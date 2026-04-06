#!/usr/bin/env python3
"""
1.track_run.py  —  Run block tracking on a prepared video.

Reads params.json from data/{name}/, processes every frame in the specified
time window, and writes a VideoCentroids msgpack to data/{name}/track1.msgpack.

Usage
-----
    python3 1.track_run.py IMG_9282
    python3 1.track_run.py 9282
    python3 1.track_run.py IMG_9282 --no-preview
"""

import os
import sys
import argparse
import math
import msgpack
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.video_io  import find_video, ensure_dataset_dir, params_path, track1_output_path, video_name
from helper.params    import TrackingParams
import helper.detection as detection_helpers
from tracking_classes import Track1Params, FrameDetections, VideoCentroids

WIN_W = 1280
WIN_H = 700


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signed_angle_from_y(dx: float, dy: float) -> float:
    """Return signed angle in radians from the image y-axis to the vector."""
    return math.atan2(dx, dy)


def _archive_rectfit_orientation_from_mask(component_mask: np.ndarray) -> float:
    """
    Fit a rotated rectangle to the component contour and choose the axis branch
    closest to the image y-axis. This yields a signed angle in [-pi/4, pi/4].
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1.0:
        return float("nan")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float64)

    best_theta = None
    best_abs_theta = None
    for i in range(2):
        p0 = box[i]
        p1 = box[(i + 1) % 4]
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        length = math.hypot(vx, vy)
        if length <= 1e-6:
            continue

        ux = vx / length
        uy = vy / length
        theta = _signed_angle_from_y(ux, uy)

        if theta < -math.pi / 4:
            theta += math.pi / 2
        elif theta > math.pi / 4:
            theta -= math.pi / 2

        abs_theta = abs(theta)
        if best_theta is None or abs_theta < best_abs_theta:
            best_theta = theta
            best_abs_theta = abs_theta

    if best_theta is None:
        return float("nan")
    return float(best_theta)


detection_helpers.component_orientation_from_mask = _archive_rectfit_orientation_from_mask
DetectionKernels = detection_helpers.DetectionKernels
detect_frame = detection_helpers.detect_frame
draw_detections = detection_helpers.draw_detections


def _apply_crop(bgr, params: TrackingParams):
    """Apply pixel crop from params. Returns the cropped frame."""
    t = params.crop_top
    b = params.crop_bottom if params.crop_bottom > 0 else None
    if b is not None:
        return bgr[t:b, :]
    if t > 0:
        return bgr[t:, :]
    return bgr


def _fit_box_from_component_mask(component_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1.0:
        return None

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.int32)


def _validated_component_boxes(
    color_mask: np.ndarray,
    white_mask: np.ndarray,
    params: TrackingParams,
    kernels: DetectionKernels,
) -> list[np.ndarray]:
    if cv2.countNonZero(color_mask) == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_mask,
        connectivity=params.ccConnectivity,
    )

    img_h, img_w = color_mask.shape
    boxes: list[tuple[float, np.ndarray]] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (float(params.min_area) <= area <= params.effective_max_area):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, _ = centroids[i]

        if params.rejectNearImageBorder and detection_helpers.is_bbox_near_border(
            x, y, w, h, img_w, img_h, params.border_margin_px
        ):
            continue

        pad = params.ringOuterRadius
        y1 = max(0, y - pad)
        y2 = min(img_h, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_w, x + w + pad)

        local_comp = (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255
        outer = cv2.dilate(local_comp, kernels.se_ring_outer)
        inner = cv2.dilate(local_comp, kernels.se_ring_inner)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))

        n_ring = cv2.countNonZero(ring)
        if n_ring == 0:
            continue

        ring_white = cv2.bitwise_and(ring, white_mask[y1:y2, x1:x2])
        if cv2.countNonZero(ring_white) / n_ring < params.minWhiteCoverageFraction:
            continue

        box = _fit_box_from_component_mask(local_comp)
        if box is None:
            continue

        box[:, 0] += x1
        box[:, 1] += y1
        boxes.append((float(cx), box))

    boxes.sort(key=lambda item: item[0])
    return [box for _, box in boxes]


def _draw_archive_preview(
    bgr_cropped: np.ndarray,
    detections,
    frame_num: int,
    params: TrackingParams,
    kernels: DetectionKernels,
    y_offset: int = 0,
) -> np.ndarray:
    overlay = draw_detections(bgr_cropped, detections, frame_num, y_offset=y_offset)

    hsv = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, kernels.lower_red1, kernels.upper_red1),
        cv2.inRange(hsv, kernels.lower_red2, kernels.upper_red2),
    )
    green_mask = cv2.inRange(hsv, kernels.lower_green, kernels.upper_green)
    white_mask = cv2.inRange(hsv, kernels.lower_white, kernels.upper_white)

    red_mask = detection_helpers.cleanup_color_mask(
        red_mask, kernels.se_color_open, kernels.se_color_close
    )
    green_mask = detection_helpers.cleanup_color_mask(
        green_mask, kernels.se_color_open, kernels.se_color_close
    )
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernels.se_white_close)
    white_mask = detection_helpers.imfill_holes(white_mask)

    for box in _validated_component_boxes(red_mask, white_mask, params, kernels):
        cv2.polylines(overlay, [box], isClosed=True, color=(0, 128, 255), thickness=2)

    for box in _validated_component_boxes(green_mask, white_mask, params, kernels):
        cv2.polylines(overlay, [box], isClosed=True, color=(255, 255, 0), thickness=2)

    return overlay


def _build_track1_params(video_path: str, out_path: str, params: TrackingParams) -> Track1Params:
    """Construct the Track1Params metadata block from TrackingParams."""
    return Track1Params(
        inputVideoPath=video_path,
        outputMatPath=out_path,
        min_area=params.min_area,
        max_area=params.effective_max_area,
        minSat_color=params.minSat_color,
        minVal_color=params.minVal_color,
        redHueLow1=params.redHueLow1,
        redHueHigh1=params.redHueHigh1,
        redHueLow2=params.redHueLow2,
        redHueHigh2=params.redHueHigh2,
        greenHueLow=params.greenHueLow,
        greenHueHigh=params.greenHueHigh,
        whiteMaxSat=params.whiteMaxSat,
        whiteMinVal=params.whiteMinVal,
        colorOpenRadius=params.colorOpenRadius,
        colorCloseRadius=params.colorCloseRadius,
        whiteCloseRadius=params.whiteCloseRadius,
        ringInnerRadius=params.ringInnerRadius,
        ringOuterRadius=params.ringOuterRadius,
        minWhiteCoverageFraction=params.minWhiteCoverageFraction,
        rejectNearImageBorder=params.rejectNearImageBorder,
        borderMarginPx=params.border_margin_px,
        ccConnectivity=params.ccConnectivity,
        assumedInputType='MJPEG AVI',
        createdOn=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track coloured blocks in a video using params.json."
    )
    parser.add_argument(
        'name',
        help="Video name or numeric suffix, e.g. IMG_9282 or 9282",
    )
    parser.add_argument(
        '--no-preview', action='store_true',
        help="Disable the live detection preview.",
    )
    args = parser.parse_args()

    # ---- Locate video ----
    video_path = find_video(args.name, "Videos")
    if video_path is None:
        print(f"Error: no video found for '{args.name}' in Videos/")
        sys.exit(1)

    name = video_name(video_path)
    print(f"Video : {video_path}")

    # ---- Load params ----
    p_path = params_path(name)
    if os.path.exists(p_path):
        params = TrackingParams.load(p_path)
        print(f"Params: {p_path}")
    else:
        print(f"Warning: no params.json found at {p_path}")
        ans = input("Continue with default parameters? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            sys.exit(0)
        params = TrackingParams.defaults()

    # ---- Setup ----
    ensure_dataset_dir(name)
    out_path = track1_output_path(name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute frame range from time crop
    start_frame = max(0, int((params.time_start_s or 0.0) * fps))
    end_frame   = (
        min(total_frames, int(params.time_end_s * fps))
        if params.time_end_s is not None
        else total_frames
    )

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kernels  = DetectionKernels(params)
    y_offset = params.crop_top

    # ---- Preview window ----
    win = None
    if not args.no_preview:
        win = f"Track: {name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)

    frames_data = []
    n_processed = 0

    print(f"Tracking frames {start_frame} – {end_frame}  "
          f"({end_frame - start_frame} total)…")

    # ---- Main loop ----
    while True:
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t_ms      = cap.get(cv2.CAP_PROP_POS_MSEC)

        if frame_num >= end_frame:
            break

        ret, bgr = cap.read()
        if not ret:
            break

        bgr_c = _apply_crop(bgr, params)
        dets  = detect_frame(bgr_c, params, kernels, y_offset=y_offset)

        frames_data.append(FrameDetections(
            frame_number=frame_num,
            frame_time_s=t_ms / 1000.0,
            detections=dets,
        ))

        if win is not None:
            overlay = _draw_archive_preview(
                bgr_c,
                dets,
                frame_num,
                params,
                kernels,
                y_offset=y_offset,
            )
            cv2.imshow(win, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("  Preview closed by user — stopping early.")
                break

        n_processed += 1

    cap.release()
    cv2.destroyAllWindows()

    # ---- Save ----
    t1_params = _build_track1_params(video_path, out_path, params)

    vc = VideoCentroids(
        filepath=video_path,
        frames=frames_data,
        params=t1_params,
        nFrames=n_processed,
        fps=fps,
    )

    with open(out_path, 'wb') as fh:
        fh.write(msgpack.packb(asdict(vc)))

    print(f"\nDone.  {n_processed} frames written → {out_path}")
    print(f"Next : python3 .archive/2.verify_and_process.py {name}")


if __name__ == '__main__':
    main()
