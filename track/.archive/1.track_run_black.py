#!/usr/bin/env python3
"""
1.track_run_black.py  —  Step 1 tracker for black blobs on a mostly white background.

This is the analogue of 1.track_run.py for the new video type:
- tracks dark/black blobs instead of red/green
- preserves centroid-based detections
- stores x, y, and angle in DetectionRecord
- writes a VideoCentroids msgpack compatible with existing data structures

Usage
-----
    python3 1.track_run_black.py /path/to/video.mov
    python3 1.track_run_black.py /path/to/video.mov --no-preview
    python3 1.track_run_black.py /path/to/video.mov --out data/MY_VIDEO/track1.msgpack

Optional:
    python3 1.track_run_black.py /path/to/video.mov --params data/MY_VIDEO/params_black.json

Notes
-----
- This is only the Step 1 detection pass.
- The output format matches VideoCentroids from tracking_classes.py.
- Since colour is no longer meaningful, detections are tagged with color='b'.
- Later downstream processing can split into X / Y / angle permanence files while keeping
  the old matrix layout unchanged.
"""

import os
import sys
import json
import math
import argparse
import msgpack

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List

import cv2
import numpy as np

from tracking_classes import Track1Params, DetectionRecord, FrameDetections, VideoCentroids


WIN_W = 1280
WIN_H = 700


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class BlackTrackingParams:
    # Crop
    crop_top: int = 0
    crop_bottom: int = 0          # 0 = full height
    time_start_s: float = 0.0
    time_end_s: Optional[float] = None

    # Detection
    dark_max_val: int = 90        # grayscale threshold: pixel <= this is "black"
    blur_kernel: int = 5          # odd; median blur preserves edges better than Gaussian
    open_radius: int = 1          # small symmetric morphology
    close_radius: int = 2

    # Blob filtering
    min_area: int = 90000
    max_area: Optional[float] = None

    # Border handling
    reject_near_image_border: bool = True
    border_margin_px: int = 3

    # CC settings
    cc_connectivity: int = 8

    @property
    def effective_max_area(self) -> float:
        return float("inf") if self.max_area is None else float(self.max_area)

    @classmethod
    def defaults(cls) -> "BlackTrackingParams":
        return cls()

    @classmethod
    def load(cls, path: str) -> "BlackTrackingParams":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        out = cls.defaults()
        for k, v in data.items():
            if hasattr(out, k):
                setattr(out, k, v)
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)


# =============================================================================
# Helpers
# =============================================================================

def video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def ensure_dataset_dir(name: str, data_root: str = "data") -> str:
    out_dir = os.path.join(data_root, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def default_out_path(video_path: str, data_root: str = "data") -> str:
    name = video_name(video_path)
    out_dir = ensure_dataset_dir(name, data_root=data_root)
    return os.path.join(out_dir, "track1.msgpack")


def default_params_path(video_path: str, data_root: str = "data") -> str:
    name = video_name(video_path)
    out_dir = ensure_dataset_dir(name, data_root=data_root)
    return os.path.join(out_dir, "params_black.json")


def apply_crop(bgr: np.ndarray, params: BlackTrackingParams) -> np.ndarray:
    t = params.crop_top
    b = params.crop_bottom if params.crop_bottom > 0 else None

    if b is not None:
        return bgr[t:b, :]
    if t > 0:
        return bgr[t:, :]
    return bgr


def disk_kernel(radius: int) -> Optional[np.ndarray]:
    if radius <= 0:
        return None
    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def build_dark_mask(gray: np.ndarray, params: BlackTrackingParams) -> np.ndarray:
    """
    Build a binary mask of dark pixels.

    Design goal:
    - preserve centroid location
    - avoid directional operators
    - keep processing symmetric
    """
    blur_k = max(1, int(params.blur_kernel))
    if blur_k % 2 == 0:
        blur_k += 1

    if blur_k > 1:
        gray = cv2.medianBlur(gray, blur_k)

    # Black objects on white background
    mask = (gray <= params.dark_max_val).astype(np.uint8) * 255

    k_open = disk_kernel(params.open_radius)
    if k_open is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    k_close = disk_kernel(params.close_radius)
    if k_close is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    return mask


def component_touches_border(stats_row, width: int, height: int, margin: int) -> bool:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])

    return (
        x <= margin or
        y <= margin or
        (x + w) >= (width - margin) or
        (y + h) >= (height - margin)
    )


def component_orientation_from_mask(component_mask: np.ndarray) -> float:
    """
    Return principal-axis orientation in radians, modulo pi, in [-pi/2, pi/2).
    component_mask must be a binary mask for ONE connected component.
    """
    ys, xs = np.nonzero(component_mask)
    if len(xs) < 2:
        return float("nan")

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    x -= x.mean()
    y -= y.mean()

    mu20 = np.mean(x * x)
    mu02 = np.mean(y * y)
    mu11 = np.mean(x * y)

    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)

    if theta >= math.pi / 2:
        theta -= math.pi
    elif theta < -math.pi / 2:
        theta += math.pi

    return float(theta)


def detect_frame_black(
    bgr_cropped: np.ndarray,
    params: BlackTrackingParams,
    y_offset: int = 0,
) -> List[DetectionRecord]:
    """
    Detect black blobs in a cropped frame.

    Returns detections with x/y in original-image coordinates and angle in radians.
    """
    gray = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2GRAY)
    mask = build_dark_mask(gray, params)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=params.cc_connectivity,
        ltype=cv2.CV_32S,
    )

    h, w = mask.shape[:2]
    detections: List[DetectionRecord] = []

    for lab in range(1, n_labels):  # skip background
        area = float(stats[lab, cv2.CC_STAT_AREA])

        if area < params.min_area:
            continue
        if area > params.effective_max_area:
            continue

        if params.reject_near_image_border:
            if component_touches_border(stats[lab], w, h, params.border_margin_px):
                continue

        cx, cy = centroids[lab]

        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        ww = int(stats[lab, cv2.CC_STAT_WIDTH])
        hh = int(stats[lab, cv2.CC_STAT_HEIGHT])

        local_labels = labels[y0:y0 + hh, x0:x0 + ww]
        component_mask = (local_labels == lab).astype(np.uint8)

        angle = component_orientation_from_mask(component_mask)

        detections.append(
            DetectionRecord(
                x=float(cx),
                y=float(cy + y_offset),
                color="b",
                area=area,
                angle=angle,
            )
        )

    detections.sort(key=lambda d: d.x)
    return detections


def draw_detections(
    bgr_cropped: np.ndarray,
    detections: List[DetectionRecord],
    frame_num: int,
    y_offset: int = 0,
) -> np.ndarray:
    """
    Preview overlay.

    Since the frame is cropped for display, convert stored original-y back to cropped-y.
    """
    overlay = bgr_cropped.copy()

    for i, det in enumerate(detections):
        x = int(round(det.x))
        y = int(round(det.y - y_offset))

        cv2.circle(overlay, (x, y), 8, (0, 0, 255), 2)

        if np.isfinite(det.angle):
            L = 40
            dx = int(round(math.cos(det.angle) * L))
            dy = int(round(math.sin(det.angle) * L))
            cv2.line(overlay, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 255), 2)

        cv2.putText(
            overlay,
            f"{i}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            f"({det.x:.1f}, {det.y:.1f})",
            (x + 8, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 128, 255),
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        overlay,
        f"frame={frame_num}  n={len(detections)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 255),
        2,
        cv2.LINE_AA
    )

    return overlay


def build_track1_params(video_path: str, out_path: str, params: BlackTrackingParams) -> Track1Params:
    """
    Reuse existing Track1Params container so msgpack stays structurally familiar.

    Fields that no longer have semantic meaning are filled with neutral values.
    """
    return Track1Params(
        inputVideoPath=video_path,
        outputMatPath=out_path,
        min_area=params.min_area,
        max_area=params.effective_max_area,

        minSat_color=0.0,
        minVal_color=0.0,
        redHueLow1=0.0,
        redHueHigh1=0.0,
        redHueLow2=0.0,
        redHueHigh2=0.0,
        greenHueLow=0.0,
        greenHueHigh=0.0,
        whiteMaxSat=0.0,
        whiteMinVal=0.0,

        colorOpenRadius=params.open_radius,
        colorCloseRadius=params.close_radius,
        whiteCloseRadius=0,
        ringInnerRadius=0,
        ringOuterRadius=0,
        minWhiteCoverageFraction=0.0,

        rejectNearImageBorder=params.reject_near_image_border,
        borderMarginPx=params.border_margin_px,
        ccConnectivity=params.cc_connectivity,

        assumedInputType="black-on-white blob tracking",
        createdOn=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track black blobs on mostly white background and write Step 1 msgpack."
    )
    parser.add_argument(
        "video_path",
        help="Path to the input video file."
    )
    parser.add_argument(
        "--params",
        help="Optional path to BlackTrackingParams JSON. Defaults to data/<video_name>/params_black.json if present."
    )
    parser.add_argument(
        "--out",
        help="Optional explicit output msgpack path. Default: data/<video_name>/track1.msgpack"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window."
    )
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    name = video_name(video_path)
    out_path = args.out if args.out else default_out_path(video_path)
    auto_params_path = default_params_path(video_path)

    if args.params:
        if not os.path.exists(args.params):
            print(f"Error: params file not found: {args.params}")
            sys.exit(1)
        params = BlackTrackingParams.load(args.params)
        print(f"Loaded params: {args.params}")
    elif os.path.exists(auto_params_path):
        params = BlackTrackingParams.load(auto_params_path)
        print(f"Loaded params: {auto_params_path}")
    else:
        params = BlackTrackingParams.defaults()
        print("Using default black-tracking parameters.")

    print(f"Video : {video_path}")
    print(f"Output: {out_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int((params.time_start_s or 0.0) * fps))
    end_frame = (
        min(total_frames, int(params.time_end_s * fps))
        if params.time_end_s is not None
        else total_frames
    )

    if start_frame >= end_frame:
        print(f"Error: empty frame range [{start_frame}, {end_frame})")
        cap.release()
        sys.exit(1)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    y_offset = params.crop_top
    frames_data: List[FrameDetections] = []
    n_processed = 0

    win = None
    if not args.no_preview:
        win = f"Track black blobs: {name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)

    print(f"Tracking frames {start_frame} – {end_frame} ({end_frame - start_frame} total)...")

    while True:
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if frame_num >= end_frame:
            break

        ret, bgr = cap.read()
        if not ret:
            break

        bgr_c = apply_crop(bgr, params)
        dets = detect_frame_black(bgr_c, params, y_offset=y_offset)

        frames_data.append(
            FrameDetections(
                frame_number=frame_num,
                frame_time_s=t_ms / 1000.0,
                detections=dets,
            )
        )

        if win is not None:
            overlay = draw_detections(bgr_c, dets, frame_num, y_offset=y_offset)
            cv2.imshow(win, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("Preview closed by user — stopping early.")
                break

        n_processed += 1

    cap.release()
    cv2.destroyAllWindows()

    t1_params = build_track1_params(video_path, out_path, params)

    vc = VideoCentroids(
        filepath=video_path,
        frames=frames_data,
        params=t1_params,
        nFrames=n_processed,
        fps=fps,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(msgpack.packb(asdict(vc), use_bin_type=True))

    print(f"\nDone. {n_processed} frames written -> {out_path}")
    print("This Step 1 output contains x, y, and angle per detection; permanence XYA split comes later.")


if __name__ == "__main__":
    main()
