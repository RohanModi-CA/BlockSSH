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
import msgpack
from dataclasses import asdict
from datetime import datetime

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.video_io  import find_video, ensure_dataset_dir, params_path, track1_output_path, video_name
from helper.params    import TrackingParams
from helper.detection import DetectionKernels, detect_frame, draw_detections
from tracking_classes import Track1Params, FrameDetections, VideoCentroids

WIN_W = 1280
WIN_H = 700


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_crop(bgr, params: TrackingParams):
    """Apply pixel crop from params. Returns the cropped frame."""
    t = params.crop_top
    b = params.crop_bottom if params.crop_bottom > 0 else None
    if b is not None:
        return bgr[t:b, :]
    if t > 0:
        return bgr[t:, :]
    return bgr


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
            overlay = draw_detections(bgr_c, dets, frame_num, y_offset=y_offset)
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
