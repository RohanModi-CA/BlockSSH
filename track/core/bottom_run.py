from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from track.tracking_classes import FrameDetections, Track1Params, VideoCentroids

from .bottom_detect import detect_frame_black, draw_detections
from .bottom_manifest import load_manifest as _load_manifest, save_manifest as _save_manifest
from .bottom_params import BottomTrackingParams
from .bottom_transform import apply_transform
from .io import find_video, save_msgpack, video_name
from .layout import (
    VIDEOS_DIR,
    ensure_dataset_dir,
    legacy_params_black_path,
    params_bottom_path,
    track1_path,
)


WIN_W = 1280
WIN_H = 760
def _build_track1_params(video_path: str, out_path: str, params: BottomTrackingParams) -> Track1Params:
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
        assumedInputType="bottom black-on-white blob tracking",
        createdOn=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_track(name: str, *, no_preview: bool = False) -> int:
    video_path = find_video(name, VIDEOS_DIR)
    if video_path is None:
        print(f"Error: no video found for '{name}' in {VIDEOS_DIR}/")
        return 1

    dataset = video_name(video_path)
    ensure_dataset_dir(dataset)
    params_path = params_bottom_path(dataset)
    legacy_params_path = legacy_params_black_path(dataset)
    if not params_path.exists() and legacy_params_path.exists():
        params = BottomTrackingParams.load(legacy_params_path)
        params.save(params_path)
        print(f"Migrated legacy params to {params_path}")
    if not params_path.exists():
        print(f"Error: params file not found: {params_path}")
        print(f"  Run first: python3 Bottom/0.VideoPrepareBottom.py {dataset}")
        return 1

    params = BottomTrackingParams.load(params_path)
    out_path = track1_path(dataset)
    print(f"Loaded params: {params_path}")
    print(f"Video : {video_path}")
    print(f"Output: {out_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int((params.time_start_s or 0.0) * fps))
    end_frame = min(total_frames, int(params.time_end_s * fps) if params.time_end_s is not None else total_frames)
    if start_frame >= end_frame:
        cap.release()
        print(f"Error: empty frame range [{start_frame}, {end_frame})")
        return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    win = None
    if not no_preview:
        win = f"Track bottom blobs: {dataset}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)

    frames_data: list[FrameDetections] = []
    n_processed = 0
    print(f"Tracking frames {start_frame} – {end_frame} ({end_frame - start_frame} total)...")
    while True:
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if frame_num >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break

        transformed, spec = apply_transform(frame, params)
        dets = detect_frame_black(
            transformed,
            params,
            x_offset=spec.crop_rect.x0,
            y_offset=spec.crop_rect.y0,
        )
        frames_data.append(
            FrameDetections(
                frame_number=frame_num,
                frame_time_s=t_ms / 1000.0,
                detections=dets,
            )
        )

        if win is not None:
            overlay = draw_detections(
                transformed,
                dets,
                frame_num,
                x_offset=spec.crop_rect.x0,
                y_offset=spec.crop_rect.y0,
            )
            cv2.imshow(win, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("Preview closed by user — stopping early.")
                break
        n_processed += 1

    cap.release()
    cv2.destroyAllWindows()

    t1_params = _build_track1_params(str(video_path), str(out_path), params)
    vc = VideoCentroids(
        filepath=str(video_path),
        frames=frames_data,
        params=t1_params,
        nFrames=n_processed,
        fps=fps,
    )
    save_msgpack(out_path, vc)

    manifest = _load_manifest(dataset)
    manifest.dataset = dataset
    manifest.source_video = str(video_path.relative_to(VIDEOS_DIR.parent))
    manifest.params_file = params_path.name
    manifest.track1_file = out_path.name
    manifest.rotation_deg = int(params.rotation_deg)
    manifest.crop_rect = asdict(params.crop_rect)
    _save_manifest(dataset, manifest)

    print(f"\nDone. {n_processed} frames written -> {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track bottom black blobs and write Step 1 msgpack.")
    parser.add_argument("name", help="Video name or numeric suffix, e.g. IMG_9282 or 9282")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window.")
    return parser
