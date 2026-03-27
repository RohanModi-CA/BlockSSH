from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import fields as dc_fields
from pathlib import Path

import cv2
import numpy as np

from .bottom_detect import detect_frame_black, draw_detections
from .bottom_manifest import load_manifest as _load_manifest, save_manifest as _save_manifest
from .bottom_params import BottomTrackingParams
from .bottom_transform import apply_transform, build_transform_spec, darken_outside_crop, rotate_frame
from .io import find_video, load_json, save_json, video_name
from .layout import (
    VIDEOS_DIR,
    ensure_dataset_dir,
    legacy_params_black_path,
    manifest_path,
    params_bottom_path,
)


WIN_W = 1280
WIN_H = 760
ROTATIONS = [0, 90, 180, 270]


def _fit_for_window(frame: np.ndarray, *, avail_w: int = WIN_W, avail_h: int = WIN_H - 110) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(avail_w / w, avail_h / h)
    disp = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    return disp, scale


def setup_rotation_and_crop(video_path: str, params: BottomTrackingParams) -> BottomTrackingParams:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  [crop] Could not read mid-video frame — skipping crop setup.")
        return params

    base_rotated = rotate_frame(frame, params.rotation_deg)
    base_h, base_w = base_rotated.shape[:2]
    initial_spec = build_transform_spec(frame, params)

    win = "Bottom Prepare  |  rotate + crop  |  C / Enter confirm  |  Q skip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    cv2.createTrackbar("Rotation", win, ROTATIONS.index(initial_spec.rotation_deg), len(ROTATIONS) - 1, lambda _v: None)
    cv2.createTrackbar("Left", win, initial_spec.crop_rect.x0, base_w - 1, lambda _v: None)
    cv2.createTrackbar("Right", win, initial_spec.crop_rect.x1 or base_w, base_w, lambda _v: None)
    cv2.createTrackbar("Top", win, initial_spec.crop_rect.y0, base_h - 1, lambda _v: None)
    cv2.createTrackbar("Bottom", win, initial_spec.crop_rect.y1 or base_h, base_h, lambda _v: None)

    print(f"\n[Step 1 — Rotation + Crop]  Frame size after rotation preview: {base_w} × {base_h} px")
    print("  Set rotation first, then crop to the useful arena.")
    print("  C / Enter = confirm      Q = skip")

    while True:
        rotation_idx = cv2.getTrackbarPos("Rotation", win)
        rotation_deg = ROTATIONS[rotation_idx]
        rotated = rotate_frame(frame, rotation_deg)
        rot_h, rot_w = rotated.shape[:2]

        left = min(cv2.getTrackbarPos("Left", win), max(0, rot_w - 1))
        right = min(max(cv2.getTrackbarPos("Right", win), left + 1), rot_w)
        top = min(cv2.getTrackbarPos("Top", win), max(0, rot_h - 1))
        bottom = min(max(cv2.getTrackbarPos("Bottom", win), top + 1), rot_h)

        disp = darken_outside_crop(rotated, type(params.crop_rect)(x0=left, x1=right, y0=top, y1=bottom))
        cv2.rectangle(disp, (left, top), (right - 1, bottom - 1), (0, 255, 255), 2)
        disp, scale = _fit_for_window(disp)

        cv2.putText(
            disp,
            f"rot={rotation_deg}  crop=({left},{top}) -> ({right},{bottom})",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 80, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("c"), 13):
            params.rotation_deg = rotation_deg
            params.crop_rect.x0 = int(left)
            params.crop_rect.x1 = int(right)
            params.crop_rect.y0 = int(top)
            params.crop_rect.y1 = int(bottom)
            break
        if key == ord("q"):
            print("  Rotation/crop setup skipped — keeping previous values.")
            cv2.destroyWindow(win)
            return params

    cv2.destroyWindow(win)
    print(
        f"  Rotation set: {params.rotation_deg} deg | "
        f"crop=({params.crop_rect.x0}, {params.crop_rect.y0}) -> "
        f"({params.crop_rect.x1}, {params.crop_rect.y1})"
    )
    return params


def setup_time_crop(video_path: str, params: BottomTrackingParams) -> BottomTrackingParams:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total_fr / fps if fps > 0 else 0.0
    print(f"\n[Step 2 — Time Crop]  Duration: {duration:.2f} s  ({total_fr} frames @ {fps:.2f} fps)")
    print("  Press Enter to keep the current value.")

    def _ask(label: str, current, fallback_display: str):
        while True:
            raw = input(f"  {label} [{current if current is not None else fallback_display}]: ").strip()
            if raw == "":
                return current
            try:
                value = float(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
            if value < 0:
                print("  Must be >= 0.")
                continue
            return value

    params.time_start_s = _ask("Start time (s)", params.time_start_s, "0.0")
    params.time_end_s = _ask("End time   (s)", params.time_end_s, f"{duration:.2f}")
    if params.time_end_s is not None and params.time_end_s <= params.time_start_s:
        print("  Warning: end <= start — clearing end time.")
        params.time_end_s = None
    end_display = f"{params.time_end_s:.2f} s" if params.time_end_s is not None else "end of video"
    print(f"  Time window: {params.time_start_s:.2f} s -> {end_display}")
    return params


def _print_summary(label: str, counts: list[int]) -> None:
    n = len(counts)
    tally = Counter(counts)
    modal = tally.most_common(1)[0][0]
    n_cons = tally[modal]
    print(f"\n  -- {label} ({n} frames) --")
    for key in sorted(tally):
        bar = "#" * max(1, int(tally[key] / n * 32))
        print(f"    {key:3d} detections : {tally[key]:4d}/{n}  ({100 * tally[key] / n:5.1f}%)  {bar}")
    print(f"  Modal count = {modal}  |  consistent: {n_cons}/{n}  ({100 * n_cons / n:.1f}%)")


def _run_frames(video_path: str, params: BottomTrackingParams, frame_indices: list[int], label: str, show_preview: bool) -> list[int]:
    cap = cv2.VideoCapture(video_path)
    counts: list[int] = []
    win = None
    if show_preview:
        win = f"Test Preview — {label}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            counts.append(0)
            continue
        transformed, spec = apply_transform(frame, params)
        dets = detect_frame_black(
            transformed,
            params,
            x_offset=spec.crop_rect.x0,
            y_offset=spec.crop_rect.y0,
        )
        counts.append(len(dets))
        if show_preview and win is not None:
            overlay = draw_detections(
                transformed,
                dets,
                frame_idx,
                x_offset=spec.crop_rect.x0,
                y_offset=spec.crop_rect.y0,
            )
            cv2.imshow(win, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if show_preview and win is not None:
        cv2.destroyWindow(win)
    return counts


def run_test_tracking(video_path: str, params: BottomTrackingParams, show_preview: bool = True) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fr_start = max(0, int((params.time_start_s or 0.0) * fps))
    fr_end = min(total_fr, int(params.time_end_s * fps) if params.time_end_s is not None else total_fr)

    consecutive = list(range(fr_start, min(fr_start + 30, fr_end)))
    counts = _run_frames(video_path, params, consecutive, "first 30", show_preview=show_preview)
    _print_summary("First 30 frames", counts)

    ans = input("\n  Also test 30 random frames? [y/N] ").strip().lower()
    if ans == "y":
        n_take = min(30, max(0, fr_end - fr_start))
        random_frames = sorted(random.sample(range(fr_start, fr_end), n_take))
        counts = _run_frames(video_path, params, random_frames, "random 30", show_preview=show_preview)
        _print_summary("Random frames", counts)


def adjust_params_interactive(params: BottomTrackingParams) -> BottomTrackingParams:
    print("\n[Step 4 — Tune Parameters]")
    print("  Format:  param_name value")
    print("  Example: min_area 70000")
    print("  Type 'done' to continue.")

    fld_map = {field.name: field for field in dc_fields(params)}
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() == "done":
            break
        parts = line.split(None, 1)
        if len(parts) != 2:
            print("  Format: param_name value")
            continue
        key, val_str = parts
        if key not in fld_map or key in {"crop_rect"}:
            print(f"  Unknown or non-inline param '{key}'.")
            continue
        old_val = getattr(params, key)
        try:
            if val_str.lower() in ("none", "null"):
                new_val = None
            elif isinstance(old_val, bool):
                new_val = val_str.lower() in ("true", "1", "yes", "y")
            elif isinstance(old_val, int):
                new_val = int(val_str)
            elif isinstance(old_val, float):
                new_val = float(val_str)
            elif old_val is None:
                try:
                    new_val = int(val_str)
                except ValueError:
                    new_val = float(val_str)
            else:
                new_val = val_str
        except (ValueError, TypeError) as exc:
            print(f"  Cannot parse '{val_str}' for {key}: {exc}")
            continue
        setattr(params, key, new_val)
        print(f"  {key}: {old_val} -> {new_val}")
    return params


def run_prepare(name: str, *, no_preview: bool = False) -> int:
    video_path = find_video(name, VIDEOS_DIR)
    if video_path is None:
        print(f"Error: no video found for '{name}' in {VIDEOS_DIR}/")
        return 1

    dataset = video_name(video_path)
    ensure_dataset_dir(dataset)
    params_path = params_bottom_path(dataset)
    legacy_params_path = legacy_params_black_path(dataset)
    manifest = _load_manifest(dataset)

    if params_path.exists():
        params = BottomTrackingParams.load(params_path)
        print(f"Loaded existing params from {params_path}")
    elif legacy_params_path.exists():
        params = BottomTrackingParams.load(legacy_params_path)
        print(f"Loaded legacy params from {legacy_params_path}")
        params.save(params_path)
        print(f"Migrated params to {params_path}")
    else:
        params = BottomTrackingParams.defaults()
        print("Starting from default bottom-tracking parameters.")

    print(f"Video: {video_path}")
    params = setup_rotation_and_crop(str(video_path), params)
    params = setup_time_crop(str(video_path), params)
    params.save(params_path)

    show_preview = not no_preview
    while True:
        run_test_tracking(str(video_path), params, show_preview=show_preview)
        ans = input("\nAdjust parameters and re-run tests? [y/N] ").strip().lower()
        if ans != "y":
            break
        params = adjust_params_interactive(params)
        params.save(params_path)

    params.save(params_path)
    manifest.dataset = dataset
    manifest.source_video = str(video_path.relative_to(VIDEOS_DIR.parent))
    manifest.params_file = params_path.name
    manifest.rotation_deg = int(params.rotation_deg)
    manifest.crop_rect = {
        "x0": int(params.crop_rect.x0),
        "x1": None if params.crop_rect.x1 is None else int(params.crop_rect.x1),
        "y0": int(params.crop_rect.y0),
        "y1": None if params.crop_rect.y1 is None else int(params.crop_rect.y1),
    }
    _save_manifest(dataset, manifest)

    print("\nSetup complete.")
    print(f"  Params   : {params_path}")
    print(f"  Manifest : {manifest_path(dataset)}")
    print(f"  Next     : python3 Bottom/1.TrackRun.py {dataset}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive preparation for bottom black-blob tracking.")
    parser.add_argument("name", help="Video name or numeric suffix, e.g. IMG_9282 or 9282")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live detection preview during tests.")
    return parser
