#!/usr/bin/env python3
"""
0.video_prepare_black.py  —  Interactive preparation for black-blob tracking.

Sets pixel crop, time crop, tests detection with live preview,
and allows inline parameter tuning. Produces data/{name}/params_black.json.

Usage
-----
    python3 0.video_prepare_black.py IMG_9282
    python3 0.video_prepare_black.py 9282
    python3 0.video_prepare_black.py IMG_9282 --no-preview
"""

import sys
import os
import argparse
import importlib.util
import random
from collections import Counter
from dataclasses import fields as dc_fields
from pathlib import Path
import cv2
import numpy as np

from helper.video_io import find_video, ensure_dataset_dir, video_name

_BLACK_TRACKER_PATH = Path(__file__).with_name("1.track_run_black.py")
_BLACK_TRACKER_SPEC = importlib.util.spec_from_file_location("track_run_black_module", _BLACK_TRACKER_PATH)
if _BLACK_TRACKER_SPEC is None or _BLACK_TRACKER_SPEC.loader is None:
    raise ImportError(f"Could not load black tracker module from {_BLACK_TRACKER_PATH}")
_BLACK_TRACKER = importlib.util.module_from_spec(_BLACK_TRACKER_SPEC)
_BLACK_TRACKER_SPEC.loader.exec_module(_BLACK_TRACKER)

BlackTrackingParams = _BLACK_TRACKER.BlackTrackingParams
detect_frame_black = _BLACK_TRACKER.detect_frame_black
draw_detections_black = _BLACK_TRACKER.draw_detections

WIN_W = 1280
WIN_H = 700


# =============================================================================
# STEP 1 — Pixel crop
# =============================================================================

def params_black_path(name: str) -> str:
    clean = video_name(name)
    return os.path.join("data", clean, "params_black.json")


def setup_pixel_crop(video_path: str, params: BlackTrackingParams) -> BlackTrackingParams:
    """
    Show a frame from the middle of the video in an OpenCV window.
    Two trackbars (Top / Bottom) control the vertical crop.
    The region between the lines is kept; outside is darkened.
    Press C or Enter to confirm; Q to skip without changing.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  [crop] Could not read mid-video frame — skipping crop setup.")
        return params

    orig_h, orig_w = frame.shape[:2]

    win = "Pixel Crop  |  adjust trackbars  |  C / Enter to confirm  |  Q to skip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    init_top    = params.crop_top
    init_bottom = params.crop_bottom if params.crop_bottom > 0 else orig_h

    cv2.createTrackbar("Top",    win, init_top,    orig_h, lambda _: None)
    cv2.createTrackbar("Bottom", win, init_bottom, orig_h, lambda _: None)

    print(f"\n[Step 1 — Pixel Crop]  Frame size: {orig_w} × {orig_h} px")
    print("  Adjust trackbars so the two lines bracket the arena.")
    print("  C / Enter = confirm      Q = skip")

    # Scale so the image fits inside WIN_W × (WIN_H - ~80px for two trackbars)
    avail_h = WIN_H - 80
    scale   = min(WIN_W / orig_w, avail_h / orig_h)
    dw, dh  = int(orig_w * scale), int(orig_h * scale)

    while True:
        top    = cv2.getTrackbarPos("Top",    win)
        bottom = cv2.getTrackbarPos("Bottom", win)
        if bottom <= top:
            bottom = min(top + 1, orig_h)

        disp     = cv2.resize(frame, (dw, dh))
        top_d    = int(top    * scale)
        bottom_d = int(bottom * scale)

        # Darken the regions outside the crop
        if top_d > 0:
            disp[:top_d, :] = (disp[:top_d, :].astype(np.float32) * 0.35).astype(np.uint8)
        if bottom_d < dh:
            disp[bottom_d:, :] = (disp[bottom_d:, :].astype(np.float32) * 0.35).astype(np.uint8)

        # Crop lines
        cv2.line(disp, (0, top_d),    (dw, top_d),    (0, 255, 255), 2)
        cv2.line(disp, (0, bottom_d), (dw, bottom_d), (255, 80, 255), 2)

        # Labels (keep within frame bounds)
        lbl_top_y    = max(top_d    + 20, 20)
        lbl_bottom_y = min(bottom_d - 8,  dh - 8)
        cv2.putText(disp, f"Top: {top} px",
                    (8, lbl_top_y),    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        bot_label = f"Bottom: {bottom} px" + ("  (full height)" if bottom >= orig_h else "")
        cv2.putText(disp, bot_label,
                    (8, lbl_bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 255), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('c'), 13):   # C or Enter
            break
        if key == ord('q'):
            print("  Crop setup skipped — keeping previous values.")
            cv2.destroyWindow(win)
            return params

    cv2.destroyWindow(win)

    params.crop_top    = top
    params.crop_bottom = bottom if bottom < orig_h else 0
    print(f"  Crop set: top = {params.crop_top} px,  "
          f"bottom = {params.crop_bottom or orig_h} px"
          + ("  (0 = full height)" if params.crop_bottom == 0 else ""))
    return params


# =============================================================================
# STEP 2 — Time crop
# =============================================================================

def setup_time_crop(video_path: str, params: BlackTrackingParams) -> BlackTrackingParams:
    """Prompt for start / end times in seconds via the terminal."""
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total_fr / fps if fps > 0 else 0.0
    print(f"\n[Step 2 — Time Crop]  Duration: {duration:.2f} s  "
          f"({total_fr} frames @ {fps:.2f} fps)")
    print("  Press Enter to keep the current value.")

    def _ask(label: str, current, fallback_display: str):
        prompt = f"  {label} [{current if current is not None else fallback_display}]: "
        while True:
            raw = input(prompt).strip()
            if raw == '':
                return current
            try:
                v = float(raw)
                if v < 0:
                    print("  Must be ≥ 0.")
                    continue
                return v
            except ValueError:
                print("  Please enter a number.")

    params.time_start_s = _ask("Start time (s)", params.time_start_s, "0.0")
    params.time_end_s   = _ask("End time   (s)", params.time_end_s,   f"{duration:.2f}")

    if params.time_end_s is not None and params.time_end_s <= params.time_start_s:
        print("  Warning: end ≤ start — clearing end time (will use full video).")
        params.time_end_s = None

    end_display = f"{params.time_end_s:.2f} s" if params.time_end_s is not None else "end of video"
    print(f"  Time window: {params.time_start_s:.2f} s → {end_display}")
    return params


# =============================================================================
# STEP 3 — Test tracking
# =============================================================================

def _apply_crop(bgr: np.ndarray, params: BlackTrackingParams):
    """Return the pixel-cropped frame."""
    t = params.crop_top
    b = params.crop_bottom if params.crop_bottom > 0 else None
    if b is not None:
        return bgr[t:b, :]
    if t > 0:
        return bgr[t:, :]
    return bgr


def _run_frames(
    video_path:    str,
    params:        BlackTrackingParams,
    frame_indices: list,
    label:         str,
    show_preview:  bool,
) -> list:
    """
    Detect black blobs in the given frame indices.
    Optionally shows a live preview window (no artificial delay).
    Returns a list of detection counts, one per frame.
    """
    cap = cv2.VideoCapture(video_path)
    counts = []

    win = None
    if show_preview:
        win = f"Test Preview — {label}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)

    for fnum in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, bgr = cap.read()
        if not ret:
            counts.append(0)
            continue

        bgr_c = _apply_crop(bgr, params)
        dets = detect_frame_black(bgr_c, params, y_offset=params.crop_top)
        counts.append(len(dets))

        if show_preview and win is not None:
            overlay = draw_detections_black(
                bgr_c,
                dets,
                fnum,
                y_offset=params.crop_top,
            )
            cv2.imshow(win, overlay)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    if show_preview and win is not None:
        cv2.destroyWindow(win)

    return counts


def _print_summary(label: str, counts: list) -> None:
    n      = len(counts)
    tally  = Counter(counts)
    modal  = tally.most_common(1)[0][0]
    n_cons = tally[modal]

    print(f"\n  ── {label} ({n} frames) ──")
    for k in sorted(tally):
        bar = "█" * max(1, int(tally[k] / n * 32))
        print(f"    {k:3d} detections : {tally[k]:4d}/{n}  ({100*tally[k]/n:5.1f}%)  {bar}")
    print(f"  Modal count = {modal}  |  consistent: {n_cons}/{n}  ({100*n_cons/n:.1f}%)")

def run_test_tracking(
    video_path:   str,
    params:       BlackTrackingParams,
    show_preview: bool = True,
) -> None:
    """
    1. Run detection on the first 30 frames of the time window.
    2. Print summary.
    3. Ask [y/N] to also run on 30 random frames.
    """
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fr_start = max(0, int((params.time_start_s or 0.0) * fps))
    fr_end   = min(
        total_fr,
        int(params.time_end_s * fps) if params.time_end_s is not None else total_fr
    )

    # first 30 frames of the time window
    n_test = min(30, fr_end - fr_start)
    consec = list(range(fr_start, fr_start + n_test))

    print(f"\n[Step 3 — Test Tracking]  Running on {len(consec)} consecutive frames "
          f"(start of time window)…")
    counts = _run_frames(video_path, params, consec, "consecutive", show_preview)
    _print_summary("consecutive frames", counts)

    ans = input("\n  Also test on 30 random frames? [y/N] ").strip().lower()
    if ans == 'y':
        pool      = list(range(fr_start, fr_end))
        n_rand    = min(30, len(pool))
        rand_idx  = sorted(random.sample(pool, n_rand))
        print(f"  Running on {n_rand} random frames…")
        r_counts  = _run_frames(video_path, params, rand_idx, "random", show_preview)
        _print_summary("random frames", r_counts)


# =============================================================================
# STEP 4 — Parameter adjustment
# =============================================================================

def adjust_params_interactive(params: BlackTrackingParams) -> BlackTrackingParams:
    """
    Interactive terminal editor for BlackTrackingParams.
    User types  'param_name value'  to update a field; 'done' to finish.
    Booleans accept true/false/yes/no/1/0.
    Optional fields accept 'none' to clear them.
    """
    fld_map = {f.name: f for f in dc_fields(params)}

    print("\n[Step 4 — Parameter Adjustment]  Current values:")
    for name, fld in fld_map.items():
        print(f"  {name:<30s} = {getattr(params, name)}")

    print("\n  Enter  'param_name value'  to update a param.")
    print("  Enter  'done'  (or blank line) to finish.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line or line.lower() == 'done':
            break

        parts = line.split(None, 1)
        if len(parts) != 2:
            print("  Format:  param_name value")
            continue

        key, val_str = parts
        if key not in fld_map:
            close = [n for n in fld_map if n.lower().startswith(key.lower()[:4])]
            hint  = f"  Did you mean: {', '.join(close[:3])}?" if close else ""
            print(f"  Unknown param '{key}'.{hint}")
            continue

        old_val = getattr(params, key)

        # Parse the new value
        try:
            if val_str.lower() in ('none', 'null'):
                new_val = None
            elif isinstance(old_val, bool):
                new_val = val_str.lower() in ('true', '1', 'yes', 'y')
            elif isinstance(old_val, int):
                new_val = int(val_str)
            elif isinstance(old_val, float):
                new_val = float(val_str)
            elif old_val is None:
                # Optional field — try int then float
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
        print(f"  {key}: {old_val}  →  {new_val}")

    return params


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive preparation for black-blob tracking."
    )
    parser.add_argument(
        'name',
        help="Video name or numeric suffix, e.g. IMG_9282 or 9282",
    )
    parser.add_argument(
        '--no-preview', action='store_true',
        help="Disable the live detection preview during tests.",
    )
    args = parser.parse_args()

    # ---- Find video ----
    video_path = find_video(args.name, "Videos")
    if video_path is None:
        print(f"Error: no video found for '{args.name}' in Videos/")
        sys.exit(1)

    name = video_name(video_path)
    print(f"Video: {video_path}")

    # ---- Dataset directory ----
    out_dir = ensure_dataset_dir(name)
    p_path = params_black_path(name)

    # ---- Load or create params ----
    if os.path.exists(p_path):
        params = BlackTrackingParams.load(p_path)
        print(f"Loaded existing params from {p_path}")
    else:
        params = BlackTrackingParams.defaults()
        print("Starting from default black-tracking parameters.")

    show_preview = not args.no_preview

    # ---- Step 1: pixel crop ----
    params = setup_pixel_crop(video_path, params)

    # ---- Step 2: time crop ----
    params = setup_time_crop(video_path, params)

    # Intermediate save so progress isn't lost
    params.save(p_path)

    # ---- Steps 3 + 4: test & tune loop ----
    while True:
        run_test_tracking(video_path, params, show_preview=show_preview)

        ans = input("\nAdjust parameters and re-run tests? [y/N] ").strip().lower()
        if ans != 'y':
            break
        params = adjust_params_interactive(params)

    # ---- Final save ----
    params.save(p_path)
    print(f"\nSetup complete.")
    print(f"  Params : {p_path}")
    print(f"  Next   : python3 1.track_run_black.py {video_path} --params {p_path}")


if __name__ == '__main__':
    main()
