#!/usr/bin/env python3
"""
2.verify_and_process_black.py  —  Verify black-blob detections and emit
X, Y, and angle permanence outputs.

Inputs
------
Reads the shared Step-1 detections from:
    data/<name>/track1.msgpack

Outputs
-------
Writes three downstream-compatible datasets:
    data/<name>_x/track2_permanence.msgpack   (X values in xPositions)
    data/<name>_y/track2_permanence.msgpack   (Y values in xPositions)
    data/<name>_a/track2_permanence.msgpack   (angle values in xPositions)

This keeps downstream code untouched while allowing X / Y / angle analysis to remain
separate.

Usage
-----
    python3 2.verify_and_process_black.py IMG_9282
    python3 2.verify_and_process_black.py 9282
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import msgpack

from helper.video_io import dataset_dir, find_video, track1_output_path, video_name
from helper.verification_black import scan_bad_frames, verify_and_sanitize
from helper.permanence_black import build_permanence_xya
from tracking_classes import VideoCentroids


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_vc(path: str) -> VideoCentroids:
    with open(path, "rb") as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def _save_msgpack(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(msgpack.packb(asdict(obj)))


def _axis_dataset_name(name: str, axis: str) -> str:
    axis = axis.lower()
    if axis not in {"x", "y", "a"}:
        raise ValueError(f"Invalid axis: {axis}")
    clean = os.path.splitext(os.path.basename(name))[0]
    return f"{clean}_{axis}"


def _axis_track2_output_path(name: str, axis: str) -> str:
    d = dataset_dir(_axis_dataset_name(name, axis))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "track2_permanence.msgpack")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify black-blob track1 detections, repair bad segments if possible, "
            "and write separate X / Y / angle permanence datasets."
        )
    )
    parser.add_argument(
        "name",
        help="Video name or numeric suffix, e.g. IMG_9282 or 9282",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.50,
        help="Min spacing ratio relative to the reference spacing (default 0.50).",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=1.50,
        help="Max spacing ratio relative to the reference spacing (default 1.50).",
    )
    parser.add_argument(
        "--no-trim-ends",
        action="store_true",
        help=(
            "Do not trim weak end columns. By default, unreliable columns may be "
            "discarded, but only if they form a contiguous block at the left and/or right edge."
        ),
    )
    parser.add_argument(
        "--min-end-support",
        type=int,
        default=3,
        help="Minimum visible support required for end columns before they are kept (default 3).",
    )
    args = parser.parse_args()

    # ---- Resolve name (video file not required for this step) ----
    video_path = find_video(args.name, "Videos")
    if video_path is not None:
        name = video_name(video_path)
    else:
        name = os.path.splitext(os.path.basename(args.name))[0]
        print(f"Note: video file not found in Videos/ — using name '{name}' for paths.")

    t1_path = track1_output_path(name)
    t2x_path = _axis_track2_output_path(name, "x")
    t2y_path = _axis_track2_output_path(name, "y")
    t2a_path = _axis_track2_output_path(name, "a")

    if not os.path.exists(t1_path):
        print(f"Error: track1 output not found: {t1_path}")
        print(f"  Run first: python3 1.track_run_black.py {args.name}")
        sys.exit(1)

    # ---- Load ----
    print(f"Loading {t1_path}…")
    vc = _load_vc(t1_path)
    print(f"  {len(vc.frames)} frames loaded.")

    # ---- Non-destructive scan ----
    print("\nScanning for bad frames…")
    n_bad, n_seg, ref_spacing = scan_bad_frames(
        vc, ratio_min=args.ratio_min, ratio_max=args.ratio_max
    )

    if n_bad > 0:
        print(f"\n  ⚠  {n_bad} bad frame(s) detected in {n_seg} segment(s).")
        print(
            f"     (reference spacing: {ref_spacing:.4f} px, "
            f"ratio bounds: [{args.ratio_min}, {args.ratio_max}])"
        )
        ans = input("\n  Apply automatic interpolation repair and continue? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted — no files written.")
            sys.exit(0)

        vc = _load_vc(t1_path)
        print("\nRepairing…")
        vc, summary = verify_and_sanitize(
            vc,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            repair=True,
            quiet=False,
        )
        print(
            f"  Repaired {summary['sanitized_frames']} frame(s) "
            f"in {summary['sanitized_runs']} segment(s)."
        )
    else:
        print("  ✓  All frames passed — no repair needed.")
        vc, summary = verify_and_sanitize(
            vc,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            repair=False,
            quiet=True,
        )

    print(f"  Mean block distance: {summary['final_mean_block_distance']:.4f} px")

    # ---- Build permanence matrices ----
    print("\nBuilding X / Y / angle permanence matrices…")
    t2x, t2y, t2a, meta = build_permanence_xya(
        vc,
        tracking_results_path=t1_path,
        quiet=False,
        trim_weak_ends=not args.no_trim_ends,
        min_end_support=args.min_end_support,
    )

    # ---- Save ----
    _save_msgpack(t2x_path, t2x)
    _save_msgpack(t2y_path, t2y)
    _save_msgpack(t2a_path, t2a)

    print("\nDone.")
    print(f"  Columns kept : {meta['n_cols_kept']}  (full solution had {meta['n_cols_full']})")
    print(f"  Trimmed ends : left={meta['trimmed_left']}, right={meta['trimmed_right']}")
    print(f"  X output     : {t2x_path}")
    print(f"  Y output     : {t2y_path}")
    print(f"  A output     : {t2a_path}")


if __name__ == "__main__":
    main()
