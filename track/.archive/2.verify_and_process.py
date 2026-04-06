#!/usr/bin/env python3
"""
2.verify_and_process.py  —  Verify detections and build the permanence matrix.

Loads data/{name}/track1.msgpack, verifies every frame, repairs any bad
segments (with user confirmation), then builds the permanence matrix and
writes:
  - data/{name}/track2_permanence.msgpack            (legacy x output)
  - data/{name}/components/x/track2_permanence.msgpack
  - data/{name}/components/y/track2_permanence.msgpack
  - data/{name}/components/a/track2_permanence.msgpack

Usage
-----
    python3 2.verify_and_process.py IMG_9282
    python3 2.verify_and_process.py 9282
"""

import os
import sys
import argparse
import msgpack
import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.video_io     import (
    component_track2_output_path,
    find_video,
    track1_output_path,
    track2_output_path,
    video_name,
)
from helper.verification import scan_bad_frames, verify_and_sanitize
from helper.permanence   import build_permanence_xya
from tracking_classes    import VideoCentroids


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_vc(path: str) -> VideoCentroids:
    with open(path, 'rb') as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def _save_msgpack(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(msgpack.packb(asdict(obj)))


def _unwrap_angle_component(track2a):
    arr = np.asarray(track2a.xPositions, dtype=float)
    # Unwrap each persistent block column independently over time using pi periodicity.
    for col in range(arr.shape[1]):
        mask = np.isfinite(arr[:, col])
        if np.any(mask):
            arr[mask, col] = np.unwrap(arr[mask, col], period=np.pi)
    track2a.xPositions = arr.tolist()
    return track2a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Track1 detections, repair bad frames, and build the permanence matrix."
    )
    parser.add_argument(
        'name',
        help="Video name or numeric suffix, e.g. IMG_9282 or 9282",
    )
    parser.add_argument(
        '--ratio-min', type=float, default=0.50,
        help="Min spacing ratio relative to reference (default 0.50).",
    )
    parser.add_argument(
        '--ratio-max', type=float, default=1.50,
        help="Max spacing ratio relative to reference (default 1.50).",
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
    t2_path = track2_output_path(name)
    t2_x_path = component_track2_output_path(name, "x")
    t2_y_path = component_track2_output_path(name, "y")
    t2_a_path = component_track2_output_path(name, "a")

    if not os.path.exists(t1_path):
        print(f"Error: track1 output not found: {t1_path}")
        print(f"  Run first: python3 .archive/1.track_run.py {name}")
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
        print(f"     (reference spacing: {ref_spacing:.4f} px, "
              f"ratio bounds: [{args.ratio_min}, {args.ratio_max}])")
        ans = input("\n  Apply automatic interpolation repair and continue? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted — no files written.")
            sys.exit(0)

        # Reload clean copy before repair (scan may have had side-effects via numpy views)
        vc = _load_vc(t1_path)
        print("\nRepairing…")
        vc, summary = verify_and_sanitize(
            vc,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            repair=True,
            quiet=False,
        )
        print(f"  Repaired {summary['sanitized_frames']} frame(s) "
              f"in {summary['sanitized_runs']} segment(s).")
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

    # ---- Build permanence matrix ----
    print("\nBuilding permanence matrices…")
    t2_x, t2_y, t2_a = build_permanence_xya(vc, quiet=False)
    t2_x.trackingResultsPath = t1_path
    t2_y.trackingResultsPath = t1_path
    t2_a.trackingResultsPath = t1_path
    t2_a = _unwrap_angle_component(t2_a)

    # ---- Save ----
    _save_msgpack(t2_path, t2_x)
    _save_msgpack(t2_x_path, t2_x)
    _save_msgpack(t2_y_path, t2_y)
    _save_msgpack(t2_a_path, t2_a)

    print(f"\nDone.")
    print(f"  Blocks   : {len(t2_x.blockColors)}  ({' '.join(t2_x.blockColors)})")
    print(f"  Frames   : {len(t2_x.xPositions)}")
    print(f"  Legacy X : {t2_path}")
    print(f"  X        : {t2_x_path}")
    print(f"  Y        : {t2_y_path}")
    print(f"  A        : {t2_a_path}")


if __name__ == '__main__':
    main()
