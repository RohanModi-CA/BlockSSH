from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from track.tracking_classes import VideoCentroids

from .bottom_manifest import load_manifest as _load_manifest, save_manifest as _save_manifest
from .bottom_permanence import build_permanence_xya
from .bottom_verification import scan_bad_frames, verify_and_sanitize
from .io import load_msgpack, save_msgpack
from .layout import (
    component_track2_path,
    track1_path,
)


def _load_vc(path: str | Path) -> VideoCentroids:
    return VideoCentroids.from_dict(load_msgpack(path))


def _unwrap_angle_component(track2a):
    arr = np.asarray(track2a.xPositions, dtype=float)
    # Unwrap each column (persistent block identity) independently over time
    for col in range(arr.shape[1]):
        mask = np.isfinite(arr[:, col])
        if np.any(mask):
            arr[mask, col] = np.unwrap(arr[mask, col], period=np.pi)
    track2a.xPositions = arr.tolist()
    return track2a


def run_process_verify(
    name: str,
    *,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
    trim_weak_ends: bool = True,
    min_end_support: int = 3,
) -> int:
    dataset = Path(name).stem
    manifest = _load_manifest(dataset)
    _save_manifest(dataset, manifest)
    t1_path = track1_path(dataset)
    if not t1_path.exists():
        print(f"Error: track1 output not found: {t1_path}")
        print(f"  Run first: python3 Bottom/1.TrackRun.py {dataset}")
        return 1

    print(f"Loading {t1_path}...")
    vc = _load_vc(t1_path)
    print(f"  {len(vc.frames)} frames loaded.")

    print("\nScanning for bad frames...")
    n_bad, n_seg, ref_spacing = scan_bad_frames(vc, ratio_min=ratio_min, ratio_max=ratio_max)

    if n_bad > 0:
        print(f"\n  Warning: {n_bad} bad frame(s) detected in {n_seg} segment(s).")
        print(
            f"     (reference spacing: {ref_spacing:.4f} px, "
            f"ratio bounds: [{ratio_min}, {ratio_max}])"
        )
        ans = input("\n  Apply automatic interpolation repair and continue? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted — no files written.")
            return 0

        vc = _load_vc(t1_path)
        print("\nRepairing...")
        try:
            vc, summary = verify_and_sanitize(
                vc,
                ratio_min=ratio_min,
                ratio_max=ratio_max,
                repair=True,
                quiet=False,
            )
        except RuntimeError as exc:
            print(f"Automatic repair was not sufficient: {exc}")
            print(f"Manual repair is required. Run: python3 Bottom/2b.ManualRepair.py {dataset}")
            return 1

        print(
            f"  Repaired {summary['sanitized_frames']} frame(s) "
            f"in {summary['sanitized_runs']} segment(s)."
        )
    else:
        print("  All frames passed — no repair needed.")
        vc, summary = verify_and_sanitize(
            vc,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            repair=False,
            quiet=True,
        )

    print(f"  Mean block distance: {summary['final_mean_block_distance']:.4f} px")
    print("\nBuilding X / Y / angle permanence matrices...")
    t2x, t2y, t2a, meta = build_permanence_xya(
        vc,
        tracking_results_path=str(t1_path),
        quiet=False,
        trim_weak_ends=trim_weak_ends,
        min_end_support=min_end_support,
    )
    t2a = _unwrap_angle_component(t2a)

    out_x = component_track2_path(dataset, "x")
    out_y = component_track2_path(dataset, "y")
    out_a = component_track2_path(dataset, "a")
    save_msgpack(out_x, t2x)
    save_msgpack(out_y, t2y)
    save_msgpack(out_a, t2a)

    manifest = _load_manifest(dataset)
    manifest.dataset = dataset
    manifest.track1_file = t1_path.name
    manifest.components = {
        "x": str(out_x.relative_to(out_x.parents[2])),
        "y": str(out_y.relative_to(out_y.parents[2])),
        "a": str(out_a.relative_to(out_a.parents[2])),
    }
    _save_manifest(dataset, manifest)

    print("\nDone.")
    print(f"  Columns kept : {meta['n_cols_kept']}  (full solution had {meta['n_cols_full']})")
    print(f"  Trimmed ends : left={meta['trimmed_left']}, right={meta['trimmed_right']}")
    print(f"  X output     : {out_x}")
    print(f"  Y output     : {out_y}")
    print(f"  A output     : {out_a}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify bottom black-blob track1 detections, repair bad segments if possible, "
            "and write X / Y / angle permanence datasets."
        )
    )
    parser.add_argument("name", help="Dataset name, e.g. IMG_9282 or 9282")
    parser.add_argument("--ratio-min", type=float, default=0.50, help="Min spacing ratio relative to reference.")
    parser.add_argument("--ratio-max", type=float, default=1.50, help="Max spacing ratio relative to reference.")
    parser.add_argument(
        "--no-trim-ends",
        action="store_true",
        help="Do not trim weak end columns.",
    )
    parser.add_argument(
        "--min-end-support",
        type=int,
        default=3,
        help="Minimum visible support required for end columns before they are kept.",
    )
    return parser
