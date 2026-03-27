#!/usr/bin/env python3
"""
2.export_area_from_existing_black.py

Build an `_area` permanence dataset from existing black-video outputs without
rerunning tracking or modifying the normal X / Y / angle pipeline.

Inputs
------
- An existing X-permanence dataset, e.g. data/CDX_10IC_x/track2_permanence.msgpack
- The corresponding track1.msgpack referenced by that X-permanence file

Output
------
- data/<name>_area/track2_permanence.msgpack

The output keeps the legacy Track2 schema and stores area values in
`xPositions`, matching how `_x`, `_y`, and `_a` reuse the same container.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import msgpack
import numpy as np

from helper.video_io import DATA_DIR, dataset_dir
from tracking_classes import Track2XPermanence, VideoCentroids


def _load_msgpack(path: str) -> dict:
    with open(path, "rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False)


def _load_vc(path: str) -> VideoCentroids:
    return VideoCentroids.from_dict(_load_msgpack(path))


def _save_msgpack(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(msgpack.packb(asdict(obj), use_bin_type=True))


def _resolve_x_track2_path(name_or_dataset: str) -> str:
    token = os.path.splitext(os.path.basename(name_or_dataset))[0]
    if token.endswith(".msgpack"):
        return token

    if token == "track2_permanence":
        return os.path.abspath(name_or_dataset)

    candidates = []
    if token.endswith("_x"):
        candidates.append(os.path.join(DATA_DIR, token, "track2_permanence.msgpack"))
    else:
        candidates.append(os.path.join(DATA_DIR, f"{token}_x", "track2_permanence.msgpack"))
        candidates.append(os.path.join(DATA_DIR, token, "track2_permanence.msgpack"))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not resolve an X permanence dataset from "
        f"'{name_or_dataset}'. Tried: {', '.join(candidates)}"
    )


def _resolve_track1_path(x_track2_payload: dict, x_dataset_name: str) -> str:
    if x_dataset_name.endswith("_x"):
        preferred = os.path.join(DATA_DIR, x_dataset_name[:-2], "track1.msgpack")
        if os.path.exists(preferred):
            return preferred

    path = str(x_track2_payload.get("trackingResultsPath", "")).strip()
    if not path:
        raise FileNotFoundError("X permanence file does not contain trackingResultsPath.")
    if os.path.exists(path):
        return path
    local = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(local):
        return local
    raise FileNotFoundError(f"Referenced track1 file not found: {path}")


def _infer_output_dataset_name(x_dataset_name: str, output_name: str | None) -> str:
    if output_name is not None:
        return output_name
    if x_dataset_name.endswith("_x"):
        return f"{x_dataset_name[:-2]}_area"
    return f"{x_dataset_name}_area"


def _frame_area_row(
    detections,
    x_row: np.ndarray,
    *,
    tol: float,
    frame_idx: int,
) -> list[float]:
    row = [float("nan")] * int(x_row.shape[0])

    ordered = sorted(detections, key=lambda d: float(d.x))
    visible_cols = np.flatnonzero(np.isfinite(x_row))
    x_visible = x_row[visible_cols]

    if len(ordered) != int(visible_cols.size):
        raise RuntimeError(
            f"Frame {frame_idx}: detection count {len(ordered)} does not match "
            f"visible permanence columns {int(visible_cols.size)}."
        )

    if ordered:
        det_x = np.array([float(d.x) for d in ordered], dtype=float)
        max_abs = float(np.max(np.abs(det_x - x_visible)))
        if max_abs > tol:
            raise RuntimeError(
                f"Frame {frame_idx}: X mismatch between track1 and X permanence "
                f"(max abs diff {max_abs:.6g} > tolerance {tol:.6g})."
            )

    for det, col in zip(ordered, visible_cols.tolist()):
        row[col] = float(det.area)

    return row


def build_area_track2(
    vc: VideoCentroids,
    x_track2: dict,
    *,
    tol: float,
) -> tuple[Track2XPermanence, dict]:
    x_positions = np.asarray(x_track2["xPositions"], dtype=float)
    frame_times = list(x_track2["frameTimes_s"])
    frame_numbers = list(x_track2.get("frameNumbers", []))
    block_colors = list(x_track2["blockColors"])

    if x_positions.ndim != 2:
        raise ValueError(f"X permanence xPositions must be 2D. Got shape {x_positions.shape}")
    if x_positions.shape[0] != len(vc.frames):
        raise ValueError(
            f"Row mismatch: X permanence has {x_positions.shape[0]} frames but track1 has {len(vc.frames)}."
        )

    area_rows: list[list[float]] = []
    visible_matches = 0
    for frame_idx, frame in enumerate(vc.frames):
        row = _frame_area_row(frame.detections, x_positions[frame_idx], tol=tol, frame_idx=frame_idx)
        visible_matches += int(np.sum(np.isfinite(row)))
        area_rows.append(row)

    area_track2 = Track2XPermanence(
        originalVideoPath=str(x_track2.get("originalVideoPath", vc.filepath)),
        trackingResultsPath=str(x_track2.get("trackingResultsPath", "")),
        blockColors=block_colors,
        xPositions=area_rows,
        frameTimes_s=frame_times,
        frameNumbers=frame_numbers if frame_numbers else [f.frame_number for f in vc.frames],
    )

    area_arr = np.asarray(area_rows, dtype=float)
    finite = area_arr[np.isfinite(area_arr)]
    meta = {
        "visible_matches": int(visible_matches),
        "n_frames": int(area_arr.shape[0]),
        "n_cols": int(area_arr.shape[1]),
        "area_min": float(np.min(finite)) if finite.size else float("nan"),
        "area_max": float(np.max(finite)) if finite.size else float("nan"),
        "area_median": float(np.median(finite)) if finite.size else float("nan"),
    }
    return area_track2, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export an `_area` permanence dataset from an existing black-video X permanence file "
            "and its referenced track1.msgpack."
        )
    )
    parser.add_argument(
        "name",
        help=(
            "Dataset token used to locate the X permanence file. Examples: CDX_10IC, "
            "CDX_10IC_x, or an explicit path to track2_permanence.msgpack."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output dataset name. Default: input with `_area` suffix.",
    )
    parser.add_argument(
        "--x-tol",
        type=float,
        default=1e-3,
        help="Maximum allowed absolute x mismatch when aligning track1 detections to permanence columns.",
    )
    args = parser.parse_args()

    x_track2_path = _resolve_x_track2_path(args.name)
    x_track2 = _load_msgpack(x_track2_path)
    x_dataset_name = os.path.basename(os.path.dirname(x_track2_path))
    track1_path = _resolve_track1_path(x_track2, x_dataset_name)
    vc = _load_vc(track1_path)

    output_dataset_name = _infer_output_dataset_name(x_dataset_name, args.output_name)
    output_path = os.path.join(dataset_dir(output_dataset_name), "track2_permanence.msgpack")

    area_track2, meta = build_area_track2(vc, x_track2, tol=float(args.x_tol))
    _save_msgpack(output_path, area_track2)

    print(f"X input     : {x_track2_path}")
    print(f"Track1 input: {track1_path}")
    print(f"Area output : {output_path}")
    print(f"Frames      : {meta['n_frames']}")
    print(f"Columns     : {meta['n_cols']}")
    print(f"Matches     : {meta['visible_matches']}")
    print(
        "Area stats  : "
        f"min={meta['area_min']:.3f}  median={meta['area_median']:.3f}  max={meta['area_max']:.3f}"
    )


if __name__ == "__main__":
    main()
