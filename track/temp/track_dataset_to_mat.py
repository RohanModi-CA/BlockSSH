#!/usr/bin/env python3
"""
Convert one tracking dataset directory from the Python/msgpack pipeline into a .mat file.

Expected dataset structure
--------------------------
DATASET/
    params_bottom.json                       (optional)
    track1.msgpack                           (optional)
    components/
        x/track2_permanence.msgpack          (optional)
        y/track2_permanence.msgpack          (optional)
        a/track2_permanence.msgpack          (optional)

Usage
-----
python track_dataset_to_mat.py /path/to/DATASET
python track_dataset_to_mat.py /path/to/DATASET -o /path/to/output.mat
python track_dataset_to_mat.py /path/to/DATASET --skip-track1
python track_dataset_to_mat.py /path/to/DATASET --pretty-json-sidecar

Notes
-----
- Saves verified track2 components as numeric matrices:
      track2.x
      track2.y
      track2.a
- Preserves metadata such as frame times, frame numbers, block colors, and paths.
- Also saves a simplified copy of track1 detections if track1.msgpack exists.
- The .mat file is written in a form that scipy.io.loadmat / MATLAB can read directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

try:
    import msgpack
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires the Python package 'msgpack'. "
        "Install it with: pip install msgpack scipy numpy"
    ) from exc


def load_msgpack(path: Path) -> Any:
    with path.open("rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)


def matlab_safe_string_list(values: list[Any]) -> np.ndarray:
    return np.array([str(v) for v in values], dtype=object)


def as_float_column(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1, 1)


def nested_numeric_matrix(values: Any) -> np.ndarray:
    """
    Convert list-of-lists numeric data into a 2D float matrix.
    Missing values become NaN.
    """
    if values is None:
        return np.empty((0, 0), dtype=float)

    if isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    if not isinstance(values, (list, tuple)):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    if len(values) == 0:
        return np.empty((0, 0), dtype=float)

    if all(not isinstance(row, (list, tuple)) for row in values):
        return np.asarray(values, dtype=float).reshape(-1, 1)

    n_rows = len(values)
    n_cols = max(len(row) if isinstance(row, (list, tuple)) else 1 for row in values)
    out = np.full((n_rows, n_cols), np.nan, dtype=float)

    for i, row in enumerate(values):
        if isinstance(row, (list, tuple)):
            for j, val in enumerate(row):
                if val is None:
                    out[i, j] = np.nan
                else:
                    out[i, j] = float(val)
        else:
            out[i, 0] = float(row)
    return out


def to_matlab_value(obj: Any) -> Any:
    """
    Recursively convert Python/msgpack structures to MATLAB-friendly values.
    """
    if obj is None:
        return np.nan

    if isinstance(obj, (bool, int, float, np.number)):
        return obj

    if isinstance(obj, str):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, dict):
        return {str(k): to_matlab_value(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return np.empty((0, 0), dtype=float)

        # Numeric vector
        if all(isinstance(x, (bool, int, float, np.number)) or x is None for x in obj):
            return np.array([np.nan if x is None else float(x) for x in obj], dtype=float)

        # List of strings
        if all(isinstance(x, str) for x in obj):
            return np.array(obj, dtype=object)

        # List of dicts -> object array of converted dicts
        if all(isinstance(x, dict) for x in obj):
            return np.array([to_matlab_value(x) for x in obj], dtype=object)

        # Nested numeric rows -> 2D matrix
        if all(isinstance(x, (list, tuple)) for x in obj):
            maybe_numeric = True
            for row in obj:
                for item in row:
                    if not (isinstance(item, (bool, int, float, np.number)) or item is None):
                        maybe_numeric = False
                        break
                if not maybe_numeric:
                    break
            if maybe_numeric:
                return nested_numeric_matrix(obj)

        return np.array([to_matlab_value(x) for x in obj], dtype=object)

    return str(obj)


def summarize_component(name: str, component: dict[str, Any]) -> dict[str, Any]:
    matrix = nested_numeric_matrix(component.get("xPositions", []))
    summary: dict[str, Any] = {
        "name": name,
        "n_frames": int(matrix.shape[0]),
        "n_blocks": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "finite_entries": int(np.isfinite(matrix).sum()) if matrix.size else 0,
    }
    if matrix.size:
        support = np.sum(np.isfinite(matrix), axis=0)
        summary["support_per_block"] = support.astype(float)
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            summary["global_min"] = float(np.min(finite))
            summary["global_max"] = float(np.max(finite))
            summary["global_mean"] = float(np.mean(finite))
    return summary


def simplify_track1(track1: dict[str, Any]) -> dict[str, Any]:
    """
    Build MATLAB-friendly arrays from track1 frame/detection data.
    Keeps the original converted object too, but this gives easy-access matrices.
    """
    frames = track1.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    frame_numbers = []
    frame_times = []
    det_counts = []
    max_dets = 0

    for frame in frames:
        dets = frame.get("detections", []) if isinstance(frame, dict) else []
        if not isinstance(dets, list):
            dets = []
        max_dets = max(max_dets, len(dets))

    x = np.full((len(frames), max_dets), np.nan, dtype=float)
    y = np.full((len(frames), max_dets), np.nan, dtype=float)
    angle = np.full((len(frames), max_dets), np.nan, dtype=float)
    area = np.full((len(frames), max_dets), np.nan, dtype=float)
    color = np.empty((len(frames), max_dets), dtype=object)
    color[:] = ""

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            frame = {}
        frame_numbers.append(frame.get("frame_number", np.nan))
        frame_times.append(frame.get("frame_time_s", np.nan))
        dets = frame.get("detections", [])
        if not isinstance(dets, list):
            dets = []
        det_counts.append(len(dets))

        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            if "x" in det and det["x"] is not None:
                x[i, j] = float(det["x"])
            if "y" in det and det["y"] is not None:
                y[i, j] = float(det["y"])
            if "angle" in det and det["angle"] is not None:
                angle[i, j] = float(det["angle"])
            if "area" in det and det["area"] is not None:
                area[i, j] = float(det["area"])
            if "color" in det and det["color"] is not None:
                color[i, j] = str(det["color"])

    out = {
        "n_frames": float(len(frames)),
        "max_detections_per_frame": float(max_dets),
        "frame_numbers": np.asarray(frame_numbers, dtype=float).reshape(-1, 1),
        "frame_times_s": np.asarray(frame_times, dtype=float).reshape(-1, 1),
        "detection_count_per_frame": np.asarray(det_counts, dtype=float).reshape(-1, 1),
        "x": x,
        "y": y,
        "angle": angle,
        "area": area,
        "color": color,
        "raw": to_matlab_value(track1),
    }
    return out


def load_component(path: Path, component_name: str) -> dict[str, Any]:
    raw = load_msgpack(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not decode to a dict-like object")

    matrix = nested_numeric_matrix(raw.get("xPositions", []))
    block_colors = raw.get("blockColors", [])
    if not isinstance(block_colors, list):
        block_colors = []

    out: dict[str, Any] = {
        "name": component_name,
        "source_file": str(path),
        "xPositions": matrix,
        "frameTimes_s": as_float_column(raw.get("frameTimes_s", [])),
        "frameNumbers": as_float_column(raw.get("frameNumbers", [])),
        "blockColors": matlab_safe_string_list(block_colors),
        "originalVideoPath": str(raw.get("originalVideoPath", "")),
        "trackingResultsPath": str(raw.get("trackingResultsPath", "")),
        "raw": to_matlab_value(raw),
    }
    out["summary"] = summarize_component(component_name, out)
    return out


def build_output_dict(dataset_dir: Path, skip_track1: bool) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()

    paths = {
        "dataset_dir": str(dataset_dir),
        "params_bottom_json": str(dataset_dir / "params_bottom.json"),
        "track1_msgpack": str(dataset_dir / "track1.msgpack"),
        "track2_x_msgpack": str(dataset_dir / "components" / "x" / "track2_permanence.msgpack"),
        "track2_y_msgpack": str(dataset_dir / "components" / "y" / "track2_permanence.msgpack"),
        "track2_a_msgpack": str(dataset_dir / "components" / "a" / "track2_permanence.msgpack"),
    }

    out: dict[str, Any] = {
        "dataset_name": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "paths": paths,
    }

    params_path = dataset_dir / "params_bottom.json"
    if params_path.is_file():
        out["params_bottom"] = to_matlab_value(json.loads(params_path.read_text()))
    else:
        out["params_bottom"] = {}

    components: dict[str, Any] = {}
    for name in ("x", "y", "a"):
        path = dataset_dir / "components" / name / "track2_permanence.msgpack"
        if path.is_file():
            components[name] = load_component(path, name)
        else:
            components[name] = {}

    out["track2"] = components

    canonical = next((components[k] for k in ("x", "y", "a") if components[k]), None)
    if canonical:
        out["frameTimes_s"] = canonical.get("frameTimes_s", np.empty((0, 1), dtype=float))
        out["frameNumbers"] = canonical.get("frameNumbers", np.empty((0, 1), dtype=float))
        out["blockColors"] = canonical.get("blockColors", np.empty((0,), dtype=object))

    if not skip_track1:
        track1_path = dataset_dir / "track1.msgpack"
        if track1_path.is_file():
            out["track1"] = simplify_track1(load_msgpack(track1_path))
        else:
            out["track1"] = {}

    # Convenience aliases for MATLAB users
    out["X"] = components["x"].get("xPositions", np.empty((0, 0), dtype=float)) if components["x"] else np.empty((0, 0), dtype=float)
    out["Y"] = components["y"].get("xPositions", np.empty((0, 0), dtype=float)) if components["y"] else np.empty((0, 0), dtype=float)
    out["A"] = components["a"].get("xPositions", np.empty((0, 0), dtype=float)) if components["a"] else np.empty((0, 0), dtype=float)

    return out


def print_summary(data: dict[str, Any]) -> None:
    print(f"Dataset: {data['dataset_name']}")
    print(f"Folder : {data['dataset_dir']}")

    params = data.get("params_bottom", {})
    if params:
        print("Params : found params_bottom.json")
    else:
        print("Params : not found")

    for name in ("x", "y", "a"):
        comp = data["track2"].get(name, {})
        if not comp:
            print(f"track2 {name}: missing")
            continue
        s = comp["summary"]
        print(
            f"track2 {name}: {s['n_frames']} frames x {s['n_blocks']} blocks, "
            f"finite={s['finite_entries']}"
        )

    track1 = data.get("track1", {})
    if track1:
        print(
            f"track1    : {int(track1['n_frames'])} frames, "
            f"max det/frame={int(track1['max_detections_per_frame'])}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one tracking dataset directory to a MAT file.")
    parser.add_argument("dataset_dir", type=Path, help="Path to one dataset directory")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output .mat path (default: DATASET_DIR/DATASET_NAME_converted.mat)"
    )
    parser.add_argument(
        "--skip-track1", action="store_true",
        help="Do not parse track1.msgpack even if it exists"
    )
    parser.add_argument(
        "--pretty-json-sidecar", action="store_true",
        help="Also write a small JSON summary sidecar next to the MAT file"
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_dir}")

    if args.output is None:
        out_path = dataset_dir / f"{dataset_dir.name}_converted.mat"
    else:
        out_path = args.output

    data = build_output_dict(dataset_dir, skip_track1=args.skip_track1)
    print_summary(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    matlab_data = to_matlab_value(data)
    savemat(str(out_path), {"dataset": matlab_data}, do_compression=True, long_field_names=True)
    print(f"\nWrote MAT file:\n  {out_path}")

    if args.pretty_json_sidecar:
        sidecar = out_path.with_suffix(".summary.json")
        summary = {
            "dataset_name": data["dataset_name"],
            "dataset_dir": data["dataset_dir"],
            "track2": {
                name: (data["track2"][name]["summary"] if data["track2"].get(name) else None)
                for name in ("x", "y", "a")
            },
            "has_track1": bool(data.get("track1")),
            "has_params_bottom": bool(data.get("params_bottom")),
        }
        sidecar.write_text(json.dumps(summary, indent=2))
        print(f"Wrote JSON summary:\n  {sidecar}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
