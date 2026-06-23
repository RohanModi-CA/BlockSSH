#!/usr/bin/env python3
"""
Convert one tracking dataset directory into a legacy MATLAB-style .mat file.

This script is meant to make the new msgpack-based tracking output look like the
older MATLAB workspace format used by analysis scripts such as:

    t
    xR
    yR

while also preserving additional track2 information when available.

Input dataset layout
--------------------
DATASET/
    params_bottom.json                       (optional)
    track1.msgpack                           (optional)
    components/
        x/track2_permanence.msgpack          (optional)
        y/track2_permanence.msgpack          (optional)
        a/track2_permanence.msgpack          (optional)

Output .mat contents
--------------------
Legacy-compatible variables:
    t               [N x 1] double
    xR              [N x M] double
    yR              [N x M] double
    oR              [N x M] double      # angle/orientation if available
    xG              [N x 0] double      # empty placeholder for old scripts
    yG              [N x 0] double      # empty placeholder for old scripts

Useful extras:
    frameNumbers    [N x 1] double
    blockColors     [1 x M] object array of strings
    nFrames         scalar
    nBlocks         scalar
    datasetName     string
    datasetDir      string

Preserved track2 extras:
    track2_x
    track2_y
    track2_a
    params_bottom
    track1_summary

The track2_* structs contain:
    xPositions
    frameTimes_s
    frameNumbers
    blockColors
    originalVideoPath
    trackingResultsPath

Usage
-----
python convert_dataset_to_legacy_mat.py /path/to/DATASET
python convert_dataset_to_legacy_mat.py /path/to/DATASET -o /path/to/output.mat
python convert_dataset_to_legacy_mat.py /path/to/DATASET --no-track1
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
except Exception as exc:
    raise SystemExit(
        "Missing dependency 'msgpack'. Install with: pip install msgpack scipy numpy"
    ) from exc


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_msgpack(path: Path) -> Any:
    with path.open("rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False, strict_map_key=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def clean_scalar(x: Any) -> Any:
    if x is None:
        return np.nan
    if isinstance(x, (bool, int, float, np.number)):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, str):
        return x
    return x


def clean_for_mat(obj: Any) -> Any:
    """
    Recursively convert Python objects to MATLAB-safe values.
    """
    if obj is None:
        return np.nan

    if isinstance(obj, (bool, int, float, np.number, str)):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            # MATLAB struct field names must start with a letter and contain
            # only letters, digits, and underscores.
            safe = []
            for i, ch in enumerate(key):
                ok = ch.isalnum() or ch == "_"
                if i == 0:
                    ok = ok and (ch.isalpha() or ch == "_")
                safe.append(ch if ok else "_")
            safe_key = "".join(safe) or "field"
            if safe_key[0].isdigit():
                safe_key = f"f_{safe_key}"
            out[safe_key] = clean_for_mat(v)
        return out

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return np.empty((0, 0), dtype=float)

        # simple numeric vector
        if all((isinstance(v, (bool, int, float, np.number)) or v is None) for v in obj):
            return np.array([np.nan if v is None else float(v) for v in obj], dtype=float)

        # string list
        if all(isinstance(v, (str, bytes)) for v in obj):
            return np.array(
                [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in obj],
                dtype=object,
            )

        # nested numeric matrix
        if all(isinstance(v, (list, tuple)) for v in obj):
            numeric = True
            for row in obj:
                for item in row:
                    if not (isinstance(item, (bool, int, float, np.number)) or item is None):
                        numeric = False
                        break
                if not numeric:
                    break
            if numeric:
                return nested_numeric_matrix(obj)

        # fallback: object array
        return np.array([clean_for_mat(v) for v in obj], dtype=object)

    return str(obj)


def nested_numeric_matrix(values: Any) -> np.ndarray:
    """
    Convert list-of-lists numeric data into [N x M] double with NaNs for missing values.
    """
    if values is None:
        return np.empty((0, 0), dtype=float)

    if isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    if not isinstance(values, (list, tuple)):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    if len(values) == 0:
        return np.empty((0, 0), dtype=float)

    if all(not isinstance(v, (list, tuple)) for v in values):
        return np.asarray(values, dtype=float).reshape(-1, 1)

    n_rows = len(values)
    n_cols = max(len(row) if isinstance(row, (list, tuple)) else 1 for row in values)
    out = np.full((n_rows, n_cols), np.nan, dtype=float)

    for i, row in enumerate(values):
        if isinstance(row, (list, tuple)):
            for j, v in enumerate(row):
                out[i, j] = np.nan if v is None else float(v)
        else:
            out[i, 0] = np.nan if row is None else float(row)

    return out


def as_float_col(values: Any) -> np.ndarray:
    arr = nested_numeric_matrix(values)
    if arr.ndim != 2:
        arr = np.asarray(arr, dtype=float).reshape(-1, 1)
    if arr.shape[1] != 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float, copy=False)


def as_string_row(values: Any) -> np.ndarray:
    if values is None:
        return np.empty((0,), dtype=object)
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1)
        return np.array([str(v) for v in flat], dtype=object)
    if not isinstance(values, (list, tuple)):
        return np.array([str(values)], dtype=object)
    return np.array([str(v) for v in values], dtype=object)


def load_track2_component(path: Path, name: str) -> dict[str, Any]:
    raw = load_msgpack(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not decode to a dict")

    comp = {
        "name": name,
        "sourceFile": str(path),
        "xPositions": nested_numeric_matrix(raw.get("xPositions", [])),
        "frameTimes_s": as_float_col(raw.get("frameTimes_s", [])),
        "frameNumbers": as_float_col(raw.get("frameNumbers", [])),
        "blockColors": as_string_row(raw.get("blockColors", [])),
        "originalVideoPath": str(raw.get("originalVideoPath", "")),
        "trackingResultsPath": str(raw.get("trackingResultsPath", "")),
    }

    X = comp["xPositions"]
    comp["nFrames"] = float(X.shape[0])
    comp["nBlocks"] = float(X.shape[1]) if X.ndim == 2 else 0.0
    comp["finiteCount"] = float(np.isfinite(X).sum()) if X.size else 0.0
    return comp


def summarize_track1(track1: dict[str, Any]) -> dict[str, Any]:
    frames = track1.get("frames", [])
    if not isinstance(frames, list):
        frames = []

    n_frames = len(frames)
    frame_numbers = np.full((n_frames, 1), np.nan, dtype=float)
    frame_times = np.full((n_frames, 1), np.nan, dtype=float)
    det_counts = np.zeros((n_frames, 1), dtype=float)

    max_dets = 0
    colors_seen: list[str] = []

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        if "frame_number" in frame and frame["frame_number"] is not None:
            frame_numbers[i, 0] = float(frame["frame_number"])
        if "frame_time_s" in frame and frame["frame_time_s"] is not None:
            frame_times[i, 0] = float(frame["frame_time_s"])

        dets = frame.get("detections", [])
        if not isinstance(dets, list):
            dets = []
        det_counts[i, 0] = float(len(dets))
        max_dets = max(max_dets, len(dets))

        for det in dets:
            if isinstance(det, dict) and "color" in det and det["color"] is not None:
                c = str(det["color"])
                if c not in colors_seen:
                    colors_seen.append(c)

    out = {
        "nFrames": float(n_frames),
        "maxDetectionsPerFrame": float(max_dets),
        "frameNumbers": frame_numbers,
        "frameTimes_s": frame_times,
        "detectionCountPerFrame": det_counts,
        "colorsSeen": np.array(colors_seen, dtype=object),
    }

    good = np.isfinite(det_counts[:, 0])
    if np.any(good):
        vals = det_counts[good, 0]
        out["minDetectionsPerFrame"] = float(np.min(vals))
        out["medianDetectionsPerFrame"] = float(np.median(vals))
        out["maxDetectionsPerFrameObserved"] = float(np.max(vals))
    return out


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def build_legacy_output(dataset_dir: Path, include_track1: bool = True) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()

    params_path = dataset_dir / "params_bottom.json"
    track1_path = dataset_dir / "track1.msgpack"
    x_path = dataset_dir / "components" / "x" / "track2_permanence.msgpack"
    y_path = dataset_dir / "components" / "y" / "track2_permanence.msgpack"
    a_path = dataset_dir / "components" / "a" / "track2_permanence.msgpack"

    out: dict[str, Any] = {}

    # Load track2 components
    track2_x = load_track2_component(x_path, "x") if x_path.is_file() else {}
    track2_y = load_track2_component(y_path, "y") if y_path.is_file() else {}
    track2_a = load_track2_component(a_path, "a") if a_path.is_file() else {}

    if not (track2_x or track2_y or track2_a):
        raise RuntimeError(
            "No track2 permanence files were found under components/x|y|a."
        )

    canonical = next((c for c in (track2_x, track2_y, track2_a) if c), None)
    assert canonical is not None

    t = canonical.get("frameTimes_s", np.empty((0, 1), dtype=float))
    frame_numbers = canonical.get("frameNumbers", np.empty((0, 1), dtype=float))
    block_colors = canonical.get("blockColors", np.empty((0,), dtype=object))

    xR = track2_x.get("xPositions", np.empty((0, 0), dtype=float)) if track2_x else np.empty((0, 0), dtype=float)
    yR = track2_y.get("xPositions", np.empty((0, 0), dtype=float)) if track2_y else np.empty((0, 0), dtype=float)
    oR = track2_a.get("xPositions", np.empty((0, 0), dtype=float)) if track2_a else np.empty((0, 0), dtype=float)

    # Legacy placeholders
    n_frames = int(max(
        t.shape[0] if isinstance(t, np.ndarray) and t.ndim >= 1 else 0,
        xR.shape[0] if isinstance(xR, np.ndarray) and xR.ndim == 2 else 0,
        yR.shape[0] if isinstance(yR, np.ndarray) and yR.ndim == 2 else 0,
        oR.shape[0] if isinstance(oR, np.ndarray) and oR.ndim == 2 else 0,
    ))
    xG = np.empty((n_frames, 0), dtype=float)
    yG = np.empty((n_frames, 0), dtype=float)

    # Legacy variables
    out["t"] = as_float_col(t)
    out["xR"] = np.asarray(xR, dtype=float)
    out["yR"] = np.asarray(yR, dtype=float)
    out["oR"] = np.asarray(oR, dtype=float)
    out["xG"] = xG
    out["yG"] = yG

    # Useful extras
    out["frameNumbers"] = as_float_col(frame_numbers)
    out["blockColors"] = as_string_row(block_colors)
    out["nFrames"] = float(n_frames)
    out["nBlocks"] = float(out["xR"].shape[1] if out["xR"].ndim == 2 else 0)
    out["datasetName"] = dataset_dir.name
    out["datasetDir"] = str(dataset_dir)

    # Preserved track2 component info
    out["track2_x"] = track2_x
    out["track2_y"] = track2_y
    out["track2_a"] = track2_a

    # Parameters
    out["params_bottom"] = clean_for_mat(read_json(params_path)) if params_path.is_file() else {}

    # Optional raw track1 summary only, not the entire raw structure
    if include_track1 and track1_path.is_file():
        track1 = load_msgpack(track1_path)
        if isinstance(track1, dict):
            out["track1_summary"] = summarize_track1(track1)
        else:
            out["track1_summary"] = {}
    else:
        out["track1_summary"] = {}

    return clean_for_mat(out)


def print_summary(data: dict[str, Any]) -> None:
    print(f"Dataset: {data['datasetName']}")
    print(f"Folder : {data['datasetDir']}")
    print(f"Frames : {int(data['nFrames'])}")
    print(f"Blocks : {int(data['nBlocks'])}")

    print(f"xR     : {tuple(data['xR'].shape)}")
    print(f"yR     : {tuple(data['yR'].shape)}")
    print(f"oR     : {tuple(data['oR'].shape)}")
    print(f"t      : {tuple(data['t'].shape)}")

    bc = data.get("blockColors", np.empty((0,), dtype=object))
    if len(bc):
        print("blockColors:", ", ".join(str(v) for v in bc))

    for name in ("track2_x", "track2_y", "track2_a"):
        comp = data.get(name, {})
        if comp:
            print(
                f"{name}: frames={int(comp.get('nFrames', 0))}, "
                f"blocks={int(comp.get('nBlocks', 0))}, "
                f"finite={int(comp.get('finiteCount', 0))}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert one msgpack tracking dataset to a legacy MATLAB-style MAT file."
    )
    p.add_argument("dataset_dir", type=Path, help="Path to dataset directory")
    p.add_argument(
        "-o", "--output", type=Path,
        help="Output MAT path (default: DATASET_DIR/DATASET_NAME_legacy.mat)"
    )
    p.add_argument(
        "--no-track1", action="store_true",
        help="Do not read track1.msgpack for summary metadata"
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_dir}")

    out_path = args.output or (dataset_dir / f"{dataset_dir.name}_legacy.mat")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_legacy_output(dataset_dir, include_track1=not args.no_track1)
    print_summary(data)

    # Save variables directly at top level, matching old MATLAB style.
    savemat(str(out_path), data, do_compression=True, long_field_names=True)
    print(f"\nWrote MAT file:\n  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
