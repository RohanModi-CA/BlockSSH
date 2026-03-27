#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import msgpack
import numpy as np

from tools.cli import add_track2_input_args
from tools.io import get_default_track_data_root, resolve_track2_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rescue a head-tail symmetric angle track2 dataset by unwrapping each "
            "block's orientation over time and writing a sibling dataset."
        )
    )
    add_track2_input_args(parser)
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Optional output dataset name. Default: append '_rescued' to the input dataset.",
    )
    parser.add_argument(
        "--output-track2",
        default=None,
        help="Optional explicit output path to the rescued track2_permanence.msgpack.",
    )
    parser.add_argument(
        "--suffix",
        default="_rescued",
        help="Suffix used when auto-generating the output dataset name. Default: _rescued",
    )
    parser.add_argument(
        "--no-recenter",
        action="store_true",
        help=(
            "Keep the unwrapped orientation as-is. By default, each column is shifted by "
            "an integer multiple of pi so its median stays near the original principal range."
        ),
    )
    return parser


def load_msgpack(path: Path) -> dict:
    with open(path, "rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False)


def save_msgpack(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))


def unwrap_orientation_column(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    idx = np.flatnonzero(finite)
    start = idx[0]
    prev = start
    for cur in idx[1:]:
        if cur != prev + 1:
            seg = out[start:prev + 1]
            out[start:prev + 1] = 0.5 * np.unwrap(2.0 * seg)
            start = cur
        prev = cur

    seg = out[start:prev + 1]
    out[start:prev + 1] = 0.5 * np.unwrap(2.0 * seg)
    return out


def recenter_orientation_column(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    median = float(np.median(out[finite]))
    out[finite] -= np.pi * np.round(median / np.pi)
    return out


def rescue_orientation_matrix(x_positions: np.ndarray, *, recenter: bool) -> np.ndarray:
    arr = np.asarray(x_positions, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"xPositions must be 2D. Got shape {arr.shape}")

    out = np.array(arr, dtype=float, copy=True)
    for col in range(out.shape[1]):
        out[:, col] = unwrap_orientation_column(out[:, col])
        if recenter:
            out[:, col] = recenter_orientation_column(out[:, col])
    return out


def count_large_jumps(x_positions: np.ndarray, threshold: float = np.pi / 2.0) -> int:
    arr = np.asarray(x_positions, dtype=float)
    if arr.ndim != 2:
        return 0
    jumps = 0
    for col in range(arr.shape[1]):
        col_vals = arr[:, col]
        finite = np.flatnonzero(np.isfinite(col_vals))
        if finite.size < 2:
            continue
        valid_prev = finite[:-1]
        valid_next = finite[1:]
        consec = (valid_next - valid_prev) == 1
        if not np.any(consec):
            continue
        delta = np.abs(col_vals[valid_next[consec]] - col_vals[valid_prev[consec]])
        jumps += int(np.sum(delta > threshold))
    return jumps


def auto_output_path(
    *,
    dataset: str | None,
    source_track2: Path,
    track_data_root: str | Path,
    output_dataset: str | None,
    suffix: str,
) -> Path:
    if output_dataset is not None:
        return Path(track_data_root) / output_dataset / "track2_permanence.msgpack"
    if dataset is not None:
        return Path(track_data_root) / f"{dataset}{suffix}" / "track2_permanence.msgpack"
    return source_track2.parent.parent / f"{source_track2.parent.name}{suffix}" / "track2_permanence.msgpack"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    src = resolve_track2_path(
        dataset=args.dataset,
        track2_path=args.track2,
        track_data_root=args.track_data_root,
    ).resolve()
    out = Path(args.output_track2).resolve() if args.output_track2 else auto_output_path(
        dataset=args.dataset,
        source_track2=src,
        track_data_root=args.track_data_root,
        output_dataset=args.output_dataset,
        suffix=str(args.suffix),
    ).resolve()

    payload = load_msgpack(src)
    x_positions = np.asarray(payload["xPositions"], dtype=float)
    rescued = rescue_orientation_matrix(x_positions, recenter=not args.no_recenter)

    payload["xPositions"] = rescued.tolist()
    payload["rescueMetadata"] = {
        "sourceTrack2Path": str(src),
        "mode": "head_tail_symmetric_orientation",
        "transform": "theta_rescued = 0.5 * unwrap(2 * theta)",
        "recenteredColumns": bool(not args.no_recenter),
    }

    before_jumps = count_large_jumps(x_positions)
    after_jumps = count_large_jumps(rescued)
    save_msgpack(out, payload)

    print(f"Source: {src}")
    print(f"Output: {out}")
    print(f"Columns: {rescued.shape[1]}")
    print(f"Frames: {rescued.shape[0]}")
    print(f"Large per-column jumps before: {before_jumps}")
    print(f"Large per-column jumps after : {after_jumps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
