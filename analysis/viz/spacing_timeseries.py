#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys

import numpy as np

from plotting.common import render_figure
from plotting.trajectory import plot_block_timeseries, plot_spacing_timeseries
from tools.cli import add_output_args, add_track2_input_args
from tools.derived import derive_spacing_dataset
from tools.io import load_track2_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stacked block-spacing time series derived from Track2 permanence.",
    )
    add_track2_input_args(parser)
    add_output_args(parser, include_title=True)
    parser.add_argument(
        "--timeseriesnorm",
        action="store_true",
        help="Rescale each displayed spacing time series to RMS 100.",
    )
    parser.add_argument(
        "--not-bonds",
        action="store_true",
        help="Plot raw block x-position traces instead of adjacent bond-spacing traces.",
    )
    return parser


def _normalize_timeseries_columns_to_rms(matrix: np.ndarray, *, target_rms: float = 100.0) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    out = matrix.copy()

    for col in range(out.shape[1]):
        y = out[:, col]
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        centered = y[finite] - float(np.mean(y[finite]))
        rms = float(np.sqrt(np.mean(np.square(centered))))
        if not np.isfinite(rms) or rms <= 0:
            continue
        out[finite, col] = centered * (float(target_rms) / rms)

    return out


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        track2 = load_track2_dataset(
            dataset=args.dataset,
            track2_path=args.track2,
            track_data_root=args.track_data_root,
        )
        spacing = derive_spacing_dataset(track2)
        if args.not_bonds:
            block_matrix = (
                _normalize_timeseries_columns_to_rms(track2.x_positions)
                if args.timeseriesnorm
                else track2.x_positions
            )

            print(f"Track2: {track2.track2_path}")
            print(f"Blocks: {track2.x_positions.shape[1]}")
            print(f"Frames: {track2.x_positions.shape[0]}")

            fig = plot_block_timeseries(
                track2.frame_times_s,
                block_matrix,
                track2.block_colors,
                title=args.title,
            )
        else:
            spacing_matrix = (
                _normalize_timeseries_columns_to_rms(spacing.spacing_matrix)
                if args.timeseriesnorm
                else spacing.spacing_matrix
            )

            print(f"Track2: {track2.track2_path}")
            print(f"Pairs: {spacing.spacing_matrix.shape[1]}")
            print(f"Frames: {spacing.spacing_matrix.shape[0]}")

            fig = plot_spacing_timeseries(
                track2.frame_times_s,
                spacing_matrix,
                spacing.pair_labels,
                title=args.title,
            )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
