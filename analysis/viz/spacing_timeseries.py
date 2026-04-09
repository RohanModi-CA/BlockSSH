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
from tools.bonds import load_bond_signal_dataset
from tools.cli import add_bond_spacing_mode_arg, add_output_args, add_track2_input_args
from tools.io import load_track2_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stacked block-spacing time series derived from Track2 permanence.",
    )
    add_track2_input_args(parser)
    add_bond_spacing_mode_arg(parser)
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
    parser.add_argument(
        "--only-index",
        type=int,
        default=None,
        help="Show only one 1-indexed site/block or bond/pair, depending on mode.",
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
        if args.only_index is not None and int(args.only_index) < 1:
            raise ValueError("--only-index must be a positive 1-indexed site/block or bond/pair number")

        track2 = load_track2_dataset(
            dataset=args.dataset,
            track2_path=args.track2,
            track_data_root=args.track_data_root,
        )
        if args.not_bonds:
            block_matrix = (
                _normalize_timeseries_columns_to_rms(track2.x_positions)
                if args.timeseriesnorm
                else track2.x_positions
            )
            block_labels = list(track2.block_colors)
            block_indices = [idx + 1 for idx in range(block_matrix.shape[1])]
            if args.only_index is not None:
                block_zero = int(args.only_index) - 1
                if block_zero >= block_matrix.shape[1]:
                    raise ValueError(
                        f"--only-index {args.only_index} is out of range for {block_matrix.shape[1]} blocks"
                    )
                block_matrix = block_matrix[:, [block_zero]]
                block_labels = [block_labels[block_zero]]
                block_indices = [int(args.only_index)]

            print(f"Track2: {track2.track2_path}")
            print(f"Blocks: {track2.x_positions.shape[1]}")
            if args.only_index is not None:
                print(f"Selected block: {args.only_index}")
            print(f"Frames: {track2.x_positions.shape[0]}")

            fig = plot_block_timeseries(
                track2.frame_times_s,
                block_matrix,
                block_labels,
                title=args.title,
                series_indices=block_indices,
            )
        else:
            bond_dataset = load_bond_signal_dataset(
                dataset=args.dataset,
                track2_path=args.track2,
                track_data_root=args.track_data_root,
                bond_spacing_mode=args.bond_spacing_mode,
            )
            spacing_matrix = (
                _normalize_timeseries_columns_to_rms(bond_dataset.signal_matrix)
                if args.timeseriesnorm
                else bond_dataset.signal_matrix
            )
            pair_labels = list(bond_dataset.pair_labels)
            pair_indices = [idx + 1 for idx in range(spacing_matrix.shape[1])]
            if args.only_index is not None:
                pair_zero = int(args.only_index) - 1
                if pair_zero >= spacing_matrix.shape[1]:
                    raise ValueError(
                        f"--only-index {args.only_index} is out of range for {spacing_matrix.shape[1]} bonds/pairs"
                    )
                spacing_matrix = spacing_matrix[:, [pair_zero]]
                pair_labels = [pair_labels[pair_zero]]
                pair_indices = [int(args.only_index)]

            print(f"Track2: {bond_dataset.source_path}")
            print(f"Pairs: {bond_dataset.signal_matrix.shape[1]}")
            if args.only_index is not None:
                print(f"Selected pair: {args.only_index}")
            print(f"Frames: {bond_dataset.signal_matrix.shape[0]}")

            fig = plot_spacing_timeseries(
                bond_dataset.frame_times_s,
                spacing_matrix,
                pair_labels,
                title=args.title,
                series_indices=pair_indices,
            )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
