#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

import numpy as np

from plotting.common import render_figure
from plotting.frequency import plot_pair_welch_frequency_grid
from tools.cli import add_colormap_arg, add_output_args, add_signal_processing_args, add_track2_input_args
from tools.derived import derive_spacing_dataset
from tools.io import load_track2_dataset
from tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from tools.spectral import analyze_spacing_dataset_with_welch_for_display


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Welch spectrum and sliding FFT of Track2-derived block spacing.",
    )
    add_track2_input_args(parser)
    add_signal_processing_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    welch_group = parser.add_mutually_exclusive_group()
    welch_group.add_argument(
        "--welch-log",
        dest="welch_log",
        action="store_true",
        help="Use a log y-axis for the Welch curve panel (default).",
    )
    welch_group.add_argument(
        "--welch-linear",
        dest="welch_log",
        action="store_false",
        help="Use a linear y-axis for the Welch curve panel.",
    )
    parser.set_defaults(welch_log=True)

    sliding_group = parser.add_mutually_exclusive_group()
    sliding_group.add_argument(
        "--sliding-log",
        dest="sliding_plot_scale",
        action="store_const",
        const="log",
        help="Use dB display for the sliding panel (default).",
    )
    sliding_group.add_argument(
        "--sliding-linear",
        dest="sliding_plot_scale",
        action="store_const",
        const="linear",
        help="Use linear power/amplitude display for the sliding panel.",
    )
    parser.set_defaults(sliding_plot_scale="log")

    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--sliding-len-s", type=float, default=20.0)
    parser.add_argument("--welch-min-hz", type=float, default=None)
    parser.add_argument("--welch-max-hz", type=float, default=None)
    parser.add_argument("--sliding-min-hz", type=float, default=None)
    parser.add_argument("--sliding-max-hz", type=float, default=None)
    parser.add_argument("--time-interval-s", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--only", choices=["welch", "sliding"], default=None)
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Replace the sliding spectrogram panel with a 2D image of the full Welch spectrum.",
    )
    parser.add_argument(
        "--disable",
        type=int,
        action="append",
        default=[],
        help="Disable pair index (0-based). Can be used multiple times.",
    )
    add_spectrasave_arg(parser)
    parser.add_argument(
        "--spectrasave-pair",
        type=int,
        default=None,
        help="0-based pair index to export when --spectrasave is used.",
    )
    return parser


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
        results = analyze_spacing_dataset_with_welch_for_display(
            spacing,
            disabled_indices=args.disable,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            sliding_len_s=args.sliding_len_s,
        )

        dt = np.diff(track2.frame_times_s)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            approx_fs = 1.0 / float(np.median(dt))
            approx_nyquist = 0.5 * approx_fs
            print(f"Track2: {track2.track2_path}")
            print(f"Approx sampling rate: {approx_fs:.4f} Hz | Approx Nyquist: {approx_nyquist:.4f} Hz")
        else:
            print(f"Track2: {track2.track2_path}")
            print("Approx sampling rate: unavailable")

        if args.spectrasave is not None:
            if args.spectrasave_pair is None:
                raise ValueError("--spectrasave requires --spectrasave-pair for see_welch_fft.py")
            matching = [result for result in results if result.pair_index == args.spectrasave_pair]
            if not matching:
                raise ValueError(f"Requested export pair {args.spectrasave_pair} was not available")
            export_result = matching[0]
            if export_result.welch_result is None:
                raise ValueError(f"Pair {args.spectrasave_pair} has no Welch result to export")

            dataset_name = track2.dataset_name or Path(track2.track2_path).resolve().parent.name
            export_path = resolve_spectrasave_path(
                args.spectrasave,
                default_name=build_default_spectrasave_name(
                    dataset_name,
                    f"pair-{export_result.pair_index}",
                    "welch",
                ),
            )
            assert export_path is not None
            saved = save_spectrum_msgpack(
                export_path,
                freq=export_result.welch_result.freq,
                amplitude=export_result.welch_result.amplitude,
                label=f"{dataset_name} pair {export_result.pair_index} Welch",
                metadata={
                    "sourceKind": "single",
                    "spectrumKind": "welch",
                    "dataset": dataset_name,
                    "track2Path": track2.track2_path,
                    "pairIndex": int(export_result.pair_index),
                    "pairLabel": export_result.label,
                    "welchLenS": float(args.welch_len_s),
                    "welchOverlap": float(args.welch_overlap),
                    "longest": bool(args.longest),
                    "handlenan": bool(args.handlenan),
                },
            )
            print(f"Spectrum saved to: {saved}")

        fig = plot_pair_welch_frequency_grid(
            results,
            welch_log=args.welch_log,
            sliding_plot_scale=args.sliding_plot_scale,
            only=args.only,
            full_image=args.full_image,
            welch_min_hz=args.welch_min_hz,
            welch_max_hz=args.welch_max_hz,
            sliding_min_hz=args.sliding_min_hz,
            sliding_max_hz=args.sliding_max_hz,
            time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
            cmap_index=args.cm,
            title=args.title,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
