#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys

from plotting.common import render_figure
from plotting.indexed import plot_localization_profiles
from tools.cli import add_normalization_args, add_output_args, add_signal_processing_args, add_track_data_root_arg
from tools.localization import compute_localization_profiles
from tools.peaks import load_peaks_csv, select_active_peak_indices
from tools.selection import build_configured_site_signals, load_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot normalized peak amplitude versus site index.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing peak frequencies.")
    add_track_data_root_arg(parser)
    add_normalization_args(parser)
    add_signal_processing_args(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--search-width-hz",
        type=float,
        default=0.25,
        help="Half-width of the frequency search window around each target peak. Default: 0.25",
    )
    parser.add_argument(
        "--allow-duplicate-sites",
        action="store_true",
        help="Average duplicate site ids instead of discarding later occurrences.",
    )
    parser.add_argument(
        "--disable-plot",
        type=int,
        nargs="+",
        default=[],
        help="List of peak indices (0-based) to skip plotting.",
    )
    parser.add_argument(
        "--only-enable-plots",
        type=int,
        nargs="+",
        default=None,
        help="List of peak indices (0-based) to plot exclusively.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        peaks = load_peaks_csv(args.peaks_csv)
        active_indices = select_active_peak_indices(
            peaks,
            disableplot=args.disable_plot,
            onlyenableplots=args.only_enable_plots,
        )
        if not active_indices:
            print("No peaks selected for plotting.", file=sys.stderr)
            return 0

        config = load_dataset_selection(args.config_json)
        records = build_configured_site_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=args.allow_duplicate_sites,
        )
        peak_targets = [(idx, peaks[idx]) for idx in active_indices]
        profiles = compute_localization_profiles(
            records,
            peak_targets,
            normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            search_width=args.search_width_hz,
            longest=args.longest,
            handlenan=args.handlenan,
        )

        if args.title:
            title = args.title
        else:
            norm_desc = args.normalize
            if args.normalize == "relative":
                norm_desc += f" [{args.relative_range[0]}-{args.relative_range[1]} Hz]"
            title = f"Site Peak Localization | Norm: {norm_desc}"

        print(f"Loaded {len(peaks)} peaks: {peaks}")
        print(f"Active peak indices (sorted high->low): {active_indices}")
        print(f"Signal records: {len(records)}")

        fig = plot_localization_profiles(
            profiles,
            xlabel="Site Index",
            title=title,
            line_color="maroon",
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
