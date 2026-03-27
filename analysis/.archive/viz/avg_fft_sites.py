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
from plotting.frequency import plot_average_spectrum
from tools.cli import (
    add_average_domain_args,
    add_bond_filter_args,
    add_colormap_arg,
    add_normalization_args,
    add_output_args,
    add_plot_scale_args,
    resolve_normalization_mode,
    add_signal_processing_args,
    add_track_data_root_arg,
)
from tools.peaks import load_peaks_csv
from tools.selection import (
    build_configured_bond_signals,
    collect_display_bond_numbers,
    filter_signal_records_by_display_bonds,
    load_dataset_selection,
)
from tools.spectral import (
    ABSOLUTE_ZERO_TOL,
    compute_average_spectrum,
    compute_fft_contributions,
    compute_reference_average_spectrum,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average FFT full-image view with site/peak overlay.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing overlay peak frequencies.")
    add_track_data_root_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_plot_scale_args(parser)
    add_signal_processing_args(parser)
    add_bond_filter_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument("--freq-min-hz", type=float, default=None)
    parser.add_argument("--freq-max-hz", type=float, default=None)
    parser.add_argument(
        "--allow-duplicate-bonds",
        action="store_true",
        help="Allow the same configured bond id to contribute multiple times.",
    )
    parser.add_argument(
        "--site-offset",
        type=float,
        default=0.0,
        help="Offset added to the site index on the x-axis. Default: 0",
    )
    parser.add_argument(
        "--site-marker-size",
        type=float,
        default=30.0,
        help="Marker size for the overlaid site peaks. Default: 30",
    )
    parser.add_argument(
        "--site-mode",
        choices=["scatter", "line"],
        default="scatter",
        help="Overlay mode. 'line' connects the points; 'scatter' does not.",
    )
    parser.add_argument(
        "--site-color",
        default="red",
        help="Color of the overlaid site peaks. Default: red",
    )
    return parser


def _format_bond_list(display_bonds: list[int]) -> str:
    if len(display_bonds) == 0:
        return "[]"
    if len(display_bonds) <= 12:
        return "[" + ", ".join(str(v) for v in display_bonds) + "]"
    head = ", ".join(str(v) for v in display_bonds[:10])
    return f"[{head}, ...] ({len(display_bonds)} total)"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    if args.freq_min_hz is not None and args.freq_max_hz is not None and args.freq_max_hz <= args.freq_min_hz:
        print("Error: --freq-max-hz must be greater than --freq-min-hz", file=sys.stderr)
        return 1

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        peaks = load_peaks_csv(args.peaks_csv)
        config = load_dataset_selection(args.config_json)
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=args.allow_duplicate_bonds,
        )
        available_display_bonds = collect_display_bond_numbers(records)
        records = filter_signal_records_by_display_bonds(
            records,
            only_bonds=args.only_bonds,
            exclude_bonds=args.exclude_bonds,
            parity=args.bond_parity,
        )
        selected_display_bonds = collect_display_bond_numbers(records)
        if len(records) == 0:
            raise ValueError("Bond selection removed all configured bond contributors")

        contributions = compute_fft_contributions(
            records,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )
        if len(contributions) == 0:
            raise ValueError("No spectra were accepted from the selected bond contributors")

        result = compute_average_spectrum(
            contributions,
            normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            average_domain=args.average_domain,
            lowest_freq=args.freq_min_hz,
            highest_freq=args.freq_max_hz,
        )

        reference_amp = None
        if args.plot_scale == "linear":
            reference = compute_reference_average_spectrum(
                contributions,
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                average_domain=args.average_domain,
            )
            reference_amp = reference.avg_amp

        n_contributors = len(result.contributors)
        n_datasets = len({contrib.record.dataset_name for contrib in result.contributors})
        accepted_display_bonds = sorted({int(contrib.record.entity_id) + 1 for contrib in result.contributors})

        if args.title:
            title = args.title
        else:
            norm_desc = args.normalize
            if args.normalize == "relative":
                title_range = tuple(args.relative_range)
                norm_desc = f"relative [{title_range[0]}, {title_range[1]}] Hz"
            title = (
                f"Average FFT with Site Overlay | contributors={n_contributors} | datasets={n_datasets} | "
                f"bonds={len(accepted_display_bonds)} | avg={args.average_domain} | norm={norm_desc}"
            )

        x_values = np.arange(len(peaks), dtype=float) + float(args.site_offset)
        x_max = len(peaks) + float(args.site_offset)

        overlay = {
            "x_values": x_values,
            "y_values": np.asarray(peaks, dtype=float),
            "mode": args.site_mode,
            "color": args.site_color,
            "marker_size": float(args.site_marker_size),
            "x_label": "Site Index",
            "x_max": float(x_max),
            "integer_ticks": True,
        }

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds)}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds)}")
        print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
        print(f"Accepted contributors: {n_contributors}")
        print(f"Unique datasets: {n_datasets}")
        print(f"Frequency window: [{result.freq_low:.6f}, {result.freq_high:.6f}] Hz")
        print(f"Normalization window: [{result.norm_low:.6f}, {result.norm_high:.6f}] Hz")
        print("Normalization band processing: linear detrend -> zero-floor -> integrate area")
        print(f"Near-zero denominator threshold: {ABSOLUTE_ZERO_TOL:.0e}")
        print(f"Common grid points: {len(result.freq_grid)}")
        print("Display mode: full image with site overlay")
        if args.plot_scale == "linear":
            print("Image color scale reference: full implicit frequency window")

        fig = plot_average_spectrum(
            result,
            full_image=True,
            plot_scale=args.plot_scale,
            cmap_index=args.cm,
            title=title,
            reference_amp_for_norm=reference_amp,
            overlay=overlay,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
