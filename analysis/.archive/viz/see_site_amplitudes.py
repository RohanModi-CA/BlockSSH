#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys

from plotting.common import render_figure
from plotting.frequency import plot_site_amplitude_previews
from plotting.indexed import plot_localization_profiles
from tools.cli import add_output_args, add_peak_integration_args, add_signal_processing_args, add_track_data_root_arg
from tools.peaks import assert_peaks_strictly_increasing, load_peaks_csv
from tools.selection import build_grouped_configured_bond_signals, load_dataset_selection
from tools.site_amplitudes import (
    analyze_grouped_bond_site_amplitudes,
    analyze_grouped_bond_site_amplitudes_phase_reconstruction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrate normalized averaged FFT amplitudes around CSV peaks for each configured global bond.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing peak frequencies in ascending order.")
    add_track_data_root_arg(parser)
    add_signal_processing_args(parser)
    add_peak_integration_args(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the normalized ROI spectra for each bond instead of the per-peak bond-index plots.",
    )
    parser.add_argument(
        "--phase-reconstruction",
        action="store_true",
        help="Reserved mode. Not implemented yet.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        peaks = assert_peaks_strictly_increasing(load_peaks_csv(args.peaks_csv))
        config = load_dataset_selection(args.config_json)
        grouped_records = build_grouped_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
        )
        if len(grouped_records) == 0:
            raise ValueError("No included datasets produced any grouped bond records")

        if args.phase_reconstruction:
            analyze_grouped_bond_site_amplitudes_phase_reconstruction()

        result = analyze_grouped_bond_site_amplitudes(
            grouped_records,
            peaks,
            integration_window_width=args.integration_window_width,
            normalization_multiplier=args.normalization_multiplier,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )

        n_datasets = sum(1 for selection in config.values() if selection.include)
        total_contributors = sum(len(bond.contributors) for bond in result.bonds)

        print(f"Loaded peaks ({len(result.peaks)}): {result.peaks.tolist()}")
        print(f"ROI window: [{result.roi_low:.6f}, {result.roi_high:.6f}] Hz")
        print(f"Included datasets: {n_datasets}")
        print(f"Accepted bond groups: {len(result.bonds)}")
        print(f"Accepted FFT contributors: {total_contributors}")
        print("Contributors per bond:")
        for bond in result.bonds:
            print(
                f"  bond {bond.display_bond_index}: {len(bond.contributors)} contributor(s) | "
                f"normalization integral={bond.normalization_integral:.6g}"
            )

        if args.title:
            title = args.title
        elif args.preview:
            title = (
                f"Bond ROI Spectra Preview | bonds={len(result.bonds)} | peaks={len(result.peaks)} | "
                f"window=±{args.integration_window_width} Hz"
            )
        else:
            title = (
                f"Integrated Site Amplitudes by Bond | bonds={len(result.bonds)} | peaks={len(result.peaks)} | "
                f"window=±{args.integration_window_width} Hz"
            )

        if args.preview:
            fig = plot_site_amplitude_previews(
                result,
                title=title,
            )
        else:
            fig = plot_localization_profiles(
                result.profiles,
                xlabel="Bond Index",
                ylabel="Integrated Normalized Amplitude",
                title=title,
                line_color=None,
            )

        render_figure(fig, save=args.save)
        return 0

    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
