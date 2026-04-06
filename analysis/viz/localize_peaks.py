#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
import warnings

from tools.bond_phase import (
    build_bond_projection_factors,
    load_bond_peak_phase_csv,
    transform_bond_phase_table,
)
from plotting.common import ensure_parent_dir, render_figure
from plotting.indexed import plot_localization_profiles
from tools.cli import (
    add_bond_spacing_mode_arg,
    add_normalization_args,
    add_output_args,
    add_signal_processing_args,
    add_track_data_root_arg,
    resolve_normalization_mode,
)
from tools.localization import build_peak_diagnostics_by_entity, compute_localization_profiles
from tools.peaks import load_peaks_csv, select_active_peak_indices
from tools.spectral import compute_fft_contributions, compute_welch_contributions
from tools.selection import build_configured_bond_signals, load_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot normalized peak amplitude versus bond index.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing peak frequencies.")
    add_track_data_root_arg(parser)
    add_bond_spacing_mode_arg(parser)
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
        "--sqrtintpower",
        action="store_true",
        help="Use sqrt(integrated power) within each peak window instead of the peak maximum amplitude.",
    )
    welch_group = parser.add_mutually_exclusive_group()
    welch_group.add_argument(
        "--welch",
        dest="welch",
        action="store_true",
        help="Use Welch spectra for localization and the diagnostic panel (default).",
    )
    welch_group.add_argument(
        "--no-welch",
        dest="welch",
        action="store_false",
        help="Use one-sided FFT spectra instead of Welch spectra.",
    )
    parser.set_defaults(welch=True)
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument(
        "--allow-duplicate-bonds",
        action="store_true",
        help="Average duplicate bond ids instead of discarding later occurrences.",
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
    parser.add_argument(
        "--one-fig",
        action="store_true",
        help="Replace the left stacked subplots with one offset overlay plot.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Optional bond-phase CSV. When set, project Welch amplitudes onto the real axis for signed plotting.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="When used with --phase, add pi to all loaded site phases before bond projection.",
    )
    parser.add_argument(
        "--forcereal",
        action="store_true",
        help=(
            "When used with --phase, snap each loaded site phase to 0 or pi before bond projection, "
            "so site phasors are purely positive or negative real."
        ),
    )
    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--zero",
        dest="show_zero",
        action="store_true",
        help="Show a dotted grey zero reference line for each profile (default).",
    )
    zero_group.add_argument(
        "--no-zero",
        dest="show_zero",
        action="store_false",
        help="Hide the dotted grey zero reference line.",
    )
    parser.set_defaults(show_zero=True)
    peak_label_group = parser.add_mutually_exclusive_group()
    peak_label_group.add_argument(
        "--peak-labels",
        dest="show_peak_labels",
        action="store_true",
        help="Show peak labels/titles on the plot (default).",
    )
    peak_label_group.add_argument(
        "--no-peak-labels",
        dest="show_peak_labels",
        action="store_false",
        help="Hide peak labels/titles on the plot.",
    )
    parser.set_defaults(show_peak_labels=True)
    parser.add_argument(
        "--max-panels",
        type=int,
        default=6,
        help="Maximum number of peak panels subplots per figure window. Excess panels spawn additional windows. Default: 6",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1
    if args.welch_len_s <= 0:
        print("Error: --welch-len-s must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= args.welch_overlap < 1.0):
        print("Error: --welch-overlap must satisfy 0 <= value < 1", file=sys.stderr)
        return 1
    if args.phase is not None and not args.welch:
        print("Error: --phase currently requires Welch mode", file=sys.stderr)
        return 1
    if args.flip and args.phase is None:
        print("Error: --flip currently requires --phase", file=sys.stderr)
        return 1
    if args.forcereal and args.phase is None:
        print("Error: --forcereal currently requires --phase", file=sys.stderr)
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
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=args.allow_duplicate_bonds,
            bond_spacing_mode=args.bond_spacing_mode,
        )
        peak_targets = [(idx, peaks[idx]) for idx in active_indices]
        projection_factors_by_peak = None
        if args.phase is not None:
            phase_table = load_bond_peak_phase_csv(args.phase)
            phase_table = transform_bond_phase_table(
                phase_table,
                flip=args.flip,
                forcereal=args.forcereal,
            )
            projection_factors_by_peak = build_bond_projection_factors(
                phase_table,
                [peak for _, peak in peak_targets],
                [int(record.entity_id) for record in records],
            )
        profiles = compute_localization_profiles(
            records,
            peak_targets,
            normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            search_width=args.search_width_hz,
            spectrum_kind="welch" if args.welch else "fft",
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            projection_factors_by_peak=projection_factors_by_peak,
            sqrtintpower=args.sqrtintpower,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )
        diagnostics_by_entity = None
        if args.welch:
            contributions = compute_welch_contributions(
                records,
                welch_len_s=args.welch_len_s,
                welch_overlap_fraction=args.welch_overlap,
                longest=args.longest,
                handlenan=args.handlenan,
                timeseriesnorm=args.timeseriesnorm,
            )
        else:
            contributions = compute_fft_contributions(
                records,
                longest=args.longest,
                handlenan=args.handlenan,
                timeseriesnorm=args.timeseriesnorm,
            )
        if contributions:
            try:
                diagnostics_by_entity = build_peak_diagnostics_by_entity(
                    contributions,
                    peak_targets,
                    normalize_mode=args.normalize,
                    relative_range=tuple(args.relative_range),
                    search_width=args.search_width_hz,
                )
            except Exception as exc:
                warnings.warn(f"Could not build peak diagnostics: {exc}")

        if args.title:
            title = args.title
        else:
            norm_desc = args.normalize
            if args.normalize == "relative":
                norm_desc += f" [{args.relative_range[0]}-{args.relative_range[1]} Hz]"
            title = f"Bond Peak Localization | {'Welch' if args.welch else 'FFT'} | Norm: {norm_desc}"

        print(f"Loaded {len(peaks)} peaks: {peaks}")
        print(f"Active peak indices (sorted high->low): {active_indices}")
        print(f"Signal records: {len(records)}")
        print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
        print(
            "Peak amplitude mode: "
            + ("sqrt(integrated power)" if args.sqrtintpower else "max amplitude in window")
        )
        if args.welch:
            print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
        if args.phase is not None:
            print(f"Phase projection CSV: {args.phase}")
            if args.flip:
                print("Phase shift: +pi")
            if args.forcereal:
                print("Phase snap: forced to {0, pi}")

        max_panels = int(args.max_panels)
        if max_panels < 1:
            print("Error: --max-panels must be >= 1", file=sys.stderr)
            return 1

        n_profiles = len(profiles)
        n_figs = 1 if args.one_fig else max(1, (n_profiles + max_panels - 1) // max_panels)

        figs = []
        for fig_idx in range(n_figs):
            if n_figs == 1:
                chunk_profiles = profiles
                chunk_diagnostics = diagnostics_by_entity
                fig_title = title
            else:
                start = fig_idx * max_panels
                end = min(start + max_panels, n_profiles)
                chunk_profiles = profiles[start:end]
                if diagnostics_by_entity is not None:
                    chunk_diagnostics = {
                        label: diags[start:end] for label, diags in diagnostics_by_entity.items()
                    }
                else:
                    chunk_diagnostics = None
                fig_title = f"{title} ({fig_idx + 1}/{n_figs})"

            fig = plot_localization_profiles(
                chunk_profiles,
                xlabel="Bond Index",
                title=fig_title,
                line_color="black",
                diagnostics_by_entity=chunk_diagnostics,
                one_fig=args.one_fig,
                show_zero=args.show_zero,
                show_peak_labels=args.show_peak_labels,
            )
            if args.save is not None:
                save_path = args.save if n_figs == 1 else f"{args.save}_{fig_idx + 1}"
                save_path = ensure_parent_dir(save_path)
                fig.savefig(save_path, dpi=300)
                print(f"Plot saved to: {save_path}")
            figs.append(fig)

        import matplotlib.pyplot as _plt
        if len(figs) > 1:
            for fig in figs:
                _plt.figure(fig.number)
            _plt.show(block=False)
            _plt.show()
        else:
            render_figure(figs[0], save=None)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
