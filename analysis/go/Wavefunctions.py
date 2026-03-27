#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.plotting.common import render_figure
from analysis.tools.bond_phase import (
    bond_phase_table_from_result,
    build_bond_projection_factors,
    estimate_bond_peak_phases,
    transform_bond_phase_table,
)
from analysis.tools.cli import (
    add_normalization_args,
    add_output_args,
    add_signal_processing_args,
    add_track_data_root_arg,
    resolve_normalization_mode,
)
from analysis.tools.groups import write_temp_component_selection_config
from analysis.tools.localization import build_peak_diagnostics_by_entity, compute_localization_profiles
from analysis.tools.models import LocalizationPeakDiagnostic, LocalizationProfile
from analysis.tools.peaks import load_peaks_csv, resolve_peaks_csv
from analysis.tools.selection import build_configured_bond_signals, load_dataset_selection
from analysis.tools.spectral import compute_fft_contributions, compute_welch_contributions


def _load_localize_peaks_module():
    module_path = Path(__file__).resolve().parents[1] / "viz" / "localize_peaks.py"
    spec = importlib.util.spec_from_file_location("analysis_go_wavefunctions_localize_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load localization implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot bond wavefunctions for a dataset from a named peaks file.",
    )
    parser.add_argument("dataset", help="Dataset name, e.g. 10M4.")
    parser.add_argument("peaks", help="Peaks name or CSV path.")
    add_track_data_root_arg(parser)
    add_normalization_args(parser)
    add_signal_processing_args(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--show",
        choices=["x", "y", "a"],
        default="x",
        help="Component to analyze. Default: x",
    )
    parser.add_argument(
        "--integration-window",
        type=float,
        default=0.25,
        help="Half-width in Hz for the peak integration/search window. Default: 0.25",
    )
    parser.add_argument(
        "--highest-bin",
        action="store_true",
        help="Use the highest bin in the window instead of sqrt(integrated power).",
    )
    welch_group = parser.add_mutually_exclusive_group()
    welch_group.add_argument(
        "--welch",
        dest="welch",
        action="store_true",
        help="Use Welch spectra for wavefunctions and diagnostics (default).",
    )
    welch_group.add_argument(
        "--no-welch",
        dest="welch",
        action="store_false",
        help="Use FFT spectra instead of Welch spectra.",
    )
    parser.set_defaults(welch=True)
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument(
        "--one-fig",
        action="store_true",
        help="Replace stacked subplots with one offset overlay plot.",
    )
    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--zero",
        dest="show_zero",
        action="store_true",
        help="Show a dotted zero reference line for each profile (default).",
    )
    zero_group.add_argument(
        "--no-zero",
        dest="show_zero",
        action="store_false",
        help="Hide the dotted zero reference line.",
    )
    parser.set_defaults(show_zero=True)
    parser.add_argument(
        "--forcereal",
        action="store_true",
        help="Force projected bond phases to be purely real by snapping to 0 or pi.",
    )
    parser.add_argument(
        "--flip",
        action="append",
        default=[],
        metavar="BOND",
        help="Flip a 1-based bond index, or use --flip all.",
    )
    parser.add_argument(
        "--reference-bond",
        type=int,
        default=None,
        help="1-based bond index used for phase gauge fixing. Default: middle bond.",
    )
    parser.add_argument(
        "--reference-peak-index",
        type=int,
        default=1,
        help="1-based peak index used as the gauge reference. Default: 1",
    )
    parser.add_argument(
        "--min-reference-fraction",
        type=float,
        default=0.05,
        help="Minimum retained reference amplitude fraction for gauge windows. Default: 0.05",
    )
    return parser


def _parse_flip_args(values: list[str], bond_ids_zero_based: list[int]) -> list[int]:
    if not values:
        return []
    lowered = [str(value).strip().lower() for value in values]
    if "all" in lowered:
        return sorted(set(int(bond_id) for bond_id in bond_ids_zero_based))

    parsed: list[int] = []
    for value in values:
        bond_display = int(value)
        if bond_display < 1:
            raise ValueError("--flip values must be positive 1-based bond indices or 'all'")
        parsed.append(bond_display - 1)
    return sorted(set(parsed))


def _shift_profiles_to_display_bonds(profiles: list[LocalizationProfile]) -> list[LocalizationProfile]:
    shifted: list[LocalizationProfile] = []
    for profile in profiles:
        shifted.append(
            LocalizationProfile(
                peak_index=int(profile.peak_index),
                frequency=float(profile.frequency),
                entity_ids=np.asarray(profile.entity_ids, dtype=int) + 1,
                mean_amplitudes=np.asarray(profile.mean_amplitudes, dtype=float),
                std_amplitudes=np.asarray(profile.std_amplitudes, dtype=float),
            )
        )
    return shifted


def _shift_diagnostics_labels(
    diagnostics_by_entity: dict[str, list[LocalizationPeakDiagnostic]] | None,
) -> dict[str, list[LocalizationPeakDiagnostic]] | None:
    if diagnostics_by_entity is None:
        return None

    shifted: dict[str, list[LocalizationPeakDiagnostic]] = {}
    for label, diagnostics in diagnostics_by_entity.items():
        if label == "All":
            shifted[label] = diagnostics
            continue
        shifted[str(int(label) + 1)] = diagnostics
    return shifted


def _default_reference_bond(records) -> int:
    bond_ids = sorted({int(record.entity_id) for record in records})
    if not bond_ids:
        raise ValueError("No configured bond records were available")
    return int(bond_ids[len(bond_ids) // 2])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1
    if args.integration_window <= 0:
        print("Error: --integration-window must be > 0", file=sys.stderr)
        return 1
    if args.welch_len_s <= 0:
        print("Error: --welch-len-s must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= args.welch_overlap < 1.0):
        print("Error: --welch-overlap must satisfy 0 <= value < 1", file=sys.stderr)
        return 1
    if args.min_reference_fraction < 0:
        print("Error: --min-reference-fraction must be >= 0", file=sys.stderr)
        return 1
    if args.reference_peak_index < 1:
        print("Error: --reference-peak-index must be >= 1", file=sys.stderr)
        return 1
    if args.reference_bond is not None and args.reference_bond < 1:
        print("Error: --reference-bond must be >= 1", file=sys.stderr)
        return 1

    try:
        peaks_path = resolve_peaks_csv(args.peaks)
        peaks = load_peaks_csv(peaks_path)
        temp_config = write_temp_component_selection_config(
            [args.dataset],
            component=args.show,
            track_data_root=args.track_data_root,
            prefix=f"analysis_wavefunctions_{args.dataset}_",
        )
        config = load_dataset_selection(temp_config.name)
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=False,
        )
        if len(records) == 0:
            raise ValueError(f"No bond records were produced for dataset '{args.dataset}'")

        reference_bond_zero = (
            int(args.reference_bond) - 1 if args.reference_bond is not None else _default_reference_bond(records)
        )
        bond_ids_zero = sorted({int(record.entity_id) for record in records})
        if reference_bond_zero not in bond_ids_zero:
            raise ValueError(
                f"Reference bond {reference_bond_zero + 1} is not in the enabled bonds: "
                f"{[bond_id + 1 for bond_id in bond_ids_zero]}"
            )

        bond_phase_result = estimate_bond_peak_phases(
            records,
            peaks,
            reference_bond_id=reference_bond_zero,
            reference_peak_index=args.reference_peak_index,
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            search_width_hz=args.integration_window,
            min_reference_fraction=args.min_reference_fraction,
            longest=args.longest,
            handlenan=args.handlenan,
        )
        phase_table = bond_phase_table_from_result(bond_phase_result)
        flip_bond_ids = _parse_flip_args(args.flip, bond_ids_zero)
        phase_table = transform_bond_phase_table(
            phase_table,
            flip_bond_ids=flip_bond_ids,
            forcereal=args.forcereal,
        )
        projection_factors_by_peak = build_bond_projection_factors(
            phase_table,
            peaks,
            [int(record.entity_id) for record in records],
        )

        peak_targets = [(idx, peaks[idx]) for idx in range(len(peaks))]
        profiles = compute_localization_profiles(
            records,
            peak_targets,
            normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            search_width=args.integration_window,
            spectrum_kind="welch" if args.welch else "fft",
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            projection_factors_by_peak=projection_factors_by_peak,
            sqrtintpower=(not args.highest_bin),
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )

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

        diagnostics_by_entity = None
        if contributions:
            diagnostics_by_entity = build_peak_diagnostics_by_entity(
                contributions,
                peak_targets,
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                search_width=args.integration_window,
            )

        profiles = _shift_profiles_to_display_bonds(profiles)
        diagnostics_by_entity = _shift_diagnostics_labels(diagnostics_by_entity)

        localize_module = _load_localize_peaks_module()
        if args.title:
            title = args.title
        else:
            mode_desc = "highest bin" if args.highest_bin else "sqrt(integrated power)"
            phase_desc = "real-projected" if args.forcereal else "phase-projected"
            title = (
                f"Bond Wavefunctions | {args.dataset} | "
                f"{'Welch' if args.welch else 'FFT'} | {mode_desc} | {phase_desc}"
            )

        print(f"Dataset: {args.dataset}")
        print(f"Component: {args.show}")
        print(f"Peaks file: {peaks_path}")
        print(f"Peaks (Hz): {peaks}")
        print(f"Bonds: {[bond_id + 1 for bond_id in bond_ids_zero]}")
        print(f"Reference bond: {reference_bond_zero + 1}")
        print(f"Reference peak index: {args.reference_peak_index}")
        print(f"Integration window half-width (Hz): {args.integration_window}")
        print(f"Amplitude mode: {'highest bin' if args.highest_bin else 'sqrt(integrated power)'}")
        if flip_bond_ids:
            print(f"Flipped bonds: {[bond_id + 1 for bond_id in flip_bond_ids]}")
        if args.forcereal:
            print("Phase snap: forced to {0, pi}")

        fig = localize_module.plot_localization_profiles(
            profiles,
            xlabel="Bond Index",
            title=title,
            line_color="black",
            diagnostics_by_entity=diagnostics_by_entity,
            one_fig=args.one_fig,
            show_zero=args.show_zero,
        )
        render_figure(fig, save=args.save)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
