#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.plotting.common import ensure_parent_dir, render_figure
from analysis.GoHit.tools.cli import add_hit_mode_args, add_hit_region_args, describe_hit_region_settings
from analysis.GoHit.tools.hits import (
    build_interhit_regions,
    build_posthit_regions,
    load_catalog_if_available,
    summarize_catalog,
)
from analysis.GoHit.tools.region_phase import estimate_region_bond_phases
from analysis.GoHit.tools.region_localization import (
    build_region_peak_diagnostics_by_entity,
    build_region_spectrum_entries,
    compute_localization_profiles_from_entries,
    compute_region_localization_profiles,
)
from analysis.GoHit.tools.region_spectra import compute_region_averaged_fft
from analysis.tools.bond_phase import (
    build_bond_projection_factors,
    transform_bond_phase_table,
)
from analysis.tools.cli import (
    add_bond_spacing_mode_arg,
    add_normalization_args,
    add_output_args,
    add_signal_processing_args,
    add_track_data_root_arg,
    resolve_normalization_mode,
)
from analysis.tools.groups import write_temp_component_selection_config
from analysis.tools.localization import build_peak_diagnostics_by_entity, compute_localization_profiles
from analysis.tools.models import FFTResult, LocalizationPeakDiagnostic, LocalizationProfile, SignalRecord, SpectrumContribution
from analysis.tools.peaks import load_peaks_csv, resolve_peaks_csv
from analysis.tools.selection import build_configured_bond_signals, load_dataset_selection
from analysis.tools.spectral import compute_fft_contributions, compute_welch_contributions
from analysis.tools.signal import normalize_processed_signal_rms, preprocess_signal


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
    add_bond_spacing_mode_arg(parser)
    add_normalization_args(parser)
    add_signal_processing_args(parser)
    add_output_args(parser, include_title=True)
    add_hit_mode_args(parser)
    add_hit_region_args(parser)

    parser.add_argument(
        "--show",
        choices=["x", "y", "a"],
        default="x",
        help="Component to analyze. Default: x",
    )
    parser.add_argument(
        "--integration-window",
        type=float,
        default=0.05,
        help="Half-width in Hz for the peak integration/search window. Default: 0.25",
    )
    parser.add_argument(
        "--highest-bin",
        action="store_true",
        help="Use the highest bin in the window instead of sqrt(integrated power).",
    )
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--baseline_subtract",
        dest="baseline_subtract",
        action="store_true",
        help="Measure peak amplitudes above a local spectral baseline before building wavefunctions.",
    )
    baseline_group.add_argument(
        "--no-baseline_subtract",
        dest="baseline_subtract",
        action="store_false",
        help="Use raw peak amplitudes without local baseline subtraction (default).",
    )
    parser.set_defaults(baseline_subtract=False)
    flatten_group = parser.add_mutually_exclusive_group()
    flatten_group.add_argument(
        "--flatten",
        dest="flatten",
        action="store_true",
        help="Apply the GoHit flattening transform before extracting wavefunction peak amplitudes.",
    )
    flatten_group.add_argument(
        "--no-flatten",
        dest="flatten",
        action="store_false",
        help="Disable the GoHit flattening transform (default).",
    )
    parser.set_defaults(flatten=False)
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
    diagnostics_group = parser.add_mutually_exclusive_group()
    diagnostics_group.add_argument(
        "--diagnostics",
        dest="show_diagnostics",
        action="store_true",
        help="Show the per-peak diagnostic side panels (default).",
    )
    diagnostics_group.add_argument(
        "--no-diagnostics",
        dest="show_diagnostics",
        action="store_false",
        help="Hide the per-peak diagnostic side panels.",
    )
    parser.set_defaults(show_diagnostics=True)
    std_group = parser.add_mutually_exclusive_group()
    std_group.add_argument(
        "--std-band",
        dest="show_std_band",
        action="store_true",
        help="Show the grey +/- std envelope around each wavefunction curve.",
    )
    std_group.add_argument(
        "--no-std-band",
        dest="show_std_band",
        action="store_false",
        help="Hide the grey +/- std envelope around each wavefunction curve (default).",
    )
    parser.set_defaults(show_std_band=False)
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
        "--forcereal",
        action="store_true",
        help="Force projected bond phases to be purely real by snapping to 0 or pi.",
    )
    parser.add_argument(
        "--posphase",
        action="store_true",
        help="Align each peak independently to be as real and positive as possible.",
    )
    parser.add_argument(
        "--flip",
        action="append",
        default=[],
        metavar="BOND",
        help="Flip a 1-based bond index, or use --flip all.",
    )
    parser.add_argument(
        "--flipmode",
        action="append",
        default=[],
        metavar="BOND",
        help="Alias for --flip. Flip a 1-based bond index, or use --flipmode all.",
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
    parser.add_argument(
        "--absvalues",
        "--absvalue",
        action="store_true",
        dest="absvalues",
        help="Use absolute values for wavefunctions instead of projecting phases.",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=6,
        help="Maximum number of peak panels subplots per figure window. Excess panels spawn additional windows. Default: 6",
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


def _maybe_hide_std_band(
    profiles: list[LocalizationProfile],
    *,
    show_std_band: bool,
) -> list[LocalizationProfile]:
    if show_std_band:
        return profiles
    stripped: list[LocalizationProfile] = []
    for profile in profiles:
        stripped.append(
            LocalizationProfile(
                peak_index=int(profile.peak_index),
                frequency=float(profile.frequency),
                entity_ids=np.asarray(profile.entity_ids, dtype=int),
                mean_amplitudes=np.asarray(profile.mean_amplitudes, dtype=float),
                std_amplitudes=np.zeros_like(np.asarray(profile.std_amplitudes, dtype=float)),
            )
        )
    return stripped


def _sort_profiles_high_to_low_frequency(
    profiles: list[LocalizationProfile],
    diagnostics_by_entity: dict[str, list[LocalizationPeakDiagnostic]] | None = None,
) -> tuple[list[LocalizationProfile], dict[str, list[LocalizationPeakDiagnostic]] | None]:
    order = sorted(
        range(len(profiles)),
        key=lambda idx: (float(profiles[idx].frequency), int(profiles[idx].peak_index)),
        reverse=True,
    )
    sorted_profiles = [profiles[idx] for idx in order]
    if diagnostics_by_entity is None:
        return sorted_profiles, None

    sorted_diagnostics = {
        label: [diagnostics[idx] for idx in order]
        for label, diagnostics in diagnostics_by_entity.items()
    }
    return sorted_profiles, sorted_diagnostics


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


def _figure_save_path(base_path: str, fig_idx: int, n_figs: int) -> str:
    if n_figs == 1:
        return base_path
    path = Path(base_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_{fig_idx + 1}{path.suffix}"))
    return f"{base_path}_{fig_idx + 1}"


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
        explicit_welch = "--welch" in sys.argv[1:]
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
            bond_spacing_mode=args.bond_spacing_mode,
        )
        if len(records) == 0:
            raise ValueError(f"No bond records were produced for dataset '{args.dataset}'")

        if args.flatten and args.welch:
            if explicit_welch:
                raise ValueError("GoHit Wavefunctions --flatten currently supports FFT mode only")
            args.welch = False
        if args.baseline_subtract and args.welch:
            if explicit_welch:
                raise ValueError("GoHit Wavefunctions --baseline_subtract currently supports FFT mode only")
            args.welch = False

        bond_ids_zero = sorted({int(record.entity_id) for record in records})
        projection_factors_by_peak = None
        phase_table = None
        flip_bond_ids = []
        reference_bond_zero = None

        if args.hits:
            if explicit_welch:
                raise ValueError("GoHit Wavefunctions --hits does not support Welch mode")
            catalog = load_catalog_if_available(args.dataset)
            if catalog is None:
                raise ValueError(
                    f"No GoHit confirmed hit catalog exists for dataset '{args.dataset}'. "
                    f"Run python3 analysis/GoHit/HitReview.py {args.dataset} first."
                )
            t_stop_s = max(float(np.nanmax(np.asarray(record.t, dtype=float))) for record in records)
            regions = (
                build_posthit_regions(
                    catalog.hit_times_s,
                    t_stop_s=t_stop_s,
                    window_s=float(args.hit_window),
                )
                if args.posthit
                else build_interhit_regions(
                    catalog.hit_times_s,
                    t_stop_s=t_stop_s,
                    exclude_after_s=float(args.delay),
                    exclude_before_s=float(args.exclude_before),
                )
            )
            if len(regions) == 0:
                raise ValueError(f"No usable {'posthit' if args.posthit else 'interhit'} regions were available")
            args.welch = False
        else:
            regions = None

        if not args.absvalues:
            reference_bond_zero = (
                int(args.reference_bond) - 1 if args.reference_bond is not None else _default_reference_bond(records)
            )
            if reference_bond_zero not in bond_ids_zero:
                raise ValueError(
                    f"Reference bond {reference_bond_zero + 1} is not in the enabled bonds: "
                    f"{[bond_id + 1 for bond_id in bond_ids_zero]}"
                )

            if args.hits:
                phase_estimate = estimate_region_bond_phases(
                    records,
                    peaks,
                    regions=regions,
                    reference_bond_id=reference_bond_zero,
                    reference_peak_index=args.reference_peak_index,
                    search_width_hz=args.integration_window,
                    min_reference_fraction=args.min_reference_fraction,
                    longest=args.longest,
                    handlenan=args.handlenan,
                    timeseriesnorm=args.timeseriesnorm,
                )
                phase_table = phase_estimate.phase_table
            else:
                from analysis.tools.bond_phase import bond_phase_table_from_result, estimate_bond_peak_phases

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
            flip_bond_ids = _parse_flip_args([*args.flip, *args.flipmode], bond_ids_zero)
            phase_table = transform_bond_phase_table(
                phase_table,
                flip_bond_ids=flip_bond_ids,
                forcereal=args.forcereal,
                posphase=args.posphase,
            )
            projection_factors_by_peak = build_bond_projection_factors(
                phase_table,
                peaks,
                [int(record.entity_id) for record in records],
            )

        peak_targets = [(idx, peaks[idx]) for idx in range(len(peaks))]
        if args.hits:
            transformed_entries = build_region_spectrum_entries(
                records,
                regions=regions,
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                flatten=args.flatten,
                longest=args.longest,
                handlenan=args.handlenan,
                timeseriesnorm=args.timeseriesnorm,
            )
            profiles = compute_localization_profiles_from_entries(
                transformed_entries,
                peak_targets,
                search_width=args.integration_window,
                phase_table=(None if args.absvalues else phase_table),
                baseline_subtract=args.baseline_subtract,
                sqrtintpower=(not args.highest_bin),
            )
        else:
            if args.baseline_subtract:
                if args.welch:
                    raise ValueError("GoHit Wavefunctions --baseline_subtract currently supports FFT mode only")
                transformed_entries = build_region_spectrum_entries(
                    records,
                    regions=None,
                    normalize_mode=args.normalize,
                    relative_range=tuple(args.relative_range),
                    flatten=args.flatten,
                    longest=args.longest,
                    handlenan=args.handlenan,
                    timeseriesnorm=args.timeseriesnorm,
                )
                profiles = compute_localization_profiles_from_entries(
                    transformed_entries,
                    peak_targets,
                    search_width=args.integration_window,
                    phase_table=(None if args.absvalues else phase_table),
                    baseline_subtract=True,
                    sqrtintpower=(not args.highest_bin),
                )
            else:
                if args.flatten:
                    if args.welch:
                        raise ValueError("GoHit Wavefunctions --flatten currently supports FFT mode only")
                    transformed_entries = build_region_spectrum_entries(
                        records,
                        regions=None,
                        normalize_mode=args.normalize,
                        relative_range=tuple(args.relative_range),
                        flatten=True,
                        longest=args.longest,
                        handlenan=args.handlenan,
                        timeseriesnorm=args.timeseriesnorm,
                    )
                    profiles = compute_localization_profiles_from_entries(
                        transformed_entries,
                        peak_targets,
                        search_width=args.integration_window,
                        phase_table=(None if args.absvalues else phase_table),
                        baseline_subtract=False,
                        sqrtintpower=(not args.highest_bin),
                    )
                else:
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
                    transformed_entries = None
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
        if args.show_diagnostics:
            if args.hits or args.flatten or args.baseline_subtract:
                if transformed_entries:
                    diagnostics_by_entity = build_region_peak_diagnostics_by_entity(
                        transformed_entries,
                        peak_targets,
                        search_width=args.integration_window,
                        baseline_subtract=args.baseline_subtract,
                    )
            elif contributions:
                diagnostics_by_entity = build_peak_diagnostics_by_entity(
                    contributions,
                    peak_targets,
                    normalize_mode=args.normalize,
                    relative_range=tuple(args.relative_range),
                    search_width=args.integration_window,
                )

        profiles = _shift_profiles_to_display_bonds(profiles)
        profiles = _maybe_hide_std_band(profiles, show_std_band=args.show_std_band)
        profiles, diagnostics_by_entity = _sort_profiles_high_to_low_frequency(profiles, diagnostics_by_entity)
        diagnostics_by_entity = _shift_diagnostics_labels(diagnostics_by_entity)

        localize_module = _load_localize_peaks_module()
        if args.title:
            title = args.title
        else:
            mode_desc = "highest bin" if args.highest_bin else "sqrt(integrated power)"
            if args.absvalues:
                phase_desc = "absolute values"
            else:
                phase_desc = "real-projected" if args.forcereal else "phase-projected"
            baseline_desc = "baseline-subtracted" if args.baseline_subtract else "raw-peak"
            flatten_desc = "flattened" if args.flatten else "unflattened"
            title = (
                f"Bond Wavefunctions | {args.dataset} | "
                f"{'GoHit Posthit FFT' if args.hits and args.posthit else 'GoHit Interhit FFT' if args.hits else 'Welch' if args.welch else 'FFT'} | {mode_desc} | {flatten_desc} | {baseline_desc} | {phase_desc}"
            )

        print(f"Dataset: {args.dataset}")
        print(f"Component: {args.show}")
        print(f"Bond spacing mode: {args.bond_spacing_mode}")
        print(f"Peaks file: {peaks_path}")
        print(f"Peaks (Hz): {peaks}")
        print(f"Bonds: {[bond_id + 1 for bond_id in bond_ids_zero]}")
        if args.hits:
            print(summarize_catalog(catalog))
            for line in describe_hit_region_settings(
                posthit=bool(args.posthit),
                delay=float(args.delay),
                exclude_before=float(args.exclude_before),
                hit_window=float(args.hit_window),
            ):
                print(line)
            print(f"Usable regions: {len(regions)}")
        if not args.absvalues and reference_bond_zero is not None:
            print(f"Reference bond: {reference_bond_zero + 1}")
            print(f"Reference peak index: {args.reference_peak_index}")
        print(f"Integration window half-width (Hz): {args.integration_window}")
        print(f"Amplitude mode: {'highest bin' if args.highest_bin else 'sqrt(integrated power)'}")
        print(f"Flatten: {'on' if args.flatten else 'off'}")
        print(f"Baseline subtraction: {'on' if args.baseline_subtract else 'off'}")
        print(f"Diagnostics panels: {'on' if args.show_diagnostics else 'off'}")
        print(f"Std band: {'on' if args.show_std_band else 'off'}")
        if not args.absvalues:
            if flip_bond_ids:
                print(f"Flipped bonds: {[bond_id + 1 for bond_id in flip_bond_ids]}")
            if args.forcereal:
                print("Phase snap: forced to {0, pi}")
        else:
            print("Phase mode: absolute values")

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

            fig = localize_module.plot_localization_profiles(
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
                save_path = _figure_save_path(args.save, fig_idx, n_figs)
                save_path = ensure_parent_dir(save_path)
                fig.savefig(save_path, dpi=300)
                print(f"Plot saved to: {save_path}")
            figs.append(fig)

        import matplotlib.pyplot as _plt
        backend = _plt.get_backend().lower()
        if len(figs) > 1 and "agg" in backend and args.save is not None:
            for fig in figs:
                _plt.close(fig)
        elif len(figs) > 1:
            for fig in figs:
                _plt.figure(fig.number)
            _plt.show(block=False)
            _plt.show()
        elif "agg" in backend and args.save is not None:
            _plt.close(figs[0])
        else:
            render_figure(figs[0], save=None)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
