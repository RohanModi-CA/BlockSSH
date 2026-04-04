#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from plotting.common import render_figure
from plotting.frequency import plot_average_component_comparison, plot_average_spectrum
from tools.cli import (
    add_average_domain_args,
    add_bond_filter_args,
    add_bond_spacing_mode_arg,
    add_colormap_arg,
    add_flattening_args,
    add_frequency_window_args,
    add_normalization_args,
    add_output_args,
    add_plot_scale_args,
    resolve_normalization_mode,
    add_signal_processing_args,
    add_tickspace_arg,
    add_track_data_root_arg,
    validate_frequency_window_args,
    validate_tickspace_arg,
)
from tools.flattening import (
    apply_flattening_to_average_result,
    flattening_metadata,
    plot_flattening_diagnostic,
)
from tools.io import default_track2_path
from tools.models import DatasetSelection, FlatteningResult
from tools.selection import (
    build_configured_bond_signals,
    collect_display_bond_numbers,
    filter_signal_records_by_display_bonds,
    load_dataset_selection,
    load_dataset_selection_entries,
)
from tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from tools.spectral import (
    ABSOLUTE_ZERO_TOL,
    compute_average_spectrum,
    compute_fft_contributions,
    compute_reference_average_spectrum,
    compute_welch_contributions,
)

CANONICAL_COMPONENTS = ("x", "y", "a")


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average normalized FFT or Welch spectra across selected configured global bonds using Track2 permanence data.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    add_track_data_root_arg(parser)
    add_bond_spacing_mode_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_plot_scale_args(parser)
    add_signal_processing_args(parser)
    add_bond_filter_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)
    add_frequency_window_args(parser)
    add_tickspace_arg(parser)
    parser.add_argument(
        "--welch",
        action="store_true",
        help="Average Welch spectra instead of FFT spectra.",
    )
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument(
        "--compare-xya",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Compare overlapping x/y/a components from configs that use ordered 'contains'. Pass --compare-xya false to use legacy single-component loading.",
    )
    parser.add_argument(
        "--allow-duplicate-bonds",
        action="store_true",
        help="Allow the same configured bond id to contribute multiple times.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Render the averaged spectrum as a 2D frequency image instead of a 1D curve.",
    )
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)
    add_flattening_args(parser)
    add_spectrasave_arg(parser)
    return parser


def _format_bond_list(display_bonds: list[int]) -> str:
    if len(display_bonds) == 0:
        return "[]"
    if len(display_bonds) <= 12:
        return "[" + ", ".join(str(v) for v in display_bonds) + "]"
    head = ", ".join(str(v) for v in display_bonds[:10])
    return f"[{head}, ...] ({len(display_bonds)} total)"


def _logical_to_physical_suffix(contains: list[str]) -> dict[str, str]:
    return {
        logical_component: CANONICAL_COMPONENTS[idx]
        for idx, logical_component in enumerate(contains)
    }


def _component_overlap(raw_config: OrderedDict[str, dict]) -> list[str]:
    included = [entry for entry in raw_config.values() if entry["include"]]
    if len(included) == 0:
        return []
    shared = set(included[0]["contains"])
    for entry in included[1:]:
        shared &= set(entry["contains"])
    return [component for component in CANONICAL_COMPONENTS if component in shared]


def _resolved_component_config(
    raw_config: OrderedDict[str, dict],
    *,
    logical_component: str,
) -> OrderedDict[str, DatasetSelection]:
    resolved: OrderedDict[str, DatasetSelection] = OrderedDict()
    for dataset_name, entry in raw_config.items():
        if entry["include"] and entry["contains"] is not None:
            physical_suffix = _logical_to_physical_suffix(entry["contains"])[logical_component]
            resolved_name = f"{dataset_name}_{physical_suffix}"
        else:
            resolved_name = dataset_name
        resolved[resolved_name] = DatasetSelection(
            include=bool(entry["include"]),
            discards=list(entry["discards"]),
            pair_ids=list(entry["pair_ids"]),
        )
    return resolved


def _save_average_result(
    args,
    *,
    config_stem: str,
    result,
    spectrum_kind: str,
    source_kind: str,
    label: str,
    component: str | None = None,
    extra_metadata: dict | None = None,
    flattening: FlatteningResult | None = None,
) -> None:
    default_name_parts = [config_stem, source_kind, spectrum_kind]
    if component is not None:
        default_name_parts.append(f"component-{component}")
    default_name = build_default_spectrasave_name(*default_name_parts)
    export_path = resolve_spectrasave_path(
        args.spectrasave,
        default_name=default_name,
        multi_suffix=component,
    )
    if export_path is None:
        return

    metadata = {
        "sourceKind": source_kind,
        "spectrumKind": spectrum_kind,
        "configPath": args.config_json,
        "component": component,
        "normalize": args.normalize,
        "relativeRange": list(map(float, args.relative_range)),
        "averageDomain": args.average_domain,
        "freqLowHz": float(result.freq_low),
        "freqHighHz": float(result.freq_high),
        "normLowHz": float(result.norm_low),
        "normHighHz": float(result.norm_high),
        "contributors": len(result.contributors),
        "datasets": sorted({contrib.record.dataset_name for contrib in result.contributors}),
        "bondIds": sorted({int(contrib.record.entity_id) for contrib in result.contributors}),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if flattening is not None:
        metadata["flattening"] = flattening_metadata(flattening)

    saved = save_spectrum_msgpack(
        export_path,
        freq=result.freq_grid,
        amplitude=result.avg_amp,
        label=label,
        metadata=metadata,
    )
    print(f"Spectrum saved to: {saved}")


def _validate_flatten_args(args) -> str | None:
    flat_low, flat_high = map(float, args.flatten_reference_band)
    if flat_high <= flat_low:
        return "Error: --flatten-reference-band STOP_HZ must be greater than START_HZ"
    return None


def _maybe_apply_flattening(result, args) -> tuple[object, FlatteningResult | None]:
    from tools.flattening import apply_global_baseline_processing_to_results
    import collections

    if not args.flatten and getattr(args, "baseline_match", None) is None:
        return result, None

    ordered: collections.OrderedDict[str, object] = collections.OrderedDict(result=result)
    processed, flattening_by_key = apply_global_baseline_processing_to_results(
        ordered,  # type: ignore[arg-type]
        flatten=args.flatten,
        baseline_match=getattr(args, "baseline_match", None),
        reference_band=tuple(float(value) for value in args.flatten_reference_band),
    )
    return processed["result"], flattening_by_key.get("result")


def _maybe_emit_flatten_plot(
    args,
    *,
    freq_grid,
    raw_amp,
    flattening: FlatteningResult | None,
    title: str,
) -> None:
    if flattening is None or (args.flatten_plot is None and not args.flatten_show_plot):
        return
    fig = plot_flattening_diagnostic(freq_grid, raw_amp, flattening, title=title)
    if args.flatten_plot is not None:
        save_path = Path(args.flatten_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Flattening plot saved to: {save_path}")
    if args.flatten_show_plot:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _validate_compare_xya_inputs(
    raw_config: OrderedDict[str, dict],
    *,
    track_data_root: str | None,
) -> list[str]:
    included = [(dataset_name, entry) for dataset_name, entry in raw_config.items() if entry["include"]]
    if len(included) == 0:
        raise ValueError("No included datasets remain in the config")

    for dataset_name, entry in included:
        if entry["contains"] is None:
            raise ValueError(
                f"--compare-xya requires every included dataset to declare ordered 'contains'; dataset '{dataset_name}' does not"
            )

        suffix_map = _logical_to_physical_suffix(entry["contains"])
        for logical_component, physical_suffix in suffix_map.items():
            resolved_name = f"{dataset_name}_{physical_suffix}"
            track2_path = default_track2_path(resolved_name, track_data_root=track_data_root)
            if not track2_path.exists():
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' declares logical component '{logical_component}' via physical suffix "
                    f"'{physical_suffix}', but {track2_path} does not exist"
                )

    overlap = _component_overlap(raw_config)
    if len(overlap) == 0:
        raise ValueError("No overlapping logical components exist across the included datasets")
    return overlap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    if args.only == "fft":
        args.full_image = False
    elif args.only == "sliding":
        args.full_image = True

    freq_window_error = validate_frequency_window_args(args)
    if freq_window_error is not None:
        print(freq_window_error, file=sys.stderr)
        return 1
    tickspace_error = validate_tickspace_arg(args)
    if tickspace_error is not None:
        print(tickspace_error, file=sys.stderr)
        return 1
    flatten_error = _validate_flatten_args(args)
    if flatten_error is not None:
        print(flatten_error, file=sys.stderr)
        return 1

    if not args.flatten and args.baseline_match is not None and args.baseline_match != "none":
        import warnings
        warnings.warn(
            f"--baseline-match={args.baseline_match} is active without --flatten; "
            "warping component baselines multiplicatively to match component "
            f"'{args.baseline_match}'s curved response envelope. "
            "Pass --flatten to also flatten to a horizontal reference line.",
            UserWarning,
        )

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        config_stem = Path(args.config_json).stem
        norm_desc = args.normalize
        if args.normalize == "relative":
            title_range = tuple(args.relative_range)
            norm_desc = f"relative [{title_range[0]}, {title_range[1]}] Hz"

        if args.compare_xya:
            raw_config = load_dataset_selection_entries(args.config_json)
            overlap = _validate_compare_xya_inputs(
                raw_config,
                track_data_root=args.track_data_root,
            )

            results_by_component = OrderedDict()
            raw_results_by_component = OrderedDict()
            flattening_by_component: dict[str, FlatteningResult] = {}
            reference_amp_by_component: dict[str, object] = {}
            available_display_bonds = None
            selected_display_bonds = None

            for logical_component in overlap:
                config = _resolved_component_config(raw_config, logical_component=logical_component)
                records = build_configured_bond_signals(
                    config,
                    track_data_root=args.track_data_root,
                    allow_duplicate_ids=args.allow_duplicate_bonds,
                    bond_spacing_mode=args.bond_spacing_mode,
                )
                component_available = collect_display_bond_numbers(records)
                records = filter_signal_records_by_display_bonds(
                    records,
                    only_bonds=args.only_bonds,
                    exclude_bonds=args.exclude_bonds,
                    parity=args.bond_parity,
                )
                component_selected = collect_display_bond_numbers(records)
                if len(records) == 0:
                    raise ValueError(
                        f"Bond selection removed all configured bond contributors for component '{logical_component}'"
                    )

                contributions = (
                    compute_welch_contributions(
                        records,
                        welch_len_s=args.welch_len_s,
                        welch_overlap_fraction=args.welch_overlap,
                        longest=args.longest,
                        handlenan=args.handlenan,
                        timeseriesnorm=args.timeseriesnorm,
                    )
                    if args.welch
                    else compute_fft_contributions(
                        records,
                        longest=args.longest,
                        handlenan=args.handlenan,
                        timeseriesnorm=args.timeseriesnorm,
                    )
                )
                if len(contributions) == 0:
                    raise ValueError(
                        f"No spectra were accepted from the selected bond contributors for component '{logical_component}'"
                    )

                raw_result = compute_average_spectrum(
                    contributions,
                    normalize_mode=args.normalize,
                    relative_range=tuple(args.relative_range),
                    average_domain=args.average_domain,
                    lowest_freq=args.freq_min_hz,
                    highest_freq=args.freq_max_hz,
                )
                result, flattening = _maybe_apply_flattening(raw_result, args)
                raw_results_by_component[logical_component] = raw_result
                results_by_component[logical_component] = result
                if flattening is not None:
                    flattening_by_component[logical_component] = flattening

                if args.full_image and args.plot_scale == "linear":
                    reference = compute_reference_average_spectrum(
                        contributions,
                        normalize_mode=args.normalize,
                        relative_range=tuple(args.relative_range),
                        average_domain=args.average_domain,
                    )
                    reference_amp_by_component[logical_component] = reference.avg_amp

                if available_display_bonds is None:
                    available_display_bonds = component_available
                if selected_display_bonds is None:
                    selected_display_bonds = component_selected

            lead_result = next(iter(results_by_component.values()))
            accepted_display_bonds = sorted({
                int(contrib.record.entity_id) + 1
                for result in results_by_component.values()
                for contrib in result.contributors
            })
            total_contributors = sum(len(result.contributors) for result in results_by_component.values())
            n_datasets = len({
                contrib.record.dataset_name
                for result in results_by_component.values()
                for contrib in result.contributors
            })

            title = args.title or (
                f"Average {'Welch' if args.welch else 'FFT'} X/Y/A Comparison | "
                f"components={','.join(results_by_component.keys())} | datasets={n_datasets} | "
                f"bonds={len(accepted_display_bonds)} | avg={args.average_domain} | norm={norm_desc}"
            )

            print(f"Available configured display bonds: {_format_bond_list(available_display_bonds or [])}")
            print(f"Selected display bonds: {_format_bond_list(selected_display_bonds or [])}")
            print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
            print(f"Compared logical components: {list(results_by_component.keys())}")
            print(f"Total accepted contributors: {total_contributors}")
            print(f"Unique datasets: {n_datasets}")
            print(f"Frequency window: [{lead_result.freq_low:.6f}, {lead_result.freq_high:.6f}] Hz")
            print(f"Normalization window: [{lead_result.norm_low:.6f}, {lead_result.norm_high:.6f}] Hz")
            print("Normalization band processing: linear detrend -> zero-floor -> integrate area")
            print(f"Near-zero denominator threshold: {ABSOLUTE_ZERO_TOL:.0e}")
            print(f"Common grid points: {len(lead_result.freq_grid)}")
            print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
            if args.welch:
                print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
            print(f"Display mode: {'full image' if args.full_image else 'curve'}")
            if args.full_image and args.plot_scale == "linear":
                print("Image color scale reference: full implicit frequency window")
            if args.flatten:
                print(
                    "Flattening: "
                    f"enabled with reference band [{args.flatten_reference_band[0]:.6g}, {args.flatten_reference_band[1]:.6g}] Hz"
                )
                component = "x" if "x" in flattening_by_component else next(iter(flattening_by_component))
                _maybe_emit_flatten_plot(
                    args,
                    freq_grid=raw_results_by_component[component].freq_grid,
                    raw_amp=raw_results_by_component[component].avg_amp,
                    flattening=flattening_by_component[component],
                    title=f"{config_stem} component {component} flattening diagnostic",
                )

            fig = plot_average_component_comparison(
                results_by_component,
                full_image=args.full_image,
                plot_scale=args.plot_scale,
                cmap_index=args.cm,
                title=title,
                reference_amp_for_norm_by_component=reference_amp_by_component or None,
                tickspace_hz=args.tickspace_hz,
            )
            if args.spectrasave is not None:
                for logical_component, result in results_by_component.items():
                    _save_average_result(
                        args,
                        config_stem=config_stem,
                        result=result,
                        spectrum_kind="welch" if args.welch else "fft",
                        source_kind="avg",
                        label=f"{config_stem} average {logical_component}",
                        component=logical_component,
                        extra_metadata={"compareXya": True},
                        flattening=flattening_by_component.get(logical_component),
                    )
        else:
            config = load_dataset_selection(args.config_json)
            records = build_configured_bond_signals(
                config,
                track_data_root=args.track_data_root,
                allow_duplicate_ids=args.allow_duplicate_bonds,
                bond_spacing_mode=args.bond_spacing_mode,
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

            contributions = (
                compute_welch_contributions(
                    records,
                    welch_len_s=args.welch_len_s,
                    welch_overlap_fraction=args.welch_overlap,
                    longest=args.longest,
                    handlenan=args.handlenan,
                    timeseriesnorm=args.timeseriesnorm,
                )
                if args.welch
                else compute_fft_contributions(
                    records,
                    longest=args.longest,
                    handlenan=args.handlenan,
                    timeseriesnorm=args.timeseriesnorm,
                )
            )
            if len(contributions) == 0:
                raise ValueError("No spectra were accepted from the selected bond contributors")

            raw_result = compute_average_spectrum(
                contributions,
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                average_domain=args.average_domain,
                lowest_freq=args.freq_min_hz,
                highest_freq=args.freq_max_hz,
            )
            result, flattening = _maybe_apply_flattening(raw_result, args)

            reference_amp = None
            if args.full_image and args.plot_scale == "linear":
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

            title = args.title or (
                f"Average {'Welch' if args.welch else 'FFT'} | contributors={n_contributors} | datasets={n_datasets} | "
                f"bonds={len(accepted_display_bonds)} | avg={args.average_domain} | norm={norm_desc}"
            )

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
            print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
            if args.welch:
                print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
            print(f"Display mode: {'full image' if args.full_image else 'curve'}")
            if args.full_image and args.plot_scale == "linear":
                print("Image color scale reference: full implicit frequency window")
            if args.flatten:
                print(
                    "Flattening: "
                    f"enabled with reference band [{args.flatten_reference_band[0]:.6g}, {args.flatten_reference_band[1]:.6g}] Hz"
                )

            _maybe_emit_flatten_plot(
                args,
                freq_grid=raw_result.freq_grid,
                raw_amp=raw_result.avg_amp,
                flattening=flattening,
                title=f"{config_stem} flattening diagnostic",
            )

            fig = plot_average_spectrum(
                result,
                full_image=args.full_image,
                plot_scale=args.plot_scale,
                cmap_index=args.cm,
                title=title,
                reference_amp_for_norm=reference_amp,
                tickspace_hz=args.tickspace_hz,
            )
            if args.spectrasave is not None:
                _save_average_result(
                    args,
                    config_stem=config_stem,
                    result=result,
                    spectrum_kind="welch" if args.welch else "fft",
                    source_kind="avg",
                    label=f"{config_stem} average",
                    extra_metadata={"compareXya": False},
                    flattening=flattening,
                )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
