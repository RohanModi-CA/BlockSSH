#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from collections.abc import Collection
from pathlib import Path

import numpy as np

from plotting.common import render_figure
from plotting.frequency import (
    plot_average_component_comparison,
    plot_average_spectrum,
    plot_component_pair_frequency_grid,
    plot_component_pair_frequency_grid_single_row_groups,
    plot_pair_frequency_grid,
    plot_pair_welch_frequency_grid,
)
from tools.bonds import load_bond_signal_dataset
from tools.cli import (
    add_bond_spacing_mode_arg,
    add_colormap_arg,
    add_flattening_args,
    add_frequency_window_args,
    add_output_args,
    add_signal_processing_args,
    add_tickspace_arg,
    add_track2_input_args,
    validate_frequency_window_args,
    validate_tickspace_arg,
)
from tools.flattening import (
    apply_flattening_to_average_result,
    apply_flattening_to_pair_result,
    flattening_metadata,
    plot_flattening_diagnostic,
)
from tools.io import load_track2_dataset
from tools.models import (
    AverageSpectrumResult,
    FFTResult,
    FlatteningResult,
    PairFrequencyAnalysisResult,
    PairWelchFrequencyAnalysisResult,
    SignalRecord,
    SpectrumContribution,
    SpacingDataset,
)
from tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from tools.spectral import (
    analyze_spacing_dataset_for_display,
    analyze_spacing_dataset_with_welch_for_display,
    compute_mean_amplitude_spectrum,
)

COMPONENT_SUFFIXES = ("x", "y", "a")


def _maybe_apply_flattening_to_results(
    args,
    results: list[PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult],
) -> list[PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult]:
    if not args.flatten:
        return results
    return [
        apply_flattening_to_pair_result(
            result,
            reference_band=tuple(float(value) for value in args.flatten_reference_band),
        )
        for result in results
    ]


def _maybe_apply_flattening_to_component_results(
    args,
    component_results: dict[str, list[PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult]],
) -> dict[str, list[PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult]]:
    if not args.flatten:
        return component_results
    return {
        component: _maybe_apply_flattening_to_results(args, results)
        for component, results in component_results.items()
    }


def _spacing_dataset_from_track2(track2, *, bond_spacing_mode: str, component: str | None = None) -> SpacingDataset:
    bond_dataset = load_bond_signal_dataset(
        dataset=track2.dataset_name,
        track2_path=track2.track2_path,
        bond_spacing_mode=bond_spacing_mode,
        component=component,
    )
    return SpacingDataset(
        track2=track2,
        pair_labels=list(bond_dataset.pair_labels),
        spacing_matrix=np.asarray(bond_dataset.signal_matrix, dtype=float),
    )


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _convert_pair_indices_to_zero_based(indices: list[int] | None) -> list[int] | None:
    if indices is None:
        return None
    return [i - 1 for i in indices]


def _filter_to_only_pairs(results: list, only_pairs_zero_based: Collection[int]) -> list:
    return [r for r in results if r.pair_index in only_pairs_zero_based]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay FFT or Welch spectra and show per-component sliding FFTs for Track2-derived block spacing.",
    )
    add_track2_input_args(parser)
    add_bond_spacing_mode_arg(parser)
    add_signal_processing_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--fft-log",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Use a log y-axis for the FFT curve panel. Pass --fft-log false for a linear axis.",
    )

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

    parser.add_argument("--sliding-len-s", type=float, default=20.0)
    welch_group = parser.add_mutually_exclusive_group()
    welch_group.add_argument(
        "--welch",
        dest="welch",
        action="store_true",
        help="Replace the left FFT panel with a Welch spectrum panel (default).",
    )
    welch_group.add_argument(
        "--no-welch",
        dest="welch",
        action="store_false",
        help="Use FFT instead of Welch for the left spectrum panel.",
    )
    parser.set_defaults(welch=True)
    parser.add_argument(
        "--welch-log",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Use a log y-axis for the Welch curve panel. Pass --welch-log false for a linear axis.",
    )
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    add_frequency_window_args(parser)
    add_tickspace_arg(parser)
    parser.add_argument("--time-interval-s", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)
    parser.add_argument(
        "--average",
        action="store_true",
        help="Average all displayed bonds together within each component instead of plotting them separately.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Replace each sliding spectrogram panel with a 2D image of the full FFT spectrum.",
    )
    parser.add_argument(
        "--full-couple",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Couple FFT-panel frequency zoom with the image-panel frequency axis. Pass --full-couple false to disable.",
    )
    parser.add_argument(
        "--disable",
        type=int,
        action="append",
        default=[],
        help="Disable pair index (1-based). Can be used multiple times.",
    )
    parser.add_argument(
        "--only-pairs",
        type=int,
        nargs="+",
        default=None,
        help="Only include these pair indices (1-based). If not specified, all non-disabled pairs are included.",
    )
    parser.add_argument(
        "--disable-component",
        choices=COMPONENT_SUFFIXES,
        action="append",
        default=[],
        help="Disable one of the suffixed component datasets: x, y, or a. Can be used multiple times.",
    )
    parser.add_argument(
        "--only-component",
        choices=COMPONENT_SUFFIXES,
        action="append",
        default=[],
        help="Only include this component (x, y, or a). Can be used multiple times.",
    )
    add_flattening_args(parser)
    add_spectrasave_arg(parser)
    return parser


def _strip_component_suffix(name: str) -> str:
    for suffix in COMPONENT_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            return name[: -len(token)]
    return name


def _base_dataset_context(
    dataset: str | None,
    track2_path: str | None,
    track_data_root: str,
) -> tuple[str, str]:
    if track2_path is not None:
        path = Path(track2_path).resolve()
        if path.name != "track2_permanence.msgpack":
            raise ValueError("--track2 must point to a track2_permanence.msgpack file")
        return _strip_component_suffix(path.parent.name), str(path.parent.parent)

    if dataset is None:
        raise ValueError("Provide either DATASET or --track2")

    return _strip_component_suffix(dataset), track_data_root


def _prompt_component_choice(available: list[str]) -> str:
    default_choice = "x" if "x" in available else available[0]
    while True:
        response = input(
            f"Which component should be exported ({'/'.join(available)})? [default: {default_choice}] "
        ).strip().lower()
        if response == "":
            return default_choice
        if response in available:
            return response
        print(f"Please choose one of: {', '.join(available)}")


def _prompt_save_component_choice(available: list[str], *, spectrum_kind: str) -> str:
    default_choice = "x" if "x" in available else available[0]
    while True:
        response = input(
            f"Which component should be saved for the averaged {spectrum_kind.upper()} spectrum "
            f"({ '/'.join(available) })? [default: {default_choice}] "
        ).strip().lower()
        if response == "":
            return default_choice
        if response in available:
            return response
        print(f"Please choose one of: {', '.join(available)}")


def _get_exportable_spectrum(result, use_welch: bool):
    return getattr(result, "welch_result" if use_welch else "fft_result", None)


def _prompt_pair_choice(component: str, results: list, *, use_welch: bool) -> int:
    available_pairs = [result for result in results if _get_exportable_spectrum(result, use_welch=use_welch) is not None]
    if not available_pairs:
        raise ValueError(f"Component '{component}' has no exportable spectrum")
    if len(available_pairs) == 1:
        return int(available_pairs[0].pair_index)

    print(f"Available pairs for component '{component}':")
    for result in available_pairs:
        print(f"  {result.pair_index + 1}: {result.label}")

    valid = {int(result.pair_index) + 1 for result in available_pairs}
    default_choice = int(available_pairs[0].pair_index) + 1
    while True:
        response = input(f"Which pair should be exported (1-based)? [default: {default_choice}] ").strip()
        if response == "":
            return default_choice - 1
        try:
            pair_index = int(response)
        except ValueError:
            print("Please enter a 1-based pair index.")
            continue
        if pair_index in valid:
            return pair_index - 1
        print(f"Please choose one of: {sorted(valid)}")


def _load_component_results(args) -> tuple[dict[str, list], dict[str, object]]:
    base_dataset, data_root = _base_dataset_context(args.dataset, args.track2, args.track_data_root)
    disabled_components = set(args.disable_component)
    only_components = set(args.only_component)

    component_results: dict[str, list] = {}
    component_track2: dict[str, object] = {}
    missing_components: list[str] = []

    for component in COMPONENT_SUFFIXES:
        if component in disabled_components:
            continue
        if only_components and component not in only_components:
            continue

        dataset_name = f"{base_dataset}_{component}"
        try:
            track2 = load_track2_dataset(
                dataset=dataset_name,
                track_data_root=data_root,
            )
        except FileNotFoundError:
            missing_components.append(component)
            continue

        spacing = _spacing_dataset_from_track2(
            track2,
            bond_spacing_mode=args.bond_spacing_mode,
            component=component,
        )
        if args.welch:
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
        else:
            results = analyze_spacing_dataset_for_display(
                spacing,
                disabled_indices=args.disable,
                longest=args.longest,
                handlenan=args.handlenan,
                timeseriesnorm=args.timeseriesnorm,
                sliding_len_s=args.sliding_len_s,
            )
        component_results[component] = results
        component_track2[component] = track2

        dt = np.diff(track2.frame_times_s)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        print(f"Track2 ({component}): {track2.track2_path}")
        if dt.size > 0:
            approx_fs = 1.0 / float(np.median(dt))
            approx_nyquist = 0.5 * approx_fs
            print(f"Approx sampling rate ({component}): {approx_fs:.4f} Hz | Approx Nyquist: {approx_nyquist:.4f} Hz")
        else:
            print(f"Approx sampling rate ({component}): unavailable")

    if missing_components:
        print(
            "Missing component datasets: "
            + ", ".join(f"{base_dataset}_{component}" for component in missing_components),
            file=sys.stderr,
        )

    return component_results, component_track2


def _analyze_spacing(args, spacing):
    if args.welch:
        return analyze_spacing_dataset_with_welch_for_display(
            spacing,
            disabled_indices=args.disable,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            sliding_len_s=args.sliding_len_s,
        )
    return analyze_spacing_dataset_for_display(
        spacing,
        disabled_indices=args.disable,
        longest=args.longest,
        handlenan=args.handlenan,
        timeseriesnorm=args.timeseriesnorm,
        sliding_len_s=args.sliding_len_s,
    )


def _print_track2_summary(track2, *, component: str | None = None) -> None:
    dt = np.diff(track2.frame_times_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    prefix = f" ({component})" if component is not None else ""
    print(f"Track2{prefix}: {track2.track2_path}")
    if dt.size > 0:
        approx_fs = 1.0 / float(np.median(dt))
        approx_nyquist = 0.5 * approx_fs
        print(f"Approx sampling rate{prefix}: {approx_fs:.4f} Hz | Approx Nyquist: {approx_nyquist:.4f} Hz")
    else:
        print(f"Approx sampling rate{prefix}: unavailable")


def _load_single_results(args):
    track2 = load_track2_dataset(
        dataset=args.dataset,
        track2_path=args.track2,
        track_data_root=args.track_data_root,
    )
    spacing = _spacing_dataset_from_track2(
        track2,
        bond_spacing_mode=args.bond_spacing_mode,
        component=_strip_component_suffix(track2.dataset_name or "") if track2.dataset_name is not None else None,
    )
    results = _analyze_spacing(args, spacing)
    _print_track2_summary(track2)
    return results, track2


def _export_selected_spectrum(args, component_results: dict[str, list], component_track2: dict[str, object]) -> None:
    available_components = [
        component
        for component, results in component_results.items()
        if any(_get_exportable_spectrum(result, use_welch=args.welch) is not None for result in results)
    ]
    if not available_components:
        raise ValueError("No exportable spectra were available")

    if len(available_components) == 1:
        export_component = available_components[0]
    else:
        export_component = _prompt_component_choice(available_components)

    export_pair = _prompt_pair_choice(export_component, component_results[export_component], use_welch=args.welch)
    matching = [result for result in component_results[export_component] if result.pair_index == export_pair]
    if not matching:
        raise ValueError(f"Requested export pair {export_pair} was not available for component '{export_component}'")
    export_result = matching[0]

    spectrum_result = _get_exportable_spectrum(export_result, use_welch=args.welch)
    spectrum_kind = "welch" if args.welch else "fft"
    if spectrum_result is None:
        raise ValueError(f"Component '{export_component}' pair {export_pair} has no {spectrum_kind.upper()} result to export")

    track2 = component_track2[export_component]
    dataset_name = track2.dataset_name or Path(track2.track2_path).resolve().parent.name
    export_path = resolve_spectrasave_path(
        args.spectrasave,
        default_name=build_default_spectrasave_name(
            dataset_name,
            f"pair-{export_result.pair_index}",
            spectrum_kind,
        ),
    )
    assert export_path is not None

    metadata = {
        "sourceKind": "single",
        "spectrumKind": spectrum_kind,
        "dataset": dataset_name,
        "track2Path": track2.track2_path,
        "component": export_component,
        "pairIndex": int(export_result.pair_index),
        "pairLabel": export_result.label,
        "longest": bool(args.longest),
        "handlenan": bool(args.handlenan),
    }
    if args.welch:
        metadata["welchLenS"] = float(args.welch_len_s)
        metadata["welchOverlap"] = float(args.welch_overlap)

    saved = save_spectrum_msgpack(
        export_path,
        freq=spectrum_result.freq,
        amplitude=spectrum_result.amplitude,
        label=f"{dataset_name} pair {export_result.pair_index} {spectrum_kind.upper()}",
        metadata=metadata,
    )
    print(f"Spectrum saved to: {saved}")


def _export_single_spectrum(args, results: list, track2) -> None:
    available_pairs = [result for result in results if _get_exportable_spectrum(result, use_welch=args.welch) is not None]
    if not available_pairs:
        raise ValueError("No exportable spectra were available")

    export_pair = _prompt_pair_choice("dataset", results, use_welch=args.welch)
    matching = [result for result in results if result.pair_index == export_pair]
    if not matching:
        raise ValueError(f"Requested export pair {export_pair} was not available")
    export_result = matching[0]

    spectrum_result = _get_exportable_spectrum(export_result, use_welch=args.welch)
    spectrum_kind = "welch" if args.welch else "fft"
    if spectrum_result is None:
        raise ValueError(f"Pair {export_pair} has no {spectrum_kind.upper()} result to export")

    dataset_name = track2.dataset_name or Path(track2.track2_path).resolve().parent.name
    export_path = resolve_spectrasave_path(
        args.spectrasave,
        default_name=build_default_spectrasave_name(
            dataset_name,
            f"pair-{export_result.pair_index}",
            spectrum_kind,
        ),
    )
    assert export_path is not None

    metadata = {
        "sourceKind": "single",
        "spectrumKind": spectrum_kind,
        "dataset": dataset_name,
        "track2Path": track2.track2_path,
        "pairIndex": int(export_result.pair_index),
        "pairLabel": export_result.label,
        "longest": bool(args.longest),
        "handlenan": bool(args.handlenan),
    }
    if args.welch:
        metadata["welchLenS"] = float(args.welch_len_s)
        metadata["welchOverlap"] = float(args.welch_overlap)

    saved = save_spectrum_msgpack(
        export_path,
        freq=spectrum_result.freq,
        amplitude=spectrum_result.amplitude,
        label=f"{dataset_name} pair {export_result.pair_index} {spectrum_kind.upper()}",
        metadata=metadata,
    )
    print(f"Spectrum saved to: {saved}")


def _average_result_from_pair_results(results: list, *, dataset_name: str, use_welch: bool) -> AverageSpectrumResult:
    contributions: list[SpectrumContribution] = []

    for result in results:
        spectrum_result = getattr(result, "welch_result", None) if use_welch else getattr(result, "fft_result", None)
        if result.error_message is not None or result.processed is None or spectrum_result is None:
            continue

        contributions.append(
            SpectrumContribution(
                record=SignalRecord(
                    dataset_name=str(dataset_name),
                    entity_id=int(result.pair_index),
                    local_index=int(result.pair_index),
                    label=str(result.label),
                    signal_kind="bond",
                    source_path="",
                    t=result.processed.t,
                    y=result.processed.y,
                ),
                processed=result.processed,
                fft_result=FFTResult(
                    freq=spectrum_result.freq,
                    amplitude=spectrum_result.amplitude,
                ),
            )
        )

    if len(contributions) == 0:
        raise ValueError("No valid bond spectra were available to average")

    averaged = compute_mean_amplitude_spectrum(contributions)
    return AverageSpectrumResult(
        freq_grid=averaged.freq_grid,
        avg_amp=averaged.mean_amplitude,
        norm_low=float(averaged.freq_low),
        norm_high=float(averaged.freq_high),
        freq_low=float(averaged.freq_low),
        freq_high=float(averaged.freq_high),
        contributors=averaged.contributors,
    )


def _save_averaged_component_spectrum(
    args,
    *,
    dataset_name: str,
    averaged_by_component: dict[str, AverageSpectrumResult],
    flattening_by_component: dict[str, FlatteningResult] | None,
    spectrum_kind: str,
) -> None:
    available_components = list(averaged_by_component)
    if not available_components:
        raise ValueError("No averaged component spectra were available to save")

    if len(available_components) == 1:
        component = available_components[0]
    else:
        component = _prompt_save_component_choice(available_components, spectrum_kind=spectrum_kind)

    result = averaged_by_component[component]
    default_name = build_default_spectrasave_name(dataset_name, "average-bonds", component, spectrum_kind)
    export_path = resolve_spectrasave_path(
        args.spectrasave,
        default_name=default_name,
    )
    if export_path is None:
        return

    saved = save_spectrum_msgpack(
        export_path,
        freq=result.freq_grid,
        amplitude=result.avg_amp,
        label=f"{dataset_name} average bonds {component} {spectrum_kind.upper()}",
        metadata={
            "sourceKind": "single",
            "spectrumKind": spectrum_kind,
            "dataset": dataset_name,
            "component": component,
            "averageBonds": True,
            "contributors": len(result.contributors),
            "bondIds": sorted({int(contrib.record.entity_id) for contrib in result.contributors}),
            "longest": bool(args.longest),
            "handlenan": bool(args.handlenan),
            "timeseriesnorm": bool(args.timeseriesnorm),
            "flattening": None
            if flattening_by_component is None or component not in flattening_by_component
            else flattening_metadata(flattening_by_component[component]),
        },
    )
    print(f"Spectrum saved to: {saved}")


def _validate_flatten_args(args) -> str | None:
    flat_low, flat_high = map(float, args.flatten_reference_band)
    if flat_high <= flat_low:
        return "Error: --flatten-reference-band STOP_HZ must be greater than START_HZ"
    return None


def _maybe_apply_flattening(
    args,
    results_by_component: dict[str, AverageSpectrumResult],
) -> tuple[dict[str, AverageSpectrumResult], dict[str, FlatteningResult]]:
    from tools.flattening import apply_global_baseline_processing_to_results
    import collections

    if not args.flatten and getattr(args, "baseline_match", None) is None:
        return dict(results_by_component), {}

    ordered_results = collections.OrderedDict(results_by_component)
    processed_results, flattening_by_component = apply_global_baseline_processing_to_results(
        ordered_results,
        flatten=args.flatten,
        baseline_match=getattr(args, "baseline_match", None),
        reference_band=tuple(float(value) for value in args.flatten_reference_band),
    )
    return dict(processed_results), flattening_by_component


def _maybe_emit_flatten_plot(
    args,
    *,
    dataset_name: str,
    results_by_component: dict[str, AverageSpectrumResult],
    flattening_by_component: dict[str, FlatteningResult],
) -> None:
    if not args.flatten or (args.flatten_plot is None and not args.flatten_show_plot):
        return
    if not flattening_by_component:
        return

    component = "x" if "x" in flattening_by_component else next(iter(flattening_by_component))
    raw_result = results_by_component[component]
    flattening = flattening_by_component[component]
    fig = plot_flattening_diagnostic(
        raw_result.freq_grid,
        raw_result.avg_amp,
        flattening,
        title=f"{dataset_name} component {component} flattening diagnostic",
    )
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.disable = _convert_pair_indices_to_zero_based(args.disable)
    args.only_pairs = _convert_pair_indices_to_zero_based(args.only_pairs)

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

    if not args.flatten and getattr(args, "baseline_match", None) is not None:
        import warnings
        warnings.warn(
            f"--baseline-match={args.baseline_match} is active without --flatten; "
            "warping component baselines multiplicatively to match component "
            f"{args.baseline_match}'s curved response envelope. "
            "Pass --flatten to also flatten to a horizontal reference line.",
            UserWarning,
        )

    if args.average:
        if args.only == "fft":
            args.full_image = False
        elif args.only == "sliding":
            args.full_image = True

    try:
        component_results, component_track2 = _load_component_results(args)
        if args.only_pairs is not None:
            only_pairs_set = set(args.only_pairs)
            component_results = {
                comp: _filter_to_only_pairs(results, only_pairs_set)
                for comp, results in component_results.items()
            }
        if len(component_results) == 0:
            print("No component sibling datasets were found; using single-dataset mode.", file=sys.stderr)
            results, track2 = _load_single_results(args)
            if args.only_pairs is not None:
                results = _filter_to_only_pairs(results, set(args.only_pairs))
            if args.average:
                dataset_name = track2.dataset_name or Path(track2.track2_path).resolve().parent.name
                averaged_raw = _average_result_from_pair_results(results, dataset_name=dataset_name, use_welch=args.welch)
                flattened_by_component, flattening_by_component = _maybe_apply_flattening(
                    args,
                    {"x": averaged_raw},
                )
                averaged = flattened_by_component["x"]
                if args.spectrasave is not None:
                    _save_averaged_component_spectrum(
                        args,
                        dataset_name=dataset_name,
                        averaged_by_component={"x": averaged},
                        flattening_by_component=flattening_by_component,
                        spectrum_kind="welch" if args.welch else "fft",
                    )
                _maybe_emit_flatten_plot(
                    args,
                    dataset_name=dataset_name,
                    results_by_component={"x": averaged_raw},
                    flattening_by_component=flattening_by_component,
                )
                fig = plot_average_spectrum(
                    averaged,
                    full_image=args.full_image,
                    plot_scale=args.sliding_plot_scale if args.full_image else ("log" if args.welch and args.welch_log else "log" if args.fft_log else "linear"),
                    cmap_index=args.cm,
                    title=args.title or f"{dataset_name} average bonds",
                    tickspace_hz=args.tickspace,
                )
            else:
                results = _maybe_apply_flattening_to_results(args, results)
                if args.spectrasave is not None:
                    _export_single_spectrum(args, results, track2)

                if args.welch:
                    fig = plot_pair_welch_frequency_grid(
                        results,
                        welch_log=args.welch_log,
                        sliding_plot_scale=args.sliding_plot_scale,
                        only="welch" if args.only == "fft" else args.only,
                        full_image=args.full_image,
                        full_couple=args.full_couple,
                        welch_min_hz=args.freq_min_hz,
                        welch_max_hz=args.freq_max_hz,
                        sliding_min_hz=args.freq_min_hz,
                        sliding_max_hz=args.freq_max_hz,
                        time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                        cmap_index=args.cm,
                        title=args.title,
                        tickspace_hz=args.tickspace,
                    )
                else:
                    fig = plot_pair_frequency_grid(
                        results,
                        fft_log=args.fft_log,
                        sliding_plot_scale=args.sliding_plot_scale,
                        only=args.only,
                        full_image=args.full_image,
                        full_couple=args.full_couple,
                        fft_min_hz=args.freq_min_hz,
                        fft_max_hz=args.freq_max_hz,
                        sliding_min_hz=args.freq_min_hz,
                        sliding_max_hz=args.freq_max_hz,
                        time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                        cmap_index=args.cm,
                        title=args.title,
                        tickspace_hz=args.tickspace,
                    )
        else:
            if args.average:
                averaged_raw_by_component = {
                    component: _average_result_from_pair_results(
                        results,
                        dataset_name=(component_track2[component].dataset_name or Path(component_track2[component].track2_path).resolve().parent.name),
                        use_welch=args.welch,
                    )
                    for component, results in component_results.items()
                }
                averaged_by_component, flattening_by_component = _maybe_apply_flattening(
                    args,
                    averaged_raw_by_component,
                )
                dataset_name = (
                    component_track2[next(iter(component_track2))].dataset_name
                    or Path(component_track2[next(iter(component_track2))].track2_path).resolve().parent.name
                )
                if args.spectrasave is not None:
                    _save_averaged_component_spectrum(
                        args,
                        dataset_name=dataset_name,
                        averaged_by_component=averaged_by_component,
                        flattening_by_component=flattening_by_component,
                        spectrum_kind="welch" if args.welch else "fft",
                    )
                _maybe_emit_flatten_plot(
                    args,
                    dataset_name=dataset_name,
                    results_by_component=averaged_raw_by_component,
                    flattening_by_component=flattening_by_component,
                )
                fig = plot_average_component_comparison(
                    averaged_by_component,
                    full_image=args.full_image,
                    plot_scale=args.sliding_plot_scale if args.full_image else ("log" if args.welch and args.welch_log else "log" if args.fft_log else "linear"),
                    cmap_index=args.cm,
                    title=args.title or f"{dataset_name} average bonds",
                    tickspace_hz=args.tickspace,
                )
            else:
                component_results = _maybe_apply_flattening_to_component_results(args, component_results)
                if args.spectrasave is not None:
                    _export_selected_spectrum(args, component_results, component_track2)
                fig = plot_component_pair_frequency_grid(
                    component_results,
                    fft_log=args.fft_log,
                    welch_log=args.welch_log,
                    sliding_plot_scale=args.sliding_plot_scale,
                    only=args.only,
                    full_image=args.full_image,
                    full_couple=args.full_couple,
                    use_welch=args.welch,
                    fft_min_hz=args.freq_min_hz,
                    fft_max_hz=args.freq_max_hz,
                    welch_min_hz=args.freq_min_hz,
                    welch_max_hz=args.freq_max_hz,
                    sliding_min_hz=args.freq_min_hz,
                    sliding_max_hz=args.freq_max_hz,
                    time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                    cmap_index=args.cm,
                    title=args.title,
                    tickspace_hz=args.tickspace,
                )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
