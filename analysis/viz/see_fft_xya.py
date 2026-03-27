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
from plotting.frequency import (
    plot_component_pair_frequency_grid,
    plot_component_pair_frequency_grid_single_row_groups,
    plot_pair_frequency_grid,
    plot_pair_welch_frequency_grid,
)
from tools.cli import add_colormap_arg, add_output_args, add_signal_processing_args, add_track2_input_args
from tools.derived import derive_spacing_dataset
from tools.io import load_track2_dataset
from tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from tools.spectral import analyze_spacing_dataset_for_display, analyze_spacing_dataset_with_welch_for_display

COMPONENT_SUFFIXES = ("x", "y", "a")


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay FFT or Welch spectra and show per-component sliding FFTs for Track2-derived block spacing.",
    )
    add_track2_input_args(parser)
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
    parser.add_argument("--fft-min-hz", type=float, default=None)
    parser.add_argument("--fft-max-hz", type=float, default=None)
    parser.add_argument("--welch-min-hz", type=float, default=None)
    parser.add_argument("--welch-max-hz", type=float, default=None)
    parser.add_argument("--sliding-min-hz", type=float, default=None)
    parser.add_argument("--sliding-max-hz", type=float, default=None)
    parser.add_argument("--time-interval-s", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)
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
        help="Disable pair index (0-based). Can be used multiple times.",
    )
    parser.add_argument(
        "--disable-component",
        choices=COMPONENT_SUFFIXES,
        action="append",
        default=[],
        help="Disable one of the suffixed component datasets: x, y, or a. Can be used multiple times.",
    )
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
        print(f"  {result.pair_index}: {result.label}")

    valid = {int(result.pair_index) for result in available_pairs}
    default_choice = int(available_pairs[0].pair_index)
    while True:
        response = input(f"Which pair should be exported? [default: {default_choice}] ").strip()
        if response == "":
            return default_choice
        try:
            pair_index = int(response)
        except ValueError:
            print("Please enter a 0-based pair index.")
            continue
        if pair_index in valid:
            return pair_index
        print(f"Please choose one of: {sorted(valid)}")


def _load_component_results(args) -> tuple[dict[str, list], dict[str, object]]:
    base_dataset, data_root = _base_dataset_context(args.dataset, args.track2, args.track_data_root)
    disabled_components = set(args.disable_component)

    component_results: dict[str, list] = {}
    component_track2: dict[str, object] = {}
    missing_components: list[str] = []

    for component in COMPONENT_SUFFIXES:
        if component in disabled_components:
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

        spacing = derive_spacing_dataset(track2)
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
    spacing = derive_spacing_dataset(track2)
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        component_results, component_track2 = _load_component_results(args)
        if len(component_results) == 0:
            print("No component sibling datasets were found; using single-dataset mode.", file=sys.stderr)
            results, track2 = _load_single_results(args)
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
                    welch_min_hz=args.welch_min_hz,
                    welch_max_hz=args.welch_max_hz,
                    sliding_min_hz=args.sliding_min_hz,
                    sliding_max_hz=args.sliding_max_hz,
                    time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                    cmap_index=args.cm,
                    title=args.title,
                )
            else:
                fig = plot_pair_frequency_grid(
                    results,
                    fft_log=args.fft_log,
                    sliding_plot_scale=args.sliding_plot_scale,
                    only=args.only,
                    full_image=args.full_image,
                    full_couple=args.full_couple,
                    fft_min_hz=args.fft_min_hz,
                    fft_max_hz=args.fft_max_hz,
                    sliding_min_hz=args.sliding_min_hz,
                    sliding_max_hz=args.sliding_max_hz,
                    time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                    cmap_index=args.cm,
                    title=args.title,
                )
        else:
            if args.spectrasave is not None:
                _export_selected_spectrum(args, component_results, component_track2)
            plot_fn = (
                plot_component_pair_frequency_grid_single_row_groups
                if args.only == "sliding"
                else plot_component_pair_frequency_grid
            )
            fig = plot_fn(
                component_results,
                fft_log=args.fft_log,
                welch_log=args.welch_log,
                sliding_plot_scale=args.sliding_plot_scale,
                only=args.only,
                full_image=args.full_image,
                full_couple=args.full_couple,
                use_welch=args.welch,
                fft_min_hz=args.fft_min_hz,
                fft_max_hz=args.fft_max_hz,
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
