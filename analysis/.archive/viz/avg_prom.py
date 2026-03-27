#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, medfilt

from plotting.common import render_figure
from plotting.frequency import COMPONENT_COLORS
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
from tools.io import default_track2_path
from tools.models import DatasetSelection
from tools.selection import (
    build_configured_bond_signals,
    collect_display_bond_numbers,
    filter_signal_records_by_display_bonds,
    load_dataset_selection,
    load_dataset_selection_entries,
)
from tools.spectral import (
    ABSOLUTE_ZERO_TOL,
    compute_average_spectrum,
    compute_fft_contributions,
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
        description="Average spectra with a local-background prominence view.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
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
    parser.add_argument("--welch", action="store_true", help="Average Welch spectra instead of FFT spectra.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument(
        "--compare-xya",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Compare overlapping x/y/a components from configs that use ordered 'contains'. Pass false to use legacy single-component loading.",
    )
    parser.add_argument(
        "--only-component",
        choices=CANONICAL_COMPONENTS,
        default=None,
        help="Restrict processing to a single logical component: x, y, or a.",
    )
    parser.add_argument("--allow-duplicate-bonds", action="store_true")
    parser.add_argument("--full-image", action="store_true")
    parser.add_argument(
        "--prominence-mode",
        choices=["subtract", "divide", "db-ratio"],
        default="db-ratio",
        help="How to express signal above local background. Default: db-ratio",
    )
    parser.add_argument(
        "--baseline-window-bins",
        type=int,
        default=101,
        help="Odd median-filter window in frequency bins for local baseline estimation. Default: 101",
    )
    parser.add_argument(
        "--peak-min-distance-bins",
        type=int,
        default=7,
        help="Minimum spacing in bins between detected transformed-spectrum peaks. Default: 7",
    )
    parser.add_argument(
        "--peak-min-prominence",
        type=float,
        default=None,
        help="Optional minimum transformed-spectrum prominence threshold for peak markers.",
    )
    parser.add_argument(
        "--show-peaks",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show automatically detected peaks on curve plots. Default: true",
    )
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


def _legacy_config_with_forced_component(
    raw_config: OrderedDict[str, dict],
    *,
    logical_component: str,
) -> OrderedDict[str, DatasetSelection]:
    resolved: OrderedDict[str, DatasetSelection] = OrderedDict()
    for dataset_name, entry in raw_config.items():
        include = bool(entry["include"])
        discards = list(entry["discards"])
        pair_ids = list(entry["pair_ids"])
        contains = entry["contains"]

        if include and contains is not None:
            if logical_component not in contains:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not provide requested logical component '{logical_component}'"
                )
            physical_suffix = _logical_to_physical_suffix(contains)[logical_component]
            resolved_name = f"{dataset_name}_{physical_suffix}"
        else:
            resolved_name = dataset_name

        resolved[resolved_name] = DatasetSelection(
            include=include,
            discards=discards,
            pair_ids=pair_ids,
        )
    return resolved


def _compute_baseline(amp: np.ndarray, window_bins: int) -> np.ndarray:
    amp = np.asarray(amp, dtype=float)
    if amp.ndim != 1:
        raise ValueError("amp must be 1D")
    if amp.size == 0:
        return amp.copy()

    kernel = max(3, int(window_bins))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > amp.size:
        kernel = amp.size if amp.size % 2 == 1 else max(1, amp.size - 1)
    if kernel < 3:
        return np.full_like(amp, float(np.nanmedian(amp)))

    baseline = medfilt(amp, kernel_size=kernel)
    positive = baseline[np.isfinite(baseline) & (baseline > 0)]
    floor = np.min(positive) if positive.size > 0 else np.finfo(float).eps
    return np.maximum(baseline, floor)


def _transform_against_baseline(amp: np.ndarray, baseline: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    eps = np.finfo(float).eps
    amp = np.asarray(amp, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if mode == "subtract":
        return amp - baseline, "Amplitude - Local Baseline"
    if mode == "divide":
        return amp / np.maximum(baseline, eps), "Amplitude / Local Baseline"
    if mode == "db-ratio":
        ratio = np.maximum(amp, eps) / np.maximum(baseline, eps)
        return 20.0 * np.log10(ratio), "Amplitude Above Local Baseline (dB)"
    raise ValueError(f"Unsupported prominence mode: {mode}")


def _find_transformed_peaks(
    transformed: np.ndarray,
    *,
    min_distance_bins: int,
    min_prominence: float | None,
) -> np.ndarray:
    kwargs = {"distance": max(1, int(min_distance_bins))}
    if min_prominence is not None:
        kwargs["prominence"] = float(min_prominence)
    peaks, _ = find_peaks(np.asarray(transformed, dtype=float), **kwargs)
    return peaks


def _plot_component_curves(
    transformed_by_component: dict[str, dict[str, np.ndarray | str]],
    *,
    plot_scale: str,
    title: str | None,
    show_peaks: bool,
    peak_min_distance_bins: int,
    peak_min_prominence: float | None,
):
    components = [component for component in CANONICAL_COMPONENTS if component in transformed_by_component]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    ax_raw, ax_trans = axes

    y_label = None
    for component in components:
        payload = transformed_by_component[component]
        freq = payload["freq"]
        raw = payload["raw"]
        transformed = payload["transformed"]
        color = COMPONENT_COLORS[component]

        if plot_scale == "log":
            ax_raw.semilogy(freq, raw, linewidth=1.2, color=color, label=component)
        else:
            ax_raw.plot(freq, raw, linewidth=1.2, color=color, label=component)
        ax_trans.plot(freq, transformed, linewidth=1.4, color=color, label=component)

        if show_peaks:
            peak_idx = _find_transformed_peaks(
                transformed,
                min_distance_bins=peak_min_distance_bins,
                min_prominence=peak_min_prominence,
            )
            if peak_idx.size > 0:
                ax_trans.scatter(
                    freq[peak_idx],
                    transformed[peak_idx],
                    s=18,
                    color=color,
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=3,
                )
        y_label = str(payload["label"])

    ax_raw.set_ylabel("Normalized Amplitude")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend()
    ax_raw.set_title(title or "Average Spectrum and Prominence View")

    ax_trans.set_xlabel("Frequency (Hz)")
    ax_trans.set_ylabel(y_label or "Prominence")
    ax_trans.grid(True, alpha=0.3)
    ax_trans.legend()

    return fig


def _apply_compact_image_axis_style(ax, *, show_right_ylabel: bool, y_label: str) -> None:
    if show_right_ylabel:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel(y_label, labelpad=10)
        ax.tick_params(axis="y", labelright=True, labelleft=False)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False, labelright=False)
    ax.tick_params(axis="both", which="both", length=2, pad=1)


def _plot_component_images(
    transformed_by_component: dict[str, dict[str, np.ndarray | str]],
    *,
    cmap_index: int,
    title: str | None,
):
    from plotting.common import centers_to_edges, colormap_name

    components = [component for component in CANONICAL_COMPONENTS if component in transformed_by_component]
    fig, axes = plt.subplots(
        1,
        len(components),
        figsize=(4.1 * len(components), 5.0),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        bottom=0.09,
        top=0.92 if title else 0.97,
        wspace=0.02,
    )

    lead_ax = axes[0]
    for ax in axes[1:]:
        ax.sharex(lead_ax)
        ax.sharey(lead_ax)

    for idx, component in enumerate(components):
        payload = transformed_by_component[component]
        freq = np.asarray(payload["freq"], dtype=float)
        transformed = np.asarray(payload["transformed"], dtype=float)
        x_cols = 64
        image_2d = np.tile(transformed[:, None], (1, x_cols))
        x_edges = np.linspace(0.0, 1.0, x_cols + 1)
        fallback_step = float(np.median(np.diff(freq))) if freq.size > 1 else 1.0
        y_edges = centers_to_edges(freq, fallback_step=fallback_step)

        pcm = axes[idx].pcolormesh(
            x_edges,
            y_edges,
            image_2d,
            shading="flat",
            cmap=colormap_name(cmap_index),
        )
        axes[idx].set_title(component.upper())
        axes[idx].set_xlabel("Arbitrary X")
        axes[idx].set_xlim(0.0, 1.0)
        axes[idx].set_xticks([])
        axes[idx].text(
            0.015,
            0.985,
            f"max {np.nanmax(transformed):.2f}\nmin {np.nanmin(transformed):.2f}",
            transform=axes[idx].transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.45,
            },
        )
        _apply_compact_image_axis_style(
            axes[idx],
            show_right_ylabel=idx == len(components) - 1,
            y_label=str(payload["label"]),
        )

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def _compute_contributions(args, records):
    return (
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
        transformed_by_component: OrderedDict[str, dict[str, np.ndarray | str]] = OrderedDict()
        available_display_bonds = None
        selected_display_bonds = None
        n_datasets = 0
        accepted_display_bonds: list[int] = []

        if args.compare_xya:
            raw_config = load_dataset_selection_entries(args.config_json)
            overlap = _validate_compare_xya_inputs(
                raw_config,
                track_data_root=args.track_data_root,
            )
            if args.only_component is not None:
                if args.only_component not in overlap:
                    raise ValueError(
                        f"Requested --only-component {args.only_component!r} is not in the shared component overlap {overlap}"
                    )
                overlap = [args.only_component]

            for logical_component in overlap:
                config = _resolved_component_config(raw_config, logical_component=logical_component)
                records = build_configured_bond_signals(
                    config,
                    track_data_root=args.track_data_root,
                    allow_duplicate_ids=args.allow_duplicate_bonds,
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

                contributions = _compute_contributions(args, records)
                if len(contributions) == 0:
                    raise ValueError(
                        f"No spectra were accepted from the selected bond contributors for component '{logical_component}'"
                    )

                result = compute_average_spectrum(
                    contributions,
                    normalize_mode=args.normalize,
                    relative_range=tuple(args.relative_range),
                    average_domain=args.average_domain,
                    lowest_freq=args.freq_min_hz,
                    highest_freq=args.freq_max_hz,
                )
                baseline = _compute_baseline(result.avg_amp, args.baseline_window_bins)
                transformed, y_label = _transform_against_baseline(
                    result.avg_amp,
                    baseline,
                    args.prominence_mode,
                )
                transformed_by_component[logical_component] = {
                    "freq": result.freq_grid,
                    "raw": result.avg_amp,
                    "baseline": baseline,
                    "transformed": transformed,
                    "label": y_label,
                }

                available_display_bonds = component_available if available_display_bonds is None else available_display_bonds
                selected_display_bonds = component_selected if selected_display_bonds is None else selected_display_bonds
                accepted_display_bonds = sorted({
                    int(contrib.record.entity_id) + 1
                    for contrib in result.contributors
                })
                n_datasets = max(
                    n_datasets,
                    len({contrib.record.dataset_name for contrib in result.contributors}),
                )
        else:
            if args.only_component is not None:
                raw_config = load_dataset_selection_entries(args.config_json)
                config = _legacy_config_with_forced_component(
                    raw_config,
                    logical_component=args.only_component,
                )
            else:
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

            contributions = _compute_contributions(args, records)
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
            baseline = _compute_baseline(result.avg_amp, args.baseline_window_bins)
            transformed, y_label = _transform_against_baseline(
                result.avg_amp,
                baseline,
                args.prominence_mode,
            )
            transformed_by_component["x"] = {
                "freq": result.freq_grid,
                "raw": result.avg_amp,
                "baseline": baseline,
                "transformed": transformed,
                "label": y_label,
            }
            accepted_display_bonds = sorted({int(contrib.record.entity_id) + 1 for contrib in result.contributors})
            n_datasets = len({contrib.record.dataset_name for contrib in result.contributors})

        title = args.title or (
            f"Average {'Welch' if args.welch else 'FFT'} Prominence | "
            f"mode={args.prominence_mode} | baseline={args.baseline_window_bins} bins | "
            f"components={','.join(transformed_by_component.keys())} | datasets={n_datasets} | "
            f"bonds={len(accepted_display_bonds)}"
        )

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds or [])}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds or [])}")
        print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
        print(f"Compared logical components: {list(transformed_by_component.keys())}")
        print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
        if args.welch:
            print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
        print("Normalization band processing: linear detrend -> zero-floor -> integrate area")
        print(f"Near-zero denominator threshold: {ABSOLUTE_ZERO_TOL:.0e}")
        print(f"Prominence mode: {args.prominence_mode}")
        print(f"Baseline window: {args.baseline_window_bins} bins")
        print(f"Display mode: {'full image' if args.full_image else 'curve'}")

        fig = (
            _plot_component_images(
                transformed_by_component,
                cmap_index=args.cm,
                title=title,
            )
            if args.full_image
            else _plot_component_curves(
                transformed_by_component,
                plot_scale=args.plot_scale,
                title=title,
                show_peaks=args.show_peaks,
                peak_min_distance_bins=args.peak_min_distance_bins,
                peak_min_prominence=args.peak_min_prominence,
            )
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
