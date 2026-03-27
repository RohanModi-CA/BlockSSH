#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import find_peaks, peak_prominences

from plotting.common import render_figure
from plotting.frequency import _plot_frequency_image
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
from tools.spectral import ABSOLUTE_ZERO_TOL, integral_over_window
from tools.selection import load_dataset_selection_entries
from viz.cleantest import CANONICAL_COMPONENTS, _average_component_result, _validate_component_inputs


OWNER_COLORS = {
    "x": "orange",
    "y": "tab:blue",
    "a": "lightgrey",
    "mixed": "tab:red",
}


@dataclass(frozen=True)
class PeakOwner:
    peak_index: int
    freq_hz: float
    low_hz: float
    high_hz: float
    total_power_window: float
    component_power_window: dict[str, float]
    component_fraction: dict[str, float]
    owner: str
    prominence: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign each X/Y/A spectral peak to the component contributing the largest "
            "fraction of total peak-window spectral power."
        )
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
    parser.add_argument("--welch", action="store_true", help="Use Welch spectra instead of FFT spectra.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--allow-duplicate-bonds", action="store_true")
    parser.add_argument(
        "--peak-window-hz",
        type=float,
        default=0.1,
        help="Half-width in Hz used to integrate power around each detected peak. Default: 0.1",
    )
    parser.add_argument(
        "--peak-min-prom-ratio",
        type=float,
        default=0.1,
        help="Keep only peaks with prominence at least this fraction of the dominant total-power prominence. Default: 0.1",
    )
    parser.add_argument(
        "--peak-min-distance-hz",
        type=float,
        default=0.15,
        help="Minimum spacing in Hz between detected total-power peaks. Default: 0.15",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=None,
        help="Optionally keep only the strongest N peaks by total-power prominence.",
    )
    parser.add_argument(
        "--mixed-threshold",
        type=float,
        default=None,
        help="If set, label a peak as mixed when its winning fraction is below this threshold.",
    )
    return parser


def _format_bond_list(display_bonds: list[int] | None) -> str:
    if not display_bonds:
        return "[]"
    if len(display_bonds) <= 12:
        return "[" + ", ".join(str(v) for v in display_bonds) + "]"
    head = ", ".join(str(v) for v in display_bonds[:10])
    return f"[{head}, ...] ({len(display_bonds)} total)"


def _positive_median_step(freq: np.ndarray) -> float:
    df = np.diff(np.asarray(freq, dtype=float))
    df = df[np.isfinite(df) & (df > 0)]
    if df.size == 0:
        raise ValueError("Could not determine a positive frequency spacing")
    return float(np.median(df))


def _compute_peak_owners(
    *,
    freq: np.ndarray,
    amp_by_component: dict[str, np.ndarray],
    peak_window_hz: float,
    peak_min_prom_ratio: float,
    peak_min_distance_hz: float,
    max_peaks: int | None,
    mixed_threshold: float | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[PeakOwner]]:
    power_by_component = {
        component: np.square(np.asarray(amp, dtype=float))
        for component, amp in amp_by_component.items()
    }
    total_power = np.zeros_like(freq, dtype=float)
    for component in CANONICAL_COMPONENTS:
        total_power = total_power + power_by_component[component]

    df = _positive_median_step(freq)
    distance_bins = max(1, int(round(peak_min_distance_hz / df)))
    peak_indices, _ = find_peaks(total_power, distance=distance_bins)
    if peak_indices.size == 0:
        return total_power, power_by_component, []

    prominences = peak_prominences(total_power, peak_indices)[0]
    if prominences.size == 0:
        return total_power, power_by_component, []

    max_prominence = float(np.max(prominences))
    prom_floor = float(peak_min_prom_ratio) * max_prominence
    keep_mask = prominences >= prom_floor
    peak_indices = peak_indices[keep_mask]
    prominences = prominences[keep_mask]
    if peak_indices.size == 0:
        return total_power, power_by_component, []

    order = np.argsort(prominences)[::-1]
    if max_peaks is not None:
        if max_peaks < 1:
            raise ValueError("--max-peaks must be at least 1 when provided")
        order = order[:max_peaks]
    selected = sorted(
        (
            (
                int(peak_indices[idx]),
                float(prominences[idx]),
            )
            for idx in order
        ),
        key=lambda item: item[0],
    )

    owners: list[PeakOwner] = []
    for peak_order, (peak_idx, prominence) in enumerate(selected):
        peak_hz = float(freq[peak_idx])
        low_hz = max(float(freq[0]), peak_hz - float(peak_window_hz))
        high_hz = min(float(freq[-1]), peak_hz + float(peak_window_hz))
        if high_hz <= low_hz:
            continue

        component_power_window = {
            component: max(
                0.0,
                integral_over_window(freq, power_by_component[component], low_hz, high_hz),
            )
            for component in CANONICAL_COMPONENTS
        }
        total_power_window = float(sum(component_power_window.values()))
        if total_power_window <= ABSOLUTE_ZERO_TOL:
            continue

        component_fraction = {
            component: float(component_power_window[component] / total_power_window)
            for component in CANONICAL_COMPONENTS
        }
        owner = max(component_fraction.items(), key=lambda item: item[1])[0]
        if mixed_threshold is not None and component_fraction[owner] < float(mixed_threshold):
            owner = "mixed"

        owners.append(
            PeakOwner(
                peak_index=int(peak_order),
                freq_hz=peak_hz,
                low_hz=float(low_hz),
                high_hz=float(high_hz),
                total_power_window=total_power_window,
                component_power_window=component_power_window,
                component_fraction=component_fraction,
                owner=owner,
                prominence=float(prominence),
            )
        )

    return total_power, power_by_component, owners


def _plot_peak_owner_images(
    *,
    freq: np.ndarray,
    amp_by_component: dict[str, np.ndarray],
    owners: list[PeakOwner],
    plot_scale: str,
    cmap_index: int,
    freq_min_hz: float,
    freq_max_hz: float,
    title: str,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 7),
        constrained_layout=True,
        sharey=True,
    )

    if len(owners) > 0:
        x_positions = np.arange(len(owners), dtype=float) + 0.5
        x_max = float(len(owners))
    else:
        x_positions = np.array([], dtype=float)
        x_max = 1.0

    for ax, component in zip(axes, CANONICAL_COMPONENTS, strict=True):
        _plot_frequency_image(
            fig,
            ax,
            freq=freq,
            amp=amp_by_component[component],
            plot_scale=plot_scale,
            cmap_index=cmap_index,
            y_min=freq_min_hz,
            y_max=freq_max_hz,
            x_label="Peak Index",
            x_max=x_max,
            title=f"{component.upper()} Spectrum",
            linear_color_label="Amplitude",
            log_color_label="Amplitude (dB)",
            show_colorbar=True,
            annotate_range=False,
        )

        if len(owners) > 0:
            y_positions = np.array([owner.freq_hz for owner in owners], dtype=float)
            colors = [OWNER_COLORS[owner.owner] for owner in owners]
            ax.scatter(
                x_positions,
                y_positions,
                c=colors,
                s=36.0,
                edgecolors="black",
                linewidths=0.4,
                zorder=4,
            )
            ax.set_xlim(0.0, x_max)
            ax.set_xticks(np.arange(len(owners), dtype=float) + 0.5)
            ax.set_xticklabels([str(owner.peak_index) for owner in owners], rotation=90)
        else:
            ax.set_xticks([])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=OWNER_COLORS[name],
            markeredgecolor="black",
            markeredgewidth=0.4,
            color="none",
            label=name,
            markersize=7,
        )
        for name in ("x", "y", "a")
    ]
    if any(owner.owner == "mixed" for owner in owners):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=OWNER_COLORS["mixed"],
                markeredgecolor="black",
                markeredgewidth=0.4,
                color="none",
                label="mixed",
                markersize=7,
            )
        )
    axes[0].legend(handles=legend_handles, loc="upper right", title="Peak owner")

    fig.suptitle(title, fontsize=14)
    return fig


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

    if args.peak_window_hz <= 0:
        print("Error: --peak-window-hz must be > 0", file=sys.stderr)
        return 1

    if args.peak_min_prom_ratio < 0:
        print("Error: --peak-min-prom-ratio must be >= 0", file=sys.stderr)
        return 1

    if args.peak_min_distance_hz <= 0:
        print("Error: --peak-min-distance-hz must be > 0", file=sys.stderr)
        return 1

    if args.mixed_threshold is not None and not (0.0 <= args.mixed_threshold <= 1.0):
        print("Error: --mixed-threshold must lie in [0, 1]", file=sys.stderr)
        return 1

    try:
        raw_config = load_dataset_selection_entries(args.config_json)
        _validate_component_inputs(
            raw_config,
            track_data_root=args.track_data_root,
            required_components=tuple(CANONICAL_COMPONENTS),
        )

        component_results: dict[str, object] = {}
        available_display_bonds: list[int] | None = None
        selected_display_bonds: list[int] | None = None
        for component in CANONICAL_COMPONENTS:
            result, available_bonds, selected_bonds = _average_component_result(args, raw_config, component)
            component_results[component] = result
            if available_display_bonds is None:
                available_display_bonds = available_bonds
                selected_display_bonds = selected_bonds

        reference_result = component_results["x"]
        freq = np.asarray(reference_result.freq_grid, dtype=float)
        for component in CANONICAL_COMPONENTS[1:]:
            result = component_results[component]
            if not np.allclose(freq, result.freq_grid, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Average spectra for 'x' and '{component}' do not share the same frequency grid"
                )

        amp_by_component = {
            component: np.asarray(component_results[component].avg_amp, dtype=float)
            for component in CANONICAL_COMPONENTS
        }
        total_power, _, owners = _compute_peak_owners(
            freq=freq,
            amp_by_component=amp_by_component,
            peak_window_hz=float(args.peak_window_hz),
            peak_min_prom_ratio=float(args.peak_min_prom_ratio),
            peak_min_distance_hz=float(args.peak_min_distance_hz),
            max_peaks=args.max_peaks,
            mixed_threshold=args.mixed_threshold,
        )

        accepted_display_bonds = sorted({
            int(contrib.record.entity_id) + 1
            for contrib in reference_result.contributors
        })
        n_datasets = len({contrib.record.dataset_name for contrib in reference_result.contributors})
        norm_desc = args.normalize
        if args.normalize == "relative":
            norm_desc = f"relative [{rel_low}, {rel_high}] Hz"

        title = args.title or (
            f"Peak Owner XYA | {'Welch' if args.welch else 'FFT'} | "
            f"datasets={n_datasets} | bonds={len(accepted_display_bonds)} | "
            f"peaks={len(owners)} | norm={norm_desc}"
        )

        freq_min_hz = float(reference_result.freq_low if args.freq_min_hz is None else args.freq_min_hz)
        freq_max_hz = float(reference_result.freq_high if args.freq_max_hz is None else args.freq_max_hz)

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds)}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds)}")
        print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
        print(f"Unique datasets: {n_datasets}")
        print(f"Frequency window: [{reference_result.freq_low:.6f}, {reference_result.freq_high:.6f}] Hz")
        print(f"Normalization window: [{reference_result.norm_low:.6f}, {reference_result.norm_high:.6f}] Hz")
        print("Normalization band processing: linear detrend -> zero-floor -> integrate area")
        print(f"Near-zero denominator threshold: {ABSOLUTE_ZERO_TOL:.0e}")
        print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
        if args.welch:
            print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
        print(f"Peak-window half-width: {args.peak_window_hz:.6g} Hz")
        print(f"Peak minimum prominence ratio: {args.peak_min_prom_ratio:.6g}")
        print(f"Peak minimum distance: {args.peak_min_distance_hz:.6g} Hz")
        if args.max_peaks is not None:
            print(f"Max peaks retained: {args.max_peaks}")
        if args.mixed_threshold is not None:
            print(f"Mixed threshold: {args.mixed_threshold:.6g}")
        print(f"Detected total-power peaks: {len(owners)}")
        print("Peak owners:")
        print("idx,freq_hz,owner,pct_x,pct_y,pct_a,window_low_hz,window_high_hz,total_power_window,prominence")
        for owner in owners:
            print(
                f"{owner.peak_index},"
                f"{owner.freq_hz:.6f},"
                f"{owner.owner},"
                f"{100.0 * owner.component_fraction['x']:.3f},"
                f"{100.0 * owner.component_fraction['y']:.3f},"
                f"{100.0 * owner.component_fraction['a']:.3f},"
                f"{owner.low_hz:.6f},"
                f"{owner.high_hz:.6f},"
                f"{owner.total_power_window:.12g},"
                f"{owner.prominence:.12g}"
            )

        if len(owners) == 0:
            print(
                "Warning: no peaks passed the current total-power detection thresholds.",
                file=sys.stderr,
            )

        fig = _plot_peak_owner_images(
            freq=freq,
            amp_by_component=amp_by_component,
            owners=owners,
            plot_scale=args.plot_scale,
            cmap_index=args.cm,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            title=title,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
