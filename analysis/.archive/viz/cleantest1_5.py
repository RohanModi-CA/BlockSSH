#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from plotting.common import ensure_parent_dir
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
from tools.selection import load_dataset_selection_entries
from viz.cleantest import (
    CANONICAL_COMPONENTS,
    DEFAULT_CLEAN_STEPS,
    _average_component_result,
    _compute_local_baseline,
    _display_state_amp,
    _format_bond_list,
    _format_state_label,
    _parse_bool_arg,
    _parse_clean_step,
    _plot_clean_step_curves,
    _plot_clean_step_images,
    _resolved_clean_steps,
    _source_component,
    _validate_component_inputs,
)


@dataclass(frozen=True)
class SourceWindows:
    windows: list[tuple[float, float]]
    centers_hz: list[float]
    reference_prominence: float
    peak_count: int


@dataclass(frozen=True)
class FitMetrics:
    alpha: float
    loss: float
    peak_count: int
    mean_width_hz: float
    mean_prominence: float
    negative_penalty: float
    outside_penalty: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply ordered scalar cleaning steps with fitted scale factors based on peak-shape loss.",
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
    parser.add_argument("--welch", action="store_true", default=True, help="Use Welch spectra instead of FFT.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--allow-duplicate-bonds", action="store_true")
    parser.add_argument("--power", action="store_true", help="Fit and subtract in power space.")
    parser.add_argument("--full-image", action="store_true", help="Show the enabled spectra as side-by-side frequency images.")
    parser.add_argument(
        "--one-fig",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Keep all cleaning rows in one figure. Pass false to open one figure per row. Default: true",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows. Useful with --save.",
    )
    parser.add_argument(
        "--clean",
        dest="clean_steps",
        action="append",
        type=_parse_clean_step,
        default=None,
        metavar="TARGET<-REMOVE",
        help=(
            "Ordered cleaning step. May be repeated. Defaults to: "
            + ", ".join(DEFAULT_CLEAN_STEPS)
        ),
    )
    parser.add_argument(
        "--display-floor",
        choices=["local-baseline", "global-baseline", "none", "zero"],
        default="local-baseline",
        help="How to floor the cleaned spectrum for display only. Default: local-baseline",
    )
    parser.add_argument(
        "--baseline-window-bins",
        type=int,
        default=101,
        help="Odd median-filter window in bins for display baselines. Default: 101",
    )
    parser.add_argument(
        "--fit-baseline-window-bins",
        type=int,
        default=101,
        help="Odd median-filter window in bins for the fit loss baseline. Default: 101",
    )
    parser.add_argument(
        "--fit-min-alpha",
        type=float,
        default=0.0,
        help="Minimum subtraction scale considered during the fit. Default: 0.0",
    )
    parser.add_argument(
        "--fit-max-alpha",
        type=float,
        default=None,
        help="Maximum subtraction scale considered during the fit. Default: auto-estimated from source windows.",
    )
    parser.add_argument(
        "--fit-auto-max-factor",
        type=float,
        default=2.0,
        help="Multiplier used when auto-estimating the max alpha from target/source ratios. Default: 2.0",
    )
    parser.add_argument(
        "--fit-alpha-count",
        type=int,
        default=121,
        help="Number of alpha values used in the grid search. Default: 121",
    )
    parser.add_argument(
        "--fit-max-peaks",
        type=int,
        default=8,
        help="Max number of source peaks used to define fit windows. Default: 8",
    )
    parser.add_argument(
        "--fit-min-prom-ratio",
        type=float,
        default=0.15,
        help="Keep source peaks with prominence at least this fraction of the dominant source prominence. Default: 0.15",
    )
    parser.add_argument(
        "--fit-window-scale",
        type=float,
        default=1.5,
        help="Expand each detected source-peak width by this factor when forming fit windows. Default: 1.5",
    )
    parser.add_argument(
        "--loss-top-k-peaks",
        type=int,
        default=4,
        help="Use the strongest K surviving peaks inside the source windows when scoring widths and prominence. Default: 4",
    )
    parser.add_argument(
        "--loss-prom-floor-ratio",
        type=float,
        default=0.1,
        help="Count cleaned peaks only if their prominence exceeds this fraction of the source reference prominence. Default: 0.1",
    )
    parser.add_argument(
        "--loss-peak-count-weight",
        type=float,
        default=1.0,
        help="Weight for the number of surviving peaks inside source windows. Default: 1.0",
    )
    parser.add_argument(
        "--loss-width-weight",
        type=float,
        default=0.75,
        help="Weight for the mean surviving peak width in Hz. Default: 0.75",
    )
    parser.add_argument(
        "--loss-prominence-weight",
        type=float,
        default=1.5,
        help="Reward weight for surviving peak prominence above baseline. Default: 1.5",
    )
    parser.add_argument(
        "--loss-negative-weight",
        type=float,
        default=4.0,
        help="Weight for oversubtraction below zero. Default: 4.0",
    )
    parser.add_argument(
        "--loss-outside-weight",
        type=float,
        default=1.0,
        help="Weight for distortion outside the source windows. Default: 1.0",
    )
    parser.add_argument(
        "--show-x",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the original X spectrum. Default: true",
    )
    parser.add_argument(
        "--show-a",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the original A spectrum. Default: true",
    )
    parser.add_argument(
        "--show-y",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the original Y spectrum. Default: true",
    )
    parser.add_argument(
        "--show-cleaned-x",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the final cleaned X spectrum. Default: true",
    )
    parser.add_argument(
        "--show-cleaned-y",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the final cleaned Y spectrum. Default: false",
    )
    return parser


def _render_figure(fig, *, save: str | None, show: bool) -> None:
    if save is not None:
        save_path = ensure_parent_dir(save)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not windows:
        return []
    windows_sorted = sorted((float(lo), float(hi)) for lo, hi in windows if hi > lo)
    merged = [windows_sorted[0]]
    for low, high in windows_sorted[1:]:
        last_low, last_high = merged[-1]
        if low <= last_high:
            merged[-1] = (last_low, max(last_high, high))
        else:
            merged.append((low, high))
    return merged


def _windows_mask(freq: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(freq.shape, dtype=bool)
    for low, high in windows:
        mask |= (freq >= low) & (freq <= high)
    return mask


def _indices_to_hz(freq: np.ndarray, positions: np.ndarray) -> np.ndarray:
    idx = np.arange(freq.size, dtype=float)
    return np.interp(positions, idx, freq)


def _detect_source_windows(freq: np.ndarray, source_spec: np.ndarray, args) -> SourceWindows:
    baseline = _compute_local_baseline(source_spec, args.fit_baseline_window_bins)
    excess = np.maximum(source_spec - baseline, 0.0)
    peaks, _ = find_peaks(excess)

    if peaks.size == 0:
        fallback_idx = int(np.argmax(excess))
        peak_hz = float(freq[fallback_idx])
        width_hz = max(float(np.median(np.diff(freq))) * 4.0, 1e-6)
        return SourceWindows(
            windows=[(max(float(freq[0]), peak_hz - width_hz), min(float(freq[-1]), peak_hz + width_hz))],
            centers_hz=[peak_hz],
            reference_prominence=max(float(np.max(excess)), 1e-12),
            peak_count=1,
        )

    prominences = peak_prominences(excess, peaks)[0]
    if prominences.size == 0 or np.max(prominences) <= 0:
        fallback_idx = int(peaks[np.argmax(excess[peaks])])
        peak_hz = float(freq[fallback_idx])
        width_hz = max(float(np.median(np.diff(freq))) * 4.0, 1e-6)
        return SourceWindows(
            windows=[(max(float(freq[0]), peak_hz - width_hz), min(float(freq[-1]), peak_hz + width_hz))],
            centers_hz=[peak_hz],
            reference_prominence=max(float(np.max(excess)), 1e-12),
            peak_count=1,
        )

    prom_ref = float(np.max(prominences))
    keep = prominences >= float(args.fit_min_prom_ratio) * prom_ref
    selected = np.where(keep)[0]
    if selected.size == 0:
        selected = np.array([int(np.argmax(prominences))], dtype=int)
    selected = selected[np.argsort(prominences[selected])[::-1]]
    selected = selected[: max(1, int(args.fit_max_peaks))]
    peak_bins = peaks[selected]

    widths, _, left_ips, right_ips = peak_widths(excess, peak_bins, rel_height=0.5)
    centers_hz = [float(freq[idx]) for idx in peak_bins]
    windows: list[tuple[float, float]] = []
    for peak_idx, width_bins, left_ip, right_ip in zip(peak_bins, widths, left_ips, right_ips, strict=True):
        center_hz = float(freq[peak_idx])
        half_width_hz = 0.5 * abs(float(_indices_to_hz(freq, np.array([right_ip]))[0] - _indices_to_hz(freq, np.array([left_ip]))[0]))
        if not np.isfinite(half_width_hz) or half_width_hz <= 0:
            half_width_hz = max(float(np.median(np.diff(freq))) * 2.0, 1e-6)
        half_width_hz *= float(args.fit_window_scale)
        low = max(float(freq[0]), center_hz - half_width_hz)
        high = min(float(freq[-1]), center_hz + half_width_hz)
        windows.append((low, high))

    return SourceWindows(
        windows=_merge_windows(windows),
        centers_hz=centers_hz,
        reference_prominence=max(prom_ref, 1e-12),
        peak_count=int(len(peak_bins)),
    )


def _estimate_alpha_max(target_spec: np.ndarray, remove_spec: np.ndarray, mask: np.ndarray, args) -> float:
    if args.fit_max_alpha is not None:
        return float(args.fit_max_alpha)

    eps = np.finfo(float).eps
    valid = mask & np.isfinite(target_spec) & np.isfinite(remove_spec) & (remove_spec > eps)
    ratios = target_spec[valid] / remove_spec[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0.0)]
    if ratios.size == 0:
        return max(float(args.fit_min_alpha) + 0.5, 2.0)

    q95 = float(np.percentile(ratios, 95.0))
    q50 = float(np.percentile(ratios, 50.0))
    auto = max(q95, q50, 1e-6) * float(args.fit_auto_max_factor)
    return max(float(args.fit_min_alpha) + 1e-6, auto)


def _evaluate_alpha(
    freq: np.ndarray,
    target_spec: np.ndarray,
    remove_spec: np.ndarray,
    windows: SourceWindows,
    alpha: float,
    args,
) -> FitMetrics:
    eps = np.finfo(float).eps
    cleaned_spec = target_spec - float(alpha) * remove_spec
    fit_signal = np.maximum(cleaned_spec, 0.0)
    baseline = _compute_local_baseline(fit_signal, args.fit_baseline_window_bins)
    excess = np.maximum(fit_signal - baseline, 0.0)

    inside_mask = _windows_mask(freq, windows.windows)
    outside_mask = ~inside_mask

    peaks, _ = find_peaks(excess)
    peak_count = 0
    mean_width_hz = 0.0
    mean_prominence = 0.0

    if peaks.size > 0:
        prominences = peak_prominences(excess, peaks)[0]
        widths_bins = peak_widths(excess, peaks, rel_height=0.5)[0]
        widths_hz = widths_bins * max(float(np.median(np.diff(freq))), eps)

        strong_mask = inside_mask[peaks] & (
            prominences >= float(args.loss_prom_floor_ratio) * float(windows.reference_prominence)
        )
        strong_idx = np.where(strong_mask)[0]
        peak_count = int(strong_idx.size)
        if strong_idx.size > 0:
            order = strong_idx[np.argsort(prominences[strong_idx])[::-1]]
            order = order[: max(1, int(args.loss_top_k_peaks))]
            mean_width_hz = float(np.mean(widths_hz[order]))
            mean_prominence = float(np.mean(prominences[order]))

    inside_scale = float(np.mean(target_spec[inside_mask] ** 2)) if np.any(inside_mask) else 1.0
    if not np.isfinite(inside_scale) or inside_scale <= 0:
        inside_scale = 1.0
    negative_penalty = float(np.mean(np.maximum(-cleaned_spec[inside_mask], 0.0) ** 2) / inside_scale) if np.any(inside_mask) else 0.0

    if np.any(outside_mask):
        outside_resid = cleaned_spec[outside_mask] - target_spec[outside_mask]
        outside_scale = target_spec[outside_mask] ** 2 + eps
        outside_penalty = float(np.mean((outside_resid**2) / outside_scale))
    else:
        outside_penalty = 0.0

    loss = (
        float(args.loss_peak_count_weight) * peak_count
        + float(args.loss_width_weight) * mean_width_hz
        - float(args.loss_prominence_weight) * (mean_prominence / max(float(windows.reference_prominence), eps))
        + float(args.loss_negative_weight) * negative_penalty
        + float(args.loss_outside_weight) * outside_penalty
    )

    return FitMetrics(
        alpha=float(alpha),
        loss=float(loss),
        peak_count=int(peak_count),
        mean_width_hz=float(mean_width_hz),
        mean_prominence=float(mean_prominence),
        negative_penalty=float(negative_penalty),
        outside_penalty=float(outside_penalty),
    )


def _run_fit_clean_pipeline(
    *,
    freq: np.ndarray,
    raw_amp_by_component: dict[str, np.ndarray],
    clean_steps: list[tuple[str, str]],
    power_mode: bool,
    args,
):
    current_amp_by_component = {
        component: np.asarray(amp, dtype=float).copy()
        for component, amp in raw_amp_by_component.items()
    }
    derived_amp_by_name: dict[str, np.ndarray] = {}
    summaries: list[dict[str, object]] = []
    step_payloads: list[dict[str, object]] = []
    removed_components_by_component = {
        component: []
        for component in raw_amp_by_component
    }

    for step_index, (target, remove_name) in enumerate(clean_steps, start=1):
        if target not in current_amp_by_component:
            raise ValueError(f"Clean step {step_index} targets unavailable component '{target}'")

        if remove_name.startswith("clean_"):
            if remove_name not in derived_amp_by_name:
                raise ValueError(
                    f"Clean step {step_index} requires '{remove_name}', but it has not been produced yet"
                )
            remove_amp = derived_amp_by_name[remove_name]
            source_removed_components = list(removed_components_by_component[_source_component(remove_name)])
        else:
            if remove_name not in raw_amp_by_component:
                raise ValueError(f"Clean step {step_index} removes unavailable component '{remove_name}'")
            remove_amp = raw_amp_by_component[remove_name]
            source_removed_components = []

        target_amp = np.asarray(current_amp_by_component[target], dtype=float)
        remove_amp = np.asarray(remove_amp, dtype=float)

        if power_mode:
            target_spec = np.maximum(target_amp, 0.0) ** 2
            remove_spec = np.maximum(remove_amp, 0.0) ** 2
        else:
            target_spec = target_amp
            remove_spec = remove_amp

        source_windows = _detect_source_windows(freq, np.maximum(remove_spec, 0.0), args)
        inside_mask = _windows_mask(freq, source_windows.windows)
        alpha_max = _estimate_alpha_max(target_spec, remove_spec, inside_mask, args)
        if alpha_max <= float(args.fit_min_alpha):
            alpha_max = float(args.fit_min_alpha) + 1e-6
        alphas = np.linspace(float(args.fit_min_alpha), alpha_max, num=max(3, int(args.fit_alpha_count)))

        metrics = [
            _evaluate_alpha(
                freq,
                target_spec,
                remove_spec,
                source_windows,
                alpha=float(alpha),
                args=args,
            )
            for alpha in alphas
        ]
        best = min(metrics, key=lambda item: item.loss)
        cleaned_target_spec = target_spec - best.alpha * remove_spec
        if power_mode:
            cleaned_target_amp = np.sqrt(np.maximum(cleaned_target_spec, 0.0))
        else:
            cleaned_target_amp = cleaned_target_spec

        target_removed_components_before = list(removed_components_by_component[target])
        target_removed_components_after = list(target_removed_components_before)
        source_component = _source_component(remove_name)
        if source_component not in target_removed_components_after:
            target_removed_components_after.append(source_component)

        current_amp_by_component[target] = cleaned_target_amp
        derived_amp_by_name[f"clean_{target}"] = cleaned_target_amp.copy()
        removed_components_by_component[target] = target_removed_components_after

        marker_hz = float(source_windows.centers_hz[0]) if source_windows.centers_hz else float(freq[int(np.argmax(remove_spec))])
        summaries.append(
            {
                "step_index": int(step_index),
                "target": target,
                "remove": remove_name,
                "alpha": float(best.alpha),
                "alpha_min": float(alphas[0]),
                "alpha_max": float(alphas[-1]),
                "loss": float(best.loss),
                "peak_count": int(best.peak_count),
                "mean_width_hz": float(best.mean_width_hz),
                "mean_prominence": float(best.mean_prominence),
                "negative_penalty": float(best.negative_penalty),
                "outside_penalty": float(best.outside_penalty),
                "source_peak_count": int(source_windows.peak_count),
                "source_windows": list(source_windows.windows),
                "marker_hz": float(marker_hz),
            }
        )
        step_payloads.append(
            {
                "step_index": int(step_index),
                "target": target,
                "remove": remove_name,
                "selected_hz": float(marker_hz),
                "source_label": _format_state_label(_source_component(remove_name), source_removed_components),
                "before_label": _format_state_label(target, target_removed_components_before),
                "after_label": _format_state_label(target, target_removed_components_after),
                "source_amp": _display_state_amp(
                    component=_source_component(remove_name),
                    amp=remove_amp,
                    removed_components=source_removed_components,
                    raw_amp_by_component=raw_amp_by_component,
                    args=args,
                ),
                "before_amp": _display_state_amp(
                    component=target,
                    amp=target_amp,
                    removed_components=target_removed_components_before,
                    raw_amp_by_component=raw_amp_by_component,
                    args=args,
                ),
                "after_amp": _display_state_amp(
                    component=target,
                    amp=cleaned_target_amp,
                    removed_components=target_removed_components_after,
                    raw_amp_by_component=raw_amp_by_component,
                    args=args,
                ),
            }
        )

    return current_amp_by_component, derived_amp_by_name, summaries, step_payloads


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    if args.freq_min_hz is not None and args.freq_max_hz is not None and args.freq_max_hz <= args.freq_min_hz:
        print("Error: --freq-max-hz must be greater than --freq-min-hz", file=sys.stderr)
        return 1
    if args.fit_alpha_count < 3:
        print("Error: --fit-alpha-count must be at least 3", file=sys.stderr)
        return 1
    if args.fit_max_peaks < 1:
        print("Error: --fit-max-peaks must be at least 1", file=sys.stderr)
        return 1
    if args.loss_top_k_peaks < 1:
        print("Error: --loss-top-k-peaks must be at least 1", file=sys.stderr)
        return 1

    try:
        clean_steps = _resolved_clean_steps(args.clean_steps)

        raw_config = load_dataset_selection_entries(args.config_json)
        required_components = {target for target, _ in clean_steps} | {
            _source_component(remove_name)
            for _, remove_name in clean_steps
        }
        if args.show_x:
            required_components.add("x")
        if args.show_a:
            required_components.add("a")
        if args.show_y or args.show_cleaned_y:
            required_components.add("y")

        _validate_component_inputs(
            raw_config,
            track_data_root=args.track_data_root,
            required_components=tuple(sorted(required_components)),
        )

        component_results: dict[str, object] = {}
        available_display_bonds = None
        selected_display_bonds = None
        for component in sorted(required_components):
            result, available_bonds, selected_bonds = _average_component_result(args, raw_config, component)
            component_results[component] = result
            if available_display_bonds is None:
                available_display_bonds = available_bonds
                selected_display_bonds = selected_bonds

        reference_component = clean_steps[0][0]
        result_ref = component_results[reference_component]
        freq = result_ref.freq_grid
        for component, result in component_results.items():
            if not np.allclose(freq, result.freq_grid, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Average spectra for '{reference_component}' and '{component}' do not share the same frequency grid"
                )

        raw_amp_by_component = {
            component: np.asarray(component_results[component].avg_amp, dtype=float)
            for component in required_components
        }
        current_amp_by_component, _, clean_summaries, step_payloads = _run_fit_clean_pipeline(
            freq=freq,
            raw_amp_by_component=raw_amp_by_component,
            clean_steps=clean_steps,
            power_mode=bool(args.power),
            args=args,
        )
        total_steps = len(step_payloads)
        for payload in step_payloads:
            payload["step_total"] = int(total_steps)

        accepted_display_bonds = sorted({
            int(contrib.record.entity_id) + 1
            for contrib in result_ref.contributors
        })
        n_datasets = len({contrib.record.dataset_name for contrib in result_ref.contributors})
        norm_desc = args.normalize
        if args.normalize == "relative":
            norm_desc = f"relative [{float(args.relative_range[0])}, {float(args.relative_range[1])}] Hz"
        pipeline_desc = ", ".join(f"{target}<-{remove_name}" for target, remove_name in clean_steps)

        title = args.title or (
            f"Clean Test 1.5 | {'Welch' if args.welch else 'FFT'} | pipeline={pipeline_desc} | "
            f"mode={'power' if args.power else 'amplitude'} | datasets={n_datasets} | "
            f"bonds={len(accepted_display_bonds)} | norm={norm_desc}"
        )

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds)}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds)}")
        print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
        print(f"Unique datasets: {n_datasets}")
        print(f"Frequency window: [{result_ref.freq_low:.6f}, {result_ref.freq_high:.6f}] Hz")
        print(f"Normalization window: [{result_ref.norm_low:.6f}, {result_ref.norm_high:.6f}] Hz")
        print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
        if args.welch:
            print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
        print(f"Cleaning pipeline: {pipeline_desc}")
        print(
            "Fit loss weights: "
            f"peak_count={args.loss_peak_count_weight:.6g}, "
            f"width={args.loss_width_weight:.6g}, "
            f"prominence={args.loss_prominence_weight:.6g}, "
            f"negative={args.loss_negative_weight:.6g}, "
            f"outside={args.loss_outside_weight:.6g}"
        )
        for summary in clean_summaries:
            windows_desc = ", ".join(f"[{low:.3f}, {high:.3f}]" for low, high in summary["source_windows"])
            print(
                f"Step {summary['step_index']}: {summary['target']} <- {summary['remove']} | "
                f"alpha={summary['alpha']:.12g} searched=[{summary['alpha_min']:.6g}, {summary['alpha_max']:.6g}] | "
                f"loss={summary['loss']:.6g} | residual_peaks={summary['peak_count']} | "
                f"mean_width={summary['mean_width_hz']:.6g} Hz | mean_prom={summary['mean_prominence']:.6g} | "
                f"neg={summary['negative_penalty']:.6g} | outside={summary['outside_penalty']:.6g} | "
                f"source_peak_count={summary['source_peak_count']} | windows={windows_desc}"
            )

        if args.one_fig:
            fig = (
                _plot_clean_step_images(
                    freq,
                    step_payloads,
                    cmap_index=args.cm,
                    plot_scale=args.plot_scale,
                    title=title,
                )
                if args.full_image
                else _plot_clean_step_curves(
                    freq,
                    step_payloads,
                    plot_scale=args.plot_scale,
                    title=title,
                )
            )
            _render_figure(fig, save=args.save, show=not args.no_show)
        else:
            for payload in step_payloads:
                step_idx = int(payload["step_index"])
                step_title = f"{title} | step {step_idx}/{total_steps} | {payload['target']} <- {payload['remove']}"
                fig = (
                    _plot_clean_step_images(
                        freq,
                        [payload | {"step_total": 1}],
                        cmap_index=args.cm,
                        plot_scale=args.plot_scale,
                        title=step_title,
                    )
                    if args.full_image
                    else _plot_clean_step_curves(
                        freq,
                        [payload | {"step_total": 1}],
                        plot_scale=args.plot_scale,
                        title=step_title,
                    )
                )
                _render_figure(fig, save=args.save, show=not args.no_show)

        show_payloads: list[tuple[str, np.ndarray]] = []
        if args.show_x and "x" in raw_amp_by_component:
            show_payloads.append(("x", raw_amp_by_component["x"]))
        if args.show_y and "y" in raw_amp_by_component:
            show_payloads.append(("y", raw_amp_by_component["y"]))
        if args.show_a and "a" in raw_amp_by_component:
            show_payloads.append(("a", raw_amp_by_component["a"]))
        if args.show_cleaned_x and "x" in current_amp_by_component:
            show_payloads.append(("clean_x", current_amp_by_component["x"]))
        if args.show_cleaned_y and "y" in current_amp_by_component:
            show_payloads.append(("clean_y", current_amp_by_component["y"]))

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
