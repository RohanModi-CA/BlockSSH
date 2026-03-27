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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.common import ensure_parent_dir
from plotting.frequency import _apply_compact_image_axis_style, _plot_frequency_image
from tools.cli import (
    add_average_domain_args,
    add_bond_filter_args,
    add_normalization_args,
    add_output_args,
    add_plot_scale_args,
    resolve_normalization_mode,
    add_signal_processing_args,
    add_track_data_root_arg,
)
from tools.selection import (
    build_configured_bond_signals,
    collect_display_bond_numbers,
    filter_signal_records_by_display_bonds,
    load_dataset_selection_entries,
)
from tools.signal import normalize_processed_signal_rms, preprocess_signal
from tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from tools.spectral import average_spectra, normalize_spectrum, resolve_normalization_window
from viz.cleantest import CANONICAL_COMPONENTS, _resolved_component_config, _validate_component_inputs


@dataclass(frozen=True)
class SpectrumRow:
    key: tuple[str, int]
    freq: np.ndarray
    amp: np.ndarray


@dataclass(frozen=True)
class PairLeakageResult:
    key: tuple[str, int]
    freq: np.ndarray
    raw_amp_by_component: dict[str, np.ndarray]
    clean_amp_by_component: dict[str, np.ndarray]
    transfer_by_target: dict[str, dict[str, np.ndarray]]
    coherence_by_target: dict[str, np.ndarray]
    residual_power_by_target: dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate frequency-dependent multichannel leakage with Welch cross-spectra "
            "and subtract the linearly predictable part of each component."
        )
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    add_track_data_root_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_plot_scale_args(parser)
    add_signal_processing_args(parser)
    add_bond_filter_args(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--components",
        nargs="+",
        choices=CANONICAL_COMPONENTS,
        default=list(CANONICAL_COMPONENTS),
        help="Logical components to jointly de-mix. Default: x y a",
    )
    parser.add_argument("--freq-min-hz", type=float, default=None)
    parser.add_argument("--freq-max-hz", type=float, default=None)
    parser.add_argument(
        "--welch-len-s",
        type=float,
        default=8.0,
        help="Segment length in seconds for cross-spectral estimation. Default: 8.0",
    )
    parser.add_argument(
        "--welch-overlap",
        type=float,
        default=0.5,
        help="Overlap fraction for cross-spectral segments. Default: 0.5",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-4,
        help="Relative ridge regularization added to predictor cross spectra. Default: 1e-4",
    )
    parser.add_argument(
        "--coherence-floor",
        type=float,
        default=0.0,
        help="Zero the fitted leakage where multiple coherence falls below this threshold. Default: 0.0",
    )
    parser.add_argument(
        "--transfer-smooth-bins",
        type=int,
        default=9,
        help="Odd moving-average width in bins used to smooth the fitted complex transfer. Default: 9",
    )
    parser.add_argument(
        "--allow-duplicate-bonds",
        action="store_true",
        help=(
            "Allow multiple datasets to contribute the same global bond id. "
            "Recommended for multi-dataset de-mixing."
        ),
    )
    parser.add_argument(
        "--show-transfers",
        action="store_true",
        help="Add a second figure with fitted leakage magnitudes and multiple coherence.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Show the averaged raw and cleaned spectra as frequency images instead of curve plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows. Useful with --save or MPLBACKEND=Agg.",
    )
    add_spectrasave_arg(parser)
    return parser


def _resolved_component_records(
    raw_config: OrderedDict[str, dict],
    *,
    logical_component: str,
    args,
) -> tuple[dict[tuple[str, int], object], list[int], list[int]]:
    config = _resolved_component_config(raw_config, logical_component=logical_component)
    resolved_names = list(config.keys())
    raw_names = list(raw_config.keys())
    resolved_to_raw = {
        resolved_name: raw_name
        for raw_name, resolved_name in zip(raw_names, resolved_names, strict=True)
    }

    records = build_configured_bond_signals(
        config,
        track_data_root=args.track_data_root,
        allow_duplicate_ids=True if args.allow_duplicate_bonds else True,
    )
    available_display_bonds = collect_display_bond_numbers(records)
    records = filter_signal_records_by_display_bonds(
        records,
        only_bonds=args.only_bonds,
        exclude_bonds=args.exclude_bonds,
        parity=args.bond_parity,
    )
    selected_display_bonds = collect_display_bond_numbers(records)

    keyed_records: dict[tuple[str, int], object] = {}
    for record in records:
        key = (resolved_to_raw[record.dataset_name], int(record.entity_id))
        if key in keyed_records:
            raise ValueError(
                f"Duplicate component record for dataset '{key[0]}' bond {key[1] + 1} in logical component '{logical_component}'"
            )
        keyed_records[key] = record

    return keyed_records, available_display_bonds, selected_display_bonds


def _align_processed_signals_multi(processed_by_component: dict[str, object]) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    components = list(processed_by_component.keys())
    starts = [float(processed_by_component[c].t[0]) for c in components]
    stops = [float(processed_by_component[c].t[-1]) for c in components]
    dts = [float(processed_by_component[c].dt) for c in components]

    start = max(starts)
    stop = min(stops)
    dt = max(dts)
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("No overlapping time range across processed components")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid aligned sampling interval")

    t = np.arange(start, stop + 0.5 * dt, dt, dtype=float)
    t = t[t <= stop + 1e-12]
    if t.size < 16:
        raise ValueError("Aligned overlap has too few samples")

    stacked = []
    for component in components:
        proc = processed_by_component[component]
        y = np.interp(t, proc.t, proc.y)
        y = y - np.mean(y)
        stacked.append(y)

    return t, np.vstack(stacked), components, dt


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=float)
    return np.hanning(n).astype(float)


def _estimate_multichannel_cross_spectra(
    signals: np.ndarray,
    *,
    fs: float,
    seg_len_s: float,
    overlap_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_channels, n_samples = signals.shape
    if n_channels < 2:
        raise ValueError("Need at least two channels for leakage estimation")

    nperseg = max(16, int(round(seg_len_s * fs)))
    nperseg = min(nperseg, n_samples)
    if nperseg < 16:
        raise ValueError("Signal too short for cross-spectral estimation")

    noverlap = int(round(overlap_frac * nperseg))
    noverlap = max(0, min(noverlap, nperseg - 1))
    step = nperseg - noverlap

    nfft = _next_pow2(nperseg)
    window = _hann_window(nperseg)
    win_pow = float(np.sum(window**2))
    if win_pow <= 0:
        raise ValueError("Invalid window power")

    starts = list(range(0, n_samples - nperseg + 1, step))
    if not starts:
        starts = [0]

    accum = None
    count = 0
    scale = 1.0 / (fs * win_pow)

    for start in starts:
        segment = signals[:, start:start + nperseg]
        if segment.shape[1] != nperseg:
            continue
        seg = (segment - np.mean(segment, axis=1, keepdims=True)) * window[None, :]
        fft_seg = np.fft.rfft(seg, n=nfft, axis=1)
        cross = scale * fft_seg[:, None, :] * np.conj(fft_seg[None, :, :])
        cross = np.moveaxis(cross, -1, 0)
        if accum is None:
            accum = cross
        else:
            accum += cross
        count += 1

    if count <= 0 or accum is None:
        raise ValueError("Failed to accumulate any spectral segments")

    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freq, accum / count


def _smooth_complex_response(values: np.ndarray, window_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    if values.size == 0:
        return values.copy()

    kernel = max(1, int(window_bins))
    if kernel % 2 == 0:
        kernel += 1
    if kernel <= 1 or values.size < 3:
        return values.copy()
    kernel = min(kernel, values.size if values.size % 2 == 1 else max(1, values.size - 1))
    if kernel <= 1:
        return values.copy()

    weights = np.ones(kernel, dtype=float) / float(kernel)
    real = np.convolve(np.real(values), weights, mode="same")
    imag = np.convolve(np.imag(values), weights, mode="same")
    return real + 1j * imag


def _fit_target_leakage(
    cross_spectra: np.ndarray,
    *,
    target_idx: int,
    regularization: float,
    coherence_floor: float,
    smooth_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_freq, n_channels, _ = cross_spectra.shape
    predictor_idx = [idx for idx in range(n_channels) if idx != target_idx]
    eps = np.finfo(float).eps

    transfer = np.zeros((n_freq, len(predictor_idx)), dtype=complex)
    coherence = np.zeros(n_freq, dtype=float)
    residual_power = np.zeros(n_freq, dtype=float)

    for f_idx in range(n_freq):
        S = cross_spectra[f_idx]
        S_tt = complex(S[target_idx, target_idx])
        S_tu = np.asarray(S[target_idx, predictor_idx], dtype=complex)
        S_uu = np.asarray(S[np.ix_(predictor_idx, predictor_idx)], dtype=complex)

        reg_scale = max(float(np.trace(np.real(S_uu))) / max(len(predictor_idx), 1), eps)
        S_uu_reg = S_uu + (float(regularization) * reg_scale) * np.eye(len(predictor_idx), dtype=complex)

        try:
            H = S_tu @ np.linalg.inv(S_uu_reg)
        except np.linalg.LinAlgError:
            H = np.zeros(len(predictor_idx), dtype=complex)

        coh_num = np.real(H @ np.conj(S_tu))
        coh_den = max(float(np.real(S_tt)), eps)
        coh = float(np.clip(coh_num / coh_den, 0.0, 1.0))

        if coherence_floor > 0.0 and coh < coherence_floor:
            H = np.zeros(len(predictor_idx), dtype=complex)
            coh = 0.0

        pred_power = float(np.real(H @ np.conj(S_tu)))
        resid = max(0.0, float(np.real(S_tt) - pred_power))

        transfer[f_idx] = H
        coherence[f_idx] = coh
        residual_power[f_idx] = resid

    for pred_idx in range(transfer.shape[1]):
        transfer[:, pred_idx] = _smooth_complex_response(transfer[:, pred_idx], smooth_bins)

    for f_idx in range(n_freq):
        S = cross_spectra[f_idx]
        S_tt = float(np.real(S[target_idx, target_idx]))
        S_tu = np.asarray(S[target_idx, predictor_idx], dtype=complex)
        H = transfer[f_idx]
        pred_power = float(np.real(H @ np.conj(S_tu)))
        coherence[f_idx] = float(np.clip(pred_power / max(S_tt, eps), 0.0, 1.0)) if S_tt > 0 else 0.0
        if coherence_floor > 0.0 and coherence[f_idx] < coherence_floor:
            transfer[f_idx] = 0.0 + 0.0j
            pred_power = 0.0
            coherence[f_idx] = 0.0
        residual_power[f_idx] = max(0.0, S_tt - pred_power)

    return transfer, coherence, residual_power


def _process_one_pair(
    records_by_component: dict[str, object],
    *,
    args,
) -> PairLeakageResult:
    processed_by_component = {}
    for component, record in records_by_component.items():
        processed, err = preprocess_signal(
            record.t,
            record.y,
            longest=args.longest,
            handlenan=args.handlenan,
            min_samples=16,
        )
        if processed is None:
            raise ValueError(f"{component} preprocessing failed: {err}")
        if args.timeseriesnorm:
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                raise ValueError(f"{component} time-series RMS normalization failed: {norm_error}")
            processed = processed_norm
        processed_by_component[component] = processed

    _, signals, components, dt = _align_processed_signals_multi(processed_by_component)
    fs = 1.0 / dt
    freq, cross = _estimate_multichannel_cross_spectra(
        signals,
        fs=fs,
        seg_len_s=args.welch_len_s,
        overlap_frac=args.welch_overlap,
    )

    raw_amp_by_component: dict[str, np.ndarray] = {}
    clean_amp_by_component: dict[str, np.ndarray] = {}
    transfer_by_target: dict[str, dict[str, np.ndarray]] = {}
    coherence_by_target: dict[str, np.ndarray] = {}
    residual_power_by_target: dict[str, np.ndarray] = {}

    for target_idx, component in enumerate(components):
        transfer, coherence, residual_power = _fit_target_leakage(
            cross,
            target_idx=target_idx,
            regularization=args.regularization,
            coherence_floor=args.coherence_floor,
            smooth_bins=args.transfer_smooth_bins,
        )

        raw_power = np.maximum(np.real(cross[:, target_idx, target_idx]), 0.0)
        raw_amp_by_component[component] = np.sqrt(raw_power)
        clean_amp_by_component[component] = np.sqrt(np.maximum(residual_power, 0.0))
        coherence_by_target[component] = coherence
        residual_power_by_target[component] = residual_power

        predictors = [name for name in components if name != component]
        transfer_by_target[component] = {
            predictor: transfer[:, pred_idx]
            for pred_idx, predictor in enumerate(predictors)
        }

    sample_component = components[0]
    return PairLeakageResult(
        key=(records_by_component[sample_component].dataset_name, int(records_by_component[sample_component].entity_id)),
        freq=freq,
        raw_amp_by_component=raw_amp_by_component,
        clean_amp_by_component=clean_amp_by_component,
        transfer_by_target=transfer_by_target,
        coherence_by_target=coherence_by_target,
        residual_power_by_target=residual_power_by_target,
    )


def _common_frequency_window(rows: list[SpectrumRow], *, lowest_freq: float | None, highest_freq: float | None) -> tuple[float, float]:
    min_supported = max(float(row.freq[0]) for row in rows)
    max_supported = min(float(row.freq[-1]) for row in rows)
    freq_low = min_supported if lowest_freq is None else max(min_supported, float(lowest_freq))
    freq_high = max_supported if highest_freq is None else min(max_supported, float(highest_freq))
    if not np.isfinite(freq_low) or not np.isfinite(freq_high) or freq_high <= freq_low:
        raise ValueError("No overlapping frequency window across accepted spectra")
    return float(freq_low), float(freq_high)


def _build_common_grid(rows: list[SpectrumRow], *, freq_low: float, freq_high: float) -> np.ndarray:
    clipped = []
    for row in rows:
        mask = (row.freq >= freq_low) & (row.freq <= freq_high)
        freq = row.freq[mask]
        if freq.size >= 2:
            clipped.append(freq)
    if not clipped:
        raise ValueError("No usable spectral rows inside the selected frequency window")

    reference = max(clipped, key=lambda arr: arr.size)
    grid = reference[(reference >= freq_low) & (reference <= freq_high)]
    if grid.size < 2:
        grid = np.linspace(freq_low, freq_high, num=512, dtype=float)
    if grid[0] > freq_low:
        grid = np.insert(grid, 0, freq_low)
    else:
        grid[0] = freq_low
    if grid[-1] < freq_high:
        grid = np.append(grid, freq_high)
    else:
        grid[-1] = freq_high
    return np.asarray(grid, dtype=float)


def _average_rows(
    rows: list[SpectrumRow],
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
    average_domain: str,
    lowest_freq: float | None,
    highest_freq: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    if len(rows) == 0:
        raise ValueError("No spectra were available to average")

    freq_low, freq_high = _common_frequency_window(rows, lowest_freq=lowest_freq, highest_freq=highest_freq)
    freq_grid = _build_common_grid(rows, freq_low=freq_low, freq_high=freq_high)

    norm_low, norm_high = resolve_normalization_window(
        freq_low,
        freq_high,
        normalize_mode=normalize_mode,
        relative_range=relative_range,
    )

    normalized_rows = []
    accepted = 0
    for row in rows:
        interp = np.interp(freq_grid, row.freq, row.amp)
        normed = normalize_spectrum(freq_grid, interp, norm_low=norm_low, norm_high=norm_high)
        if normed is None:
            continue
        normalized_rows.append(normed)
        accepted += 1

    if not normalized_rows:
        raise ValueError("All spectra were rejected during normalization")

    avg = average_spectra(np.vstack(normalized_rows), average_domain)
    return freq_grid, avg, float(norm_low), float(norm_high), int(accepted)


def _plot_component_curves(
    freq: np.ndarray,
    raw_avg: dict[str, np.ndarray],
    clean_avg: dict[str, np.ndarray],
    *,
    plot_scale: str,
    title: str,
):
    components = list(raw_avg.keys())
    fig, axes = plt.subplots(
        2,
        len(components),
        figsize=(4.6 * len(components), 8.0),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )

    for col_idx, component in enumerate(components):
        raw_ax = axes[0, col_idx]
        clean_ax = axes[1, col_idx]

        raw = np.asarray(raw_avg[component], dtype=float)
        clean = np.asarray(clean_avg[component], dtype=float)

        if plot_scale == "log":
            raw_ax.semilogy(freq, np.where(raw > 0, raw, np.nan), linewidth=1.3, label=f"{component} raw")
            clean_ax.semilogy(freq, np.where(raw > 0, raw, np.nan), linewidth=0.9, alpha=0.45, label="raw")
            clean_ax.semilogy(freq, np.where(clean > 0, clean, np.nan), linewidth=1.3, label="clean")
        else:
            raw_ax.plot(freq, raw, linewidth=1.3, label=f"{component} raw")
            clean_ax.plot(freq, raw, linewidth=0.9, alpha=0.45, label="raw")
            clean_ax.plot(freq, clean, linewidth=1.3, label="clean")
            raw_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.2)
            clean_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.2)

        raw_ax.set_title(f"{component} observed")
        clean_ax.set_title(f"{component} residual")
        raw_ax.grid(True, alpha=0.3)
        clean_ax.grid(True, alpha=0.3)
        raw_ax.set_xlim(freq[0], freq[-1])
        clean_ax.set_xlim(freq[0], freq[-1])
        raw_ax.set_ylabel("Normalized amplitude")
        clean_ax.set_ylabel("Normalized amplitude")
        clean_ax.set_xlabel("Frequency (Hz)")
        raw_ax.legend()
        clean_ax.legend()

    fig.suptitle(title, fontsize=14)
    return fig


def _plot_component_images(
    freq: np.ndarray,
    raw_avg: dict[str, np.ndarray],
    clean_avg: dict[str, np.ndarray],
    *,
    plot_scale: str,
    title: str,
):
    components = list(raw_avg.keys())
    fig, axes = plt.subplots(
        2,
        len(components),
        figsize=(4.2 * len(components), 8.2),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.92,
        hspace=0.2,
        wspace=0.05,
    )

    lead_ax = axes[0, 0]
    for ax in axes.ravel()[1:]:
        ax.sharex(lead_ax)
        ax.sharey(lead_ax)

    for col_idx, component in enumerate(components):
        panels = [
            (axes[0, col_idx], f"{component} observed", raw_avg[component]),
            (axes[1, col_idx], f"{component} residual", clean_avg[component]),
        ]
        for row_idx, (ax, panel_title, amp) in enumerate(panels):
            _plot_frequency_image(
                fig,
                ax,
                freq=np.asarray(freq, dtype=float),
                amp=np.asarray(amp, dtype=float),
                plot_scale=plot_scale,
                cmap_index=6,
                y_min=float(freq[0]),
                y_max=float(freq[-1]),
                title=panel_title,
                linear_color_label="Normalized amplitude",
                log_color_label="Amplitude (dB)",
                show_colorbar=False,
                annotate_range=True,
            )
            _apply_compact_image_axis_style(
                ax,
                show_right_ylabel=col_idx == len(components) - 1,
            )
            if col_idx == 0:
                ax.text(
                    -0.3,
                    0.5,
                    "Observed" if row_idx == 0 else "Residual",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

    fig.suptitle(title, fontsize=14)
    return fig


def _plot_transfer_diagnostics(
    pair_results: list[PairLeakageResult],
    avg_freq: np.ndarray,
    *,
    title: str,
):
    components = list(pair_results[0].transfer_by_target.keys())
    fig, axes = plt.subplots(
        len(components),
        2,
        figsize=(12.0, 3.3 * len(components)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )

    for row_idx, target in enumerate(components):
        ax_mag = axes[row_idx, 0]
        ax_coh = axes[row_idx, 1]

        predictor_names = list(pair_results[0].transfer_by_target[target].keys())
        for predictor in predictor_names:
            mag_rows = []
            for result in pair_results:
                mag = np.abs(result.transfer_by_target[target][predictor])
                mag_rows.append(np.interp(avg_freq, result.freq, mag))
            mean_mag = np.mean(np.vstack(mag_rows), axis=0)
            ax_mag.plot(avg_freq, mean_mag, linewidth=1.2, label=f"{target}<-{predictor}")

        coh_rows = [
            np.interp(avg_freq, result.freq, result.coherence_by_target[target])
            for result in pair_results
        ]
        mean_coh = np.mean(np.vstack(coh_rows), axis=0)
        ax_coh.plot(avg_freq, mean_coh, linewidth=1.2, color="black", label=f"{target} coherence")

        ax_mag.set_title(f"{target} fitted leakage magnitude")
        ax_coh.set_title(f"{target} multiple coherence")
        ax_mag.set_ylabel("|H(f)|")
        ax_coh.set_ylabel("coherence")
        ax_mag.grid(True, alpha=0.3)
        ax_coh.grid(True, alpha=0.3)
        ax_coh.set_ylim(0.0, 1.05)
        ax_mag.legend()
        ax_coh.legend()
        if row_idx == len(components) - 1:
            ax_mag.set_xlabel("Frequency (Hz)")
            ax_coh.set_xlabel("Frequency (Hz)")

    fig.suptitle(title, fontsize=14)
    return fig


def _render_figure(fig, *, save: str | None, show: bool) -> None:
    if save is not None:
        save_path = ensure_parent_dir(save)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _format_bond_list(display_bonds: list[int]) -> str:
    return ", ".join(str(v) for v in display_bonds) if display_bonds else "(none)"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    if args.freq_min_hz is not None and args.freq_max_hz is not None and args.freq_max_hz <= args.freq_min_hz:
        print("Error: --freq-max-hz must be greater than --freq-min-hz", file=sys.stderr)
        return 1
    if args.welch_len_s <= 0:
        print("Error: --welch-len-s must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= args.welch_overlap < 1.0):
        print("Error: --welch-overlap must satisfy 0 <= value < 1", file=sys.stderr)
        return 1
    if not (0.0 <= args.coherence_floor <= 1.0):
        print("Error: --coherence-floor must satisfy 0 <= value <= 1", file=sys.stderr)
        return 1

    try:
        config_stem = Path(args.config_json).stem
        raw_config = load_dataset_selection_entries(args.config_json)
        required_components = tuple(sorted(set(args.components)))
        _validate_component_inputs(
            raw_config,
            track_data_root=args.track_data_root,
            required_components=required_components,
        )

        keyed_records_by_component: dict[str, dict[tuple[str, int], object]] = {}
        available_display_bonds = None
        selected_display_bonds = None
        for component in required_components:
            keyed_records, available_bonds, selected_bonds = _resolved_component_records(
                raw_config,
                logical_component=component,
                args=args,
            )
            keyed_records_by_component[component] = keyed_records
            if available_display_bonds is None:
                available_display_bonds = available_bonds
                selected_display_bonds = selected_bonds

        common_keys = set.intersection(*(set(records.keys()) for records in keyed_records_by_component.values()))
        common_keys = sorted(common_keys, key=lambda item: (item[0], item[1]))
        if not common_keys:
            raise ValueError("No dataset/bond records were common to all selected components after filtering")

        pair_results: list[PairLeakageResult] = []
        skipped: list[tuple[tuple[str, int], str]] = []
        for key in common_keys:
            records = {component: keyed_records_by_component[component][key] for component in required_components}
            try:
                pair_results.append(_process_one_pair(records, args=args))
            except Exception as exc:
                skipped.append((key, str(exc)))

        if not pair_results:
            raise ValueError("All matched dataset/bond tuples failed during preprocessing or leakage estimation")

        raw_rows_by_component: dict[str, list[SpectrumRow]] = {component: [] for component in required_components}
        clean_rows_by_component: dict[str, list[SpectrumRow]] = {component: [] for component in required_components}
        for result in pair_results:
            for component in required_components:
                raw_rows_by_component[component].append(
                    SpectrumRow(key=result.key, freq=result.freq, amp=result.raw_amp_by_component[component])
                )
                clean_rows_by_component[component].append(
                    SpectrumRow(key=result.key, freq=result.freq, amp=result.clean_amp_by_component[component])
                )

        avg_freq = None
        raw_avg: dict[str, np.ndarray] = {}
        clean_avg: dict[str, np.ndarray] = {}
        norm_low = None
        norm_high = None
        accepted_count_by_component: dict[str, int] = {}

        for component in required_components:
            freq_raw, avg_raw, this_norm_low, this_norm_high, accepted_raw = _average_rows(
                raw_rows_by_component[component],
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                average_domain=args.average_domain,
                lowest_freq=args.freq_min_hz,
                highest_freq=args.freq_max_hz,
            )
            freq_clean, avg_clean, _, _, accepted_clean = _average_rows(
                clean_rows_by_component[component],
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                average_domain=args.average_domain,
                lowest_freq=args.freq_min_hz,
                highest_freq=args.freq_max_hz,
            )
            if not np.allclose(freq_raw, freq_clean, rtol=0.0, atol=1e-12):
                raise ValueError(f"Raw and cleaned averaged frequency grids differ for component '{component}'")
            if avg_freq is None:
                avg_freq = freq_raw
                norm_low = this_norm_low
                norm_high = this_norm_high
            elif not np.allclose(avg_freq, freq_raw, rtol=0.0, atol=1e-12):
                avg_raw = np.interp(avg_freq, freq_raw, avg_raw)
                avg_clean = np.interp(avg_freq, freq_clean, avg_clean)

            raw_avg[component] = avg_raw
            clean_avg[component] = avg_clean
            accepted_count_by_component[component] = min(accepted_raw, accepted_clean)

        assert avg_freq is not None
        n_datasets = len({key[0] for key in common_keys})
        title = args.title or (
            f"Clean Test 2 | multichannel Welch de-mix | components={'/'.join(required_components)} | "
            f"pairs={len(pair_results)} | datasets={n_datasets}"
        )

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds or [])}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds or [])}")
        print(f"Matched dataset/bond tuples before preprocessing: {len(common_keys)}")
        print(f"Accepted tuples after preprocessing/leakage fit: {len(pair_results)}")
        print(f"Skipped tuples: {len(skipped)}")
        for key, message in skipped[:10]:
            print(f"  skipped {key[0]} bond {key[1] + 1}: {message}")
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} additional skipped tuples")

        print(f"Frequency window: [{avg_freq[0]:.6f}, {avg_freq[-1]:.6f}] Hz")
        print(f"Normalization window: [{norm_low:.6f}, {norm_high:.6f}] Hz")
        print(
            f"Cross-spectral fit: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g} "
            f"ridge={args.regularization:.6g} coherence_floor={args.coherence_floor:.6g} "
            f"smooth_bins={args.transfer_smooth_bins}"
        )

        for component in required_components:
            raw_area = float(np.trapz(raw_avg[component], avg_freq))
            clean_area = float(np.trapz(clean_avg[component], avg_freq))
            reduction = 100.0 * (1.0 - clean_area / raw_area) if raw_area > 0 else float("nan")
            coh_stack = np.vstack([
                np.interp(avg_freq, result.freq, result.coherence_by_target[component])
                for result in pair_results
            ])
            print(
                f"{component}: accepted={accepted_count_by_component[component]} | "
                f"area raw={raw_area:.6g} clean={clean_area:.6g} reduction={reduction:.3f}% | "
                f"mean coherence={float(np.mean(coh_stack)):.4f}"
            )

        fig = (
            _plot_component_images(
                avg_freq,
                raw_avg,
                clean_avg,
                plot_scale=args.plot_scale,
                title=title,
            )
            if args.full_image
            else _plot_component_curves(
                avg_freq,
                raw_avg,
                clean_avg,
                plot_scale=args.plot_scale,
                title=title,
            )
        )
        _render_figure(fig, save=args.save, show=not args.no_show)

        if args.spectrasave is not None:
            for component, amp in clean_avg.items():
                default_name = build_default_spectrasave_name(
                    config_stem,
                    "cleantest2",
                    "clean",
                    f"component-{component}",
                )
                export_path = resolve_spectrasave_path(
                    args.spectrasave,
                    default_name=default_name,
                    multi_suffix=component,
                )
                assert export_path is not None
                saved = save_spectrum_msgpack(
                    export_path,
                    freq=avg_freq,
                    amplitude=amp,
                    label=f"{config_stem} cleantest2 {component}",
                    metadata={
                        "sourceKind": "cleantest2",
                        "spectrumKind": "welch",
                        "configPath": args.config_json,
                        "component": component,
                        "components": list(required_components),
                        "normalize": args.normalize,
                        "relativeRange": list(map(float, args.relative_range)),
                        "averageDomain": args.average_domain,
                        "freqLowHz": float(avg_freq[0]),
                        "freqHighHz": float(avg_freq[-1]),
                        "normLowHz": float(norm_low),
                        "normHighHz": float(norm_high),
                        "matchedTupleCount": len(common_keys),
                        "acceptedTupleCount": len(pair_results),
                        "welchLenS": float(args.welch_len_s),
                        "welchOverlap": float(args.welch_overlap),
                        "regularization": float(args.regularization),
                        "coherenceFloor": float(args.coherence_floor),
                        "transferSmoothBins": int(args.transfer_smooth_bins),
                    },
                )
                print(f"Spectrum saved to: {saved}")

        if args.show_transfers:
            fig_diag = _plot_transfer_diagnostics(
                pair_results,
                avg_freq,
                title=f"{title} | transfer diagnostics",
            )
            _render_figure(fig_diag, save=None, show=not args.no_show)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
