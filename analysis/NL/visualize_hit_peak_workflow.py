#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.peaks import load_peaks_csv, resolve_peaks_csv
from analysis.tools.signal import compute_complex_spectrogram, compute_one_sided_fft, preprocess_signal


@dataclass(frozen=True)
class HitInput:
    dataset: str | None
    component: str | None
    index: int
    time_s: float
    strength_peak: float
    strength_integral: float
    broadband_at_peak: float


@dataclass(frozen=True)
class AcceptedHit:
    index: int
    time_s: float
    strength_peak: float
    strength_integral: float
    broadband_at_peak: float
    window_start_s: float
    window_stop_s: float
    peak_amplitudes: np.ndarray
    freq: np.ndarray
    amplitude: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explain the hit-strength and per-hit peak-scaling workflow with more explicit diagnostics.",
    )
    parser.add_argument("dataset")
    parser.add_argument("hit_csv")
    parser.add_argument("peaks")
    parser.add_argument("--component", default="x", choices=["x", "y", "a"])
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--bond-spacing-mode", default="purecomoving", choices=["default", "purecomoving"])
    parser.add_argument("--sliding-len-s", type=float, default=1.5)
    parser.add_argument("--broadband-min-hz", type=float, default=3.0)
    parser.add_argument("--broadband-max-hz", type=float, default=25.0)
    parser.add_argument("--smooth-window-s", type=float, default=0.35)
    parser.add_argument("--baseline-window-s", type=float, default=18.0)
    parser.add_argument("--threshold-sigma", type=float, default=3.0)
    parser.add_argument("--strength-window-s", type=float, default=2.5)
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--window-offset-s", type=float, default=3.0)
    parser.add_argument("--isolation-window-s", type=float, default=20.0)
    parser.add_argument("--df", type=float, default=0.05)
    parser.add_argument("--focus-hit", type=int, action="append", default=[])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _repair_time_vector(frame_times_s: np.ndarray, frame_numbers: np.ndarray) -> tuple[np.ndarray, float, bool]:
    t = np.asarray(frame_times_s, dtype=float)
    frame_numbers = np.asarray(frame_numbers, dtype=float)
    positive_dt = np.diff(t)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if positive_dt.size == 0:
        raise ValueError("Could not infer a positive frame interval from frameTimes_s")
    dt = float(np.median(positive_dt))
    rebuilt = (frame_numbers - float(frame_numbers[0])) * dt
    invalid_tail = bool(np.any(t[-max(1, min(32, t.size)) :] == 0.0))
    nonmonotone_fraction = float(np.mean(np.diff(t) <= 0)) if t.size > 1 else 0.0
    clearly_bad = invalid_tail or nonmonotone_fraction > 0.001 or (not np.all(np.isfinite(t)))
    return (rebuilt if clearly_bad else t), dt, clearly_bad


def _moving_average(y: np.ndarray, window_samples: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.size == 0 or window_samples <= 1:
        return arr.copy()
    kernel = np.ones(int(window_samples), dtype=float) / float(window_samples)
    return np.convolve(arr, kernel, mode="same")


def _robust_threshold(y: np.ndarray, sigma: float) -> tuple[float, float, float]:
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_std = 1.4826 * mad
    if robust_std <= 0 or not np.isfinite(robust_std):
        robust_std = float(np.std(finite))
    return median + float(sigma) * robust_std, median, robust_std


def _load_hits_csv(path: str | Path) -> list[HitInput]:
    out: list[HitInput] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(
                HitInput(
                    dataset=(row.get("dataset") or None),
                    component=(row.get("component") or None),
                    index=int(row["hit_index"]),
                    time_s=float(row["time_s"]),
                    strength_peak=float(row["strength_peak"]),
                    strength_integral=float(row["strength_integral"]),
                    broadband_at_peak=float(row["broadband_at_peak"]),
                )
            )
    if not out:
        raise ValueError(f"No hits loaded from {path}")
    return out


def _aggregate(processed_signals: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(processed_signals), axis=0)


def _build_broadband_trace(processed_signals: list[np.ndarray], fs: float, sliding_len_s: float, fmin: float, fmax: float):
    spec_results = [compute_complex_spectrogram(y, fs, sliding_len_s) for y in processed_signals]
    if any(result is None for result in spec_results):
        raise ValueError("Sliding FFT window too short")
    ref = spec_results[0]
    assert ref is not None
    freq_mask = (ref.f >= fmin) & (ref.f <= fmax)
    traces = [np.mean(np.abs(result.S_complex[freq_mask, :]) ** 2, axis=0) for result in spec_results if result is not None]
    broadband_raw = np.sqrt(np.mean(np.vstack(traces), axis=0))
    return ref, broadband_raw


def _integrated_peak_amplitude(freq: np.ndarray, amp: np.ndarray, peak_hz: float, df: float) -> float:
    low = max(float(freq[0]), float(peak_hz - df))
    high = min(float(freq[-1]), float(peak_hz + df))
    if high <= low:
        return 0.0
    interior_mask = (freq > low) & (freq < high)
    edge_freq = np.array([low, high], dtype=float)
    edge_amp = np.interp(edge_freq, freq, amp)
    if np.any(interior_mask):
        window_freq = np.concatenate(([low], freq[interior_mask], [high]))
        window_amp = np.concatenate(([edge_amp[0]], amp[interior_mask], [edge_amp[1]]))
    else:
        window_freq = edge_freq
        window_amp = edge_amp
    power = np.square(np.abs(window_amp))
    integral = float(np.trapz(power, window_freq))
    return 0.0 if (not np.isfinite(integral) or integral <= 0.0) else float(np.sqrt(integral))


def _accept_hits(
    hits: list[HitInput],
    processed_t: np.ndarray,
    aggregate_signal: np.ndarray,
    peaks: list[float],
    *,
    window_s: float,
    window_offset_s: float,
    isolation_window_s: float,
    df: float,
) -> list[AcceptedHit]:
    hit_times = np.asarray([hit.time_s for hit in hits], dtype=float)
    accepted: list[AcceptedHit] = []
    for idx, hit in enumerate(hits):
        prev_gap = None if idx == 0 else float(hit.time_s - hit_times[idx - 1])
        next_gap = None if idx == (len(hits) - 1) else float(hit_times[idx + 1] - hit.time_s)
        nearest_gap = min([gap for gap in (prev_gap, next_gap) if gap is not None], default=float("inf"))
        if nearest_gap < isolation_window_s:
            continue
        start_s = float(hit.time_s + window_offset_s)
        stop_s = start_s + window_s
        mask = (processed_t >= start_s) & (processed_t <= stop_s)
        if np.count_nonzero(mask) < 16:
            continue
        t_window = processed_t[mask]
        y_window = aggregate_signal[mask]
        dt = float(np.median(np.diff(t_window)))
        spectrum = compute_one_sided_fft(y_window, dt)
        peak_amps = np.asarray(
            [_integrated_peak_amplitude(spectrum.freq, spectrum.amplitude, peak_hz, df) for peak_hz in peaks],
            dtype=float,
        )
        accepted.append(
            AcceptedHit(
                index=hit.index,
                time_s=hit.time_s,
                strength_peak=hit.strength_peak,
                strength_integral=hit.strength_integral,
                broadband_at_peak=hit.broadband_at_peak,
                window_start_s=start_s,
                window_stop_s=stop_s,
                peak_amplitudes=peak_amps,
                freq=np.asarray(spectrum.freq, dtype=float),
                amplitude=np.asarray(spectrum.amplitude, dtype=float),
            )
        )
    return accepted


def _choose_focus_hits(accepted_hits: list[AcceptedHit], explicit: list[int]) -> list[AcceptedHit]:
    by_id = {hit.index: hit for hit in accepted_hits}
    chosen: list[AcceptedHit] = []
    for hit_id in explicit:
        if hit_id in by_id:
            chosen.append(by_id[hit_id])
    if chosen:
        return chosen[:4]
    if not accepted_hits:
        return []
    strengths = np.asarray([hit.strength_integral for hit in accepted_hits], dtype=float)
    order = np.argsort(strengths)
    picks = [order[0], order[len(order) // 2], order[-1]]
    if 3 in by_id and all(by_id[3].index != accepted_hits[int(idx)].index for idx in picks):
        chosen.append(by_id[3])
    for idx in picks:
        hit = accepted_hits[int(idx)]
        if all(existing.index != hit.index for existing in chosen):
            chosen.append(hit)
    return chosen[:4]


def main() -> int:
    args = build_parser().parse_args()
    hits = _load_hits_csv(args.hit_csv)
    peaks_path = resolve_peaks_csv(args.peaks)
    peaks = load_peaks_csv(peaks_path)

    dataset_name = f"{args.dataset}_{args.component}"
    bond_dataset = load_bond_signal_dataset(
        dataset=dataset_name,
        track_data_root=args.track_data_root,
        bond_spacing_mode=args.bond_spacing_mode,
        component=args.component,
    )
    from analysis.tools.io import load_track2_dataset

    track2 = load_track2_dataset(dataset=dataset_name, track_data_root=args.track_data_root)
    repaired_t, repaired_dt, repaired_flag = _repair_time_vector(track2.frame_times_s, track2.frame_numbers)

    processed_signals: list[np.ndarray] = []
    processed_t: np.ndarray | None = None
    for bond_idx in range(bond_dataset.signal_matrix.shape[1]):
        processed, error = preprocess_signal(repaired_t, bond_dataset.signal_matrix[:, bond_idx])
        if processed is None:
            print(f"Skipping bond {bond_idx + 1}: {error}", file=sys.stderr)
            continue
        if processed_t is None:
            processed_t = np.asarray(processed.t, dtype=float)
        processed_signals.append(np.asarray(processed.y, dtype=float))
    if processed_t is None or not processed_signals:
        raise ValueError("No processed signals available")

    aggregate_signal = _aggregate(processed_signals)
    fs = 1.0 / float(np.median(np.diff(processed_t)))
    spec_ref, broadband_raw = _build_broadband_trace(
        processed_signals, fs, args.sliding_len_s, args.broadband_min_hz, args.broadband_max_hz
    )
    t_spec = spec_ref.t + float(processed_t[0])
    smooth_samples = max(1, int(round(args.smooth_window_s / float(np.median(np.diff(t_spec))))))
    baseline_samples = max(1, int(round(args.baseline_window_s / float(np.median(np.diff(t_spec))))))
    broadband = _moving_average(broadband_raw, smooth_samples)
    baseline = _moving_average(broadband, baseline_samples)
    excess = np.maximum(0.0, broadband - baseline)
    threshold, excess_median, excess_std = _robust_threshold(excess, args.threshold_sigma)

    accepted_hits = _accept_hits(
        hits,
        processed_t,
        aggregate_signal,
        peaks,
        window_s=args.window_s,
        window_offset_s=args.window_offset_s,
        isolation_window_s=args.isolation_window_s,
        df=args.df,
    )
    focus_hits = _choose_focus_hits(accepted_hits, args.focus_hit)
    peak_matrix = np.vstack([hit.peak_amplitudes for hit in accepted_hits]) if accepted_hits else np.empty((0, len(peaks)))

    overview_fig, overview_axes = plt.subplots(4, 1, figsize=(16, 15), constrained_layout=True)
    ax0, ax1, ax2, ax3 = overview_axes

    ax0.plot(processed_t, aggregate_signal, color="black", linewidth=0.8)
    for hit in hits:
        ax0.axvline(hit.time_s, color="0.85", linestyle=":", linewidth=0.8)
    for hit in accepted_hits:
        ax0.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
        ax0.axvspan(hit.window_start_s, hit.window_stop_s, color="tab:green", alpha=0.12)
    ax0.set_title(f"{args.dataset} aggregate signal with accepted FFT windows")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.25)

    ax1.plot(t_spec, broadband_raw, color="0.8", linewidth=0.8, label="raw broadband")
    ax1.plot(t_spec, broadband, color="navy", linewidth=1.2, label="smoothed broadband")
    ax1.plot(t_spec, baseline, color="0.2", linestyle="--", linewidth=1.0, label="baseline")
    ax1.plot(t_spec, excess, color="darkorange", linewidth=1.1, label="excess")
    ax1.axhline(excess_median, color="0.55", linestyle=":", linewidth=1.0, label="excess median")
    ax1.axhline(threshold, color="crimson", linestyle="--", linewidth=1.1, label="excess threshold")
    for hit in hits:
        ax1.axvspan(hit.time_s, hit.time_s + args.strength_window_s, color="crimson", alpha=0.06)
        ax1.text(hit.time_s, hit.strength_peak, f"H{hit.index}", fontsize=7, color="crimson", ha="left", va="bottom")
    ax1.set_title("Broadband strength construction: raw, baseline, excess, and strength windows")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Broadband amplitude")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", ncol=3, fontsize=8)

    ax2.scatter([hit.strength_peak for hit in hits], [hit.strength_integral for hit in hits], color="black", s=26)
    for hit in hits:
        ax2.text(hit.strength_peak, hit.strength_integral, str(hit.index), fontsize=8, color="crimson", ha="left", va="bottom")
    ax2.set_title("Hit strength proxy comparison")
    ax2.set_xlabel("strength_peak")
    ax2.set_ylabel("strength_integral")
    ax2.grid(True, alpha=0.25)

    if accepted_hits:
        image = peak_matrix.T
        im = ax3.imshow(image, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
        overview_fig.colorbar(im, ax=ax3, label="sqrt(int power)")
        ax3.set_yticks(np.arange(len(peaks)))
        ax3.set_yticklabels([f"{peak:.3f}" for peak in peaks], fontsize=8)
        ax3.set_xticks(np.arange(len(accepted_hits)))
        ax3.set_xticklabels([str(hit.index) for hit in accepted_hits], fontsize=8)
        ax3.set_xlabel("Accepted hit index")
        ax3.set_ylabel("Peak frequency (Hz)")
        ax3.set_title("Final output shape: per-hit peak amplitudes")
    else:
        ax3.set_title("No accepted hits")

    n_focus = max(1, len(focus_hits))
    focus_fig, focus_axes = plt.subplots(n_focus, 3, figsize=(17, 4.5 * n_focus), constrained_layout=True)
    if n_focus == 1:
        focus_axes = np.asarray([focus_axes], dtype=object)

    for row_idx, hit in enumerate(focus_hits):
        ax_sig, ax_broad, ax_fft = focus_axes[row_idx]
        local_low = max(float(processed_t[0]), hit.time_s - 8.0)
        local_high = min(float(processed_t[-1]), hit.window_stop_s + 4.0)

        signal_mask = (processed_t >= local_low) & (processed_t <= local_high)
        ax_sig.plot(processed_t[signal_mask], aggregate_signal[signal_mask], color="black", linewidth=0.8)
        ax_sig.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_sig.axvspan(hit.window_start_s, hit.window_stop_s, color="tab:green", alpha=0.18)
        ax_sig.set_title(f"Hit {hit.index}: time signal and FFT window")
        ax_sig.set_xlabel("Time (s)")
        ax_sig.set_ylabel("Amplitude")
        ax_sig.grid(True, alpha=0.25)

        broad_mask = (t_spec >= local_low) & (t_spec <= local_high)
        ax_broad.plot(t_spec[broad_mask], broadband[broad_mask], color="navy", linewidth=1.2, label="smoothed")
        ax_broad.plot(t_spec[broad_mask], baseline[broad_mask], color="0.2", linestyle="--", linewidth=1.0, label="baseline")
        ax_broad.plot(t_spec[broad_mask], excess[broad_mask], color="darkorange", linewidth=1.1, label="excess")
        strength_mask = (t_spec >= hit.time_s) & (t_spec <= (hit.time_s + args.strength_window_s))
        local_strength_mask = strength_mask & broad_mask
        if np.any(local_strength_mask):
            ax_broad.fill_between(
                t_spec[local_strength_mask],
                0.0,
                excess[local_strength_mask],
                color="crimson",
                alpha=0.18,
                label="strength integral",
            )
        ax_broad.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_broad.axhline(threshold, color="crimson", linestyle=":", linewidth=1.0)
        ax_broad.set_title(
            f"Hit {hit.index}: strength proxy | peak={hit.strength_peak:.3g} int={hit.strength_integral:.3g}"
        )
        ax_broad.set_xlabel("Time (s)")
        ax_broad.set_ylabel("Broadband")
        ax_broad.grid(True, alpha=0.25)
        ax_broad.legend(fontsize=7, loc="upper right")

        ax_fft.plot(hit.freq, hit.amplitude, color="black", linewidth=1.0)
        for peak_hz, peak_amp in zip(peaks, hit.peak_amplitudes):
            ax_fft.axvspan(peak_hz - args.df, peak_hz + args.df, color="crimson", alpha=0.08)
            ax_fft.axvline(peak_hz, color="crimson", linestyle=":", linewidth=0.8)
            ax_fft.text(peak_hz, peak_amp, f"{peak_hz:.2f}", fontsize=7, color="crimson", ha="left", va="bottom")
        ax_fft.set_xlim(0.0, max(float(max(peaks) + 1.5), 1.0))
        ax_fft.set_title(f"Hit {hit.index}: per-hit FFT and measured peak bands")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")
        ax_fft.grid(True, alpha=0.25)

    print(f"Dataset: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Peaks file: {peaks_path}")
    print(f"Frame-time repaired: {'yes' if repaired_flag else 'no'}")
    print(f"Repaired dt from frame numbers: {repaired_dt:.9f} s")
    print(f"Hit CSV rows: {len(hits)}")
    print(f"Accepted hits after offset/isolation: {len(accepted_hits)}")
    print(f"Broadband threshold: median={excess_median:.6g} robust_std={excess_std:.6g} threshold={threshold:.6g}")
    print(f"Focus hits: {[hit.index for hit in focus_hits]}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.component}__workflow"
        overview_path = save_dir / f"{stem}__overview.png"
        focus_path = save_dir / f"{stem}__focus.png"
        overview_fig.savefig(overview_path, dpi=160)
        focus_fig.savefig(focus_path, dpi=160)
        print(f"Saved overview figure: {overview_path}")
        print(f"Saved focus figure: {focus_path}")

    if args.no_show:
        plt.close(overview_fig)
        plt.close(focus_fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
