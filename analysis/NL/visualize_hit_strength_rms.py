#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.peaks import load_peaks_csv, resolve_peaks_csv
from analysis.tools.signal import compute_one_sided_fft, preprocess_signal
from analysis.tools.flattening import compute_flattening


FFT_LOG_FLOOR = 1e-8


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
class RMSHit:
    index: int
    time_s: float
    baseline_value: float
    rms_strength: float
    mean_abs_strength: float
    peak_abs_strength: float
    prev_gap_s: float | None
    next_gap_s: float | None
    baseline_start_s: float
    baseline_stop_s: float
    analysis_start_s: float
    analysis_stop_s: float
    freq: np.ndarray
    amplitude: np.ndarray
    peak_amplitudes: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alternative hit-strength visualization using local time-domain RMS after local baseline subtraction.",
    )
    parser.add_argument("dataset")
    parser.add_argument("hit_csv")
    parser.add_argument("peaks")
    parser.add_argument("--component", default="x", choices=["x", "y", "a"])
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--bond-spacing-mode", default="purecomoving", choices=["default", "purecomoving"])
    parser.add_argument("--window-s", type=float, default=30.0, help="Post-hit FFT/response window length in seconds. Default: 30")
    parser.add_argument("--window-offset-s", type=float, default=3.0, help="Delay after hit before response window starts.")
    parser.add_argument("--baseline-before-s", type=float, default=2.0, help="Gap between hit time and the end of the baseline window.")
    parser.add_argument("--baseline-window-s", type=float, default=4.0, help="Length of the local baseline window taken before the hit.")
    parser.add_argument("--isolation-window-s", type=float, default=20.0, help="Reject hits with a neighbor closer than this.")
    parser.add_argument("--df", type=float, default=0.05)
    parser.add_argument("--focus-span-s", type=float, default=50.0, help="Total time span shown in each focus-row diagnostic plot. Default: 50")
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


def _nearest_neighbor_gaps(hit_times: np.ndarray, idx: int) -> tuple[float | None, float | None]:
    prev_gap = None if idx == 0 else float(hit_times[idx] - hit_times[idx - 1])
    next_gap = None if idx == (len(hit_times) - 1) else float(hit_times[idx + 1] - hit_times[idx])
    return prev_gap, next_gap


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


def _build_rms_hits(
    hits: list[HitInput],
    processed_t: np.ndarray,
    aggregate_signal: np.ndarray,
    peaks: list[float],
    *,
    window_s: float,
    window_offset_s: float,
    baseline_before_s: float,
    baseline_window_s: float,
    isolation_window_s: float,
    df: float,
    flatten: bool = False,
) -> list[RMSHit]:
    hit_times = np.asarray([hit.time_s for hit in hits], dtype=float)
    out: list[RMSHit] = []
    for idx, hit in enumerate(hits):
        prev_gap, next_gap = _nearest_neighbor_gaps(hit_times, idx)
        nearest_gap = min([gap for gap in (prev_gap, next_gap) if gap is not None], default=float("inf"))
        if nearest_gap < isolation_window_s:
            continue

        baseline_stop = float(hit.time_s - baseline_before_s)
        baseline_start = float(baseline_stop - baseline_window_s)
        analysis_start = float(hit.time_s + window_offset_s)
        analysis_stop = float(analysis_start + window_s)

        overlap_hit_mask = (hit_times > float(hit.time_s)) & (hit_times < analysis_stop)
        if np.any(overlap_hit_mask):
            continue

        if baseline_start < float(processed_t[0]) or analysis_stop > float(processed_t[-1]):
            continue

        baseline_mask = (processed_t >= baseline_start) & (processed_t <= baseline_stop)
        analysis_mask = (processed_t >= analysis_start) & (processed_t <= analysis_stop)
        if np.count_nonzero(baseline_mask) < 16 or np.count_nonzero(analysis_mask) < 16:
            continue

        baseline_values = aggregate_signal[baseline_mask]
        analysis_values = aggregate_signal[analysis_mask]
        baseline_value = float(np.mean(baseline_values))
        centered = np.asarray(analysis_values - baseline_value, dtype=float)

        rms_strength = float(np.sqrt(np.mean(np.square(centered))))
        mean_abs_strength = float(np.mean(np.abs(centered)))
        peak_abs_strength = float(np.max(np.abs(centered)))

        t_window = processed_t[analysis_mask]
        dt = float(np.median(np.diff(t_window)))
        spectrum = compute_one_sided_fft(centered, dt)
        
        amp = spectrum.amplitude
        if flatten:
            try:
                flattening = compute_flattening(spectrum.freq, amp)
                amp = flattening.flattened
            except Exception as e:
                print(f"Warning: Flattening failed for hit {hit.index}: {e}", file=sys.stderr)

        peak_amplitudes = np.asarray(
            [_integrated_peak_amplitude(spectrum.freq, amp, peak_hz, df) for peak_hz in peaks],
            dtype=float,
        )
        out.append(
            RMSHit(
                index=hit.index,
                time_s=hit.time_s,
                baseline_value=baseline_value,
                rms_strength=rms_strength,
                mean_abs_strength=mean_abs_strength,
                peak_abs_strength=peak_abs_strength,
                prev_gap_s=prev_gap,
                next_gap_s=next_gap,
                baseline_start_s=baseline_start,
                baseline_stop_s=baseline_stop,
                analysis_start_s=analysis_start,
                analysis_stop_s=analysis_stop,
                freq=np.asarray(spectrum.freq, dtype=float),
                amplitude=np.asarray(amp, dtype=float),
                peak_amplitudes=peak_amplitudes,
            )
        )
    return out


def _choose_focus_hits(rms_hits: list[RMSHit], explicit: list[int]) -> list[RMSHit]:
    by_id = {hit.index: hit for hit in rms_hits}
    chosen: list[RMSHit] = []
    for hit_id in explicit:
        if hit_id in by_id:
            chosen.append(by_id[hit_id])
    if chosen:
        return chosen[:4]
    if not rms_hits:
        return []
    strengths = np.asarray([hit.rms_strength for hit in rms_hits], dtype=float)
    order = np.argsort(strengths)
    picks = [order[0], order[len(order) // 2], order[-1]]
    if 3 in by_id and all(by_id[3].index != rms_hits[int(idx)].index for idx in picks):
        chosen.append(by_id[3])
    for idx in picks:
        hit = rms_hits[int(idx)]
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
    rms_hits = _build_rms_hits(
        hits,
        processed_t,
        aggregate_signal,
        peaks,
        window_s=args.window_s,
        window_offset_s=args.window_offset_s,
        baseline_before_s=args.baseline_before_s,
        baseline_window_s=args.baseline_window_s,
        isolation_window_s=args.isolation_window_s,
        df=args.df,
    )
    if not rms_hits:
        raise ValueError("No hits survived the local RMS window checks")
    focus_hits = _choose_focus_hits(rms_hits, args.focus_hit)
    peak_matrix = np.vstack([hit.peak_amplitudes for hit in rms_hits])
    mean_abs_strengths = np.asarray([hit.mean_abs_strength for hit in rms_hits], dtype=float)

    overview_fig, overview_axes = plt.subplots(4, 1, figsize=(16, 15), constrained_layout=True)
    ax0, ax1, ax2, ax3 = overview_axes

    ax0.plot(processed_t, aggregate_signal, color="black", linewidth=0.8)
    for hit in hits:
        ax0.axvline(hit.time_s, color="0.86", linestyle=":", linewidth=0.8)
    for hit in rms_hits:
        ax0.axvspan(hit.baseline_start_s, hit.baseline_stop_s, color="tab:blue", alpha=0.10)
        ax0.axvspan(hit.analysis_start_s, hit.analysis_stop_s, color="tab:green", alpha=0.12)
        ax0.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.set_title(f"{args.dataset} local-window RMS workflow | blue=baseline window | green=response window")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.25)

    old_peak = np.asarray([hit.strength_peak for hit in hits], dtype=float)
    old_integral = np.asarray([hit.strength_integral for hit in hits], dtype=float)
    ax1.scatter(old_peak, old_integral, color="0.6", s=24, label="old broadband proxy")
    ax1.scatter(
        [hit.peak_abs_strength for hit in rms_hits],
        [hit.mean_abs_strength for hit in rms_hits],
        color="black",
        s=28,
        label="new local mean|x| proxy",
    )
    for hit in rms_hits:
        ax1.text(hit.peak_abs_strength, hit.mean_abs_strength, str(hit.index), fontsize=8, color="crimson", ha="left", va="bottom")
    ax1.set_title("Old broadband strength vs new local mean|x| strength")
    ax1.set_xlabel("x-axis: peak_abs_strength")
    ax1.set_ylabel("y-axis: mean|x| strength")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.scatter(mean_abs_strengths, [hit.time_s for hit in rms_hits], color="black", s=28)
    for hit in rms_hits:
        ax2.text(hit.mean_abs_strength, hit.time_s, str(hit.index), fontsize=8, color="crimson", ha="left", va="bottom")
    ax2.set_title("Accepted hit mean|x| strength vs time")
    ax2.set_xlabel("Local mean|x| strength")
    ax2.set_ylabel("Hit time (s)")
    ax2.grid(True, alpha=0.25)

    im = ax3.imshow(peak_matrix.T, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
    overview_fig.colorbar(im, ax=ax3, label="sqrt(int power)")
    ax3.set_yticks(np.arange(len(peaks)))
    ax3.set_yticklabels([f"{peak:.3f}" for peak in peaks], fontsize=8)
    ax3.set_xticks(np.arange(len(rms_hits)))
    ax3.set_xticklabels([str(hit.index) for hit in rms_hits], fontsize=8)
    ax3.set_xlabel("Accepted hit index")
    ax3.set_ylabel("Peak frequency (Hz)")
    ax3.set_title("Final output shape with local RMS strength workflow")

    n_focus = max(1, len(focus_hits))
    focus_fig, focus_axes = plt.subplots(n_focus, 3, figsize=(17, 4.5 * n_focus), constrained_layout=True)
    if n_focus == 1:
        focus_axes = np.asarray([focus_axes], dtype=object)

    for row_idx, hit in enumerate(focus_hits):
        ax_sig, ax_centered, ax_fft = focus_axes[row_idx]
        half_span = 0.5 * float(args.focus_span_s)
        local_low = max(float(processed_t[0]), hit.time_s - half_span)
        local_high = min(float(processed_t[-1]), hit.time_s + half_span)
        mask = (processed_t >= local_low) & (processed_t <= local_high)
        local_t = processed_t[mask]
        local_y = aggregate_signal[mask]
        ax_sig.plot(local_t, local_y, color="black", linewidth=0.8)
        ax_sig.axvspan(hit.baseline_start_s, hit.baseline_stop_s, color="tab:blue", alpha=0.12, label="baseline")
        ax_sig.axvspan(hit.analysis_start_s, hit.analysis_stop_s, color="tab:green", alpha=0.12, label="response")
        ax_sig.axhline(hit.baseline_value, color="tab:blue", linestyle="--", linewidth=1.0)
        ax_sig.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_sig.set_title(f"Hit {hit.index}: local windows")
        ax_sig.set_xlabel("Time (s)")
        ax_sig.set_ylabel("Signal")
        ax_sig.grid(True, alpha=0.25)
        ax_sig.legend(fontsize=7, loc="upper right")

        response_mask = (processed_t >= hit.analysis_start_s) & (processed_t <= hit.analysis_stop_s)
        response_t = processed_t[response_mask]
        centered = aggregate_signal[response_mask] - hit.baseline_value
        ax_centered.plot(response_t, centered, color="black", linewidth=0.9)
        ax_centered.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        ax_centered.fill_between(response_t, 0.0, centered, where=(centered >= 0), color="tab:green", alpha=0.18)
        ax_centered.fill_between(response_t, 0.0, centered, where=(centered < 0), color="tab:orange", alpha=0.18)
        ax_centered.set_title(
            f"Hit {hit.index}: centered response | mean|x|={hit.mean_abs_strength:.3g} rms={hit.rms_strength:.3g}"
        )
        ax_centered.set_xlabel("Time (s)")
        ax_centered.set_ylabel("Centered signal")
        ax_centered.grid(True, alpha=0.25)

        ax_fft.plot(hit.freq, hit.amplitude, color="black", linewidth=1.0)
        ax_fft.set_yscale("log")
        for peak_hz, peak_amp in zip(peaks, hit.peak_amplitudes):
            ax_fft.axvspan(peak_hz - args.df, peak_hz + args.df, color="crimson", alpha=0.08)
            ax_fft.axvline(peak_hz, color="crimson", linestyle=":", linewidth=0.8)
            ax_fft.text(
                peak_hz,
                max(float(peak_amp), FFT_LOG_FLOOR),
                f"{peak_hz:.2f}",
                fontsize=7,
                color="crimson",
                ha="left",
                va="bottom",
            )
        ax_fft.set_xlim(0.0, max(float(max(peaks) + 1.5), 1.0))
        positive = hit.amplitude[np.isfinite(hit.amplitude) & (hit.amplitude > 0)]
        if positive.size > 0:
            ymin = max(float(np.min(positive)) * 0.7, FFT_LOG_FLOOR)
            ymax = float(np.max(positive)) * 1.3
            if ymax > ymin:
                ax_fft.set_ylim(ymin, ymax)
        ax_fft.set_title(f"Hit {hit.index}: FFT of centered response")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude (log)")
        ax_fft.grid(True, alpha=0.25)

    print(f"Dataset: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Peaks file: {peaks_path}")
    print(f"Frame-time repaired: {'yes' if repaired_flag else 'no'}")
    print(f"Repaired dt from frame numbers: {repaired_dt:.9f} s")
    print(f"Input hits: {len(hits)}")
    print(f"Accepted RMS hits: {len(rms_hits)}")
    print("Primary local strength proxy: mean|x| of the centered response window")
    print(f"Focus plot span: {args.focus_span_s:.6g} s")
    print(f"Focus hits: {[hit.index for hit in focus_hits]}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.component}__rms_workflow"
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
