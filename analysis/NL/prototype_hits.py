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
from analysis.tools.signal import compute_complex_spectrogram, preprocess_signal


@dataclass(frozen=True)
class HitRecord:
    index: int
    time_s: float
    strength_peak: float
    strength_integral: float
    broadband_at_peak: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype hit identification and hit-strength estimation from broadband sliding-spectrum energy.",
    )
    parser.add_argument("dataset", help="Dataset stem, e.g. IMG_0681_rot270")
    parser.add_argument("--component", default="x", choices=["x", "y", "a"])
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--bond-spacing-mode", default="purecomoving", choices=["default", "purecomoving"])
    parser.add_argument("--sliding-len-s", type=float, default=1.5)
    parser.add_argument("--broadband-min-hz", type=float, default=3.0)
    parser.add_argument("--broadband-max-hz", type=float, default=25.0)
    parser.add_argument("--smooth-window-s", type=float, default=0.35)
    parser.add_argument("--baseline-window-s", type=float, default=18.0)
    parser.add_argument("--threshold-sigma", type=float, default=3.0)
    parser.add_argument("--min-hit-separation-s", type=float, default=6.0)
    parser.add_argument("--strength-window-s", type=float, default=2.5)
    parser.add_argument("--max-hits", type=int, default=None)
    parser.add_argument("--save-dir", default=None, help="Optional directory for PNG/CSV outputs.")
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

    if not np.all(np.isfinite(frame_numbers)):
        repaired = np.arange(t.size, dtype=float) * dt
        return repaired, dt, True

    rebuilt = (frame_numbers - float(frame_numbers[0])) * dt
    nonmonotone_fraction = float(np.mean(np.diff(t) <= 0)) if t.size > 1 else 0.0
    invalid_tail = bool(np.any(t[-max(1, min(32, t.size)) :] == 0.0))
    clearly_bad = nonmonotone_fraction > 0.001 or invalid_tail or (not np.all(np.isfinite(t)))
    return (rebuilt if clearly_bad else t), dt, clearly_bad


def _moving_average(y: np.ndarray, window_samples: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if window_samples <= 1 or arr.size == 0:
        return arr.copy()
    kernel = np.ones(int(window_samples), dtype=float) / float(window_samples)
    return np.convolve(arr, kernel, mode="same")


def _robust_threshold(y: np.ndarray, sigma: float) -> tuple[float, float, float]:
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot build a threshold from an all-NaN broadband trace")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_std = 1.4826 * mad
    if robust_std <= 0 or not np.isfinite(robust_std):
        robust_std = float(np.std(finite))
    threshold = median + float(sigma) * robust_std
    return threshold, median, robust_std


def _aggregate_for_display(processed_signals: list[np.ndarray]) -> np.ndarray:
    stack = np.vstack([np.asarray(y, dtype=float) for y in processed_signals])
    return np.mean(stack, axis=0)


def _build_broadband_trace(
    processed_signals: list[np.ndarray],
    fs: float,
    sliding_len_s: float,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec_results = []
    for y in processed_signals:
        result = compute_complex_spectrogram(y, fs, sliding_len_s)
        if result is None:
            raise ValueError("Sliding spectrogram window is too short for the processed signal")
        spec_results.append(result)

    ref = spec_results[0]
    freq_mask = (ref.f >= float(fmin)) & (ref.f <= float(fmax))
    if not np.any(freq_mask):
        raise ValueError(
            f"No spectrogram bins fall inside the requested broadband range [{fmin:.3g}, {fmax:.3g}] Hz"
        )

    power_traces = []
    for result in spec_results:
        band_power = np.mean(np.abs(result.S_complex[freq_mask, :]) ** 2, axis=0)
        power_traces.append(band_power)

    broadband = np.sqrt(np.mean(np.vstack(power_traces), axis=0))
    return ref.t, ref.f, broadband


def _detect_hits(
    t_spec: np.ndarray,
    excess_broadband: np.ndarray,
    *,
    threshold_sigma: float,
    min_hit_separation_s: float,
    max_hits: int | None,
) -> tuple[np.ndarray, float, float, float]:
    if t_spec.size < 2:
        raise ValueError("Need at least two sliding-spectrum time samples to detect hits")

    dt_spec = float(np.median(np.diff(t_spec)))
    min_distance = max(1, int(round(float(min_hit_separation_s) / dt_spec)))
    threshold, median, robust_std = _robust_threshold(excess_broadband, threshold_sigma)
    prominence = max(robust_std, 1e-12)
    peaks, _ = find_peaks(
        excess_broadband,
        height=threshold,
        distance=min_distance,
        prominence=prominence,
    )
    if max_hits is not None and peaks.size > int(max_hits):
        order = np.argsort(excess_broadband[peaks])[::-1][: int(max_hits)]
        peaks = np.sort(peaks[order])
    return peaks, threshold, median, robust_std


def _summarize_hits(
    t_spec: np.ndarray,
    broadband: np.ndarray,
    excess_broadband: np.ndarray,
    peaks: np.ndarray,
    *,
    strength_window_s: float,
) -> list[HitRecord]:
    out: list[HitRecord] = []
    if t_spec.size < 2:
        return out
    dt_spec = float(np.median(np.diff(t_spec)))

    for display_idx, peak_idx in enumerate(peaks, start=1):
        t0 = float(t_spec[peak_idx])
        mask = (t_spec >= t0) & (t_spec <= (t0 + float(strength_window_s)))
        if not np.any(mask):
            strength_peak = float(excess_broadband[peak_idx])
            strength_integral = 0.0
        else:
            strength_peak = float(np.max(excess_broadband[mask]))
            strength_integral = float(np.trapz(excess_broadband[mask], t_spec[mask]))
        out.append(
            HitRecord(
                index=display_idx,
                time_s=t0,
                strength_peak=strength_peak,
                strength_integral=strength_integral,
                broadband_at_peak=float(broadband[peak_idx]),
            )
        )
    return out


def _save_hits_csv(path: Path, hits: list[HitRecord], *, dataset: str, component: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["dataset", "component", "hit_index", "time_s", "strength_peak", "strength_integral", "broadband_at_peak"]
        )
        for hit in hits:
            writer.writerow(
                [
                    str(dataset),
                    str(component),
                    int(hit.index),
                    f"{hit.time_s:.9f}",
                    f"{hit.strength_peak:.9f}",
                    f"{hit.strength_integral:.9f}",
                    f"{hit.broadband_at_peak:.9f}",
                ]
            )


def main() -> int:
    args = build_parser().parse_args()

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
    proc_msgs: list[str] = []
    for bond_idx in range(bond_dataset.signal_matrix.shape[1]):
        processed, error = preprocess_signal(repaired_t, bond_dataset.signal_matrix[:, bond_idx])
        if processed is None:
            print(f"Skipping bond {bond_idx + 1}: {error}", file=sys.stderr)
            continue
        if processed_t is None:
            processed_t = processed.t
        processed_signals.append(np.asarray(processed.y, dtype=float))
        proc_msgs.append(processed.proc_msg)

    if processed_t is None or not processed_signals:
        raise ValueError("No bond signals were successfully preprocessed")

    aggregate_signal = _aggregate_for_display(processed_signals)
    fs = 1.0 / float(np.median(np.diff(processed_t)))

    t_spec_rel, f_spec, broadband_raw = _build_broadband_trace(
        processed_signals,
        fs,
        args.sliding_len_s,
        args.broadband_min_hz,
        args.broadband_max_hz,
    )
    t_spec = t_spec_rel + float(processed_t[0])
    smooth_samples = max(1, int(round(float(args.smooth_window_s) / float(np.median(np.diff(t_spec))))))
    baseline_samples = max(1, int(round(float(args.baseline_window_s) / float(np.median(np.diff(t_spec))))))
    broadband = _moving_average(broadband_raw, smooth_samples)
    broadband_baseline = _moving_average(broadband, baseline_samples)
    excess_broadband = np.maximum(0.0, broadband - broadband_baseline)
    hit_indices, threshold, bg_median, bg_std = _detect_hits(
        t_spec,
        excess_broadband,
        threshold_sigma=args.threshold_sigma,
        min_hit_separation_s=args.min_hit_separation_s,
        max_hits=args.max_hits,
    )
    hits = _summarize_hits(
        t_spec,
        broadband,
        excess_broadband,
        hit_indices,
        strength_window_s=args.strength_window_s,
    )

    aggregate_spec = compute_complex_spectrogram(aggregate_signal, fs, args.sliding_len_s)
    if aggregate_spec is None:
        raise ValueError("Could not build aggregate sliding FFT")

    t_edges = np.empty(aggregate_spec.t.size + 1, dtype=float)
    t_step = float(np.median(np.diff(aggregate_spec.t))) if aggregate_spec.t.size > 1 else args.sliding_len_s
    t_centers = aggregate_spec.t + float(processed_t[0])
    t_edges[1:-1] = 0.5 * (t_centers[:-1] + t_centers[1:])
    t_edges[0] = t_centers[0] - 0.5 * t_step
    t_edges[-1] = t_centers[-1] + 0.5 * t_step

    f_edges = np.empty(aggregate_spec.f.size + 1, dtype=float)
    f_step = float(np.median(np.diff(aggregate_spec.f))) if aggregate_spec.f.size > 1 else 1.0
    f_edges[1:-1] = 0.5 * (aggregate_spec.f[:-1] + aggregate_spec.f[1:])
    f_edges[0] = aggregate_spec.f[0] - 0.5 * f_step
    f_edges[-1] = aggregate_spec.f[-1] + 0.5 * f_step

    s_db = 20.0 * np.log10(np.abs(aggregate_spec.S_complex) + np.finfo(float).eps)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        constrained_layout=True,
        sharex=False,
    )

    ax0, ax1, ax2 = axes
    ax0.plot(processed_t, aggregate_signal, color="black", linewidth=0.8)
    for hit in hits:
        ax0.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.75)
        ax0.axvspan(hit.time_s, hit.time_s + args.strength_window_s, color="crimson", alpha=0.08)
        ax0.text(hit.time_s, ax0.get_ylim()[1] if np.isfinite(ax0.get_ylim()[1]) else 0.0, f"H{hit.index}",
                 color="crimson", fontsize=8, ha="left", va="top")
    ax0.set_title(
        f"{args.dataset} {args.component.upper()} aggregate bond signal | "
        f"bonds={len(processed_signals)} | repaired_time={'yes' if repaired_flag else 'no'}"
    )
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.25)

    ax1.plot(t_spec, broadband_raw, color="0.8", linewidth=0.8, label="raw broadband")
    ax1.plot(t_spec, broadband, color="navy", linewidth=1.3, label="smoothed broadband")
    ax1.plot(t_spec, broadband_baseline, color="0.25", linestyle="--", linewidth=1.1, label="slow baseline")
    ax1.plot(t_spec, excess_broadband, color="darkorange", linewidth=1.1, label="baseline-subtracted")
    ax1.axhline(bg_median, color="0.5", linestyle=":", linewidth=1.0, label="excess median")
    ax1.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2, label="excess threshold")
    if hit_indices.size > 0:
        ax1.scatter(
            t_spec[hit_indices],
            excess_broadband[hit_indices],
            color="crimson",
            s=24,
            zorder=3,
            label="hits",
        )
    for hit in hits:
        ax1.axvspan(hit.time_s, hit.time_s + args.strength_window_s, color="crimson", alpha=0.08)
        ax1.text(
            hit.time_s,
            hit.broadband_at_peak,
            f"H{hit.index}\nP={hit.strength_peak:.3g}\nI={hit.strength_integral:.3g}",
            color="crimson",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax1.set_title(
        f"Broadband hit-energy trace | band=[{args.broadband_min_hz:.1f}, {args.broadband_max_hz:.1f}] Hz | "
        f"sigma={args.threshold_sigma:.2f} | baseline={args.baseline_window_s:.1f} s | excess robust_std={bg_std:.3g}"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Broadband amplitude")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    pcm = ax2.pcolormesh(t_edges, f_edges, s_db, shading="flat", cmap="viridis")
    fig.colorbar(pcm, ax=ax2, label="Amplitude (dB)")
    for hit in hits:
        ax2.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
        ax2.axvspan(hit.time_s, hit.time_s + args.strength_window_s, color="crimson", alpha=0.08)
    ax2.set_ylim(0.0, min(args.broadband_max_hz * 1.2, 0.5 * fs))
    ax2.set_title(f"Aggregate sliding FFT | window={args.sliding_len_s:.2f} s")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    print(f"Dataset: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Bond spacing mode: {args.bond_spacing_mode}")
    print(f"Bonds used: {len(processed_signals)}")
    print(f"Processed duration: {processed_t[0]:.3f} s to {processed_t[-1]:.3f} s")
    print(f"Estimated sample rate: {fs:.6f} Hz")
    print(f"Frame-time repaired: {'yes' if repaired_flag else 'no'}")
    print(f"Repaired dt from frame numbers: {repaired_dt:.9f} s")
    print(f"Preprocess modes: {sorted(set(proc_msgs))}")
    print(f"Broadband band: [{args.broadband_min_hz:.3f}, {args.broadband_max_hz:.3f}] Hz")
    print(f"Detected hits: {len(hits)}")
    for hit in hits:
        print(
            f"  Hit {hit.index:02d}: t={hit.time_s:9.3f} s | "
            f"excess_peak={hit.strength_peak:.6g} | excess_integral={hit.strength_integral:.6g} | "
            f"raw_peak={hit.broadband_at_peak:.6g}"
        )

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.component}__prototype_hits"
        fig_path = save_dir / f"{stem}.png"
        csv_path = save_dir / f"{stem}.csv"
        fig.savefig(fig_path, dpi=160)
        _save_hits_csv(csv_path, hits, dataset=args.dataset, component=args.component)
        print(f"Saved figure: {fig_path}")
        print(f"Saved hits CSV: {csv_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
