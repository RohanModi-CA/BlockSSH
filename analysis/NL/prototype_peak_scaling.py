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
    mean_abs: float
    prev_gap_s: float | None
    next_gap_s: float | None
    window_start_s: float
    window_stop_s: float
    freq: np.ndarray
    amplitude: np.ndarray
    peak_amplitudes: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype per-hit peak scaling analysis using a hit CSV and a peaks CSV.",
    )
    parser.add_argument("dataset", help="Dataset stem, e.g. IMG_0681_rot270")
    parser.add_argument("hit_csv", help="CSV from prototype_hits.py")
    parser.add_argument("peaks", help="Peaks name or CSV path")
    parser.add_argument("--component", default="x", choices=["x", "y", "a"])
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--bond-spacing-mode", default="default", choices=["default", "comoving"])
    parser.add_argument("--window-s", type=float, default=4.0, help="Post-hit FFT window length in seconds.")
    parser.add_argument("--window-offset-s", type=float, default=3.0, help="Delay after hit before FFT window starts. Default: 3")
    parser.add_argument("--isolation-window-s", type=float, default=20.0, help="Reject hits with a neighbor closer than this. Default: 20")
    parser.add_argument("--df", type=float, default=0.05, help="Half-width in Hz used for peak integration. Default: 0.05")
    parser.add_argument("--max-hits", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--use-meanabs", action="store_true", help="Plot against local mean|x| of the response window instead of hit strength.")
    parser.add_argument("--flatten", action="store_true", help="Apply transfer function flattening to the hit spectra.")
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
        raise ValueError(f"No hits were loaded from {path}")
    return out


def _aggregate_for_display(processed_signals: list[np.ndarray]) -> np.ndarray:
    stack = np.vstack([np.asarray(y, dtype=float) for y in processed_signals])
    return np.mean(stack, axis=0)


def _nearest_neighbor_gaps(hit_times: np.ndarray, idx: int) -> tuple[float | None, float | None]:
    prev_gap = None if idx == 0 else float(hit_times[idx] - hit_times[idx - 1])
    next_gap = None if idx == (hit_times.size - 1) else float(hit_times[idx + 1] - hit_times[idx])
    return prev_gap, next_gap


def _slice_uniform_window(t: np.ndarray, y: np.ndarray, start_s: float, stop_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    mask = (t >= float(start_s)) & (t <= float(stop_s))
    if np.count_nonzero(mask) < 16:
        return None
    return np.asarray(t[mask], dtype=float), np.asarray(y[mask], dtype=float)


def _integrated_peak_amplitude(freq: np.ndarray, amp: np.ndarray, peak_hz: float, df: float) -> float:
    freq = np.asarray(freq, dtype=float)
    amp = np.asarray(amp, dtype=float)
    low = float(peak_hz - df)
    high = float(peak_hz + df)
    if freq.size < 2 or high <= low:
        return 0.0
    if high < float(freq[0]) or low > float(freq[-1]):
        return 0.0

    low = max(low, float(freq[0]))
    high = min(high, float(freq[-1]))
    if high <= low:
        return 0.0

    interior_mask = (freq > low) & (freq < high)
    local_freq = freq[interior_mask]
    local_amp = amp[interior_mask]
    edge_freq = np.array([low, high], dtype=float)
    edge_amp = np.interp(edge_freq, freq, amp)

    if local_freq.size == 0:
        window_freq = edge_freq
        window_amp = edge_amp
    else:
        window_freq = np.concatenate(([low], local_freq, [high]))
        window_amp = np.concatenate(([edge_amp[0]], local_amp, [edge_amp[1]]))

    power = np.square(np.abs(window_amp))
    integral = float(np.trapz(power, window_freq))
    if not np.isfinite(integral) or integral <= 0.0:
        return 0.0
    return float(np.sqrt(integral))


def _accept_hits(
    hits: list[HitInput],
    *,
    processed_t: np.ndarray,
    aggregate_signal: np.ndarray,
    window_s: float,
    window_offset_s: float,
    isolation_window_s: float,
    max_hits: int | None,
    peaks: list[float],
    df: float,
    flatten: bool = False,
) -> list[AcceptedHit]:
    hit_times = np.asarray([hit.time_s for hit in hits], dtype=float)
    accepted: list[AcceptedHit] = []

    for idx, hit in enumerate(hits):
        prev_gap, next_gap = _nearest_neighbor_gaps(hit_times, idx)
        nearest_gap = min(
            [gap for gap in (prev_gap, next_gap) if gap is not None],
            default=float("inf"),
        )
        if nearest_gap < float(isolation_window_s):
            continue

        start_s = float(hit.time_s + window_offset_s)
        stop_s = float(start_s + window_s)
        if start_s < float(processed_t[0]) or stop_s > float(processed_t[-1]):
            continue

        sliced = _slice_uniform_window(processed_t, aggregate_signal, start_s, stop_s)
        if sliced is None:
            continue
        t_window, y_window = sliced
        
        # Calculate mean absolute amplitude after removing local mean
        y_centered = y_window - np.mean(y_window)
        mean_abs_val = float(np.mean(np.abs(y_centered)))
        
        dt = float(np.median(np.diff(t_window)))
        spectrum = compute_one_sided_fft(y_centered, dt)
        
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
        accepted.append(
            AcceptedHit(
                index=int(hit.index),
                time_s=float(hit.time_s),
                strength_peak=float(hit.strength_peak),
                strength_integral=float(hit.strength_integral),
                broadband_at_peak=float(hit.broadband_at_peak),
                mean_abs=mean_abs_val,
                prev_gap_s=prev_gap,
                next_gap_s=next_gap,
                window_start_s=start_s,
                window_stop_s=stop_s,
                freq=np.asarray(spectrum.freq, dtype=float),
                amplitude=np.asarray(amp, dtype=float),
                peak_amplitudes=peak_amplitudes,
            )
        )

    if max_hits is not None and len(accepted) > int(max_hits):
        accepted = accepted[: int(max_hits)]
    return accepted


def _save_scaling_csv(path: Path, accepted_hits: list[AcceptedHit], peaks: list[float]) -> None:
    fieldnames = [
        "hit_index",
        "time_s",
        "strength_peak",
        "strength_integral",
        "broadband_at_peak",
        "prev_gap_s",
        "next_gap_s",
        "window_start_s",
        "window_stop_s",
    ] + [f"peak_{peak_hz:.6f}_Hz" for peak_hz in peaks]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for hit in accepted_hits:
            row = {
                "hit_index": int(hit.index),
                "time_s": f"{hit.time_s:.9f}",
                "strength_peak": f"{hit.strength_peak:.9f}",
                "strength_integral": f"{hit.strength_integral:.9f}",
                "broadband_at_peak": f"{hit.broadband_at_peak:.9f}",
                "prev_gap_s": "" if hit.prev_gap_s is None else f"{hit.prev_gap_s:.9f}",
                "next_gap_s": "" if hit.next_gap_s is None else f"{hit.next_gap_s:.9f}",
                "window_start_s": f"{hit.window_start_s:.9f}",
                "window_stop_s": f"{hit.window_stop_s:.9f}",
            }
            for peak_hz, peak_amp in zip(peaks, hit.peak_amplitudes):
                row[f"peak_{peak_hz:.6f}_Hz"] = f"{float(peak_amp):.9f}"
            writer.writerow(row)


def main() -> int:
    args = build_parser().parse_args()
    isolation_window_s = float(args.isolation_window_s)
    if args.window_s <= 0:
        raise ValueError("--window-s must be > 0")
    if args.df <= 0:
        raise ValueError("--df must be > 0")

    hits = _load_hits_csv(args.hit_csv)
    hit_datasets = sorted({hit.dataset for hit in hits if hit.dataset})
    hit_components = sorted({hit.component for hit in hits if hit.component})
    if hit_datasets and hit_datasets != [args.dataset]:
        raise ValueError(
            f"Hit CSV dataset metadata {hit_datasets} does not match requested dataset '{args.dataset}'"
        )
    if hit_components and hit_components != [args.component]:
        raise ValueError(
            f"Hit CSV component metadata {hit_components} does not match requested component '{args.component}'"
        )
    if (not hit_datasets) and (args.dataset not in Path(args.hit_csv).stem):
        print(
            "Warning: hit CSV does not carry dataset metadata, and its filename does not mention the requested dataset. "
            "Make sure the hit CSV came from the same dataset.",
            file=sys.stderr,
        )
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
        raise ValueError("No bond signals were successfully preprocessed")

    aggregate_signal = _aggregate_for_display(processed_signals)
    accepted_hits = _accept_hits(
        hits,
        processed_t=processed_t,
        aggregate_signal=aggregate_signal,
        window_s=float(args.window_s),
        window_offset_s=float(args.window_offset_s),
        isolation_window_s=isolation_window_s,
        max_hits=args.max_hits,
        peaks=peaks,
        df=float(args.df),
        flatten=args.flatten,
    )
    rejected_count = len(hits) - len(accepted_hits)
    if not accepted_hits:
        raise ValueError("No hits survived the isolation/window checks")

    if args.use_meanabs:
        strength_values = np.asarray([hit.mean_abs for hit in accepted_hits], dtype=float)
        strength_label = "Hit local mean|x| amplitude"
    else:
        strength_values = np.asarray([hit.strength_integral for hit in accepted_hits], dtype=float)
        strength_label = "Hit strength integral"

    peak_matrix = np.vstack([hit.peak_amplitudes for hit in accepted_hits])

    n_peaks = len(peaks)
    ncols = 3
    nrows = 1 + int(np.ceil(n_peaks / ncols))
    fig = plt.figure(figsize=(17, 4.5 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(processed_t, aggregate_signal, color="black", linewidth=0.8)
    for hit in hits:
        ax0.axvline(hit.time_s, color="0.8", linestyle=":", linewidth=0.8)
    for hit in accepted_hits:
        ax0.axvspan(hit.window_start_s, hit.window_stop_s, color="tab:green", alpha=0.12)
        ax0.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax0.text(hit.time_s, ax0.get_ylim()[1], f"H{hit.index}", color="crimson", fontsize=8, ha="left", va="top")
    ax0.set_title(
        f"{args.dataset} {args.component.upper()} accepted per-hit FFT windows | "
        f"accepted={len(accepted_hits)} rejected={rejected_count} | window={args.window_s:.2f} s | df={args.df:.3f} Hz"
    )
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Aggregate amplitude")
    ax0.grid(True, alpha=0.25)

    ax1 = fig.add_subplot(gs[1, 0])
    order = np.argsort(strength_values)
    for rank, hit_idx in enumerate(order):
        hit = accepted_hits[int(hit_idx)]
        label = f"H{hit.index}" if rank < 10 else None
        ax1.plot(hit.freq, hit.amplitude, linewidth=1.0, alpha=0.65, label=label)
    for peak_hz in peaks:
        ax1.axvspan(peak_hz - args.df, peak_hz + args.df, color="crimson", alpha=0.08)
        ax1.axvline(peak_hz, color="crimson", linestyle=":", linewidth=0.9)
    ax1.set_xlim(0.0, max(float(max(peaks) + 2.0 * args.df), 1.0))
    ax1.set_title("Accepted hit spectra")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(strength_values, [hit.time_s for hit in accepted_hits], c=np.arange(len(accepted_hits)), cmap="viridis", s=35)
    ax2.set_title("Accepted hit strength vs time")
    ax2.set_xlabel(strength_label)
    ax2.set_ylabel("Hit time (s)")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 2])
    for peak_idx, peak_hz in enumerate(peaks):
        ax3.plot(
            strength_values,
            peak_matrix[:, peak_idx],
            "o",
            markersize=4,
            alpha=0.8,
            label=f"{peak_hz:.3f} Hz",
        )
    ax3.set_title("All peak amplitudes vs hit strength")
    ax3.set_xlabel(strength_label)
    ax3.set_ylabel("Peak amplitude")
    ax3.grid(True, alpha=0.25)
    if n_peaks <= 10:
        ax3.legend(fontsize=8)

    for peak_idx, peak_hz in enumerate(peaks):
        row = 2 + (peak_idx // ncols)
        col = peak_idx % ncols
        if row >= nrows:
            break
        ax = fig.add_subplot(gs[row, col])
        y = peak_matrix[:, peak_idx]
        ax.scatter(strength_values, y, color="black", s=28)
        for hit, x_val, y_val in zip(accepted_hits, strength_values, y):
            ax.text(x_val, y_val, str(hit.index), fontsize=7, color="crimson", ha="left", va="bottom")
        ax.set_title(f"{peak_hz:.3f} Hz")
        ax.set_xlabel("Hit strength integral")
        ax.set_ylabel("sqrt(int power)")
        ax.grid(True, alpha=0.25)

    print(f"Dataset: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Peaks file: {peaks_path}")
    print(f"Peaks (Hz): {peaks}")
    print(f"Window length: {args.window_s:.6g} s")
    print(f"Window offset: {args.window_offset_s:.6g} s")
    print(f"Isolation window: {isolation_window_s:.6g} s")
    print(f"Peak half-width df: {args.df:.6g} Hz")
    print(f"Frame-time repaired: {'yes' if repaired_flag else 'no'}")
    print(f"Repaired dt from frame numbers: {repaired_dt:.9f} s")
    print(f"Input hits: {len(hits)}")
    print(f"Accepted hits: {len(accepted_hits)}")
    print(f"Rejected hits: {rejected_count}")
    for hit in accepted_hits:
        pieces = ", ".join(
            f"{peak_hz:.3f}Hz={amp:.4g}" for peak_hz, amp in zip(peaks, hit.peak_amplitudes)
        )
        print(
            f"  Hit {hit.index:02d}: t={hit.time_s:9.3f} s | "
            f"strength_int={hit.strength_integral:.6g} | {pieces}"
        )

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.component}__peak_scaling"
        fig_path = save_dir / f"{stem}.png"
        csv_path = save_dir / f"{stem}.csv"
        fig.savefig(fig_path, dpi=160)
        _save_scaling_csv(csv_path, accepted_hits, peaks)
        print(f"Saved figure: {fig_path}")
        print(f"Saved scaling CSV: {csv_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
