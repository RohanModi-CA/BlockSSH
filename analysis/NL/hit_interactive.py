#!/usr/bin/env python3
"""Interactive hit detection, editing, and FFT analysis.

Workflow:
1. Auto-detect hits from broadband energy trace (same logic as prototype_hits.py)
2. Display hits on timeseries + broadband panels — click to add, right-click to delete
3. Press Enter to accept hits and compute FFTs
4. FFT panel shows inter-hit and hit-region spectra (all bonds, averaged)

Usage:
    python3 analysis/NL/hit_interactive.py DATASET
    python3 analysis/NL/hit_interactive.py DATASET --component x --delay 1.0 --exclude-before 1.0
    python3 analysis/NL/hit_interactive.py DATASET --mode psd --hit-window 5.0
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.io import load_track2_dataset
from analysis.tools.signal import compute_complex_spectrogram, compute_one_sided_fft, preprocess_signal


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HitRecord:
    index: int
    time_s: float
    strength_peak: float
    strength_integral: float
    broadband_at_peak: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive hit detection, editing, and FFT analysis.",
    )
    p.add_argument("dataset", help="Dataset stem, e.g. 11triv or IMG_0681_rot270")
    p.add_argument("--component", default="x", choices=["x", "y", "a"])
    p.add_argument("--track-data-root", default=None)
    p.add_argument("--bond-spacing-mode", default="purecomoving", choices=["default", "purecomoving"],
                   help="Bond spacing mode: 'purecomoving' (needs x+y) or 'default' (x-only, no rotation correction)")

    # Detection params
    p.add_argument("--sliding-len-s", type=float, default=1.5)
    p.add_argument("--broadband-min-hz", type=float, default=3.0)
    p.add_argument("--broadband-max-hz", type=float, default=25.0)
    p.add_argument("--smooth-window-s", type=float, default=0.35)
    p.add_argument("--baseline-window-s", type=float, default=18.0)
    p.add_argument("--threshold-sigma", type=float, default=3.0)
    p.add_argument("--min-hit-separation-s", type=float, default=6.0)
    p.add_argument("--strength-window-s", type=float, default=2.5)
    p.add_argument("--max-hits", type=int, default=None)

    # FFT analysis params
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds to exclude after each hit for inter-hit regions")
    p.add_argument("--exclude-before", type=float, default=1.0,
                   help="Seconds to exclude before next hit for inter-hit regions")
    p.add_argument("--hit-window", type=float, default=5.0,
                   help="Duration of hit region to analyze after each hit")
    p.add_argument("--mode", default="fft", choices=["fft", "psd"],
                   help="Spectrum type: fft (amplitude) or psd (power)")
    p.add_argument("--no-auto-detect", action="store_true",
                   help="Skip auto-detection, start with existing CSV or empty")
    p.add_argument("--no-load-hits", action="store_true",
                   help="Don't load existing hits CSV even if present")

    p.add_argument("--save", default=None, help="Output plot path for FFT panel")
    p.add_argument("--no-show", action="store_true")
    return p


def _repair_time_vector(frame_times_s, frame_numbers):
    t = np.asarray(frame_times_s, dtype=float)
    frame_numbers = np.asarray(frame_numbers, dtype=float)
    positive_dt = np.diff(t)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if positive_dt.size == 0:
        raise ValueError("Could not infer a positive frame interval")
    dt = float(np.median(positive_dt))
    if not np.all(np.isfinite(frame_numbers)):
        return np.arange(t.size, dtype=float) * dt, dt, True
    rebuilt = (frame_numbers - float(frame_numbers[0])) * dt
    nonmonotone_fraction = float(np.mean(np.diff(t) <= 0)) if t.size > 1 else 0.0
    invalid_tail = bool(np.any(t[-max(1, min(32, t.size)):] == 0.0))
    clearly_bad = nonmonotone_fraction > 0.001 or invalid_tail or (not np.all(np.isfinite(t)))
    return (rebuilt if clearly_bad else t), dt, clearly_bad


def _moving_average(y, window_samples):
    arr = np.asarray(y, dtype=float)
    if window_samples <= 1 or arr.size == 0:
        return arr.copy()
    kernel = np.ones(int(window_samples), dtype=float) / float(window_samples)
    return np.convolve(arr, kernel, mode="same")


def _robust_threshold(y, sigma):
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot build threshold from all-NaN trace")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_std = 1.4826 * mad
    if robust_std <= 0 or not np.isfinite(robust_std):
        robust_std = float(np.std(finite))
    threshold = median + float(sigma) * robust_std
    return threshold, median, robust_std


def _build_broadband_trace(processed_signals, fs, sliding_len_s, fmin, fmax):
    spec_results = []
    for y in processed_signals:
        result = compute_complex_spectrogram(y, fs, sliding_len_s)
        if result is None:
            raise ValueError("Sliding spectrogram window is too short")
        spec_results.append(result)
    ref = spec_results[0]
    freq_mask = (ref.f >= float(fmin)) & (ref.f <= float(fmax))
    if not np.any(freq_mask):
        raise ValueError(f"No spectrogram bins in [{fmin:.3g}, {fmax:.3g}] Hz")
    power_traces = []
    for result in spec_results:
        band_power = np.mean(np.abs(result.S_complex[freq_mask, :]) ** 2, axis=0)
        power_traces.append(band_power)
    broadband = np.sqrt(np.mean(np.vstack(power_traces), axis=0))
    return ref.t, ref.f, broadband


def _detect_hits(t_spec, excess_broadband, threshold_sigma, min_hit_separation_s, max_hits):
    if t_spec.size < 2:
        raise ValueError("Need at least two time samples to detect hits")
    dt_spec = float(np.median(np.diff(t_spec)))
    min_distance = max(1, int(round(float(min_hit_separation_s) / dt_spec)))
    threshold, median, robust_std = _robust_threshold(excess_broadband, threshold_sigma)
    prominence = max(robust_std, 1e-12)
    peaks, _ = find_peaks(excess_broadband, height=threshold, distance=min_distance, prominence=prominence)
    if max_hits is not None and peaks.size > int(max_hits):
        order = np.argsort(excess_broadband[peaks])[::-1][: int(max_hits)]
        peaks = np.sort(peaks[order])
    return peaks, threshold, median, robust_std


def _summarize_hits(t_spec, broadband, excess_broadband, peaks, strength_window_s):
    out = []
    if t_spec.size < 2:
        return out
    for display_idx, peak_idx in enumerate(peaks, start=1):
        t0 = float(t_spec[peak_idx])
        mask = (t_spec >= t0) & (t_spec <= (t0 + float(strength_window_s)))
        if not np.any(mask):
            strength_peak = float(excess_broadband[peak_idx])
            strength_integral = 0.0
        else:
            strength_peak = float(np.max(excess_broadband[mask]))
            strength_integral = float(np.trapz(excess_broadband[mask], t_spec[mask]))
        out.append(HitRecord(
            index=display_idx, time_s=t0, strength_peak=strength_peak,
            strength_integral=strength_integral, broadband_at_peak=float(broadband[peak_idx]),
        ))
    return out


def _hits_csv_path(dataset, component):
    return REPO_ROOT / "analysis/NL/out" / f"{dataset}_comparison_purecomoving" / f"{dataset}__{component}__prototype_hits.csv"


def _load_hits_csv(path):
    times = []
    if not path.exists():
        return times
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            times.append(float(row["time_s"]))
    return times


def _save_hits_csv(path, hit_times, dataset, component):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "component", "hit_index", "time_s", "strength_peak", "strength_integral", "broadband_at_peak"])
        for i, t in enumerate(hit_times, start=1):
            writer.writerow([str(dataset), str(component), i, f"{t:.9f}", "0.0", "0.0", "0.0"])


def _inter_hit_regions(hit_times, t_end, delay, exclude_before):
    regions = []
    for i in range(len(hit_times)):
        start = hit_times[i] + delay
        end = hit_times[i + 1] - exclude_before if i < len(hit_times) - 1 else t_end - 1.0
        if end > start + 0.5:
            regions.append((start, end))
    return regions


def _hit_regions(hit_times, t_end, hit_window):
    regions = []
    for ht in hit_times:
        end = min(ht + hit_window, t_end)
        if end > ht + 0.5:
            regions.append((ht, end))
    return regions


def _compute_region_spectrum(pt, sig, fs, start, end, mode):
    mask = (pt >= start) & (pt <= end)
    seg = sig[mask]
    if seg.size < 16:
        return None, None
    dt = 1.0 / fs
    if mode == "psd":
        fft_res = compute_one_sided_fft(seg, dt)
        return fft_res.freq, fft_res.amplitude ** 2
    else:
        fft_res = compute_one_sided_fft(seg, dt)
        return fft_res.freq, fft_res.amplitude


def _average_spectra(pt, sig_matrix, fs, regions, mode):
    if not regions:
        return None, None
    n_bonds = sig_matrix.shape[1]
    all_spectra = []
    common_freq = None
    for start, end in regions:
        bond_spectra = []
        for b in range(n_bonds):
            mask = (pt >= start) & (pt <= end)
            seg = sig_matrix[mask, b]
            if seg.size < 16:
                continue
            freq, amp = _compute_region_spectrum(pt, sig_matrix[:, b], fs, start, end, mode)
            if freq is not None:
                bond_spectra.append((freq, amp))
        if not bond_spectra:
            continue
        if common_freq is None:
            common_freq = bond_spectra[0][0]
        avg_bond = np.mean([np.interp(common_freq, f, a) for f, a in bond_spectra], axis=0)
        all_spectra.append(avg_bond)
    if not all_spectra:
        return None, None
    avg = np.mean(all_spectra, axis=0)
    return common_freq, avg


class HitEditor:
    def __init__(self, args, processed_t, aggregate_signal, t_spec, f_spec,
                 broadband_raw, broadband, broadband_baseline, excess_broadband,
                 hit_times, threshold, bg_median, bg_std,
                 pt, sig_matrix, fs,
                 delay, exclude_before, hit_window, mode):
        self.args = args
        self.processed_t = processed_t
        self.aggregate_signal = aggregate_signal
        self.t_spec = t_spec
        self.f_spec = f_spec
        self.broadband_raw = broadband_raw
        self.broadband = broadband
        self.broadband_baseline = broadband_baseline
        self.excess_broadband = excess_broadband
        self.hit_times = sorted(hit_times)
        self.threshold = threshold
        self.bg_median = bg_median
        self.bg_std = bg_std
        self.pt = pt
        self.sig_matrix = sig_matrix
        self.fs = fs
        self.delay = delay
        self.exclude_before = exclude_before
        self.hit_window = hit_window
        self.mode = mode
        self.accepted = False
        self._hit_lines_ts = []
        self._hit_lines_bb = []
        self._hit_lines_sp = []
        self._hit_spans_ts = []
        self._hit_spans_bb = []
        self._hit_spans_sp = []
        self._interhit_spans = []
        self._delete_mode = False

    def _clear_hit_artists(self):
        for l in self._hit_lines_ts:
            l.remove()
        for l in self._hit_lines_bb:
            l.remove()
        for l in self._hit_lines_sp:
            l.remove()
        for s in self._hit_spans_ts:
            s.remove()
        for s in self._hit_spans_bb:
            s.remove()
        for s in self._hit_spans_sp:
            s.remove()
        for s in self._interhit_spans:
            s.remove()
        self._hit_lines_ts = []
        self._hit_lines_bb = []
        self._hit_lines_sp = []
        self._hit_spans_ts = []
        self._hit_spans_bb = []
        self._hit_spans_sp = []
        self._interhit_spans = []

    def _draw_hits(self, ax0, ax1, ax2):
        self._clear_hit_artists()
        strength_window = self.args.strength_window_s
        for ht in self.hit_times:
            l0 = ax0.axvline(ht, color="crimson", linestyle="--", linewidth=1.0, alpha=0.75)
            s0 = ax0.axvspan(ht, ht + strength_window, color="crimson", alpha=0.08)
            self._hit_lines_ts.append(l0)
            self._hit_spans_ts.append(s0)

            l1 = ax1.axvline(ht, color="crimson", linestyle="--", linewidth=1.0, alpha=0.75)
            s1 = ax1.axvspan(ht, ht + strength_window, color="crimson", alpha=0.08)
            self._hit_lines_bb.append(l1)
            self._hit_spans_bb.append(s1)

            l2 = ax2.axvline(ht, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
            s2 = ax2.axvspan(ht, ht + strength_window, color="crimson", alpha=0.08)
            self._hit_lines_sp.append(l2)
            self._hit_spans_sp.append(s2)

        regions = _inter_hit_regions(np.array(self.hit_times), self.pt[-1], self.delay, self.exclude_before)
        for start, end in regions:
            s = ax0.axvspan(start, end, color="tab:green", alpha=0.12)
            self._interhit_spans.append(s)

    def _find_nearest_hit(self, x_time, radius=2.0):
        best_idx = None
        best_dist = radius
        for i, ht in enumerate(self.hit_times):
            d = abs(ht - x_time)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _on_click(self, event):
        if event.inaxes is None:
            return
        if event.xdata is None:
            return
        x_time = event.xdata
        fig = event.inaxes.figure

        if event.button == 1:
            self.hit_times.append(x_time)
            self.hit_times.sort()
            print(f"  Added hit at t={x_time:.3f}s (total: {len(self.hit_times)})")
        elif event.button == 3:
            idx = self._find_nearest_hit(x_time)
            if idx is not None:
                removed = self.hit_times.pop(idx)
                print(f"  Removed hit at t={removed:.3f}s (total: {len(self.hit_times)})")
            else:
                print("  No hit near click to remove")
                return

        self._draw_hits(fig.axes[0], fig.axes[1], fig.axes[2])
        self._update_title(fig)
        fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "enter":
            self.accepted = True
            if event.inaxes is not None:
                plt.close(event.inaxes.figure)
            return
        if event.key in ("d", "D"):
            self._delete_mode = not self._delete_mode
            status = "ON — right-click near a hit to remove it" if self._delete_mode else "OFF"
            print(f"  Delete mode {status}")

    def _update_title(self, fig):
        ax0 = fig.axes[0]
        ax0.set_title(
            f"{self.args.dataset} {self.args.component.upper()} | "
            f"{len(self.hit_times)} hits | Click to add, right-click to remove | "
            f"{'DEL MODE' if self._delete_mode else 'ADD MODE'} | Press Enter to accept"
        )

    def run(self):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)

        ax0.plot(self.processed_t, self.aggregate_signal, color="black", linewidth=0.6)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Amplitude")
        ax0.grid(True, alpha=0.25)

        ax1.plot(self.t_spec, self.broadband_raw, color="0.8", linewidth=0.6, label="raw broadband")
        ax1.plot(self.t_spec, self.broadband, color="navy", linewidth=1.2, label="smoothed")
        ax1.plot(self.t_spec, self.broadband_baseline, color="0.25", linestyle="--", linewidth=1.0, label="baseline")
        ax1.plot(self.t_spec, self.excess_broadband, color="darkorange", linewidth=1.0, label="excess")
        ax1.axhline(self.bg_median, color="0.5", linestyle=":", linewidth=1.0, label="median")
        ax1.axhline(self.threshold, color="crimson", linestyle="--", linewidth=1.2, label="threshold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Broadband amplitude")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="upper right", fontsize=7)

        t_edges = np.empty(self.t_spec.size + 1, dtype=float)
        t_step = float(np.median(np.diff(self.t_spec))) if self.t_spec.size > 1 else self.args.sliding_len_s
        t_centers = self.t_spec + float(self.processed_t[0])
        t_edges[1:-1] = 0.5 * (t_centers[:-1] + t_centers[1:])
        t_edges[0] = t_centers[0] - 0.5 * t_step
        t_edges[-1] = t_centers[-1] + 0.5 * t_step

        spec = compute_complex_spectrogram(self.aggregate_signal, self.fs, self.args.sliding_len_s)
        if spec is not None:
            f_edges = np.empty(spec.f.size + 1, dtype=float)
            f_step = float(np.median(np.diff(spec.f))) if spec.f.size > 1 else 1.0
            f_edges[1:-1] = 0.5 * (spec.f[:-1] + spec.f[1:])
            f_edges[0] = spec.f[0] - 0.5 * f_step
            f_edges[-1] = spec.f[-1] + 0.5 * f_step
            s_db = 20.0 * np.log10(np.abs(spec.S_complex) + np.finfo(float).eps)
            pcm = ax2.pcolormesh(t_edges, f_edges, s_db, shading="flat", cmap="viridis")
            fig.colorbar(pcm, ax=ax2, label="Amplitude (dB)")
            ax2.set_ylim(0.0, min(self.args.broadband_max_hz * 1.2, 0.5 * self.fs))

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")

        self._draw_hits(ax0, ax1, ax2)
        self._update_title(fig)

        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        print(f"\n{'='*60}")
        print(f"Interactive Hit Editor — {self.args.dataset} {self.args.component.upper()}")
        print(f"{'='*60}")
        print(f"  Hits loaded: {len(self.hit_times)}")
        print(f"  Left-click on signal/broadband to ADD a hit")
        print(f"  Right-click near a hit to REMOVE it")
        print(f"  Press 'd' to toggle delete mode info")
        print(f"  Press ENTER to accept hits and compute FFTs")
        print(f"{'='*60}\n")

        if not self.args.no_show:
            plt.show()
        else:
            plt.close(fig)
            self.accepted = True

        return self.accepted


def run_fft_analysis(pt, sig_matrix, fs, hit_times, delay, exclude_before, hit_window, mode, dataset, component, args):
    hit_times_sorted = sorted(hit_times)
    t_end = pt[-1]

    interhit_regions = _inter_hit_regions(np.array(hit_times_sorted), t_end, delay, exclude_before)
    hit_regs = _hit_regions(np.array(hit_times_sorted), t_end, hit_window)

    print(f"\n{'='*60}")
    print(f"FFT Analysis — {dataset} {component.upper()}")
    print(f"{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Inter-hit regions: {len(interhit_regions)}")
    for i, (s, e) in enumerate(interhit_regions):
        print(f"    Region {i+1}: {s:.2f}s – {e:.2f}s ({e-s:.1f}s)")
    print(f"  Hit regions: {len(hit_regs)}")
    for i, (s, e) in enumerate(hit_regs):
        print(f"    Hit {i+1}: {s:.2f}s – {e:.2f}s ({e-s:.1f}s)")
    print(f"{'='*60}\n")

    inter_freq, inter_amp = _average_spectra(pt, sig_matrix, fs, interhit_regions, mode)
    hit_freq, hit_amp = _average_spectra(pt, sig_matrix, fs, hit_regs, mode)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    ax1.plot(pt, np.nanmean(sig_matrix, axis=1), color="black", lw=0.5, alpha=0.7)
    for start, end in interhit_regions:
        ax1.axvspan(start, end, color="tab:green", alpha=0.15)
    for start, end in hit_regs:
        ax1.axvspan(start, end, color="crimson", alpha=0.12)
    for ht in hit_times_sorted:
        ax1.axvline(ht, color="crimson", ls="--", lw=0.8, alpha=0.5)
    ax1.set_title(f"{dataset} {component.upper()} — Analysis Regions\nGreen = inter-hit (FFT'd), Red = hit regions (FFT'd)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.25)

    label = "Power Spectral Density" if mode == "psd" else "FFT Amplitude"
    if inter_freq is not None and inter_amp is not None:
        ax2.plot(inter_freq, inter_amp, color="tab:green", lw=1.5, label=f"Inter-hit ({len(interhit_regions)} regions)")
    if hit_freq is not None and hit_amp is not None:
        ax2.plot(hit_freq, hit_amp, color="crimson", lw=1.5, label=f"Hit regions ({len(hit_regs)} regions)")
    ax2.set_yscale("log")
    ax2.set_xlim(0, 30)
    ax2.set_title(f"{label} — Inter-hit vs Hit Regions")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(label)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend()

    out_path = args.save or f"analysis/NL/out/{dataset}_{component}_hit_interactive_{mode}.png"
    out_full = REPO_ROOT / out_path
    out_full.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_full, dpi=200)
    print(f"Saved FFT plot: {out_full}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()

    if not args.no_show:
        matplotlib.use("TkAgg")

    dataset_name = f"{args.dataset}_{args.component}"
    print(f"Loading {dataset_name} in {args.bond_spacing_mode} mode...")
    bond_dataset = load_bond_signal_dataset(
        dataset=dataset_name,
        track_data_root=args.track_data_root,
        bond_spacing_mode=args.bond_spacing_mode,
        component=args.component,
    )

    track2 = load_track2_dataset(dataset=dataset_name, track_data_root=args.track_data_root)
    repaired_t, repaired_dt, repaired_flag = _repair_time_vector(track2.frame_times_s, track2.frame_numbers)

    processed_signals = []
    processed_t = None
    for bond_idx in range(bond_dataset.signal_matrix.shape[1]):
        processed, error = preprocess_signal(repaired_t, bond_dataset.signal_matrix[:, bond_idx])
        if processed is None:
            print(f"  Skipping bond {bond_idx + 1}: {error}", file=sys.stderr)
            continue
        if processed_t is None:
            processed_t = processed.t
        processed_signals.append(np.asarray(processed.y, dtype=float))

    if processed_t is None or not processed_signals:
        raise ValueError("No bond signals were successfully preprocessed")

    aggregate_signal = np.mean(np.vstack([np.asarray(y, dtype=float) for y in processed_signals]), axis=0)
    fs = 1.0 / float(np.median(np.diff(processed_t)))

    sig_matrix = np.column_stack(processed_signals)

    t_spec_rel, f_spec, broadband_raw = _build_broadband_trace(
        processed_signals, fs, args.sliding_len_s, args.broadband_min_hz, args.broadband_max_hz,
    )
    t_spec = t_spec_rel + float(processed_t[0])
    smooth_samples = max(1, int(round(float(args.smooth_window_s) / float(np.median(np.diff(t_spec))))))
    baseline_samples = max(1, int(round(float(args.baseline_window_s) / float(np.median(np.diff(t_spec))))))
    broadband = _moving_average(broadband_raw, smooth_samples)
    broadband_baseline = _moving_average(broadband, baseline_samples)
    excess_broadband = np.maximum(0.0, broadband - broadband_baseline)

    hit_times = []

    if not args.no_load_hits:
        csv_path = _hits_csv_path(args.dataset, args.component)
        loaded = _load_hits_csv(csv_path)
        if loaded:
            print(f"Loaded {len(loaded)} hits from existing CSV: {csv_path}")
            hit_times.extend(loaded)

    if not args.no_auto_detect:
        hit_indices, threshold, bg_median, bg_std = _detect_hits(
            t_spec, excess_broadband,
            threshold_sigma=args.threshold_sigma,
            min_hit_separation_s=args.min_hit_separation_s,
            max_hits=args.max_hits,
        )
        hits = _summarize_hits(t_spec, broadband, excess_broadband, hit_indices, args.strength_window_s)
        auto_times = [h.time_s for h in hits]
        print(f"Auto-detected {len(auto_times)} hits")
        for h in hits:
            print(f"  Hit {h.index:02d}: t={h.time_s:9.3f}s | peak={h.strength_peak:.6g} | integral={h.strength_integral:.6g}")
        existing_set = set(round(t, 3) for t in hit_times)
        for t in auto_times:
            if round(t, 3) not in existing_set:
                hit_times.append(t)
        hit_times.sort()
        threshold_val = threshold
        bg_med = bg_median
        bg_s = bg_std
    else:
        threshold_val, bg_med, bg_s = _robust_threshold(excess_broadband, args.threshold_sigma)

    print(f"\nTotal hits for editing: {len(hit_times)}")

    editor = HitEditor(
        args=args,
        processed_t=processed_t,
        aggregate_signal=aggregate_signal,
        t_spec=t_spec,
        f_spec=f_spec,
        broadband_raw=broadband_raw,
        broadband=broadband,
        broadband_baseline=broadband_baseline,
        excess_broadband=excess_broadband,
        hit_times=hit_times,
        threshold=threshold_val,
        bg_median=bg_med,
        bg_std=bg_s,
        pt=processed_t,
        sig_matrix=sig_matrix,
        fs=fs,
        delay=args.delay,
        exclude_before=args.exclude_before,
        hit_window=args.hit_window,
        mode=args.mode,
    )

    accepted = editor.run()
    if not accepted:
        print("Cancelled.")
        return 0

    final_hits = sorted(editor.hit_times)
    csv_path = _hits_csv_path(args.dataset, args.component)
    _save_hits_csv(csv_path, final_hits, args.dataset, args.component)
    print(f"Saved {len(final_hits)} hits to {csv_path}")

    run_fft_analysis(
        pt=processed_t,
        sig_matrix=sig_matrix,
        fs=fs,
        hit_times=final_hits,
        delay=args.delay,
        exclude_before=args.exclude_before,
        hit_window=args.hit_window,
        mode=args.mode,
        dataset=args.dataset,
        component=args.component,
        args=args,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
