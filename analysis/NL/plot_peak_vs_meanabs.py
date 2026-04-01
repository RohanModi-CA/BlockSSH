#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.NL.visualize_hit_strength_rms import (
    _aggregate,
    _build_rms_hits,
    _load_hits_csv,
    _repair_time_vector,
)
from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.peaks import load_peaks_csv, resolve_peaks_csv
from analysis.tools.signal import compute_complex_spectrogram, preprocess_signal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-hit peak amplitudes against local mean|x| strength using the current RMS-workflow defaults.",
    )
    parser.add_argument("dataset")
    parser.add_argument("hit_csv")
    parser.add_argument("peaks")
    parser.add_argument("--component", default="x", choices=["x", "y", "a"])
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--bond-spacing-mode", default="purecomoving", choices=["default", "purecomoving"])
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--window-offset-s", type=float, default=3.0)
    parser.add_argument("--baseline-before-s", type=float, default=2.0)
    parser.add_argument("--baseline-window-s", type=float, default=4.0)
    parser.add_argument("--isolation-window-s", type=float, default=20.0)
    parser.add_argument("--df", type=float, default=0.05)
    parser.add_argument("--sliding-len-s", type=float, default=1.5)
    parser.add_argument("--spectrogram-max-hz", type=float, default=25.0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--flatten", action="store_true", help="Apply transfer function flattening to hit spectra.")
    return parser


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
    aggregate_spec = compute_complex_spectrogram(
        aggregate_signal,
        1.0 / float(np.median(np.diff(processed_t))),
        float(args.sliding_len_s),
    )
    if aggregate_spec is None:
        raise ValueError("Could not build aggregate spectrogram")
    rms_hits = _build_rms_hits(
        hits,
        processed_t,
        aggregate_signal,
        peaks,
        window_s=float(args.window_s),
        window_offset_s=float(args.window_offset_s),
        baseline_before_s=float(args.baseline_before_s),
        baseline_window_s=float(args.baseline_window_s),
        isolation_window_s=float(args.isolation_window_s),
        df=float(args.df),
        flatten=args.flatten,
    )
    if not rms_hits:
        raise ValueError("No accepted hits survived the local mean|x| workflow")

    mean_abs = np.asarray([hit.mean_abs_strength for hit in rms_hits], dtype=float)
    peak_matrix = np.vstack([hit.peak_amplitudes for hit in rms_hits])
    hit_ids = [hit.index for hit in rms_hits]

    n_peaks = len(peaks)
    ncols = 3
    nrows = 3 + int(np.ceil(n_peaks / ncols))
    fig = plt.figure(figsize=(17, 4.4 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(processed_t, aggregate_signal, color="black", linewidth=0.8)
    for hit in rms_hits:
        ax0.axvspan(hit.baseline_start_s, hit.baseline_stop_s, color="tab:blue", alpha=0.10)
        ax0.axvspan(hit.analysis_start_s, hit.analysis_stop_s, color="tab:green", alpha=0.12)
        ax0.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
        ylim_top = ax0.get_ylim()[1]
        ax0.text(
            hit.time_s,
            ylim_top,
            str(hit.index),
            color="crimson",
            fontsize=8,
            ha="left",
            va="top",
        )
    ax0.set_title(
        f"{args.dataset} accepted local mean|x| windows | accepted={len(rms_hits)} | "
        f"response={args.window_s:.1f}s offset={args.window_offset_s:.1f}s"
    )
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.25)

    ax_spec = fig.add_subplot(gs[1, :])
    s_db = 20.0 * np.log10(np.abs(aggregate_spec.S_complex) + np.finfo(float).eps)
    t_centers = aggregate_spec.t + float(processed_t[0])
    if t_centers.size > 1:
        t_step = float(np.median(np.diff(t_centers)))
    else:
        t_step = float(args.sliding_len_s)
    if aggregate_spec.f.size > 1:
        f_step = float(np.median(np.diff(aggregate_spec.f)))
    else:
        f_step = 1.0
    t_edges = np.empty(t_centers.size + 1, dtype=float)
    t_edges[1:-1] = 0.5 * (t_centers[:-1] + t_centers[1:])
    t_edges[0] = t_centers[0] - 0.5 * t_step
    t_edges[-1] = t_centers[-1] + 0.5 * t_step
    f_edges = np.empty(aggregate_spec.f.size + 1, dtype=float)
    f_edges[1:-1] = 0.5 * (aggregate_spec.f[:-1] + aggregate_spec.f[1:])
    f_edges[0] = aggregate_spec.f[0] - 0.5 * f_step
    f_edges[-1] = aggregate_spec.f[-1] + 0.5 * f_step
    pcm = ax_spec.pcolormesh(t_edges, f_edges, s_db, shading="flat", cmap="viridis")
    fig.colorbar(pcm, ax=ax_spec, label="Amplitude (dB)")
    for hit in rms_hits:
        ax_spec.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_spec.text(
            hit.time_s,
            float(args.spectrogram_max_hz),
            str(hit.index),
            color="crimson",
            fontsize=8,
            ha="left",
            va="top",
        )
    ax_spec.set_ylim(0.0, float(args.spectrogram_max_hz))
    ax_spec.set_title("Aggregate spectrogram with accepted hit labels")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")

    ax1 = fig.add_subplot(gs[2, 0])
    for peak_idx, peak_hz in enumerate(peaks):
        ax1.plot(mean_abs, peak_matrix[:, peak_idx], "o", markersize=4, alpha=0.8, label=f"{peak_hz:.3f} Hz")
    ax1.set_title("All peak amplitudes vs mean|x|")
    ax1.set_xlabel("mean|x| strength")
    ax1.set_ylabel("sqrt(int power)")
    ax1.grid(True, alpha=0.25)
    if len(peaks) <= 10:
        ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[2, 1])
    eps = 1e-12
    for peak_idx, peak_hz in enumerate(peaks):
        x = np.maximum(mean_abs, eps)
        y = np.maximum(peak_matrix[:, peak_idx], eps)
        ax2.plot(x, y, "o", markersize=4, alpha=0.8, label=f"{peak_hz:.3f} Hz")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("All peak amplitudes vs mean|x| (log-log)")
    ax2.set_xlabel("mean|x| strength")
    ax2.set_ylabel("sqrt(int power)")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 2])
    norm_peak_matrix = peak_matrix.copy()
    for col in range(norm_peak_matrix.shape[1]):
        vmax = float(np.max(norm_peak_matrix[:, col]))
        if vmax > 0:
            norm_peak_matrix[:, col] /= vmax
    im = ax3.imshow(norm_peak_matrix.T, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
    fig.colorbar(im, ax=ax3, label="peak amplitude / peak max")
    ax3.set_yticks(np.arange(len(peaks)))
    ax3.set_yticklabels([f"{peak:.3f}" for peak in peaks], fontsize=8)
    ax3.set_xticks(np.arange(len(rms_hits)))
    ax3.set_xticklabels([str(hit_id) for hit_id in hit_ids], fontsize=8)
    ax3.set_xlabel("Accepted hit index")
    ax3.set_ylabel("Peak frequency (Hz)")
    ax3.set_title("Per-hit peak amplitudes (column-normalized)")

    for peak_idx, peak_hz in enumerate(peaks):
        row = 3 + (peak_idx // ncols)
        col = peak_idx % ncols
        ax = fig.add_subplot(gs[row, col])
        y = peak_matrix[:, peak_idx]
        ax.scatter(mean_abs, y, color="black", s=28)
        for hit, x_val, y_val in zip(rms_hits, mean_abs, y):
            ax.text(x_val, y_val, str(hit.index), fontsize=7, color="crimson", ha="left", va="bottom")
        ax.set_title(f"{peak_hz:.3f} Hz")
        ax.set_xlabel("mean|x| strength")
        ax.set_ylabel("sqrt(int power)")
        ax.grid(True, alpha=0.25)

    print(f"Dataset: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Peaks file: {peaks_path}")
    print(f"Frame-time repaired: {'yes' if repaired_flag else 'no'}")
    print(f"Repaired dt from frame numbers: {repaired_dt:.9f} s")
    print(f"Accepted hits: {len(rms_hits)}")
    print("Strength axis: mean|x| of the centered 30 s response window")
    print(f"Hit ids: {hit_ids}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_path = save_dir / f"{args.dataset}__{args.component}__peak_vs_meanabs.png"
        fig.savefig(fig_path, dpi=160)
        print(f"Saved figure: {fig_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
