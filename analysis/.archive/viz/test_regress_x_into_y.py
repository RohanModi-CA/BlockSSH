#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.derived import derive_spacing_dataset
from tools.io import load_track2_dataset
from tools.signal import compute_complex_spectrogram, compute_one_sided_fft, preprocess_signal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether x->y leakage can be removed by time-domain regression "
            "y_clean = y - a*x, where a = (x·y)/(x·x)."
        )
    )

    parser.add_argument(
        "x_dataset",
        help="Dataset stem for the _x data, e.g. IMG_0662_rot90_x",
    )
    parser.add_argument(
        "y_dataset",
        help="Dataset stem for the _y data, e.g. IMG_0662_rot90_y",
    )
    parser.add_argument(
        "--track-data-root",
        default=None,
        help="Optional track/data root. Defaults to the repo's standard location.",
    )

    parser.add_argument(
        "--pair",
        type=int,
        nargs="+",
        default=None,
        help="Only analyze these 0-based pair indices. Default: all common pairs.",
    )

    parser.add_argument(
        "--longest",
        action="store_true",
        help="Use the longest contiguous valid segment instead of interpolating through gaps.",
    )
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help="Discard invalid time/value samples before uniform resampling.",
    )
    parser.add_argument(
        "--sliding-len-s",
        type=float,
        default=20.0,
        help="Sliding FFT window length in seconds. Default: 20.0",
    )
    parser.add_argument(
        "--peak-min-hz",
        type=float,
        default=2.0,
        help="Lower bound for x-dominant peak search. Default: 2.0",
    )
    parser.add_argument(
        "--peak-max-hz",
        type=float,
        default=20.0,
        help="Upper bound for x-dominant peak search. Default: 20.0",
    )
    parser.add_argument(
        "--fft-max-hz",
        type=float,
        default=23.0,
        help="Max frequency shown in FFT/spectrogram plots. Default: 23.0",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save per-pair plots instead of only showing them.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows. Useful with --save-dir.",
    )

    return parser


def align_processed_signals(
    t_x: np.ndarray,
    y_x: np.ndarray,
    dt_x: float,
    t_y: np.ndarray,
    y_y: np.ndarray,
    dt_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = max(float(t_x[0]), float(t_y[0]))
    stop = min(float(t_x[-1]), float(t_y[-1]))
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("No overlapping time range between processed x and y signals")

    dt = max(float(dt_x), float(dt_y))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid sample spacing after preprocessing")

    t = np.arange(start, stop + 0.5 * dt, dt, dtype=float)
    t = t[t <= stop + 1e-12]
    if t.size < 10:
        raise ValueError("Aligned overlap has too few samples")

    x = np.interp(t, t_x, y_x)
    y = np.interp(t, t_y, y_y)

    # Keep the aligned signals centered. They were already detrended upstream,
    # but this keeps the regression focused on shared oscillatory content.
    x = x - np.mean(x)
    y = y - np.mean(y)

    return t, x, y, dt


def regression_scale(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if not np.isfinite(denom) or denom <= 1e-15:
        raise ValueError("x signal has near-zero energy; cannot fit regression coefficient")
    return float(np.dot(x, y) / denom)


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def find_peak_in_band(freq: np.ndarray, amp: np.ndarray, fmin: float, fmax: float) -> tuple[float, float, int]:
    mask = (freq >= fmin) & (freq <= fmax)
    if not np.any(mask):
        raise ValueError(f"No FFT bins found in requested peak-search band [{fmin}, {fmax}] Hz")
    idx_local = int(np.argmax(amp[mask]))
    idx = int(np.where(mask)[0][idx_local])
    return float(freq[idx]), float(amp[idx]), idx


def amplitude_at_frequency(freq: np.ndarray, amp: np.ndarray, target_hz: float) -> tuple[float, float, int]:
    idx = int(np.argmin(np.abs(freq - target_hz)))
    return float(freq[idx]), float(amp[idx]), idx


def db_mag(s_complex: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    return 20.0 * np.log10(np.abs(s_complex) + eps)


def centers_to_edges(vals: np.ndarray, fallback_step: float = 1.0) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if vals.size == 1:
        half = 0.5 * fallback_step
        return np.array([vals[0] - half, vals[0] + half], dtype=float)

    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    return edges


def plot_spectrogram_panel(
    fig,
    ax,
    *,
    spec,
    t_start: float,
    fmax: float,
    title: str,
) -> None:
    if spec is None:
        ax.set_title(f"{title} | window too short")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        return

    s_plot = db_mag(spec.S_complex)
    t_global = spec.t + float(t_start)

    t_step = (t_global[1] - t_global[0]) if t_global.size > 1 else 1.0
    f_step = (spec.f[1] - spec.f[0]) if spec.f.size > 1 else 1.0

    t_edges = centers_to_edges(t_global, fallback_step=t_step)
    f_edges = centers_to_edges(spec.f, fallback_step=f_step)

    pcm = ax.pcolormesh(
        t_edges,
        f_edges,
        s_plot,
        shading="flat",
        cmap="turbo",
    )
    fig.colorbar(pcm, ax=ax, label="Amplitude (dB)")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0.0, fmax)


def analyze_one_pair(
    *,
    pair_idx: int,
    label_x: str,
    label_y: str,
    raw_t_x: np.ndarray,
    raw_x: np.ndarray,
    raw_t_y: np.ndarray,
    raw_y: np.ndarray,
    longest: bool,
    handlenan: bool,
    sliding_len_s: float,
    peak_min_hz: float,
    peak_max_hz: float,
    fft_max_hz: float,
):
    proc_x, err_x = preprocess_signal(
        raw_t_x,
        raw_x,
        longest=longest,
        handlenan=handlenan,
        min_samples=10,
    )
    if proc_x is None:
        raise ValueError(f"x preprocessing failed: {err_x}")

    proc_y, err_y = preprocess_signal(
        raw_t_y,
        raw_y,
        longest=longest,
        handlenan=handlenan,
        min_samples=10,
    )
    if proc_y is None:
        raise ValueError(f"y preprocessing failed: {err_y}")

    t, x, y, dt = align_processed_signals(
        proc_x.t, proc_x.y, proc_x.dt,
        proc_y.t, proc_y.y, proc_y.dt,
    )

    a = regression_scale(x, y)
    y_clean = y - a * x

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    fcap = min(float(fft_max_hz), float(nyquist))

    fft_x = compute_one_sided_fft(x, dt)
    fft_y = compute_one_sided_fft(y, dt)
    fft_clean = compute_one_sided_fft(y_clean, dt)

    peak_hz, peak_amp_x, _ = find_peak_in_band(
        fft_x.freq,
        fft_x.amplitude,
        peak_min_hz,
        min(peak_max_hz, nyquist),
    )
    sel_hz_y_before, amp_y_before, _ = amplitude_at_frequency(fft_y.freq, fft_y.amplitude, peak_hz)
    sel_hz_y_after, amp_y_after, _ = amplitude_at_frequency(fft_clean.freq, fft_clean.amplitude, peak_hz)

    spec_y = compute_complex_spectrogram(y, fs, sliding_len_s)
    spec_clean = compute_complex_spectrogram(y_clean, fs, sliding_len_s)

    corr_before = corrcoef_safe(x, y)
    corr_after = corrcoef_safe(x, y_clean)

    metrics = {
        "pair_idx": pair_idx,
        "label_x": label_x,
        "label_y": label_y,
        "n_samples": int(t.size),
        "dt": float(dt),
        "fs": float(fs),
        "nyquist": float(nyquist),
        "scale_a": float(a),
        "corr_before": corr_before,
        "corr_after": corr_after,
        "x_peak_hz": peak_hz,
        "x_peak_amp": peak_amp_x,
        "y_amp_before": amp_y_before,
        "y_amp_after": amp_y_after,
        "reduction_ratio": float(amp_y_after / amp_y_before) if amp_y_before > 0 else float("nan"),
        "reduction_percent": float(100.0 * (1.0 - amp_y_after / amp_y_before)) if amp_y_before > 0 else float("nan"),
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        constrained_layout=True,
    )

    ax_ts = axes[0, 0]
    ax_fft = axes[0, 1]
    ax_spec_y = axes[1, 0]
    ax_spec_clean = axes[1, 1]

    ax_ts.plot(t, x, label="x", linewidth=1.0)
    ax_ts.plot(t, y, label="y", linewidth=1.0)
    ax_ts.plot(t, y_clean, label="y_clean", linewidth=1.2)
    ax_ts.set_title(
        f"Pair {pair_idx} time series | a={a:.6g} | corr(x,y)={corr_before:.3f} -> corr(x,y_clean)={corr_after:.3f}"
    )
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Amplitude")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend()

    fft_mask_x = fft_x.freq <= fcap
    fft_mask_y = fft_y.freq <= fcap
    fft_mask_c = fft_clean.freq <= fcap

    ax_fft.semilogy(fft_x.freq[fft_mask_x], fft_x.amplitude[fft_mask_x], label="x", linewidth=1.0)
    ax_fft.semilogy(fft_y.freq[fft_mask_y], fft_y.amplitude[fft_mask_y], label="y", linewidth=1.0)
    ax_fft.semilogy(fft_clean.freq[fft_mask_c], fft_clean.amplitude[fft_mask_c], label="y_clean", linewidth=1.2)
    ax_fft.axvline(peak_hz, linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    ax_fft.set_title(
        f"FFT | x peak={peak_hz:.3f} Hz | y@peak {amp_y_before:.3g} -> {amp_y_after:.3g}"
    )
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_xlim(0.0, fcap)
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend()

    plot_spectrogram_panel(
        fig,
        ax_spec_y,
        spec=spec_y,
        t_start=float(t[0]),
        fmax=fcap,
        title="Sliding FFT of y",
    )
    plot_spectrogram_panel(
        fig,
        ax_spec_clean,
        spec=spec_clean,
        t_start=float(t[0]),
        fmax=fcap,
        title="Sliding FFT of y_clean",
    )

    fig.suptitle(
        f"Regression leakage test | pair {pair_idx} | x={label_x.upper()} | y={label_y.upper()}",
        fontsize=14,
    )

    return fig, metrics


def main() -> int:
    args = build_parser().parse_args()

    try:
        track2_x = load_track2_dataset(
            dataset=args.x_dataset,
            track_data_root=args.track_data_root,
        )
        track2_y = load_track2_dataset(
            dataset=args.y_dataset,
            track_data_root=args.track_data_root,
        )

        spacing_x = derive_spacing_dataset(track2_x)
        spacing_y = derive_spacing_dataset(track2_y)

        n_pairs_x = int(spacing_x.spacing_matrix.shape[1])
        n_pairs_y = int(spacing_y.spacing_matrix.shape[1])
        n_pairs = min(n_pairs_x, n_pairs_y)

        if n_pairs <= 0:
            raise ValueError("No spacing pairs available")

        if args.pair is None:
            pair_indices = list(range(n_pairs))
        else:
            pair_indices = sorted(set(int(i) for i in args.pair))
            bad = [i for i in pair_indices if i < 0 or i >= n_pairs]
            if bad:
                raise ValueError(f"Requested pair indices out of range: {bad}. Valid range is [0, {n_pairs - 1}]")

        save_dir = None
        if args.save_dir is not None:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f"x Track2: {track2_x.track2_path}")
        print(f"y Track2: {track2_y.track2_path}")
        print(f"Common pair count: {n_pairs}")
        print(f"Analyzing pairs: {pair_indices}")
        print()

        for pair_idx in pair_indices:
            label_x = spacing_x.pair_labels[pair_idx] if pair_idx < len(spacing_x.pair_labels) else "?"
            label_y = spacing_y.pair_labels[pair_idx] if pair_idx < len(spacing_y.pair_labels) else "?"

            fig, metrics = analyze_one_pair(
                pair_idx=pair_idx,
                label_x=label_x,
                label_y=label_y,
                raw_t_x=track2_x.frame_times_s,
                raw_x=np.asarray(spacing_x.spacing_matrix[:, pair_idx], dtype=float),
                raw_t_y=track2_y.frame_times_s,
                raw_y=np.asarray(spacing_y.spacing_matrix[:, pair_idx], dtype=float),
                longest=args.longest,
                handlenan=args.handlenan,
                sliding_len_s=args.sliding_len_s,
                peak_min_hz=args.peak_min_hz,
                peak_max_hz=args.peak_max_hz,
                fft_max_hz=args.fft_max_hz,
            )

            print(
                f"Pair {metrics['pair_idx']} ({label_x.upper()} / {label_y.upper()}): "
                f"a={metrics['scale_a']:.6g}, "
                f"corr {metrics['corr_before']:.4f} -> {metrics['corr_after']:.4f}, "
                f"x peak={metrics['x_peak_hz']:.4f} Hz, "
                f"y@peak {metrics['y_amp_before']:.6g} -> {metrics['y_amp_after']:.6g}, "
                f"reduction={metrics['reduction_percent']:.2f}%"
            )

            if save_dir is not None:
                out_path = save_dir / f"pair_{pair_idx:02d}_regression_test.png"
                fig.savefig(out_path, dpi=220)
                print(f"Saved: {out_path}")

            if args.no_show:
                plt.close(fig)

        if not args.no_show:
            plt.show()

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
