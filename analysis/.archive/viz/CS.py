#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
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
            "Cross-spectral leakage removal test: estimate frequency-dependent "
            "transfer H(f)=Sxy/Sxx, then remove Y_leak(f)=H(f)X(f) from y."
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
        "--csd-segment-len-s",
        type=float,
        default=8.0,
        help="Segment length in seconds used to estimate cross spectra. Default: 8.0",
    )
    parser.add_argument(
        "--csd-overlap-frac",
        type=float,
        default=0.5,
        help="Overlap fraction for cross-spectral segments. Default: 0.5",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-8,
        help="Relative regularization added to Sxx before division. Default: 1e-8",
    )
    parser.add_argument(
        "--coherence-floor",
        type=float,
        default=0.0,
        help=(
            "Optional coherence floor in [0,1]. Where coherence is below this, "
            "set H(f)=0. Default: 0.0"
        ),
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

    x = x - np.mean(x)
    y = y - np.mean(y)

    return t, x, y, dt


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


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=float)
    return np.hanning(n).astype(float)


def estimate_cross_spectra(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    seg_len_s: float,
    overlap_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Welch-style cross-spectral estimate.
    Returns:
        f      : one-sided frequency grid
        Sxx    : auto spectrum of x
        Syy    : auto spectrum of y
        Sxy    : cross spectrum x->y
    """
    n = int(x.size)
    if n != y.size:
        raise ValueError("x and y must have same length")

    nperseg = max(16, int(round(seg_len_s * fs)))
    nperseg = min(nperseg, n)
    if nperseg < 16:
        raise ValueError("Signal too short for cross-spectral estimation")

    noverlap = int(round(overlap_frac * nperseg))
    noverlap = max(0, min(noverlap, nperseg - 1))
    step = nperseg - noverlap

    nfft = next_pow2(nperseg)
    win = hann_window(nperseg)
    win_pow = float(np.sum(win**2))
    if win_pow <= 0:
        raise ValueError("Invalid window power")

    starts = list(range(0, n - nperseg + 1, step))
    if not starts:
        starts = [0]

    Sxx_acc = None
    Syy_acc = None
    Sxy_acc = None
    count = 0

    for start in starts:
        xs = x[start:start + nperseg]
        ys = y[start:start + nperseg]
        if xs.size != nperseg or ys.size != nperseg:
            continue

        xs = (xs - np.mean(xs)) * win
        ys = (ys - np.mean(ys)) * win

        X = np.fft.rfft(xs, n=nfft)
        Y = np.fft.rfft(ys, n=nfft)

        # Scale is enough for relative transfer estimation; absolute PSD units
        # are not critical here, but keep a consistent normalization anyway.
        scale = 1.0 / (fs * win_pow)

        Sxx_seg = scale * (X * np.conj(X))
        Syy_seg = scale * (Y * np.conj(Y))
        Sxy_seg = scale * (X * np.conj(Y))

        if Sxx_acc is None:
            Sxx_acc = Sxx_seg
            Syy_acc = Syy_seg
            Sxy_acc = Sxy_seg
        else:
            Sxx_acc += Sxx_seg
            Syy_acc += Syy_seg
            Sxy_acc += Sxy_seg

        count += 1

    if count <= 0:
        raise ValueError("Failed to accumulate any spectral segments")

    Sxx = Sxx_acc / count
    Syy = Syy_acc / count
    Sxy = Sxy_acc / count
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    return f, Sxx, Syy, Sxy


def estimate_transfer_function(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    seg_len_s: float,
    overlap_frac: float,
    regularization: float,
    coherence_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate H(f)=Sxy/Sxx and coherence gamma^2(f)=|Sxy|^2/(Sxx*Syy).
    Returns:
        f
        H
        coh
    """
    f, Sxx, Syy, Sxy = estimate_cross_spectra(
        x=x,
        y=y,
        fs=fs,
        seg_len_s=seg_len_s,
        overlap_frac=overlap_frac,
    )

    eps = np.finfo(float).eps
    reg = float(regularization) * max(float(np.max(np.real(Sxx))), eps)
    denom = Sxx + reg

    H = Sxy / denom

    coh = (np.abs(Sxy) ** 2) / (np.maximum(np.real(Sxx) * np.real(Syy), eps))
    coh = np.clip(np.real(coh), 0.0, 1.0)

    if coherence_floor > 0.0:
        H = np.where(coh >= coherence_floor, H, 0.0 + 0.0j)

    return f, H, coh


def remove_cross_spectral_leakage(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    seg_len_s: float,
    overlap_frac: float,
    regularization: float,
    coherence_floor: float,
) -> tuple[np.ndarray, dict]:
    """
    Estimate frequency-dependent leakage from x into y, remove it in the
    frequency domain, and return y_clean in time domain.
    """
    if x.size != y.size:
        raise ValueError("x and y must have same length")

    fs = 1.0 / dt
    n = int(x.size)

    f_H, H, coh = estimate_transfer_function(
        x=x,
        y=y,
        fs=fs,
        seg_len_s=seg_len_s,
        overlap_frac=overlap_frac,
        regularization=regularization,
        coherence_floor=coherence_floor,
    )

    # Use whole-record FFT for actual subtraction, with H interpolated to that grid.
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    f_full = np.fft.rfftfreq(n, d=dt)

    # Interpolate real/imag parts separately.
    H_real = np.interp(f_full, f_H, np.real(H), left=np.real(H[0]), right=np.real(H[-1]))
    H_imag = np.interp(f_full, f_H, np.imag(H), left=np.imag(H[0]), right=np.imag(H[-1]))
    H_full = H_real + 1j * H_imag

    coh_full = np.interp(f_full, f_H, coh, left=coh[0], right=coh[-1])

    Y_leak = H_full * X
    Y_clean = Y - Y_leak
    y_clean = np.fft.irfft(Y_clean, n=n)

    details = {
        "f_transfer": f_H,
        "H": H,
        "coherence": coh,
        "f_full": f_full,
        "H_full": H_full,
        "coherence_full": coh_full,
        "Y_leak": Y_leak,
        "Y_clean": Y_clean,
    }
    return y_clean, details


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
    csd_segment_len_s: float,
    csd_overlap_frac: float,
    regularization: float,
    coherence_floor: float,
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

    y_clean, leak = remove_cross_spectral_leakage(
        x=x,
        y=y,
        dt=dt,
        seg_len_s=csd_segment_len_s,
        overlap_frac=csd_overlap_frac,
        regularization=regularization,
        coherence_floor=coherence_floor,
    )

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

    # Transfer/coherence at peak frequency
    f_full = leak["f_full"]
    H_full = leak["H_full"]
    coh_full = leak["coherence_full"]
    idx_peak = int(np.argmin(np.abs(f_full - peak_hz)))
    H_peak = H_full[idx_peak]
    coh_peak = float(coh_full[idx_peak])

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
        "corr_before": corr_before,
        "corr_after": corr_after,
        "x_peak_hz": peak_hz,
        "x_peak_amp": peak_amp_x,
        "y_amp_before": amp_y_before,
        "y_amp_after": amp_y_after,
        "reduction_ratio": float(amp_y_after / amp_y_before) if amp_y_before > 0 else float("nan"),
        "reduction_percent": float(100.0 * (1.0 - amp_y_after / amp_y_before)) if amp_y_before > 0 else float("nan"),
        "H_peak_real": float(np.real(H_peak)),
        "H_peak_imag": float(np.imag(H_peak)),
        "H_peak_mag": float(np.abs(H_peak)),
        "H_peak_phase_rad": float(np.angle(H_peak)),
        "coherence_peak": coh_peak,
    }

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
        constrained_layout=True,
    )

    ax_ts = axes[0, 0]
    ax_fft = axes[0, 1]
    ax_H = axes[1, 0]
    ax_coh = axes[1, 1]
    ax_spec_y = axes[2, 0]
    ax_spec_clean = axes[2, 1]

    ax_ts.plot(t, x, label="x", linewidth=1.0)
    ax_ts.plot(t, y, label="y", linewidth=1.0)
    ax_ts.plot(t, y_clean, label="y_clean", linewidth=1.2)
    ax_ts.set_title(
        f"Pair {pair_idx} time series | corr(x,y)={corr_before:.3f} -> corr(x,y_clean)={corr_after:.3f}"
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

    H_mask = f_full <= fcap
    ax_H.plot(f_full[H_mask], np.abs(H_full[H_mask]), linewidth=1.0, label="|H(f)|")
    ax_H.axvline(peak_hz, linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    ax_H.set_title(
        f"Estimated transfer | |H({peak_hz:.3f} Hz)|={np.abs(H_peak):.3g}, phase={np.angle(H_peak):.3f} rad"
    )
    ax_H.set_xlabel("Frequency (Hz)")
    ax_H.set_ylabel("|H(f)|")
    ax_H.set_xlim(0.0, fcap)
    ax_H.grid(True, alpha=0.3)

    ax_coh.plot(f_full[H_mask], coh_full[H_mask], linewidth=1.0, label="coherence")
    ax_coh.axvline(peak_hz, linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    ax_coh.set_title(f"Coherence | peak-band coherence @ {peak_hz:.3f} Hz = {coh_peak:.3f}")
    ax_coh.set_xlabel("Frequency (Hz)")
    ax_coh.set_ylabel("Coherence")
    ax_coh.set_xlim(0.0, fcap)
    ax_coh.set_ylim(0.0, 1.05)
    ax_coh.grid(True, alpha=0.3)

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
        f"Cross-spectral leakage removal | pair {pair_idx} | x={label_x.upper()} | y={label_y.upper()}",
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
                csd_segment_len_s=args.csd_segment_len_s,
                csd_overlap_frac=args.csd_overlap_frac,
                regularization=args.regularization,
                coherence_floor=args.coherence_floor,
            )

            print(
                f"Pair {metrics['pair_idx']} ({label_x.upper()} / {label_y.upper()}): "
                f"corr {metrics['corr_before']:.4f} -> {metrics['corr_after']:.4f}, "
                f"x peak={metrics['x_peak_hz']:.4f} Hz, "
                f"|H(peak)|={metrics['H_peak_mag']:.6g}, "
                f"phase={metrics['H_peak_phase_rad']:.4f} rad, "
                f"coh={metrics['coherence_peak']:.4f}, "
                f"y@peak {metrics['y_amp_before']:.6g} -> {metrics['y_amp_after']:.6g}, "
                f"reduction={metrics['reduction_percent']:.2f}%"
            )

            if save_dir is not None:
                out_path = save_dir / f"pair_{pair_idx:02d}_cross_spectral_test.png"
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
