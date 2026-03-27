from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.signal as sp_signal

from .models import (
    ComplexFFTResult,
    FFTResult,
    ProcessedSignal,
    SpectrogramResult,
    TargetFrequencyResult,
    WelchSpectrumResult,
)


def get_longest_valid_segment(arr) -> tuple[int, int]:
    mask = ~np.isnan(arr)
    bounded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(bounded.astype(int))

    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return 0, 0

    lengths = stops - starts
    best_idx = int(np.argmax(lengths))
    return int(starts[best_idx]), int(stops[best_idx])


def get_longest_true_segment(mask) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    bounded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(bounded.astype(int))

    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return 0, 0

    lengths = stops - starts
    best_idx = int(np.argmax(lengths))
    return int(starts[best_idx]), int(stops[best_idx])


def collapse_duplicate_times(t, y):
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]

    uniq_t, idx, counts = np.unique(t_sorted, return_index=True, return_counts=True)
    if uniq_t.size == 0:
        return uniq_t, np.array([], dtype=float)

    sums = np.add.reduceat(y_sorted, idx)
    uniq_y = sums / counts
    return uniq_t, uniq_y


def build_uniform_signal(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]

    if t.size < 2:
        return None, None, None

    t, y = collapse_duplicate_times(t, y)
    if t.size < 2:
        return None, None, None

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None, None, None

    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        return None, None, None

    t_uniform = np.arange(t[0], t[-1] + 0.5 * median_dt, median_dt, dtype=float)
    t_uniform = t_uniform[t_uniform <= (t[-1] + 1e-12)]
    if t_uniform.size < 2:
        return None, None, None

    y_uniform = np.interp(t_uniform, t, y)
    return t_uniform, y_uniform, median_dt


def preprocess_signal(
    T,
    y,
    *,
    longest: bool = False,
    handlenan: bool = False,
    min_samples: int = 10,
) -> tuple[ProcessedSignal | None, str | None]:
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    if handlenan:
        valid_mask = np.isfinite(T) & np.isfinite(y)

        if longest:
            start, stop = get_longest_true_segment(valid_mask)
            if stop - start < min_samples:
                return None, "no valid data"

            t_sel = T[start:stop]
            y_sel = y[start:stop]
            finite_local = np.isfinite(t_sel) & np.isfinite(y_sel)
            t_sel = t_sel[finite_local]
            y_sel = y_sel[finite_local]
            proc_msg = f"Longest Seq: {stop - start}"
        else:
            t_sel = T[valid_mask]
            y_sel = y[valid_mask]
            if t_sel.size < 2:
                return None, "insufficient valid data"
            proc_msg = "Finite samples"
    else:
        if longest:
            start, stop = get_longest_valid_segment(y)
            if stop - start < min_samples:
                return None, "no valid data"

            t_sel = T[start:stop]
            y_sel = y[start:stop]
            proc_msg = f"Longest Seq: {stop - start}"
        else:
            nan_mask = np.isnan(y)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                if valid_idx.size < 2:
                    return None, "insufficient valid data"
                y_sel = np.interp(np.arange(len(y)), valid_idx, y[valid_idx])
            else:
                y_sel = y.copy()

            t_sel = T.copy()
            proc_msg = "Interpolated"

    t_uniform, y_uniform, dt = build_uniform_signal(t_sel, y_sel)
    if t_uniform is None or y_uniform is None or dt is None:
        return None, "could not build uniform signal"

    if len(y_uniform) < min_samples:
        return None, "insufficient valid data"

    y_uniform = y_uniform - np.mean(y_uniform)
    y_uniform = sp_signal.detrend(y_uniform, type="linear")

    Fs = 1.0 / dt
    nyquist = 0.5 * Fs

    return ProcessedSignal(
        t=t_uniform,
        y=y_uniform,
        dt=dt,
        Fs=Fs,
        nyquist=nyquist,
        proc_msg=proc_msg,
    ), None


def normalize_processed_signal_rms(
    processed: ProcessedSignal,
    *,
    target_rms: float = 100.0,
) -> tuple[ProcessedSignal | None, float | None, str | None]:
    y = np.asarray(processed.y, dtype=float)
    if y.size == 0:
        return None, None, "empty processed signal"
    if not np.isfinite(target_rms) or target_rms <= 0:
        return None, None, "target RMS must be positive and finite"

    rms = float(np.sqrt(np.mean(np.square(y))))
    if not np.isfinite(rms) or rms <= 0:
        return None, None, "processed signal RMS is non-positive or invalid"

    scale = float(target_rms) / rms
    return replace(processed, y=(y * scale)), scale, None


def hann_window_symmetric(n: int):
    return sp_signal.windows.hann(n, sym=True)


def hann_window_periodic(n: int):
    return sp_signal.windows.hann(n, sym=False)


def _compute_one_sided_fft_internal(y, dt):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 2:
        raise ValueError("Signal must have at least 2 samples for FFT")

    w = hann_window_symmetric(n)
    x = y * w

    X = np.fft.rfft(x) / n
    if X.size > 2:
        X = X.copy()
        X[1:-1] *= 2.0

    f = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(X)
    return f, X, amp


def compute_one_sided_fft(y, dt) -> FFTResult:
    f, _, amp = _compute_one_sided_fft_internal(y, dt)
    return FFTResult(freq=f, amplitude=amp)


def compute_one_sided_fft_complex(y, dt) -> ComplexFFTResult:
    f, X, amp = _compute_one_sided_fft_internal(y, dt)
    return ComplexFFTResult(freq=f, spectrum=X, amplitude=amp)


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(np.ceil(np.log2(n)))


def compute_complex_spectrogram(y, Fs: float, sliding_len_s: float) -> SpectrogramResult | None:
    n = len(y)
    win_samp = max(8, int(round(sliding_len_s * Fs)))
    win_samp = min(win_samp, n)

    if win_samp < 4:
        return None

    noverlap = min(int(round(0.90 * win_samp)), win_samp - 1)
    nfft = max(win_samp, next_power_of_two(win_samp))
    w = hann_window_periodic(win_samp)

    f_spec, t_spec, S_complex = sp_signal.spectrogram(
        y,
        fs=Fs,
        window=w,
        nperseg=win_samp,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
        mode="complex",
    )

    return SpectrogramResult(
        f=f_spec,
        t=t_spec,
        S_complex=S_complex,
        win_samp=win_samp,
        noverlap=noverlap,
        nfft=nfft,
    )


def compute_welch_spectrum(
    y,
    Fs: float,
    welch_len_s: float,
    *,
    overlap_fraction: float = 0.5,
) -> WelchSpectrumResult | None:
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 4:
        return None
    if welch_len_s <= 0:
        raise ValueError("welch_len_s must be > 0")
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in [0, 1)")

    nperseg = max(8, int(round(welch_len_s * Fs)))
    nperseg = min(nperseg, n)
    if nperseg < 4:
        return None

    noverlap = min(int(round(overlap_fraction * nperseg)), nperseg - 1)
    nfft = max(nperseg, next_power_of_two(nperseg))
    window = hann_window_periodic(nperseg)

    freq, power = sp_signal.welch(
        y,
        fs=Fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
    )
    amplitude = np.sqrt(np.maximum(power, 0.0))

    return WelchSpectrumResult(
        freq=freq,
        power=power,
        amplitude=amplitude,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )


def get_nearest_frequency_bin(freqs: np.ndarray, target: float) -> int:
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        raise ValueError("Frequency grid is empty")
    return int(np.argmin(np.abs(freqs - target)))


def get_complex_at_frequency(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    target: float,
    width: float | None = None,
    strategy: str = "max_amplitude",
) -> TargetFrequencyResult:
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum)

    if freqs.ndim != 1 or spectrum.ndim != 1 or freqs.size != spectrum.size:
        raise ValueError("freqs and spectrum must be 1D arrays of equal length")

    if freqs.size == 0:
        return TargetFrequencyResult(
            target_freq=float(target),
            selected_freq=float("nan"),
            index=-1,
            complex_value=0.0 + 0.0j,
            amplitude=0.0,
            found=False,
        )

    if width is None:
        idx = get_nearest_frequency_bin(freqs, target)
        val = complex(spectrum[idx])
        return TargetFrequencyResult(
            target_freq=float(target),
            selected_freq=float(freqs[idx]),
            index=idx,
            complex_value=val,
            amplitude=float(np.abs(val)),
            found=True,
        )

    if width < 0:
        raise ValueError("width must be >= 0")

    mask = (freqs >= (target - width)) & (freqs <= (target + width))
    if not np.any(mask):
        idx = get_nearest_frequency_bin(freqs, target)
        if abs(freqs[idx] - target) > width:
            return TargetFrequencyResult(
                target_freq=float(target),
                selected_freq=float(freqs[idx]),
                index=idx,
                complex_value=0.0 + 0.0j,
                amplitude=0.0,
                found=False,
            )
        val = complex(spectrum[idx])
        return TargetFrequencyResult(
            target_freq=float(target),
            selected_freq=float(freqs[idx]),
            index=idx,
            complex_value=val,
            amplitude=float(np.abs(val)),
            found=True,
        )

    idx_candidates = np.where(mask)[0]
    if strategy == "nearest":
        idx = idx_candidates[np.argmin(np.abs(freqs[idx_candidates] - target))]
    elif strategy == "max_amplitude":
        amps = np.abs(spectrum[idx_candidates])
        idx = idx_candidates[int(np.argmax(amps))]
    else:
        raise ValueError("strategy must be 'nearest' or 'max_amplitude'")

    val = complex(spectrum[idx])
    return TargetFrequencyResult(
        target_freq=float(target),
        selected_freq=float(freqs[idx]),
        index=int(idx),
        complex_value=val,
        amplitude=float(np.abs(val)),
        found=True,
    )


def align_complex_mode_shape(mode: np.ndarray, ref_index: int | None = None) -> np.ndarray:
    mode = np.asarray(mode, dtype=complex).copy()
    if mode.ndim != 1:
        raise ValueError("mode must be a 1D array")

    if mode.size == 0:
        return mode

    if ref_index is None:
        mags = np.abs(mode)
        if np.all(mags == 0):
            return mode
        ref_index = int(np.argmax(mags))

    ref_val = mode[ref_index]
    if np.abs(ref_val) == 0:
        return mode

    phase = np.angle(ref_val)
    return mode * np.exp(-1j * phase)


def complex_mode_to_bonds(site_mode: np.ndarray) -> np.ndarray:
    site_mode = np.asarray(site_mode, dtype=complex)
    if site_mode.ndim != 1:
        raise ValueError("site_mode must be 1D")
    if site_mode.size < 2:
        return np.array([], dtype=complex)
    return site_mode[1:] - site_mode[:-1]


def bonds_to_sites(
    bond_mode: np.ndarray,
    *,
    anchor_value: complex = 0.0 + 0.0j,
) -> np.ndarray:
    bond_mode = np.asarray(bond_mode, dtype=complex)
    n_bonds = bond_mode.size
    out = np.empty(n_bonds + 1, dtype=complex)
    out[0] = anchor_value
    for i in range(n_bonds):
        out[i + 1] = out[i] + bond_mode[i]
    return out


def extract_complex_mode_from_signals(
    signals: np.ndarray,
    t: np.ndarray,
    target_freq: float,
    *,
    width: float | None = None,
    strategy: str = "max_amplitude",
) -> tuple[np.ndarray, float, bool]:
    signals = np.asarray(signals, dtype=float)
    t = np.asarray(t, dtype=float)

    if signals.ndim != 2:
        raise ValueError("signals must be 2D, shape (n_samples, n_channels)")
    if t.ndim != 1 or t.size != signals.shape[0]:
        raise ValueError("t must be 1D and match signals.shape[0]")
    if t.size < 2:
        raise ValueError("Need at least 2 time samples")

    dt = float(np.median(np.diff(t)))
    mode = np.zeros(signals.shape[1], dtype=complex)
    selected_freq = float("nan")
    found_any = False

    for j in range(signals.shape[1]):
        res = compute_one_sided_fft_complex(signals[:, j], dt)
        picked = get_complex_at_frequency(
            res.freq,
            res.spectrum,
            target_freq,
            width=width,
            strategy=strategy,
        )
        mode[j] = picked.complex_value
        if picked.found and not found_any:
            selected_freq = picked.selected_freq
            found_any = True

    return mode, selected_freq, found_any
