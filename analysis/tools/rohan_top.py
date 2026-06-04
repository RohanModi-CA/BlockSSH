from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .io import join_dataset_component, load_track2_dataset, split_dataset_component


DEFAULT_TARGET_FREQUENCIES = np.array([3.3, 6.2, 8.85, 11.6, 15.25, 16.58, 20.35, 21.5, 22.2, 23.0], dtype=float)


@dataclass(frozen=True)
class RohanTopResult:
    dataset_name: str
    component: str
    source_path: str
    tmin: float
    tmax: float
    fs: float
    win: int
    av: int
    offset: int
    time_s: np.ndarray
    bond_signal: np.ndarray
    freq_hz: np.ndarray
    avg_amplitude: np.ndarray
    target_frequencies: np.ndarray
    target_signed: np.ndarray
    target_amplitude: np.ndarray
    chirality: np.ndarray
    chirality_smoothed: np.ndarray
    window_count: int


def resolve_component_dataset_name(dataset: str, component: str) -> str:
    base, _ = split_dataset_component(str(dataset))
    return join_dataset_component(base, component)


def load_component_track2_dataset(
    dataset: str,
    component: str,
    *,
    track_data_root: str | Path | None = None,
):
    component_dataset = resolve_component_dataset_name(dataset, component)
    return load_track2_dataset(dataset=component_dataset, track_data_root=track_data_root)


def _interpolate_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr.copy()
    if np.count_nonzero(finite) < 2:
        raise ValueError("Need at least two finite samples to interpolate a bond trace")
    idx = np.arange(arr.size, dtype=float)
    return np.interp(idx, idx[finite], arr[finite])


def _fill_nan_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("bond signal matrix must be 2D")
    if arr.shape[1] == 0:
        return arr.copy()
    return np.column_stack([_interpolate_1d(arr[:, idx]) for idx in range(arr.shape[1])])


def derive_bond_signal_matrix(x_positions: np.ndarray) -> np.ndarray:
    x_positions = np.asarray(x_positions, dtype=float)
    if x_positions.ndim != 2:
        raise ValueError("x_positions must be a 2D array")
    if x_positions.shape[1] < 2:
        return np.empty((x_positions.shape[0], 0), dtype=float)
    return np.abs(x_positions[:, 1:] - x_positions[:, :-1])


def select_time_window(
    frame_times_s: np.ndarray,
    bond_signal: np.ndarray,
    *,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(frame_times_s, dtype=float)
    signal = np.asarray(bond_signal, dtype=float)
    if t.ndim != 1:
        raise ValueError("frame_times_s must be 1D")
    if signal.ndim != 2:
        raise ValueError("bond_signal must be 2D")
    if signal.shape[0] != t.shape[0]:
        raise ValueError("frame_times_s length must match bond_signal rows")
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        raise ValueError("tmin and tmax must be finite")
    if tmax <= tmin:
        raise ValueError("tmax must be greater than tmin")

    mask = (t > float(tmin)) & (t < float(tmax))
    return t[mask], signal[mask]


def _windowed_fft(
    series: np.ndarray,
    *,
    fs: float,
    win: int,
    av: int,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    arr = _interpolate_1d(series)
    win = int(win)
    av = int(av)
    offset = int(offset)
    if win < 2:
        raise ValueError("--win must be at least 2")
    if av < 1:
        raise ValueError("--av must be at least 1")
    if offset < 0:
        raise ValueError("--offset must be non-negative")
    if arr.size < win:
        raise ValueError("Selected time window is shorter than --win")

    available = arr.size - win - offset + 1
    if available <= 0:
        raise ValueError("Selected time window is too short for the requested offset")

    window_count = min(av, available)
    if window_count < av:
        print(
            f"Warning: requested --av={av} windows but only {window_count} are available in this time window"
        )

    windows = np.lib.stride_tricks.sliding_window_view(arr, win)
    starts = offset + np.arange(window_count, dtype=int)
    selected = windows[starts]
    fft_windows = np.fft.fft(selected, axis=-1) / float(win)
    avg_amp = np.mean(np.abs(fft_windows), axis=0)
    freq = np.arange(win, dtype=float) * float(fs) / float(win)
    return freq, avg_amp, fft_windows[0], window_count


def _target_response(
    freq: np.ndarray,
    avg_amp: np.ndarray,
    ref_fft: np.ndarray,
    target_freqs: np.ndarray,
    *,
    fs: float,
    win: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(target_freqs, dtype=float)
    signed = np.zeros(targets.shape[0], dtype=float)
    amplitude = np.zeros(targets.shape[0], dtype=float)
    freq = np.asarray(freq, dtype=float)
    avg_amp = np.asarray(avg_amp, dtype=float)
    ref_fft = np.asarray(ref_fft, dtype=complex)

    for idx, target in enumerate(targets):
        center = int(round(float(target) * float(win) / float(fs)))
        lo = max(0, center - 3)
        hi = min(freq.size, center + 4)
        if hi <= lo:
            lo = max(0, min(center, freq.size - 1))
            hi = min(freq.size, lo + 1)
        local = avg_amp[lo:hi]
        if local.size == 0:
            selected = min(max(center, 0), freq.size - 1)
        else:
            peaks, _ = find_peaks(local, distance=4, prominence=0.02)
            if peaks.size == 0:
                selected = lo + int(np.argmax(local))
            else:
                selected = lo + int(peaks[int(np.argmax(local[peaks]))])
        amplitude[idx] = float(avg_amp[selected])
        signed[idx] = float(avg_amp[selected] * np.sign(np.real(ref_fft[selected])))
    return signed, amplitude


def _chirality_trace(long_signal: np.ndarray, *, smooth_window: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(long_signal, dtype=float)
    if arr.ndim != 2:
        raise ValueError("long_signal must be 2D")
    if arr.shape[1] == 0:
        empty = np.empty((arr.shape[0],), dtype=float)
        return empty, empty

    centered = _fill_nan_columns(arr)
    means = np.mean(centered, axis=0, keepdims=True)
    centered = centered - means
    xprobe = centered.T

    weights = np.array([((idx // 2) + 1) * (1.0 if idx % 2 == 0 else -1.0) for idx in range(xprobe.shape[0])], dtype=float)
    norms = np.linalg.norm(xprobe, axis=0)
    xnorm = np.full_like(xprobe, np.nan, dtype=float)
    valid = np.isfinite(norms) & (norms > 0)
    if np.any(valid):
        xnorm[:, valid] = xprobe[:, valid] / norms[valid]

    chirality = np.nansum((xnorm * xnorm) * weights[:, None], axis=0)
    smoothed = smooth_series(chirality, smooth_window)
    return chirality, smoothed


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    window = int(max(1, min(int(window), arr.size)))
    if window == 1:
        return arr.copy()

    finite = np.isfinite(arr).astype(float)
    data = np.where(np.isfinite(arr), arr, 0.0)
    kernel = np.ones(window, dtype=float)
    summed = np.convolve(data, kernel, mode="same")
    counts = np.convolve(finite, kernel, mode="same")
    out = np.full_like(arr, np.nan, dtype=float)
    valid = counts > 0
    out[valid] = summed[valid] / counts[valid]
    return out


def analyze_rohan_top(
    dataset: str,
    *,
    component: str,
    tmin: float,
    tmax: float,
    track_data_root: str | Path | None = None,
    fs: float = 120.0,
    win: int = 400,
    av: int = 2000,
    offset: int = 0,
    target_frequencies: np.ndarray | None = None,
    smooth_window: int = 1000,
) -> RohanTopResult:
    track2 = load_component_track2_dataset(dataset, component, track_data_root=track_data_root)
    bond_signal = derive_bond_signal_matrix(track2.x_positions)
    if bond_signal.shape[1] == 0:
        raise ValueError(f"Dataset '{dataset}' has fewer than 2 tracked sites")

    time_s, selected_signal = select_time_window(
        track2.frame_times_s,
        bond_signal,
        tmin=tmin,
        tmax=tmax,
    )
    if time_s.size == 0:
        raise ValueError(f"No samples fall inside the requested window ({tmin}, {tmax})")

    filled_signal = _fill_nan_columns(selected_signal)

    freqs: np.ndarray | None = None
    avg_amp_rows: list[np.ndarray] = []
    target_signed_rows: list[np.ndarray] = []
    target_amp_rows: list[np.ndarray] = []
    window_count: int | None = None

    targets = np.asarray(DEFAULT_TARGET_FREQUENCIES if target_frequencies is None else target_frequencies, dtype=float)

    for bond_idx in range(filled_signal.shape[1]):
        freq, avg_amp, ref_fft, windows_used = _windowed_fft(
            filled_signal[:, bond_idx],
            fs=fs,
            win=win,
            av=av,
            offset=offset,
        )
        if freqs is None:
            freqs = freq
            window_count = windows_used
        avg_amp_rows.append(avg_amp)
        signed, amplitude = _target_response(freq, avg_amp, ref_fft, targets, fs=fs, win=win)
        target_signed_rows.append(signed)
        target_amp_rows.append(amplitude)

    assert freqs is not None
    assert window_count is not None

    avg_amp_matrix = np.vstack(avg_amp_rows)
    target_signed = np.vstack(target_signed_rows).T
    target_amp = np.vstack(target_amp_rows).T
    chirality, chirality_smoothed = _chirality_trace(filled_signal, smooth_window=smooth_window)

    base_dataset, _ = split_dataset_component(str(dataset))
    return RohanTopResult(
        dataset_name=base_dataset,
        component=str(component),
        source_path=str(track2.track2_path),
        tmin=float(tmin),
        tmax=float(tmax),
        fs=float(fs),
        win=int(win),
        av=int(av),
        offset=int(offset),
        time_s=np.asarray(time_s, dtype=float),
        bond_signal=np.asarray(filled_signal, dtype=float),
        freq_hz=np.asarray(freqs, dtype=float),
        avg_amplitude=np.asarray(avg_amp_matrix, dtype=float),
        target_frequencies=np.asarray(targets, dtype=float),
        target_signed=np.asarray(target_signed, dtype=float),
        target_amplitude=np.asarray(target_amp, dtype=float),
        chirality=np.asarray(chirality, dtype=float),
        chirality_smoothed=np.asarray(chirality_smoothed, dtype=float),
        window_count=int(window_count),
    )

