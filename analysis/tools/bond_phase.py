from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal as sp_signal

from .models import SignalRecord
from .peaks import assert_peaks_strictly_increasing
from .signal import hann_window_periodic, next_power_of_two, preprocess_signal


@dataclass(frozen=True)
class BondPeakPhaseResult:
    dataset_name: str
    bond_ids: np.ndarray
    peaks_hz: np.ndarray
    mean_selected_freq_hz: np.ndarray
    relative_phase_rad: np.ndarray
    mean_bond_magnitude: np.ndarray
    bond_coherence: np.ndarray
    n_windows_used: np.ndarray
    reference_bond_id: int
    reference_peak_hz: float


@dataclass(frozen=True)
class LoadedBondPhaseTable:
    peaks_hz: np.ndarray
    bond_ids: np.ndarray
    relative_phase_rad: np.ndarray


def _enabled_dataset_names(records: list[SignalRecord]) -> list[str]:
    return sorted({str(record.dataset_name) for record in records})


def _align_processed_bond_signals(
    records: list[SignalRecord],
    *,
    longest: bool = False,
    handlenan: bool = False,
    min_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    processed_rows: list[tuple[int, object]] = []
    dataset_names = _enabled_dataset_names(records)
    if len(dataset_names) != 1:
        raise ValueError(
            f"Expected exactly one enabled dataset for bond phase generation, found {len(dataset_names)}: {dataset_names}"
        )

    for record in sorted(records, key=lambda rec: int(rec.entity_id)):
        processed, error_msg = preprocess_signal(
            record.t,
            record.y,
            longest=longest,
            handlenan=handlenan,
            min_samples=min_samples,
        )
        if processed is None:
            raise ValueError(
                f"Could not preprocess bond {record.entity_id} in dataset '{record.dataset_name}': {error_msg}"
            )
        processed_rows.append((int(record.entity_id), processed))

    if len(processed_rows) == 0:
        raise ValueError("Need at least one bond signal to estimate bond phases")

    common_start = max(float(proc.t[0]) for _, proc in processed_rows)
    common_stop = min(float(proc.t[-1]) for _, proc in processed_rows)
    if common_stop <= common_start:
        raise ValueError("No overlapping time span across the selected bond signals")

    dts = np.asarray([float(proc.dt) for _, proc in processed_rows], dtype=float)
    dt = float(np.median(dts))
    if not np.allclose(dts, dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Preprocessed bond signals do not share a consistent sampling interval")

    t_common = np.arange(common_start, common_stop + 0.5 * dt, dt, dtype=float)
    t_common = t_common[t_common <= (common_stop + 1e-12)]
    if t_common.size < max(16, min_samples):
        raise ValueError("Common aligned time grid is too short for bond phase estimation")

    bond_ids = np.asarray([bond_id for bond_id, _ in processed_rows], dtype=int)
    signals = np.empty((t_common.size, len(processed_rows)), dtype=float)
    for col, (_, proc) in enumerate(processed_rows):
        signals[:, col] = np.interp(t_common, proc.t, proc.y)

    return t_common, signals, bond_ids, dataset_names[0]


def _compute_complex_spectrogram_with_overlap(
    y: np.ndarray,
    fs: float,
    segment_len_s: float,
    overlap_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        raise ValueError("Need at least 4 samples to compute a complex spectrogram")
    if segment_len_s <= 0:
        raise ValueError("segment_len_s must be > 0")
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be in [0, 1)")

    nperseg = max(8, int(round(segment_len_s * fs)))
    nperseg = min(nperseg, n)
    if nperseg < 4:
        raise ValueError("Segment length is too short for a stable complex spectrogram")

    noverlap = min(int(round(overlap_fraction * nperseg)), nperseg - 1)
    nfft = max(nperseg, next_power_of_two(nperseg))
    window = hann_window_periodic(nperseg)
    freq, time, s_complex = sp_signal.spectrogram(
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
        mode="complex",
    )
    return np.asarray(freq, dtype=float), np.asarray(time, dtype=float), np.asarray(s_complex, dtype=complex)


def _pick_frequency_index(freqs: np.ndarray, target_hz: float, search_width_hz: float) -> int:
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("Frequency grid must be a non-empty 1D array")

    mask = (freqs >= (target_hz - search_width_hz)) & (freqs <= (target_hz + search_width_hz))
    if np.any(mask):
        idx_candidates = np.where(mask)[0]
        return int(idx_candidates[np.argmin(np.abs(freqs[idx_candidates] - target_hz))])

    idx = int(np.argmin(np.abs(freqs - target_hz)))
    if abs(float(freqs[idx]) - float(target_hz)) > float(search_width_hz):
        raise ValueError(
            f"Could not find a frequency bin within {search_width_hz:.6g} Hz of target {target_hz:.6g} Hz"
        )
    return idx


def _find_bond_index(bond_ids: np.ndarray, bond_id: int) -> int:
    matches = np.where(np.asarray(bond_ids, dtype=int) == int(bond_id))[0]
    if matches.size == 0:
        raise ValueError(f"Required bond id {bond_id} was not found in the configured bond signals")
    return int(matches[0])


def _pick_windowwise_bond_values(
    bond_spectra: np.ndarray,
    freq_grid: np.ndarray,
    target_hz: float,
    search_width_hz: float,
    reference_bond_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (freq_grid >= (target_hz - search_width_hz)) & (freq_grid <= (target_hz + search_width_hz))
    idx_candidates = np.where(mask)[0]
    if idx_candidates.size == 0:
        idx_candidates = np.array([_pick_frequency_index(freq_grid, float(target_hz), float(search_width_hz))], dtype=int)

    ref_spec = bond_spectra[idx_candidates, :, reference_bond_idx]
    best_local = np.argmax(np.abs(ref_spec), axis=0)
    chosen_idx = idx_candidates[best_local]

    n_windows = bond_spectra.shape[1]
    chosen_values = np.empty((n_windows, bond_spectra.shape[2]), dtype=complex)
    for window_idx in range(n_windows):
        chosen_values[window_idx, :] = bond_spectra[chosen_idx[window_idx], window_idx, :]
    chosen_freqs = freq_grid[chosen_idx]
    return chosen_values, chosen_freqs


def estimate_bond_peak_phases(
    records: list[SignalRecord],
    peaks_hz: list[float] | np.ndarray,
    *,
    reference_bond_id: int,
    reference_peak_index: int,
    welch_len_s: float = 20.0,
    welch_overlap_fraction: float = 0.5,
    search_width_hz: float = 0.25,
    min_reference_fraction: float = 0.05,
    longest: bool = False,
    handlenan: bool = False,
    min_samples: int = 10,
) -> BondPeakPhaseResult:
    peaks_arr = assert_peaks_strictly_increasing(peaks_hz)
    if not (1 <= int(reference_peak_index) <= len(peaks_arr)):
        raise ValueError(
            f"reference_peak_index must be between 1 and {len(peaks_arr)}; got {reference_peak_index}"
        )

    t_common, signals, bond_ids, dataset_name = _align_processed_bond_signals(
        records,
        longest=longest,
        handlenan=handlenan,
        min_samples=min_samples,
    )

    dt = float(np.median(np.diff(t_common)))
    fs = 1.0 / dt

    spectrograms = []
    freq_grid = None
    time_grid = None
    for col_idx in range(signals.shape[1]):
        freq, time, s_complex = _compute_complex_spectrogram_with_overlap(
            signals[:, col_idx],
            fs,
            welch_len_s,
            welch_overlap_fraction,
        )
        if freq_grid is None:
            freq_grid = np.asarray(freq, dtype=float)
            time_grid = np.asarray(time, dtype=float)
        else:
            if s_complex.shape != spectrograms[0].shape:
                raise ValueError("Configured bond spectrogram shapes did not match")
            if not np.allclose(freq, freq_grid, rtol=1e-6, atol=1e-9):
                raise ValueError("Configured bond frequency grids did not match")
            if not np.allclose(time, time_grid, rtol=1e-6, atol=1e-9):
                raise ValueError("Configured bond time grids did not match")
        spectrograms.append(np.asarray(s_complex, dtype=complex))

    if freq_grid is None or time_grid is None or len(spectrograms) == 0:
        raise ValueError("No complex bond spectrograms were produced")

    bond_spectra = np.stack(spectrograms, axis=2)
    reference_bond_idx = _find_bond_index(bond_ids, int(reference_bond_id))
    reference_peak_hz = float(peaks_arr[int(reference_peak_index) - 1])

    ref_windows, _ = _pick_windowwise_bond_values(
        bond_spectra,
        freq_grid,
        reference_peak_hz,
        search_width_hz,
        reference_bond_idx,
    )
    ref_vals = np.asarray(ref_windows[:, reference_bond_idx], dtype=complex)
    ref_amp = np.abs(ref_vals)
    finite_ref = np.isfinite(ref_amp)
    positive_ref = finite_ref & (ref_amp > 0)
    if not np.any(positive_ref):
        raise ValueError(
            f"Reference bond {reference_bond_id} at peak {reference_peak_hz:.6g} Hz had no usable short-time windows"
        )

    ref_median = float(np.median(ref_amp[positive_ref]))
    ref_threshold = float(min_reference_fraction) * ref_median
    gauge_mask = positive_ref & (ref_amp >= ref_threshold)
    if not np.any(gauge_mask):
        raise ValueError(
            f"Reference gating removed all windows; median amplitude={ref_median:.6g}, "
            f"threshold={ref_threshold:.6g}"
        )

    rotations = np.ones(ref_vals.shape[0], dtype=complex)
    rotations[gauge_mask] = np.exp(-1j * np.angle(ref_vals[gauge_mask]))

    n_bonds = len(bond_ids)
    n_peaks = len(peaks_arr)
    phase = np.full((n_bonds, n_peaks), np.nan, dtype=float)
    mean_mag = np.full((n_bonds, n_peaks), np.nan, dtype=float)
    coherence = np.full((n_bonds, n_peaks), np.nan, dtype=float)
    n_windows_used = np.zeros(n_peaks, dtype=int)
    mean_selected_freq_hz = np.full(n_peaks, np.nan, dtype=float)

    for peak_idx, peak_hz in enumerate(peaks_arr):
        bond_windows, chosen_freqs = _pick_windowwise_bond_values(
            bond_spectra,
            freq_grid,
            float(peak_hz),
            search_width_hz,
            reference_bond_idx,
        )
        finite_mask = np.all(np.isfinite(bond_windows), axis=1)
        keep_mask = gauge_mask & finite_mask
        peak_windows = bond_windows[keep_mask]
        peak_freqs = chosen_freqs[keep_mask]
        if peak_windows.shape[0] == 0:
            continue

        rotated = peak_windows * rotations[keep_mask][:, None]
        mean_selected_freq_hz[peak_idx] = float(np.mean(peak_freqs))
        n_windows_used[peak_idx] = int(rotated.shape[0])

        mean_complex = np.mean(rotated, axis=0)
        phase[:, peak_idx] = np.angle(mean_complex)
        mean_mag[:, peak_idx] = np.abs(mean_complex)

        normalized = np.zeros_like(rotated)
        amp = np.abs(rotated)
        nonzero = amp > 1e-12
        normalized[nonzero] = rotated[nonzero] / amp[nonzero]
        coherence[:, peak_idx] = np.abs(np.mean(normalized, axis=0))

    return BondPeakPhaseResult(
        dataset_name=dataset_name,
        bond_ids=np.asarray(bond_ids, dtype=int),
        peaks_hz=np.asarray(peaks_arr, dtype=float),
        mean_selected_freq_hz=mean_selected_freq_hz,
        relative_phase_rad=phase,
        mean_bond_magnitude=mean_mag,
        bond_coherence=coherence,
        n_windows_used=n_windows_used,
        reference_bond_id=int(reference_bond_id),
        reference_peak_hz=float(reference_peak_hz),
    )


def write_bond_peak_phase_csv(result: BondPeakPhaseResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["bond_id", *[f"{peak:.12g}" for peak in result.peaks_hz]]
        writer.writerow(header)
        for row_idx, bond_id in enumerate(result.bond_ids):
            row = [int(bond_id), *[f"{val:.12g}" for val in result.relative_phase_rad[row_idx, :]]]
            writer.writerow(row)

    return output_path


def load_bond_peak_phase_csv(path: str | Path) -> LoadedBondPhaseTable:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Bond phase CSV not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise ValueError("Bond phase CSV must contain a header and at least one bond row")

    header = rows[0]
    if len(header) < 2 or header[0].strip().lower() != "bond_id":
        raise ValueError("Bond phase CSV header must start with 'bond_id'")

    try:
        peaks_hz = np.asarray([float(cell) for cell in header[1:]], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse bond phase CSV header peak frequencies: {exc}") from exc

    bond_ids: list[int] = []
    phase_rows: list[list[float]] = []
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError("Each bond phase CSV row must have the same number of columns as the header")
        try:
            bond_id = int(row[0])
        except ValueError as exc:
            raise ValueError(f"Could not parse bond id '{row[0]}': {exc}") from exc
        try:
            phases = [float(cell) for cell in row[1:]]
        except ValueError as exc:
            raise ValueError(f"Could not parse phase row for bond {bond_id}: {exc}") from exc
        bond_ids.append(bond_id)
        phase_rows.append(phases)

    return LoadedBondPhaseTable(
        peaks_hz=np.asarray(peaks_hz, dtype=float),
        bond_ids=np.asarray(bond_ids, dtype=int),
        relative_phase_rad=np.asarray(phase_rows, dtype=float),
    )


def transform_bond_phase_table(
    phase_table: LoadedBondPhaseTable,
    *,
    flip: bool = False,
    flip_bond_ids: list[int] | None = None,
    forcereal: bool = False,
) -> LoadedBondPhaseTable:
    phase = np.asarray(phase_table.relative_phase_rad, dtype=float)
    if flip:
        phase = phase + np.pi
    if flip_bond_ids:
        requested = {int(bond_id) for bond_id in flip_bond_ids}
        for row_idx, bond_id in enumerate(np.asarray(phase_table.bond_ids, dtype=int)):
            if int(bond_id) in requested:
                phase[row_idx, :] = phase[row_idx, :] + np.pi
    if forcereal:
        wrapped = np.angle(np.exp(1j * phase))
        phase = np.where(np.cos(wrapped) >= 0.0, 0.0, np.pi)

    return LoadedBondPhaseTable(
        peaks_hz=np.asarray(phase_table.peaks_hz, dtype=float),
        bond_ids=np.asarray(phase_table.bond_ids, dtype=int),
        relative_phase_rad=np.asarray(phase, dtype=float),
    )


def bond_phase_table_from_result(result: BondPeakPhaseResult) -> LoadedBondPhaseTable:
    return LoadedBondPhaseTable(
        peaks_hz=np.asarray(result.peaks_hz, dtype=float),
        bond_ids=np.asarray(result.bond_ids, dtype=int),
        relative_phase_rad=np.asarray(result.relative_phase_rad, dtype=float),
    )


def build_bond_projection_factors(
    phase_table: LoadedBondPhaseTable,
    target_peaks_hz: list[float] | np.ndarray,
    bond_ids: list[int] | np.ndarray,
    *,
    peak_tol_hz: float = 1e-6,
) -> dict[float, dict[int, float]]:
    target_peaks = np.asarray(target_peaks_hz, dtype=float)
    requested_bond_ids = [int(bond_id) for bond_id in np.asarray(bond_ids, dtype=int).tolist()]
    phase_by_bond_id = {
        int(bond_id): np.asarray(phase_table.relative_phase_rad[row_idx, :], dtype=float)
        for row_idx, bond_id in enumerate(phase_table.bond_ids)
    }

    projection_by_peak: dict[float, dict[int, float]] = {}
    for peak_hz in target_peaks.tolist():
        diffs = np.abs(phase_table.peaks_hz - float(peak_hz))
        col_idx = int(np.argmin(diffs))
        if float(diffs[col_idx]) > float(peak_tol_hz):
            raise ValueError(
                f"Bond phase CSV does not contain a matching peak for {peak_hz:.9g} Hz within tolerance {peak_tol_hz:.3g} Hz"
            )

        bond_projection: dict[int, float] = {}
        for bond_id in requested_bond_ids:
            if bond_id not in phase_by_bond_id:
                raise ValueError(f"Bond phase CSV is missing phase data for bond {bond_id}")
            phase_val = float(phase_by_bond_id[bond_id][col_idx])
            bond_projection[bond_id] = float(np.cos(phase_val))
        projection_by_peak[float(peak_hz)] = bond_projection

    return projection_by_peak
