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
class SitePeakPhaseResult:
    dataset_name: str
    site_ids: np.ndarray
    peaks_hz: np.ndarray
    selected_freq_hz: np.ndarray
    relative_phase_rad: np.ndarray
    mode_magnitude: np.ndarray
    coherence_fraction: np.ndarray
    reference_site_id: int
    n_windows_used: np.ndarray


@dataclass(frozen=True)
class LoadedSitePhaseTable:
    peaks_hz: np.ndarray
    site_ids: np.ndarray
    relative_phase_rad: np.ndarray


def _enabled_dataset_names(records: list[SignalRecord]) -> list[str]:
    return sorted({str(record.dataset_name) for record in records})


def _align_processed_site_signals(
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
            f"Expected exactly one enabled dataset for phase generation, found {len(dataset_names)}: {dataset_names}"
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
                f"Could not preprocess site {record.entity_id} in dataset '{record.dataset_name}': {error_msg}"
            )
        processed_rows.append((int(record.entity_id), processed))

    if len(processed_rows) < 2:
        raise ValueError("Need at least two site signals to estimate relative phase")

    common_start = max(float(proc.t[0]) for _, proc in processed_rows)
    common_stop = min(float(proc.t[-1]) for _, proc in processed_rows)
    if common_stop <= common_start:
        raise ValueError("No overlapping time span across the selected site signals")

    dts = np.asarray([float(proc.dt) for _, proc in processed_rows], dtype=float)
    dt = float(np.median(dts))
    if not np.allclose(dts, dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Preprocessed site signals do not share a consistent sampling interval")

    t_common = np.arange(common_start, common_stop + 0.5 * dt, dt, dtype=float)
    t_common = t_common[t_common <= (common_stop + 1e-12)]
    if t_common.size < max(16, min_samples):
        raise ValueError("Common aligned time grid is too short for phase estimation")

    site_ids = np.asarray([site_id for site_id, _ in processed_rows], dtype=int)
    signals = np.empty((t_common.size, len(processed_rows)), dtype=float)
    for col, (_, proc) in enumerate(processed_rows):
        signals[:, col] = np.interp(t_common, proc.t, proc.y)

    return t_common, signals, site_ids, dataset_names[0]


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


def estimate_site_peak_phases(
    records: list[SignalRecord],
    peaks_hz: list[float] | np.ndarray,
    *,
    welch_len_s: float = 100.0,
    welch_overlap_fraction: float = 0.5,
    search_width_hz: float = 0.25,
    longest: bool = False,
    handlenan: bool = False,
    min_samples: int = 10,
) -> SitePeakPhaseResult:
    peaks_arr = assert_peaks_strictly_increasing(peaks_hz)
    t_common, signals, site_ids, dataset_name = _align_processed_site_signals(
        records,
        longest=longest,
        handlenan=handlenan,
        min_samples=min_samples,
    )

    dt = float(np.median(np.diff(t_common)))
    fs = 1.0 / dt

    specs = []
    freq_grid = None
    time_grid = None
    for site_idx in range(signals.shape[1]):
        freq, time, s_complex = _compute_complex_spectrogram_with_overlap(
            signals[:, site_idx],
            fs,
            welch_len_s,
            welch_overlap_fraction,
        )
        if freq_grid is None:
            freq_grid = freq
            time_grid = time
        else:
            if s_complex.shape != specs[0].shape:
                raise ValueError("Site spectrogram shapes did not match")
            if not np.allclose(freq, freq_grid, rtol=1e-6, atol=1e-9):
                raise ValueError("Site spectrogram frequency grids did not match")
            if not np.allclose(time, time_grid, rtol=1e-6, atol=1e-9):
                raise ValueError("Site spectrogram time grids did not match")
        specs.append(s_complex)

    if freq_grid is None or time_grid is None or len(specs) == 0:
        raise ValueError("No complex spectrogram data were produced")

    n_sites = len(site_ids)
    n_peaks = len(peaks_arr)
    relative_phase = np.full((n_sites, n_peaks), np.nan, dtype=float)
    mode_magnitude = np.full((n_sites, n_peaks), np.nan, dtype=float)
    selected_freq_hz = np.full(n_peaks, np.nan, dtype=float)
    coherence_fraction = np.full(n_peaks, np.nan, dtype=float)
    n_windows_used = np.zeros(n_peaks, dtype=int)

    spectra_by_site = np.stack([np.asarray(spec, dtype=complex) for spec in specs], axis=2)
    reference_site_id = int(site_ids[0])

    for peak_idx, peak_hz in enumerate(peaks_arr):
        freq_idx = _pick_frequency_index(freq_grid, float(peak_hz), float(search_width_hz))
        selected_freq_hz[peak_idx] = float(freq_grid[freq_idx])

        site_window_vectors = np.asarray(spectra_by_site[freq_idx, :, :], dtype=complex)
        if site_window_vectors.ndim != 2 or site_window_vectors.shape[1] != n_sites:
            raise ValueError("Unexpected complex spectrogram slice shape")

        energies = np.linalg.norm(site_window_vectors, axis=1)
        finite_mask = np.all(np.isfinite(site_window_vectors), axis=1)
        valid_mask = finite_mask & np.isfinite(energies) & (energies > 0)
        if not np.any(valid_mask):
            raise ValueError(f"Peak {peak_hz:.6g} Hz had no valid spectrogram windows")

        valid_vectors = site_window_vectors[valid_mask]
        n_windows_used[peak_idx] = int(valid_vectors.shape[0])

        csd = np.einsum("wi,wj->ij", valid_vectors, np.conjugate(valid_vectors)) / float(valid_vectors.shape[0])
        evals, evecs = np.linalg.eigh(csd)
        lead_idx = int(np.argmax(np.real(evals)))
        mode = np.asarray(evecs[:, lead_idx], dtype=complex)

        total_power = float(np.sum(np.clip(np.real(evals), 0.0, None)))
        lead_power = float(max(np.real(evals[lead_idx]), 0.0))
        coherence_fraction[peak_idx] = lead_power / total_power if total_power > 0 else np.nan

        ref_val = mode[0]
        if abs(ref_val) <= 1e-12:
            raise ValueError(
                f"Reference site {reference_site_id} had near-zero modal magnitude at {peak_hz:.6g} Hz"
            )
        mode = mode * np.exp(-1j * np.angle(ref_val))

        relative_phase[:, peak_idx] = np.angle(mode)
        mode_magnitude[:, peak_idx] = np.abs(mode)

    return SitePeakPhaseResult(
        dataset_name=dataset_name,
        site_ids=site_ids,
        peaks_hz=peaks_arr,
        selected_freq_hz=selected_freq_hz,
        relative_phase_rad=relative_phase,
        mode_magnitude=mode_magnitude,
        coherence_fraction=coherence_fraction,
        reference_site_id=reference_site_id,
        n_windows_used=n_windows_used,
    )


def write_site_peak_phase_csv(result: SitePeakPhaseResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["site_id", *[f"{peak:.9g}" for peak in result.peaks_hz]])
        for row_idx, site_id in enumerate(result.site_ids):
            row = [int(site_id), *[f"{val:.12g}" for val in result.relative_phase_rad[row_idx, :]]]
            writer.writerow(row)

    return output_path


def load_site_peak_phase_csv(path: str | Path) -> LoadedSitePhaseTable:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise ValueError("Phase CSV must contain a header row and at least one site row")

    header = rows[0]
    if len(header) < 2:
        raise ValueError("Phase CSV header must contain 'site_id' and at least one peak column")
    if header[0].strip().lower() != "site_id":
        raise ValueError("Phase CSV first column must be 'site_id'")

    try:
        peaks_hz = np.asarray([float(cell) for cell in header[1:] if str(cell).strip() != ""], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse phase CSV header peak frequencies: {exc}") from exc

    if peaks_hz.size != (len(header) - 1):
        raise ValueError("Phase CSV header contains empty peak columns")
    if peaks_hz.size == 0:
        raise ValueError("Phase CSV must contain at least one peak column")

    site_ids: list[int] = []
    phase_rows: list[list[float]] = []
    for row in rows[1:]:
        if len(row) == 0 or all(str(cell).strip() == "" for cell in row):
            continue
        if len(row) != len(header):
            raise ValueError("Each phase CSV row must have the same number of columns as the header")
        try:
            site_id = int(row[0])
        except ValueError as exc:
            raise ValueError(f"Could not parse site_id '{row[0]}'") from exc
        try:
            phases = [float(cell) for cell in row[1:]]
        except ValueError as exc:
            raise ValueError(f"Could not parse phase row for site {site_id}: {exc}") from exc
        site_ids.append(site_id)
        phase_rows.append(phases)

    if len(site_ids) == 0:
        raise ValueError("Phase CSV contained no site rows")

    return LoadedSitePhaseTable(
        peaks_hz=np.asarray(peaks_hz, dtype=float),
        site_ids=np.asarray(site_ids, dtype=int),
        relative_phase_rad=np.asarray(phase_rows, dtype=float),
    )


def build_bond_projection_factors(
    phase_table: LoadedSitePhaseTable,
    target_peaks_hz: list[float] | np.ndarray,
    bond_ids: list[int] | np.ndarray,
    *,
    peak_tol_hz: float = 1e-6,
) -> dict[float, dict[int, float]]:
    target_peaks = np.asarray(target_peaks_hz, dtype=float)
    unique_bond_ids = [int(bond_id) for bond_id in np.asarray(bond_ids, dtype=int).tolist()]
    site_phase_by_id = {
        int(site_id): np.asarray(phase_table.relative_phase_rad[row_idx, :], dtype=float)
        for row_idx, site_id in enumerate(phase_table.site_ids)
    }

    projection_by_peak: dict[float, dict[int, float]] = {}
    for peak_hz in target_peaks.tolist():
        diffs = np.abs(phase_table.peaks_hz - float(peak_hz))
        col_idx = int(np.argmin(diffs))
        if float(diffs[col_idx]) > float(peak_tol_hz):
            raise ValueError(
                f"Phase CSV does not contain a matching peak for {peak_hz:.9g} Hz within tolerance {peak_tol_hz:.3g} Hz"
            )

        bond_projection: dict[int, float] = {}
        for bond_id in unique_bond_ids:
            left_site = int(bond_id)
            right_site = int(bond_id + 1)
            if left_site not in site_phase_by_id or right_site not in site_phase_by_id:
                raise ValueError(
                    f"Phase CSV is missing site phase data needed for bond {bond_id} (sites {left_site}, {right_site})"
                )

            left = np.exp(1j * complex(site_phase_by_id[left_site][col_idx]))
            right = np.exp(1j * complex(site_phase_by_id[right_site][col_idx]))
            bond_mode = right - left
            mag = float(np.abs(bond_mode))
            bond_projection[bond_id] = float(np.real(bond_mode) / mag) if mag > 1e-12 else 0.0

        projection_by_peak[float(peak_hz)] = bond_projection

    return projection_by_peak
