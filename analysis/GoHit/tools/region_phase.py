from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.interpolate as _interp

from analysis.GoHit.tools.hits import HitRegion
from analysis.tools.bond_phase import LoadedBondPhaseTable
from analysis.tools.models import SignalRecord
from analysis.tools.peaks import assert_peaks_strictly_increasing
from analysis.tools.signal import compute_one_sided_fft_complex, normalize_processed_signal_rms, preprocess_signal


@dataclass(frozen=True)
class RegionBondPhaseEstimate:
    phase_table: LoadedBondPhaseTable
    n_regions_total: int
    n_regions_used: int
    reference_bond_id: int
    reference_peak_hz: float


def _interp_complex(freq: np.ndarray, spectrum: np.ndarray, target_hz: float, *, kind: str = "linear") -> complex:
    if freq.size == 0:
        return complex(np.nan, np.nan)
    if target_hz < float(freq[0]) or target_hz > float(freq[-1]):
        return complex(np.nan, np.nan)
    if kind == "linear":
        real = np.interp(float(target_hz), freq, np.real(spectrum))
        imag = np.interp(float(target_hz), freq, np.imag(spectrum))
    else:
        min_pts = 3 if kind == "quadratic" else 4
        if freq.size < min_pts:
            real = np.interp(float(target_hz), freq, np.real(spectrum))
            imag = np.interp(float(target_hz), freq, np.imag(spectrum))
        else:
            fn_real = _interp.interp1d(freq, np.real(spectrum), kind=kind, bounds_error=False, fill_value="extrapolate")
            fn_imag = _interp.interp1d(freq, np.imag(spectrum), kind=kind, bounds_error=False, fill_value="extrapolate")
            real = float(fn_real(float(target_hz)))
            imag = float(fn_imag(float(target_hz)))
    return complex(real, imag)


def _pick_reference_frequency(freq: np.ndarray, amp: np.ndarray, target_hz: float, search_width_hz: float) -> float | None:
    mask = (freq >= (float(target_hz) - float(search_width_hz))) & (freq <= (float(target_hz) + float(search_width_hz)))
    if np.any(mask):
        idx_local = np.where(mask)[0]
        best_idx = int(idx_local[np.argmax(amp[idx_local])])
        return float(freq[best_idx])
    if freq.size == 0:
        return None
    idx = int(np.argmin(np.abs(freq - float(target_hz))))
    if abs(float(freq[idx]) - float(target_hz)) <= float(search_width_hz):
        return float(freq[idx])
    return None


def _preprocess_records(
    records: list[SignalRecord],
    *,
    longest: bool,
    handlenan: bool,
    timeseriesnorm: bool,
    min_samples: int,
) -> dict[int, tuple[SignalRecord, object]]:
    processed_by_bond: dict[int, tuple[SignalRecord, object]] = {}
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
        if timeseriesnorm and record.signal_kind == "bond":
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                raise ValueError(
                    f"Could not RMS-normalize bond {record.entity_id} in dataset '{record.dataset_name}': {norm_error}"
                )
            processed = processed_norm
        processed_by_bond[int(record.entity_id)] = (record, processed)
    if not processed_by_bond:
        raise ValueError("No bond records were available")
    return processed_by_bond


def estimate_region_bond_phases(
    records: list[SignalRecord],
    peaks_hz: list[float] | np.ndarray,
    *,
    regions: list[HitRegion],
    reference_bond_id: int,
    reference_peak_index: int,
    search_width_hz: float,
    min_reference_fraction: float,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    min_samples: int = 10,
) -> RegionBondPhaseEstimate:
    peaks = assert_peaks_strictly_increasing(peaks_hz)
    if not (1 <= int(reference_peak_index) <= len(peaks)):
        raise ValueError(f"reference_peak_index must be between 1 and {len(peaks)}")

    processed_by_bond = _preprocess_records(
        records,
        longest=longest,
        handlenan=handlenan,
        timeseriesnorm=timeseriesnorm,
        min_samples=min_samples,
    )
    bond_ids = np.asarray(sorted(processed_by_bond.keys()), dtype=int)
    if int(reference_bond_id) not in processed_by_bond:
        raise ValueError(f"Reference bond {reference_bond_id} was not found in the configured records")

    n_regions = len(regions)
    n_bonds = len(bond_ids)
    n_peaks = len(peaks)
    responses = np.full((n_regions, n_bonds, n_peaks), np.nan + 0j, dtype=complex)

    ref_peak_hz = float(peaks[int(reference_peak_index) - 1])
    ref_bond_idx = int(np.where(bond_ids == int(reference_bond_id))[0][0])

    for region_idx, region in enumerate(regions):
        ref_record, ref_processed = processed_by_bond[int(reference_bond_id)]
        ref_mask = (ref_processed.t >= float(region.start_s)) & (ref_processed.t <= float(region.stop_s))
        ref_seg = np.asarray(ref_processed.y[ref_mask], dtype=float)
        if ref_seg.size < max(16, min_samples):
            continue
        ref_fft = compute_one_sided_fft_complex(ref_seg, ref_processed.dt)

        region_target_freqs: list[float | None] = []
        for peak_hz in peaks.tolist():
            region_target_freqs.append(
                _pick_reference_frequency(ref_fft.freq, ref_fft.amplitude, float(peak_hz), float(search_width_hz))
            )

        for bond_col, bond_id in enumerate(bond_ids.tolist()):
            _, processed = processed_by_bond[int(bond_id)]
            mask = (processed.t >= float(region.start_s)) & (processed.t <= float(region.stop_s))
            seg = np.asarray(processed.y[mask], dtype=float)
            if seg.size < max(16, min_samples):
                continue
            fft = compute_one_sided_fft_complex(seg, processed.dt)
            for peak_idx, chosen_freq in enumerate(region_target_freqs):
                if chosen_freq is None:
                    continue
                responses[region_idx, bond_col, peak_idx] = _interp_complex(fft.freq, fft.spectrum, chosen_freq)

    ref_vals = responses[:, ref_bond_idx, int(reference_peak_index) - 1]
    ref_amp = np.abs(ref_vals)
    valid_ref = np.isfinite(ref_amp) & (ref_amp > 0)
    if not np.any(valid_ref):
        raise ValueError(
            f"Reference bond {reference_bond_id} at peak {ref_peak_hz:.6g} Hz had no usable HIT regions"
        )
    ref_median = float(np.median(ref_amp[valid_ref]))
    ref_threshold = float(min_reference_fraction) * ref_median
    keep_mask = valid_ref & (ref_amp >= ref_threshold)
    if not np.any(keep_mask):
        raise ValueError(
            f"Reference gating removed all HIT regions; median amplitude={ref_median:.6g}, "
            f"threshold={ref_threshold:.6g}"
        )

    rotations = np.ones(n_regions, dtype=complex)
    rotations[keep_mask] = np.exp(-1j * np.angle(ref_vals[keep_mask]))

    phase = np.full((n_bonds, n_peaks), np.nan, dtype=float)
    for peak_idx in range(n_peaks):
        for bond_col in range(n_bonds):
            region_vals = responses[:, bond_col, peak_idx]
            usable = keep_mask & np.isfinite(np.real(region_vals)) & np.isfinite(np.imag(region_vals))
            if not np.any(usable):
                continue
            rotated = region_vals[usable] * rotations[usable]
            mean_complex = np.mean(rotated)
            phase[bond_col, peak_idx] = float(np.angle(mean_complex))

    return RegionBondPhaseEstimate(
        phase_table=LoadedBondPhaseTable(
            peaks_hz=np.asarray(peaks, dtype=float),
            bond_ids=np.asarray(bond_ids, dtype=int),
            relative_phase_rad=np.asarray(phase, dtype=float),
        ),
        n_regions_total=n_regions,
        n_regions_used=int(np.sum(keep_mask)),
        reference_bond_id=int(reference_bond_id),
        reference_peak_hz=float(ref_peak_hz),
    )
