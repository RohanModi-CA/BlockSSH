from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from analysis.GoHit.tools.baseline_peaks import local_baseline_subtracted_peak_amplitude
from analysis.GoHit.tools.hits import HitRegion
from analysis.tools.bond_phase import LoadedBondPhaseTable
from analysis.tools.flattening import compute_flattening
from analysis.tools.localization import (
    LocalizationPeakDiagnostic,
    build_peak_diagnostics,
    compute_normalization_factor,
    get_peak_amplitude,
    get_peak_sqrt_integrated_power,
)
from analysis.tools.models import FFTResult, LocalizationProfile, SignalRecord, SpectrumContribution
from analysis.tools.signal import compute_one_sided_fft_complex, normalize_processed_signal_rms, preprocess_signal
from analysis.tools.spectral import compute_mean_amplitude_spectrum, process_spectrum_window


@dataclass(frozen=True)
class RegionSpectrumEntry:
    record: SignalRecord
    processed: object
    region: HitRegion
    freq: np.ndarray
    amplitude: np.ndarray
    spectrum: np.ndarray


def _phase_lookup(
    phase_table: LoadedBondPhaseTable | None,
    *,
    bond_id: int,
    peak_hz: float,
    peak_tol_hz: float = 1e-6,
) -> float | None:
    if phase_table is None:
        return None
    diffs = np.abs(np.asarray(phase_table.peaks_hz, dtype=float) - float(peak_hz))
    col_idx = int(np.argmin(diffs))
    if float(diffs[col_idx]) > float(peak_tol_hz):
        raise ValueError(f"Phase table does not contain a matching peak for {peak_hz:.9g} Hz")
    row_matches = np.where(np.asarray(phase_table.bond_ids, dtype=int) == int(bond_id))[0]
    if row_matches.size == 0:
        raise ValueError(f"Phase table does not contain bond {bond_id}")
    return float(phase_table.relative_phase_rad[int(row_matches[0]), col_idx])


def _representative_complex(freq: np.ndarray, spectrum: np.ndarray, target: float, width: float) -> complex | None:
    if freq.size == 0:
        return None
    mask = (freq >= (float(target) - float(width))) & (freq <= (float(target) + float(width)))
    if np.any(mask):
        idx_local = np.where(mask)[0]
        amps = np.abs(spectrum[idx_local])
        return complex(spectrum[int(idx_local[np.argmax(amps)])])
    idx = int(np.argmin(np.abs(freq - float(target))))
    if abs(float(freq[idx]) - float(target)) <= float(width):
        return complex(spectrum[idx])
    return None


def _effective_regions(processed, regions: list[HitRegion] | None) -> list[HitRegion]:
    if regions is not None:
        return regions
    return [
        HitRegion(
            index=1,
            start_s=float(processed.t[0]),
            stop_s=float(processed.t[-1]),
            mode="full",
        )
    ]


def build_region_spectrum_entries(
    records: list[SignalRecord],
    *,
    regions: list[HitRegion] | None,
    normalize_mode: str,
    relative_range: tuple[float, float],
    flatten: bool = False,
    flatten_reference_band: tuple[float, float] = (20.0, 30.0),
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    min_samples: int = 10,
) -> list[RegionSpectrumEntry]:
    raw_entries: list[RegionSpectrumEntry] = []

    for record in records:
        processed, error_msg = preprocess_signal(
            record.t,
            record.y,
            longest=longest,
            handlenan=handlenan,
            min_samples=min_samples,
        )
        if processed is None:
            warnings.warn(
                f"{record.signal_kind.capitalize()} {record.entity_id} in dataset '{record.dataset_name}' "
                f"has invalid signal ({error_msg}); omitting from transformed spectra"
            )
            continue

        if timeseriesnorm and record.signal_kind == "bond":
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                warnings.warn(
                    f"{record.signal_kind.capitalize()} {record.entity_id} in dataset '{record.dataset_name}' "
                    f"could not be RMS-normalized ({norm_error}); omitting from transformed spectra"
                )
                continue
            processed = processed_norm

        for region in _effective_regions(processed, regions):
            mask = (processed.t >= float(region.start_s)) & (processed.t <= float(region.stop_s))
            seg = np.asarray(processed.y[mask], dtype=float)
            if seg.size < max(16, min_samples):
                continue
            fft = compute_one_sided_fft_complex(seg, processed.dt)
            norm_factor = compute_normalization_factor(
                fft.freq,
                fft.amplitude,
                normalize_mode,
                relative_range,
            )
            if norm_factor <= 1e-12 or not np.isfinite(norm_factor):
                continue
            raw_entries.append(
                RegionSpectrumEntry(
                    record=record,
                    processed=processed,
                    region=region,
                    freq=np.asarray(fft.freq, dtype=float),
                    amplitude=np.asarray(fft.amplitude / norm_factor, dtype=float),
                    spectrum=np.asarray(fft.spectrum / norm_factor, dtype=complex),
                )
            )

    if not flatten or not raw_entries:
        return raw_entries

    flattenings = [
        compute_flattening(entry.freq, entry.amplitude, reference_band=flatten_reference_band)
        for entry in raw_entries
    ]
    shared_target = float(np.median([flat.reference_level for flat in flattenings]))

    transformed: list[RegionSpectrumEntry] = []
    for entry, flat in zip(raw_entries, flattenings):
        transfer = shared_target / np.maximum(flat.baseline_smooth, np.finfo(float).tiny)
        transformed.append(
            RegionSpectrumEntry(
                record=entry.record,
                processed=entry.processed,
                region=entry.region,
                freq=np.asarray(entry.freq, dtype=float),
                amplitude=np.asarray(entry.amplitude * transfer, dtype=float),
                spectrum=np.asarray(entry.spectrum * transfer, dtype=complex),
            )
        )
    return transformed


def compute_region_localization_profiles(
    records: list[SignalRecord],
    peak_targets: list[tuple[int, float]],
    *,
    regions: list[HitRegion] | None,
    normalize_mode: str,
    relative_range: tuple[float, float],
    search_width: float,
    phase_table: LoadedBondPhaseTable | None = None,
    baseline_subtract: bool = False,
    flatten: bool = False,
    flatten_reference_band: tuple[float, float] = (20.0, 30.0),
    sqrtintpower: bool = False,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    min_samples: int = 10,
) -> list[LocalizationProfile]:
    entries = build_region_spectrum_entries(
        records,
        regions=regions,
        normalize_mode=normalize_mode,
        relative_range=relative_range,
        flatten=flatten,
        flatten_reference_band=flatten_reference_band,
        longest=longest,
        handlenan=handlenan,
        timeseriesnorm=timeseriesnorm,
        min_samples=min_samples,
    )
    return compute_localization_profiles_from_entries(
        entries,
        peak_targets,
        search_width=search_width,
        phase_table=phase_table,
        baseline_subtract=baseline_subtract,
        sqrtintpower=sqrtintpower,
    )


def compute_localization_profiles_from_entries(
    entries: list[RegionSpectrumEntry],
    peak_targets: list[tuple[int, float]],
    *,
    search_width: float,
    phase_table: LoadedBondPhaseTable | None = None,
    baseline_subtract: bool = False,
    sqrtintpower: bool = False,
) -> list[LocalizationProfile]:
    data_store: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        for peak_index, target_freq in peak_targets:
            if baseline_subtract:
                magnitude, found = local_baseline_subtracted_peak_amplitude(
                    entry.freq,
                    entry.amplitude,
                    target_freq,
                    search_width,
                    sqrtintpower=sqrtintpower,
                )
            else:
                if sqrtintpower:
                    magnitude, found = get_peak_sqrt_integrated_power(
                        entry.freq,
                        entry.amplitude,
                        target_freq,
                        search_width,
                    )
                else:
                    magnitude, found = get_peak_amplitude(
                        entry.freq,
                        entry.amplitude,
                        target_freq,
                        search_width,
                    )

            value = float(magnitude)
            target_phase = _phase_lookup(phase_table, bond_id=int(entry.record.entity_id), peak_hz=float(target_freq))
            if target_phase is not None:
                rep = _representative_complex(entry.freq, entry.spectrum, float(target_freq), float(search_width))
                if rep is not None and np.isfinite(np.real(rep)) and np.isfinite(np.imag(rep)):
                    value = float(magnitude) * float(np.cos(np.angle(rep) - float(target_phase)))

            if not found:
                warnings.warn(
                    f"Could not find peak {peak_index} ({target_freq} Hz) "
                    f"for {entry.record.signal_kind} {entry.record.entity_id} in dataset '{entry.record.dataset_name}' "
                    f"region [{entry.region.start_s:.6g}, {entry.region.stop_s:.6g}]"
                )
            data_store[peak_index][int(entry.record.entity_id)].append(value)

    profiles: list[LocalizationProfile] = []
    for peak_index, frequency in peak_targets:
        entity_data = data_store.get(peak_index, {})
        entity_ids = np.array(sorted(entity_data.keys()), dtype=int)
        if entity_ids.size == 0:
            profiles.append(
                LocalizationProfile(
                    peak_index=int(peak_index),
                    frequency=float(frequency),
                    entity_ids=np.array([], dtype=int),
                    mean_amplitudes=np.array([], dtype=float),
                    std_amplitudes=np.array([], dtype=float),
                )
            )
            continue

        means: list[float] = []
        stds: list[float] = []
        for entity_id in entity_ids:
            vals = np.asarray(entity_data[entity_id], dtype=float)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if vals.size > 1 else 0.0)

        profiles.append(
            LocalizationProfile(
                peak_index=int(peak_index),
                frequency=float(frequency),
                entity_ids=np.asarray(entity_ids, dtype=int),
                mean_amplitudes=np.asarray(means, dtype=float),
                std_amplitudes=np.asarray(stds, dtype=float),
            )
        )

    return profiles


def _entries_to_contributions(entries: list[RegionSpectrumEntry]) -> list[SpectrumContribution]:
    contributions: list[SpectrumContribution] = []
    for entry in entries:
        contributions.append(
            SpectrumContribution(
                record=entry.record,
                processed=entry.processed,
                fft_result=FFTResult(
                    freq=np.asarray(entry.freq, dtype=float),
                    amplitude=np.asarray(entry.amplitude, dtype=float),
                ),
            )
        )
    return contributions


def _baseline_subtracted_diagnostics(
    freq: np.ndarray,
    amplitude: np.ndarray,
    peak_targets: list[tuple[int, float]],
    *,
    search_width: float,
) -> list[LocalizationPeakDiagnostic]:
    diagnostics: list[LocalizationPeakDiagnostic] = []
    display_width = max(0.75, 3.0 * float(search_width))
    for peak_index, target_freq in peak_targets:
        display_low = max(float(freq[0]), float(target_freq) - float(display_width))
        display_high = min(float(freq[-1]), float(target_freq) + float(display_width))
        processed = process_spectrum_window(freq, amplitude, display_low, display_high)
        display_freq = np.asarray(processed.freq, dtype=float)
        display_amp = np.asarray(np.maximum(processed.shifted_amplitude, 0.0), dtype=float)
        selected_amp, found = get_peak_amplitude(display_freq, display_amp, target_freq, search_width)
        selected_freq = float("nan")
        if found:
            mask = (display_freq >= (float(target_freq) - float(search_width))) & (
                display_freq <= (float(target_freq) + float(search_width))
            )
            if np.any(mask):
                idx_local = np.where(mask)[0]
                best_idx = int(idx_local[np.argmax(display_amp[idx_local])])
                selected_freq = float(display_freq[best_idx])
        diagnostics.append(
            LocalizationPeakDiagnostic(
                peak_index=int(peak_index),
                target_frequency=float(target_freq),
                selected_frequency=float(selected_freq),
                found=bool(found and np.isfinite(selected_amp)),
                window_low=float(target_freq - search_width),
                window_high=float(target_freq + search_width),
                display_freq=display_freq,
                display_amplitude=display_amp,
            )
        )
    return diagnostics


def build_region_peak_diagnostics_by_entity(
    entries: list[RegionSpectrumEntry],
    peak_targets: list[tuple[int, float]],
    *,
    search_width: float,
    baseline_subtract: bool = False,
    include_all: bool = True,
) -> dict[str, list[LocalizationPeakDiagnostic]]:
    grouped: dict[str, list[RegionSpectrumEntry]] = {}
    if include_all and entries:
        grouped["All"] = list(entries)

    entity_ids = sorted({int(entry.record.entity_id) for entry in entries})
    for entity_id in entity_ids:
        grouped[str(entity_id)] = [entry for entry in entries if int(entry.record.entity_id) == int(entity_id)]

    diagnostics_by_entity: dict[str, list[LocalizationPeakDiagnostic]] = {}
    for label, grouped_entries in grouped.items():
        contributions = _entries_to_contributions(grouped_entries)
        if not contributions:
            continue
        averaged = compute_mean_amplitude_spectrum(contributions)
        if baseline_subtract:
            diagnostics_by_entity[label] = _baseline_subtracted_diagnostics(
                averaged.freq_grid,
                averaged.mean_amplitude,
                peak_targets,
                search_width=search_width,
            )
        else:
            diagnostics_by_entity[label] = build_peak_diagnostics(
                averaged.freq_grid,
                averaged.mean_amplitude,
                peak_targets,
                search_width=search_width,
            )
    return diagnostics_by_entity
