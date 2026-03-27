from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from .models import LocalizationPeakDiagnostic, LocalizationProfile, SignalRecord
from .signal import (
    compute_one_sided_fft,
    compute_welch_spectrum,
    normalize_processed_signal_rms,
    preprocess_signal,
)
from .spectral import compute_average_spectrum, integral_over_window, resolve_normalization_window


def get_peak_amplitude(
    freqs: np.ndarray,
    amps: np.ndarray,
    target: float,
    width: float,
) -> tuple[float, bool]:
    f_min = target - width
    f_max = target + width

    if freqs.size == 0:
        return 0.0, False

    if f_max < freqs[0] or f_min > freqs[-1]:
        return 0.0, False

    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(freqs - target)))
        nearest_f = freqs[idx]
        if abs(nearest_f - target) <= width:
            return float(amps[idx]), True
        return 0.0, False

    return float(np.max(amps[mask])), True


def get_peak_sqrt_integrated_power(
    freqs: np.ndarray,
    amps: np.ndarray,
    target: float,
    width: float,
) -> tuple[float, bool]:
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)

    f_min = target - width
    f_max = target + width

    if freqs.size == 0:
        return 0.0, False

    if f_max < freqs[0] or f_min > freqs[-1]:
        return 0.0, False

    low = max(float(freqs[0]), float(f_min))
    high = min(float(freqs[-1]), float(f_max))
    if high <= low:
        return 0.0, False

    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0, False

    power = np.square(np.abs(amps))
    integrated_power = integral_over_window(freqs, power, low, high)
    if not np.isfinite(integrated_power) or integrated_power <= 0.0:
        return 0.0, False
    return float(np.sqrt(integrated_power)), True


def compute_normalization_factor(
    freq: np.ndarray,
    amp: np.ndarray,
    mode: str,
    relative_range: tuple[float, float],
) -> float:
    low, high = resolve_normalization_window(
        float(freq[0]),
        float(freq[-1]),
        normalize_mode=mode,
        relative_range=relative_range,
    )
    val = integral_over_window(freq, amp, low, high)
    return float(val)


def compute_localization_profiles(
    records: list[SignalRecord],
    peak_targets: list[tuple[int, float]],
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
    search_width: float = 0.25,
    spectrum_kind: str = "fft",
    welch_len_s: float = 100.0,
    welch_overlap_fraction: float = 0.5,
    projection_factors_by_peak: dict[float, dict[int, float]] | None = None,
    sqrtintpower: bool = False,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    min_samples: int = 10,
) -> list[LocalizationProfile]:
    data_store: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

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
                f"has invalid signal ({error_msg}); recording 0 amplitudes"
            )
            for peak_index, _ in peak_targets:
                data_store[peak_index][record.entity_id].append(0.0)
            continue

        if timeseriesnorm and record.signal_kind == "bond":
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                warnings.warn(
                    f"{record.signal_kind.capitalize()} {record.entity_id} in dataset '{record.dataset_name}' "
                    f"could not be RMS-normalized ({norm_error}); recording 0 amplitudes"
                )
                for peak_index, _ in peak_targets:
                    data_store[peak_index][record.entity_id].append(0.0)
                continue
            processed = processed_norm

        if spectrum_kind == "fft":
            spectrum_result = compute_one_sided_fft(processed.y, processed.dt)
        elif spectrum_kind == "welch":
            spectrum_result = compute_welch_spectrum(
                processed.y,
                processed.Fs,
                welch_len_s,
                overlap_fraction=welch_overlap_fraction,
            )
            if spectrum_result is None:
                warnings.warn(
                    f"{record.signal_kind.capitalize()} {record.entity_id} in dataset '{record.dataset_name}' "
                    "has invalid Welch spectrum; recording 0 amplitudes"
                )
                for peak_index, _ in peak_targets:
                    data_store[peak_index][record.entity_id].append(0.0)
                continue
        else:
            raise ValueError(f"Unsupported spectrum kind: {spectrum_kind}")

        freqs = spectrum_result.freq
        amps = spectrum_result.amplitude

        norm_factor = compute_normalization_factor(
            freqs,
            amps,
            normalize_mode,
            relative_range,
        )
        if norm_factor <= 1e-12 or not np.isfinite(norm_factor):
            warnings.warn(
                f"{record.signal_kind.capitalize()} {record.entity_id} in dataset '{record.dataset_name}' "
                "has zero or invalid normalization factor; recording 0 amplitudes"
            )
            for peak_index, _ in peak_targets:
                data_store[peak_index][record.entity_id].append(0.0)
            continue

        normalized_amps = amps / norm_factor

        for peak_index, target_freq in peak_targets:
            if sqrtintpower:
                val, found = get_peak_sqrt_integrated_power(freqs, normalized_amps, target_freq, search_width)
            else:
                val, found = get_peak_amplitude(freqs, normalized_amps, target_freq, search_width)
            if projection_factors_by_peak is not None:
                peak_projection = projection_factors_by_peak.get(float(target_freq))
                if peak_projection is None:
                    raise ValueError(f"Missing projection factors for peak {target_freq:.9g} Hz")
                if int(record.entity_id) not in peak_projection:
                    raise ValueError(
                        f"Missing projection factor for entity {record.entity_id} at peak {target_freq:.9g} Hz"
                    )
                val *= float(peak_projection[int(record.entity_id)])
            if not found:
                warnings.warn(
                    f"Could not find peak {peak_index} ({target_freq} Hz) "
                    f"for {record.signal_kind} {record.entity_id} in dataset '{record.dataset_name}'; recording 0"
                )
            data_store[peak_index][record.entity_id].append(val)

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
                entity_ids=entity_ids,
                mean_amplitudes=np.asarray(means, dtype=float),
                std_amplitudes=np.asarray(stds, dtype=float),
            )
        )

    return profiles


def build_peak_diagnostics(
    freqs: np.ndarray,
    amps: np.ndarray,
    peak_targets: list[tuple[int, float]],
    *,
    search_width: float,
    display_width: float | None = None,
) -> list[LocalizationPeakDiagnostic]:
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)

    if freqs.ndim != 1 or amps.ndim != 1 or freqs.size != amps.size:
        raise ValueError("freqs and amps must be 1D arrays of equal length")

    if freqs.size == 0:
        raise ValueError("freqs must not be empty")

    if display_width is None:
        display_width = max(0.75, 3.0 * float(search_width))

    diagnostics: list[LocalizationPeakDiagnostic] = []
    for peak_index, target_freq in peak_targets:
        selected_amp, found = get_peak_amplitude(freqs, amps, target_freq, search_width)

        window_mask = (freqs >= (target_freq - search_width)) & (freqs <= (target_freq + search_width))
        selected_freq = float("nan")
        if found:
            if np.any(window_mask):
                window_indices = np.where(window_mask)[0]
                local_amps = amps[window_indices]
                best_local = int(np.argmax(local_amps))
                selected_freq = float(freqs[window_indices[best_local]])
            else:
                idx = int(np.argmin(np.abs(freqs - target_freq)))
                if abs(float(freqs[idx]) - float(target_freq)) <= float(search_width):
                    selected_freq = float(freqs[idx])

        display_low = max(float(freqs[0]), float(target_freq) - float(display_width))
        display_high = min(float(freqs[-1]), float(target_freq) + float(display_width))
        display_mask = (freqs >= display_low) & (freqs <= display_high)

        if np.any(display_mask):
            display_freq = freqs[display_mask]
            display_amp = amps[display_mask]
        else:
            idx = int(np.argmin(np.abs(freqs - target_freq)))
            display_freq = freqs[idx : idx + 1]
            display_amp = amps[idx : idx + 1]

        diagnostics.append(
            LocalizationPeakDiagnostic(
                peak_index=int(peak_index),
                target_frequency=float(target_freq),
                selected_frequency=float(selected_freq),
                found=bool(found and np.isfinite(selected_amp)),
                window_low=float(target_freq - search_width),
                window_high=float(target_freq + search_width),
                display_freq=np.asarray(display_freq, dtype=float),
                display_amplitude=np.asarray(display_amp, dtype=float),
            )
        )

    return diagnostics


def build_peak_diagnostics_by_entity(
    contributions,
    peak_targets: list[tuple[int, float]],
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
    search_width: float,
    include_all: bool = True,
) -> dict[str, list[LocalizationPeakDiagnostic]]:
    grouped: dict[str, list] = {}
    if include_all and contributions:
        grouped["All"] = list(contributions)

    entity_ids = sorted({int(contrib.record.entity_id) for contrib in contributions})
    for entity_id in entity_ids:
        grouped[str(entity_id)] = [
            contrib for contrib in contributions if int(contrib.record.entity_id) == int(entity_id)
        ]

    diagnostics_by_entity: dict[str, list[LocalizationPeakDiagnostic]] = {}
    for label, entity_contribs in grouped.items():
        if not entity_contribs:
            continue
        averaged = compute_average_spectrum(
            entity_contribs,
            normalize_mode=normalize_mode,
            relative_range=relative_range,
            average_domain="linear",
        )
        diagnostics_by_entity[label] = build_peak_diagnostics(
            averaged.freq_grid,
            averaged.avg_amp,
            peak_targets,
            search_width=search_width,
        )

    return diagnostics_by_entity
