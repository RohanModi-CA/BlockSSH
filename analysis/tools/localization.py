from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from .models import LocalizationProfile, SignalRecord
from .signal import compute_one_sided_fft, preprocess_signal
from .spectral import integral_over_window


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


def compute_normalization_factor(
    freq: np.ndarray,
    amp: np.ndarray,
    mode: str,
    relative_range: tuple[float, float],
) -> float:
    if mode == "absolute":
        val = np.trapz(amp, freq)
    elif mode == "relative":
        low, high = relative_range
        val = integral_over_window(freq, amp, low, high)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    return float(val)


def compute_localization_profiles(
    records: list[SignalRecord],
    peak_targets: list[tuple[int, float]],
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
    search_width: float = 0.25,
    longest: bool = False,
    handlenan: bool = False,
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

        fft_result = compute_one_sided_fft(processed.y, processed.dt)
        freqs = fft_result.freq
        amps = fft_result.amplitude

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
            val, found = get_peak_amplitude(freqs, normalized_amps, target_freq, search_width)
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
