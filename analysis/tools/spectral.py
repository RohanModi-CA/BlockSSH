from __future__ import annotations

import warnings

import numpy as np
import scipy.signal as sp_signal

from .models import (
    AverageSpectrumResult,
    AveragedAmplitudeSpectrum,
    PairFrequencyAnalysisResult,
    PairWelchFrequencyAnalysisResult,
    ProcessedSpectrumWindow,
    SignalRecord,
    SpectrumContribution,
    SpacingDataset,
)
from .signal import (
    compute_complex_spectrogram,
    compute_one_sided_fft,
    compute_welch_spectrum,
    normalize_processed_signal_rms,
    preprocess_signal,
)


ABSOLUTE_ZERO_TOL = 1e-10
ABSOLUTE_NORMALIZATION_MARGIN_HZ = 0.5


def is_close_to_zero(value: float, *, tol: float = ABSOLUTE_ZERO_TOL) -> bool:
    return (not np.isfinite(value)) or (abs(float(value)) <= float(tol))


def compute_fft_contributions(
    records: list[SignalRecord],
    *,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    min_samples: int = 10,
) -> list[SpectrumContribution]:
    contributions: list[SpectrumContribution] = []

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
                f"Skipping {record.signal_kind} id {record.entity_id} from dataset '{record.dataset_name}' "
                f"because preprocessing failed: {error_msg}"
            )
            continue

        if timeseriesnorm and record.signal_kind == "bond":
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                warnings.warn(
                    f"Skipping {record.signal_kind} id {record.entity_id} from dataset '{record.dataset_name}' "
                    f"because time-series RMS normalization failed: {norm_error}"
                )
                continue
            processed = processed_norm

        fft_result = compute_one_sided_fft(processed.y, processed.dt)
        if fft_result.freq.size < 2 or fft_result.amplitude.size != fft_result.freq.size:
            warnings.warn(
                f"Skipping {record.signal_kind} id {record.entity_id} from dataset '{record.dataset_name}' "
                "because FFT output was invalid"
            )
            continue

        contributions.append(
            SpectrumContribution(
                record=record,
                processed=processed,
                fft_result=fft_result,
            )
        )

    return contributions


def compute_welch_contributions(
    records: list[SignalRecord],
    *,
    welch_len_s: float = 100.0,
    welch_overlap_fraction: float = 0.5,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    min_samples: int = 10,
) -> list[SpectrumContribution]:
    contributions: list[SpectrumContribution] = []

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
                f"Skipping {record.signal_kind} id {record.entity_id} from dataset '{record.dataset_name}' "
                f"because preprocessing failed: {error_msg}"
            )
            continue

        if timeseriesnorm and record.signal_kind == "bond":
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                warnings.warn(
                    f"Skipping {record.signal_kind} id {record.entity_id} from dataset '{record.dataset_name}' "
                    f"because time-series RMS normalization failed: {norm_error}"
                )
                continue
            processed = processed_norm

        welch_result = compute_welch_spectrum(
            processed.y,
            processed.Fs,
            welch_len_s,
            overlap_fraction=welch_overlap_fraction,
        )
        if welch_result is None or welch_result.freq.size < 2 or welch_result.amplitude.size != welch_result.freq.size:
            warnings.warn(
                f"Skipping {record.signal_kind} id {record.entity_id} from dataset '{record.dataset_name}' "
                "because Welch output was invalid"
            )
            continue

        contributions.append(
            SpectrumContribution(
                record=record,
                processed=processed,
                fft_result=welch_result,
            )
        )

    return contributions


def analyze_spacing_dataset_for_display(
    spacing_dataset: SpacingDataset,
    *,
    disabled_indices: list[int] | None = None,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    sliding_len_s: float = 20.0,
    min_samples: int = 10,
) -> list[PairFrequencyAnalysisResult]:
    disabled = set(disabled_indices or [])
    results: list[PairFrequencyAnalysisResult] = []

    spacing = spacing_dataset.spacing_matrix
    T = spacing_dataset.track2.frame_times_s
    n_pairs = int(spacing.shape[1])

    for pair_idx in range(n_pairs):
        if pair_idx in disabled:
            continue

        label = spacing_dataset.pair_labels[pair_idx] if pair_idx < len(spacing_dataset.pair_labels) else "?"
        y = spacing[:, pair_idx]

        processed, error_msg = preprocess_signal(
            T,
            y,
            longest=longest,
            handlenan=handlenan,
            min_samples=min_samples,
        )
        if processed is None:
            results.append(
                PairFrequencyAnalysisResult(
                    pair_index=pair_idx,
                    label=label,
                    processed=None,
                    fft_result=None,
                    spectrogram_result=None,
                    error_message=error_msg,
                )
            )
            continue

        if timeseriesnorm:
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                results.append(
                    PairFrequencyAnalysisResult(
                        pair_index=pair_idx,
                        label=label,
                        processed=None,
                        fft_result=None,
                        spectrogram_result=None,
                        error_message=f"time-series RMS normalization failed: {norm_error}",
                    )
                )
                continue
            processed = processed_norm

        fft_result = compute_one_sided_fft(processed.y, processed.dt)
        spec_result = compute_complex_spectrogram(processed.y, processed.Fs, sliding_len_s)
        spec_error = None if spec_result is not None else "window too short"

        results.append(
            PairFrequencyAnalysisResult(
                pair_index=pair_idx,
                label=label,
                processed=processed,
                fft_result=fft_result,
                spectrogram_result=spec_result,
                spectrogram_error_message=spec_error,
            )
        )

    return results


def analyze_spacing_dataset_with_welch_for_display(
    spacing_dataset: SpacingDataset,
    *,
    disabled_indices: list[int] | None = None,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = False,
    welch_len_s: float = 100.0,
    welch_overlap_fraction: float = 0.5,
    sliding_len_s: float = 20.0,
    min_samples: int = 10,
) -> list[PairWelchFrequencyAnalysisResult]:
    disabled = set(disabled_indices or [])
    results: list[PairWelchFrequencyAnalysisResult] = []

    spacing = spacing_dataset.spacing_matrix
    T = spacing_dataset.track2.frame_times_s
    n_pairs = int(spacing.shape[1])

    for pair_idx in range(n_pairs):
        if pair_idx in disabled:
            continue

        label = spacing_dataset.pair_labels[pair_idx] if pair_idx < len(spacing_dataset.pair_labels) else "?"
        y = spacing[:, pair_idx]

        processed, error_msg = preprocess_signal(
            T,
            y,
            longest=longest,
            handlenan=handlenan,
            min_samples=min_samples,
        )
        if processed is None:
            results.append(
                PairWelchFrequencyAnalysisResult(
                    pair_index=pair_idx,
                    label=label,
                    processed=None,
                    welch_result=None,
                    spectrogram_result=None,
                    error_message=error_msg,
                )
            )
            continue

        if timeseriesnorm:
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                results.append(
                    PairWelchFrequencyAnalysisResult(
                        pair_index=pair_idx,
                        label=label,
                        processed=None,
                        welch_result=None,
                        spectrogram_result=None,
                        error_message=f"time-series RMS normalization failed: {norm_error}",
                    )
                )
                continue
            processed = processed_norm

        welch_result = compute_welch_spectrum(
            processed.y,
            processed.Fs,
            welch_len_s,
            overlap_fraction=welch_overlap_fraction,
        )
        if welch_result is None:
            results.append(
                PairWelchFrequencyAnalysisResult(
                    pair_index=pair_idx,
                    label=label,
                    processed=processed,
                    welch_result=None,
                    spectrogram_result=None,
                    error_message="Welch window too short",
                )
            )
            continue

        spec_result = compute_complex_spectrogram(processed.y, processed.Fs, sliding_len_s)
        spec_error = None if spec_result is not None else "window too short"

        results.append(
            PairWelchFrequencyAnalysisResult(
                pair_index=pair_idx,
                label=label,
                processed=processed,
                welch_result=welch_result,
                spectrogram_result=spec_result,
                spectrogram_error_message=spec_error,
            )
        )

    return results


def median_positive_step(x: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    dx = np.diff(x)
    dx = dx[np.isfinite(dx) & (dx > 0)]
    if dx.size == 0:
        return None
    return float(np.median(dx))


def build_common_grid(contributions: list[SpectrumContribution], freq_low: float, freq_high: float) -> np.ndarray:
    dfs: list[float] = []
    for contrib in contributions:
        mask = (contrib.fft_result.freq >= freq_low) & (contrib.fft_result.freq <= freq_high)
        f_local = contrib.fft_result.freq[mask]
        df = median_positive_step(f_local)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise ValueError("Could not determine a common positive FFT spacing in the overlap window")

    df_target = max(dfs)
    grid = np.arange(freq_low, freq_high + 0.5 * df_target, df_target, dtype=float)
    grid = grid[(grid >= freq_low - 1e-12) & (grid <= freq_high + 1e-12)]

    if grid.size < 2:
        grid = np.array([freq_low, freq_high], dtype=float)

    if grid.size < 2 or not np.all(np.isfinite(grid)) or grid[-1] <= grid[0]:
        raise ValueError("Failed to build a valid common frequency grid")

    return grid


def interp_amplitude(freq_src: np.ndarray, amp_src: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    return np.interp(freq_dst, freq_src, amp_src)


def slice_spectrum_window(freq: np.ndarray, amp: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    freq = np.asarray(freq, dtype=float)
    amp = np.asarray(amp, dtype=float)

    if freq.ndim != 1 or amp.ndim != 1 or freq.size != amp.size:
        raise ValueError("freq and amp must be 1D arrays of equal length")
    if freq.size < 2:
        raise ValueError("Need at least two spectrum samples")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(amp)):
        raise ValueError("freq and amp must be finite")
    if high <= low:
        raise ValueError("Window high must be greater than low")
    if low < freq[0] or high > freq[-1]:
        raise ValueError(
            f"Requested window [{low:.6g}, {high:.6g}] Hz lies outside supported range "
            f"[{freq[0]:.6g}, {freq[-1]:.6g}] Hz"
        )

    interior_mask = (freq > low) & (freq < high)
    interior_freq = freq[interior_mask]
    interior_amp = amp[interior_mask]

    edge_freq = np.array([low, high], dtype=float)
    edge_amp = np.interp(edge_freq, freq, amp)

    if interior_freq.size == 0:
        window_freq = edge_freq
        window_amp = edge_amp
    else:
        window_freq = np.concatenate(([low], interior_freq, [high]))
        window_amp = np.concatenate(([edge_amp[0]], interior_amp, [edge_amp[1]]))

    if window_freq.size < 2:
        raise ValueError("Window extraction produced too few samples")

    return window_freq, window_amp


def integral_over_window(freq: np.ndarray, amp: np.ndarray, low: float, high: float) -> float:
    window_freq, window_amp = slice_spectrum_window(freq, amp, low, high)
    return float(np.trapz(window_amp, window_freq))


def process_spectrum_window(freq: np.ndarray, amp: np.ndarray, low: float, high: float) -> ProcessedSpectrumWindow:
    window_freq, window_amp = slice_spectrum_window(freq, amp, low, high)
    detrended = sp_signal.detrend(window_amp, type="linear")
    shifted = detrended - float(np.min(detrended))
    integral = float(np.trapz(shifted, window_freq))

    return ProcessedSpectrumWindow(
        low_hz=float(low),
        high_hz=float(high),
        freq=window_freq,
        raw_amplitude=window_amp,
        detrended_amplitude=detrended,
        shifted_amplitude=shifted,
        integral=integral,
    )


def normalize_spectrum(
    freq: np.ndarray,
    amp: np.ndarray,
    *,
    norm_low: float,
    norm_high: float,
) -> np.ndarray | None:
    processed_window = process_spectrum_window(freq, amp, norm_low, norm_high)
    if is_close_to_zero(processed_window.integral):
        return None
    return amp / processed_window.integral


def choose_frequency_window(
    contributions: list[SpectrumContribution],
    *,
    lowest_freq: float | None = None,
    highest_freq: float | None = None,
) -> tuple[float, float]:
    min_supported = max(float(contrib.fft_result.freq[0]) for contrib in contributions)
    max_supported = min(float(contrib.fft_result.freq[-1]) for contrib in contributions)

    freq_low = min_supported if lowest_freq is None else max(min_supported, lowest_freq)
    freq_high = max_supported if highest_freq is None else min(max_supported, highest_freq)

    if not np.isfinite(freq_low) or not np.isfinite(freq_high) or freq_high <= freq_low:
        raise ValueError("No overlapping frequency window across accepted spectra")

    return float(freq_low), float(freq_high)


def resolve_normalization_window(
    freq_low: float,
    freq_high: float,
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
) -> tuple[float, float]:
    rel_low, rel_high = map(float, relative_range)
    if normalize_mode == "absolute":
        norm_low = float(freq_low) + ABSOLUTE_NORMALIZATION_MARGIN_HZ
        norm_high = float(freq_high) - ABSOLUTE_NORMALIZATION_MARGIN_HZ
        if norm_high <= norm_low:
            raise ValueError(
                "Absolute normalization range [min + 0.5, max - 0.5] Hz does not fit within the selected frequency window"
            )
    elif normalize_mode == "relative":
        norm_low = max(float(freq_low), rel_low)
        norm_high = min(float(freq_high), rel_high)
        if norm_high <= norm_low:
            raise ValueError(
                "Relative normalization range does not overlap the selected frequency window"
            )
    else:
        raise ValueError(f"Unsupported normalization mode: {normalize_mode}")
    return float(norm_low), float(norm_high)


def average_spectra(normalized_stack: np.ndarray, domain: str) -> np.ndarray:
    eps = np.finfo(float).tiny

    if domain == "linear":
        return np.mean(normalized_stack, axis=0)

    if domain == "log":
        db_stack = 20.0 * np.log10(np.maximum(normalized_stack, eps))
        mean_db = np.mean(db_stack, axis=0)
        return 10.0 ** (mean_db / 20.0)

    raise ValueError(f"Unsupported averaging domain: {domain}")


def compute_mean_amplitude_spectrum(
    contributions: list[SpectrumContribution],
    *,
    lowest_freq: float | None = None,
    highest_freq: float | None = None,
) -> AveragedAmplitudeSpectrum:
    if len(contributions) == 0:
        raise ValueError("No FFT contributions were available")

    freq_low, freq_high = choose_frequency_window(
        contributions,
        lowest_freq=lowest_freq,
        highest_freq=highest_freq,
    )
    freq_grid = build_common_grid(contributions, freq_low, freq_high)
    interpolated = [interp_amplitude(c.fft_result.freq, c.fft_result.amplitude, freq_grid) for c in contributions]
    stack = np.vstack(interpolated)
    mean_amplitude = np.mean(stack, axis=0)

    return AveragedAmplitudeSpectrum(
        freq_grid=freq_grid,
        mean_amplitude=mean_amplitude,
        freq_low=float(freq_low),
        freq_high=float(freq_high),
        contributors=list(contributions),
    )


def compute_average_spectrum(
    contributions: list[SpectrumContribution],
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
    average_domain: str,
    lowest_freq: float | None = None,
    highest_freq: float | None = None,
) -> AverageSpectrumResult:
    if len(contributions) == 0:
        raise ValueError("No FFT contributions were available")

    averaged = compute_mean_amplitude_spectrum(
        contributions,
        lowest_freq=lowest_freq,
        highest_freq=highest_freq,
    )
    freq_grid = averaged.freq_grid
    freq_low = averaged.freq_low
    freq_high = averaged.freq_high

    norm_low, norm_high = resolve_normalization_window(
        freq_low,
        freq_high,
        normalize_mode=normalize_mode,
        relative_range=relative_range,
    )

    normalized_rows: list[np.ndarray] = []
    accepted_contributors: list[SpectrumContribution] = []

    for contrib in contributions:
        amp_interp = interp_amplitude(contrib.fft_result.freq, contrib.fft_result.amplitude, freq_grid)
        amp_norm = normalize_spectrum(
            freq_grid,
            amp_interp,
            norm_low=norm_low,
            norm_high=norm_high,
        )
        if amp_norm is None:
            warnings.warn(
                f"Skipping {contrib.record.signal_kind} id {contrib.record.entity_id} from "
                f"dataset '{contrib.record.dataset_name}' because normalization denominator "
                f"in [{norm_low:.6g}, {norm_high:.6g}] Hz was <= {ABSOLUTE_ZERO_TOL:.0e} after detrend/zero-floor"
            )
            continue
        normalized_rows.append(amp_norm)
        accepted_contributors.append(contrib)

    if len(normalized_rows) == 0:
        raise ValueError("All contributors were rejected during normalization")

    normalized_stack = np.vstack(normalized_rows)
    avg_amp = average_spectra(normalized_stack, average_domain)

    return AverageSpectrumResult(
        freq_grid=freq_grid,
        avg_amp=avg_amp,
        norm_low=float(norm_low),
        norm_high=float(norm_high),
        freq_low=float(freq_low),
        freq_high=float(freq_high),
        contributors=accepted_contributors,
    )


def compute_reference_average_spectrum(
    contributions: list[SpectrumContribution],
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
    average_domain: str,
) -> AverageSpectrumResult:
    return compute_average_spectrum(
        contributions,
        normalize_mode=normalize_mode,
        relative_range=relative_range,
        average_domain=average_domain,
        lowest_freq=None,
        highest_freq=None,
    )
