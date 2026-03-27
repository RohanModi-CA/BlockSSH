from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .models import (
    BondSiteAmplitudeResult,
    LocalizationProfile,
    PeakWindowIntegral,
    SignalRecord,
    SiteAmplitudeAnalysisResult,
)
from .peaks import assert_peaks_strictly_increasing
from .spectral import (
    ABSOLUTE_ZERO_TOL,
    compute_fft_contributions,
    compute_mean_amplitude_spectrum,
    integral_over_window,
    is_close_to_zero,
    process_spectrum_window,
)


def compute_roi_bounds(
    peaks: np.ndarray,
    integration_window_width: float,
    normalization_multiplier: float,
) -> tuple[float, float]:
    if integration_window_width <= 0:
        raise ValueError("integration_window_width must be > 0")
    if normalization_multiplier <= 0:
        raise ValueError("normalization_multiplier must be > 0")

    padding = float(integration_window_width) * float(normalization_multiplier)
    return float(peaks[0] - padding), float(peaks[-1] + padding)


def process_average_spectrum_roi(
    freq: np.ndarray,
    amplitude: np.ndarray,
    *,
    roi_low: float,
    roi_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    processed = process_spectrum_window(freq, amplitude, roi_low, roi_high)
    if is_close_to_zero(processed.integral, tol=ABSOLUTE_ZERO_TOL):
        raise ValueError(
            f"ROI normalization integral in [{roi_low:.6g}, {roi_high:.6g}] Hz was <= {ABSOLUTE_ZERO_TOL:.0e}"
        )
    return (
        processed.freq,
        processed.raw_amplitude,
        processed.detrended_amplitude,
        processed.shifted_amplitude,
        processed.integral,
    )


def integrate_peak_windows(
    freq: np.ndarray,
    normalized_amplitude: np.ndarray,
    peaks: np.ndarray,
    *,
    integration_window_width: float,
) -> list[PeakWindowIntegral]:
    out: list[PeakWindowIntegral] = []
    for peak_hz in peaks:
        low_hz = float(peak_hz - integration_window_width)
        high_hz = float(peak_hz + integration_window_width)
        area = integral_over_window(freq, normalized_amplitude, low_hz, high_hz)
        if area < -ABSOLUTE_ZERO_TOL:
            raise ValueError(
                f"Integrated normalized amplitude for peak {peak_hz:.6g} Hz was negative: {area:.6g}"
            )
        out.append(
            PeakWindowIntegral(
                peak_hz=float(peak_hz),
                low_hz=low_hz,
                high_hz=high_hz,
                integrated_amplitude=float(max(area, 0.0)),
            )
        )
    return out


def analyze_grouped_bond_site_amplitudes(
    grouped_records: OrderedDict[int, list[SignalRecord]] | dict[int, list[SignalRecord]],
    peaks: list[float] | np.ndarray,
    *,
    integration_window_width: float = 0.1,
    normalization_multiplier: float = 4.0,
    longest: bool = False,
    handlenan: bool = False,
    min_samples: int = 10,
) -> SiteAmplitudeAnalysisResult:
    peaks_arr = assert_peaks_strictly_increasing(peaks)
    if len(grouped_records) == 0:
        raise ValueError("No grouped bond records were provided")

    roi_low, roi_high = compute_roi_bounds(
        peaks_arr,
        integration_window_width=integration_window_width,
        normalization_multiplier=normalization_multiplier,
    )

    bond_results: list[BondSiteAmplitudeResult] = []
    for bond_id in sorted(int(b) for b in grouped_records.keys()):
        records = list(grouped_records[bond_id])
        if len(records) == 0:
            raise ValueError(f"Bond {bond_id} has no signal records")

        contributions = compute_fft_contributions(
            records,
            longest=longest,
            handlenan=handlenan,
            min_samples=min_samples,
        )
        if len(contributions) == 0:
            raise ValueError(f"Bond {bond_id} had no accepted FFT contributions")

        averaged = compute_mean_amplitude_spectrum(contributions)
        if roi_low < averaged.freq_low or roi_high > averaged.freq_high:
            raise ValueError(
                f"ROI [{roi_low:.6g}, {roi_high:.6g}] Hz crosses the averaged FFT support for bond {bond_id} "
                f"([{averaged.freq_low:.6g}, {averaged.freq_high:.6g}] Hz)"
            )

        roi_freq, roi_amplitude, detrended, shifted, normalization_integral = process_average_spectrum_roi(
            averaged.freq_grid,
            averaged.mean_amplitude,
            roi_low=roi_low,
            roi_high=roi_high,
        )
        normalized = shifted / normalization_integral
        peak_integrals = integrate_peak_windows(
            roi_freq,
            normalized,
            peaks_arr,
            integration_window_width=integration_window_width,
        )

        bond_results.append(
            BondSiteAmplitudeResult(
                bond_id=int(bond_id),
                display_bond_index=int(bond_id) + 1,
                contributors=list(contributions),
                freq_grid=averaged.freq_grid,
                mean_amplitude=averaged.mean_amplitude,
                roi_low=float(roi_low),
                roi_high=float(roi_high),
                roi_freq=roi_freq,
                roi_mean_amplitude=roi_amplitude,
                roi_detrended_amplitude=detrended,
                roi_shifted_amplitude=shifted,
                roi_normalized_amplitude=normalized,
                normalization_integral=float(normalization_integral),
                peak_integrals=peak_integrals,
            )
        )

    if len(bond_results) == 0:
        raise ValueError("No bond amplitude results were produced")

    sorted_bonds = sorted(bond_results, key=lambda bond: bond.display_bond_index)
    profiles: list[LocalizationProfile] = []
    for peak_idx, peak_hz in enumerate(peaks_arr):
        entity_ids = np.asarray([bond.display_bond_index for bond in sorted_bonds], dtype=int)
        mean_amplitudes = np.asarray(
            [bond.peak_integrals[peak_idx].integrated_amplitude for bond in sorted_bonds],
            dtype=float,
        )
        std_amplitudes = np.zeros_like(mean_amplitudes)
        profiles.append(
            LocalizationProfile(
                peak_index=int(peak_idx),
                frequency=float(peak_hz),
                entity_ids=entity_ids,
                mean_amplitudes=mean_amplitudes,
                std_amplitudes=std_amplitudes,
            )
        )

    return SiteAmplitudeAnalysisResult(
        peaks=peaks_arr,
        integration_window_width=float(integration_window_width),
        normalization_multiplier=float(normalization_multiplier),
        roi_low=float(roi_low),
        roi_high=float(roi_high),
        bonds=sorted_bonds,
        profiles=profiles,
    )


def analyze_grouped_bond_site_amplitudes_phase_reconstruction(*args, **kwargs):
    raise NotImplementedError("Phase reconstruction mode is not implemented yet")
