from __future__ import annotations

import numpy as np

from analysis.tools.spectral import process_spectrum_window, slice_spectrum_window


BASELINE_ROI_MULTIPLIER = 3.0


def local_baseline_subtracted_peak_amplitude(
    freqs: np.ndarray,
    amps: np.ndarray,
    target: float,
    width: float,
    *,
    sqrtintpower: bool,
    roi_multiplier: float = BASELINE_ROI_MULTIPLIER,
) -> tuple[float, bool]:
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)

    if freqs.size < 2 or amps.size != freqs.size:
        return 0.0, False

    peak_low = max(float(freqs[0]), float(target) - float(width))
    peak_high = min(float(freqs[-1]), float(target) + float(width))
    if peak_high <= peak_low:
        return 0.0, False

    roi_low = max(float(freqs[0]), float(target) - float(roi_multiplier) * float(width))
    roi_high = min(float(freqs[-1]), float(target) + float(roi_multiplier) * float(width))
    if roi_high <= roi_low:
        return 0.0, False

    processed = process_spectrum_window(freqs, amps, roi_low, roi_high)
    local_freq, local_shifted = slice_spectrum_window(
        processed.freq,
        processed.shifted_amplitude,
        peak_low,
        peak_high,
    )

    finite = np.isfinite(local_shifted)
    if not np.any(finite):
        return 0.0, False

    shifted = np.maximum(local_shifted, 0.0)
    if sqrtintpower:
        power = np.square(shifted)
        integral = float(np.trapz(power, local_freq))
        if not np.isfinite(integral) or integral <= 0.0:
            return 0.0, False
        return float(np.sqrt(integral)), True

    return float(np.max(shifted[finite])), True
