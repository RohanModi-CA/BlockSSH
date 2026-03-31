from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from .models import (
    AverageSpectrumResult,
    FFTResult,
    FlatteningResult,
    PairFrequencyAnalysisResult,
    PairWelchFrequencyAnalysisResult,
    WelchSpectrumResult,
)


DEFAULT_FLATTEN_METHOD = "quantile-envelope"


def rolling_quantile_baseline(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    *,
    quantile: float = 0.15,
    envelope_hz: float = 1.5,
    smooth_hz: float = 1.2,
) -> np.ndarray:
    step = float(np.median(np.diff(freq_hz)))
    half_window = max(1, int(round(envelope_hz / step / 2.0)))
    envelope = np.empty_like(amplitude)
    for idx in range(amplitude.size):
        lo = max(0, idx - half_window)
        hi = min(amplitude.size, idx + half_window + 1)
        envelope[idx] = np.quantile(amplitude[lo:hi], quantile)

    smooth_points = max(5, int(round(smooth_hz / step)))
    if smooth_points % 2 == 0:
        smooth_points += 1
    baseline = savgol_filter(envelope, smooth_points, polyorder=2, mode="interp")
    return np.minimum(baseline, amplitude)


def smooth_log_envelope(
    freq_hz: np.ndarray,
    values: np.ndarray,
    *,
    smooth_hz: float = 4.0,
) -> np.ndarray:
    step = float(np.median(np.diff(freq_hz)))
    smooth_points = max(7, int(round(smooth_hz / step)))
    if smooth_points % 2 == 0:
        smooth_points += 1
    log_values = np.log(np.maximum(values, np.finfo(float).tiny))
    smoothed = savgol_filter(log_values, smooth_points, polyorder=2, mode="interp")
    return np.exp(smoothed)


def compute_flattening(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    *,
    reference_band: tuple[float, float] = (20.0, 30.0),
    baseline_quantile: float = 0.15,
    baseline_envelope_hz: float = 1.5,
    baseline_smooth_hz: float = 1.2,
    response_smooth_hz: float = 4.0,
    method: str = DEFAULT_FLATTEN_METHOD,
) -> FlatteningResult:
    if method != DEFAULT_FLATTEN_METHOD:
        raise ValueError(f"Unsupported flattening method: {method}")

    baseline = rolling_quantile_baseline(
        freq_hz,
        amplitude,
        quantile=baseline_quantile,
        envelope_hz=baseline_envelope_hz,
        smooth_hz=baseline_smooth_hz,
    )
    baseline_smooth = smooth_log_envelope(
        freq_hz,
        baseline,
        smooth_hz=response_smooth_hz,
    )
    mask = (freq_hz >= reference_band[0]) & (freq_hz <= reference_band[1])
    if not np.any(mask):
        raise ValueError("Flattening reference band does not overlap frequency grid")
    reference_level = float(
        np.exp(np.mean(np.log(np.maximum(baseline_smooth[mask], np.finfo(float).tiny))))
    )
    transfer = reference_level / np.maximum(baseline_smooth, np.finfo(float).tiny)
    flattened = np.asarray(amplitude, dtype=float) * transfer
    return FlatteningResult(
        baseline=np.asarray(baseline, dtype=float),
        baseline_smooth=np.asarray(baseline_smooth, dtype=float),
        transfer=np.asarray(transfer, dtype=float),
        flattened=np.asarray(flattened, dtype=float),
        reference_level=reference_level,
        reference_band=(float(reference_band[0]), float(reference_band[1])),
        method=method,
        baseline_quantile=float(baseline_quantile),
        baseline_envelope_hz=float(baseline_envelope_hz),
        baseline_smooth_hz=float(baseline_smooth_hz),
        response_smooth_hz=float(response_smooth_hz),
    )


def apply_flattening_to_average_result(
    result: AverageSpectrumResult,
    *,
    reference_band: tuple[float, float] = (20.0, 30.0),
    baseline_quantile: float = 0.15,
    baseline_envelope_hz: float = 1.5,
    baseline_smooth_hz: float = 1.2,
    response_smooth_hz: float = 4.0,
    method: str = DEFAULT_FLATTEN_METHOD,
) -> tuple[AverageSpectrumResult, FlatteningResult]:
    flattening = compute_flattening(
        result.freq_grid,
        result.avg_amp,
        reference_band=reference_band,
        baseline_quantile=baseline_quantile,
        baseline_envelope_hz=baseline_envelope_hz,
        baseline_smooth_hz=baseline_smooth_hz,
        response_smooth_hz=response_smooth_hz,
        method=method,
    )
    return replace(result, avg_amp=flattening.flattened), flattening


def apply_flattening_to_pair_result(
    result: PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult,
    *,
    reference_band: tuple[float, float] = (20.0, 30.0),
    baseline_quantile: float = 0.15,
    baseline_envelope_hz: float = 1.5,
    baseline_smooth_hz: float = 1.2,
    response_smooth_hz: float = 4.0,
    method: str = DEFAULT_FLATTEN_METHOD,
) -> PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult:
    if result.error_message is not None:
        return result

    if isinstance(result, PairFrequencyAnalysisResult):
        if result.fft_result is None:
            return result
        freq = result.fft_result.freq
        amp = result.fft_result.amplitude
    else:
        if result.welch_result is None:
            return result
        freq = result.welch_result.freq
        amp = result.welch_result.amplitude

    flattening = compute_flattening(
        freq,
        amp,
        reference_band=reference_band,
        baseline_quantile=baseline_quantile,
        baseline_envelope_hz=baseline_envelope_hz,
        baseline_smooth_hz=baseline_smooth_hz,
        response_smooth_hz=response_smooth_hz,
        method=method,
    )

    if isinstance(result, PairFrequencyAnalysisResult):
        new_spectral = FFTResult(freq=freq, amplitude=flattening.flattened)
        result = replace(result, fft_result=new_spectral)
    else:
        new_spectral = replace(
            result.welch_result,
            amplitude=flattening.flattened,
            power=flattening.flattened**2,
        )
        result = replace(result, welch_result=new_spectral)

    if result.spectrogram_result is not None:
        f_spec = result.spectrogram_result.f
        # Interpolate the transfer function to the spectrogram frequency grid
        transfer_spec = np.interp(f_spec, freq, flattening.transfer)
        new_s_complex = result.spectrogram_result.S_complex * transfer_spec[:, None]
        new_spec_result = replace(result.spectrogram_result, S_complex=new_s_complex)
        result = replace(result, spectrogram_result=new_spec_result)

    return result


def flattening_metadata(flattening: FlatteningResult) -> dict[str, object]:
    return {
        "method": flattening.method,
        "referenceBandHz": [float(flattening.reference_band[0]), float(flattening.reference_band[1])],
        "referenceLevel": float(flattening.reference_level),
        "baselineQuantile": float(flattening.baseline_quantile),
        "baselineEnvelopeHz": float(flattening.baseline_envelope_hz),
        "baselineSmoothHz": float(flattening.baseline_smooth_hz),
        "responseSmoothHz": float(flattening.response_smooth_hz),
        "baseline": flattening.baseline,
        "baselineSmooth": flattening.baseline_smooth,
        "transfer": flattening.transfer,
    }


def plot_flattening_diagnostic(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    flattening: FlatteningResult,
    *,
    title: str | None = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    ax0, ax1 = axes

    ax0.semilogy(freq_hz, amplitude, linewidth=1.2, label="raw average")
    ax0.semilogy(freq_hz, flattening.baseline, linewidth=1.8, color="black", label="baseline")
    ax0.semilogy(
        freq_hz,
        flattening.baseline_smooth,
        linewidth=1.4,
        color="tab:orange",
        label="smoothed response envelope",
    )
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")
    ax0.set_title(title or "Flattening Diagnostic")

    ax1.semilogy(freq_hz, flattening.transfer, linewidth=1.3, color="tab:purple", label="transfer")
    ax1.semilogy(freq_hz, flattening.flattened, linewidth=1.2, color="tab:blue", label="flattened")
    ax1.axhline(flattening.reference_level, color="0.6", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Gain / Flattened amp.")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    return fig
