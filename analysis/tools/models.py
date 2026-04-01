from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Track2Dataset:
    dataset_name: str | None
    track2_path: str
    original_video_path: str
    tracking_results_path: str
    block_colors: list[str]
    x_positions: np.ndarray
    frame_times_s: np.ndarray
    frame_numbers: np.ndarray
    purecomoving_signal: np.ndarray | None = None


@dataclass(frozen=True)
class SpacingDataset:
    track2: Track2Dataset
    pair_labels: list[str]
    spacing_matrix: np.ndarray


@dataclass(frozen=True)
class DatasetSelection:
    include: bool
    discards: list[int]
    pair_ids: list[int]


@dataclass(frozen=True)
class SignalRecord:
    dataset_name: str
    entity_id: int
    local_index: int
    label: str
    signal_kind: Literal["bond", "site"]
    source_path: str
    t: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ProcessedSignal:
    t: np.ndarray
    y: np.ndarray
    dt: float
    Fs: float
    nyquist: float
    proc_msg: str


@dataclass(frozen=True)
class FFTResult:
    freq: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class ComplexFFTResult:
    freq: np.ndarray
    spectrum: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class SpectrogramResult:
    f: np.ndarray
    t: np.ndarray
    S_complex: np.ndarray
    win_samp: int
    noverlap: int
    nfft: int


@dataclass(frozen=True)
class WelchSpectrumResult:
    freq: np.ndarray
    power: np.ndarray
    amplitude: np.ndarray
    nperseg: int
    noverlap: int
    nfft: int


@dataclass(frozen=True)
class TargetFrequencyResult:
    target_freq: float
    selected_freq: float
    index: int
    complex_value: complex
    amplitude: float
    found: bool


@dataclass(frozen=True)
class SpectrumContribution:
    record: SignalRecord
    processed: ProcessedSignal
    fft_result: FFTResult


@dataclass(frozen=True)
class AverageSpectrumResult:
    freq_grid: np.ndarray
    avg_amp: np.ndarray
    norm_low: float
    norm_high: float
    freq_low: float
    freq_high: float
    contributors: list[SpectrumContribution]


@dataclass(frozen=True)
class FlatteningResult:
    baseline: np.ndarray
    baseline_smooth: np.ndarray
    transfer: np.ndarray
    flattened: np.ndarray
    reference_level: float
    reference_band: tuple[float, float]
    method: str
    baseline_quantile: float
    baseline_envelope_hz: float
    baseline_smooth_hz: float
    response_smooth_hz: float


@dataclass(frozen=True)
class AveragedAmplitudeSpectrum:
    freq_grid: np.ndarray
    mean_amplitude: np.ndarray
    freq_low: float
    freq_high: float
    contributors: list[SpectrumContribution]


@dataclass(frozen=True)
class ProcessedSpectrumWindow:
    low_hz: float
    high_hz: float
    freq: np.ndarray
    raw_amplitude: np.ndarray
    detrended_amplitude: np.ndarray
    shifted_amplitude: np.ndarray
    integral: float


@dataclass(frozen=True)
class PairFrequencyAnalysisResult:
    pair_index: int
    label: str
    processed: ProcessedSignal | None
    fft_result: FFTResult | None
    spectrogram_result: SpectrogramResult | None
    error_message: str | None = None
    spectrogram_error_message: str | None = None


@dataclass(frozen=True)
class PairWelchFrequencyAnalysisResult:
    pair_index: int
    label: str
    processed: ProcessedSignal | None
    welch_result: WelchSpectrumResult | None
    spectrogram_result: SpectrogramResult | None
    error_message: str | None = None
    spectrogram_error_message: str | None = None


@dataclass(frozen=True)
class LocalizationProfile:
    peak_index: int
    frequency: float
    entity_ids: np.ndarray
    mean_amplitudes: np.ndarray
    std_amplitudes: np.ndarray


@dataclass(frozen=True)
class LocalizationPeakDiagnostic:
    peak_index: int
    target_frequency: float
    selected_frequency: float
    found: bool
    window_low: float
    window_high: float
    display_freq: np.ndarray
    display_amplitude: np.ndarray


@dataclass(frozen=True)
class PeakWindowIntegral:
    peak_hz: float
    low_hz: float
    high_hz: float
    integrated_amplitude: float


@dataclass(frozen=True)
class BondSiteAmplitudeResult:
    bond_id: int
    display_bond_index: int
    contributors: list[SpectrumContribution]
    freq_grid: np.ndarray
    mean_amplitude: np.ndarray
    roi_low: float
    roi_high: float
    roi_freq: np.ndarray
    roi_mean_amplitude: np.ndarray
    roi_detrended_amplitude: np.ndarray
    roi_shifted_amplitude: np.ndarray
    roi_normalized_amplitude: np.ndarray
    normalization_integral: float
    peak_integrals: list[PeakWindowIntegral]


@dataclass(frozen=True)
class SiteAmplitudeAnalysisResult:
    peaks: np.ndarray
    integration_window_width: float
    normalization_multiplier: float
    roi_low: float
    roi_high: float
    bonds: list[BondSiteAmplitudeResult]
    profiles: list[LocalizationProfile]
