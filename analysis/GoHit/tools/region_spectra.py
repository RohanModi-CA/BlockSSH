from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.interpolate as _interp

from analysis.GoHit.tools.hits import HitRegion
from analysis.tools.models import FFTResult, ProcessedSignal
from analysis.tools.signal import compute_one_sided_fft


@dataclass(frozen=True)
class RegionAveragedFFT:
    fft_result: FFTResult | None
    n_regions_used: int


def _interp_spectrum(freq_src, amp_src, freq_dst, kind):
    if kind == "linear":
        return np.interp(freq_dst, freq_src, amp_src)
    if kind in ("quadratic", "cubic"):
        min_pts = 3 if kind == "quadratic" else 4
        if freq_src.size < min_pts:
            return np.interp(freq_dst, freq_src, amp_src)
        fn = _interp.interp1d(freq_src, amp_src, kind=kind, bounds_error=False, fill_value="extrapolate")
        return np.asarray(fn(freq_dst), dtype=float)
    return np.interp(freq_dst, freq_src, amp_src)


def compute_region_averaged_fft(
    processed: ProcessedSignal,
    regions: list[HitRegion],
    *,
    min_samples: int = 16,
    grid_mode: str = "finest",
    interp_kind: str = "cubic",
) -> RegionAveragedFFT:
    region_ffts: list[FFTResult] = []
    for region in regions:
        mask = (processed.t >= float(region.start_s)) & (processed.t <= float(region.stop_s))
        seg = np.asarray(processed.y[mask], dtype=float)
        if seg.size < int(min_samples):
            continue
        fft = compute_one_sided_fft(seg, processed.dt)
        if fft.freq.size < 2 or fft.amplitude.size != fft.freq.size:
            continue
        region_ffts.append(fft)

    if not region_ffts:
        return RegionAveragedFFT(fft_result=None, n_regions_used=0)

    freq_low = max(float(fft.freq[0]) for fft in region_ffts)
    freq_high = min(float(fft.freq[-1]) for fft in region_ffts)
    if not np.isfinite(freq_low) or not np.isfinite(freq_high) or freq_high <= freq_low:
        return RegionAveragedFFT(fft_result=None, n_regions_used=0)

    dfs: list[float] = []
    for fft in region_ffts:
        mask = (fft.freq >= freq_low) & (fft.freq <= freq_high)
        local_freq = fft.freq[mask]
        if local_freq.size >= 2:
            diffs = np.diff(local_freq)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size > 0:
                dfs.append(float(np.median(diffs)))
    if not dfs:
        return RegionAveragedFFT(fft_result=None, n_regions_used=0)

    df_target = max(dfs) if grid_mode == "coarsest" else min(dfs)
    freq_grid = np.arange(freq_low, freq_high + 0.5 * df_target, df_target, dtype=float)
    if freq_grid.size < 2:
        return RegionAveragedFFT(fft_result=None, n_regions_used=0)

    stack = np.vstack([_interp_spectrum(fft.freq, fft.amplitude, freq_grid, interp_kind) for fft in region_ffts])
    return RegionAveragedFFT(
        fft_result=FFTResult(freq=freq_grid, amplitude=np.mean(stack, axis=0)),
        n_regions_used=len(region_ffts),
    )
