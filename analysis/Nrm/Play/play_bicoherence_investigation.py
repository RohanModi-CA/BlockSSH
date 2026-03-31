#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks

# Add repo root to path to resolve analysis package imports
SCRIPT_DIR = Path(__file__).resolve().parent
def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")
REPO_ROOT = add_repo_root_to_path()

# Re-using necessary components
from analysis.Nrm.Tools.post_hit_regions import EnabledRegionConfig, extract_post_hit_regions
from analysis.tools.signal import hann_window_periodic, next_power_of_two, preprocess_signal
from analysis.tools.bonds import load_bond_signal_dataset # Needed for preprocess_signal indirectly
from analysis.tools.io import split_dataset_component # Needed for preprocess_signal indirectly

OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _build_hit_mask(times: np.ndarray, peak_times_s: np.ndarray, half_width_s: float) -> np.ndarray:
    mask = np.ones(times.shape, dtype=bool)
    if half_width_s <= 0 or peak_times_s.size == 0:
        return mask
    for hit_time in peak_times_s:
        mask &= np.abs(times - float(hit_time)) > half_width_s
    return mask

def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    bounded = np.concatenate(([False], mask, [False]))
    changes = np.diff(bounded.astype(int))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]

def collect_welch_segments(cfg: Config) -> tuple[np.ndarray, np.ndarray, float]:
    all_complex_segments: list[np.ndarray] = []
    
    # We will use the processed signal for each bond, then segment it
    for bond_id in cfg.bond_ids:
        # extract_post_hit_regions is used here to get the full processed signal for the bond
        region_config = EnabledRegionConfig(bond_spacing_mode="default") # Using default spacing mode
        result = extract_post_hit_regions(
            dataset=cfg.dataset,
            component=cfg.component,
            bond_id=bond_id,
            config=region_config,
        )
        processed, err = preprocess_signal(result.frame_times_s, result.signal, longest=False, handlenan=False)
        if processed is None:
            print(f"Skipping bond {bond_id}: Failed to preprocess full trace: {err}", file=sys.stderr)
            continue

        local_fs = processed.Fs
        local_nperseg = max(8, int(round(cfg.welch_len_s * local_fs)))
        local_nperseg = min(local_nperseg, processed.y.size)
        if local_nperseg < 8:
            print(f"Skipping bond {bond_id}: Segment length is too short for data size.", file=sys.stderr)
            continue

        local_noverlap = min(int(round(cfg.welch_overlap_fraction * local_nperseg)), local_nperseg - 1)
        local_step = local_nperseg - local_noverlap
        if local_step <= 0:
            print(f"Skipping bond {bond_id}: Segment overlap leaves no forward step.", file=sys.stderr)
            continue

        nfft = max(local_nperseg, next_power_of_two(local_nperseg))
        window = hann_window_periodic(local_nperseg)
        window_norm = float(np.sum(window))

        if cfg.no_mask_hits:
            usable_mask = np.ones(processed.t.shape, dtype=bool)
        else:
            usable_mask = _build_hit_mask(
                processed.t,
                result.peak_times_s,
                half_width_s=float(cfg.mask_hit_window_s),
            )
        
        # Iterate over usable runs of the signal
        for run_start, run_stop in _true_runs(usable_mask):
            if run_stop - run_start < local_nperseg:
                continue
            for start_idx in range(run_start, run_stop - local_nperseg + 1, local_step):
                end_idx = start_idx + local_nperseg
                segment = processed.y[start_idx:end_idx] * window
                spectrum = np.fft.rfft(segment, n=nfft) / window_norm
                if spectrum.size > 2:
                    spectrum = spectrum.copy()
                    spectrum[1:-1] *= 2.0 # Adjust for one-sided spectrum power scaling
                all_complex_segments.append(spectrum)
    
    if not all_complex_segments:
        raise ValueError("No valid segments were collected. Check input data, segment length, and masking.")

    # Assuming all segments have the same frequency bins (due to fixed nfft and dt)
    # Use the dt from the last successfully processed signal
    freqs = np.fft.rfftfreq(nfft, d=processed.dt)

    return freqs, np.vstack(all_complex_segments), processed.Fs

def extract_local_component_from_segments(
    freqs: np.ndarray,
    all_complex_segments: np.ndarray,
    center_hz: float,
    half_width_hz: float
) -> np.ndarray:
    mask = np.abs(freqs - float(center_hz)) <= float(half_width_hz)
    if not np.any(mask):
        raise ValueError(f"No frequency bins found near {center_hz:.4f} Hz")
    
    band_spec = all_complex_segments[:, mask] # Select frequency bins across all segments
    
    # For each segment, find the complex value at the peak amplitude within the band
    amp = np.abs(band_spec)
    idx = np.argmax(amp, axis=1) # Index of max amplitude within band for each segment

    chosen_complex = band_spec[np.arange(band_spec.shape[0]), idx]
    return chosen_complex

def pooled_bicoherence_from_complex_segments(z1: np.ndarray, z2: np.ndarray, z3: np.ndarray, null_trials: int = 0, rng: np.random.Generator | None = None):
    # Compute Bispectrum
    bispec_terms = z1 * z2 * np.conj(z3)
    bispec = np.mean(bispec_terms)
    
    # Compute Normalization factor
    denom = np.mean(np.abs(z1 * z2)**2) * np.mean(np.abs(z3)**2)
    
    b_sq_observed = float(np.abs(bispec)**2 / denom) if denom > 1e-18 else np.nan
    phase_plv_observed = float(np.abs(np.mean(np.exp(1j * np.angle(bispec_terms))))) if bispec_terms.size > 0 else np.nan

    null_values: list[float] = []
    if null_trials > 0 and rng is not None and z1.size > 0:
        for _ in range(null_trials):
            shuffled_z3 = np.copy(z3)
            if shuffled_z3.size > 1:
                shift = int(rng.integers(1, shuffled_z3.size))
                shuffled_z3 = np.roll(shuffled_z3, shift)
            
            null_bispec_terms = z1 * z2 * np.conj(shuffled_z3)
            null_bispec = np.mean(null_bispec_terms)
            null_denom = np.mean(np.abs(z1 * z2)**2) * np.mean(np.abs(shuffled_z3)**2)
            null_values.append(float(np.abs(null_bispec)**2 / null_denom) if null_denom > 1e-18 else np.nan)
    
    null_arr = np.asarray(null_values, dtype=float)
    b_sq_null_mean = float(np.nanmean(null_arr)) if null_arr.size else np.nan
    b_sq_null_q95 = float(np.nanquantile(null_arr, 0.95)) if null_arr.size else np.nan

    return {
        "b_sq_observed": b_sq_observed,
        "phase_plv_observed": phase_plv_observed,
        "b_sq_null_mean": b_sq_null_mean,
        "b_sq_null_q95": b_sq_null_q95,
    }


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    component: str = "x"
    bond_ids: tuple[int, ...] = (0, 1, 2)
    welch_len_s: float = 100.0 # User's requested Welch window length
    welch_overlap_fraction: float = 0.5
    search_half_width_hz: float = 0.25
    amplitude_gate_fraction: float = 0.20
    min_windows_per_region: int = 3
    null_trials: int = 400
    null_seed: int = 0
    mask_hit_window_s: float = 6.0
    no_mask_hits: bool = True


@dataclass
class BicoherenceResult:
    f1: float
    f2: float
    f_sum_expected: float
    f_sum_actual: float
    b_sq_observed: float
    b_sq_null_mean: float
    b_sq_null_q95: float
    label: str
    n_total_segments: int
    max_segment_bic: float = np.nan
    max_segment_plv: float = np.nan












def get_bicoherence_for_triplet(f1: float, f2: float, f3_expected: float, all_freqs: np.ndarray, all_complex_segments: np.ndarray, cfg: Config) -> BicoherenceResult:
    rng = np.random.default_rng(cfg.null_seed)

    # Extract complex components for f1, f2, and f3_expected
    z1 = extract_local_component_from_segments(all_freqs, all_complex_segments, f1, cfg.search_half_width_hz)
    z2 = extract_local_component_from_segments(all_freqs, all_complex_segments, f2, cfg.search_half_width_hz)
    z3 = extract_local_component_from_segments(all_freqs, all_complex_segments, f3_expected, cfg.search_half_width_hz)

    # Perform pooled bicoherence analysis with null testing
    pooled_bic_result = pooled_bicoherence_from_complex_segments(
        z1, z2, z3, null_trials=cfg.null_trials, rng=rng
    )

    # For f_sum_actual, find the closest frequency bin to f3_expected
    f_sum_actual = all_freqs[np.argmin(np.abs(all_freqs - f3_expected))]

    return BicoherenceResult(
        f1=f1,
        f2=f2,
        f_sum_expected=f3_expected,
        f_sum_actual=f_sum_actual,
        b_sq_observed=pooled_bic_result["b_sq_observed"],
        b_sq_null_mean=pooled_bic_result["b_sq_null_mean"],
        b_sq_null_q95=pooled_bic_result["b_sq_null_q95"],
        label=f"({f1:.2f} + {f2:.2f} -> {f3_expected:.2f})",
        n_total_segments=all_complex_segments.shape[0],
        max_segment_bic=np.nan, # Not directly applicable in this pooled context
        max_segment_plv=np.nan, # Not directly applicable in this pooled context
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Investigate specific frequency combinations using Welch-style bicoherence analysis."
    )
    parser.add_argument("--dataset", default="IMG_0681_rot270", help="Dataset base name.")
    parser.add_argument("--component", choices=("x", "y", "a"), default="x", help="Component to analyze.")
    parser.add_argument("--bond-ids", type=int, nargs="+", default=[0, 1, 2], help="Space-separated list of bond IDs.")
    parser.add_argument("--welch-len-s", type=float, default=10.0, help="Welch window length in seconds.")
    parser.add_argument("--welch-overlap-fraction", type=float, default=0.5, help="Overlap fraction for Welch windows.")
    parser.add_argument("--search-half-width-hz", type=float, default=0.25, help="Half-width for frequency band search.")
    parser.add_argument("--amplitude-gate-fraction", type=float, default=0.20, help="Fraction of median amplitude for gating.")
    parser.add_argument("--min-windows-per-region", type=int, default=3, help="Minimum Welch windows per region to use.")
    parser.add_argument("--null-trials", type=int, default=400, help="Number of null trials for significance testing.")
    parser.add_argument("--null-seed", type=int, default=0, help="Random seed for null trials.")
    parser.add_argument("--mask-hit-window-s", type=float, default=6.0, help="Half-width in seconds masked around each detected/manual hit time.")
    parser.add_argument("--no-mask-hits", action="store_true", default=True, help="Disable hit masking and use all segments from the full traces.")
    parser.add_argument("--show", action="store_true", help="Show the final figure with matplotlib.")

    args = parser.parse_args()

    current_config = Config(
        dataset=args.dataset,
        component=args.component,
        bond_ids=tuple(args.bond_ids),
        welch_len_s=args.welch_len_s,
        welch_overlap_fraction=args.welch_overlap_fraction,
        search_half_width_hz=args.search_half_width_hz,
        amplitude_gate_fraction=args.amplitude_gate_fraction,
        min_windows_per_region=args.min_windows_per_region,
        null_trials=args.null_trials,
        null_seed=args.null_seed,
        mask_hit_window_s=args.mask_hit_window_s,
        no_mask_hits=args.no_mask_hits
    )

    print(f"--- Welch-style Bicoherence Investigation: {current_config.dataset} Component {current_config.component} ---")
    print(f"Welch segment length: {current_config.welch_len_s}s, Overlap: {current_config.welch_overlap_fraction}, Null Trials: {current_config.null_trials}")
    print(f"Hit Masking: {'Disabled' if current_config.no_mask_hits else f'+/- {current_config.mask_hit_window_s}s around hits'}")
    
    all_freqs, all_complex_segments, processed_Fs = collect_welch_segments(current_config)
    n_segments = all_complex_segments.shape[0]

    if n_segments == 0:
        print("No segments collected for analysis. Exiting.")
        return 1

    print(f"Total {n_segments} segments collected. Approx Fs: {processed_Fs:.2f}Hz")


    # Define Triplets to Test (same as play_bicoherence_investigation.py for comparison)
    triplets_to_test = [
        (0.42, 0.42, 0.84, "Harmonic (2 * 0.42)"),
        (6.34, 12.00, 18.34, "Sum (6.34 + 12.00)"),
        (3.97, 3.97, 7.94, "Harmonic (2 * 3.97)"),
        (3.35, 7.96, 11.31, "Difference Implied (3.35 + 7.96 -> 11.31)"),
        (8.96, 8.96, 18.0, "Project Key Triad (8.96 + 8.96 -> 18.0)"),
        (2.0, 16.34, 18.34, "Negative Control (2.0 + 16.34 -> 18.34)"),
        (1.0, 6.94, 7.94, "Negative Control (1.0 + 6.94 -> 7.94)"),
    ]

    results: list[BicoherenceResult] = []
    print(f"\n{'Relationship':<50} | {'f1':<8} | {'f2':<8} | {'f_sum_exp':<11} | {'f_sum_act':<11} | {'Observed B^2':<14} | {'Null Mean B^2':<14} | {'Null 95% B^2':<14} | {'Status':<10} | {'Segments':<8}")
    print("-" * 190)
    for f1, f2, f3_expected, label_prefix in triplets_to_test:
        try:
            result = get_bicoherence_for_triplet(f1, f2, f3_expected, all_freqs, all_complex_segments, current_config)
            result.label = label_prefix
            results.append(result)
            status = "Coupled" if result.b_sq_observed > result.b_sq_null_q95 else "Independent"
            print(f"{result.label:<50} | {result.f1:<8.3f} | {result.f2:<8.3f} | {result.f_sum_expected:<11.3f} | {result.f_sum_actual:<11.3f} | {result.b_sq_observed:<14.6f} | {result.b_sq_null_mean:<14.6f} | {result.b_sq_null_q95:<14.6f} | {status:<10} | {result.n_total_segments:<8}")
        except ValueError as e:
            print(f"{label_prefix:<50} | {'Error':<8} | {'-':<8} | {'-':<11} | {'-':<11} | {'-':<14} | {'-':<14} | {'-':<14} | {'-':<10} | {'-':<8} | Error: {e}")

    # 4. Simple Visualization (Bar chart)
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [r.label for r in results]
    observed_vals = [r.b_sq_observed for r in results]
    null_q95_vals = [r.b_sq_null_q95 for r in results]
    
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, observed_vals, width, label='Observed B^2', color='C0')
    ax.bar(x + width/2, null_q95_vals, width, label='Null 95% B^2', color='C1', alpha=0.7)
    
    ax.set_ylabel("Squared Bicoherence")
    ax.set_title(f"Welch-style Bicoherence (Full Trace, Observed vs. Null 95%) - {current_config.dataset} Component {current_config.component}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "welch_bicoherence_investigation_results.png")
    print(f"\nResults plot saved to {OUTPUT_DIR / 'welch_bicoherence_investigation_results.png'}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
emExit(main())
