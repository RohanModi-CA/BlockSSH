#!/usr/bin/env python3
from __future__ import annotations

import sys
import numpy as np
import scipy.signal as sp_signal
from pathlib import Path
import csv

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")
REPO_ROOT = add_repo_root_to_path()

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import preprocess_signal
from analysis.tools.io import load_track2_dataset


def extract_analytic_envelope(y: np.ndarray, fs: float, f_center: float, df: float) -> np.ndarray:
    """Bandpass filter a signal around f_center +/- df, then extract its complex analytic signal."""
    nyq = 0.5 * fs
    low = max(0.01, f_center - df) / nyq
    high = min(nyq - 0.01, f_center + df) / nyq
    
    b, a = sp_signal.butter(4, [low, high], btype='bandpass')
    y_filt = sp_signal.filtfilt(b, a, y)
    return sp_signal.hilbert(y_filt)


def test_event_locked_phase(f1: float, f2: float, f3: float, hit_times: list[float], y_agg: np.ndarray, fs: float, processed_t: np.ndarray):
    """
    Event-locked Phase Locking Value (PLV) using impact-core censoring and ensemble pooling.
    """
    # 1. Extract Narrowband Analytic Signals
    Z1 = extract_analytic_envelope(y_agg, fs, f1, 0.2)
    Z2 = extract_analytic_envelope(y_agg, fs, f2, 0.2)
    Z3 = extract_analytic_envelope(y_agg, fs, f3, 0.2)
    
    # Analysis parameters
    window_s = 15.0  # Look at 15s after each hit
    exclude_top_percentile = 5.0  # Discard the top 5% of energy samples (the impact core)
    
    all_phase_vectors = []
    hit_phase_vectors = []
    
    # 2. Extract Event-Locked Post-Impact Sweet Spots
    for t_hit in hit_times:
        start_idx = np.searchsorted(processed_t, t_hit)
        stop_idx = start_idx + int(window_s * fs)
        if stop_idx >= len(processed_t): continue
        
        z1_hit = Z1[start_idx:stop_idx]
        z2_hit = Z2[start_idx:stop_idx]
        z3_hit = Z3[start_idx:stop_idx]
        
        # Calculate total narrowband energy to find the "impact core"
        E_hit = np.abs(z1_hit)**2 + np.abs(z2_hit)**2 + np.abs(z3_hit)**2
        
        # Censor the core (e.g. top 5% of energy)
        threshold = np.percentile(E_hit, 100.0 - exclude_top_percentile)
        valid_mask = E_hit <= threshold
        
        if np.count_nonzero(valid_mask) < 10:
            continue
            
        # Calculate Phase Residuals (Phase-Only, Amplitude stripped)
        phase_diff = np.angle(z1_hit[valid_mask]) + np.angle(z2_hit[valid_mask]) - np.angle(z3_hit[valid_mask])
        complex_vectors = np.exp(1j * phase_diff)
        
        # Store for ensemble pooling
        hit_phase_vectors.append(complex_vectors)
        all_phase_vectors.extend(complex_vectors)
        
    if not all_phase_vectors:
        print(f"Triad {f1:.2f} + {f2:.2f} -> {f3:.2f} | Failed: No valid samples after censoring.")
        return
        
    # 3. True Pooled PLV
    real_plv = np.abs(np.mean(all_phase_vectors))
    
    # 4. Surrogate Testing (Envelope-Preserving Random Phase Shift per Event)
    rng = np.random.default_rng(42)
    n_surrogates = 200
    null_plvs = []
    
    for _ in range(n_surrogates):
        surrogate_vectors = []
        # We apply a single random phase shift to Z3 for EACH hit event
        for vectors in hit_phase_vectors:
            random_shift = rng.uniform(0, 2 * np.pi)
            surrogate_vectors.extend(vectors * np.exp(-1j * random_shift))
        null_plvs.append(np.abs(np.mean(surrogate_vectors)))
        
    null_plvs = np.array(null_plvs)
    p_val = np.mean(null_plvs >= real_plv)
    
    print(f"Triad {f1:5.2f} + {f2:5.2f} -> {f3:5.2f} | Pooled PLV = {real_plv:.3f} | p-value = {p_val:.3f}")


def load_hits(csv_path: str) -> list[float]:
    times = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
    return times


def main():
    print("Loading CDX_10IC dataset and prototype hits...")
    bond_dataset = load_bond_signal_dataset(dataset='CDX_10IC_x', bond_spacing_mode='purecomoving', component='x')
    track2 = load_track2_dataset(dataset='CDX_10IC_x')
    t = track2.frame_times_s
    
    processed_signals = []
    processed_t = None
    for i in range(bond_dataset.signal_matrix.shape[1]):
        sig = bond_dataset.signal_matrix[:, i]
        processed, _ = preprocess_signal(t, sig)
        if processed:
            if processed_t is None: processed_t = processed.t
            processed_signals.append(processed.y)
    
    y_agg = np.mean(processed_signals, axis=0)
    fs = 1.0 / np.median(np.diff(processed_t))
    
    hits_csv = REPO_ROOT / "analysis/NL/out/CDX_10IC_more_hits/CDX_10IC__x__prototype_hits.csv"
    hit_times = load_hits(str(hits_csv))
    
    print(f"Found {len(hit_times)} hit events. Running Impact-Core Censored Ensemble PLV...\n")
    
    print("--- Negative Controls (Targeting 16.61 Fundamental) ---")
    test_event_locked_phase(8.96, 16.61 - 8.96, 16.61, hit_times, y_agg, fs, processed_t)
    test_event_locked_phase(12.00, 16.61 - 12.00, 16.61, hit_times, y_agg, fs, processed_t)
    test_event_locked_phase(3.35, 16.61 - 3.35, 16.61, hit_times, y_agg, fs, processed_t)

    print("\n--- Positive Control (Suspected Real Cascade) ---")
    test_event_locked_phase(8.96, 9.42, 18.35, hit_times, y_agg, fs, processed_t)

if __name__ == "__main__":
    main()
