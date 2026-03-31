#!/usr/bin/env python3
from __future__ import annotations

import sys
import numpy as np
import scipy.signal as sp_signal
import argparse
from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt

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

def get_Z_complex(y, fs, f_center, df=0.2):
    nyq = 0.5 * fs
    low = max(0.01, f_center - df) / nyq
    high = min(nyq - 0.01, f_center + df) / nyq
    if low >= high: return np.zeros_like(y, dtype=complex)
    b, a = sp_signal.butter(4, [low, high], btype='bandpass')
    y_filt = sp_signal.filtfilt(b, a, y)
    return sp_signal.hilbert(y_filt)

def load_hits(csv_path: str) -> list[float]:
    times = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
    return times

def calculate_plv_ensemble(phase_vectors_list):
    all_vecs = []
    for v in phase_vectors_list:
        all_vecs.extend(v)
    if not all_vecs: return 0.0
    return np.abs(np.mean(all_vecs))

def run_surrogate_test(hit_phase_vectors, n_surrogates=200):
    rng = np.random.default_rng(42)
    real_plv = calculate_plv_ensemble(hit_phase_vectors)
    null_plvs = []
    for _ in range(n_surrogates):
        surr_vecs = []
        for v in hit_phase_vectors:
            # Shift each hit ensemble randomly
            shift = rng.uniform(0, 2*np.pi)
            surr_vecs.extend(v * np.exp(1j * shift))
        null_plvs.append(np.abs(np.mean(surr_vecs)))
    p_val = np.mean(np.array(null_plvs) >= real_plv)
    return real_plv, p_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=float, default=8.96)
    parser.add_argument("--f2", type=float, default=9.43)
    parser.add_argument("--f3", type=float, default=18.35)
    args = parser.parse_args()

    dataset = "CDX_10IC"
    component = "x"
    
    bond_dataset = load_bond_signal_dataset(dataset=f"{dataset}_{component}", bond_spacing_mode="comoving", component=component)
    track2 = load_track2_dataset(dataset=f"{dataset}_{component}")
    t_raw = track2.frame_times_s
    
    hits_csv = REPO_ROOT / "analysis/NL/out/CDX_10IC_more_hits/CDX_10IC__x__prototype_hits.csv"
    hit_times = load_hits(str(hits_csv))
    
    n_bonds = bond_dataset.signal_matrix.shape[1]
    signals = []
    processed_t = None
    for i in range(n_bonds):
        sig = bond_dataset.signal_matrix[:, i]
        res, _ = preprocess_signal(t_raw, sig)
        if res:
            if processed_t is None: processed_t = res.t
            signals.append(res.y)
    
    fs = 1.0 / np.median(np.diff(processed_t))
    
    # Method A: Average Time Series first
    y_agg = np.mean(signals, axis=0)
    Z1_agg = get_Z_complex(y_agg, fs, args.f1)
    Z2_agg = get_Z_complex(y_agg, fs, args.f2)
    Z3_agg = get_Z_complex(y_agg, fs, args.f3)
    
    # Method B: Process per Bond
    Z1_bonds = [get_Z_complex(s, fs, args.f1) for s in signals]
    Z2_bonds = [get_Z_complex(s, fs, args.f2) for s in signals]
    Z3_bonds = [get_Z_complex(s, fs, args.f3) for s in signals]
    
    # Parameters
    win_s = 15.0
    perc = 5.0
    
    # Collect ensembles
    ensemble_agg = [] # list of arrays
    ensemble_spatial = [] # list of arrays (one per hit-bond combo)
    
    bond_amps = np.zeros((n_bonds, 3)) # f1, f2, f3 average amp
    
    for hit_t in hit_times:
        idx = np.searchsorted(processed_t, hit_t)
        stop = idx + int(win_s * fs)
        if stop >= len(processed_t): continue
        
        # Agg Method
        e1a, e2a, e3a = Z1_agg[idx:stop], Z2_agg[idx:stop], Z3_agg[idx:stop]
        energy_a = np.abs(e1a)**2 + np.abs(e2a)**2 + np.abs(e3a)**2
        mask_a = energy_a <= np.percentile(energy_a, 100-perc)
        if np.count_nonzero(mask_a) > 10:
            phi_a = np.angle(e1a[mask_a]) + np.angle(e2a[mask_a]) - np.angle(e3a[mask_a])
            ensemble_agg.append(np.exp(1j * phi_a))
            
        # Spatial Method
        for b in range(n_bonds):
            e1b, e2b, e3b = Z1_bonds[b][idx:stop], Z2_bonds[b][idx:stop], Z3_bonds[b][idx:stop]
            energy_b = np.abs(e1b)**2 + np.abs(e2b)**2 + np.abs(e3b)**2
            # Per-bond SNR check: if signal is too weak, maybe skip?
            # Let's just collect all and see.
            mask_b = energy_b <= np.percentile(energy_b, 100-perc)
            if np.count_nonzero(mask_b) > 10:
                phi_b = np.angle(e1b[mask_b]) + np.angle(e2b[mask_b]) - np.angle(e3b[mask_b])
                ensemble_spatial.append(np.exp(1j * phi_b))
                
                # Record amplitudes
                bond_amps[b, 0] += np.mean(np.abs(e1b[mask_b]))
                bond_amps[b, 1] += np.mean(np.abs(e2b[mask_b]))
                bond_amps[b, 2] += np.mean(np.abs(e3b[mask_b]))
    
    bond_amps /= len(hit_times)

    print(f"--- Triad {args.f1} + {args.f2} -> {args.f3} ---")
    
    plv_a, p_a = run_surrogate_test(ensemble_agg)
    print(f"METHOD A (Aggregated Time Series): PLV = {plv_a:.4f}, p = {p_a:.4f}")
    
    plv_b, p_b = run_surrogate_test(ensemble_spatial)
    print(f"METHOD B (Spatial Ensemble):     PLV = {plv_b:.4f}, p = {p_b:.4f}")
    
    print("\nPer-Bond Mean Amplitudes (Normalized to max bond):")
    for b in range(n_bonds):
        print(f"Bond {b}: f1={bond_amps[b,0]/np.max(bond_amps[:,0]):.2f}, f2={bond_amps[b,1]/np.max(bond_amps[:,1]):.2f}, f3={bond_amps[b,2]/np.max(bond_amps[:,2]):.2f}")

if __name__ == "__main__":
    main()
