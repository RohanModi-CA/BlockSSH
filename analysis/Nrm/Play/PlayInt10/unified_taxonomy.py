#!/usr/bin/env python3
from __future__ import annotations

import sys
import numpy as np
import scipy.signal as sp_signal
import argparse
from pathlib import Path
import csv
import json

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
from analysis.go.Play.fft_flattening import compute_flattened_component_spectra

analytic_cache = {}
def get_Z(f_center, y_agg, fs, df=0.2):
    k = round(f_center, 3)
    if k not in analytic_cache:
        nyq = 0.5 * fs
        low = max(0.01, f_center - df) / nyq
        high = min(nyq - 0.01, f_center + df) / nyq
        if low >= high:
            analytic_cache[k] = np.zeros_like(y_agg, dtype=complex)
        else:
            b, a = sp_signal.butter(4, [low, high], btype='bandpass')
            y_filt = sp_signal.filtfilt(b, a, y_agg)
            analytic_cache[k] = sp_signal.hilbert(y_filt)
    return analytic_cache[k]


def test_triad(f1, f2, f3, hit_times, y_agg, fs, processed_t, rng):
    Z1 = get_Z(f1, y_agg, fs, 0.2)
    Z2 = get_Z(f2, y_agg, fs, 0.2)
    Z3 = get_Z(f3, y_agg, fs, 0.2)
    
    window_s = 15.0
    exclude_top_percentile = 5.0
    
    hit_phase_vectors = []
    all_phase_vectors = []
    
    # Try both SUM and DIFF hypothesis, pick the one that fits best mathematically
    # (Since we know f3 and {f1, f2}, we check which relation matches)
    if abs(f1 + f2 - f3) < abs(abs(f1 - f2) - f3):
        is_sum = True
    else:
        is_sum = False
    
    for t_hit in hit_times:
        start_idx = np.searchsorted(processed_t, t_hit)
        stop_idx = start_idx + int(window_s * fs)
        if stop_idx >= len(processed_t): continue
        
        z1_hit = Z1[start_idx:stop_idx]
        z2_hit = Z2[start_idx:stop_idx]
        z3_hit = Z3[start_idx:stop_idx]
        
        E_hit = np.abs(z1_hit)**2 + np.abs(z2_hit)**2 + np.abs(z3_hit)**2
        threshold = np.percentile(E_hit, 100.0 - exclude_top_percentile)
        valid_mask = E_hit <= threshold
        
        if np.count_nonzero(valid_mask) < 10: continue
        
        if is_sum:
            phase_diff = np.angle(z1_hit[valid_mask]) + np.angle(z2_hit[valid_mask]) - np.angle(z3_hit[valid_mask])
        else:
            # For diff, check which parent is larger to maintain p1 - p2 = child
            if f1 > f2:
                phase_diff = np.angle(z1_hit[valid_mask]) - np.angle(z2_hit[valid_mask]) - np.angle(z3_hit[valid_mask])
            else:
                phase_diff = np.angle(z2_hit[valid_mask]) - np.angle(z1_hit[valid_mask]) - np.angle(z3_hit[valid_mask])

        complex_vectors = np.exp(1j * phase_diff)
        hit_phase_vectors.append(complex_vectors)
        all_phase_vectors.extend(complex_vectors)
        
    if not all_phase_vectors:
        return 0.0, 1.0
        
    real_plv = np.abs(np.mean(all_phase_vectors))
    
    n_surrogates = 200
    null_plvs = []
    for _ in range(n_surrogates):
        surrogate_vectors = []
        for vectors in hit_phase_vectors:
            random_shift = rng.uniform(0, 2 * np.pi)
            surrogate_vectors.extend(vectors * np.exp(-1j * random_shift))
        null_plvs.append(np.abs(np.mean(surrogate_vectors)))
        
    p_val = np.mean(np.array(null_plvs) >= real_plv)
    return real_plv, p_val


def load_hits(csv_path: str) -> list[float]:
    times = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
    return times


def get_flattened_peaks(dataset, component, bond_spacing_mode):
    results = compute_flattened_component_spectra(
        dataset=dataset,
        bond_spacing_mode=bond_spacing_mode,
        components=(component,),
        use_welch=True,
    )
    freqs = np.asarray(results[component].freq_hz)
    amps = np.asarray(results[component].flattened)
    df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    distance = max(1, int(0.1 / df))
    peaks, _ = sp_signal.find_peaks(amps, prominence=0.03, distance=distance)
    peak_freqs = freqs[peaks]
    peak_amps = amps[peaks]
    sort_idx = np.argsort(peak_amps)[::-1]
    return peak_freqs[sort_idx], peak_amps[sort_idx], df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="CDX_10IC", nargs='?')
    parser.add_argument("--component", default="x")
    parser.add_argument("--bond-spacing-mode", default="comoving")
    parser.add_argument("--seed-f", type=float, default=0.41)
    parser.add_argument("--n-peaks", type=int, default=50)
    parser.add_argument("--p-thresh", type=float, default=0.05)
    parser.add_argument("--tol-bins", type=float, default=3.0)
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    bond_dataset = load_bond_signal_dataset(dataset=f"{args.dataset}_{args.component}", bond_spacing_mode=args.bond_spacing_mode, component=args.component)
    track2 = load_track2_dataset(dataset=f"{args.dataset}_{args.component}")
    t_raw = track2.frame_times_s
    
    processed_signals = []
    processed_t = None
    for i in range(bond_dataset.signal_matrix.shape[1]):
        sig = bond_dataset.signal_matrix[:, i]
        res, _ = preprocess_signal(t_raw, sig)
        if res:
            if processed_t is None: processed_t = res.t
            processed_signals.append(res.y)
    
    y_agg = np.mean(processed_signals, axis=0)
    fs = 1.0 / np.median(np.diff(processed_t))
    
    hits_csv = REPO_ROOT / "analysis/NL/out/CDX_10IC_more_hits/CDX_10IC__x__prototype_hits.csv"
    hit_times = load_hits(str(hits_csv))

    peak_freqs, peak_amps, df_spectrum = get_flattened_peaks(args.dataset, args.component, args.bond_spacing_mode)
    tol_hz = args.tol_bins * df_spectrum
    print(f"Tolerance: {args.tol_bins} bins ({tol_hz:.4f} Hz)")

    # Pool of already classified peaks (Fundamentals and Children)
    pool = [] # List of dicts: {'hz': float, 'role': str, 'parents': list}
    
    # 1. Seed the first fundamental (0.41 Hz)
    idx_seed = np.argmin(np.abs(peak_freqs - args.seed_f))
    f0_hz = peak_freqs[idx_seed]
    pool.append({'hz': f0_hz, 'role': 'fundamental', 'parents': []})
    
    # Remaining peaks to check, in descending amplitude order
    remaining_indices = [i for i in range(len(peak_freqs)) if i != idx_seed]
    # Limit to top N
    remaining_indices = remaining_indices[:args.n_peaks-1]

    rng = np.random.default_rng(42)
    
    print(f"\n--- Initialized Fundamental Seed: {f0_hz:.3f} Hz ---")
    
    for idx in remaining_indices:
        target_f = peak_freqs[idx]
        found_explanation = None
        
        # Check every pair in the current pool to see if they explain target_f
        for i in range(len(pool)):
            p1 = pool[i]
            for j in range(i, len(pool)):
                p2 = pool[j]
                
                # Check Sum, Diff, Harmonic
                math_targets = [
                    (p1['hz'] + p2['hz'], "SUM"),
                    (abs(p1['hz'] - p2['hz']), "DIFF")
                ]
                
                for math_f, rel_type in math_targets:
                    if math_f < 0.2: continue # Ignore extremely low freq noise
                    
                    if abs(math_f - target_f) <= tol_hz:
                        # Candidate found! Run the PLV test.
                        plv, p_val = test_triad(p1['hz'], p2['hz'], target_f, hit_times, y_agg, fs, processed_t, rng)
                        
                        if p_val < args.p_thresh:
                            found_explanation = {
                                'hz': target_f,
                                'role': 'child',
                                'parents': [p1['hz'], p2['hz']],
                                'type': rel_type,
                                'plv': plv,
                                'p_val': p_val
                            }
                            break
                if found_explanation: break
            if found_explanation: break
            
        if found_explanation:
            print(f"[+] CHILD: {target_f:6.3f} Hz <- {found_explanation['type']} of {found_explanation['parents'][0]:.3f} and {found_explanation['parents'][1]:.3f} (p={found_explanation['p_val']:.3f})")
            pool.append(found_explanation)
        else:
            print(f"[!] FUND : {target_f:6.3f} Hz (No math match or failed PLV)")
            pool.append({'hz': target_f, 'role': 'fundamental', 'parents': []})

    # Save results
    fundamentals = [p['hz'] for p in pool if p['role'] == 'fundamental']
    children = [p for p in pool if p['role'] == 'child']
    
    out_file = SCRIPT_DIR / "output" / f"{args.dataset}_unified_taxonomy.json"
    with open(out_file, 'w') as f:
        json.dump({
            "fundamentals": fundamentals,
            "children": children
        }, f, indent=2)
    
    print(f"\nFinal Taxonomy: {len(fundamentals)} Fundamentals, {len(children)} Children.")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
