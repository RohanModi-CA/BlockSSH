#!/usr/bin/env python3
from __future__ import annotations

import sys
import numpy as np
import scipy.signal as sp_signal
import argparse
from pathlib import Path
import csv
import json

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
    
    is_sum = abs(f1 + f2 - f3) < abs(abs(f1 - f2) - f3)
    
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
    
    peaks, props = sp_signal.find_peaks(amps, prominence=0.03, distance=distance)
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
    parser.add_argument("--n-fund", type=int, default=7)
    parser.add_argument("--p-thresh", type=float, default=0.05)
    parser.add_argument("--tol-bins", type=float, default=3.0)
    parser.add_argument("--continue-from", type=str, default=None, help="JSON file to resume from")
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
    if not hits_csv.exists():
        raise FileNotFoundError(f"Hits CSV not found: {hits_csv}")
    hit_times = load_hits(str(hits_csv))
    print(f"Loaded {len(hit_times)} hit events.")

    peak_freqs, peak_amps, df_spectrum = get_flattened_peaks(args.dataset, args.component, args.bond_spacing_mode)
    
    tol_hz = args.tol_bins * df_spectrum
    print(f"Tolerance set to {args.tol_bins} bins = {tol_hz:.4f} Hz (df={df_spectrum:.4f} Hz)")
    
    valid_mask = peak_freqs > 0.1
    peak_freqs = peak_freqs[valid_mask]
    peak_amps = peak_amps[valid_mask]
    print(f"Found {len(peak_freqs)} peaks in flattened spectrum.")
    
    unclassified_peaks = list(peak_freqs)
    
    fundamentals = []
    children = []
    tested_pairs = set()
    
    if args.continue_from:
        print(f"Resuming from {args.continue_from}...")
        with open(args.continue_from, 'r') as f:
            tax = json.load(f)
        fundamentals = tax.get("fundamentals", [])
        for c in tax.get("children", []):
            children.append((c["child_hz"], c["parent1_hz"], c["parent2_hz"], c["type"], c["plv"], c["p_val"]))
            
        for i in range(len(fundamentals)):
            fi = fundamentals[i]
            tested_pairs.add((fi, fi))
            for j in range(i):
                fj = fundamentals[j]
                tested_pairs.add(tuple(sorted([fi, fj])))
                
        classified_hz = set(fundamentals) | set(c[0] for c in children)
        filtered_unclassified = []
        for p in unclassified_peaks:
            is_classified = False
            for cls_f in classified_hz:
                if abs(p - cls_f) < 1e-4:
                    is_classified = True
                    break
            if not is_classified:
                filtered_unclassified.append(p)
        unclassified_peaks = filtered_unclassified
        print(f"Loaded {len(fundamentals)} fundamentals and {len(children)} children.")
    else:
        # 1. Seed the first fundamental
        idx_seed = np.argmin(np.abs(np.array(unclassified_peaks) - args.seed_f))
        f0 = unclassified_peaks.pop(idx_seed)
        fundamentals.append(f0)
        print(f"\n--- Initialized Fundamental: {f0:.3f} Hz ---")
        
    rng = np.random.default_rng(42)
    
    while len(fundamentals) < args.n_fund:
        new_pairs = []
        latest_f = fundamentals[-1]
        
        # self harmonic
        if (latest_f, latest_f) not in tested_pairs:
            new_pairs.append((latest_f, latest_f))
            tested_pairs.add((latest_f, latest_f))
            
        # pairs with previous fundamentals
        for f_prev in fundamentals[:-1]:
            pair = tuple(sorted([latest_f, f_prev]))
            if pair not in tested_pairs:
                new_pairs.append(pair)
                tested_pairs.add(pair)
                
        for f1, f2 in new_pairs:
            targets = [
                (f1 + f2, "SUM"),
                (abs(f1 - f2), "DIFF")
            ]
            
            for target_f, rel_type in targets:
                if target_f < 0.2: continue
                if not unclassified_peaks: break
                
                closest_idx = np.argmin(np.abs(np.array(unclassified_peaks) - target_f))
                dist = abs(unclassified_peaks[closest_idx] - target_f)
                
                if dist <= tol_hz:
                    candidate_f = unclassified_peaks[closest_idx]
                    
                    plv, p_val = test_triad(f1, f2, candidate_f, hit_times, y_agg, fs, processed_t, rng)
                    
                    if p_val < args.p_thresh:
                        print(f"  [+] {rel_type} {f1:.3f} and {f2:.3f} -> Found {candidate_f:.3f} Hz | PLV={plv:.3f}, p={p_val:.3f} (CHILD)")
                        children.append((candidate_f, f1, f2, rel_type, plv, p_val))
                        # Remove from unclassified so it doesn't get promoted to fundamental
                        unclassified_peaks.pop(closest_idx)
                    else:
                        print(f"  [-] {rel_type} {f1:.3f} and {f2:.3f} -> Tested {candidate_f:.3f} Hz | PLV={plv:.3f}, p={p_val:.3f} (REJECTED)")
        
        if len(fundamentals) < args.n_fund and unclassified_peaks:
            next_f = unclassified_peaks.pop(0)
            fundamentals.append(next_f)
            print(f"\n--- Promoted New Fundamental: {next_f:.3f} Hz ---")
            
    print("\n================ FINAL TAXONOMY ================")
    print(f"Fundamentals ({len(fundamentals)}):")
    for i, f in enumerate(fundamentals):
        print(f"  {i+1}. {f:.3f} Hz")
        
    print(f"\n1st Generation Children ({len(children)}):")
    children.sort(key=lambda x: x[0])
    for c, p1, p2, rel_type, plv, p_val in children:
        print(f"  {c:.3f} Hz <- {rel_type} of {p1:.3f} and {p2:.3f} (PLV={plv:.3f}, p={p_val:.3f})")
    print("================================================")

    out_file = SCRIPT_DIR / f"{args.dataset}_taxonomy.json"
    with open(out_file, 'w') as f:
        json.dump({
            "fundamentals": fundamentals,
            "children": [
                {"child_hz": c, "parent1_hz": p1, "parent2_hz": p2, "type": rel_type, "plv": plv, "p_val": p_val}
                for c, p1, p2, rel_type, plv, p_val in children
            ]
        }, f, indent=2)
    print(f"Saved taxonomy data to {out_file}")
    
if __name__ == "__main__":
    main()
