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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="CDX_10IC", nargs='?')
    parser.add_argument("--component", default="x")
    parser.add_argument("--bond-spacing-mode", default="purecomoving")
    parser.add_argument("--json", required=True, help="Path to taxonomy JSON file")
    parser.add_argument("--p-thresh", type=float, default=0.005)
    parser.add_argument("--tol-hz", type=float, default=0.1)
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        tax = json.load(f)

    fundamentals = tax.get("fundamentals", [])
    gen1_children = tax.get("children", [])
    
    print(f"Loaded {len(fundamentals)} fundamentals and {len(gen1_children)} 1st-Gen children from {json_path}")

    print(f"\nLoading dataset {args.dataset}...")
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
    print(f"Loaded {len(hit_times)} hit events.")

    rng = np.random.default_rng(42)
    
    gen2_children = tax.get("gen2_children", [])
    demoted_fundamentals = set()
    
    print("\n--- Starting Demotion Sweep ---")
    
    for f_target in fundamentals:
        # Try to explain f_target as F_i +/- C_j
        demoted = False
        
        for fi in fundamentals:
            if fi == f_target: continue
            
            for child in gen1_children:
                cj = child["child_hz"]
                
                # Causality Rule: Don't use a child to explain a target if the target
                # is one of the parents that gave birth to the child!
                if abs(child["parent1_hz"] - f_target) < 1e-4 or abs(child["parent2_hz"] - f_target) < 1e-4:
                    continue
                
                targets = [
                    (fi + cj, "SUM"),
                    (abs(fi - cj), "DIFF")
                ]
                
                for math_target, rel_type in targets:
                    if abs(math_target - f_target) <= args.tol_hz:
                        plv, p_val = test_triad(fi, cj, f_target, hit_times, y_agg, fs, processed_t, rng)
                        if p_val < args.p_thresh:
                            print(f"[!] DEMOTED Fundamental {f_target:.3f} Hz -> {rel_type} of F({fi:.3f}) and C1({cj:.3f}) (PLV={plv:.3f}, p={p_val:.3f})")
                            gen2_children.append({
                                "child_hz": f_target,
                                "parent_fundamental_hz": fi,
                                "parent_child_hz": cj,
                                "type": rel_type,
                                "plv": plv,
                                "p_val": p_val
                            })
                            demoted_fundamentals.add(f_target)
                            demoted = True
                            break
                if demoted: break
            if demoted: break

    # Reconstruct the Fundamentals list, removing demoted ones
    new_fundamentals = [f for f in fundamentals if f not in demoted_fundamentals]
    
    print(f"\nSweep complete. Demoted {len(demoted_fundamentals)} fundamentals to 2nd-Generation Children.")
    
    tax["fundamentals"] = new_fundamentals
    tax["gen2_children"] = gen2_children
    
    out_path = SCRIPT_DIR / f"{args.dataset}_taxonomy_demoted.json"
    with open(out_path, 'w') as f:
        json.dump(tax, f, indent=2)
        
    print(f"Saved updated taxonomy to {out_path}")

if __name__ == "__main__":
    main()
