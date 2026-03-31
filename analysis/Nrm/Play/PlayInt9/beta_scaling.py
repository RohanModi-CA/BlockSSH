#!/usr/bin/env python3
from __future__ import annotations

import sys
import numpy as np
import scipy.signal as sp_signal
import argparse
import json
import matplotlib.pyplot as plt
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
    nyq = 0.5 * fs
    low = max(0.01, f_center - df) / nyq
    high = min(nyq - 0.01, f_center + df) / nyq
    b, a = sp_signal.butter(4, [low, high], btype='bandpass')
    y_filt = sp_signal.filtfilt(b, a, y)
    return np.abs(sp_signal.hilbert(y_filt))

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
    parser.add_argument("--json", required=True, help="Path to taxonomy JSON")
    parser.add_argument("--component", default="x")
    parser.add_argument("--bond-spacing-mode", default="comoving")
    parser.add_argument("--window-s", type=float, default=15.0)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        tax = json.load(f)
    
    children = tax.get("children", [])
    # Sort children by PLV to pick the most "solid" ones first
    children.sort(key=lambda x: x['plv'], reverse=True)
    
    print(f"Loading {args.dataset} and hits...")
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

    # Pick top 6 children for visualization
    top_children = children[:6]
    
    fig, axes = plt.subplots(len(top_children), 2, figsize=(15, 4 * len(top_children)), constrained_layout=True)
    
    for i, child in enumerate(top_children):
        f1 = child['parent1_hz']
        f2 = child['parent2_hz']
        f3 = child['child_hz']
        ctype = child['type']
        
        print(f"Analyzing Triad {i+1}: {f1:.2f} {ctype} {f2:.2f} -> {f3:.2f}")
        
        # 1. Get analytic envelopes
        env1 = extract_analytic_envelope(y_agg, fs, f1, 0.2)
        env2 = extract_analytic_envelope(y_agg, fs, f2, 0.2)
        env3 = extract_analytic_envelope(y_agg, fs, f3, 0.2)
        
        hit_strengths = [] # mean|x|
        beta_values = []
        child_amps = []
        parent_prod_amps = [] # A1 * A2
        
        for t_hit in hit_times:
            start_idx = np.searchsorted(processed_t, t_hit)
            stop_idx = start_idx + int(args.window_s * fs)
            if stop_idx >= len(processed_t): continue
            
            # Censor impact core (as before)
            y_win = y_agg[start_idx:stop_idx]
            e1 = env1[start_idx:stop_idx]
            e2 = env2[start_idx:stop_idx]
            e3 = env3[start_idx:stop_idx]
            
            local_energy = e1**2 + e2**2 + e3**2
            threshold = np.percentile(local_energy, 95)
            mask = local_energy <= threshold
            
            if np.count_nonzero(mask) < 10: continue
            
            # 2. Calculate average amplitudes in the sweet spot
            a1 = np.mean(e1[mask])
            a2 = np.mean(e2[mask])
            a3 = np.mean(e3[mask])
            
            # 3. Hit strength: local mean|x| (after removing local baseline mean)
            strength = np.mean(np.abs(y_win[mask] - np.mean(y_win[mask])))
            
            hit_strengths.append(strength)
            child_amps.append(a3)
            parent_prod_amps.append(a1 * a2)
            beta_values.append(a3 / (a1 * a2) if (a1*a2) > 0 else 0)

        # Plot A: A_child vs (A1 * A2) -> Should be linear
        ax_lin = axes[i, 0]
        ax_lin.scatter(parent_prod_amps, child_amps, color='tab:blue', s=40, edgecolors='black', alpha=0.8)
        # Linear fit through origin
        X_fit = np.array(parent_prod_amps).reshape(-1, 1)
        Y_fit = np.array(child_amps)
        if X_fit.size > 0:
            beta_avg, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)
            x_line = np.linspace(0, max(parent_prod_amps)*1.1, 100)
            ax_lin.plot(x_line, beta_avg[0] * x_line, color='tab:red', linestyle='--', alpha=0.6, label=f'beta_avg = {beta_avg[0]:.4f}')
        
        ax_lin.set_title(f"Triad {f3:.2f} <- {f1:.2f} {ctype} {f2:.2f}")
        ax_lin.set_xlabel("$A_1 \\times A_2$ (Parent Product)")
        ax_lin.set_ylabel("$A_{child}$")
        ax_lin.legend(fontsize=8)
        ax_lin.grid(True, alpha=0.3)

        # Plot B: Beta vs mean|x| -> Should be flat
        ax_beta = axes[i, 1]
        ax_beta.scatter(hit_strengths, beta_values, color='tab:purple', s=40, edgecolors='black', alpha=0.8)
        ax_beta.axhline(np.mean(beta_values), color='tab:red', linestyle='--', alpha=0.6)
        ax_beta.set_title(f"Coupling Stability: $\\beta$ vs Hit Strength")
        ax_beta.set_xlabel("Local Mean|x| (Hit Strength)")
        ax_beta.set_ylabel("$\\beta = A_3 / (A_1 A_2)$")
        ax_beta.set_ylim(0, np.mean(beta_values)*2 if beta_values else 1)
        ax_beta.grid(True, alpha=0.3)

    out_path = SCRIPT_DIR / "output" / f"{args.dataset}_beta_scaling.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nAnalysis complete! Saved scaling dashboard to {out_path}")

if __name__ == "__main__":
    main()
