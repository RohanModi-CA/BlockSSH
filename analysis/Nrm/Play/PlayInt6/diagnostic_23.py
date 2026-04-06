#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import numpy as np
import scipy.signal as sp_signal
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
    return sp_signal.hilbert(y_filt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="CDX_10IC")
    parser.add_argument("--f1", type=float, required=True)
    parser.add_argument("--f2", type=float, required=True)
    parser.add_argument("--f3", type=float, required=True)
    parser.add_argument("--hit-idx", type=int, default=10)
    args = parser.parse_args()

    dataset = args.dataset
    f1, f2, f3 = args.f1, args.f2, args.f3
    hit_idx_to_plot = args.hit_idx
    
    print(f"Loading {dataset} for diagnostic look at {f1} + {f2} -> {f3}...")
    bond_dataset = load_bond_signal_dataset(dataset=f"{dataset}_x", bond_spacing_mode='purecomoving', component='x')
    track2 = load_track2_dataset(dataset=f"{dataset}_x")
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
    hit_times = []
    with open(hits_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: hit_times.append(float(row['time_s']))
    
    t_hit = hit_times[hit_idx_to_plot]
    print(f"Targeting Hit {hit_idx_to_plot+1} at t={t_hit:.2f}s")
    
    # Extract analytic signals
    df = 0.3 # Slightly wider for the diagnostic
    Z1 = extract_analytic_envelope(y_agg, fs, f1, df)
    Z2 = extract_analytic_envelope(y_agg, fs, f2, df)
    Z3 = extract_analytic_envelope(y_agg, fs, f3, df)
    
    # Analysis Window: -2s to +15s around the hit
    view_start = t_hit - 2.0
    view_stop = t_hit + 15.0
    mask = (processed_t >= view_start) & (processed_t <= view_stop)
    
    t_win = processed_t[mask]
    z1w, z2w, z3w = Z1[mask], Z2[mask], Z3[mask]
    
    # Phase Residual
    dphi = np.angle(z1w) + np.angle(z2w) - np.angle(z3w)
    # Wrap to [-pi, pi]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    
    # Energy for censoring (Impact Core vs Sweet Spot)
    energy = np.abs(z1w)**2 + np.abs(z2w)**2 + np.abs(z3w)**2
    core_thresh = np.percentile(energy, 95)
    is_sweet_spot = energy <= core_thresh
    
    # Plotting
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    
    # 1. Broad look
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_win, y_agg[mask], color='0.7', label='Aggregate')
    ax0.plot(t_win, np.abs(z3w), color='tab:red', linewidth=2, label=f'|Z({f3}Hz)|')
    ax0.axvspan(t_hit-0.1, t_hit+0.1, color='orange', alpha=0.3, label='Impact Core')
    ax0.set_title(f"Hit Anatomy: {f1} + {f2} -> {f3} Hz")
    ax0.legend()
    
    # 2. Envelopes
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(t_win, np.abs(z1w), label=f'|Z({f1})|', color='tab:blue')
    ax1.plot(t_win, np.abs(z2w), label=f'|Z({f2})|', color='tab:green')
    ax1.plot(t_win, np.abs(z3w), label=f'|Z({f3})|', color='tab:red')
    ax1.set_ylabel("Analytic Envelopes")
    ax1.set_yscale('log')
    ax1.legend()
    
    # 3. Phase Residual vs Time
    ax2 = fig.add_subplot(gs[2, :])
    ax2.scatter(t_win[~is_sweet_spot], dphi[~is_sweet_spot], color='orange', s=5, label='Impact Core')
    ax2.scatter(t_win[is_sweet_spot], dphi[is_sweet_spot], color='tab:purple', s=5, label='Sweet Spot')
    ax2.set_ylabel("Phase Residual (rad)")
    ax2.set_ylim(-np.pi, np.pi)
    ax2.axvline(t_hit, color='black', linestyle='--')
    ax2.legend()
    
    # 4. Polar Plot (The "Clustering" Proof)
    ax_polar = fig.add_subplot(gs[3, 0], projection='polar')
    # Core vectors
    ax_polar.scatter(dphi[~is_sweet_spot], np.ones_like(dphi[~is_sweet_spot]), color='orange', alpha=0.2, s=2)
    # Sweet spot vectors
    ax_polar.scatter(dphi[is_sweet_spot], np.ones_like(dphi[is_sweet_spot]), color='tab:purple', alpha=0.5, s=2)
    
    # Compute and plot vectors
    vec_core = np.mean(np.exp(1j * dphi[~is_sweet_spot]))
    vec_sweet = np.mean(np.exp(1j * dphi[is_sweet_spot]))
    
    ax_polar.annotate("", xy=(np.angle(vec_core), np.abs(vec_core)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='orange', lw=3))
    ax_polar.annotate("", xy=(np.angle(vec_sweet), np.abs(vec_sweet)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='tab:purple', lw=3))
    ax_polar.set_title("Phase Lock Vectors (Length = PLV)")
    
    # 5. Summary Text
    ax_text = fig.add_subplot(gs[3, 1])
    ax_text.axis('off')
    text = (
        f"Triad: {f1} + {f2} = {f3}\n\n"
        f"IMPACT CORE (Top 5% Energy):\n"
        f"  PLV = {np.abs(vec_core):.3f}\n\n"
        f"SWEET SPOT (Censored):\n"
        f"  PLV = {np.abs(vec_sweet):.3f}\n\n"
        f"If PLV stays high in the Sweet Spot,\n"
        f"the coupling is PHYSICALLY ACTIVE.\n"
        f"If it drops to near zero, it was just\n"
        f"an amplitude spike artifact."
    )
    ax_text.text(0.1, 0.5, text, fontsize=12, family='monospace', va='center')

    out_path = SCRIPT_DIR / "output" / f"diagnostic_{f3:.1f}Hz.png"
    fig.savefig(out_path)
    print(f"Diagnostic plot saved to {out_path}")

if __name__ == "__main__":
    main()
