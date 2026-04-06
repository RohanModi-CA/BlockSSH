#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp_signal
import scipy.stats as sp_stats

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
    
    # Extract analytic signal via Hilbert transform
    z = sp_signal.hilbert(y_filt)
    return z


def fit_qar(Z1: np.ndarray, Z2: np.ndarray, Z3: np.ndarray, p: int) -> tuple[float, float]:
    """
    Fits Linear AR(p) and Quadratic AR(p) models predicting Z3 from its past + Z1*Z2.
    Returns: (F-statistic, p-value)
    """
    N_full = len(Z3)
    N = N_full - p
    
    # Target
    Y = Z3[p:]
    
    # Linear Regressors X_L (AR terms only)
    X_L = np.zeros((N, p), dtype=complex)
    for k in range(p):
        X_L[:, k] = Z3[p - k - 1 : N_full - k - 1]
        
    # Quadratic Regressors X_Q (AR terms + Z1*Z2 interaction)
    Z_int = Z1[p:] * Z2[p:]
    X_Q = np.hstack([X_L, Z_int.reshape(-1, 1)])
    
    # Fit models (Complex Least Squares)
    w_L, resid_L, _, _ = np.linalg.lstsq(X_L, Y, rcond=None)
    w_Q, resid_Q, _, _ = np.linalg.lstsq(X_Q, Y, rcond=None)
    
    # Calculate Residual Sum of Squares
    RSS_L = np.sum(np.abs(Y - X_L @ w_L)**2) if resid_L.size == 0 else resid_L[0].real
    RSS_Q = np.sum(np.abs(Y - X_Q @ w_Q)**2) if resid_Q.size == 0 else resid_Q[0].real
    
    if RSS_L <= 0 or RSS_Q <= 0 or RSS_Q >= RSS_L:
        return 0.0, 1.0
        
    # F-test for nested models (Complex data means 2 real DOF per parameter)
    # df1 = 2 (for the single complex interaction parameter)
    # df2 = 2*N - 2*(p + 1)
    df1 = 2
    df2 = 2 * N - 2 * (p + 1)
    
    if df2 <= 0:
        return 0.0, 1.0
        
    F = ((RSS_L - RSS_Q) / df1) / (RSS_Q / df2)
    p_val = sp_stats.f.sf(F, df1, df2)
    
    return F, p_val


def main():
    parser = argparse.ArgumentParser(description="Quadratic Autoregressive (QAR) Phase Lock Analysis")
    parser.add_argument("dataset")
    parser.add_argument("--component", default="x")
    parser.add_argument("--f1", type=float, default=8.96, help="Target frequency 1 (Hz)")
    parser.add_argument("--f2", type=float, default=9.42, help="Target frequency 2 (Hz)")
    parser.add_argument("--f3", type=float, default=18.35, help="Child frequency (Hz)")
    parser.add_argument("--df", type=float, default=0.2, help="Filter half-bandwidth (Hz)")
    parser.add_argument("--window-s", type=float, default=10.0, help="Sliding window size (s)")
    parser.add_argument("--step-s", type=float, default=2.5, help="Sliding window step (s)")
    parser.add_argument("--ar-order", type=int, default=2, help="Linear AR order (p)")
    args = parser.parse_args()

    bond_dataset = load_bond_signal_dataset(
        dataset=f"{args.dataset}_{args.component}",
        bond_spacing_mode="purecomoving",
        component=args.component,
    )
    track2 = load_track2_dataset(dataset=f"{args.dataset}_{args.component}")
    t = track2.frame_times_s
    
    processed_signals = []
    processed_t = None
    for i in range(bond_dataset.signal_matrix.shape[1]):
        sig = bond_dataset.signal_matrix[:, i]
        processed, _ = preprocess_signal(t, sig)
        if processed is None: continue
        if processed_t is None: processed_t = processed.t
        processed_signals.append(processed.y)
        
    y_agg = np.mean(processed_signals, axis=0)
    fs = 1.0 / np.median(np.diff(processed_t))
    
    print(f"Extracting analytic signals for f1={args.f1}Hz, f2={args.f2}Hz, f3={args.f3}Hz...")
    Z1 = extract_analytic_envelope(y_agg, fs, args.f1, args.df)
    Z2 = extract_analytic_envelope(y_agg, fs, args.f2, args.df)
    Z3 = extract_analytic_envelope(y_agg, fs, args.f3, args.df)
    
    win_samples = int(args.window_s * fs)
    step_samples = int(args.step_s * fs)
    
    t_centers = []
    p_values = []
    F_stats = []
    
    print(f"Sliding QAR analysis: Window={args.window_s}s, Step={args.step_s}s, AR(p)={args.ar_order}")
    for start in range(0, len(y_agg) - win_samples, step_samples):
        stop = start + win_samples
        
        Z1_win = Z1[start:stop]
        Z2_win = Z2[start:stop]
        Z3_win = Z3[start:stop]
        
        F, p_val = fit_qar(Z1_win, Z2_win, Z3_win, p=args.ar_order)
        
        t_centers.append(processed_t[start + win_samples//2])
        F_stats.append(F)
        p_values.append(p_val)
        
    t_centers = np.array(t_centers)
    p_values = np.array(p_values)
    log_p = -np.log10(np.maximum(p_values, 1e-15))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. Raw aggregate signal envelope
    axes[0].plot(processed_t, y_agg, color='black', alpha=0.8, linewidth=0.5)
    axes[0].set_title(f"{args.dataset} {args.component} - Aggregate Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # 2. QAR F-Statistic
    axes[1].plot(t_centers, F_stats, color='tab:blue', linewidth=1.5)
    axes[1].set_title(f"QAR F-Statistic (Model Improvement from {args.f1} * {args.f2} -> {args.f3})")
    axes[1].set_ylabel("F-Stat")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Log p-value (Statistical Significance)
    axes[2].plot(t_centers, log_p, color='crimson', linewidth=1.5)
    axes[2].axhline(-np.log10(0.01), color='black', linestyle='--', label='p=0.01 (99% Confidence)')
    axes[2].axhline(-np.log10(0.001), color='tab:orange', linestyle='--', label='p=0.001 (99.9% Confidence)')
    axes[2].set_title(f"Quadratic Coupling Significance (-log10 p-value)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("-log10(p)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_qar_{args.f1}_{args.f2}_{args.f3}.png"
    fig.savefig(out_path)
    print(f"\nAnalysis complete! Saved plot to {out_path}")


if __name__ == "__main__":
    main()
