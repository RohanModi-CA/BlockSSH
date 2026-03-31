#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# Importing your existing infrastructure
import sys
from play1 import CONFIG, collect_region_spectra, SCRIPT_DIR, OUTPUT_DIR

# We need the raw extraction tool to get complex values
from analysis.Nrm.Tools.post_hit_regions import extract_post_hit_regions

@dataclass
class BicoherenceResult:
    f1: float
    f2: float
    f_sum: float
    b_sq: float
    label: str

def compute_complex_spectra(cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts complex FFTs for every hit region to preserve phase information.
    Returns: (freq_grid, complex_matrix) where matrix is (n_hits, n_freqs)
    """
    all_complex_ffts = []
    freq_grid = None

    for bond_id in cfg.bond_ids:
        result = extract_post_hit_regions(
            dataset=cfg.dataset,
            component=cfg.component,
            bond_id=bond_id,
        )
        for region in result.regions:
            y = region.processed_y
            dt = float(region.processed_t[1] - region.processed_t[0])
            
            # Use rfft to get complex coefficients
            n = len(y)
            freqs = np.fft.rfftfreq(n, d=dt)
            complex_fft = np.fft.rfft(y)

            if freq_grid is None:
                freq_grid = freqs
            
            # Interpolate complex values to a common grid if lengths differ
            if len(freqs) != len(freq_grid):
                real_interp = np.interp(freq_grid, freqs, complex_fft.real)
                imag_interp = np.interp(freq_grid, freqs, complex_fft.imag)
                all_complex_ffts.append(real_interp + 1j * imag_interp)
            else:
                all_complex_ffts.append(complex_fft)

    return freq_grid, np.vstack(all_complex_ffts)

def get_bicoherence(f1: float, f2: float, freqs: np.ndarray, X: np.ndarray) -> float:
    """
    Calculates Squared Bicoherence for the triplet (f1, f2, f1+f2).
    X is the complex matrix (n_hits, n_freqs).
    """
    # Find closest indices in the frequency grid
    idx1 = np.argmin(np.abs(freqs - f1))
    idx2 = np.argmin(np.abs(freqs - f2))
    idx3 = np.argmin(np.abs(freqs - (f1 + f2)))

    X1 = X[:, idx1]
    X2 = X[:, idx2]
    X3 = X[:, idx3]

    # Bispectrum: E[X(f1) * X(f2) * conj(X(f1+f2))]
    bispec = np.mean(X1 * X2 * np.conj(X3))
    
    # Normalization factor
    denom = np.mean(np.abs(X1 * X2)**2) * np.mean(np.abs(X3)**2)
    
    if denom == 0: return 0.0
    return np.abs(bispec)**2 / denom

def main():
    print(f"--- Bicoherence Analysis: {CONFIG.dataset} ---")
    
    # 1. Collect Data
    freqs, X_complex = compute_complex_spectra(CONFIG)
    n_hits = X_complex.shape[0]
    print(f"Analyzed {n_hits} hit regions.")

    # 2. Define Triplets to Test
    # Target 1: Is 18.393 the sum of 6.34 and 12.05?
    # Target 2: Is 18.803 a sideband of 18.393 and 0.41?
    # Target 3: Negative Control (Random frequencies that sum to 18.393)
    triplets = [
        (6.34, 12.053, "Main Nonlinear Sum (6.34 + 12.05)"),
        (18.393, 0.41, "Pendulum Sideband (18.39 + 0.41)"),
        (3.74, 14.653, "Negative Control (Coincidental Sum)"),
        (6.34, 6.34,   "Second Harmonic (6.34 + 6.34)")
    ]

    results = []
    for f1, f2, label in triplets:
        b2 = get_bicoherence(f1, f2, freqs, X_complex)
        results.append(BicoherenceResult(f1, f2, f1+f2, b2, label))

    # 3. Report Results
    print(f"{'Relationship':<40} | {'f1':<7} | {'f2':<7} | {'f_sum':<7} | {'Bicoherence^2':<15}")
    print("-" * 90)
    for r in results:
        status = "!!! COUPLED !!!" if r.b_sq > 0.5 else "Independent"
        print(f"{r.label:<40} | {r.f1:<7.3f} | {r.f2:<7.3f} | {r.f_sum:<7.3f} | {r.b_sq:<15.4f} {status}")

    # 4. Simple Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{r.f1}+{r.f2}" for r in results]
    vals = [r.b_sq for r in results]
    colors = ['green' if v > 0.5 else 'gray' for v in vals]
    
    ax.bar(labels, vals, color=colors)
    ax.axhline(1/n_hits, color='red', linestyle='--', label='Statistical Noise Floor')
    ax.set_ylabel("Squared Bicoherence")
    ax.set_title(f"Phase Coupling Strength - {CONFIG.dataset}")
    ax.legend()
    
    plt.savefig(OUTPUT_DIR / "play3_bicoherence_results.png")
    print(f"\nResults saved to {OUTPUT_DIR / 'play3_bicoherence_results.png'}")
    plt.show()

if __name__ == "__main__":
    main()
