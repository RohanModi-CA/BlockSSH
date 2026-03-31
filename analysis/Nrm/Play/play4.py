#!/usr/bin/env python3
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from play1 import CONFIG, SCRIPT_DIR, OUTPUT_DIR
from analysis.Nrm.Tools.post_hit_regions import extract_post_hit_regions

@dataclass
class BicoherenceSearchTask:
    f1_target: float
    f2_target: float
    label: str

def compute_complex_spectra(cfg):
    """Extracts complex FFTs from all hit regions using a unified grid."""
    all_regions_y = []
    dts = []
    
    # First pass: Collect all data and find the maximum length
    for bond_id in cfg.bond_ids:
        result = extract_post_hit_regions(dataset=cfg.dataset, component=cfg.component, bond_id=bond_id)
        for region in result.regions:
            all_regions_y.append(region.processed_y)
            dts.append(float(region.processed_t[1] - region.processed_t[0]))
    
    if not all_regions_y:
        raise ValueError("No hit regions found.")

    # Find global max length to set a unified n_fft
    max_len = max(len(y) for y in all_regions_y)
    n_fft = 2**int(np.ceil(np.log2(max_len))) 
    
    # Second pass: Compute FFTs with padding to n_fft
    all_complex_ffts = []
    for y in all_regions_y:
        # np.fft.rfft automatically zero-pads if n > len(y)
        complex_fft = np.fft.rfft(y, n=n_fft)
        all_complex_ffts.append(complex_fft)
    
    # Assume dt is consistent (take the average or just the first)
    avg_dt = sum(dts) / len(dts)
    freqs = np.fft.rfftfreq(n_fft, d=avg_dt)
    
    return freqs, np.vstack(all_complex_ffts)

def bicoherence_grid_search(f1_target, f2_target, freqs, X, search_bins=5):
    """
    Searches a neighborhood for the strongest phase coupling.
    Ensures that idx3 is exactly idx1 + idx2.
    """
    df = freqs[1] - freqs[0]
    i_center = int(round(f1_target / df))
    j_center = int(round(f2_target / df))
    
    best_b2 = -1.0
    best_triplet = (0, 0, 0)

    # Search the local grid
    for i in range(max(0, i_center - search_bins), i_center + search_bins + 1):
        for j in range(max(0, j_center - search_bins), j_center + search_bins + 1):
            k = i + j # Force the sum frequency bin
            
            if k >= X.shape[1]: continue
            
            X1 = X[:, i]
            X2 = X[:, j]
            X3 = X[:, k]
            
            # Squared Bicoherence Calculation
            # E[X1 * X2 * conj(X3)]
            bispec = np.mean(X1 * X2 * np.conj(X3))
            # Normalization
            denom = np.mean(np.abs(X1 * X2)**2) * np.mean(np.abs(X3)**2)
            b2 = (np.abs(bispec)**2 / denom) if denom > 1e-18 else 0
            
            if b2 > best_b2:
                best_b2 = b2
                best_triplet = (freqs[i], freqs[j], freqs[k])
                
    return best_b2, best_triplet

def main():
    print(f"--- Play 4: Grid-Search Bicoherence (Fixed Grid) ---")
    freqs, X_complex = compute_complex_spectra(CONFIG)
    n_hits = X_complex.shape[0]
    
    # These are the triplets you are suspicious of
    tasks = [
        BicoherenceSearchTask(6.34, 12.053, "Sum: 6.34 + 12.05 -> 18.39"),
        BicoherenceSearchTask(18.393, 0.41, "Sideband: 18.39 + 0.41 -> 18.80"),
        BicoherenceSearchTask(6.34, 6.34,   "Harmonic: 2 * 6.34 -> 12.68"),
        BicoherenceSearchTask(3.74, 14.653, "Control: 3.74 + 14.65 -> 18.39"),
    ]

    print(f"Resolution: {freqs[1]-freqs[0]:.4f} Hz | Hits: {n_hits}")
    print(f"{'Target Relationship':<35} | {'Best B^2':<10} | {'Found Triplet (Hz)':<25}")
    print("-" * 85)

    plot_data = []

    for t in tasks:
        # Search locally (+/- 5 bins) for the best coupling
        b2, triplet = bicoherence_grid_search(t.f1_target, t.f2_target, freqs, X_complex, search_bins=5)
        
        # Color coding the verdict based on noise floor (1/N)
        noise_floor = 1.0 / n_hits
        if b2 > 0.4: 
            verdict = "COUPLED"
        elif b2 > 3 * noise_floor:
            verdict = "WEAK/PARTIAL"
        else:
            verdict = "NOISE"
        
        print(f"{t.label:<35} | {b2:<10.4f} | {triplet[0]:.2f}+{triplet[1]:.2f}={triplet[2]:.2f} ({verdict})")
        plot_data.append((t.label, b2))

    # Visualization
    labels, values = zip(*plot_data)
    plt.figure(figsize=(10, 5))
    # Green for strong, Orange for partial, Red for noise
    colors = ['#2ca02c' if v > 0.4 else '#ff7f0e' if v > 3/n_hits else '#d62728' for v in values]
    plt.barh(labels, values, color=colors)
    plt.axvline(1/n_hits, color='black', linestyle='--', label='Stat. Noise Floor (1/N)')
    plt.xlim(0, 1.0)
    plt.title(f"Bicoherence Grid Search Results (N={n_hits} Hits)")
    plt.xlabel("Squared Bicoherence (Coupling Strength)")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "play4_search_results.png")
    plt.show()

if __name__ == "__main__":
    main()
