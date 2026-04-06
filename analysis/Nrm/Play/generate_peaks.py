#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

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
from analysis.tools.signal import compute_one_sided_fft, preprocess_signal
from analysis.tools.io import load_track2_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name, e.g., CDX_10IC")
    parser.add_argument("--component", default="x")
    parser.add_argument("--bond-spacing-mode", default="purecomoving")
    args = parser.parse_args()

    dataset = args.dataset
    component = args.component
    bond_spacing_mode = args.bond_spacing_mode
    
    print(f"Loading dataset {dataset} ({component}) with {bond_spacing_mode}...")
    bond_dataset = load_bond_signal_dataset(
        dataset=f"{dataset}_{component}",
        bond_spacing_mode=bond_spacing_mode,
        component=component,
    )
    
    track2 = load_track2_dataset(dataset=f"{dataset}_{component}")
    t = track2.frame_times_s
    
    # Preprocess and average FFT across all bonds
    all_amps = []
    freqs = None
    
    for i in range(bond_dataset.signal_matrix.shape[1]):
        sig = bond_dataset.signal_matrix[:, i]
        processed, _ = preprocess_signal(t, sig)
        if processed is None: continue
        
        dt = float(processed.t[1] - processed.t[0])
        res = compute_one_sided_fft(processed.y, dt)
        if freqs is None: freqs = res.freq
        all_amps.append(res.amplitude)
        
    avg_amp = np.mean(all_amps, axis=0)
    
    # Find peaks
    # 0.1 Hz spacing
    df = freqs[1] - freqs[0]
    distance = int(0.1 / df)
    
    peak_indices, _ = find_peaks(avg_amp, distance=distance, prominence=0.001)
    peak_freqs = freqs[peak_indices]
    peak_amps = avg_amp[peak_indices]
    
    # Sort by amplitude and take top 7
    top_indices = np.argsort(peak_amps)[-7:][::-1]
    top_freqs = peak_freqs[top_indices]
    top_amps = peak_amps[top_indices]
    
    # Sort by frequency for the CSV
    sort_idx = np.argsort(top_freqs)
    final_freqs = top_freqs[sort_idx]
    
    print("Top 7 peaks found:")
    for f in final_freqs:
        print(f"  {f:.3f} Hz")
        
    # Save to CSV
    out_path = REPO_ROOT / "analysis" / "configs" / "peaks" / f"{dataset}_top7.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write("frequency_hz,label\n")
        for freq in final_freqs:
            label = f"{freq:.2f}Hz"
            f.write(f"{freq:.4f},{label}\n")
            
    print(f"Saved peaks to {out_path}")

if __name__ == "__main__":
    main()
