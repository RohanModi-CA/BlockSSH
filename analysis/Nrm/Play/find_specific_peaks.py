#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

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
    dataset = "CDX_10IC"
    component = "x"
    bond_spacing_mode = "purecomoving"
    
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
    
    # Target approximations requested by user
    targets = [0.41, 3.35, 11.3, 12.0, 15.9, 16.6, 18.0, 18.3]
    exact_peaks = []
    
    search_half_width = 0.4  # Hz
    
    for target in targets:
        mask = (freqs >= target - search_half_width) & (freqs <= target + search_half_width)
        local_freqs = freqs[mask]
        local_amps = avg_amp[mask]
        
        if len(local_freqs) > 0:
            max_idx = np.argmax(local_amps)
            exact_freq = local_freqs[max_idx]
            exact_peaks.append(exact_freq)
            print(f"Target ~{target} Hz -> Found exact peak at {exact_freq:.4f} Hz")
        else:
            print(f"Warning: No frequencies found near target {target} Hz")
            
    # Sort and remove duplicates just in case
    exact_peaks = sorted(list(set(exact_peaks)))
    
    # Save to CSV
    out_path = REPO_ROOT / "analysis" / "configs" / "peaks" / "CDX_10IC_custom.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write("frequency_hz,label\n")
        for freq in exact_peaks:
            label = f"{freq:.2f}Hz"
            f.write(f"{freq:.4f},{label}\n")
            
    print(f"Saved custom peaks to {out_path}")

if __name__ == "__main__":
    main()
