#!/usr/bin/env python3
import argparse
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import preprocess_signal

def main():
    parser = argparse.ArgumentParser(description="Ensemble-averaged spectrogram of hit decay across all hits and bonds.")
    parser.add_argument("--dataset", default="11triv", help="Dataset name")
    parser.add_argument("--component", default="x", help="Component")
    parser.add_argument("--duration-s", type=float, default=15.0, help="Seconds to plot after hit")
    parser.add_argument("--output", help="Output plot path")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        plt.switch_backend("Agg")

    # 1. Load Data (All Bonds)
    print(f"Loading {args.dataset}_{args.component} in purecomoving mode...")
    ds = load_bond_signal_dataset(
        dataset=f"{args.dataset}_{args.component}",
        bond_spacing_mode="purecomoving",
        component=args.component
    )
    n_bonds = ds.signal_matrix.shape[1]
    
    # 2. Load Hits
    hits_csv = REPO_ROOT / "analysis/NL/out" / f"{args.dataset}_comparison_purecomoving" / f"{args.dataset}__{args.component}__prototype_hits.csv"
    hit_times = []
    with hits_csv.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hit_times.append(float(row['time_s']))
    print(f"Loaded {len(hit_times)} hits.")

    all_specs = []
    all_sigs = []
    
    # 3. Process each bond
    for b_idx in range(n_bonds):
        print(f"Processing bond {b_idx}...")
        raw_signal = ds.signal_matrix[:, b_idx]
        processed, _ = preprocess_signal(ds.frame_times_s, raw_signal)
        if processed is None: continue
        sig = processed.y
        t = processed.t
        fs = processed.Fs
        
        # Window settings (consistent across all)
        nperseg = int(round(0.5 * fs))
        noverlap = int(round(0.9 * nperseg))
        
        # 4. Extract segments for each hit
        for ht in hit_times:
            t_start = ht - 2.0
            t_end = ht + args.duration_s
            mask = (t >= t_start) & (t <= t_end)
            if np.sum(mask) < nperseg: continue
            
            seg_y = sig[mask]
            
            # Timeseries accumulation
            # Ensure consistent length by interpolation if needed (but fs is stable)
            rel_t = np.linspace(-2.0, args.duration_s, int((args.duration_s + 2.0) * fs))
            seg_y_interp = np.interp(rel_t, t[mask] - ht, seg_y)
            all_sigs.append(seg_y_interp)
            
            # Spectrogram
            f_spec, t_spec, Sxx = spectrogram(seg_y, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
            all_specs.append(Sxx)
            
    # 5. Average Results
    avg_sig = np.mean(all_sigs, axis=0)
    avg_Sxx = np.mean(all_specs, axis=0)
    t_spec_rel = t_spec - 2.0 # Adjust to hit-relative time
    rel_t_axis = np.linspace(-2.0, args.duration_s, avg_sig.size)

    # 6. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    
    # Timeseries
    ax1.plot(rel_t_axis, avg_sig, color='black', lw=0.8)
    ax1.axvline(0, color='crimson', ls='--', label="Hit (T=0)")
    ax1.axvspan(1.0, args.duration_s, color='tab:green', alpha=0.1, label="Inter-hit region")
    ax1.set_title(f"Ensemble-Averaged Hit Decay: {args.dataset} {args.component} ({len(all_sigs)} segments)")
    ax1.set_ylabel("Average Amplitude")
    ax1.legend()
    
    # Spectrogram
    Sxx_db = 10 * np.log10(avg_Sxx + 1e-15)
    pcm = ax2.pcolormesh(t_spec_rel, f_spec, Sxx_db, shading='gouraud', cmap='magma', vmin=Sxx_db.max()-40)
    fig.colorbar(pcm, ax=ax2, label="Average Power (dB)")
    ax2.set_ylim(0, 30)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time from Hit (s)")
    
    # Highlight 16Hz
    ax2.axhline(16, color='cyan', ls=':', alpha=0.5, label="16Hz Harmonic")
    ax2.legend(loc='upper right')

    out_name = args.output or f"analysis/NL/out/{args.dataset}_ensemble_decay.png"
    out_path = REPO_ROOT / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
