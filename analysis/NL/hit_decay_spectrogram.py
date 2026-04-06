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
    parser = argparse.ArgumentParser(description="High-resolution spectrogram of hit decay.")
    parser.add_argument("--dataset", default="11triv", help="Dataset name")
    parser.add_argument("--component", default="x", help="Component")
    parser.add_argument("--hit-index", type=int, default=2, help="Which hit to zoom in on (1-based)")
    parser.add_argument("--duration-s", type=float, default=15.0, help="Seconds to plot after hit")
    parser.add_argument("--output", help="Output plot path")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        plt.switch_backend("Agg")

    # 1. Load Data (Last Bond)
    print(f"Loading {args.dataset}_{args.component} last bond...")
    ds = load_bond_signal_dataset(
        dataset=f"{args.dataset}_{args.component}",
        bond_spacing_mode="purecomoving",
        component=args.component
    )
    raw_signal = ds.signal_matrix[:, -1]
    processed, _ = preprocess_signal(ds.frame_times_s, raw_signal)
    sig = processed.y
    t = processed.t
    fs = processed.Fs

    # 2. Load Hits
    hits_csv = REPO_ROOT / "analysis/NL/out" / f"{args.dataset}_comparison_purecomoving" / f"{args.dataset}__{args.component}__prototype_hits.csv"
    hit_times = []
    with hits_csv.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hit_times.append(float(row['time_s']))
    
    t_hit = hit_times[args.hit_index - 1]
    print(f"Zooming on Hit {args.hit_index} at {t_hit:.3f}s")

    # 3. Extract Segment (2s before, duration_s after)
    t_start = t_hit - 2.0
    t_end = t_hit + args.duration_s
    mask = (t >= t_start) & (t <= t_end)
    seg_t = t[mask] - t_hit # Relative to hit
    seg_y = sig[mask]

    # 4. Compute Spectrogram
    # Using short windows for high time resolution
    nperseg = int(round(0.5 * fs)) # 500ms window
    noverlap = int(round(0.9 * nperseg)) # 90% overlap
    f_spec, t_spec, Sxx = spectrogram(seg_y, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    t_spec = t_spec - 2.0 # Adjust to hit-relative time

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    
    # Timeseries
    ax1.plot(seg_t, seg_y, color='black', lw=0.8)
    ax1.axvline(0, color='crimson', ls='--', label="Hit")
    ax1.axvspan(1.0, args.duration_s, color='tab:green', alpha=0.1, label="Inter-hit Region Start")
    ax1.set_title(f"Hit Decay Zoom: {args.dataset} {args.component} (Hit {args.hit_index})")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    
    # Spectrogram
    Sxx_db = 10 * np.log10(Sxx + 1e-15)
    pcm = ax2.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='magma', vmin=Sxx_db.max()-40)
    fig.colorbar(pcm, ax=ax2, label="Power (dB)")
    ax2.set_ylim(0, 30)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time from Hit (s)")
    
    # Highlight 16Hz
    ax2.axhline(16, color='cyan', ls=':', alpha=0.5, label="16Hz Harmonic")
    ax2.legend(loc='upper right')

    out_name = args.output or f"analysis/NL/out/{args.dataset}_hit{args.hit_index}_decay.png"
    out_path = REPO_ROOT / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
