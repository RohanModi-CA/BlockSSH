#!/usr/bin/env python3
import argparse
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import preprocess_signal

def main():
    parser = argparse.ArgumentParser(description="Weighted Welch spectral estimation suppressing hits.")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--component", default="x", help="Component (x, y, or a)")
    parser.add_argument("--bond-index", type=int, default=None, help="Specific bond (0-based). If None, averages all. Use -1 for final.")
    parser.add_argument("--window-len-s", type=float, default=10.0, help="Welch window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction (0-1)")
    parser.add_argument("--sigma-s", type=float, default=1.0, help="Width of hit suppression (sigma of Gaussian dip)")
    parser.add_argument("--output", help="Output plot path")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        plt.switch_backend("Agg")

    # 1. Load Data
    print(f"Loading {args.dataset}_{args.component} in purecomoving mode...")
    ds = load_bond_signal_dataset(
        dataset=f"{args.dataset}_{args.component}",
        bond_spacing_mode="purecomoving",
        component=args.component
    )
    
    n_bonds = ds.signal_matrix.shape[1]
    if args.bond_index is not None:
        idx = args.bond_index if args.bond_index >= 0 else n_bonds + args.bond_index
        raw_signal = ds.signal_matrix[:, idx]
        label = f"Bond {idx}"
    else:
        raw_signal = np.nanmean(ds.signal_matrix, axis=1)
        label = "Average of all bonds"
    
    processed, _ = preprocess_signal(ds.frame_times_s, raw_signal)
    sig = processed.y
    t = processed.t
    fs = processed.Fs

    # 2. Load Hits
    hits_csv = REPO_ROOT / "analysis/NL/out" / f"{args.dataset}_comparison_purecomoving" / f"{args.dataset}__{args.component}__prototype_hits.csv"
    if not hits_csv.exists():
        print(f"Hits CSV not found: {hits_csv}. Run prototype_hits.py first.")
        return
        
    hit_times = []
    with hits_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hit_times.append(float(row['time_s']))
    print(f"Loaded {len(hit_times)} hits.")

    # 3. Create Continuous Weight Trace
    # w(t) = 1 - sum(exp(-(t-hit)^2 / 2sigma^2)) -> clamped to [0, 1]
    # We use a product for a cleaner "dip" near multiple hits
    weight_trace = np.ones_like(t)
    for ht in hit_times:
        weight_trace *= (1.0 - np.exp(-((t - ht)**2) / (2 * args.sigma_s**2)))
    
    # 4. Manual Weighted Welch
    nperseg = int(round(args.window_len_s * fs))
    noverlap = int(round(args.overlap * nperseg))
    step = nperseg - noverlap
    
    win = windows.hann(nperseg)
    win_sq_sum = np.sum(win**2)
    
    freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
    psd_weighted_acc = np.zeros_like(freqs)
    psd_normal_acc = np.zeros_like(freqs)
    total_weight = 0.0
    count = 0
    
    print("Computing weighted segments...")
    for start_idx in range(0, len(sig) - nperseg, step):
        end_idx = start_idx + nperseg
        segment = sig[start_idx:end_idx]
        w_segment = weight_trace[start_idx:end_idx]
        
        # Window Weight: average weight of the segment
        W_i = np.mean(w_segment)
        
        # Periodogram
        fft = np.fft.rfft(segment * win)
        psd = (np.abs(fft)**2) / (fs * win_sq_sum)
        if nperseg % 2 == 0:
            psd[1:-1] *= 2
        else:
            psd[1:] *= 2
            
        psd_weighted_acc += W_i * psd
        psd_normal_acc += psd
        total_weight += W_i
        count += 1
        
    avg_psd_weighted = psd_weighted_acc / total_weight
    avg_psd_normal = psd_normal_acc / count
    
    # Amplitudes
    amp_weighted = np.sqrt(avg_psd_weighted)
    amp_normal = np.sqrt(avg_psd_normal)

    # 5. Plotting
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    
    # Timeseries + Weight
    ax1.plot(t, sig, color='0.7', lw=0.5, label="Signal")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, weight_trace, color='tab:blue', lw=1.2, alpha=0.6, label="Weight Trace")
    ax1_twin.set_ylabel("Weight", color='tab:blue')
    ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
    ax1_twin.set_ylim(-0.1, 1.1)
    
    for ht in hit_times:
        ax1.axvline(ht, color='crimson', ls='--', alpha=0.4)
        
    ax1.set_title(f"Weighted Welch Analysis: {args.dataset} {args.component} ({label})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper left')
    
    # Spectrum Comparison
    ax2.plot(freqs, amp_normal, color='0.5', lw=1.0, alpha=0.5, label="Standard Welch (unweighted)")
    ax2.plot(freqs, amp_weighted, color='black', lw=1.3, label=f"Weighted Welch (sigma={args.sigma_s}s)")
    ax2.set_yscale('log')
    ax2.set_xlim(0, 30)
    ax2.set_title("Spectral Comparison: Standard vs Weighted Welch")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()

    out_name = args.output or f"analysis/NL/out/{args.dataset}_weighted_welch.png"
    out_path = REPO_ROOT / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")
    
    if args.show:
        plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
