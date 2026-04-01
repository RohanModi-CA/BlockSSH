#!/usr/bin/env python3
import argparse
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows, find_peaks

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import preprocess_signal

def main():
    parser = argparse.ArgumentParser(description="Sweep Welch Wall Strength.")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--component", default="x", help="Component (x, y, or a)")
    parser.add_argument("--bond-index", type=int, default=None, help="Specific bond (0-based). None for average. -1 for final.")
    parser.add_argument("--window-len-s", type=float, default=30.0, help="Welch window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.75, help="Overlap fraction (0-1)")
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
    hit_times = np.array(hit_times)
    print(f"Loaded {len(hit_times)} hits.")

    # 3. Setup Welch Segments
    nperseg = int(round(args.window_len_s * fs))
    noverlap = int(round(args.overlap * nperseg))
    step = nperseg - noverlap
    
    win = windows.hann(nperseg)
    win_sq_sum = np.sum(win**2)
    freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
    
    start_indices = list(range(0, len(sig) - nperseg + 1, step))
    n_segments = len(start_indices)
    PSD_matrix = np.zeros((n_segments, len(freqs)))
    C_i = np.zeros(n_segments) # Contamination metric [0, 1]
    
    D_safe = (args.window_len_s / 2.0) + 1.0 # Window half-length + 1s hit buffer
    
    print(f"Pre-computing {n_segments} Welch segments...")
    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + nperseg
        segment = sig[start_idx:end_idx]
        
        # Periodogram
        fft = np.fft.rfft(segment * win)
        psd = (np.abs(fft)**2) / (fs * win_sq_sum)
        if nperseg % 2 == 0: psd[1:-1] *= 2
        else: psd[1:] *= 2
        PSD_matrix[i, :] = psd
        
        # Window Contamination
        t_center = t[start_idx] + (args.window_len_s / 2.0)
        d_i = np.min(np.abs(hit_times - t_center))
        C_i[i] = max(0.0, 1.0 - (d_i / D_safe))

    # 4. Sweep Wall Strength (alpha)
    alphas = np.linspace(0, 1, 100)
    A_matrix = np.zeros((len(alphas), len(freqs)))
    
    f_mask = (freqs >= 3.0) & (freqs <= 28.0)
    
    print("Sweeping wall strength alpha [0 -> 1]...")
    for j, alpha in enumerate(alphas):
        k = alpha / (1.001 - alpha) # Maps alpha [0,1] -> k [0, ~1000]
        W = (1.0 - C_i)**k
        
        if np.sum(W) < 1e-9:
            A_matrix[j, :] = np.nan
            continue
            
        psd_avg = np.sum(W[:, None] * PSD_matrix, axis=0) / np.sum(W)
        amp = np.sqrt(psd_avg)
        
        # Normalize
        rms = np.sqrt(np.mean(amp[f_mask]**2))
        if rms > 0:
            amp = amp / rms
            
        A_matrix[j, :] = amp

    # 5. Plotting
    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)
    
    # Plot 1: Heatmap
    # X: Freqs, Y: Alphas, Z: Normalized Amplitude (dB for visibility)
    # Filter heatmap to interesting freqs
    plot_f_mask = (freqs >= 0) & (freqs <= 30)
    freqs_plot = freqs[plot_f_mask]
    A_plot = A_matrix[:, plot_f_mask]
    
    pcm = ax1.pcolormesh(freqs_plot, alphas, 20 * np.log10(A_plot + 1e-12), 
                         shading='auto', cmap='turbo', vmin=-10, vmax=25)
    fig.colorbar(pcm, ax=ax1, label="Normalized Amplitude (dB)")
    ax1.set_title(f"Impact Sensitivity Heatmap: {args.dataset}_{args.component} ({label})")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Wall Strength ($\\alpha$) [0=No Mask, 1=Strict Exclusion]")
    
    # Plot 2: Peak Trajectories
    # Find most interesting peaks by looking at extreme alphas
    peaks_0, _ = find_peaks(A_matrix[0, f_mask], prominence=0.5)
    peaks_1, _ = find_peaks(A_matrix[-1, f_mask], prominence=0.5)
    
    # Map back to full frequency indices
    f_indices = np.where(f_mask)[0]
    all_peak_indices = np.unique(np.concatenate([f_indices[peaks_0], f_indices[peaks_1]]))
    
    # Sort by maximum normalized amplitude achieved anywhere in the sweep
    peak_max_amps = [np.nanmax(A_matrix[:, p]) for p in all_peak_indices]
    top_peak_indices = [all_peak_indices[i] for i in np.argsort(peak_max_amps)[::-1][:8]] # Top 8 peaks
    
    for p_idx in top_peak_indices:
        f_val = freqs[p_idx]
        traj = A_matrix[:, p_idx]
        
        # Determine behavior: does it grow or shrink?
        change = traj[-1] - traj[0]
        if np.isnan(change): change = 0
        style = '-' if change > 0 else '--'
        
        ax2.plot(alphas, traj, style, lw=2.0, label=f"{f_val:.2f} Hz (Change: {change:+.2f})")

    ax2.set_yscale('log')
    ax2.set_title("Peak Trajectories: Normalized Amplitude vs. Wall Strength")
    ax2.set_xlabel("Wall Strength ($\\alpha$)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    out_name = args.output or f"analysis/NL/out/{args.dataset}_wall_sweep.png"
    out_path = REPO_ROOT / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {out_path}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
