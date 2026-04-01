#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import compute_complex_spectrogram, compute_one_sided_fft, preprocess_signal

def main():
    parser = argparse.ArgumentParser(description="Average FFTs of inter-hit regions using purecomoving mode.")
    parser.add_argument("--dataset", default="11triv", help="Dataset name")
    parser.add_argument("--component", default="x", help="Component (x, y, or a)")
    parser.add_argument("--bond-index", type=int, default=None, help="Specific bond index to use (0-based). If None, averages all. Use -1 for final bond.")
    parser.add_argument("--output", default="analysis/NL/out/purecomoving_hit_avg_fft_normalized.png", help="Output plot path")
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize each FFT by its RMS in norm-range")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--norm-range", type=float, nargs=2, default=[3.0, 28.0], help="Frequency range for normalization RMS")
    parser.add_argument("--log-average", action="store_true", help="Perform log-averaging instead of arithmetic mean")
    parser.add_argument("--exclude-after", type=float, default=1.0, help="Seconds after each hit to exclude. Default: 1.0")
    parser.add_argument("--exclude-before", type=float, default=1.0, help="Seconds before next hit to exclude. Default: 1.0")
    parser.add_argument("--show", action="store_true", default=True, help="Show the plot on screen (default: True)")
    parser.add_argument("--no-show", action="store_false", dest="show")
    args = parser.parse_args()

    if not args.show:
        plt.switch_backend("Agg")

    # 1. Load purecomoving dataset
    print(f"Loading {args.dataset}_{args.component} in purecomoving mode...")
    ds = load_bond_signal_dataset(
        dataset=f"{args.dataset}_{args.component}",
        bond_spacing_mode="purecomoving",
        component=args.component
    )
    
    n_bonds = ds.signal_matrix.shape[1]
    if args.bond_index is not None:
        idx = args.bond_index
        if idx < 0:
            idx = n_bonds + idx
        print(f"Using bond index {idx} of {n_bonds}")
        raw_signal = ds.signal_matrix[:, idx]
        label = f"Bond {idx}"
    else:
        # Average across all bonds for a cleaner representative signal
        print(f"Averaging across all {n_bonds} bonds")
        raw_signal = np.nanmean(ds.signal_matrix, axis=1)
        label = "Average of all bonds"
    
    t = ds.frame_times_s
    
    # Preprocess (handle NaNs, detrend, etc.)
    processed, err = preprocess_signal(t, raw_signal)
    if processed is None:
        print(f"Preprocessing failed: {err}")
        return
    
    sig = processed.y
    pt = processed.t
    fs = 1.0 / np.median(np.diff(pt))

    # 2. Load Existing Hits
    hits_csv = REPO_ROOT / "analysis/NL/out" / f"{args.dataset}_comparison_purecomoving" / f"{args.dataset}__{args.component}__prototype_hits.csv"
    if not hits_csv.exists():
        print(f"Hits CSV not found: {hits_csv}. Please run prototype_hits.py first.")
        return
        
    import csv
    hit_times = []
    with hits_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hit_times.append(float(row['time_s']))
    hit_times = np.asarray(hit_times)
    print(f"Loaded {len(hit_times)} hits from CSV.")

    # 3. Define Inter-hit Regions
    # Regions: (Hit N + exclude_after) to (Hit N+1 - exclude_before)
    # Plus the region after the last hit
    regions = []
    for i in range(len(hit_times)):
        start_t = hit_times[i] + args.exclude_after
        if i < len(hit_times) - 1:
            end_t = hit_times[i+1] - args.exclude_before
        else:
            end_t = pt[-1] - 1.0
            
        if end_t > start_t + 1.0: # Minimum 1s duration
            regions.append((start_t, end_t))

    # 4. Compute and Average FFTs
    print(f"Averaging FFTs from {len(regions)} regions...")
    all_ffts = []
    freq_grid = None
    
    for start, end in regions:
        mask = (pt >= start) & (pt <= end)
        seg_y = sig[mask]
        if seg_y.size < 100: continue
        
        fft_res = compute_one_sided_fft(seg_y, 1.0/fs)
        if freq_grid is None:
            freq_grid = fft_res.freq
        
        # Interpolate to common grid if necessary (though fs is constant here)
        amp = np.interp(freq_grid, fft_res.freq, fft_res.amplitude)
        
        if args.normalize:
            f_mask = (freq_grid >= args.norm_range[0]) & (freq_grid <= args.norm_range[1])
            if np.any(f_mask):
                rms = np.sqrt(np.mean(amp[f_mask]**2))
                if rms > 0:
                    amp = amp / rms
                    
        all_ffts.append(amp)
    
    if args.log_average:
        eps = 1e-12
        avg_fft = 10**(np.mean(np.log10(np.array(all_ffts) + eps), axis=0))
    else:
        avg_fft = np.mean(all_ffts, axis=0)

    # 5. Plotting
    print(f"Saving plot to {args.output}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # Timeseries
    ax1.plot(pt, sig, color='black', lw=0.5, alpha=0.7)
    for start, end in regions:
        ax1.axvspan(start, end, color='tab:green', alpha=0.2)
    for ht in hit_times:
        ax1.axvline(ht, color='crimson', ls='--', lw=1)
    ax1.set_title(f"{args.dataset} {args.component} Purecomoving Signal ({label}) with Inter-hit Analysis Regions")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    
    # Averaged FFT
    ax2.plot(freq_grid, avg_fft, color='black', lw=1.2)
    ax2.set_yscale('log')
    ax2.set_xlim(0, 30) # Focus on interesting range
    mode_str = "Log-Averaged" if args.log_average else "Averaged"
    norm_str = f"Normalized [{args.norm_range[0]}-{args.norm_range[1]}Hz]" if args.normalize else "Unnormalized"
    ax2.set_title(f"{mode_str} Spectrum of Inter-hit Regions ({norm_str})")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude (normalized)" if args.normalize else "Amplitude")
    ax2.grid(True, which='both', alpha=0.2)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")
    
    if args.show:
        plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
