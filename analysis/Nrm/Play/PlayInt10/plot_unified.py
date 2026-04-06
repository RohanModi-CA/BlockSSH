#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")
REPO_ROOT = add_repo_root_to_path()

from analysis.go.Play.fft_flattening import compute_flattened_component_spectra

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="CDX_10IC", nargs='?')
    parser.add_argument("--json", required=True)
    parser.add_argument("--component", default="x")
    parser.add_argument("--bond-spacing-mode", default="purecomoving")
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        taxonomy = json.load(f)

    fundamentals = taxonomy.get("fundamentals", [])
    children = taxonomy.get("children", [])

    print("Computing flattened spectrum...")
    results = compute_flattened_component_spectra(
        dataset=args.dataset,
        bond_spacing_mode=args.bond_spacing_mode,
        components=(args.component,),
        use_welch=True,
    )
    
    freqs = np.asarray(results[args.component].freq_hz)
    amps = np.asarray(results[args.component].flattened)

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    ax.plot(freqs, amps, color="black", linewidth=1.0, alpha=0.5, label="Flattened Welch Spectrum")

    # Plot fundamentals
    for f in fundamentals:
        if f > 28.0: continue
        ax.axvline(f, color="tab:red", linestyle="-", linewidth=1.5, alpha=0.6)
        idx = np.argmin(np.abs(freqs - f))
        ax.text(f, amps[idx] * 1.5, f"F({f:.2f})", color="tab:red", fontsize=7, rotation=90, va="bottom", ha="center")

    # Plot all children
    for c in children:
        f = c["hz"]
        if f > 28.0: continue
        ax.axvline(f, color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.8)
        idx = np.argmin(np.abs(freqs - f))
        label = f"C({f:.2f})"
        ax.text(f, amps[idx] * 1.1, label, color="tab:blue", fontsize=7, rotation=90, va="bottom", ha="center")

    ax.set_yscale("log")
    ax.set_xlim(0, 25.0)
    ax.set_ylim(bottom=max(1e-4, np.min(amps)), top=np.max(amps) * 10)
    ax.set_title(f"Unified Greedy Taxonomy (Pass 1) - {args.dataset}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Flattened Amplitude")
    ax.grid(True, alpha=0.3)

    out_path = SCRIPT_DIR / "output" / f"{args.dataset}_unified_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
