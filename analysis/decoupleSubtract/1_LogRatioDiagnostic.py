#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the analysis package is in the path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.cli import (
    add_track_data_root_arg, add_bond_spacing_mode_arg, add_normalization_args,
    add_signal_processing_args, add_average_domain_args,
)
from analysis.tools.groups import write_temp_component_selection_config
from analysis.tools.selection import build_configured_bond_signals, load_dataset_selection
from analysis.tools.spectral import compute_average_spectrum, compute_welch_contributions
from analysis.tools.flattening import apply_global_baseline_processing_to_results
from analysis.plotting.common import ensure_parent_dir

CANONICAL_COMPONENTS = ("x", "y", "a")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Numerical diagnostics for component coupling (with flattening)."
    )
    parser.add_argument("dataset", help="Dataset name, e.g. 11triv.")
    add_track_data_root_arg(parser)
    add_bond_spacing_mode_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_signal_processing_args(parser)
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="analysis/decoupleSubtract/plots")
    parser.add_argument("--flatten-ref-band", type=float, nargs=2, default=[20.0, 30.0],
                        help="Reference band for flattening in Hz. Default: 20 30")
    parser.add_argument("--baseline-match", choices=["none", "x", "y", "a"], default="x",
                        help="Component to baseline-match all others to. Default: x")
    parser.add_argument(
        "--interp-kind",
        default="cubic",
        choices=["linear", "quadratic", "cubic"],
        help="Interpolation kind for common frequency grid. Default: cubic",
    )
    parser.add_argument(
        "--coarsest",
        action="store_true",
        help="Use the coarsest (max df) frequency grid instead of finest (default)",
    )
    return parser


def compute_component_spectra(args) -> OrderedDict[str, object]:
    results: OrderedDict[str, object] = OrderedDict()
    for component in CANONICAL_COMPONENTS:
        print(f"  Processing {component}...")
        temp_config = write_temp_component_selection_config(
            [args.dataset], component=component,
            track_data_root=args.track_data_root,
            prefix=f"decouple_diag_{args.dataset}_{component}_",
        )
        config = load_dataset_selection(temp_config.name)
        records = build_configured_bond_signals(
            config, track_data_root=args.track_data_root,
            bond_spacing_mode=args.bond_spacing_mode,
        )
        if not records:
            print(f"    Skip: no records for {component}")
            continue

        contributions = compute_welch_contributions(
            records, welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            longest=args.longest, handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )
        if not contributions:
            print(f"    Skip: no contributions for {component}")
            continue

        avg_result = compute_average_spectrum(
            contributions, normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            average_domain=args.average_domain,
            grid_mode="coarsest" if args.coarsest else "finest",
            interp_kind=args.interp_kind,
        )
        results[component] = avg_result
        print(f"    {len(contributions)} contributors, "
              f"freq [{avg_result.freq_grid[0]:.4f}, {avg_result.freq_grid[-1]:.4f}] Hz")
    return results


def get_common_freq(results: OrderedDict) -> np.ndarray:
    min_f = max(r.freq_grid[0] for r in results.values())
    max_f = min(r.freq_grid[-1] for r in results.values())
    lead = next(iter(results.values())).freq_grid
    mask = (lead >= min_f) & (lead <= max_f)
    return lead[mask]


def interp_onto_common(results: OrderedDict, common_freq: np.ndarray) -> dict:
    out = {}
    for comp, result in results.items():
        amp = np.interp(common_freq, result.freq_grid, result.avg_amp)
        out[comp] = amp
    return out


def compute_numerical_diagnostics(common_freq: np.ndarray, amps: dict,
                                   flat_amps: dict, args) -> dict:
    """Compute quantitative coupling metrics from flattened spectra."""
    eps = np.finfo(float).eps
    pairs = [("x", "y"), ("x", "a"), ("y", "a")]
    diagnostics = {"pairs": {}, "global": {}}

    for c1, c2 in pairs:
        if c1 not in flat_amps or c2 not in flat_amps:
            continue

        a1 = flat_amps[c1]
        a2 = flat_amps[c2]
        ratio = a1 / (a2 + eps)
        log_ratio = np.log10(ratio + eps)

        # Key numerical stats
        pair_diag = {
            "mean_ratio": float(np.mean(ratio)),
            "median_ratio": float(np.median(ratio)),
            "std_ratio": float(np.std(ratio)),
            "mean_log10_ratio": float(np.mean(log_ratio)),
            "median_log10_ratio": float(np.median(log_ratio)),
            "std_log10_ratio": float(np.std(log_ratio)),
            "min_ratio": float(np.min(ratio)),
            "max_ratio": float(np.max(ratio)),
            "fraction_within_2x": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
            "fraction_within_10x": float(np.mean((ratio > 0.1) & (ratio < 10.0))),
        }

        # Rolling statistics to find stable regions
        window = max(3, len(common_freq) // 50)  # ~50 windows across spectrum
        if window % 2 == 0:
            window += 1

        rolling_medians = []
        rolling_stds = []
        rolling_freqs = []
        for i in range(0, len(ratio) - window + 1, window // 2):
            chunk = ratio[i:i + window]
            chunk_log = log_ratio[i:i + window]
            rolling_medians.append(float(np.median(chunk)))
            rolling_stds.append(float(np.std(chunk_log)))
            rolling_freqs.append(float(np.mean(common_freq[i:i + window])))

        # Find most stable regions (lowest std in log-ratio)
        if rolling_stds:
            best_idx = int(np.argmin(rolling_stds))
            pair_diag["most_stable_region"] = {
                "center_freq": rolling_freqs[best_idx],
                "median_ratio": rolling_medians[best_idx],
                "log10_ratio_std": rolling_stds[best_idx],
            }
            worst_idx = int(np.argmax(rolling_stds))
            pair_diag["least_stable_region"] = {
                "center_freq": rolling_freqs[worst_idx],
                "median_ratio": rolling_medians[worst_idx],
                "log10_ratio_std": rolling_stds[worst_idx],
            }

        # Percentiles of the ratio distribution
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pair_diag["ratio_percentiles"] = {
            f"p{p}": float(np.percentile(ratio, p)) for p in percentiles
        }
        pair_diag["log10_ratio_percentiles"] = {
            f"p{p}": float(np.percentile(log_ratio, p)) for p in percentiles
        }

        diagnostics["pairs"][f"{c1}_vs_{c2}"] = pair_diag

    # Global: how much variance is explained by a constant ratio model?
    for c1, c2 in pairs:
        if c1 not in flat_amps or c2 not in flat_amps:
            continue
        a1 = flat_amps[c1]
        a2 = flat_amps[c2]
        # R^2 of log(a1) ~ log(a2) + constant
        log_a1 = np.log10(a1 + eps)
        log_a2 = np.log10(a2 + eps)
        # Simple linear regression: log_a1 = alpha * log_a2 + beta
        corr = np.corrcoef(log_a1, log_a2)[0, 1]
        diagnostics["pairs"][f"{c1}_vs_{c2}"]["correlation_log_amplitudes"] = float(corr)
        diagnostics["pairs"][f"{c1}_vs_{c2}"]["r_squared"] = float(corr ** 2)

    return diagnostics


def plot_diagnostics(common_freq: np.ndarray, amps: dict, flat_amps: dict,
                     diagnostics: dict, output_dir: Path, dataset: str):
    """Save diagnostic plots."""
    eps = np.finfo(float).eps
    pairs = [("x", "y"), ("x", "a"), ("y", "a")]

    # Plot 1: Flattened spectra
    fig, ax = plt.subplots(figsize=(12, 5))
    for comp in flat_amps:
        ax.plot(common_freq, flat_amps[comp], label=comp.upper(), linewidth=1.0)
    ax.set_title(f"Flattened Spectra — {dataset}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Flattened Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = output_dir / f"{dataset}_flat_spectra.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(f"  Saved: {p}")

    # Plot 2: Ratio histograms (flattened)
    for c1, c2 in pairs:
        if c1 not in flat_amps or c2 not in flat_amps:
            continue
        ratio = flat_amps[c1] / (flat_amps[c2] + eps)
        log_r = np.log10(ratio + eps)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(log_r, bins=100, color="steelblue", edgecolor="white", linewidth=0.5)
        ax1.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax1.set_title(f"log10(|{c1.upper()}|/|{c2.upper()}|) histogram — {dataset}")
        ax1.set_xlabel("log10(Ratio)")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.3)

        ax2.plot(common_freq, log_r, linewidth=0.8, color="steelblue")
        ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax2.set_title(f"log10(|{c1.upper()}|/|{c2.upper()}|) vs frequency — {dataset}")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("log10(Ratio)")
        ax2.grid(True, alpha=0.3)

        p = output_dir / f"{dataset}_ratio_hist_{c1}_vs_{c2}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        print(f"  Saved: {p}")

    # Plot 3: Rolling median ratio
    for c1, c2 in pairs:
        if c1 not in flat_amps or c2 not in flat_amps:
            continue
        ratio = flat_amps[c1] / (flat_amps[c2] + eps)
        window = max(3, len(common_freq) // 50)
        if window % 2 == 0:
            window += 1

        rolling_med = []
        rolling_freqs = []
        for i in range(0, len(ratio) - window + 1, window // 2):
            chunk = ratio[i:i + window]
            rolling_med.append(float(np.median(chunk)))
            rolling_freqs.append(float(np.mean(common_freq[i:i + window])))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rolling_freqs, rolling_med, linewidth=1.5, color="steelblue")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(f"Rolling Median Ratio |{c1.upper()}|/|{c2.upper()}| — {dataset}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Median Ratio (log scale)")
        ax.grid(True, alpha=0.3)

        p = output_dir / f"{dataset}_rolling_ratio_{c1}_vs_{c2}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        print(f"  Saved: {p}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Component Coupling Diagnostics: {args.dataset} ===")
    print(f"Normalization: {args.normalize}")
    print(f"Flatten ref band: {args.flatten_ref_band} Hz")
    print(f"Baseline match target: {args.baseline_match}")

    print("\n--- Computing average spectra ---")
    results = compute_component_spectra(args)
    if len(results) < 2:
        print("Error: need >= 2 components.", file=sys.stderr)
        return 1

    common_freq = get_common_freq(results)
    print(f"\nCommon frequency grid: {len(common_freq)} bins, "
          f"[{common_freq[0]:.4f}, {common_freq[-1]:.4f}] Hz")

    # Raw amplitudes on common grid
    raw_amps = interp_onto_common(results, common_freq)

    # Apply flattening with baseline matching
    print("\n--- Applying flattening ---")
    flat_results, flattenings = apply_global_baseline_processing_to_results(
        results,
        flatten=True,
        baseline_match=args.baseline_match if args.baseline_match != "none" else None,
        reference_band=tuple(args.flatten_ref_band),
    )

    flat_amps = {}
    for comp, result in flat_results.items():
        flat_amps[comp] = np.interp(common_freq, result.freq_grid, result.avg_amp)

    # Compute numerical diagnostics
    print("\n--- Computing numerical diagnostics ---")
    diagnostics = compute_numerical_diagnostics(common_freq, raw_amps, flat_amps, args)

    # Print results
    print("\n" + "=" * 60)
    print("NUMERICAL DIAGNOSTICS (FLATTENED SPECTRA)")
    print("=" * 60)

    for pair_name, pd in diagnostics["pairs"].items():
        print(f"\n--- {pair_name} ---")
        print(f"  Mean ratio:          {pd['mean_ratio']:.6g}")
        print(f"  Median ratio:        {pd['median_ratio']:.6g}")
        print(f"  Std of ratio:        {pd['std_ratio']:.6g}")
        print(f"  Mean log10(ratio):   {pd['mean_log10_ratio']:.6g}")
        print(f"  Std log10(ratio):    {pd['std_log10_ratio']:.6g}")
        print(f"  Fraction within 2x:  {pd['fraction_within_2x']:.4f}")
        print(f"  Fraction within 10x: {pd['fraction_within_10x']:.4f}")
        print(f"  Correlation(log amp):{pd['correlation_log_amplitudes']:.6g}")
        print(f"  R-squared:           {pd['r_squared']:.6g}")

        if "most_stable_region" in pd:
            sr = pd["most_stable_region"]
            print(f"  MOST STABLE REGION:")
            print(f"    Center freq: {sr['center_freq']:.4f} Hz")
            print(f"    Median ratio: {sr['median_ratio']:.6g}")
            print(f"    log10 ratio std: {sr['log10_ratio_std']:.6g}")

        if "least_stable_region" in pd:
            wr = pd["least_stable_region"]
            print(f"  LEAST STABLE REGION:")
            print(f"    Center freq: {wr['center_freq']:.4f} Hz")
            print(f"    Median ratio: {wr['median_ratio']:.6g}")
            print(f"    log10 ratio std: {wr['log10_ratio_std']:.6g}")

        print(f"  Ratio percentiles:")
        for p_name, p_val in pd["ratio_percentiles"].items():
            print(f"    {p_name}: {p_val:.6g}")

    # Save diagnostics as JSON
    json_path = output_dir / f"{args.dataset}_diagnostics.json"
    with open(json_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nSaved diagnostics JSON: {json_path}")

    # Generate plots
    print("\n--- Generating plots ---")
    plot_diagnostics(common_freq, raw_amps, flat_amps, diagnostics, output_dir, args.dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
