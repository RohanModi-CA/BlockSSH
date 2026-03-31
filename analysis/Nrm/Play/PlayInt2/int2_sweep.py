#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from int2 import (
    OUTPUT_DIR,
    Peak,
    average_welch_spectrum,
    build_second_order_predictions,
    choose_generators,
    classify_matches,
    configure_matplotlib,
    detect_observed_peaks,
    flattened_welch_spectrum,
    robust_log_stats,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep second-order match tolerance and compare raw vs flattened k_eff stability.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--bond-spacing-mode", choices=("default", "comoving"), default="comoving")
    parser.add_argument("--peak-prominence", type=float, default=0.05)
    parser.add_argument("--merge-hz", type=float, default=0.12)
    parser.add_argument("--ambiguity-gap-hz", type=float, default=0.05)
    parser.add_argument("--min-freq-hz", type=float, default=0.2)
    parser.add_argument("--max-freq-hz", type=float, default=20.0)
    parser.add_argument("--top-n-generators", type=int, default=5)
    parser.add_argument("--min-child-prominence", type=float, default=0.08)
    parser.add_argument("--tol-values", type=float, nargs="+", default=(0.05, 0.06, 0.08, 0.10, 0.12, 0.14))
    return parser


def save_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_sweep(rows: list[dict[str, float]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tol = np.array([row["match_tol_hz"] for row in rows], dtype=float)
    raw_std = np.array([row["raw_log10_std"] for row in rows], dtype=float)
    flat_std = np.array([row["flat_log10_std"] for row in rows], dtype=float)
    raw_mad = np.array([row["raw_log10_mad"] for row in rows], dtype=float)
    flat_mad = np.array([row["flat_log10_mad"] for row in rows], dtype=float)
    count = np.array([row["accepted_count"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    ax = axes[0]
    ax.plot(tol, raw_std, "o-", color="#777777", label="raw std")
    ax.plot(tol, flat_std, "o-", color="#d95f02", label="flattened std")
    ax.plot(tol, raw_mad, "s--", color="#777777", alpha=0.8, label="raw MAD")
    ax.plot(tol, flat_mad, "s--", color="#d95f02", alpha=0.8, label="flattened MAD")
    ax.set_xlabel("Match tolerance (Hz)")
    ax.set_ylabel("log10(k_eff) spread")
    ax.set_title("k_eff Stability vs Match Tolerance")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[1]
    ax.plot(tol, count, "o-", color="#1b9e77")
    ax.set_xlabel("Match tolerance (Hz)")
    ax.set_ylabel("Accepted child count")
    ax.set_title("Accepted Second-Order Matches")
    ax.grid(alpha=0.25)

    path = OUTPUT_DIR / "quadratic_keff_tolerance_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)

    raw_freqs, raw_amp = average_welch_spectrum(
        bond_spacing_mode=str(args.bond_spacing_mode),
        welch_len_s=float(args.welch_len_s),
        welch_overlap=float(args.welch_overlap),
    )
    flat_freqs, flat_amp = flattened_welch_spectrum(bond_spacing_mode=str(args.bond_spacing_mode))

    flat_peaks0 = detect_observed_peaks(
        flat_freqs,
        flat_amp,
        peak_prominence=float(args.peak_prominence),
        merge_hz=float(args.merge_hz),
        min_freq_hz=float(args.min_freq_hz),
        max_freq_hz=float(args.max_freq_hz),
    )
    flat_peaks = [Peak(p.freq_hz, p.amplitude, p.prominence, p.generator_rank) for p in flat_peaks0]
    generators = choose_generators(flat_peaks, int(args.top_n_generators))
    generators = [Peak(p.freq_hz, p.amplitude, p.prominence, i) for i, p in enumerate(sorted(generators, key=lambda peak: peak.amplitude, reverse=True), start=1)]
    predictions = build_second_order_predictions(generators, float(args.max_freq_hz))

    rows: list[dict[str, float]] = []
    for tol in args.tol_values:
        matches = classify_matches(
            flat_peaks=flat_peaks,
            generators=generators,
            predictions=predictions,
            raw_freqs=raw_freqs,
            raw_amp=raw_amp,
            flat_freqs=flat_freqs,
            flat_amp=flat_amp,
            match_tol_hz=float(tol),
            ambiguity_gap_hz=float(args.ambiguity_gap_hz),
            min_child_prominence=float(args.min_child_prominence),
        )
        if len(matches) < 2:
            continue
        raw_vals = np.array([row.raw_keff for row in matches], dtype=float)
        flat_vals = np.array([row.flat_keff for row in matches], dtype=float)
        _, raw_mad, raw_std = robust_log_stats(raw_vals)
        _, flat_mad, flat_std = robust_log_stats(flat_vals)
        rows.append(
            {
                "match_tol_hz": float(tol),
                "accepted_count": float(len(matches)),
                "raw_log10_mad": float(raw_mad),
                "raw_log10_std": float(raw_std),
                "flat_log10_mad": float(flat_mad),
                "flat_log10_std": float(flat_std),
            }
        )

    if not rows:
        raise ValueError("No sweep points produced at least two accepted matches.")

    csv_path = save_csv(OUTPUT_DIR / "quadratic_keff_tolerance_sweep.csv", rows)
    fig_path = plot_sweep(rows)
    print(f"Saved sweep figure to {fig_path}")
    print(f"Saved sweep csv to {csv_path}")
    for row in rows:
        print(
            f"tol={row['match_tol_hz']:.3f} | n={int(row['accepted_count'])} | "
            f"raw std={row['raw_log10_std']:.3f}, flat std={row['flat_log10_std']:.3f}"
        )
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
