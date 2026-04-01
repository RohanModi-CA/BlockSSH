#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from play1 import CONFIG, OUTPUT_DIR
from play6 import TripletSpec, collect_segments, configure_matplotlib, resolve_triplet_bins


@dataclass(frozen=True)
class WindowSpec:
    start_s: float
    stop_s: float
    segment_indices: np.ndarray


@dataclass(frozen=True)
class ControlBins:
    label: str
    i1: int
    i2: int
    i3: int
    f1_sel: float
    f2_sel: float
    f3_sel: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirmatory bicoherence test with fixed protocol, surrogates, and matched control triads.",
    )
    parser.add_argument("--show", action="store_true", help="Show the final figure.")
    parser.add_argument("--segment-len-s", type=float, default=100.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--analysis-window-s",
        type=float,
        default=100.0,
        help="Fixed sliding analysis window over segment midpoints. Default: 100.0 s",
    )
    parser.add_argument(
        "--analysis-step-s",
        type=float,
        default=25.0,
        help="Sliding analysis-window step. Default: 25.0 s",
    )
    parser.add_argument(
        "--min-window-segments",
        type=int,
        default=6,
        help="Minimum segment count in an analysis window. Default: 6",
    )
    parser.add_argument(
        "--scan-mode",
        choices=("contiguous_subsets", "time_windows"),
        default="contiguous_subsets",
        help="Fixed scan family used for the confirmatory max statistic. Default: contiguous_subsets",
    )
    parser.add_argument(
        "--max-window-segments",
        type=int,
        default=0,
        help="Optional maximum contiguous-subset size. Use 0 for no cap. Default: 0",
    )
    parser.add_argument(
        "--snap-bins",
        type=int,
        default=3,
        help="Snap target frequencies to local mean-amplitude peaks within this many bins. Default: 3",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=200,
        help="Number of surrogate realizations for the target triad max statistic. Default: 200",
    )
    parser.add_argument(
        "--max-controls",
        type=int,
        default=200,
        help="Maximum number of matched control triads to evaluate. Default: 200",
    )
    parser.add_argument(
        "--control-min-freq-hz",
        type=float,
        default=0.5,
        help="Minimum frequency allowed in matched controls. Default: 0.5 Hz",
    )
    parser.add_argument(
        "--control-exclusion-bins",
        type=int,
        default=4,
        help="Exclude control bins near target/known triads by this many bins. Default: 4",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility. Default: 0")
    parser.add_argument(
        "--bond-spacing-mode",
        choices=("default", "purecomoving"),
        default="default",
        help="Bond signal representation to analyze. Default: default",
    )
    return parser


def build_windows(
    mids: np.ndarray,
    *,
    analysis_window_s: float,
    analysis_step_s: float,
    min_segments: int,
) -> list[WindowSpec]:
    mids = np.asarray(mids, dtype=float)
    if analysis_window_s <= 0:
        raise ValueError("--analysis-window-s must be > 0")
    if analysis_step_s <= 0:
        raise ValueError("--analysis-step-s must be > 0")
    if mids.size == 0:
        return []

    start_min = float(np.min(mids))
    start_max = float(np.max(mids))
    starts = np.arange(start_min, start_max + 1e-12, analysis_step_s, dtype=float)
    windows: list[WindowSpec] = []
    for start_s in starts:
        stop_s = start_s + analysis_window_s
        indices = np.where((mids >= start_s) & (mids <= stop_s))[0]
        if indices.size < min_segments:
            continue
        windows.append(
            WindowSpec(
                start_s=float(start_s),
                stop_s=float(stop_s),
                segment_indices=np.asarray(indices, dtype=int),
            )
        )
    if not windows:
        windows.append(
            WindowSpec(
                start_s=float(np.min(mids)),
                stop_s=float(np.max(mids)),
                segment_indices=np.arange(mids.size, dtype=int),
            )
        )
    return windows


def build_contiguous_subset_windows(
    records,
    *,
    min_segments: int,
    max_segments: int,
) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    n = len(records)
    upper = n if max_segments <= 0 else min(n, int(max_segments))
    for start in range(n):
        for stop in range(start + min_segments - 1, min(n, start + upper) ):
            indices = np.arange(start, stop + 1, dtype=int)
            windows.append(
                WindowSpec(
                    start_s=float(records[start].mid_s),
                    stop_s=float(records[stop].mid_s),
                    segment_indices=indices,
                )
            )
    return windows


def bicoherence_for_bins(X: np.ndarray, i1: int, i2: int, i3: int) -> float:
    X1 = X[:, i1]
    X2 = X[:, i2]
    X3 = X[:, i3]
    z = X1 * X2 * np.conj(X3)
    denom = np.mean(np.abs(X1 * X2) ** 2) * np.mean(np.abs(X3) ** 2)
    return 0.0 if denom <= 1e-18 else float(np.abs(np.mean(z)) ** 2 / denom)


def surrogate_bicoherence_for_bins(
    X: np.ndarray,
    i1: int,
    i2: int,
    i3: int,
    rng: np.random.Generator,
) -> float:
    if X.shape[0] < 2:
        return 0.0
    perm2 = rng.permutation(X.shape[0])
    perm3 = rng.permutation(X.shape[0])
    X1 = X[:, i1]
    X2 = X[perm2, i2]
    X3 = X[perm3, i3]
    z = X1 * X2 * np.conj(X3)
    denom = np.mean(np.abs(X1 * X2) ** 2) * np.mean(np.abs(X3) ** 2)
    return 0.0 if denom <= 1e-18 else float(np.abs(np.mean(z)) ** 2 / denom)


def score_windows(
    X: np.ndarray,
    windows: list[WindowSpec],
    *,
    i1: int,
    i2: int,
    i3: int,
) -> np.ndarray:
    scores = []
    for window in windows:
        scores.append(bicoherence_for_bins(X[window.segment_indices], i1, i2, i3))
    return np.asarray(scores, dtype=float)


def score_windows_surrogate(
    X: np.ndarray,
    windows: list[WindowSpec],
    *,
    i1: int,
    i2: int,
    i3: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scores = []
    for window in windows:
        scores.append(
            surrogate_bicoherence_for_bins(
                X[window.segment_indices],
                i1,
                i2,
                i3,
                rng,
            )
        )
    return np.asarray(scores, dtype=float)


def build_matched_controls(
    freqs: np.ndarray,
    target_bins,
    known_bins: dict[str, object],
    *,
    min_freq_hz: float,
    exclusion_bins: int,
    max_controls: int,
) -> list[ControlBins]:
    i3 = int(target_bins.i3)
    min_bin = int(np.searchsorted(freqs, float(min_freq_hz), side="left"))
    forbidden: set[int] = set()
    for bins in known_bins.values():
        for idx in (int(bins.i1), int(bins.i2)):
            lo = max(min_bin, idx - exclusion_bins)
            hi = min(freqs.size - 1, idx + exclusion_bins)
            forbidden.update(range(lo, hi + 1))

    controls: list[ControlBins] = []
    for i1 in range(min_bin, i3 - min_bin + 1):
        i2 = i3 - i1
        if i2 < min_bin or i2 >= freqs.size:
            continue
        if i1 > i2:
            continue
        if i1 in forbidden or i2 in forbidden:
            continue
        if abs(i1 - int(target_bins.i1)) <= exclusion_bins and abs(i2 - int(target_bins.i2)) <= exclusion_bins:
            continue
        controls.append(
            ControlBins(
                label=f"{freqs[i1]:.3f}+{freqs[i2]:.3f}",
                i1=int(i1),
                i2=int(i2),
                i3=int(i3),
                f1_sel=float(freqs[i1]),
                f2_sel=float(freqs[i2]),
                f3_sel=float(freqs[i3]),
            )
        )
    if len(controls) <= max_controls:
        return controls
    picks = np.linspace(0, len(controls) - 1, max_controls, dtype=int)
    return [controls[int(idx)] for idx in picks]


def empirical_pvalue(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    return float((1 + np.count_nonzero(null_values >= observed)) / (1 + null_values.size))


def main() -> int:
    args = build_parser().parse_args()
    if args.segment_len_s <= 0:
        raise ValueError("--segment-len-s must be > 0")
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1)")
    if args.snap_bins < 0:
        raise ValueError("--snap-bins must be >= 0")
    if args.n_surrogates < 1:
        raise ValueError("--n-surrogates must be >= 1")
    if args.max_controls < 1:
        raise ValueError("--max-controls must be >= 1")

    configure_matplotlib(args.show)
    rng = np.random.default_rng(args.seed)

    triplets = [
        TripletSpec("Main Nonlinear Sum", 6.34, 12.053),
        TripletSpec("Possible 2*8.97 Peak", 8.97, 8.97),
        TripletSpec("Pendulum Sideband", 18.393, 0.41),
        TripletSpec("Negative Control", 3.74, 14.653),
        TripletSpec("Second Harmonic", 6.34, 6.34),
    ]

    freqs, records, mean_amplitude = collect_segments(
        args.segment_len_s,
        args.overlap,
        bond_spacing_mode=str(args.bond_spacing_mode),
    )
    X = np.vstack([record.spectrum for record in records])
    mids = np.asarray([record.mid_s for record in records], dtype=float)
    bins_by_label = resolve_triplet_bins(freqs, mean_amplitude, triplets, args.snap_bins)
    if args.scan_mode == "time_windows":
        windows = build_windows(
            mids,
            analysis_window_s=float(args.analysis_window_s),
            analysis_step_s=float(args.analysis_step_s),
            min_segments=int(args.min_window_segments),
        )
        scan_desc = (
            f"time_windows len={args.analysis_window_s:.1f}s step={args.analysis_step_s:.1f}s"
        )
    else:
        windows = build_contiguous_subset_windows(
            records,
            min_segments=int(args.min_window_segments),
            max_segments=int(args.max_window_segments),
        )
        if args.max_window_segments > 0:
            scan_desc = (
                f"contiguous_subsets n={args.min_window_segments}..{args.max_window_segments}"
            )
        else:
            scan_desc = f"contiguous_subsets n>={args.min_window_segments}"

    target_bins = bins_by_label["Main Nonlinear Sum"]
    target_window_scores = score_windows(
        X,
        windows,
        i1=int(target_bins.i1),
        i2=int(target_bins.i2),
        i3=int(target_bins.i3),
    )
    observed_max = float(np.max(target_window_scores))
    best_window_index = int(np.argmax(target_window_scores))
    best_window = windows[best_window_index]

    surrogate_maxima = np.empty(args.n_surrogates, dtype=float)
    for idx in range(args.n_surrogates):
        scores = score_windows_surrogate(
            X,
            windows,
            i1=int(target_bins.i1),
            i2=int(target_bins.i2),
            i3=int(target_bins.i3),
            rng=rng,
        )
        surrogate_maxima[idx] = float(np.max(scores))

    controls = build_matched_controls(
        freqs,
        target_bins,
        bins_by_label,
        min_freq_hz=float(args.control_min_freq_hz),
        exclusion_bins=int(args.control_exclusion_bins),
        max_controls=int(args.max_controls),
    )
    if not controls:
        raise ValueError("No matched control triads were available under the requested constraints.")

    control_maxima = np.empty(len(controls), dtype=float)
    control_envelope = np.zeros(len(windows), dtype=float)
    for idx, control in enumerate(controls):
        scores = score_windows(
            X,
            windows,
            i1=int(control.i1),
            i2=int(control.i2),
            i3=int(control.i3),
        )
        control_maxima[idx] = float(np.max(scores))
        control_envelope = np.maximum(control_envelope, scores)

    best_window_metrics = {
        label: bicoherence_for_bins(
            X[best_window.segment_indices],
            int(bins.i1),
            int(bins.i2),
            int(bins.i3),
        )
        for label, bins in bins_by_label.items()
    }

    p_surrogate = empirical_pvalue(observed_max, surrogate_maxima)
    p_control = empirical_pvalue(observed_max, control_maxima)

    print(f"--- Play 7: Confirmatory Bicoherence Test ({CONFIG.dataset}) ---")
    print(
        f"Segments: {len(records)} | segment_len={args.segment_len_s:.1f}s | "
        f"scan={scan_desc} | windows={len(windows)} | mode={args.bond_spacing_mode}"
    )
    print(f"Resolved target bins: {target_bins.f1_sel:.3f} + {target_bins.f2_sel:.3f} -> {target_bins.f3_sel:.3f}")
    print(f"Matched controls evaluated: {len(controls)}")
    print(
        f"Observed max target bicoherence: {observed_max:.4f} "
        f"at {best_window.start_s:.1f}s to {best_window.stop_s:.1f}s "
        f"(n={best_window.segment_indices.size})"
    )
    print(f"Empirical p vs surrogates: {p_surrogate:.4f}")
    print(f"Empirical p vs matched controls: {p_control:.4f}")
    print("Best-window metrics:")
    for label, value in best_window_metrics.items():
        print(f"  {label:<20} {value:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax = axes[0]
    window_centers = np.asarray([0.5 * (window.start_s + window.stop_s) for window in windows], dtype=float)
    ax.plot(window_centers, target_window_scores, color="tab:blue", linewidth=1.8, label="Target triad")
    ax.plot(window_centers, control_envelope, color="tab:red", linewidth=1.2, label="Matched control envelope")
    ax.axvspan(best_window.start_s, best_window.stop_s, color="tab:green", alpha=0.15, label="Max target window")
    ax.set_ylabel("Squared Bicoherence")
    ax.set_title("Fixed Sliding-Window Confirmatory Scan")
    ax.legend()

    ax = axes[1]
    ax.hist(surrogate_maxima, bins=24, alpha=0.65, color="tab:gray", label="Surrogate max distribution")
    ax.hist(control_maxima, bins=24, alpha=0.45, color="tab:red", label="Matched control max distribution")
    ax.axvline(observed_max, color="tab:blue", linewidth=2.0, label=f"Observed max = {observed_max:.4f}")
    ax.set_xlabel("Max Windowed Bicoherence")
    ax.set_ylabel("Count")
    ax.set_title("Null Comparisons")
    ax.legend()

    output_path = OUTPUT_DIR / "play7_confirmatory_bicoherence.png"
    fig.savefig(output_path)
    print(f"\nResults saved to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
