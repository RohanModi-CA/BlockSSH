#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")


REPO_ROOT = add_repo_root_to_path()

from analysis.go.Play.fft_flattening import compute_flattened_component_spectra
from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import compute_welch_spectrum, preprocess_signal


DATASET = "IMG_0681_rot270"
COMPONENT = "x"
BOND_IDS = (0, 1, 2)


@dataclass(frozen=True)
class ObservedPeak:
    freq_hz: float
    amplitude: float
    prominence: float
    generator_rank: int


@dataclass(frozen=True)
class Explanation:
    generator_count: int
    observed_freq_hz: float
    generator_rank: int
    relation: str
    source_a_hz: float
    source_b_hz: float
    predicted_hz: float
    error_hz: float


@dataclass(frozen=True)
class SpectrumAnalysis:
    label: str
    freqs: np.ndarray
    amplitude: np.ndarray
    observed_peaks: list[ObservedPeak]
    explanations: list[Explanation]
    summaries: list[dict[str, float]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrated peak-family ledger from corrected comoving Welch spectra.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--bond-spacing-mode", choices=("default", "purecomoving"), default="purecomoving")
    parser.add_argument("--peak-prominence", type=float, default=0.05)
    parser.add_argument("--merge-hz", type=float, default=0.12)
    parser.add_argument("--match-tol-hz", type=float, default=0.12)
    parser.add_argument("--min-freq-hz", type=float, default=0.2)
    parser.add_argument("--max-freq-hz", type=float, default=20.0)
    parser.add_argument("--top-k-max", type=int, default=8)
    parser.add_argument("--max-harmonic-order", type=int, default=6)
    parser.add_argument(
        "--spectrum-mode",
        choices=("raw", "flattened", "compare"),
        default="compare",
        help="Use raw averaged Welch amplitude, flattened amplitude, or compare both. Default: compare",
    )
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def average_welch_spectrum(*, bond_spacing_mode: str, welch_len_s: float, welch_overlap: float) -> tuple[np.ndarray, np.ndarray]:
    ds = load_bond_signal_dataset(
        dataset=f"{DATASET}_{COMPONENT}",
        bond_spacing_mode=bond_spacing_mode,
        component=COMPONENT,
    )
    amplitudes: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for bond_id in BOND_IDS:
        processed, err = preprocess_signal(ds.frame_times_s, ds.signal_matrix[:, bond_id], longest=False, handlenan=False)
        if processed is None:
            raise ValueError(f"Preprocess failed for bond {bond_id}: {err}")
        welch = compute_welch_spectrum(
            processed.y,
            processed.Fs,
            welch_len_s,
            overlap_fraction=welch_overlap,
        )
        if welch is None:
            raise ValueError(f"Welch failed for bond {bond_id}")
        freqs = np.asarray(welch.freq, dtype=float)
        amplitudes.append(np.asarray(welch.amplitude, dtype=float))
    if freqs is None or not amplitudes:
        raise ValueError("No Welch spectra were produced")
    return freqs, np.mean(np.vstack(amplitudes), axis=0)


def flattened_welch_spectrum(*, bond_spacing_mode: str) -> tuple[np.ndarray, np.ndarray]:
    results = compute_flattened_component_spectra(
        dataset=DATASET,
        bond_spacing_mode=bond_spacing_mode,
        components=(COMPONENT,),
        use_welch=True,
    )
    result = results[COMPONENT]
    return np.asarray(result.freq_hz, dtype=float), np.asarray(result.flattened, dtype=float)


def detect_observed_peaks(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    *,
    peak_prominence: float,
    merge_hz: float,
    min_freq_hz: float,
    max_freq_hz: float,
) -> list[ObservedPeak]:
    peak_idx, props = find_peaks(amplitude, prominence=peak_prominence)
    mask = (freqs[peak_idx] >= min_freq_hz) & (freqs[peak_idx] <= max_freq_hz)
    peak_idx = peak_idx[mask]
    prominences = np.asarray(props["prominences"], dtype=float)[mask]

    order = np.argsort(amplitude[peak_idx])[::-1]
    selected: list[int] = []
    prom_by_idx: dict[int, float] = {}
    for order_idx in order:
        idx = int(peak_idx[order_idx])
        if any(abs(float(freqs[idx]) - float(freqs[other])) <= merge_hz for other in selected):
            continue
        selected.append(idx)
        prom_by_idx[idx] = float(prominences[order_idx])

    selected = sorted(selected, key=lambda idx: float(freqs[idx]))
    peaks = [
        ObservedPeak(
            freq_hz=float(freqs[idx]),
            amplitude=float(amplitude[idx]),
            prominence=float(prom_by_idx[idx]),
            generator_rank=0,
        )
        for idx in selected
    ]
    ranked = sorted(peaks, key=lambda peak: peak.amplitude, reverse=True)
    rank_map = {peak.freq_hz: rank + 1 for rank, peak in enumerate(ranked)}
    return [
        ObservedPeak(
            freq_hz=peak.freq_hz,
            amplitude=peak.amplitude,
            prominence=peak.prominence,
            generator_rank=int(rank_map[peak.freq_hz]),
        )
        for peak in peaks
    ]


def build_predictions(
    generators: list[ObservedPeak],
    *,
    max_freq_hz: float,
    max_harmonic_order: int,
) -> list[tuple[str, float, float, float]]:
    predictions: list[tuple[str, float, float, float]] = []
    for peak in generators:
        for order in range(2, max_harmonic_order + 1):
            freq = float(order * peak.freq_hz)
            if freq <= max_freq_hz:
                predictions.append((f"{order}x", peak.freq_hz, 0.0, freq))

    for i, peak_a in enumerate(generators):
        for peak_b in generators[i:]:
            sum_freq = float(peak_a.freq_hz + peak_b.freq_hz)
            if sum_freq <= max_freq_hz:
                predictions.append(("sum", peak_a.freq_hz, peak_b.freq_hz, sum_freq))
            diff_freq = float(abs(peak_a.freq_hz - peak_b.freq_hz))
            if diff_freq > 0:
                predictions.append(("diff", peak_a.freq_hz, peak_b.freq_hz, diff_freq))
    return predictions


def explain_peaks(
    observed_peaks: list[ObservedPeak],
    *,
    top_k_max: int,
    match_tol_hz: float,
    max_freq_hz: float,
    max_harmonic_order: int,
) -> tuple[list[Explanation], list[dict[str, float]]]:
    ranked = sorted(observed_peaks, key=lambda peak: peak.amplitude, reverse=True)
    summaries: list[dict[str, float]] = []
    explanations: list[Explanation] = []

    for generator_count in range(1, min(top_k_max, len(ranked)) + 1):
        generators = ranked[:generator_count]
        predictions = build_predictions(
            generators,
            max_freq_hz=max_freq_hz,
            max_harmonic_order=max_harmonic_order,
        )
        explained_freqs: set[float] = set()
        explained_amp = 0.0
        for peak in observed_peaks:
            if peak.freq_hz in [generator.freq_hz for generator in generators]:
                explained_freqs.add(peak.freq_hz)
                explained_amp += peak.amplitude
                explanations.append(
                    Explanation(
                        generator_count=generator_count,
                        observed_freq_hz=peak.freq_hz,
                        generator_rank=peak.generator_rank,
                        relation="generator",
                        source_a_hz=peak.freq_hz,
                        source_b_hz=0.0,
                        predicted_hz=peak.freq_hz,
                        error_hz=0.0,
                    )
                )
                continue

            best: tuple[float, tuple[str, float, float, float]] | None = None
            for prediction in predictions:
                error = abs(peak.freq_hz - prediction[3])
                if error <= match_tol_hz and (best is None or error < best[0]):
                    best = (error, prediction)
            if best is None:
                continue
            error, prediction = best
            explained_freqs.add(peak.freq_hz)
            explained_amp += peak.amplitude
            explanations.append(
                Explanation(
                    generator_count=generator_count,
                    observed_freq_hz=peak.freq_hz,
                    generator_rank=peak.generator_rank,
                    relation=prediction[0],
                    source_a_hz=prediction[1],
                    source_b_hz=prediction[2],
                    predicted_hz=prediction[3],
                    error_hz=float(error),
                )
            )

        total_amp = float(sum(peak.amplitude for peak in observed_peaks))
        summaries.append(
            {
                "generator_count": float(generator_count),
                "explained_peak_count": float(len(explained_freqs)),
                "observed_peak_count": float(len(observed_peaks)),
                "explained_fraction": float(len(explained_freqs) / len(observed_peaks)),
                "explained_amplitude_fraction": float(explained_amp / total_amp) if total_amp > 0 else np.nan,
            }
        )
    return explanations, summaries


def save_csv(
    observed_peaks: list[ObservedPeak],
    explanations: list[Explanation],
    summaries: list[dict[str, float]],
    *,
    prefix: str,
) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    peaks_csv = OUTPUT_DIR / f"{prefix}_observed_peaks.csv"
    with peaks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_hz", "amplitude", "prominence", "generator_rank"])
        for peak in sorted(observed_peaks, key=lambda peak: peak.freq_hz):
            writer.writerow([f"{peak.freq_hz:.6f}", f"{peak.amplitude:.6f}", f"{peak.prominence:.6f}", peak.generator_rank])

    expl_csv = OUTPUT_DIR / f"{prefix}_explanations.csv"
    with expl_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generator_count", "observed_freq_hz", "generator_rank", "relation", "source_a_hz", "source_b_hz", "predicted_hz", "error_hz"])
        for row in explanations:
            writer.writerow(
                [
                    row.generator_count,
                    f"{row.observed_freq_hz:.6f}",
                    row.generator_rank,
                    row.relation,
                    f"{row.source_a_hz:.6f}",
                    f"{row.source_b_hz:.6f}",
                    f"{row.predicted_hz:.6f}",
                    f"{row.error_hz:.6f}",
                ]
            )

    summ_csv = OUTPUT_DIR / f"{prefix}_generator_ladder.csv"
    with summ_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    return peaks_csv, expl_csv, summ_csv


def plot_spectrum_and_explanations(analysis: SpectrumAnalysis) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    freqs = analysis.freqs
    amplitude = analysis.amplitude
    observed_peaks = analysis.observed_peaks
    explanations = analysis.explanations
    summaries = analysis.summaries
    ranked = sorted(observed_peaks, key=lambda peak: peak.amplitude, reverse=True)
    first_explained: dict[float, int] = {}
    best_rel: dict[tuple[float, int], str] = {}
    for row in explanations:
        first_explained.setdefault(row.observed_freq_hz, row.generator_count)
        best_rel[(row.observed_freq_hz, row.generator_count)] = row.relation

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    ax = axes[0]
    mask = (freqs >= 0.0) & (freqs <= 20.0)
    ax.plot(freqs[mask], amplitude[mask], color="black", linewidth=1.2)
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
    for peak in sorted(observed_peaks, key=lambda peak: peak.freq_hz):
        stage = first_explained.get(peak.freq_hz)
        color = "#bbbbbb" if stage is None else palette[min(stage - 1, len(palette) - 1)]
        ax.axvline(peak.freq_hz, color=color, alpha=0.55, linewidth=1.0)
        label = f"{peak.freq_hz:.2f}"
        if peak.generator_rank <= 3:
            label += f" G{peak.generator_rank}"
        ax.text(peak.freq_hz, peak.amplitude, label, fontsize=7, rotation=60, ha="left", va="bottom", color=color)
    ax.set_yscale("log")
    ax.set_title(f"{analysis.label}: Averaged Spectrum With Auto Peaks and First Explanation Stage")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ks = np.array([row["generator_count"] for row in summaries], dtype=float)
    explained_frac = np.array([row["explained_fraction"] for row in summaries], dtype=float)
    explained_amp = np.array([row["explained_amplitude_fraction"] for row in summaries], dtype=float)
    ax.plot(ks, explained_frac, "o-", label="Explained peak fraction")
    ax.plot(ks, explained_amp, "s-", label="Explained amplitude fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Top-K generators included")
    ax.set_ylabel("Fraction explained")
    ax.set_title("Generator Ladder Coverage")
    ax.grid(alpha=0.25)
    ax.legend()
    fig_path = OUTPUT_DIR / f"{analysis.label}_spectrum_ladder.png"
    fig.savefig(fig_path)
    plt.close(fig)

    top_k = int(min(3, len(ranked)))
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for peak in observed_peaks:
        relation = next(
            (row.relation for row in explanations if row.generator_count == top_k and row.observed_freq_hz == peak.freq_hz),
            "unexplained",
        )
        color = {
            "generator": "#1b9e77",
            "sum": "#d95f02",
            "diff": "#7570b3",
            "2x": "#e7298a",
            "3x": "#66a61e",
            "4x": "#e6ab02",
            "5x": "#a6761d",
            "6x": "#666666",
            "unexplained": "#bbbbbb",
        }.get(relation, "#444444")
        ax.scatter(peak.freq_hz, peak.amplitude, color=color, s=55)
        ax.text(peak.freq_hz, peak.amplitude, f"{peak.freq_hz:.2f}", fontsize=7, ha="left", va="bottom", color=color)
    ax.plot(freqs[mask], amplitude[mask], color="black", linewidth=0.8, alpha=0.45)
    ax.set_yscale("log")
    ax.set_title(f"{analysis.label}: Top-{top_k} Generator Explanation Map")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    fig2_path = OUTPUT_DIR / f"{analysis.label}_topk_explanations.png"
    fig.savefig(fig2_path)
    plt.close(fig)

    return fig_path, fig2_path


def write_summary(
    analysis: SpectrumAnalysis,
    fig_paths: tuple[Path, Path],
    csv_paths: tuple[Path, Path, Path],
    args: argparse.Namespace,
) -> Path:
    observed_peaks = analysis.observed_peaks
    explanations = analysis.explanations
    summaries = analysis.summaries
    ranked = sorted(observed_peaks, key=lambda peak: peak.amplitude, reverse=True)
    top3 = ranked[:3]
    top3_freqs = {peak.freq_hz for peak in top3}
    top3_expl = [row for row in explanations if row.generator_count == 3]
    unresolved_top3 = [
        peak for peak in observed_peaks
        if peak.freq_hz not in {row.observed_freq_hz for row in top3_expl} and peak.freq_hz not in top3_freqs
    ]

    lines = []
    lines.append(f"dataset: {DATASET}")
    lines.append(f"component: {COMPONENT}")
    lines.append(f"bond_spacing_mode: {args.bond_spacing_mode}")
    lines.append(f"spectrum_label: {analysis.label}")
    lines.append(f"welch_len_s: {args.welch_len_s}")
    lines.append(f"welch_overlap: {args.welch_overlap}")
    lines.append(f"peak_prominence: {args.peak_prominence}")
    lines.append(f"merge_hz: {args.merge_hz}")
    lines.append(f"match_tol_hz: {args.match_tol_hz}")
    lines.append("")
    lines.append("top generators:")
    for peak in top3:
        lines.append(f"  G{peak.generator_rank}: {peak.freq_hz:.3f} Hz | amplitude={peak.amplitude:.4f} | prominence={peak.prominence:.4f}")
    lines.append("")
    lines.append("generator ladder:")
    for row in summaries:
        lines.append(
            f"  K={int(row['generator_count'])}: explained_peaks={int(row['explained_peak_count'])}/{int(row['observed_peak_count'])} "
            f"({row['explained_fraction']:.3f}) | explained_amplitude_fraction={row['explained_amplitude_fraction']:.3f}"
        )
    lines.append("")
    lines.append("top-3 explanation assignments:")
    for row in sorted((r for r in top3_expl if r.relation != 'generator'), key=lambda r: r.observed_freq_hz):
        lhs = f"{row.source_a_hz:.3f}" if row.source_b_hz == 0.0 else f"{row.source_a_hz:.3f},{row.source_b_hz:.3f}"
        lines.append(
            f"  {row.observed_freq_hz:.3f} Hz <- {row.relation}({lhs}) => {row.predicted_hz:.3f} Hz | err={row.error_hz:.3f}"
        )
    lines.append("")
    lines.append("top-3 unresolved peaks:")
    for peak in unresolved_top3[:12]:
        lines.append(f"  {peak.freq_hz:.3f} Hz | amplitude={peak.amplitude:.4f}")
    lines.append("")
    lines.append("saved_files:")
    for path in (*fig_paths, *csv_paths):
        lines.append(str(path))

    summary_path = OUTPUT_DIR / f"{analysis.label}_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def plot_compare_ladders(analyses: list[SpectrumAnalysis]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    ax = axes[0]
    for analysis in analyses:
        mask = (analysis.freqs >= 0.0) & (analysis.freqs <= 20.0)
        ax.plot(analysis.freqs[mask], analysis.amplitude[mask], linewidth=1.2, label=analysis.label)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Raw vs Flattened Comoving Spectra")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for analysis in analyses:
        ks = np.array([row["generator_count"] for row in analysis.summaries], dtype=float)
        explained_amp = np.array([row["explained_amplitude_fraction"] for row in analysis.summaries], dtype=float)
        explained_frac = np.array([row["explained_fraction"] for row in analysis.summaries], dtype=float)
        ax.plot(ks, explained_amp, "o-", label=f"{analysis.label} amp")
        ax.plot(ks, explained_frac, "s--", label=f"{analysis.label} peaks")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Top-K generators included")
    ax.set_ylabel("Fraction explained")
    ax.set_title("Generator Ladder: Raw vs Flattened")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    path = OUTPUT_DIR / "compare_raw_flattened.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)

    requested = ["raw", "flattened"] if args.spectrum_mode == "compare" else [str(args.spectrum_mode)]
    analyses: list[SpectrumAnalysis] = []
    summary_paths: list[Path] = []

    for label in requested:
        if label == "raw":
            freqs, amplitude = average_welch_spectrum(
                bond_spacing_mode=str(args.bond_spacing_mode),
                welch_len_s=float(args.welch_len_s),
                welch_overlap=float(args.welch_overlap),
            )
        else:
            freqs, amplitude = flattened_welch_spectrum(
                bond_spacing_mode=str(args.bond_spacing_mode),
            )

        observed_peaks = detect_observed_peaks(
            freqs,
            amplitude,
            peak_prominence=float(args.peak_prominence),
            merge_hz=float(args.merge_hz),
            min_freq_hz=float(args.min_freq_hz),
            max_freq_hz=float(args.max_freq_hz),
        )
        explanations, summaries = explain_peaks(
            observed_peaks,
            top_k_max=int(args.top_k_max),
            match_tol_hz=float(args.match_tol_hz),
            max_freq_hz=float(args.max_freq_hz),
            max_harmonic_order=int(args.max_harmonic_order),
        )
        analysis = SpectrumAnalysis(
            label=label,
            freqs=freqs,
            amplitude=amplitude,
            observed_peaks=observed_peaks,
            explanations=explanations,
            summaries=summaries,
        )
        analyses.append(analysis)

        csv_paths = save_csv(observed_peaks, explanations, summaries, prefix=label)
        fig_paths = plot_spectrum_and_explanations(analysis)
        summary_paths.append(write_summary(analysis, fig_paths, csv_paths, args))

        ranked = sorted(observed_peaks, key=lambda peak: peak.amplitude, reverse=True)
        print(f"[{label}] detected {len(observed_peaks)} merged peaks in {args.min_freq_hz:.1f}-{args.max_freq_hz:.1f} Hz")
        for peak in ranked[:5]:
            print(f"  G{peak.generator_rank}: {peak.freq_hz:.3f} Hz | amp={peak.amplitude:.4f} | prom={peak.prominence:.4f}")
        for row in summaries:
            print(
                f"  K={int(row['generator_count'])}: explained {int(row['explained_peak_count'])}/{int(row['observed_peak_count'])} "
                f"peaks | amp frac {row['explained_amplitude_fraction']:.3f}"
            )

    compare_path = None
    if len(analyses) >= 2:
        compare_path = plot_compare_ladders(analyses)
        print(f"Saved comparison to {compare_path}")

    print("Saved summaries:")
    for path in summary_paths:
        print(f"  {path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
