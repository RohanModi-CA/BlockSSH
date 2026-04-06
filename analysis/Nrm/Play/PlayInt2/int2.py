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

try:
    from int1 import BOND_IDS, COMPONENT, DATASET, detect_observed_peaks
except ModuleNotFoundError:
    from PlayInt2.int1 import BOND_IDS, COMPONENT, DATASET, detect_observed_peaks


@dataclass(frozen=True)
class Peak:
    freq_hz: float
    amplitude: float
    prominence: float
    rank: int


@dataclass(frozen=True)
class Prediction:
    relation: str
    source_a_hz: float
    source_b_hz: float
    predicted_hz: float


@dataclass(frozen=True)
class Match:
    observed_freq_hz: float
    relation: str
    source_a_hz: float
    source_b_hz: float
    predicted_hz: float
    error_hz: float
    second_best_error_hz: float
    raw_child_amp: float
    raw_parent_product: float
    raw_keff: float
    flat_child_amp: float
    flat_parent_product: float
    flat_keff: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Investigate raw vs flattened effective quadratic coefficients on matched second-order peaks.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--bond-spacing-mode", choices=("default", "purecomoving"), default="purecomoving")
    parser.add_argument("--peak-prominence", type=float, default=0.05)
    parser.add_argument("--merge-hz", type=float, default=0.12)
    parser.add_argument("--match-tol-hz", type=float, default=0.12)
    parser.add_argument("--ambiguity-gap-hz", type=float, default=0.05)
    parser.add_argument("--min-freq-hz", type=float, default=0.2)
    parser.add_argument("--max-freq-hz", type=float, default=20.0)
    parser.add_argument("--top-n-generators", type=int, default=5)
    parser.add_argument("--min-child-prominence", type=float, default=0.08)
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


def nearest_amplitude(freqs: np.ndarray, amplitude: np.ndarray, target_hz: float) -> float:
    return float(np.interp(target_hz, freqs, amplitude))


def choose_generators(flat_peaks: list[Peak], top_n: int) -> list[Peak]:
    ranked = sorted(flat_peaks, key=lambda peak: peak.amplitude, reverse=True)
    return ranked[: min(top_n, len(ranked))]


def build_second_order_predictions(generators: list[Peak], max_freq_hz: float) -> list[Prediction]:
    predictions: list[Prediction] = []
    for idx, peak_a in enumerate(generators):
        double_freq = 2.0 * peak_a.freq_hz
        if double_freq <= max_freq_hz:
            predictions.append(Prediction("2x", peak_a.freq_hz, peak_a.freq_hz, double_freq))
        for peak_b in generators[idx + 1 :]:
            sum_freq = peak_a.freq_hz + peak_b.freq_hz
            if sum_freq <= max_freq_hz:
                predictions.append(Prediction("sum", peak_a.freq_hz, peak_b.freq_hz, sum_freq))
            diff_freq = abs(peak_a.freq_hz - peak_b.freq_hz)
            if diff_freq > 0:
                predictions.append(Prediction("diff", peak_a.freq_hz, peak_b.freq_hz, diff_freq))
    return predictions


def classify_matches(
    *,
    flat_peaks: list[Peak],
    generators: list[Peak],
    predictions: list[Prediction],
    raw_freqs: np.ndarray,
    raw_amp: np.ndarray,
    flat_freqs: np.ndarray,
    flat_amp: np.ndarray,
    match_tol_hz: float,
    ambiguity_gap_hz: float,
    min_child_prominence: float,
) -> list[Match]:
    generator_freqs = {peak.freq_hz for peak in generators}
    matches: list[Match] = []
    for peak in flat_peaks:
        if peak.freq_hz in generator_freqs:
            continue
        if peak.prominence < min_child_prominence:
            continue
        scored: list[tuple[float, Prediction]] = []
        for prediction in predictions:
            error = abs(peak.freq_hz - prediction.predicted_hz)
            if error <= match_tol_hz:
                scored.append((error, prediction))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0])
        best_error, best = scored[0]
        second_best_error = scored[1][0] if len(scored) > 1 else np.inf
        if second_best_error - best_error < ambiguity_gap_hz:
            continue

        raw_child_amp = nearest_amplitude(raw_freqs, raw_amp, peak.freq_hz)
        flat_child_amp = nearest_amplitude(flat_freqs, flat_amp, peak.freq_hz)
        raw_a = nearest_amplitude(raw_freqs, raw_amp, best.source_a_hz)
        raw_b = nearest_amplitude(raw_freqs, raw_amp, best.source_b_hz)
        flat_a = nearest_amplitude(flat_freqs, flat_amp, best.source_a_hz)
        flat_b = nearest_amplitude(flat_freqs, flat_amp, best.source_b_hz)
        raw_parent_product = raw_a * raw_b
        flat_parent_product = flat_a * flat_b
        if raw_parent_product <= 0 or flat_parent_product <= 0:
            continue
        matches.append(
            Match(
                observed_freq_hz=peak.freq_hz,
                relation=best.relation,
                source_a_hz=best.source_a_hz,
                source_b_hz=best.source_b_hz,
                predicted_hz=best.predicted_hz,
                error_hz=float(best_error),
                second_best_error_hz=float(second_best_error),
                raw_child_amp=float(raw_child_amp),
                raw_parent_product=float(raw_parent_product),
                raw_keff=float(raw_child_amp / raw_parent_product),
                flat_child_amp=float(flat_child_amp),
                flat_parent_product=float(flat_parent_product),
                flat_keff=float(flat_child_amp / flat_parent_product),
            )
        )
    return sorted(matches, key=lambda row: row.observed_freq_hz)


def robust_log_stats(values: np.ndarray) -> tuple[float, float, float]:
    logs = np.log10(values)
    median = float(np.median(logs))
    mad = float(np.median(np.abs(logs - median)))
    std = float(np.std(logs))
    return median, mad, std


def write_matches_csv(path: Path, matches: list[Match]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "observed_freq_hz",
                "relation",
                "source_a_hz",
                "source_b_hz",
                "predicted_hz",
                "error_hz",
                "second_best_error_hz",
                "raw_child_amp",
                "raw_parent_product",
                "raw_keff",
                "flat_child_amp",
                "flat_parent_product",
                "flat_keff",
            ]
        )
        for row in matches:
            writer.writerow(
                [
                    f"{row.observed_freq_hz:.6f}",
                    row.relation,
                    f"{row.source_a_hz:.6f}",
                    f"{row.source_b_hz:.6f}",
                    f"{row.predicted_hz:.6f}",
                    f"{row.error_hz:.6f}",
                    f"{row.second_best_error_hz:.6f}" if np.isfinite(row.second_best_error_hz) else "",
                    f"{row.raw_child_amp:.6f}",
                    f"{row.raw_parent_product:.6f}",
                    f"{row.raw_keff:.6f}",
                    f"{row.flat_child_amp:.6f}",
                    f"{row.flat_parent_product:.6f}",
                    f"{row.flat_keff:.6f}",
                ]
            )
    return path


def plot_results(
    *,
    raw_freqs: np.ndarray,
    raw_amp: np.ndarray,
    flat_freqs: np.ndarray,
    flat_amp: np.ndarray,
    generators: list[Peak],
    matches: list[Match],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)

    ax = axes[0]
    raw_mask = (raw_freqs >= 0.0) & (raw_freqs <= 20.0)
    flat_mask = (flat_freqs >= 0.0) & (flat_freqs <= 20.0)
    ax.plot(raw_freqs[raw_mask], raw_amp[raw_mask], color="#777777", lw=1.0, label="raw")
    ax.plot(flat_freqs[flat_mask], flat_amp[flat_mask], color="#111111", lw=1.1, label="flattened")
    for peak in generators:
        ax.axvline(peak.freq_hz, color="#1b9e77", alpha=0.5, lw=1.0)
        ax.text(peak.freq_hz, peak.amplitude, f"G{peak.rank} {peak.freq_hz:.2f}", color="#1b9e77", fontsize=7, rotation=60, ha="left", va="bottom")
    for row in matches:
        ax.axvline(row.observed_freq_hz, color="#d95f02", alpha=0.2, lw=0.8)
    ax.set_yscale("log")
    ax.set_title("Raw vs Flattened Spectra With Chosen Generators and Accepted Second-Order Children")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    raw_prod = np.array([row.raw_parent_product for row in matches], dtype=float)
    raw_child = np.array([row.raw_child_amp for row in matches], dtype=float)
    flat_prod = np.array([row.flat_parent_product for row in matches], dtype=float)
    flat_child = np.array([row.flat_child_amp for row in matches], dtype=float)
    ax.scatter(raw_prod, raw_child, color="#777777", s=40, label="raw")
    ax.scatter(flat_prod, flat_child, color="#d95f02", s=40, label="flattened")
    for label, x, y, color in (
        ("raw", raw_prod, raw_child, "#777777"),
        ("flattened", flat_prod, flat_child, "#d95f02"),
    ):
        beta = 10.0 ** np.median(np.log10(y / x))
        xline = np.array([np.min(x) * 0.8, np.max(x) * 1.2], dtype=float)
        ax.plot(xline, beta * xline, color=color, lw=1.5, alpha=0.85, label=f"{label} median k={beta:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Child Amplitude vs Parent Product")
    ax.set_xlabel("Parent amplitude product")
    ax.set_ylabel("Child amplitude")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2]
    child_hz = np.array([row.observed_freq_hz for row in matches], dtype=float)
    raw_keff = np.array([row.raw_keff for row in matches], dtype=float)
    flat_keff = np.array([row.flat_keff for row in matches], dtype=float)
    ax.scatter(child_hz, raw_keff, color="#777777", s=38, label="raw")
    ax.scatter(child_hz, flat_keff, color="#d95f02", s=38, label="flattened")
    raw_med = 10.0 ** np.median(np.log10(raw_keff))
    flat_med = 10.0 ** np.median(np.log10(flat_keff))
    ax.axhline(raw_med, color="#777777", ls="--", alpha=0.8)
    ax.axhline(flat_med, color="#d95f02", ls="--", alpha=0.8)
    for row in matches:
        label = f"{row.observed_freq_hz:.2f} {row.relation}"
        ax.text(row.observed_freq_hz, row.flat_keff, label, fontsize=6.5, color="#d95f02", ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_title("Effective Quadratic Coefficient by Child Frequency")
    ax.set_xlabel("Child frequency (Hz)")
    ax.set_ylabel("k_eff = A_child / (A_parent1 A_parent2)")
    ax.grid(alpha=0.25)
    ax.legend()

    path = OUTPUT_DIR / "quadratic_keff_stability.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(
    *,
    args: argparse.Namespace,
    generators: list[Peak],
    matches: list[Match],
    figure_path: Path,
    csv_path: Path,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_vals = np.array([row.raw_keff for row in matches], dtype=float)
    flat_vals = np.array([row.flat_keff for row in matches], dtype=float)
    raw_med, raw_mad, raw_std = robust_log_stats(raw_vals)
    flat_med, flat_mad, flat_std = robust_log_stats(flat_vals)

    by_relation: dict[str, list[Match]] = {}
    for row in matches:
        by_relation.setdefault(row.relation, []).append(row)

    lines: list[str] = []
    lines.append(f"dataset: {DATASET}")
    lines.append(f"component: {COMPONENT}")
    lines.append(f"bond_spacing_mode: {args.bond_spacing_mode}")
    lines.append(f"welch_len_s: {args.welch_len_s}")
    lines.append(f"welch_overlap: {args.welch_overlap}")
    lines.append(f"peak_prominence: {args.peak_prominence}")
    lines.append(f"merge_hz: {args.merge_hz}")
    lines.append(f"match_tol_hz: {args.match_tol_hz}")
    lines.append(f"ambiguity_gap_hz: {args.ambiguity_gap_hz}")
    lines.append(f"min_child_prominence: {args.min_child_prominence}")
    lines.append("")
    lines.append("chosen generators:")
    for idx, peak in enumerate(generators, start=1):
        lines.append(f"  G{idx}: {peak.freq_hz:.3f} Hz | amplitude={peak.amplitude:.4f} | prominence={peak.prominence:.4f}")
    lines.append("")
    lines.append(f"accepted second-order children: {len(matches)}")
    lines.append("")
    lines.append("global coefficient spread in log10 space:")
    lines.append(f"  raw:       median={raw_med:.3f} | MAD={raw_mad:.3f} | std={raw_std:.3f}")
    lines.append(f"  flattened: median={flat_med:.3f} | MAD={flat_mad:.3f} | std={flat_std:.3f}")
    lines.append("")
    lines.append("per-relation spread in log10 space:")
    for relation in sorted(by_relation):
        subset = by_relation[relation]
        if len(subset) < 2:
            continue
        sub_raw = np.array([row.raw_keff for row in subset], dtype=float)
        sub_flat = np.array([row.flat_keff for row in subset], dtype=float)
        _, raw_rel_mad, raw_rel_std = robust_log_stats(sub_raw)
        _, flat_rel_mad, flat_rel_std = robust_log_stats(sub_flat)
        lines.append(
            f"  {relation}: n={len(subset)} | raw MAD={raw_rel_mad:.3f}, raw std={raw_rel_std:.3f} | "
            f"flat MAD={flat_rel_mad:.3f}, flat std={flat_rel_std:.3f}"
        )
    lines.append("")
    lines.append("accepted matches:")
    for row in matches:
        lhs = f"{row.source_a_hz:.3f}" if row.source_a_hz == row.source_b_hz else f"{row.source_a_hz:.3f},{row.source_b_hz:.3f}"
        lines.append(
            f"  {row.observed_freq_hz:.3f} Hz <- {row.relation}({lhs}) => {row.predicted_hz:.3f} | "
            f"err={row.error_hz:.3f} | raw_k={row.raw_keff:.3f} | flat_k={row.flat_keff:.3f}"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(figure_path))
    lines.append(str(csv_path))

    path = OUTPUT_DIR / "quadratic_keff_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    matches = classify_matches(
        flat_peaks=flat_peaks,
        generators=generators,
        predictions=predictions,
        raw_freqs=raw_freqs,
        raw_amp=raw_amp,
        flat_freqs=flat_freqs,
        flat_amp=flat_amp,
        match_tol_hz=float(args.match_tol_hz),
        ambiguity_gap_hz=float(args.ambiguity_gap_hz),
        min_child_prominence=float(args.min_child_prominence),
    )
    if not matches:
        raise ValueError("No sufficiently unambiguous second-order matches were found.")

    csv_path = write_matches_csv(OUTPUT_DIR / "quadratic_keff_matches.csv", matches)
    figure_path = plot_results(
        raw_freqs=raw_freqs,
        raw_amp=raw_amp,
        flat_freqs=flat_freqs,
        flat_amp=flat_amp,
        generators=generators,
        matches=matches,
    )
    summary_path = write_summary(
        args=args,
        generators=generators,
        matches=matches,
        figure_path=figure_path,
        csv_path=csv_path,
    )

    print(f"Accepted {len(matches)} second-order child peaks.")
    print("Generators:")
    for peak in generators:
        print(f"  G{peak.rank}: {peak.freq_hz:.3f} Hz | amp={peak.amplitude:.4f} | prom={peak.prominence:.4f}")
    raw_vals = np.array([row.raw_keff for row in matches], dtype=float)
    flat_vals = np.array([row.flat_keff for row in matches], dtype=float)
    raw_med, raw_mad, raw_std = robust_log_stats(raw_vals)
    flat_med, flat_mad, flat_std = robust_log_stats(flat_vals)
    print(f"log10(raw k_eff): median={raw_med:.3f}, MAD={raw_mad:.3f}, std={raw_std:.3f}")
    print(f"log10(flat k_eff): median={flat_med:.3f}, MAD={flat_mad:.3f}, std={flat_std:.3f}")
    print(f"Saved figure to {figure_path}")
    print(f"Saved summary to {summary_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
