from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_peak_bicoherence_analysis import (
    CONFIG as BIC_CONFIG,
    PlotWriter,
    Triad,
    detect_regions,
    evaluate_triad,
    load_primary_processed,
)
from run_peak_coherence_analysis import (
    CONFIG as PHASE_CONFIG,
    HarmonicHypothesis,
    build_regions,
    detect_peak_times,
    evaluate_hypothesis,
    load_primary_dataset,
    summarize_line_tracks,
)


@dataclass(frozen=True)
class Config:
    peak_csv: str = "/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/configs/peaks/0681ROT270X.csv"
    dataset: str = "IMG_0681_rot270"
    bond_index: int = 0
    pair_tol_hz: float = 0.08
    harmonic_tol_hz: float = 0.12
    max_harmonic_order: int = 5
    triad_null_trials: int = 150
    phase_null_trials: int = 400
    child_like_threshold: float = 0.03


CONFIG = Config()
OUTPUT_DIR = SCRIPT_DIR / "taxonomy_output"


def load_peak_list(path: str) -> list[float]:
    with Path(path).open() as f:
        rows = list(csv.reader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected a single-row CSV of peaks, got {len(rows)} rows")
    return [float(cell) for cell in rows[0] if str(cell).strip()]


def plot_taxonomy_scatter(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    child = np.asarray([float(row["child_evidence"]) for row in rows], dtype=float)
    outgoing = np.asarray([float(row["outgoing_count"]) for row in rows], dtype=float)
    freq = np.asarray([float(row["peak_hz"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    scatter = ax.scatter(child, outgoing, c=freq, cmap="turbo", s=40, alpha=0.8, edgecolor="black", linewidth=0.3)
    fig.colorbar(scatter, ax=ax, label="peak frequency (Hz)")
    ax.axvline(CONFIG.child_like_threshold, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("incoming child evidence")
    ax.set_ylabel("outgoing significant pair count")
    ax.set_title("0681 first-pass peak taxonomy")
    ax.grid(alpha=0.25)
    return writer.save(fig, "taxonomy_scatter")


def plot_top_child_like(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    ranked = sorted(rows, key=lambda row: float(row["child_evidence"]), reverse=True)[:15]
    labels = [f"{float(row['peak_hz']):.3f}" for row in ranked]
    values = np.asarray([float(row["child_evidence"]) for row in ranked], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, color="C1")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(CONFIG.child_like_threshold, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("incoming child evidence")
    ax.set_title("Top child-like peaks from first-pass screening")
    ax.grid(alpha=0.25, axis="y")
    return writer.save(fig, "top_child_like_peaks")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    peaks = load_peak_list(CONFIG.peak_csv)

    bic_cfg = replace(BIC_CONFIG, dataset=CONFIG.dataset, bond_index=CONFIG.bond_index, null_trials=CONFIG.triad_null_trials)
    _, ds_bic, processed_bic = load_primary_processed(bic_cfg)
    regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg)

    phase_cfg = replace(PHASE_CONFIG, dataset=CONFIG.dataset, primary_bond_index=CONFIG.bond_index, null_trials=CONFIG.phase_null_trials)
    _, ds_phase, processed_phase = load_primary_dataset(phase_cfg.segmentation_mode, phase_cfg)
    t_global, peak_times = detect_peak_times(processed_phase, phase_cfg)
    regions_phase = build_regions(ds_phase, phase_cfg.primary_bond_index, t_global, peak_times, phase_cfg)

    pair_cache: dict[tuple[float, float, float], dict[str, object]] = {}
    harmonic_cache: dict[tuple[float, int, float], dict[str, object]] = {}

    for idx, f3 in enumerate(peaks):
        for i, f1 in enumerate(peaks[:idx]):
            for f2 in peaks[i:idx]:
                if abs((f1 + f2) - f3) <= CONFIG.pair_tol_hz:
                    key = (float(f1), float(f2), float(f3))
                    if key not in pair_cache:
                        pair_cache[key] = evaluate_triad(
                            regions_bic,
                            Triad(label="", f1_hz=float(f1), f2_hz=float(f2), f3_hz=float(f3)),
                            bic_cfg,
                        )

    for idx, f3 in enumerate(peaks):
        for f1 in peaks[:idx]:
            for order in range(2, CONFIG.max_harmonic_order + 1):
                if abs(order * f1 - f3) <= CONFIG.harmonic_tol_hz:
                    key = (float(f1), int(order), float(f3))
                    if key not in harmonic_cache:
                        harmonic_cache[key] = evaluate_hypothesis(
                            regions_phase,
                            HarmonicHypothesis(label="", parent_hz=float(f1), target_hz=float(f3), order=int(order)),
                            phase_cfg,
                        )

    outgoing_count: dict[float, int] = {float(peak): 0 for peak in peaks}
    for (f1, f2, _), result in pair_cache.items():
        excess = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
        if excess > 0:
            outgoing_count[f1] += 1
            outgoing_count[f2] += 1

    rows: list[dict[str, object]] = []
    for peak in peaks:
        line_summary = summarize_line_tracks(regions_phase, float(peak), phase_cfg)
        pair_candidates = []
        for (f1, f2, f3), result in pair_cache.items():
            if abs(f3 - peak) > 1e-12:
                continue
            excess = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
            pair_candidates.append((excess, f1, f2, result))
        pair_candidates.sort(key=lambda item: item[0], reverse=True)

        harmonic_candidates = []
        for (f1, order, f3), result in harmonic_cache.items():
            if abs(f3 - peak) > 1e-12:
                continue
            excess = float(result["observed_plv"]) - float(result["null_q95_plv"])
            harmonic_candidates.append((excess, f1, order, result))
        harmonic_candidates.sort(key=lambda item: item[0], reverse=True)

        best_pair_excess = float(pair_candidates[0][0]) if pair_candidates else np.nan
        best_harmonic_excess = float(harmonic_candidates[0][0]) if harmonic_candidates else np.nan
        child_evidence = float(np.nanmax([best_pair_excess, best_harmonic_excess]))
        if not np.isfinite(child_evidence):
            child_evidence = np.nan

        if pair_candidates and np.isfinite(best_pair_excess) and (best_harmonic_excess <= best_pair_excess or not np.isfinite(best_harmonic_excess)):
            best_relation = f"{pair_candidates[0][1]:.3f} + {pair_candidates[0][2]:.3f}"
        elif harmonic_candidates and np.isfinite(best_harmonic_excess):
            best_relation = f"{harmonic_candidates[0][2]} x {harmonic_candidates[0][1]:.3f}"
        else:
            best_relation = ""

        if np.isfinite(child_evidence) and child_evidence > CONFIG.child_like_threshold:
            tag = "child-like"
        else:
            tag = "unexplained-or-parent-like"

        rows.append(
            {
                "peak_hz": float(peak),
                "mean_freq_hz": float(line_summary["global_mean_freq_hz"]),
                "freq_std_hz": float(line_summary["global_std_freq_hz"]),
                "mean_amp": float(line_summary["global_mean_amp"]),
                "best_pair_excess": best_pair_excess,
                "best_harmonic_excess": best_harmonic_excess,
                "child_evidence": child_evidence,
                "best_relation": best_relation,
                "outgoing_count": int(outgoing_count[float(peak)]),
                "tag": tag,
            }
        )

    rows.sort(key=lambda row: float(row["peak_hz"]))

    csv_path = OUTPUT_DIR / "peak_taxonomy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "peak_hz",
                "mean_freq_hz",
                "freq_std_hz",
                "mean_amp",
                "best_pair_excess",
                "best_harmonic_excess",
                "child_evidence",
                "best_relation",
                "outgoing_count",
                "tag",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    saved_paths = [
        plot_taxonomy_scatter(writer, rows),
        plot_top_child_like(writer, rows),
    ]

    summary_lines = []
    summary_lines.append(f"peak_csv: {CONFIG.peak_csv}")
    summary_lines.append(f"dataset: {CONFIG.dataset}")
    summary_lines.append(f"bond_index: {CONFIG.bond_index}")
    summary_lines.append(f"n_peaks: {len(peaks)}")
    summary_lines.append(f"n_pair_candidates_evaluated: {len(pair_cache)}")
    summary_lines.append(f"n_harmonic_candidates_evaluated: {len(harmonic_cache)}")
    summary_lines.append("")
    summary_lines.append("top child-like peaks:")
    for row in sorted(rows, key=lambda row: float(row["child_evidence"]), reverse=True)[:20]:
        summary_lines.append(
            f"{float(row['peak_hz']):.3f} Hz | child_evidence={float(row['child_evidence']):.5f} | "
            f"best_relation={row['best_relation']} | outgoing_count={int(row['outgoing_count'])} | tag={row['tag']}"
        )
    summary_lines.append("")
    summary_lines.append("lowest child-evidence peaks:")
    valid_low = [row for row in rows if np.isfinite(float(row["child_evidence"]))]
    for row in sorted(valid_low, key=lambda row: float(row["child_evidence"]))[:20]:
        summary_lines.append(
            f"{float(row['peak_hz']):.3f} Hz | child_evidence={float(row['child_evidence']):.5f} | "
            f"best_relation={row['best_relation']} | outgoing_count={int(row['outgoing_count'])} | tag={row['tag']}"
        )
    summary_lines.append("")
    summary_lines.append("notes:")
    summary_lines.append("This is only a first-pass taxonomy. Low child evidence does not yet prove fundamentality; it only means the peak resisted the tested sum-rule and low-order harmonic explanations.")
    summary_lines.append("High child evidence means the peak already has at least one lower-frequency explanation that survives the current null tests.")
    summary_lines.append("")
    summary_lines.append("saved_files:")
    summary_lines.append(str(csv_path))
    for path in saved_paths:
        summary_lines.append(str(path))

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"[saved] {csv_path.name}")
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
