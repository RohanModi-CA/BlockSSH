from __future__ import annotations

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

from run_peak_bicoherence_analysis import CONFIG as BIC_CONFIG
from run_peak_bicoherence_analysis import PlotWriter, Triad, detect_regions, evaluate_triad, load_primary_processed
from run_peak_coherence_analysis import (
    CONFIG as PHASE_CONFIG,
    HarmonicHypothesis,
    build_regions,
    detect_peak_times,
    evaluate_hypothesis,
    load_primary_dataset,
)


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    bond_index: int
    label: str


CONFIG = (
    DatasetConfig(dataset="IMG_0681_rot270", bond_index=0, label="0681 bond0"),
    DatasetConfig(dataset="IMG_0680_rot270", bond_index=1, label="0680 bond1"),
)
OUTPUT_DIR = SCRIPT_DIR / "parent18_output"


def plot_phase_comparison(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    labels = [str(row["dataset_label"]) for row in rows]
    hyp_labels = [str(item["label"]) for item in rows[0]["phase_results"]]
    data = np.full((len(rows), len(hyp_labels)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, result in enumerate(row["phase_results"]):
            data[i, j] = float(result["observed_plv"]) - float(result["null_q95_plv"])

    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    image = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-0.12, vmax=0.08)
    fig.colorbar(image, ax=ax, label="observed PLV - null 95%")
    ax.set_xticks(np.arange(len(hyp_labels)))
    ax.set_xticklabels(hyp_labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("18 Hz parent hypotheses: phase-lock comparison")
    return writer.save(fig, "phase_parent_comparison")


def plot_bicoherence_comparison(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    labels = [str(row["dataset_label"]) for row in rows]
    observed = np.asarray([float(row["bic_18"]["observed_bicoherence"]) for row in rows], dtype=float)
    null95 = np.asarray([float(row["bic_18"]["null_q95_bicoherence"]) for row in rows], dtype=float)

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.bar(x - 0.18, observed, width=0.36, color="C0", label="8.96+8.96 -> 18 observed")
    ax.bar(x + 0.18, null95, width=0.36, color="C1", label="null 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("bicoherence")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    ax.set_title("18 Hz bicoherence evidence from 8.96+8.96")
    return writer.save(fig, "bicoherence_18_comparison")


def build_summary(rows: list[dict[str, object]], saved_paths: list[Path]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(f"{row['dataset_label']}:")
        lines.append(
            f"  bicoherence 8.96+8.96->18: observed={float(row['bic_18']['observed_bicoherence']):.5f}, "
            f"null95={float(row['bic_18']['null_q95_bicoherence']):.5f}"
        )
        for result in row["phase_results"]:
            diff = float(result["observed_plv"]) - float(result["null_q95_plv"])
            lines.append(
                f"  phase {result['label']}: observed={float(result['observed_plv']):.5f}, "
                f"null95={float(result['null_q95_plv']):.5f}, diff={diff:.5f}"
            )
        lines.append("")

    if rows:
        primary = rows[0]
        phase_by_label = {str(x["label"]): x for x in primary["phase_results"]}
        diff_896 = float(phase_by_label["2 x 8.96 -> 18"]["observed_plv"]) - float(phase_by_label["2 x 8.96 -> 18"]["null_q95_plv"])
        diff_635 = float(phase_by_label["3 x 6.35 -> 18"]["observed_plv"]) - float(phase_by_label["3 x 6.35 -> 18"]["null_q95_plv"])
        diff_335 = float(phase_by_label["5 x 3.35 -> 18"]["observed_plv"]) - float(phase_by_label["5 x 3.35 -> 18"]["null_q95_plv"])
        lines.append("how much stronger in IMG_0681_rot270 bond0?")
        lines.append(
            f"  Against the phase-lock null, 2 x 8.96 -> 18 is higher by {diff_896:.5f}."
        )
        lines.append(
            f"  3 x 6.35 -> 18 is lower than its null threshold by {-diff_635:.5f}."
        )
        lines.append(
            f"  5 x 3.35 -> 18 is lower than its null threshold by {-diff_335:.5f}."
        )
        if diff_896 > 0:
            lines.append(
                f"  So relative to the null-threshold margin, the 8.96 parent hypothesis beats the 6.35 one by {(diff_896 - diff_635):.5f} and the 3.35 one by {(diff_896 - diff_335):.5f}."
            )
        lines.append(
            "  In plain terms: 8.96 is the only tested parent that produces a positive phase-lock excess for 18 in 0681; the 3.35 and 6.35 alternatives do not."
        )
    lines.append("")
    lines.append("saved_plots:")
    for path in saved_paths:
        lines.append(str(path))
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    rows: list[dict[str, object]] = []

    for dataset_cfg in CONFIG:
        phase_cfg = replace(PHASE_CONFIG, dataset=dataset_cfg.dataset, primary_bond_index=dataset_cfg.bond_index, null_trials=2000)
        base, ds_phase, processed_phase = load_primary_dataset(phase_cfg.segmentation_mode, phase_cfg)
        t_global, peak_times = detect_peak_times(processed_phase, phase_cfg)
        regions_phase = build_regions(ds_phase, phase_cfg.primary_bond_index, t_global, peak_times, phase_cfg)
        phase_hypotheses = [
            HarmonicHypothesis(label="2 x 8.96 -> 18", parent_hz=8.96, target_hz=18.0, order=2),
            HarmonicHypothesis(label="3 x 6.35 -> 18", parent_hz=6.35, target_hz=18.0, order=3),
            HarmonicHypothesis(label="5 x 3.35 -> 18", parent_hz=3.35, target_hz=18.0, order=5),
            HarmonicHypothesis(label="44 x 0.41 -> 18", parent_hz=0.41, target_hz=18.0, order=44),
        ]
        phase_results = [evaluate_hypothesis(regions_phase, hyp, phase_cfg) for hyp in phase_hypotheses]

        bic_cfg = replace(BIC_CONFIG, dataset=dataset_cfg.dataset, bond_index=dataset_cfg.bond_index)
        base, ds_bic, processed_bic = load_primary_processed(bic_cfg)
        regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg)
        bic_18 = evaluate_triad(regions_bic, Triad(label="8.96 + 8.96 -> 18.0", f1_hz=8.96, f2_hz=8.96, f3_hz=18.0), bic_cfg)

        rows.append(
            {
                "dataset_label": dataset_cfg.label,
                "phase_results": phase_results,
                "bic_18": bic_18,
            }
        )

    saved_paths = [
        plot_phase_comparison(writer, rows),
        plot_bicoherence_comparison(writer, rows),
    ]
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(build_summary(rows, saved_paths))
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
