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


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    bond_index: int
    label: str


DATASETS = (
    DatasetConfig(dataset="IMG_0681_rot270", bond_index=1, label="0681 y-like bond1"),
    DatasetConfig(dataset="IMG_0680_rot270", bond_index=2, label="0680 y-like bond2"),
    DatasetConfig(dataset="CDX_10IC", bond_index=6, label="CDX x-like bond6"),
)

TRIADS = (
    Triad(label="8.106 + 8.557 -> 16.663", f1_hz=8.10635126866, f2_hz=8.55719347087, f3_hz=16.66354473953),
    Triad(label="7.956 + 8.742 -> 16.698", f1_hz=7.95607053459, f2_hz=8.74162891723, f3_hz=16.69769945182),
    Triad(label="4.766 + 12.016 -> 16.782", f1_hz=4.76602040683, f2_hz=12.0158503351, f3_hz=16.78187074193),
    Triad(label="3.329 + 13.257 -> 16.586", f1_hz=3.32856015604, f2_hz=13.2568917893, f3_hz=16.58545194534),
    Triad(label="1.902 + 14.701 -> 16.602", f1_hz=1.90186098613, f2_hz=14.7008972301, f3_hz=16.60275821623),
    Triad(label="8.950 + 8.950 -> 17.901", f1_hz=8.95047794335, f2_hz=8.95047794335, f3_hz=17.9009558867),
    Triad(label="8.950 + 8.950 -> 18.361", f1_hz=8.95047794335, f2_hz=8.95047794335, f3_hz=18.3613165321),
)

OUTPUT_DIR = SCRIPT_DIR / "parent166_output"


def plot_matrix(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    labels = [str(row["dataset_label"]) for row in rows]
    triad_labels = [triad.label for triad in TRIADS]
    data = np.full((len(rows), len(triad_labels)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, result in enumerate(row["results"]):
            data[i, j] = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-0.1, vmax=0.2)
    fig.colorbar(im, ax=ax, label="observed - null95 bicoherence")
    ax.set_xticks(np.arange(len(triad_labels)))
    ax.set_xticklabels(triad_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Candidate parent explanations for the 16.6 neighborhood, with 18 controls")
    return writer.save(fig, "parent166_matrix")


def plot_control_compare(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    labels = [str(row["dataset_label"]) for row in rows]
    triad_by_label = []
    for row in rows:
        triad_by_label.append({result["label"]: result for result in row["results"]})
    candidates = ["8.106 + 8.557 -> 16.663", "7.956 + 8.742 -> 16.698", "8.950 + 8.950 -> 18.361"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    x = np.arange(len(labels), dtype=float)
    offsets = np.linspace(-0.25, 0.25, num=len(candidates))
    for offset, label in zip(offsets, candidates):
        values = [
            float(triad_by_label[i][label]["observed_bicoherence"]) - float(triad_by_label[i][label]["null_q95_bicoherence"])
            for i in range(len(labels))
        ]
        ax.bar(x + offset, values, width=0.22, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("observed - null95")
    ax.set_title("16.6 candidate parents versus the 18.361 control relation")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    return writer.save(fig, "parent166_control_compare")


def build_summary(rows: list[dict[str, object]], saved_paths: list[Path]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(f"{row['dataset_label']}:")
        for result in row["results"]:
            diff = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
            status = "above null" if diff > 0 else "not above null"
            lines.append(
                f"  {result['label']}: observed={float(result['observed_bicoherence']):.5f}, "
                f"null95={float(result['null_q95_bicoherence']):.5f}, diff={diff:.5f} | {status}"
            )
        lines.append("")

    lines.append("working read:")
    lines.append("If 16.6 were a child line in the same strong sense that 18 is, at least one 16.6 candidate parent pair should rise clearly above the null across multiple datasets/views.")
    lines.append("The control relation 8.95 + 8.95 -> 18.361 is included so we can check that the method still lights up on a known follower family.")
    lines.append("")
    lines.append("saved_plots:")
    for path in saved_paths:
        lines.append(str(path))
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    rows: list[dict[str, object]] = []
    for dataset_cfg in DATASETS:
        bic_cfg = replace(BIC_CONFIG, dataset=dataset_cfg.dataset, bond_index=dataset_cfg.bond_index)
        _, ds_bic, processed_bic = load_primary_processed(bic_cfg)
        regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg)
        results = []
        for triad in TRIADS:
            result = evaluate_triad(regions_bic, triad, bic_cfg)
            results.append({"label": triad.label, **result})
        rows.append({"dataset_label": dataset_cfg.label, "results": results})

    saved_paths = [
        plot_matrix(writer, rows),
        plot_control_compare(writer, rows),
    ]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(rows, saved_paths))
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
