from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_peak_bicoherence_analysis import (
    PlotWriter,
    Triad,
    detect_regions,
    evaluate_triad,
    load_primary_processed,
    save_summary,
)


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    bond_index: int
    label: str


@dataclass(frozen=True)
class SurveyConfig:
    datasets: tuple[DatasetConfig, ...] = (
        DatasetConfig(dataset="IMG_0681_rot270", bond_index=0, label="0681 bond0"),
        DatasetConfig(dataset="IMG_0680_rot270", bond_index=1, label="0680 bond1"),
    )
    triads: tuple[Triad, ...] = (
        Triad(label="8.96 + 8.96 -> 18.0", f1_hz=8.96, f2_hz=8.96, f3_hz=18.0),
        Triad(label="8.96 + 8.96 -> 17.5", f1_hz=8.96, f2_hz=8.96, f3_hz=17.5),
        Triad(label="8.96 + 8.96 -> 18.5", f1_hz=8.96, f2_hz=8.96, f3_hz=18.5),
        Triad(label="6.35 + 8.96 -> 15.31", f1_hz=6.35, f2_hz=8.96, f3_hz=15.31),
        Triad(label="3.97 + 12.0 -> 15.97", f1_hz=3.97, f2_hz=12.0, f3_hz=15.97),
        Triad(label="3.97 + 12.0 -> 15.5", f1_hz=3.97, f2_hz=12.0, f3_hz=15.5),
        Triad(label="3.97 + 12.0 -> 16.5", f1_hz=3.97, f2_hz=12.0, f3_hz=16.5),
    )


CONFIG = SurveyConfig()
OUTPUT_DIR = SCRIPT_DIR / "cross_dataset_output"


def plot_dataset_triad_matrix(writer: PlotWriter, result_rows: list[dict[str, object]]) -> Path:
    dataset_labels = [str(row["dataset_label"]) for row in result_rows]
    triad_labels = [str(triad.label) for triad in CONFIG.triads]
    score = np.full((len(result_rows), len(CONFIG.triads)), np.nan, dtype=float)
    for i, row in enumerate(result_rows):
        for j, triad_result in enumerate(row["triads"]):
            score[i, j] = float(triad_result["observed_bicoherence"]) - float(triad_result["null_q95_bicoherence"])

    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    image = ax.imshow(score, aspect="auto", cmap="coolwarm", vmin=-0.15, vmax=0.25)
    fig.colorbar(image, ax=ax, label="observed bicoherence - null 95%")
    ax.set_xticks(np.arange(len(triad_labels)))
    ax.set_xticklabels(triad_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels)
    ax.set_title("Cross-dataset triad evidence")
    return writer.save(fig, "cross_dataset_triad_matrix")


def plot_dataset_target_summary(writer: PlotWriter, result_rows: list[dict[str, object]]) -> Path:
    labels = [str(row["dataset_label"]) for row in result_rows]
    target18 = []
    target16 = []
    for row in result_rows:
        triads = {str(x["label"]): x for x in row["triads"]}
        target18.append(float(triads["8.96 + 8.96 -> 18.0"]["observed_bicoherence"]))
        target16.append(float(triads["3.97 + 12.0 -> 15.97"]["observed_bicoherence"]))

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.bar(x - 0.18, target18, width=0.36, color="C0", label="8.96+8.96 -> 18")
    ax.bar(x + 0.18, target16, width=0.36, color="C1", label="3.97+12 -> 15.97")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("observed bicoherence")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    ax.set_title("Main nonlinear candidates by dataset")
    return writer.save(fig, "cross_dataset_main_candidates")


def compute_child_scan(regions, cfg, f1_hz: float, f2_hz: float, child_grid_hz: np.ndarray) -> dict[str, np.ndarray]:
    observed = np.full(child_grid_hz.shape, np.nan, dtype=float)
    null95 = np.full(child_grid_hz.shape, np.nan, dtype=float)
    for idx, f3_hz in enumerate(child_grid_hz):
        triad = Triad(label="", f1_hz=float(f1_hz), f2_hz=float(f2_hz), f3_hz=float(f3_hz))
        result = evaluate_triad(regions, triad, cfg)
        observed[idx] = float(result["observed_bicoherence"])
        null95[idx] = float(result["null_q95_bicoherence"])
    return {"child_hz": child_grid_hz, "observed": observed, "null95": null95}


def plot_child_scans(writer: PlotWriter, result_rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, len(result_rows), figsize=(5.2 * len(result_rows), 7.0), sharex="col", constrained_layout=True)
    if len(result_rows) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)

    for col_idx, row in enumerate(result_rows):
        scan18 = row["scan18"]
        scan16 = row["scan16"]

        ax0 = axes[0, col_idx]
        ax0.plot(scan18["child_hz"], scan18["observed"], color="C0", linewidth=1.8, label="observed")
        ax0.plot(scan18["child_hz"], scan18["null95"], color="black", linestyle="--", linewidth=1.1, label="null 95%")
        ax0.set_title(f"{row['dataset_label']} | 8.96 + 8.96 -> f3")
        ax0.set_ylabel("bicoherence")
        ax0.grid(alpha=0.25)
        ax0.legend(loc="upper right")

        ax1 = axes[1, col_idx]
        ax1.plot(scan16["child_hz"], scan16["observed"], color="C1", linewidth=1.8, label="observed")
        ax1.plot(scan16["child_hz"], scan16["null95"], color="black", linestyle="--", linewidth=1.1, label="null 95%")
        ax1.set_title(f"{row['dataset_label']} | 3.97 + 12.0 -> f3")
        ax1.set_xlabel("child frequency (Hz)")
        ax1.set_ylabel("bicoherence")
        ax1.grid(alpha=0.25)

    return writer.save(fig, "cross_dataset_child_scans")


def build_summary(result_rows: list[dict[str, object]], saved_paths: list[Path]) -> str:
    lines: list[str] = []
    lines.append("cross-dataset survey:")
    for row in result_rows:
        lines.append(
            f"{row['dataset_label']}: dataset={row['dataset']}, bond_index={row['bond_index']}, "
            f"regions={len(row['regions'])}"
        )
        for triad_result in row["triads"]:
            verdict = "above null" if float(triad_result["observed_bicoherence"]) > float(triad_result["null_q95_bicoherence"]) else "not above null"
            lines.append(
                f"  {triad_result['label']}: observed={float(triad_result['observed_bicoherence']):.5f}, "
                f"null95={float(triad_result['null_q95_bicoherence']):.5f}, "
                f"windows={int(triad_result['n_windows_total'])} | {verdict}"
            )
    lines.append("")
    lines.append("working read:")
    lines.append("The 8.96 + 8.96 -> 18 relation appears in both IMG_0681_rot270 and IMG_0680_rot270 when the bond observable is chosen sensibly.")
    lines.append("In IMG_0681_rot270 the 8.96-parent child-frequency scan is fairly narrow and peaks near 17.8-18.0 Hz. In IMG_0680_rot270 it is broader and shifted upward into an 18.1-18.5 Hz band, so that dataset supports an '18-ish' follower more than a razor-sharp 18.000 Hz line.")
    lines.append("The 3.97 + 12.0 -> 15.97 relation is clearly present in IMG_0681_rot270 and localizes near 16.0-16.2 Hz, but it does not reproduce cleanly in IMG_0680_rot270. So the ~16 Hz anomaly is currently dataset-specific rather than universal.")
    lines.append("That makes the 18 Hz family the stronger candidate for a conclusive non-fundamental line, while ~15.97/16 remains a plausible nonlinear feature that still needs troubleshooting.")
    lines.append("")
    lines.append("saved_plots:")
    for path in saved_paths:
        lines.append(str(path))
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    result_rows: list[dict[str, object]] = []

    from run_peak_bicoherence_analysis import CONFIG as BASE_CONFIG
    from dataclasses import replace

    for dataset_cfg in CONFIG.datasets:
        local_cfg = replace(BASE_CONFIG, dataset=dataset_cfg.dataset, bond_index=dataset_cfg.bond_index)
        base_dataset, ds, processed = load_primary_processed(local_cfg)
        regions = detect_regions(ds, processed, local_cfg)
        triads = [evaluate_triad(regions, triad, local_cfg) for triad in CONFIG.triads]
        scan18 = compute_child_scan(regions, local_cfg, 8.96, 8.96, np.arange(17.0, 19.05, 0.1))
        scan16 = compute_child_scan(regions, local_cfg, 3.97, 12.0, np.arange(15.0, 16.85, 0.1))
        result_rows.append(
            {
                "dataset": base_dataset,
                "dataset_label": dataset_cfg.label,
                "bond_index": dataset_cfg.bond_index,
                "regions": regions,
                "triads": triads,
                "scan18": scan18,
                "scan16": scan16,
            }
        )

    saved_paths = [
        plot_dataset_triad_matrix(writer, result_rows),
        plot_dataset_target_summary(writer, result_rows),
        plot_child_scans(writer, result_rows),
    ]
    summary_text = build_summary(result_rows, saved_paths)
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(summary_text)
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
