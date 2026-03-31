from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_mode_family_scan import PlotWriter


@dataclass(frozen=True)
class Config:
    cluster_gap_hz: float = 0.7


CONFIG = Config()
OUTPUT_DIR = Path(__file__).resolve().parent / "high_frequency_cluster_output"
HF_CSV = Path(__file__).resolve().parent / "high_frequency_catalog_output" / "high_frequency_catalog.csv"
REFINED_CSV = Path(__file__).resolve().parent / "catalog_refined_output" / "peak_catalog_refined.csv"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_peak_rows() -> list[dict[str, object]]:
    hf = {float(row["peak_hz"]): row for row in load_csv(HF_CSV)}
    refined = {float(row["peak_hz"]): row for row in load_csv(REFINED_CSV)}
    peaks = sorted(hf)
    rows = []
    for peak in peaks:
        rows.append(
            {
                "peak_hz": peak,
                "hf_score": float(hf[peak]["high_frequency_score"]),
                "parent_penalty": float(refined[peak]["weighted_child_penalty"]),
                "best_view": hf[peak]["best_view"],
            }
        )
    return rows


def cluster_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = sorted(rows, key=lambda row: float(row["peak_hz"]))
    clusters: list[list[dict[str, object]]] = []
    current = [rows[0]]
    for row in rows[1:]:
        if float(row["peak_hz"]) - float(current[-1]["peak_hz"]) <= CONFIG.cluster_gap_hz:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    clusters.append(current)

    out = []
    for cluster in clusters:
        peaks = [float(row["peak_hz"]) for row in cluster]
        out.append(
            {
                "label": ", ".join(f"{peak:.3f}" for peak in peaks),
                "center_hz": float(np.mean(peaks)),
                "members": len(peaks),
                "max_hf_score": float(np.max([float(row["hf_score"]) for row in cluster])),
                "mean_hf_score": float(np.mean([float(row["hf_score"]) for row in cluster])),
                "min_parent_penalty": float(np.min([float(row["parent_penalty"]) for row in cluster])),
                "mean_parent_penalty": float(np.mean([float(row["parent_penalty"]) for row in cluster])),
            }
        )
    out.sort(key=lambda row: float(row["max_hf_score"]), reverse=True)
    return out


def plot_clusters(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    x = np.asarray([float(row["max_hf_score"]) for row in rows], dtype=float)
    y = np.asarray([float(row["min_parent_penalty"]) for row in rows], dtype=float)
    s = np.asarray([80 + 25 * int(row["members"]) for row in rows], dtype=float)
    ax.scatter(x, y, s=s, color="tab:blue", edgecolors="black", linewidths=0.35)
    for row in rows:
        ax.annotate(row["label"], (float(row["max_hf_score"]), float(row["min_parent_penalty"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("cluster max high-frequency score")
    ax.set_ylabel("cluster min parent penalty")
    ax.set_title("High-frequency clusters: recurrence versus child-specificity")
    ax.grid(alpha=0.25)
    return writer.save(fig, "cluster_frontier")


def build_summary(rows: list[dict[str, object]], plot_paths: list[Path]) -> str:
    lines = [
        "idea:",
        "Nearby listed peaks may belong to one physical family. This view clusters nearby high-frequency peaks before comparing recurrence against parent-specificity.",
        "",
        "clusters:",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} | center={float(row['center_hz']):.3f} Hz | members={int(row['members'])} | "
            f"max_hf={float(row['max_hf_score']):.3f} | mean_hf={float(row['mean_hf_score']):.3f} | "
            f"min_parent_penalty={float(row['min_parent_penalty']):.5f}"
        )
    lines.append("")
    lines.append("saved_plots:")
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    clusters = cluster_rows(build_peak_rows())
    plot_paths = [plot_clusters(writer, clusters)]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(clusters, plot_paths))
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
