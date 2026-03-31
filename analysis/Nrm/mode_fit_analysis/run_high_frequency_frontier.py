from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_mode_family_scan import PlotWriter


OUTPUT_DIR = Path(__file__).resolve().parent / "high_frequency_frontier_output"
HF_CSV = Path(__file__).resolve().parent / "high_frequency_catalog_output" / "high_frequency_catalog.csv"
REFINED_CSV = Path(__file__).resolve().parent / "catalog_refined_output" / "peak_catalog_refined.csv"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_rows():
    hf_rows = {float(row["peak_hz"]): row for row in load_csv(HF_CSV)}
    ref_rows = {float(row["peak_hz"]): row for row in load_csv(REFINED_CSV)}
    peaks = sorted(set(hf_rows) & set(ref_rows))
    rows = []
    for peak in peaks:
        rows.append(
            {
                "peak_hz": peak,
                "high_frequency_score": float(hf_rows[peak]["high_frequency_score"]),
                "best_view": hf_rows[peak]["best_view"],
                "weighted_child_penalty": float(ref_rows[peak]["weighted_child_penalty"]),
                "parent_support_geomean": float(ref_rows[peak]["parent_support_geomean"]),
                "relation": ref_rows[peak]["best_relation"],
            }
        )
    return rows


def plot_frontier(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    x = np.asarray([float(row["high_frequency_score"]) for row in rows], dtype=float)
    y = np.asarray([float(row["weighted_child_penalty"]) for row in rows], dtype=float)
    f = np.asarray([float(row["peak_hz"]) for row in rows], dtype=float)
    sc = ax.scatter(x, y, c=f, cmap="turbo", s=55, edgecolors="black", linewidths=0.35)
    fig.colorbar(sc, ax=ax, label="peak frequency (Hz)")
    ax.set_xlabel("high-frequency mode-likeness")
    ax.set_ylabel("specific-parent penalty")
    ax.set_title("10-20 Hz frontier: mode-like recurrence vs parent-specificity")
    ax.grid(alpha=0.25)
    for peak in (11.3332775353, 12.0158503351, 16.6009768024, 18.3613165321):
        row = next(item for item in rows if abs(float(item["peak_hz"]) - peak) < 1e-12)
        ax.annotate(f"{peak:.3f}", (float(row["high_frequency_score"]), float(row["weighted_child_penalty"])), xytext=(6, 6), textcoords="offset points", fontsize=8)
    return writer.save(fig, "frontier")


def build_summary(rows: list[dict[str, object]], plot_paths: list[Path]) -> str:
    rows_sorted = sorted(rows, key=lambda row: (float(row["high_frequency_score"]), -float(row["weighted_child_penalty"])), reverse=True)
    lines = [
        "idea:",
        "This frontier separates high-frequency recurrence from parent-specificity. The peaks we most want to call fundamental-like should sit to the right and low; follower-like peaks should move upward because their child explanation is specific and supported.",
        "",
        "rows sorted by high-frequency score:",
    ]
    for row in rows_sorted:
        lines.append(
            f"{float(row['peak_hz']):.3f} Hz | hf_score={float(row['high_frequency_score']):.3f} | "
            f"parent_penalty={float(row['weighted_child_penalty']):.5f} | parent_support={float(row['parent_support_geomean']):.3f} | "
            f"best_view={row['best_view']} | relation={row['relation']}"
        )
    lines.append("")
    lines.append("reference comparison:")
    for peak in (11.3332775353, 12.0158503351, 16.6009768024, 18.3613165321):
        row = next(item for item in rows if abs(float(item["peak_hz"]) - peak) < 1e-12)
        lines.append(
            f"{peak:.3f} Hz | hf_score={float(row['high_frequency_score']):.3f} | "
            f"parent_penalty={float(row['weighted_child_penalty']):.5f} | relation={row['relation']}"
        )
    lines.append("")
    lines.append("saved_plots:")
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    rows = merge_rows()
    plot_paths = [plot_frontier(writer, rows)]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(rows, plot_paths))
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
