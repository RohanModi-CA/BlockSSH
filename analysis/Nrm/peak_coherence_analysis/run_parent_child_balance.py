from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_peak_bicoherence_analysis import PlotWriter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "balance_output"
INPUT_CSV = SCRIPT_DIR / "calibrated_taxonomy_output" / "calibrated_taxonomy.csv"


def classify(row: dict[str, float]) -> str:
    incoming = float(row["incoming_best"]) if np.isfinite(float(row["incoming_best"])) else -np.inf
    outgoing = float(row["outgoing_best"]) if np.isfinite(float(row["outgoing_best"])) else -np.inf
    outgoing_count = int(row["outgoing_count"])
    score = float(row["mode_score"])

    if outgoing_count >= 1 and outgoing > 0.02 and (not np.isfinite(incoming) or incoming < 0.08):
        return "parent-like"
    if incoming > 0.10 and outgoing_count == 0:
        return "child-like"
    if incoming > 0.06 and outgoing_count >= 1:
        return "mixed"
    if score > 2.3:
        return "parent-like"
    return "unclear"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    rows = []
    with INPUT_CSV.open() as f:
        for row in csv.DictReader(f):
            for key in ("repr_hz", "repr_amp", "persistence", "region_mean_amp", "amp_rank", "amp_0680", "reproducible", "incoming_best", "outgoing_best", "mode_score"):
                row[key] = float(row[key]) if row[key] not in ("", "nan") else np.nan
            row["outgoing_count"] = int(row["outgoing_count"])
            row["class"] = classify(row)
            rows.append(row)

    class_order = ["parent-like", "mixed", "child-like", "unclear"]
    colors = {"parent-like": "C2", "mixed": "C0", "child-like": "C1", "unclear": "0.5"}

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for cls in class_order:
        sub = [row for row in rows if row["class"] == cls]
        if not sub:
            continue
        x = np.asarray([float(row["incoming_best"]) if np.isfinite(float(row["incoming_best"])) else -0.02 for row in sub], dtype=float)
        y = np.asarray([float(row["outgoing_best"]) if np.isfinite(float(row["outgoing_best"])) else -0.02 for row in sub], dtype=float)
        s = np.asarray([35 + 220 * float(row["repr_amp"]) for row in sub], dtype=float)
        ax.scatter(x, y, s=s, color=colors[cls], alpha=0.8, edgecolor="black", linewidth=0.4, label=cls)
        for xi, yi, row in zip(x, y, sub):
            ax.text(xi, yi, str(row["family_label"]), fontsize=8, ha="left", va="bottom")
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.02, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("best incoming evidence")
    ax.set_ylabel("best outgoing evidence")
    ax.set_title("Parent / child balance from calibrated taxonomy")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    scatter_path = writer.save(fig, "parent_child_balance")

    summary_lines = []
    for cls in class_order:
        summary_lines.append(cls + ":")
        for row in rows:
            if row["class"] != cls:
                continue
            summary_lines.append(
                f"  {row['family_label']} | {row['repr_hz']:.3f} Hz | amp={row['repr_amp']:.4f} | "
                f"incoming={row['incoming_best']:.5f} | outgoing={row['outgoing_best']:.5f} | "
                f"outgoing_count={row['outgoing_count']} | mode_score={row['mode_score']:.4f}"
            )
        summary_lines.append("")
    summary_lines.append("saved_files:")
    summary_lines.append(str(scatter_path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"[saved] summary.txt")


if __name__ == "__main__":
    main()
