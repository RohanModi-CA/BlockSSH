from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_peak_bicoherence_analysis import PlotWriter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "synthesis_output"
BALANCE_CSV = SCRIPT_DIR / "edge_survey_output" / "family_balance.csv"
PROM_CSV = SCRIPT_DIR / "prominence_output" / "family_prominence.csv"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def classify(row: dict[str, float]) -> str:
    incoming = float(row["best_incoming_edge"]) if np.isfinite(float(row["best_incoming_edge"])) else -np.inf
    outgoing = float(row["best_outgoing_edge"]) if np.isfinite(float(row["best_outgoing_edge"])) else -np.inf
    prom1 = float(row["contrast_db_0681"])
    prom0 = float(row["contrast_db_0680"])
    amp = float(row["repr_amp"])

    if incoming > 0.22 and outgoing < 0.02 and prom1 < 8.0:
        return "nonfundamental-leaning"
    if incoming > 0.18 and outgoing < 0.04 and prom1 < 6.5:
        return "nonfundamental-leaning"
    if outgoing > 0.08 and prom1 > 10.0 and (not np.isfinite(incoming) or incoming < 0.10):
        return "fundamental-leaning"
    if outgoing > 0.02 and incoming > 0.08:
        return "mixed-coupled"
    if prom1 > 12.0 and prom0 > 5.0 and amp > 0.05:
        return "fundamental-leaning"
    return "unclear"


def plot_synthesis(writer: PlotWriter, rows: list[dict[str, float]]) -> Path:
    classes = {
        "fundamental-leaning": "C2",
        "mixed-coupled": "C0",
        "nonfundamental-leaning": "C1",
        "unclear": "0.5",
    }
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    for cls, color in classes.items():
        sub = [row for row in rows if row["class"] == cls]
        if not sub:
            continue
        x = np.asarray([float(row["best_incoming_edge"]) if np.isfinite(float(row["best_incoming_edge"])) else -0.03 for row in sub], dtype=float)
        y = np.asarray([float(row["contrast_db_0681"]) for row in sub], dtype=float)
        s = np.asarray([35.0 + 220.0 * float(row["repr_amp"]) for row in sub], dtype=float)
        ax.scatter(x, y, s=s, color=color, edgecolor="black", linewidth=0.4, alpha=0.85, label=cls)
        for xi, yi, row in zip(x, y, sub):
            ax.text(xi, yi, str(row["family_label"]), fontsize=8, ha="left", va="bottom")
    ax.axvline(0.18, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(8.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("best incoming edge score")
    ax.set_ylabel("0681 local prominence (dB)")
    ax.set_title("Family synthesis: incoming edge evidence vs line prominence")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    return writer.save(fig, "family_synthesis")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    balance_rows = load_csv(BALANCE_CSV)
    prom_rows = load_csv(PROM_CSV)
    prom_by_label = {row["family_label"]: row for row in prom_rows}

    rows = []
    for brow in balance_rows:
        prow = prom_by_label[brow["family_label"]]
        row = {
            "family_label": brow["family_label"],
            "repr_hz": float(brow["repr_hz"]),
            "repr_amp": float(brow["repr_amp"]),
            "best_incoming_edge": float(brow["best_incoming_edge"]) if brow["best_incoming_edge"] not in ("", "nan") else np.nan,
            "best_outgoing_edge": float(brow["best_outgoing_edge"]) if brow["best_outgoing_edge"] not in ("", "nan") else np.nan,
            "incoming_reproducible_count": int(brow["incoming_reproducible_count"]),
            "outgoing_reproducible_count": int(brow["outgoing_reproducible_count"]),
            "contrast_db_0681": float(prow["contrast_db_0681"]),
            "contrast_db_0680": float(prow["contrast_db_0680"]),
        }
        row["class"] = classify(row)
        rows.append(row)

    rows.sort(key=lambda row: (row["class"], -float(row["repr_amp"])))

    csv_path = OUTPUT_DIR / "family_synthesis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    saved = [plot_synthesis(writer, rows)]
    lines = []
    lines.append("Family synthesis report")
    lines.append("Classes combine cross-dataset incoming/outgoing edges with local spectral prominence.")
    lines.append("")
    for cls in ("fundamental-leaning", "mixed-coupled", "nonfundamental-leaning", "unclear"):
        lines.append(cls + ":")
        for row in rows:
            if row["class"] != cls:
                continue
            lines.append(
                f"  {row['family_label']} | {row['repr_hz']:.3f} Hz | amp={row['repr_amp']:.4f} | "
                f"in={row['best_incoming_edge']:.5f} | out={row['best_outgoing_edge']:.5f} | "
                f"prom0681={row['contrast_db_0681']:.2f} dB | prom0680={row['contrast_db_0680']:.2f} dB | "
                f"in_repro={row['incoming_reproducible_count']} | out_repro={row['outgoing_reproducible_count']}"
            )
        lines.append("")
    lines.append("saved_files:")
    lines.append(str(csv_path))
    for path in saved:
        lines.append(str(path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("[saved] family_synthesis.csv")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
