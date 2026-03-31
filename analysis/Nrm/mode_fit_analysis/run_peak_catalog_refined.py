from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_mode_family_scan import PlotWriter


OUTPUT_DIR = Path(__file__).resolve().parent / "catalog_refined_output"
CATALOG_CSV = Path(__file__).resolve().parent / "catalog_output" / "peak_catalog_evidence.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_relation(text: str) -> list[float]:
    text = text.strip()
    if not text:
        return []
    if " + " in text:
        return [float(part) for part in text.split(" + ")]
    if " x " in text:
        _, parent = text.split(" x ")
        return [float(parent)]
    return []


def nearest_peak(rows: list[dict[str, str]], target: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(float(row["peak_hz"]) - target))


def parent_support(row: dict[str, str]) -> float:
    best = float(row["best_view_score"]) if row["best_view_score"] not in ("", "nan") else np.nan
    strong = float(row["n_strong_views"])
    family = float(row["n_family_matches"])
    if not np.isfinite(best):
        return 0.0
    score_term = min(max(best, 0.0), 1.6) / 1.6
    recurrence_term = min(strong, 5.0) / 5.0
    family_term = min(family, 5.0) / 5.0
    return float(0.55 * score_term + 0.30 * recurrence_term + 0.15 * family_term)


def build_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        peak = float(row["peak_hz"])
        child = float(row["child_evidence"]) if row["child_evidence"] not in ("", "nan") else np.nan
        top3 = float(row["top3_mean_score"]) if row["top3_mean_score"] not in ("", "nan") else np.nan
        n_strong = int(row["n_strong_views"])
        n_good = int(row["n_good_views"])
        n_family = int(row["n_family_matches"])
        relation = row["best_relation"]

        parents = parse_relation(relation)
        supports = [parent_support(nearest_peak(rows, parent)) for parent in parents]
        relation_support = float(np.prod(supports) ** (1.0 / len(supports))) if supports else 0.0
        raw_penalty = max(0.0, child) if np.isfinite(child) else 0.0
        weighted_penalty = raw_penalty * relation_support

        refined = (
            (top3 if np.isfinite(top3) else -1.0)
            + 0.20 * n_strong
            + 0.08 * n_good
            + 0.18 * n_family
            - 1.3 * weighted_penalty
        )

        out.append(
            {
                **row,
                "parent_support_geomean": relation_support,
                "weighted_child_penalty": weighted_penalty,
                "refined_score": refined,
            }
        )
    out.sort(key=lambda row: float(row["refined_score"]), reverse=True)
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_penalty_compare(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    top = rows[:25]
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(top))
    ax.bar(x - 0.18, [float(row["combined_score"]) for row in top], width=0.36, label="old", color="tab:blue")
    ax.bar(x + 0.18, [float(row["refined_score"]) for row in top], width=0.36, label="refined", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(row['peak_hz']):.3f}" for row in top], rotation=45, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Catalog ranking before and after relation-weighted child penalty")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    return writer.save(fig, "penalty_compare")


def plot_relation_support(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    child = np.asarray([float(row["child_evidence"]) if row["child_evidence"] not in ("", "nan") else 0.0 for row in rows], dtype=float)
    support = np.asarray([float(row["parent_support_geomean"]) for row in rows], dtype=float)
    freq = np.asarray([float(row["peak_hz"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sc = ax.scatter(child, support, c=freq, cmap="turbo", s=42, edgecolors="black", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="peak frequency (Hz)")
    ax.set_xlabel("child evidence")
    ax.set_ylabel("relation support from proposed parents")
    ax.set_title("When child-evidence is actually backed by strong parents")
    ax.grid(alpha=0.25)
    return writer.save(fig, "relation_support")


def build_summary(rows: list[dict[str, object]], csv_path: Path, plot_paths: list[Path]) -> str:
    lines = [
        f"source_catalog: {CATALOG_CSV}",
        "",
        "idea:",
        "Child evidence should count less when the proposed parent relation is itself weak. This refined pass weights the child penalty by the catalog support of the proposed parent peaks.",
        "",
        "top refined-score peaks:",
    ]
    for row in rows[:25]:
        lines.append(
            f"{float(row['peak_hz']):.3f} Hz | refined={float(row['refined_score']):.3f} | old={float(row['combined_score']):.3f} | "
            f"best={float(row['best_view_score']) if row['best_view_score'] not in ('', 'nan') else float('nan'):.3f} @ {row['best_view']} | "
            f"n_strong={int(row['n_strong_views'])} | child={float(row['child_evidence']) if row['child_evidence'] not in ('', 'nan') else float('nan'):.5f} | "
            f"parent_support={float(row['parent_support_geomean']):.3f} | relation={row['best_relation']}"
        )
    lines.append("")
    lines.append("reference peaks:")
    for peak in (3.32856015604, 6.36911171871, 8.95047794335, 12.0158503351, 16.6009768024, 18.3613165321):
        row = next(item for item in rows if abs(float(item["peak_hz"]) - peak) < 1e-12)
        lines.append(
            f"{peak:.3f} Hz | refined={float(row['refined_score']):.3f} | old={float(row['combined_score']):.3f} | "
            f"parent_support={float(row['parent_support_geomean']):.3f} | weighted_child_penalty={float(row['weighted_child_penalty']):.5f} | "
            f"relation={row['best_relation']}"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(csv_path))
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    rows = build_rows(load_rows(CATALOG_CSV))
    csv_path = write_csv(rows, OUTPUT_DIR / "peak_catalog_refined.csv")
    plot_paths = [
        plot_penalty_compare(writer, rows),
        plot_relation_support(writer, rows),
    ]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(rows, csv_path, plot_paths))
    print(f"[saved] {csv_path.name}")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()

