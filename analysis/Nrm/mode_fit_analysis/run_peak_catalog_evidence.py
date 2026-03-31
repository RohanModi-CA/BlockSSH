from __future__ import annotations

import csv
import pickle
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

from run_mode_family_scan import CONFIG as FAMILY_CONFIG
from run_mode_family_scan import PlotWriter, run_scan_for_view


@dataclass(frozen=True)
class Config:
    peak_csv: str = "/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/configs/peaks/0681ROT270X.csv"
    taxonomy_csv: str = "/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/Nrm/peak_coherence_analysis/taxonomy_output/peak_taxonomy.csv"
    family_match_tol_hz: float = 0.25
    strong_view_score: float = 1.0
    good_view_score: float = 0.7
    strong_corr: float = 0.9
    slope_tol: float = 0.2
    reference_hz: tuple[float, ...] = (3.35, 6.35, 8.96, 12.0, 16.65, 18.0)


CONFIG = Config()
OUTPUT_DIR = SCRIPT_DIR / "catalog_output"
CACHE_PATH = OUTPUT_DIR / "view_results.pkl"


def load_peak_list(path: str) -> list[float]:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected single-row peak CSV, got {len(rows)} rows")
    return [float(cell) for cell in rows[0] if str(cell).strip()]


def load_taxonomy(path: str) -> dict[float, dict[str, object]]:
    with Path(path).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out: dict[float, dict[str, object]] = {}
    for row in rows:
        peak = float(row["peak_hz"])
        out[peak] = row
    return out


def nearest_record(records: list[dict], peak_hz: float) -> dict:
    return min(records, key=lambda item: abs(float(item["target_hz"]) - peak_hz))


def match_family(candidates: list[dict], peak_hz: float, tol_hz: float) -> dict | None:
    if not candidates:
        return None
    candidate = min(candidates, key=lambda item: abs(float(item["family_center_hz"]) - peak_hz))
    if abs(float(candidate["family_center_hz"]) - peak_hz) <= tol_hz:
        return candidate
    return None


def build_catalog_rows(peaks: list[float], taxonomy: dict[float, dict[str, object]], view_results: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for peak_hz in peaks:
        per_view = []
        for result in view_results:
            if not (FAMILY_CONFIG.scan_range_hz[0] <= peak_hz <= FAMILY_CONFIG.scan_range_hz[1]):
                per_view.append(
                    {
                        "label": f"{result['dataset']}:{result['component']}",
                        "dataset": result["dataset"],
                        "component": result["component"],
                        "score": np.nan,
                        "best_corr": np.nan,
                        "best_slope": np.nan,
                        "best_mean_r2": np.nan,
                        "best_freq_rms": np.nan,
                        "best_mean_fft_prom": np.nan,
                        "best_response_bond": -1,
                        "family_center_hz": np.nan,
                        "family_score": np.nan,
                        "family_match": False,
                    }
                )
                continue

            record = nearest_record(result["records"], peak_hz)
            family = match_family(result["candidates"], peak_hz, CONFIG.family_match_tol_hz)
            per_view.append(
                {
                    "label": f"{result['dataset']}:{result['component']}",
                    "dataset": result["dataset"],
                    "component": result["component"],
                    "score": float(record["score"]),
                    "best_corr": float(record["best_corr"]),
                    "best_slope": float(record["best_slope"]),
                    "best_mean_r2": float(record["best_mean_r2"]),
                    "best_freq_rms": float(record["best_freq_rms"]),
                    "best_mean_fft_prom": float(record["best_mean_fft_prom"]),
                    "best_response_bond": int(record["best_response_bond"]),
                    "family_center_hz": float(family["family_center_hz"]) if family is not None else np.nan,
                    "family_score": float(family["smooth_score"]) if family is not None else np.nan,
                    "family_match": family is not None,
                }
            )

        scores = np.asarray([view["score"] for view in per_view], dtype=float)
        corrs = np.asarray([view["best_corr"] for view in per_view], dtype=float)
        slopes = np.asarray([view["best_slope"] for view in per_view], dtype=float)
        family_matches = np.asarray([bool(view["family_match"]) for view in per_view], dtype=bool)
        valid_scores = np.isfinite(scores)
        good_slope = np.isfinite(slopes) & (np.abs(slopes - 1.0) <= CONFIG.slope_tol)
        strong_mask = valid_scores & (scores >= CONFIG.strong_view_score) & (corrs >= CONFIG.strong_corr) & good_slope
        good_mask = valid_scores & (scores >= CONFIG.good_view_score) & good_slope

        best_idx = int(np.nanargmax(scores)) if np.any(valid_scores) else 0
        best_view = per_view[best_idx]
        if np.any(family_matches):
            candidate_scores = scores[family_matches & valid_scores]
        else:
            candidate_scores = scores[valid_scores]
        sorted_scores = np.sort(candidate_scores)[::-1] if candidate_scores.size else np.asarray([], dtype=float)
        top3_mean = float(np.mean(sorted_scores[: min(3, sorted_scores.size)])) if sorted_scores.size else np.nan
        median_score = float(np.nanmedian(scores)) if np.any(valid_scores) else np.nan
        taxonomy_row = taxonomy.get(peak_hz)
        child_evidence = float(taxonomy_row["child_evidence"]) if taxonomy_row is not None and taxonomy_row["child_evidence"] not in ("", "nan") else np.nan
        outgoing_count = int(taxonomy_row["outgoing_count"]) if taxonomy_row is not None else -1
        tag = str(taxonomy_row["tag"]) if taxonomy_row is not None else ""
        best_relation = str(taxonomy_row["best_relation"]) if taxonomy_row is not None else ""

        child_penalty = max(0.0, child_evidence) if np.isfinite(child_evidence) else 0.0
        base_score = top3_mean if np.isfinite(top3_mean) else -1.0
        combined_score = (
            base_score
            + 0.20 * float(np.sum(strong_mask))
            + 0.08 * float(np.sum(good_mask))
            + 0.18 * float(np.sum(family_matches))
            - 1.3 * child_penalty
        )

        row = {
            "peak_hz": peak_hz,
            "best_view": best_view["label"],
            "best_view_score": float(best_view["score"]),
            "best_view_corr": float(best_view["best_corr"]),
            "best_view_slope": float(best_view["best_slope"]),
            "best_view_freq_rms": float(best_view["best_freq_rms"]),
            "best_view_prom": float(best_view["best_mean_fft_prom"]),
            "best_response_bond": int(best_view["best_response_bond"]),
            "top3_mean_score": top3_mean,
            "median_view_score": median_score,
            "n_strong_views": int(np.sum(strong_mask)),
            "n_good_views": int(np.sum(good_mask)),
            "n_family_matches": int(np.sum(family_matches)),
            "child_evidence": child_evidence,
            "outgoing_count": outgoing_count,
            "taxonomy_tag": tag,
            "best_relation": best_relation,
            "combined_score": combined_score,
        }
        for view in per_view:
            prefix = view["label"].replace(":", "__")
            row[f"{prefix}__score"] = view["score"]
            row[f"{prefix}__corr"] = view["best_corr"]
            row[f"{prefix}__slope"] = view["best_slope"]
            row[f"{prefix}__family_match"] = int(view["family_match"])
        rows.append(row)
    rows.sort(key=lambda row: float(row["combined_score"]), reverse=True)
    return rows


def write_catalog_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_combined_ranking(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    top_rows = rows[:25]
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    y = np.arange(len(top_rows))
    ax.barh(y, [float(row["combined_score"]) for row in top_rows], color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{float(row['peak_hz']):.3f} Hz | {row['best_view']}" for row in top_rows])
    ax.invert_yaxis()
    ax.set_xlabel("combined evidence score")
    ax.set_title("Top peaks by combined mode-like recurrence and anti-child evidence")
    ax.grid(alpha=0.25, axis="x")
    return writer.save(fig, "combined_ranking")


def plot_child_vs_recurrence(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    child = np.asarray([float(row["child_evidence"]) if np.isfinite(float(row["child_evidence"])) else 0.0 for row in rows], dtype=float)
    recurrence = np.asarray([float(row["n_strong_views"]) + 0.5 * float(row["n_family_matches"]) for row in rows], dtype=float)
    freq = np.asarray([float(row["peak_hz"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    scatter = ax.scatter(child, recurrence, c=freq, cmap="turbo", s=42, edgecolors="black", linewidths=0.3)
    fig.colorbar(scatter, ax=ax, label="peak frequency (Hz)")
    ax.set_xlabel("child evidence")
    ax.set_ylabel("recurrence score")
    ax.set_title("Peaks that recur across views while resisting child explanations")
    ax.grid(alpha=0.25)
    return writer.save(fig, "child_vs_recurrence")


def plot_reference_panel(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(len(CONFIG.reference_hz), 1, figsize=(12, 2.5 * len(CONFIG.reference_hz)), constrained_layout=True)
    if len(CONFIG.reference_hz) == 1:
        axes = np.asarray([axes])
    sorted_by_peak = sorted(rows, key=lambda row: abs(float(row["peak_hz"])))
    for ax, ref_hz in zip(axes, CONFIG.reference_hz):
        nearest = sorted(rows, key=lambda row: abs(float(row["peak_hz"]) - ref_hz))[:6]
        x = np.arange(len(nearest))
        ax.bar(x - 0.18, [float(row["combined_score"]) for row in nearest], width=0.36, label="combined", color="tab:blue")
        ax.bar(x + 0.18, [float(row["best_view_score"]) for row in nearest], width=0.36, label="best view", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(row['peak_hz']):.3f}" for row in nearest], rotation=20, ha="right")
        ax.set_title(f"Nearest listed peaks to {ref_hz:.2f} Hz")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(loc="upper right")
    return writer.save(fig, "reference_panel")


def build_summary(rows: list[dict[str, object]], csv_path: Path, plot_paths: list[Path]) -> str:
    lines = [
        f"peak_csv: {CONFIG.peak_csv}",
        f"taxonomy_csv: {CONFIG.taxonomy_csv}",
        f"n_listed_peaks: {len(rows)}",
        "",
        "idea:",
        "A candidate looks more fundamental-like if it recurs as a mode-like family across multiple views, lands on a strong bond-agnostic scaling score, and does not already look well explained as a child of lower frequencies.",
        "",
        "top combined-evidence peaks:",
    ]
    for row in rows[:25]:
        lines.append(
            f"{float(row['peak_hz']):.3f} Hz | combined={float(row['combined_score']):.3f} | "
            f"best={float(row['best_view_score']):.3f} @ {row['best_view']} | "
            f"n_strong={int(row['n_strong_views'])} | n_family={int(row['n_family_matches'])} | "
            f"child={float(row['child_evidence']) if np.isfinite(float(row['child_evidence'])) else float('nan'):.5f} | "
            f"tag={row['taxonomy_tag']} | relation={row['best_relation']}"
        )
    lines.append("")
    lines.append("reference neighborhoods:")
    for ref_hz in CONFIG.reference_hz:
        nearest = sorted(rows, key=lambda row: abs(float(row["peak_hz"]) - ref_hz))[:5]
        lines.append(f"near {ref_hz:.2f} Hz")
        for row in nearest:
            lines.append(
                f"  {float(row['peak_hz']):.3f} Hz | combined={float(row['combined_score']):.3f} | "
                f"best={float(row['best_view_score']):.3f} @ {row['best_view']} | "
                f"n_strong={int(row['n_strong_views'])} | child={float(row['child_evidence']) if np.isfinite(float(row['child_evidence'])) else float('nan'):.5f}"
            )
    lines.append("")
    lines.append("notes:")
    lines.append("This ranking is still heuristic. It is designed to elevate peaks that repeatedly look mode-like across views while discounting peaks that already have strong lower-frequency explanations.")
    lines.append("Higher-frequency fundamentals do not have to win in every observable; recurrence across several views matters more than a single best view.")
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(csv_path))
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    peaks = load_peak_list(CONFIG.peak_csv)
    taxonomy = load_taxonomy(CONFIG.taxonomy_csv)
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            view_results = pickle.load(f)
        print(f"[cache] loaded {CACHE_PATH.name}")
    else:
        view_results = []
        for dataset in FAMILY_CONFIG.datasets:
            for component in FAMILY_CONFIG.components:
                print(f"[catalog] scanning {dataset} | {component}")
                view_results.append(run_scan_for_view(dataset, component, FAMILY_CONFIG))
        with CACHE_PATH.open("wb") as f:
            pickle.dump(view_results, f)
        print(f"[cache] saved {CACHE_PATH.name}")
    rows = build_catalog_rows(peaks, taxonomy, view_results)
    csv_path = write_catalog_csv(rows, OUTPUT_DIR / "peak_catalog_evidence.csv")
    plot_paths = [
        plot_combined_ranking(writer, rows),
        plot_child_vs_recurrence(writer, rows),
        plot_reference_panel(writer, rows),
    ]
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(build_summary(rows, csv_path, plot_paths))
    print(f"[saved] {csv_path.name}")
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
