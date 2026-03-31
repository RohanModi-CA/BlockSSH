from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_mode_family_scan import Config as BaseConfig
from run_mode_family_scan import PlotWriter, detect_regions, collect_processed_regions, summarize_target


@dataclass(frozen=True)
class Config:
    peak_csv: str = "/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/configs/peaks/0681ROT270X.csv"
    freq_range_hz: tuple[float, float] = (10.0, 20.0)
    views: tuple[tuple[str, str], ...] = (
        ("IMG_0681_rot270", "x"),
        ("IMG_0681_rot270", "y"),
        ("IMG_0681_rot270", "a"),
        ("IMG_0680_rot270", "x"),
        ("IMG_0680_rot270", "y"),
        ("IMG_0680_rot270", "a"),
        ("CDX_10IC", "x"),
        ("CDX_10IC", "y"),
        ("CDX_10IC", "a"),
    )


CONFIG = Config()
OUTPUT_DIR = Path(__file__).resolve().parent / "high_frequency_catalog_output"


def load_peaks() -> list[float]:
    with Path(CONFIG.peak_csv).open(encoding="utf-8") as f:
        peaks = [float(cell) for cell in next(csv.reader(f)) if str(cell).strip()]
    return [peak for peak in peaks if CONFIG.freq_range_hz[0] <= peak <= CONFIG.freq_range_hz[1]]


def choose_half_width(peak_hz: float) -> float:
    if peak_hz >= 15.0:
        return 0.15
    if peak_hz >= 12.0:
        return 0.20
    return 0.25


def collect_view_data():
    out = {}
    for dataset, component in CONFIG.views:
        ds, regions = detect_regions(dataset, component, BaseConfig())
        out[(dataset, component)] = collect_processed_regions(ds, regions)
    return out


def build_rows(peaks: list[float], view_data) -> list[dict[str, object]]:
    rows = []
    for peak in peaks:
        per_view = []
        for dataset, component in CONFIG.views:
            cfg = BaseConfig(half_width_hz=choose_half_width(peak), n_local_scan=121)
            rec = summarize_target(view_data[(dataset, component)], peak, cfg)
            per_view.append(
                {
                    "label": f"{dataset}:{component}",
                    "score": float(rec["score"]),
                    "corr": float(rec["best_corr"]),
                    "slope": float(rec["best_slope"]),
                    "freq_rms": float(rec["best_freq_rms"]),
                    "prom": float(rec["best_mean_fft_prom"]),
                    "bond": int(rec["best_response_bond"]),
                }
            )
        scores = np.asarray([item["score"] for item in per_view], dtype=float)
        corrs = np.asarray([item["corr"] for item in per_view], dtype=float)
        slopes = np.asarray([item["slope"] for item in per_view], dtype=float)
        good = (scores >= 0.9) & (corrs >= 0.85) & (np.abs(slopes - 1.0) <= 0.2)
        solid = (scores >= 0.7) & (corrs >= 0.8) & (np.abs(slopes - 1.0) <= 0.25)
        best_idx = int(np.nanargmax(scores))
        top3 = float(np.mean(np.sort(scores)[::-1][:3]))
        score = top3 + 0.22 * float(np.sum(good)) + 0.10 * float(np.sum(solid))
        rows.append(
            {
                "peak_hz": peak,
                "adaptive_half_width_hz": choose_half_width(peak),
                "best_view": per_view[best_idx]["label"],
                "best_score": float(per_view[best_idx]["score"]),
                "best_corr": float(per_view[best_idx]["corr"]),
                "best_slope": float(per_view[best_idx]["slope"]),
                "n_good_views": int(np.sum(good)),
                "n_solid_views": int(np.sum(solid)),
                "top3_mean_score": top3,
                "high_frequency_score": score,
            }
        )
    rows.sort(key=lambda row: float(row["high_frequency_score"]), reverse=True)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_ranking(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y, [float(row["high_frequency_score"]) for row in rows], color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{float(row['peak_hz']):.3f} Hz | {row['best_view']}" for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("high-frequency-only score")
    ax.set_title("Adaptive-window ranking of listed 10-20 Hz peaks")
    ax.grid(alpha=0.25, axis="x")
    return writer.save(fig, "ranking")


def build_summary(rows: list[dict[str, object]], csv_path: Path, plot_paths: list[Path]) -> str:
    lines = [
        "idea:",
        "This pass ranks only the listed 10-20 Hz peaks using adaptive windows and no child penalty, to reduce low-frequency bias in the main catalog.",
        "",
        "ranking:",
    ]
    for row in rows:
        lines.append(
            f"{float(row['peak_hz']):.3f} Hz | score={float(row['high_frequency_score']):.3f} | "
            f"best={float(row['best_score']):.3f} @ {row['best_view']} | "
            f"n_good={int(row['n_good_views'])} | n_solid={int(row['n_solid_views'])} | "
            f"hw={float(row['adaptive_half_width_hz']):.2f}"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(csv_path))
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    peaks = load_peaks()
    view_data = collect_view_data()
    rows = build_rows(peaks, view_data)
    csv_path = write_csv(rows, OUTPUT_DIR / "high_frequency_catalog.csv")
    plot_paths = [plot_ranking(writer, rows)]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(rows, csv_path, plot_paths))
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
