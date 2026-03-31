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

from run_mode_family_scan import Config as BaseConfig
from run_mode_family_scan import PlotWriter, detect_regions, collect_processed_regions, summarize_target


@dataclass(frozen=True)
class Config:
    views: tuple[tuple[str, str], ...] = (
        ("IMG_0681_rot270", "x"),
        ("IMG_0681_rot270", "y"),
        ("IMG_0680_rot270", "y"),
        ("CDX_10IC", "x"),
    )
    targets_hz: tuple[float, ...] = (12.0158503351, 16.6009768024, 18.3613165321)
    half_widths_hz: tuple[float, ...] = (0.15, 0.20, 0.25, 0.35, 0.50)


CONFIG = Config()
OUTPUT_DIR = SCRIPT_DIR / "high_frequency_output"


def collect_results() -> list[dict[str, object]]:
    rows = []
    for dataset, component in CONFIG.views:
        ds, regions = detect_regions(dataset, component, BaseConfig())
        processed = collect_processed_regions(ds, regions)
        for target_hz in CONFIG.targets_hz:
            for half_width_hz in CONFIG.half_widths_hz:
                cfg = BaseConfig(half_width_hz=half_width_hz, n_local_scan=121)
                record = summarize_target(processed, target_hz, cfg)
                rows.append(
                    {
                        "dataset": dataset,
                        "component": component,
                        "target_hz": target_hz,
                        "half_width_hz": half_width_hz,
                        "score": float(record["score"]),
                        "best_response_bond": int(record["best_response_bond"]),
                        "best_corr": float(record["best_corr"]),
                        "best_slope": float(record["best_slope"]),
                        "best_freq_rms": float(record["best_freq_rms"]),
                        "best_mean_fft_prom": float(record["best_mean_fft_prom"]),
                    }
                )
    return rows


def plot_score_vs_width(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(len(CONFIG.views), len(CONFIG.targets_hz), figsize=(13, 9), constrained_layout=True, sharex=True)
    for row_idx, (dataset, component) in enumerate(CONFIG.views):
        for col_idx, target_hz in enumerate(CONFIG.targets_hz):
            ax = axes[row_idx, col_idx]
            subset = [row for row in rows if row["dataset"] == dataset and row["component"] == component and row["target_hz"] == target_hz]
            subset.sort(key=lambda row: float(row["half_width_hz"]))
            widths = [float(row["half_width_hz"]) for row in subset]
            scores = [float(row["score"]) for row in subset]
            corr = [float(row["best_corr"]) for row in subset]
            ax.plot(widths, scores, "o-", color="tab:blue", label="score")
            ax.plot(widths, corr, "s--", color="tab:green", label="corr")
            ax.set_title(f"{dataset} | {component} | {target_hz:.3f} Hz")
            ax.grid(alpha=0.25)
            if row_idx == len(CONFIG.views) - 1:
                ax.set_xlabel("half-width (Hz)")
            if col_idx == 0:
                ax.set_ylabel("score / corr")
    axes[0, 0].legend(loc="best")
    return writer.save(fig, "score_vs_width")


def plot_best_widths(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    best_rows = []
    for dataset, component in CONFIG.views:
        for target_hz in CONFIG.targets_hz:
            subset = [row for row in rows if row["dataset"] == dataset and row["component"] == component and row["target_hz"] == target_hz]
            best_rows.append(max(subset, key=lambda row: float(row["score"])))
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    y = np.arange(len(best_rows))
    ax.barh(y, [float(row["score"]) for row in best_rows], color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['dataset']}:{row['component']} | {float(row['target_hz']):.3f} Hz | hw={float(row['half_width_hz']):.2f}" for row in best_rows])
    ax.invert_yaxis()
    ax.set_xlabel("best score")
    ax.set_title("Best half-width per view and target")
    ax.grid(alpha=0.25, axis="x")
    return writer.save(fig, "best_widths")


def build_summary(rows: list[dict[str, object]], plot_paths: list[Path]) -> str:
    lines = [
        "idea:",
        "Higher-frequency lines may need narrower local windows than the broad defaults used for the low-frequency scan. This study sweeps half-width for 12.0, 16.6, and 18.36 across selected views.",
        "",
    ]
    for dataset, component in CONFIG.views:
        lines.append(f"{dataset} | {component}")
        for target_hz in CONFIG.targets_hz:
            subset = [row for row in rows if row["dataset"] == dataset and row["component"] == component and row["target_hz"] == target_hz]
            best = max(subset, key=lambda row: float(row["score"]))
            lines.append(
                f"  {target_hz:.3f} Hz | best_hw={float(best['half_width_hz']):.2f} | score={float(best['score']):.3f} | "
                f"corr={float(best['best_corr']):.3f} | slope={float(best['best_slope']):.3f} | "
                f"freq_rms={float(best['best_freq_rms']):.4f} | prom={float(best['best_mean_fft_prom']):.3f}"
            )
        lines.append("")
    lines.append("saved_files:")
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    rows = collect_results()
    plot_paths = [
        plot_score_vs_width(writer, rows),
        plot_best_widths(writer, rows),
    ]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(rows, plot_paths))
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
