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


OUTPUT_DIR = SCRIPT_DIR / "midband_scan_output"


@dataclass(frozen=True)
class ScanCase:
    label: str
    f1_hz: float
    f2_hz: float
    scan_lo_hz: float
    scan_hi_hz: float
    step_hz: float
    marked_targets_hz: tuple[float, ...]


CASES: tuple[ScanCase, ...] = (
    ScanCase(
        label="4.766 + 7.956 -> f3",
        f1_hz=4.76602040683,
        f2_hz=7.95607053459,
        scan_lo_hz=12.2,
        scan_hi_hz=12.9,
        step_hz=0.05,
        marked_targets_hz=(12.537, 12.607, 12.700),
    ),
    ScanCase(
        label="4.359 + 8.557 -> f3",
        f1_hz=4.35862456299,
        f2_hz=8.55719347087,
        scan_lo_hz=12.7,
        scan_hi_hz=13.4,
        step_hz=0.05,
        marked_targets_hz=(12.934, 13.049, 13.108),
    ),
    ScanCase(
        label="4.359 + 8.950 -> f3",
        f1_hz=4.35862456299,
        f2_hz=8.95047794335,
        scan_lo_hz=12.8,
        scan_hi_hz=13.5,
        step_hz=0.05,
        marked_targets_hz=(13.049, 13.108, 13.257),
    ),
)


def prepare(dataset: str, bond_index: int, null_trials: int):
    cfg = replace(BIC_CONFIG, dataset=dataset, bond_index=bond_index, null_trials=null_trials)
    _, ds, processed = load_primary_processed(cfg)
    regions = detect_regions(ds, processed, cfg)
    return cfg, regions


def run_scan(case: ScanCase, regions, cfg) -> dict[str, np.ndarray]:
    f3_grid = np.arange(case.scan_lo_hz, case.scan_hi_hz + 0.5 * case.step_hz, case.step_hz, dtype=float)
    excess = np.full_like(f3_grid, np.nan)
    for i, f3 in enumerate(f3_grid):
        result = evaluate_triad(regions, Triad(label="", f1_hz=case.f1_hz, f2_hz=case.f2_hz, f3_hz=float(f3)), cfg)
        excess[i] = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
    return {"f3_grid_hz": f3_grid, "excess": excess}


def plot_scans(writer: PlotWriter, scan_rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(len(CASES), 1, figsize=(10.0, 3.0 * len(CASES)), sharex=False, constrained_layout=True)
    if len(CASES) == 1:
        axes = [axes]
    for ax, case, row in zip(axes, CASES, scan_rows):
        f3 = np.asarray(row["f3_grid_hz"], dtype=float)
        ax.plot(f3, np.asarray(row["excess_0681"], dtype=float), color="C0", linewidth=2.0, label="0681 excess")
        ax.plot(f3, np.asarray(row["excess_0680"], dtype=float), color="C1", linewidth=2.0, label="0680 excess")
        for target in case.marked_targets_hz:
            ax.axvline(float(target), color="0.5", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_title(case.label)
        ax.set_ylabel("excess")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("child frequency f3 (Hz)")
    return writer.save(fig, "midband_child_scans")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    cfg_0681, regions_0681 = prepare("IMG_0681_rot270", 0, 220)
    cfg_0680, regions_0680 = prepare("IMG_0680_rot270", 1, 180)

    scan_rows = []
    lines = []
    lines.append("Midband child-frequency scans")
    lines.append("Datasets: 0681 bond0 and 0680 bond1")
    lines.append("")
    for case in CASES:
        s1 = run_scan(case, regions_0681, cfg_0681)
        s0 = run_scan(case, regions_0680, cfg_0680)
        i1 = int(np.nanargmax(s1["excess"]))
        i0 = int(np.nanargmax(s0["excess"]))
        scan_rows.append({"f3_grid_hz": s1["f3_grid_hz"], "excess_0681": s1["excess"], "excess_0680": s0["excess"]})
        lines.append(case.label)
        lines.append(f"  0681 best: f3={float(s1['f3_grid_hz'][i1]):.3f}, excess={float(s1['excess'][i1]):.5f}")
        lines.append(f"  0680 best: f3={float(s0['f3_grid_hz'][i0]):.3f}, excess={float(s0['excess'][i0]):.5f}")
        for target in case.marked_targets_hz:
            j1 = int(np.argmin(np.abs(s1['f3_grid_hz'] - float(target))))
            j0 = int(np.argmin(np.abs(s0['f3_grid_hz'] - float(target))))
            lines.append(
                f"  marked {target:.3f}: 0681 excess={float(s1['excess'][j1]):.5f}, 0680 excess={float(s0['excess'][j0]):.5f}"
            )
        lines.append("")

    saved = [plot_scans(writer, scan_rows)]
    lines.append("saved_files:")
    for path in saved:
        lines.append(str(path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
