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


OUTPUT_DIR = SCRIPT_DIR / "key_peak_output"


@dataclass(frozen=True)
class Case:
    label: str
    f1_hz: float
    f2_hz: float
    f3_hz: float
    controls_hz: tuple[float, ...]


CASES: tuple[Case, ...] = (
    Case(
        label="8.950 + 8.950 -> 18.361",
        f1_hz=8.95047794335,
        f2_hz=8.95047794335,
        f3_hz=18.3613165321,
        controls_hz=(18.00, 18.70),
    ),
    Case(
        label="8.106 + 8.106 -> 16.025",
        f1_hz=8.10635126866,
        f2_hz=8.10635126866,
        f3_hz=16.0249450640,
        controls_hz=(15.75, 16.30),
    ),
    Case(
        label="7.956 + 8.106 -> 15.823",
        f1_hz=7.95607053459,
        f2_hz=8.10635126866,
        f3_hz=15.8234583338,
        controls_hz=(15.55, 16.10),
    ),
    Case(
        label="7.628 + 8.950 -> 16.601",
        f1_hz=7.62818529662,
        f2_hz=8.95047794335,
        f3_hz=16.6009768024,
        controls_hz=(16.30, 16.90),
    ),
    Case(
        label="16.601 + 6.369 -> 23.159",
        f1_hz=16.6009768024,
        f2_hz=6.36911171871,
        f3_hz=23.1592639881,
        controls_hz=(22.85, 23.45),
    ),
    Case(
        label="16.601 + 5.190 -> 21.960",
        f1_hz=16.6009768024,
        f2_hz=5.18953883921,
        f3_hz=21.9597771241,
        controls_hz=(21.65, 22.25),
    ),
)


def prepare(dataset: str, bond_index: int, null_trials: int):
    cfg = replace(BIC_CONFIG, dataset=dataset, bond_index=bond_index, null_trials=null_trials)
    _, ds, processed = load_primary_processed(cfg)
    regions = detect_regions(ds, processed, cfg)
    return cfg, regions


def evaluate_case(case: Case, regions, cfg, label_suffix: str) -> list[dict[str, object]]:
    rows = []
    targets = (case.f3_hz, *case.controls_hz)
    for target in targets:
        triad = Triad(
            label=f"{case.label} | f3={target:.3f} | {label_suffix}",
            f1_hz=case.f1_hz,
            f2_hz=case.f2_hz,
            f3_hz=float(target),
        )
        result = evaluate_triad(regions, triad, cfg)
        rows.append(
            {
                "target_hz": float(target),
                "observed": float(result["observed_bicoherence"]),
                "null95": float(result["null_q95_bicoherence"]),
                "excess": float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"]),
                "windows": int(result["n_windows_total"]),
                "ampcorr": float(result["amp_prod_corr"]),
            }
        )
    return rows


def plot_cases(writer: PlotWriter, combined_rows: list[dict[str, object]]) -> Path:
    labels = []
    ex1 = []
    ex0 = []
    for row in combined_rows:
        labels.append(f"{row['short_label']} @ {row['target_hz']:.2f}")
        ex1.append(float(row["excess_0681"]))
        ex0.append(float(row["excess_0680"]))
    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    ax.bar(x - 0.18, np.asarray(ex1, dtype=float), width=0.36, color="C0", label="0681 excess")
    ax.bar(x + 0.18, np.asarray(ex0, dtype=float), width=0.36, color="C1", label="0680 excess")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylabel("bicoherence excess over null95")
    ax.set_title("Specific raw-peak triads and nearby controls")
    ax.legend(loc="upper right")
    return writer.save(fig, "key_peak_specificity")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    cfg_0681, regions_0681 = prepare("IMG_0681_rot270", 0, 300)
    cfg_0680, regions_0680 = prepare("IMG_0680_rot270", 1, 240)

    combined_rows = []
    lines = []
    lines.append("Key raw-peak triad specificity checks")
    lines.append("Datasets: 0681 bond0 and 0680 bond1")
    lines.append("")
    for case in CASES:
        r1 = evaluate_case(case, regions_0681, cfg_0681, "0681")
        r0 = evaluate_case(case, regions_0680, cfg_0680, "0680")
        lines.append(case.label)
        for row1, row0 in zip(r1, r0):
            is_true_target = abs(float(row1["target_hz"]) - case.f3_hz) < 1e-9
            tag = "target" if is_true_target else "control"
            short_label = case.label.split(" -> ")[1]
            combined_rows.append(
                {
                    "short_label": short_label,
                    "target_hz": float(row1["target_hz"]),
                    "excess_0681": float(row1["excess"]),
                    "excess_0680": float(row0["excess"]),
                }
            )
            lines.append(
                f"  {tag} f3={row1['target_hz']:.3f} | "
                f"0681: obs={row1['observed']:.5f}, null95={row1['null95']:.5f}, excess={row1['excess']:.5f}, windows={row1['windows']} | "
                f"0680: obs={row0['observed']:.5f}, null95={row0['null95']:.5f}, excess={row0['excess']:.5f}, windows={row0['windows']}"
            )
        lines.append("")

    saved = [plot_cases(writer, combined_rows)]
    lines.append("saved_files:")
    for path in saved:
        lines.append(str(path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
