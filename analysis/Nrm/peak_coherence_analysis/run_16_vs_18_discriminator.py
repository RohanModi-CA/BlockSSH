from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_peak_bicoherence_analysis import CONFIG as BIC_CONFIG
from run_peak_bicoherence_analysis import PlotWriter, Triad, detect_regions, evaluate_triad, load_primary_processed


@dataclass(frozen=True)
class TargetCase:
    label: str
    dataset: str
    bond_index: int
    triads: tuple[Triad, ...]


CASES = (
    TargetCase(
        label="16.6 family in its stronger views",
        dataset="IMG_0681_rot270",
        bond_index=1,
        triads=(
            Triad(label="8.106 + 8.557 -> 16.663", f1_hz=8.10635126866, f2_hz=8.55719347087, f3_hz=16.66354473953),
            Triad(label="7.956 + 8.742 -> 16.698", f1_hz=7.95607053459, f2_hz=8.74162891723, f3_hz=16.69769945182),
            Triad(label="4.766 + 12.016 -> 16.782", f1_hz=4.76602040683, f2_hz=12.0158503351, f3_hz=16.78187074193),
            Triad(label="3.329 + 13.257 -> 16.586", f1_hz=3.32856015604, f2_hz=13.2568917893, f3_hz=16.58545194534),
            Triad(label="1.902 + 14.701 -> 16.602", f1_hz=1.90186098613, f2_hz=14.7008972301, f3_hz=16.60275821623),
        ),
    ),
    TargetCase(
        label="18 family in its stronger control view",
        dataset="IMG_0681_rot270",
        bond_index=0,
        triads=(
            Triad(label="8.950 + 8.950 -> 18.361", f1_hz=8.95047794335, f2_hz=8.95047794335, f3_hz=18.3613165321),
            Triad(label="7.082 + 11.333 -> 18.415", f1_hz=7.0817099, f2_hz=11.3332775353, f3_hz=18.4149874353),
            Triad(label="6.369 + 12.016 -> 18.385", f1_hz=6.36911171871, f2_hz=12.0158503351, f3_hz=18.38496205381),
        ),
    ),
)

OUTPUT_DIR = SCRIPT_DIR / "discriminator_output"


def plot_case_bars(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3.4 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])
    for ax, row in zip(axes, rows):
        labels = [result["label"] for result in row["results"]]
        diffs = [float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"]) for result in row["results"]]
        ax.bar(np.arange(len(labels)), diffs, color="tab:blue")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("obs - null95")
        ax.set_title(str(row["label"]))
        ax.grid(alpha=0.25, axis="y")
    return writer.save(fig, "case_bars")


def build_summary(rows: list[dict[str, object]], saved_paths: list[Path]) -> str:
    lines = []
    for row in rows:
        diffs = [float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"]) for result in row["results"]]
        lines.append(f"{row['label']}:")
        for result, diff in zip(row["results"], diffs):
            status = "above null" if diff > 0 else "not above null"
            lines.append(
                f"  {result['label']}: observed={float(result['observed_bicoherence']):.5f}, "
                f"null95={float(result['null_q95_bicoherence']):.5f}, diff={diff:.5f} | {status}"
            )
        lines.append(f"  best_parent_diff={max(diffs):.5f}")
        lines.append(f"  second_best_diff={sorted(diffs, reverse=True)[1]:.5f}" if len(diffs) > 1 else "")
        lines.append("")

    lines.append("working read:")
    lines.append("A follower line should have a specific parent relation that stands out clearly over alternatives in a sensible view.")
    lines.append("If the target only shows scattered weak positives across many different parent guesses, that is not the same thing as a clean follower signature.")
    lines.append("")
    lines.append("saved_plots:")
    for path in saved_paths:
        lines.append(str(path))
    return "\n".join(line for line in lines if line != "") + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    rows = []
    for case in CASES:
        bic_cfg = replace(BIC_CONFIG, dataset=case.dataset, bond_index=case.bond_index)
        _, ds_bic, processed_bic = load_primary_processed(bic_cfg)
        regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg)
        results = [evaluate_triad(regions_bic, triad, bic_cfg) | {"label": triad.label} for triad in case.triads]
        rows.append({"label": case.label, "results": results})
    saved_paths = [plot_case_bars(writer, rows)]
    (OUTPUT_DIR / "summary.txt").write_text(build_summary(rows, saved_paths))
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
