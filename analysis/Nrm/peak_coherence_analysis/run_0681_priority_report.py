from __future__ import annotations

import csv
import sys
from dataclasses import replace
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
from run_peak_coherence_analysis import (
    CONFIG as PHASE_CONFIG,
    HarmonicHypothesis,
    build_regions,
    detect_peak_times,
    evaluate_hypothesis,
    load_primary_dataset,
)

OUTPUT_DIR = SCRIPT_DIR / "priority_output"
FAMILY_CSV = SCRIPT_DIR / "family_taxonomy_output" / "family_taxonomy.csv"


def load_family_rows():
    rows = []
    with FAMILY_CSV.open() as f:
        for row in csv.DictReader(f):
            row["repr_hz"] = float(row["repr_hz"])
            row["repr_amp"] = float(row["repr_amp"])
            row["child_evidence"] = float(row["child_evidence"]) if row["child_evidence"] not in ("", "nan") else np.nan
            row["outgoing_count"] = int(row["outgoing_count"])
            row["family_id"] = int(row["family_id"])
            rows.append(row)
    return rows


def plot_priority_summary(writer: PlotWriter, priority_rows: list[dict[str, object]]) -> Path:
    x = np.arange(len(priority_rows), dtype=float)
    amp = np.asarray([float(row["repr_amp"]) for row in priority_rows], dtype=float)
    child = np.asarray([float(row["best_incoming_excess"]) if np.isfinite(float(row["best_incoming_excess"])) else 0.0 for row in priority_rows], dtype=float)
    labels = [str(row["family_label"]) for row in priority_rows]

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax1.bar(x - 0.18, amp, width=0.36, color="C0", label="family amplitude")
    ax1.set_ylabel("tracked amplitude", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(alpha=0.25, axis="y")

    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, child, width=0.36, color="C1", label="best incoming child evidence")
    ax2.set_ylabel("incoming child evidence", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("0681 strongest families: amplitude vs incoming-child evidence")
    return writer.save(fig, "priority_summary")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    families = load_family_rows()
    priority = sorted(families, key=lambda row: row["repr_amp"], reverse=True)[:12]
    families_by_label = {row["family_label"]: row for row in families}

    phase_cfg = replace(PHASE_CONFIG, dataset="IMG_0681_rot270", primary_bond_index=0, null_trials=600)
    _, ds_phase, processed_phase = load_primary_dataset(phase_cfg.segmentation_mode, phase_cfg)
    t_global, peak_times = detect_peak_times(processed_phase, phase_cfg)
    regions_phase = build_regions(ds_phase, phase_cfg.primary_bond_index, t_global, peak_times, phase_cfg)

    bic_cfg = replace(BIC_CONFIG, dataset="IMG_0681_rot270", bond_index=0, null_trials=250)
    _, ds_bic, processed_bic = load_primary_processed(bic_cfg)
    regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg)

    priority_rows = []
    for child in priority:
        incoming = []
        lower = [row for row in families if row["repr_hz"] < child["repr_hz"]]
        for i, left in enumerate(lower):
            for right in lower[i:]:
                if abs((left["repr_hz"] + right["repr_hz"]) - child["repr_hz"]) <= 0.12:
                    result = evaluate_triad(
                        regions_bic,
                        Triad(label="", f1_hz=left["repr_hz"], f2_hz=right["repr_hz"], f3_hz=child["repr_hz"]),
                        bic_cfg,
                    )
                    excess = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
                    incoming.append(
                        {
                            "kind": "pair",
                            "relation": f"{left['family_label']} + {right['family_label']}",
                            "excess": excess,
                            "parent_amp": float(np.sqrt(left["repr_amp"] * right["repr_amp"])),
                            "obs": float(result["observed_bicoherence"]),
                            "null95": float(result["null_q95_bicoherence"]),
                        }
                    )
        for parent in lower:
            for order in range(2, 6):
                if abs(order * parent["repr_hz"] - child["repr_hz"]) <= 0.18:
                    result = evaluate_hypothesis(
                        regions_phase,
                        HarmonicHypothesis(label="", parent_hz=parent["repr_hz"], target_hz=child["repr_hz"], order=order),
                        phase_cfg,
                    )
                    excess = float(result["observed_plv"]) - float(result["null_q95_plv"])
                    incoming.append(
                        {
                            "kind": "harmonic",
                            "relation": f"{order} x {parent['family_label']}",
                            "excess": excess,
                            "parent_amp": float(parent["repr_amp"]),
                            "obs": float(result["observed_plv"]),
                            "null95": float(result["null_q95_plv"]),
                        }
                    )

        incoming.sort(key=lambda row: row["excess"], reverse=True)
        best = incoming[0] if incoming else None
        ratio = float(child["repr_amp"] / best["parent_amp"]) if best and best["parent_amp"] > 0 else np.nan
        priority_rows.append(
            {
                "family_label": child["family_label"],
                "repr_hz": child["repr_hz"],
                "repr_amp": child["repr_amp"],
                "outgoing_count": child["outgoing_count"],
                "best_incoming_relation": best["relation"] if best else "",
                "best_incoming_kind": best["kind"] if best else "",
                "best_incoming_excess": best["excess"] if best else np.nan,
                "best_incoming_obs": best["obs"] if best else np.nan,
                "best_incoming_null95": best["null95"] if best else np.nan,
                "parent_amp_proxy": best["parent_amp"] if best else np.nan,
                "child_to_parent_amp_ratio": ratio,
                "all_incoming": incoming[:6],
            }
        )

    csv_path = OUTPUT_DIR / "priority_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "family_label",
                "repr_hz",
                "repr_amp",
                "outgoing_count",
                "best_incoming_relation",
                "best_incoming_kind",
                "best_incoming_excess",
                "best_incoming_obs",
                "best_incoming_null95",
                "parent_amp_proxy",
                "child_to_parent_amp_ratio",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows([{k: v for k, v in row.items() if k != "all_incoming"} for row in priority_rows])

    saved_paths = [plot_priority_summary(writer, priority_rows)]

    notes = []
    notes.append("0681 priority report for strongest >3 Hz families")
    notes.append("")
    for row in priority_rows:
        notes.append(
            f"{row['family_label']} | {row['repr_hz']:.3f} Hz | amp={row['repr_amp']:.4f} | outgoing={row['outgoing_count']} | "
            f"best_incoming={row['best_incoming_relation']} ({row['best_incoming_kind']}) | "
            f"excess={row['best_incoming_excess']:.5f} | child/parent_amp={row['child_to_parent_amp_ratio']:.3f}"
        )
        for item in row["all_incoming"]:
            notes.append(
                f"  {item['relation']} | kind={item['kind']} | excess={item['excess']:.5f} | "
                f"obs={item['obs']:.5f} | null95={item['null95']:.5f} | parent_amp={item['parent_amp']:.5f}"
            )
    notes.append("")
    notes.append("saved_files:")
    notes.append(str(csv_path))
    for path in saved_paths:
        notes.append(str(path))

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(notes) + "\n")
    print(f"[saved] {csv_path.name}")
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
