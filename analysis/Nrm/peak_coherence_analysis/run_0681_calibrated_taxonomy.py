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
    summarize_line_tracks,
)


OUTPUT_DIR = SCRIPT_DIR / "calibrated_taxonomy_output"
FAMILY_CSV = SCRIPT_DIR / "family_taxonomy_output" / "family_taxonomy.csv"


def load_families() -> list[dict[str, object]]:
    rows = []
    with FAMILY_CSV.open() as f:
        for row in csv.DictReader(f):
            row["family_id"] = int(row["family_id"])
            row["repr_hz"] = float(row["repr_hz"])
            row["repr_amp"] = float(row["repr_amp"])
            row["raw_peaks"] = [float(x) for x in row["raw_peaks"].split()]
            rows.append(row)
    return rows


def region_presence(regions_phase, phase_cfg, freq_hz: float) -> tuple[int, int, float]:
    present = 0
    total = 0
    amp_vals = []
    for region in regions_phase:
        processed = region["processed"]
        summary = summarize_line_tracks([region], float(freq_hz), phase_cfg)
        if len(summary["rows"]) == 0:
            continue
        total += 1
        amp = float(summary["rows"][0]["mean_amp"])
        amp_vals.append(amp)
        if amp > 0:
            present += 1
    return present, total, float(np.mean(amp_vals)) if amp_vals else np.nan


def dataset_mean_amp(dataset: str, bond_index: int, freq_hz: float) -> float:
    phase_cfg = replace(PHASE_CONFIG, dataset=dataset, primary_bond_index=bond_index)
    _, _, processed = load_primary_dataset(phase_cfg.segmentation_mode, phase_cfg)
    summary = summarize_line_tracks(
        [{"region_index": -1, "left": 0.0, "right": 0.0, "processed": processed}],
        float(freq_hz),
        phase_cfg,
    )
    return float(summary["global_mean_amp"])


def plot_scorecard(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    rows_sorted = sorted(rows, key=lambda row: float(row["mode_score"]), reverse=True)
    labels = [str(row["family_label"]) for row in rows_sorted]
    score = np.asarray([float(row["mode_score"]) for row in rows_sorted], dtype=float)
    incoming = np.asarray([max(0.0, float(row["incoming_best"])) if np.isfinite(float(row["incoming_best"])) else 0.0 for row in rows_sorted], dtype=float)
    outgoing = np.asarray([max(0.0, float(row["outgoing_best"])) if np.isfinite(float(row["outgoing_best"])) else 0.0 for row in rows_sorted], dtype=float)

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.bar(x - 0.24, score, width=0.24, color="C0", label="mode score")
    ax.bar(x, outgoing, width=0.24, color="C2", label="best outgoing excess")
    ax.bar(x + 0.24, incoming, width=0.24, color="C1", label="best incoming excess")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    ax.set_title("0681 calibrated family scorecard")
    return writer.save(fig, "calibrated_scorecard")


def plot_amp_vs_score(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    amp = np.asarray([float(row["repr_amp"]) for row in rows], dtype=float)
    score = np.asarray([float(row["mode_score"]) for row in rows], dtype=float)
    freq = np.asarray([float(row["repr_hz"]) for row in rows], dtype=float)
    labels = [str(row["family_label"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    sc = ax.scatter(amp, score, c=freq, cmap="turbo", s=85, edgecolor="black", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="frequency (Hz)")
    for x, y, text in zip(amp, score, labels):
        ax.text(x, y, text, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("tracked amplitude")
    ax.set_ylabel("mode score")
    ax.grid(alpha=0.25)
    ax.set_title("Amplitude vs calibrated mode score")
    return writer.save(fig, "amp_vs_mode_score")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    families = load_families()

    phase_cfg_0681 = replace(PHASE_CONFIG, dataset="IMG_0681_rot270", primary_bond_index=0, null_trials=500)
    _, ds_phase, processed_phase = load_primary_dataset(phase_cfg_0681.segmentation_mode, phase_cfg_0681)
    t_global, peak_times = detect_peak_times(processed_phase, phase_cfg_0681)
    regions_phase = build_regions(ds_phase, phase_cfg_0681.primary_bond_index, t_global, peak_times, phase_cfg_0681)

    bic_cfg_0681 = replace(BIC_CONFIG, dataset="IMG_0681_rot270", bond_index=0, null_trials=200)
    _, ds_bic, processed_bic = load_primary_processed(bic_cfg_0681)
    regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg_0681)

    # Secondary dataset for reproducibility / calibration.
    phase_cfg_0680 = replace(PHASE_CONFIG, dataset="IMG_0680_rot270", primary_bond_index=1)

    amp_values = np.asarray([float(row["repr_amp"]) for row in families], dtype=float)
    amp_rank = np.argsort(np.argsort(amp_values)) / max(1, len(amp_values) - 1)

    rows = []
    for idx, family in enumerate(families):
        f = float(family["repr_hz"])
        lower = [row for row in families if float(row["repr_hz"]) < f]
        higher = [row for row in families if float(row["repr_hz"]) > f]

        incoming_excesses = []
        for i, left in enumerate(lower):
            for right in lower[i:]:
                if abs((float(left["repr_hz"]) + float(right["repr_hz"])) - f) <= 0.12:
                    result = evaluate_triad(
                        regions_bic,
                        Triad(label="", f1_hz=float(left["repr_hz"]), f2_hz=float(right["repr_hz"]), f3_hz=f),
                        bic_cfg_0681,
                    )
                    incoming_excesses.append(float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"]))
        for parent in lower:
            for order in range(2, 6):
                if abs(order * float(parent["repr_hz"]) - f) <= 0.18:
                    result = evaluate_hypothesis(
                        regions_phase,
                        HarmonicHypothesis(label="", parent_hz=float(parent["repr_hz"]), target_hz=f, order=order),
                        phase_cfg_0681,
                    )
                    incoming_excesses.append(float(result["observed_plv"]) - float(result["null_q95_plv"]))

        outgoing_excesses = []
        for child in higher:
            fc = float(child["repr_hz"])
            for partner in families:
                fp = float(partner["repr_hz"])
                if fp == f:
                    if abs((f + f) - fc) > 0.12:
                        continue
                else:
                    if abs((f + fp) - fc) > 0.12:
                        continue
                result = evaluate_triad(
                    regions_bic,
                    Triad(label="", f1_hz=f, f2_hz=fp, f3_hz=fc),
                    bic_cfg_0681,
                )
                outgoing_excesses.append(float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"]))

        incoming_best = max(incoming_excesses) if incoming_excesses else np.nan
        outgoing_best = max(outgoing_excesses) if outgoing_excesses else np.nan
        outgoing_count = int(np.sum(np.asarray(outgoing_excesses, dtype=float) > 0.0)) if outgoing_excesses else 0

        present, total, region_mean_amp = region_presence(regions_phase, phase_cfg_0681, f)
        persistence = present / total if total > 0 else np.nan
        amp_0680 = dataset_mean_amp("IMG_0680_rot270", 1, f)
        reproducible = float(amp_0680 > 0.004)

        mode_score = (
            1.2 * float(amp_rank[idx])
            + 0.8 * float(persistence if np.isfinite(persistence) else 0.0)
            + 0.5 * reproducible
            + 0.6 * max(0.0, float(outgoing_best) if np.isfinite(outgoing_best) else 0.0)
            + 0.12 * outgoing_count
            - 0.35 * max(0.0, float(incoming_best) if np.isfinite(incoming_best) else 0.0)
        )

        rows.append(
            {
                "family_label": family["family_label"],
                "repr_hz": f,
                "repr_amp": float(family["repr_amp"]),
                "persistence": persistence,
                "region_mean_amp": region_mean_amp,
                "amp_rank": float(amp_rank[idx]),
                "amp_0680": amp_0680,
                "reproducible": reproducible,
                "incoming_best": incoming_best,
                "outgoing_best": outgoing_best,
                "outgoing_count": outgoing_count,
                "mode_score": mode_score,
            }
        )

    rows_sorted = sorted(rows, key=lambda row: float(row["mode_score"]), reverse=True)

    csv_path = OUTPUT_DIR / "calibrated_taxonomy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows_sorted)

    saved_paths = [
        plot_scorecard(writer, rows_sorted),
        plot_amp_vs_score(writer, rows_sorted),
    ]

    lines = []
    lines.append("0681 calibrated family taxonomy")
    lines.append("Scoring intent: reward amplitude, persistence, reproducibility, and outgoing couplings; penalize only positive incoming-coupling evidence.")
    lines.append("")
    lines.append("ranked families:")
    for row in rows_sorted:
        lines.append(
            f"{row['family_label']} | {row['repr_hz']:.3f} Hz | score={row['mode_score']:.4f} | "
            f"amp={row['repr_amp']:.4f} | persist={row['persistence']:.2f} | "
            f"incoming={row['incoming_best']:.5f} | outgoing={row['outgoing_best']:.5f} | "
            f"outgoing_count={row['outgoing_count']} | amp0680={row['amp_0680']:.5f}"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(csv_path))
    for path in saved_paths:
        lines.append(str(path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[saved] {csv_path.name}")
    print(f"[saved] summary.txt")


if __name__ == "__main__":
    main()
