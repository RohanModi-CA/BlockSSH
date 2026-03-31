from __future__ import annotations

import csv
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

from run_peak_bicoherence_analysis import (
    CONFIG as BIC_CONFIG,
    PlotWriter,
    Triad,
    detect_regions,
    evaluate_triad,
    load_primary_processed,
)
from run_peak_coherence_analysis import (
    CONFIG as PHASE_CONFIG,
    HarmonicHypothesis,
    build_regions,
    detect_peak_times,
    evaluate_hypothesis,
    load_primary_dataset,
    summarize_line_tracks,
)


@dataclass(frozen=True)
class Config:
    peak_csv: str = "/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/configs/peaks/0681ROT270X.csv"
    dataset: str = "IMG_0681_rot270"
    bond_index: int = 0
    min_freq_hz: float = 3.0
    family_merge_hz: float = 0.10
    pair_tol_hz: float = 0.12
    harmonic_tol_hz: float = 0.18
    max_harmonic_order: int = 5
    triad_null_trials: int = 200
    phase_null_trials: int = 500
    child_like_threshold: float = 0.04


CONFIG = Config()
OUTPUT_DIR = SCRIPT_DIR / "family_taxonomy_output"


def load_peak_list(path: str) -> list[float]:
    with Path(path).open() as f:
        rows = list(csv.reader(f))
    return [float(cell) for cell in rows[0] if str(cell).strip()]


def build_peak_track_rows(peaks: list[float], regions_phase, phase_cfg) -> list[dict[str, float]]:
    rows = []
    for peak in peaks:
        summary = summarize_line_tracks(regions_phase, float(peak), phase_cfg)
        rows.append(
            {
                "peak_hz": float(peak),
                "tracked_mean_hz": float(summary["global_mean_freq_hz"]),
                "tracked_std_hz": float(summary["global_std_freq_hz"]),
                "tracked_amp": float(summary["global_mean_amp"]),
            }
        )
    rows.sort(key=lambda row: row["tracked_mean_hz"])
    return rows


def cluster_peak_rows(rows: list[dict[str, float]], merge_hz: float) -> list[dict[str, object]]:
    families: list[list[dict[str, float]]] = []
    for row in rows:
        if not families:
            families.append([row])
            continue
        last = families[-1]
        last_mean = float(np.mean([item["tracked_mean_hz"] for item in last]))
        if abs(float(row["tracked_mean_hz"]) - last_mean) <= merge_hz:
            last.append(row)
        else:
            families.append([row])

    out = []
    for idx, fam in enumerate(families, start=1):
        raw_peaks = np.asarray([item["peak_hz"] for item in fam], dtype=float)
        tracked_means = np.asarray([item["tracked_mean_hz"] for item in fam], dtype=float)
        tracked_stds = np.asarray([item["tracked_std_hz"] for item in fam], dtype=float)
        tracked_amp = np.asarray([item["tracked_amp"] for item in fam], dtype=float)
        out.append(
            {
                "family_id": idx,
                "family_label": f"F{idx:02d}",
                "raw_peaks": raw_peaks,
                "repr_hz": float(np.mean(tracked_means)),
                "repr_std_hz": float(np.mean(tracked_stds)),
                "repr_amp": float(np.max(tracked_amp)),
            }
        )
    return out


def plot_family_overview(writer: PlotWriter, families: list[dict[str, object]]) -> Path:
    freq = np.asarray([float(row["repr_hz"]) for row in families], dtype=float)
    amp = np.asarray([float(row["repr_amp"]) for row in families], dtype=float)
    size = np.asarray([len(row["raw_peaks"]) for row in families], dtype=float)
    label = [str(row["family_label"]) for row in families]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.scatter(freq, amp, s=40 + 35 * size, color="C0", edgecolor="black", linewidth=0.4, alpha=0.8)
    for x, y, text in zip(freq, amp, label):
        ax.text(x, y, text, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("family representative frequency (Hz)")
    ax.set_ylabel("max tracked amplitude")
    ax.set_title("0681 >3 Hz frequency families")
    ax.grid(alpha=0.25)
    return writer.save(fig, "family_overview")


def plot_family_taxonomy(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    x = np.asarray([float(row["child_evidence"]) for row in rows], dtype=float)
    y = np.asarray([float(row["outgoing_count"]) for row in rows], dtype=float)
    c = np.asarray([float(row["repr_hz"]) for row in rows], dtype=float)
    labels = [str(row["family_label"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    sc = ax.scatter(x, y, c=c, cmap="turbo", s=85, edgecolor="black", linewidth=0.4, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="representative frequency (Hz)")
    for xi, yi, text in zip(x, y, labels):
        ax.text(xi, yi, text, fontsize=8, ha="left", va="bottom")
    ax.axvline(CONFIG.child_like_threshold, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("incoming child evidence")
    ax.set_ylabel("outgoing significant pair count")
    ax.set_title("0681 family-level taxonomy")
    ax.grid(alpha=0.25)
    return writer.save(fig, "family_taxonomy")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    raw_peaks = [x for x in load_peak_list(CONFIG.peak_csv) if float(x) >= CONFIG.min_freq_hz]

    phase_cfg = replace(PHASE_CONFIG, dataset=CONFIG.dataset, primary_bond_index=CONFIG.bond_index, null_trials=CONFIG.phase_null_trials)
    _, ds_phase, processed_phase = load_primary_dataset(phase_cfg.segmentation_mode, phase_cfg)
    t_global, peak_times = detect_peak_times(processed_phase, phase_cfg)
    regions_phase = build_regions(ds_phase, phase_cfg.primary_bond_index, t_global, peak_times, phase_cfg)

    bic_cfg = replace(BIC_CONFIG, dataset=CONFIG.dataset, bond_index=CONFIG.bond_index, null_trials=CONFIG.triad_null_trials)
    _, ds_bic, processed_bic = load_primary_processed(bic_cfg)
    regions_bic = detect_regions(ds_bic, processed_bic, bic_cfg)

    peak_rows = build_peak_track_rows(raw_peaks, regions_phase, phase_cfg)
    families = cluster_peak_rows(peak_rows, CONFIG.family_merge_hz)

    pair_cache: dict[tuple[int, int, int], dict[str, object]] = {}
    harmonic_cache: dict[tuple[int, int, int], dict[str, object]] = {}

    for child_idx, child in enumerate(families):
        f3 = float(child["repr_hz"])
        for i, left in enumerate(families[:child_idx]):
            f1 = float(left["repr_hz"])
            for j, right in enumerate(families[i:child_idx], start=i):
                f2 = float(right["repr_hz"])
                if abs((f1 + f2) - f3) <= CONFIG.pair_tol_hz:
                    key = (i, j, child_idx)
                    pair_cache[key] = evaluate_triad(
                        regions_bic,
                        Triad(label="", f1_hz=f1, f2_hz=f2, f3_hz=f3),
                        bic_cfg,
                    )

        for parent_idx, parent in enumerate(families[:child_idx]):
            f1 = float(parent["repr_hz"])
            for order in range(2, CONFIG.max_harmonic_order + 1):
                if abs(order * f1 - f3) <= CONFIG.harmonic_tol_hz:
                    key = (parent_idx, order, child_idx)
                    harmonic_cache[key] = evaluate_hypothesis(
                        regions_phase,
                        HarmonicHypothesis(label="", parent_hz=f1, target_hz=f3, order=order),
                        phase_cfg,
                    )

    outgoing_count = {idx: 0 for idx in range(len(families))}
    for (i, j, _), result in pair_cache.items():
        excess = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
        if excess > 0:
            outgoing_count[i] += 1
            outgoing_count[j] += 1

    family_rows = []
    for child_idx, family in enumerate(families):
        pair_candidates = []
        for (i, j, k), result in pair_cache.items():
            if k != child_idx:
                continue
            excess = float(result["observed_bicoherence"]) - float(result["null_q95_bicoherence"])
            pair_candidates.append((excess, i, j, result))
        pair_candidates.sort(key=lambda item: item[0], reverse=True)

        harmonic_candidates = []
        for (i, order, k), result in harmonic_cache.items():
            if k != child_idx:
                continue
            excess = float(result["observed_plv"]) - float(result["null_q95_plv"])
            harmonic_candidates.append((excess, i, order, result))
        harmonic_candidates.sort(key=lambda item: item[0], reverse=True)

        best_pair_excess = float(pair_candidates[0][0]) if pair_candidates else np.nan
        best_harmonic_excess = float(harmonic_candidates[0][0]) if harmonic_candidates else np.nan
        child_evidence = np.nanmax([best_pair_excess, best_harmonic_excess])
        if not np.isfinite(child_evidence):
            child_evidence = np.nan

        best_relation = ""
        if pair_candidates and (not harmonic_candidates or best_pair_excess >= best_harmonic_excess):
            best_relation = f"{families[pair_candidates[0][1]]['family_label']} + {families[pair_candidates[0][2]]['family_label']}"
        elif harmonic_candidates:
            best_relation = f"{harmonic_candidates[0][2]} x {families[harmonic_candidates[0][1]]['family_label']}"

        tag = "child-like" if np.isfinite(child_evidence) and child_evidence > CONFIG.child_like_threshold else "parent-like-or-unresolved"

        family_rows.append(
            {
                "family_id": int(family["family_id"]),
                "family_label": str(family["family_label"]),
                "repr_hz": float(family["repr_hz"]),
                "repr_std_hz": float(family["repr_std_hz"]),
                "repr_amp": float(family["repr_amp"]),
                "raw_peaks": " ".join(f"{x:.3f}" for x in family["raw_peaks"]),
                "n_raw_peaks": int(len(family["raw_peaks"])),
                "best_pair_excess": best_pair_excess,
                "best_harmonic_excess": best_harmonic_excess,
                "child_evidence": child_evidence,
                "best_relation": best_relation,
                "outgoing_count": int(outgoing_count[child_idx]),
                "tag": tag,
            }
        )

    csv_path = OUTPUT_DIR / "family_taxonomy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=list(family_rows[0].keys()),
        )
        writer_csv.writeheader()
        writer_csv.writerows(family_rows)

    saved_paths = [
        plot_family_overview(writer, families),
        plot_family_taxonomy(writer, family_rows),
    ]

    summary_lines = []
    summary_lines.append(f"peak_csv: {CONFIG.peak_csv}")
    summary_lines.append(f"dataset: {CONFIG.dataset}")
    summary_lines.append(f"bond_index: {CONFIG.bond_index}")
    summary_lines.append(f"min_freq_hz: {CONFIG.min_freq_hz}")
    summary_lines.append(f"n_raw_peaks_used: {len(raw_peaks)}")
    summary_lines.append(f"n_families: {len(families)}")
    summary_lines.append(f"n_pair_candidates_evaluated: {len(pair_cache)}")
    summary_lines.append(f"n_harmonic_candidates_evaluated: {len(harmonic_cache)}")
    summary_lines.append("")
    summary_lines.append("family table:")
    for row in sorted(family_rows, key=lambda item: item["repr_hz"]):
        summary_lines.append(
            f"{row['family_label']} | repr={row['repr_hz']:.3f} Hz | child_evidence={row['child_evidence']:.5f} | "
            f"outgoing={row['outgoing_count']} | relation={row['best_relation']} | tag={row['tag']} | peaks={row['raw_peaks']}"
        )
    summary_lines.append("")
    summary_lines.append("high child-evidence families:")
    for row in sorted(family_rows, key=lambda item: item["child_evidence"], reverse=True)[:12]:
        summary_lines.append(
            f"{row['family_label']} | repr={row['repr_hz']:.3f} Hz | child_evidence={row['child_evidence']:.5f} | "
            f"relation={row['best_relation']} | outgoing={row['outgoing_count']}"
        )
    summary_lines.append("")
    summary_lines.append("low child-evidence families:")
    valid_rows = [row for row in family_rows if np.isfinite(row["child_evidence"])]
    for row in sorted(valid_rows, key=lambda item: item["child_evidence"])[:12]:
        summary_lines.append(
            f"{row['family_label']} | repr={row['repr_hz']:.3f} Hz | child_evidence={row['child_evidence']:.5f} | "
            f"relation={row['best_relation']} | outgoing={row['outgoing_count']}"
        )
    summary_lines.append("")
    summary_lines.append("saved_files:")
    summary_lines.append(str(csv_path))
    for path in saved_paths:
        summary_lines.append(str(path))

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"[saved] {csv_path.name}")
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
