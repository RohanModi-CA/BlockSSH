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


OUTPUT_DIR = SCRIPT_DIR / "edge_survey_output"
FAMILY_CSV = SCRIPT_DIR / "family_taxonomy_output" / "family_taxonomy.csv"


def load_families() -> list[dict[str, object]]:
    rows = []
    with FAMILY_CSV.open() as f:
        for row in csv.DictReader(f):
            row["family_id"] = int(row["family_id"])
            row["repr_hz"] = float(row["repr_hz"])
            row["repr_amp"] = float(row["repr_amp"])
            row["outgoing_count"] = int(row["outgoing_count"])
            rows.append(row)
    return rows


def prepare_regions(dataset: str, bond_index: int, null_trials: int):
    cfg = replace(BIC_CONFIG, dataset=dataset, bond_index=bond_index, null_trials=null_trials)
    _, ds, processed = load_primary_processed(cfg)
    regions = detect_regions(ds, processed, cfg)
    return cfg, regions


def evaluate_edges(
    families: list[dict[str, object]],
    regions_0681,
    cfg_0681,
    regions_0680,
    cfg_0680,
    pair_tol_hz: float = 0.12,
) -> list[dict[str, object]]:
    edges = []
    for child_idx, child in enumerate(families):
        f3 = float(child["repr_hz"])
        lowers = families[:child_idx]
        for i, left in enumerate(lowers):
            f1 = float(left["repr_hz"])
            for right in lowers[i:]:
                f2 = float(right["repr_hz"])
                if abs((f1 + f2) - f3) > pair_tol_hz:
                    continue
                triad = Triad(
                    label=f"{left['family_label']} + {right['family_label']} -> {child['family_label']}",
                    f1_hz=f1,
                    f2_hz=f2,
                    f3_hz=f3,
                )
                r1 = evaluate_triad(regions_0681, triad, cfg_0681)
                r0 = evaluate_triad(regions_0680, triad, cfg_0680)
                ex1 = float(r1["observed_bicoherence"]) - float(r1["null_q95_bicoherence"])
                ex0 = float(r0["observed_bicoherence"]) - float(r0["null_q95_bicoherence"])
                reproducible = float(ex1 > 0.0 and ex0 > 0.0)
                child_amp = float(child["repr_amp"])
                parent_amp_geom = float(np.sqrt(float(left["repr_amp"]) * float(right["repr_amp"])))
                edges.append(
                    {
                        "parent1": str(left["family_label"]),
                        "parent2": str(right["family_label"]),
                        "child": str(child["family_label"]),
                        "f1_hz": f1,
                        "f2_hz": f2,
                        "f3_hz": f3,
                        "amp_child": child_amp,
                        "amp_parent_geom": parent_amp_geom,
                        "child_parent_amp_ratio": child_amp / parent_amp_geom if parent_amp_geom > 0 else np.nan,
                        "bic_0681": float(r1["observed_bicoherence"]),
                        "null95_0681": float(r1["null_q95_bicoherence"]),
                        "excess_0681": ex1,
                        "bic_0680": float(r0["observed_bicoherence"]),
                        "null95_0680": float(r0["null_q95_bicoherence"]),
                        "excess_0680": ex0,
                        "ampcorr_0681": float(r1["amp_prod_corr"]),
                        "ampcorr_0680": float(r0["amp_prod_corr"]),
                        "nwin_0681": int(r1["n_windows_total"]),
                        "nwin_0680": int(r0["n_windows_total"]),
                        "reproducible_positive": reproducible,
                        "combined_edge_score": ex1 + 0.6 * max(0.0, ex0),
                    }
                )
    edges.sort(key=lambda row: float(row["combined_edge_score"]), reverse=True)
    return edges


def summarize_families(
    families: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_child: dict[str, list[dict[str, object]]] = {}
    by_parent: dict[str, list[dict[str, object]]] = {}
    for edge in edges:
        by_child.setdefault(str(edge["child"]), []).append(edge)
        by_parent.setdefault(str(edge["parent1"]), []).append(edge)
        by_parent.setdefault(str(edge["parent2"]), []).append(edge)

    rows = []
    for family in families:
        label = str(family["family_label"])
        incoming = by_child.get(label, [])
        outgoing = by_parent.get(label, [])
        best_incoming = max((float(edge["combined_edge_score"]) for edge in incoming), default=np.nan)
        best_outgoing = max((float(edge["combined_edge_score"]) for edge in outgoing), default=np.nan)
        incoming_repro = int(sum(int(edge["reproducible_positive"]) for edge in incoming))
        outgoing_repro = int(sum(int(edge["reproducible_positive"]) for edge in outgoing))
        rows.append(
            {
                "family_label": label,
                "repr_hz": float(family["repr_hz"]),
                "repr_amp": float(family["repr_amp"]),
                "best_incoming_edge": best_incoming,
                "best_outgoing_edge": best_outgoing,
                "incoming_reproducible_count": incoming_repro,
                "outgoing_reproducible_count": outgoing_repro,
                "incoming_edge_count": len(incoming),
                "outgoing_edge_count": len(outgoing),
            }
        )
    rows.sort(key=lambda row: (float(row["repr_amp"]), float(np.nan_to_num(row["best_outgoing_edge"], nan=-1.0))), reverse=True)
    return rows


def plot_top_edges(writer: PlotWriter, edges: list[dict[str, object]], top_n: int = 16) -> Path:
    top = edges[:top_n]
    labels = [f"{row['parent1']}+{row['parent2']}->{row['child']}" for row in top]
    ex1 = np.asarray([float(row["excess_0681"]) for row in top], dtype=float)
    ex0 = np.asarray([float(row["excess_0680"]) for row in top], dtype=float)
    x = np.arange(len(top), dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.bar(x - 0.18, ex1, width=0.36, color="C0", label="0681 excess over null95")
    ax.bar(x + 0.18, ex0, width=0.36, color="C1", label="0680 excess over null95")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("bicoherence excess")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    ax.set_title("Strongest family-level triad edges")
    return writer.save(fig, "top_family_edges")


def plot_family_balance(writer: PlotWriter, rows: list[dict[str, object]]) -> Path:
    x = np.asarray([float(row["best_incoming_edge"]) if np.isfinite(float(row["best_incoming_edge"])) else -0.03 for row in rows], dtype=float)
    y = np.asarray([float(row["best_outgoing_edge"]) if np.isfinite(float(row["best_outgoing_edge"])) else -0.03 for row in rows], dtype=float)
    s = np.asarray([35.0 + 210.0 * float(row["repr_amp"]) for row in rows], dtype=float)
    c = np.asarray([float(row["repr_hz"]) for row in rows], dtype=float)
    labels = [str(row["family_label"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    sc = ax.scatter(x, y, s=s, c=c, cmap="turbo", edgecolor="black", linewidth=0.4, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="frequency (Hz)")
    for xi, yi, text in zip(x, y, labels):
        ax.text(xi, yi, text, fontsize=8, ha="left", va="bottom")
    ax.axvline(0.08, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("best incoming edge score")
    ax.set_ylabel("best outgoing edge score")
    ax.set_title("Family parent/child balance from cross-dataset edges")
    ax.grid(alpha=0.25)
    return writer.save(fig, "family_edge_balance")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    families = load_families()
    cfg_0681, regions_0681 = prepare_regions("IMG_0681_rot270", 0, 250)
    cfg_0680, regions_0680 = prepare_regions("IMG_0680_rot270", 1, 200)

    edges = evaluate_edges(families, regions_0681, cfg_0681, regions_0680, cfg_0680)
    family_rows = summarize_families(families, edges)

    edge_csv = OUTPUT_DIR / "family_edges.csv"
    with edge_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(edges[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(edges)

    family_csv = OUTPUT_DIR / "family_balance.csv"
    with family_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(family_rows)

    saved = [
        plot_top_edges(writer, edges),
        plot_family_balance(writer, family_rows),
    ]

    lines = []
    lines.append("Cross-dataset family edge survey")
    lines.append("Datasets: 0681 bond0 and 0680 bond1")
    lines.append("Edge score = excess_0681 + 0.6 * max(0, excess_0680)")
    lines.append("")
    lines.append("top edges:")
    for row in edges[:20]:
        lines.append(
            f"{row['parent1']} + {row['parent2']} -> {row['child']} | "
            f"freqs=({row['f1_hz']:.3f}, {row['f2_hz']:.3f}, {row['f3_hz']:.3f}) | "
            f"ex0681={row['excess_0681']:.5f} | ex0680={row['excess_0680']:.5f} | "
            f"child/parent_amp={row['child_parent_amp_ratio']:.3f} | "
            f"repro={int(row['reproducible_positive'])}"
        )
    lines.append("")
    lines.append("family balance:")
    for row in family_rows:
        lines.append(
            f"{row['family_label']} | {row['repr_hz']:.3f} Hz | amp={row['repr_amp']:.4f} | "
            f"in_best={row['best_incoming_edge']:.5f} | out_best={row['best_outgoing_edge']:.5f} | "
            f"in_repro={row['incoming_reproducible_count']} | out_repro={row['outgoing_reproducible_count']} | "
            f"in_edges={row['incoming_edge_count']} | out_edges={row['outgoing_edge_count']}"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(edge_csv))
    lines.append(str(family_csv))
    for path in saved:
        lines.append(str(path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("[saved] family_edges.csv")
    print("[saved] family_balance.csv")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
