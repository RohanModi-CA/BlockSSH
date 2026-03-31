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
from run_peak_bicoherence_analysis import PlotWriter, compute_complex_spectrogram_with_overlap, detect_regions, load_primary_processed


OUTPUT_DIR = SCRIPT_DIR / "prominence_output"
FAMILY_CSV = SCRIPT_DIR / "family_taxonomy_output" / "family_taxonomy.csv"


def load_families() -> list[dict[str, object]]:
    rows = []
    with FAMILY_CSV.open() as f:
        for row in csv.DictReader(f):
            row["repr_hz"] = float(row["repr_hz"])
            row["repr_amp"] = float(row["repr_amp"])
            rows.append(row)
    return rows


def prepare(dataset: str, bond_index: int):
    cfg = replace(BIC_CONFIG, dataset=dataset, bond_index=bond_index, null_trials=100)
    _, ds, processed = load_primary_processed(cfg)
    regions = detect_regions(ds, processed, cfg)
    return cfg, regions


def line_prominence(freq: np.ndarray, s_complex: np.ndarray, center_hz: float) -> tuple[float, float]:
    amp = np.mean(np.abs(s_complex), axis=1)
    center_mask = np.abs(freq - float(center_hz)) <= 0.12
    side_mask = ((np.abs(freq - float(center_hz)) >= 0.20) & (np.abs(freq - float(center_hz)) <= 0.55))
    if not np.any(center_mask) or np.sum(side_mask) < 2:
        return np.nan, np.nan
    line_amp = float(np.max(amp[center_mask]))
    side_med = float(np.median(amp[side_mask]))
    if side_med <= 0:
        return np.nan, np.nan
    contrast = line_amp / side_med
    contrast_db = 20.0 * np.log10(max(contrast, 1e-12))
    return contrast, contrast_db


def evaluate_dataset(families: list[dict[str, object]], dataset: str, bond_index: int) -> list[dict[str, object]]:
    cfg, regions = prepare(dataset, bond_index)
    rows = []
    for family in families:
        center = float(family["repr_hz"])
        region_contrast = []
        region_db = []
        for region in regions:
            processed = region["processed"]
            spec = compute_complex_spectrogram_with_overlap(
                processed.y,
                processed.Fs,
                cfg.welch_len_s,
                cfg.welch_overlap_fraction,
            )
            if spec is None:
                continue
            freq, _, s_complex = spec
            contrast, contrast_db = line_prominence(freq, s_complex, center)
            if np.isfinite(contrast):
                region_contrast.append(contrast)
                region_db.append(contrast_db)
        rows.append(
            {
                "family_label": family["family_label"],
                "repr_hz": center,
                "repr_amp": float(family["repr_amp"]),
                "contrast_mean": float(np.mean(region_contrast)) if region_contrast else np.nan,
                "contrast_db_mean": float(np.mean(region_db)) if region_db else np.nan,
                "contrast_db_std": float(np.std(region_db)) if region_db else np.nan,
                "n_regions": len(region_db),
            }
        )
    return rows


def plot_prominence(writer: PlotWriter, merged_rows: list[dict[str, object]]) -> Path:
    x = np.asarray([float(row["contrast_db_0681"]) for row in merged_rows], dtype=float)
    y = np.asarray([float(row["contrast_db_0680"]) for row in merged_rows], dtype=float)
    s = np.asarray([35.0 + 180.0 * float(row["repr_amp"]) for row in merged_rows], dtype=float)
    c = np.asarray([float(row["repr_hz"]) for row in merged_rows], dtype=float)
    labels = [str(row["family_label"]) for row in merged_rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    sc = ax.scatter(x, y, s=s, c=c, cmap="turbo", edgecolor="black", linewidth=0.4, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="frequency (Hz)")
    for xi, yi, text in zip(x, y, labels):
        ax.text(xi, yi, text, fontsize=8, ha="left", va="bottom")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("mean local prominence in 0681 (dB)")
    ax.set_ylabel("mean local prominence in 0680 (dB)")
    ax.set_title("Local spectral prominence of family lines")
    ax.grid(alpha=0.25)
    return writer.save(fig, "family_prominence")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    families = load_families()
    rows_0681 = evaluate_dataset(families, "IMG_0681_rot270", 0)
    rows_0680 = evaluate_dataset(families, "IMG_0680_rot270", 1)
    by_0680 = {row["family_label"]: row for row in rows_0680}

    merged_rows = []
    for row in rows_0681:
        peer = by_0680[row["family_label"]]
        merged_rows.append(
            {
                "family_label": row["family_label"],
                "repr_hz": row["repr_hz"],
                "repr_amp": row["repr_amp"],
                "contrast_db_0681": row["contrast_db_mean"],
                "contrast_db_0680": peer["contrast_db_mean"],
                "contrast_mean_0681": row["contrast_mean"],
                "contrast_mean_0680": peer["contrast_mean"],
                "std_db_0681": row["contrast_db_std"],
                "std_db_0680": peer["contrast_db_std"],
            }
        )
    merged_rows.sort(key=lambda row: float(np.nan_to_num(row["contrast_db_0681"], nan=-999.0)), reverse=True)

    csv_path = OUTPUT_DIR / "family_prominence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(merged_rows)

    saved = [plot_prominence(writer, merged_rows)]
    lines = []
    lines.append("Family local spectral prominence")
    lines.append("Higher dB means the line stands out more strongly above its nearby frequency neighborhood.")
    lines.append("")
    for row in merged_rows:
        lines.append(
            f"{row['family_label']} | {row['repr_hz']:.3f} Hz | amp={row['repr_amp']:.4f} | "
            f"0681={row['contrast_db_0681']:.2f} dB | 0680={row['contrast_db_0680']:.2f} dB"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(csv_path))
    for path in saved:
        lines.append(str(path))
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("[saved] family_prominence.csv")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
