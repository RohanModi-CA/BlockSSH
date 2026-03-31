from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "band_mapping_output"
PEAK_CSV = Path("/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/configs/peaks/0681ROT270X.csv")
FAMILY_CSV = SCRIPT_DIR / "family_taxonomy_output" / "family_taxonomy.csv"
SYNTHESIS_CSV = SCRIPT_DIR / "synthesis_output" / "family_synthesis.csv"


@dataclass(frozen=True)
class Band:
    label: str
    lo_hz: float
    hi_hz: float
    status: str
    note: str


BANDS: tuple[Band, ...] = (
    Band("B12p6", 12.50, 12.75, "nonfundamental-leaning", "strong incoming family edge around 12.61"),
    Band("B13p0", 12.95, 13.28, "nonfundamental-leaning", "strong incoming family edges around 13.05-13.21"),
    Band("B16p0", 15.95, 16.20, "nonfundamental-leaning", "child band from 8.106+8.106 and 7.956+8.106"),
    Band("B16p6", 16.35, 16.95, "mixed-coupled", "broad coupled band containing 16.60; not a clean follower proof"),
    Band("B18ish", 17.90, 18.45, "mixed-coupled", "18-ish child band linked to 8.95+8.95; exact peak shifts by dataset"),
    Band("B22p8", 22.75, 23.20, "mixed-coupled", "broad 22.8-23.2 band seen in 16.60+6.37 scan"),
)


def load_peak_list() -> list[float]:
    with PEAK_CSV.open() as f:
        row = next(csv.reader(f))
    return [float(cell) for cell in row if cell.strip()]


def load_families() -> list[dict[str, str]]:
    with FAMILY_CSV.open() as f:
        return list(csv.DictReader(f))


def load_synthesis() -> dict[str, dict[str, str]]:
    with SYNTHESIS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return {row["family_label"]: row for row in rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    peaks = [peak for peak in load_peak_list() if peak >= 3.0]
    families = load_families()
    synthesis = load_synthesis()

    family_rows = []
    for family in families:
        raw_peaks = [float(x) for x in family["raw_peaks"].split()]
        syn = synthesis[family["family_label"]]
        for peak in raw_peaks:
            band_hits = [band for band in BANDS if band.lo_hz <= peak <= band.hi_hz]
            family_rows.append(
                {
                    "raw_peak_hz": peak,
                    "family_label": family["family_label"],
                    "family_repr_hz": float(family["repr_hz"]),
                    "family_class": syn["class"],
                    "family_incoming": float(syn["best_incoming_edge"]) if syn["best_incoming_edge"] not in ("", "nan") else "",
                    "family_outgoing": float(syn["best_outgoing_edge"]) if syn["best_outgoing_edge"] not in ("", "nan") else "",
                    "family_prominence_db_0681": float(syn["contrast_db_0681"]),
                    "band_labels": " ".join(band.label for band in band_hits),
                    "band_status": " ".join(band.status for band in band_hits),
                    "band_notes": " | ".join(band.note for band in band_hits),
                }
            )

    family_rows.sort(key=lambda row: float(row["raw_peak_hz"]))

    csv_path = OUTPUT_DIR / "peak_band_mapping.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_rows)

    lines = []
    lines.append("Peak-to-family band mapping for 0681ROT270X > 3 Hz")
    lines.append("")
    for row in family_rows:
        lines.append(
            f"{row['raw_peak_hz']:.3f} Hz | {row['family_label']} @ {row['family_repr_hz']:.3f} Hz | "
            f"class={row['family_class']} | bands={row['band_labels'] or '-'} | notes={row['band_notes'] or '-'}"
        )
    lines.append("")
    lines.append("bands:")
    for band in BANDS:
        lines.append(f"{band.label} | [{band.lo_hz:.2f}, {band.hi_hz:.2f}] Hz | {band.status} | {band.note}")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n")
    print("[saved] peak_band_mapping.csv")
    print("[saved] summary.txt")


if __name__ == "__main__":
    main()
