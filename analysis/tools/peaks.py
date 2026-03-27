from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DEFAULT_PEAKS_DIR = Path(__file__).resolve().parents[1] / "configs" / "peaks"


def resolve_peaks_csv(name_or_path: str | Path, peaks_dir: str | Path | None = None) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path

    root = Path(peaks_dir) if peaks_dir is not None else DEFAULT_PEAKS_DIR
    text = str(name_or_path).strip()
    candidates = [
        root / text,
        root / f"{text}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Peaks file not found for '{name_or_path}'. Tried: {tried}")


def load_peaks_csv(path: str | Path) -> list[float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peaks file not found: {path}")

    peaks: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                try:
                    val = float(text)
                except ValueError:
                    continue
                if val > 0:
                    peaks.append(val)

    if not peaks:
        raise ValueError("No valid positive floating point peaks found in CSV")

    return peaks


def assert_peaks_strictly_increasing(peaks: list[float] | np.ndarray) -> np.ndarray:
    peaks_arr = np.asarray(peaks, dtype=float)
    if peaks_arr.ndim != 1 or peaks_arr.size == 0:
        raise ValueError("Peaks must be a non-empty 1D sequence")
    if not np.all(np.isfinite(peaks_arr)):
        raise ValueError("Peaks must all be finite")
    if np.any(peaks_arr <= 0):
        raise ValueError("Peaks must all be positive")
    diffs = np.diff(peaks_arr)
    if np.any(diffs <= 0):
        raise ValueError("CSV peaks must be strictly increasing from lowest to highest")
    return peaks_arr


def select_active_peak_indices(
    peaks: list[float],
    *,
    disableplot: list[int] | None = None,
    onlyenableplots: list[int] | None = None,
) -> list[int]:
    all_indices = set(range(len(peaks)))
    if onlyenableplots is not None:
        active = set(int(i) for i in onlyenableplots)
    else:
        active = set(all_indices)

    active -= set(int(i) for i in (disableplot or []))
    active = {idx for idx in active if idx in all_indices}
    return sorted(active, key=lambda idx: peaks[idx], reverse=True)
