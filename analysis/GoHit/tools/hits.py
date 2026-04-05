from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HitCatalog:
    dataset: str
    source_component: str
    hit_times_s: tuple[float, ...]
    detector: dict[str, object]


@dataclass(frozen=True)
class HitRegion:
    index: int
    start_s: float
    stop_s: float
    mode: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def catalog_dir() -> Path:
    return _repo_root() / "analysis" / "GoHit" / "out" / "hit_catalogs"


def default_catalog_path(dataset: str) -> Path:
    return catalog_dir() / f"{dataset}__hits.json"


def default_catalog_csv_path(dataset: str) -> Path:
    return catalog_dir() / f"{dataset}__hits.csv"


def legacy_hits_csv_path(dataset: str, component: str) -> Path:
    return _repo_root() / "analysis" / "NL" / "out" / f"{dataset}_comparison_purecomoving" / f"{dataset}__{component}__prototype_hits.csv"


def save_hit_catalog(catalog: HitCatalog, path: str | Path | None = None) -> Path:
    out_path = Path(path) if path is not None else default_catalog_path(catalog.dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(catalog)
    payload["hit_times_s"] = [float(value) for value in catalog.hit_times_s]
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = default_catalog_csv_path(catalog.dataset)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "source_component", "hit_index", "time_s"])
        for idx, hit_time in enumerate(catalog.hit_times_s, start=1):
            writer.writerow([catalog.dataset, catalog.source_component, idx, f"{float(hit_time):.9f}"])
    return out_path


def load_hit_catalog(path: str | Path) -> HitCatalog:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HitCatalog(
        dataset=str(payload["dataset"]),
        source_component=str(payload["source_component"]),
        hit_times_s=tuple(float(value) for value in payload.get("hit_times_s", [])),
        detector=dict(payload.get("detector", {})),
    )


def load_catalog_if_available(dataset: str) -> HitCatalog | None:
    path = default_catalog_path(dataset)
    if path.is_file():
        return load_hit_catalog(path)
    return None


def load_legacy_hit_times(dataset: str, component: str) -> list[float]:
    path = legacy_hits_csv_path(dataset, component)
    if not path.is_file():
        return []
    times: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            times.append(float(row["time_s"]))
    return times


def _final_regions(bounds: list[tuple[float, float]], *, mode: str) -> list[HitRegion]:
    regions: list[HitRegion] = []
    for idx, (start_s, stop_s) in enumerate(bounds, start=1):
        if stop_s <= start_s:
            continue
        regions.append(HitRegion(index=idx, start_s=float(start_s), stop_s=float(stop_s), mode=mode))
    return regions


def build_interhit_regions(
    hit_times_s: list[float] | tuple[float, ...] | np.ndarray,
    *,
    t_stop_s: float,
    exclude_after_s: float,
    exclude_before_s: float,
    min_duration_s: float = 0.5,
) -> list[HitRegion]:
    hit_times = np.sort(np.asarray(hit_times_s, dtype=float))
    bounds: list[tuple[float, float]] = []
    for idx in range(hit_times.size):
        start_s = float(hit_times[idx] + exclude_after_s)
        stop_s = float(hit_times[idx + 1] - exclude_before_s) if idx < hit_times.size - 1 else float(t_stop_s - 1.0)
        if stop_s > start_s + float(min_duration_s):
            bounds.append((start_s, stop_s))
    return _final_regions(bounds, mode="interhit")


def build_posthit_regions(
    hit_times_s: list[float] | tuple[float, ...] | np.ndarray,
    *,
    t_stop_s: float,
    window_s: float,
    min_duration_s: float = 0.5,
) -> list[HitRegion]:
    hit_times = np.sort(np.asarray(hit_times_s, dtype=float))
    bounds: list[tuple[float, float]] = []
    for hit_time in hit_times:
        start_s = float(hit_time)
        stop_s = min(float(t_stop_s), start_s + float(window_s))
        if stop_s > start_s + float(min_duration_s):
            bounds.append((start_s, stop_s))
    return _final_regions(bounds, mode="posthit")


def summarize_catalog(catalog: HitCatalog) -> str:
    return (
        f"dataset={catalog.dataset} source_component={catalog.source_component} "
        f"hits={len(catalog.hit_times_s)} catalog={default_catalog_path(catalog.dataset)}"
    )
