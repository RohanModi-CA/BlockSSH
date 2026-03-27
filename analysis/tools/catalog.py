from __future__ import annotations

import json
from pathlib import Path

from .io import get_default_track_data_root


DEFAULT_GROUPS_DIR = Path(__file__).resolve().parents[1] / "configs" / "groups"
DEFAULT_PEAKS_DIR = Path(__file__).resolve().parents[1] / "configs" / "peaks"


def list_base_datasets(track_data_root: str | Path | None = None) -> list[str]:
    root = Path(track_data_root) if track_data_root is not None else get_default_track_data_root()
    if not root.is_dir():
        return []

    names: set[str] = set()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name.endswith("_BACKUP"):
            continue
        stripped = False
        for suffix in ("_x", "_y", "_a", "_fx", "_fy", "_fa", "_area"):
            if path.name.endswith(suffix):
                names.add(path.name[: -len(suffix)])
                stripped = True
                break
        if stripped:
            continue
        if (path / "components").is_dir() or (path / "track2_permanence.msgpack").exists():
            names.add(path.name)
            continue

    return sorted(names)


def list_group_names(groups_dir: str | Path | None = None) -> list[str]:
    root = Path(groups_dir) if groups_dir is not None else DEFAULT_GROUPS_DIR
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.iterdir() if path.is_file() and path.suffix == ".json")


def list_peak_names(peaks_dir: str | Path | None = None) -> list[str]:
    root = Path(peaks_dir) if peaks_dir is not None else DEFAULT_PEAKS_DIR
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.iterdir() if path.is_file() and path.suffix == ".csv")


def save_group(path: str | Path, *, name: str, datasets: list[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": str(name),
        "version": 1,
        "datasets": [str(dataset) for dataset in datasets],
    }
    with output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return output
