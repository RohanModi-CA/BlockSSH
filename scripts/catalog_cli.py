#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.tools.catalog import (
    DEFAULT_GROUPS_DIR,
    DEFAULT_PEAKS_DIR,
    list_base_datasets,
    list_group_names,
    list_peak_names,
)


ROOT = Path(__file__).resolve().parents[1]
TRACK_DATA_ROOT = ROOT / "track" / "data"
TRACK_VIDEOS_ROOT = ROOT / "track" / "Videos"
SPECTRASAVE_ROOT = ROOT / "analysis" / "spectrasave"


def list_video_stems() -> list[str]:
    if not TRACK_VIDEOS_ROOT.is_dir():
        return []
    stems: set[str] = set()
    for path in TRACK_VIDEOS_ROOT.iterdir():
        if path.is_file():
            stems.add(path.stem)
    return sorted(stems)


def list_prepare_targets() -> list[str]:
    return sorted(set(list_base_datasets(track_data_root=TRACK_DATA_ROOT)) | set(list_video_stems()))


def list_spectrasaves() -> list[str]:
    if not SPECTRASAVE_ROOT.is_dir():
        return []
    return sorted(path.name for path in SPECTRASAVE_ROOT.iterdir() if path.is_file())


def list_components(dataset: str) -> list[str]:
    dataset_root = TRACK_DATA_ROOT / str(dataset)
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        components = payload.get("components", {})
        if isinstance(components, dict):
            return sorted(str(component) for component in components.keys())

    components_dir = dataset_root / "components"
    if components_dir.is_dir():
        return sorted(path.name for path in components_dir.iterdir() if path.is_dir())

    out: list[str] = []
    for component in ("x", "y", "a", "fx", "fy", "fa", "area"):
        if (TRACK_DATA_ROOT / f"{dataset}_{component}" / "track2_permanence.msgpack").is_file():
            out.append(component)
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print project catalog values for shell completion.")
    parser.add_argument(
        "kind",
        choices=["datasets", "prepare-targets", "groups", "peaks", "spectrasaves", "components"],
    )
    parser.add_argument("dataset", nargs="?", help="Dataset name for 'components'.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.kind == "datasets":
        values = list_base_datasets(track_data_root=TRACK_DATA_ROOT)
    elif args.kind == "prepare-targets":
        values = list_prepare_targets()
    elif args.kind == "groups":
        values = list_group_names(groups_dir=DEFAULT_GROUPS_DIR)
    elif args.kind == "peaks":
        values = list_peak_names(peaks_dir=DEFAULT_PEAKS_DIR)
    elif args.kind == "spectrasaves":
        values = list_spectrasaves()
    elif args.kind == "components":
        if not args.dataset:
            raise SystemExit("components requires a dataset argument")
        values = list_components(args.dataset)
    else:
        values = []

    for value in values:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
