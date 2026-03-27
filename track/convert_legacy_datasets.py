#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import msgpack

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track.core.bottom_params import BottomTrackingParams
from track.core.models import CropRect, DatasetManifest


TRACK_DIR = Path(__file__).resolve().parent
DATA_DIR = TRACK_DIR / "data"
KNOWN_COMPONENT_SUFFIXES = ("x", "y", "a", "fx", "fy", "fa", "area")
REPORT_PATH = DATA_DIR / "legacy_conversion_report.json"


def load_msgpack(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy track/data layouts into the new dataset-root/component-subfolder format."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write files. Default is dry-run with only a conversion report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files under the new layout if they already exist.",
    )
    parser.add_argument(
        "--report",
        default=str(REPORT_PATH),
        help=f"Path for the JSON conversion report. Default: {REPORT_PATH}",
    )
    return parser.parse_args()


def try_component_suffix(name: str) -> tuple[str, str] | None:
    for suffix in KNOWN_COMPONENT_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            return name[: -len(token)], suffix
    return None


def infer_rotation_deg(name: str) -> int:
    match = re.search(r"_rot(90|180|270)$", name)
    if match:
        return int(match.group(1))
    return 0


def infer_tracking_mode(track1_data: dict[str, Any] | None, *, has_track1: bool, has_root_track2: bool) -> str:
    if track1_data is not None:
        params = track1_data.get("params", {}) or {}
        assumed = str(params.get("assumedInputType", "")).strip().lower()
        if "black-on-white blob tracking" in assumed:
            return "bottom_black"
        if "matlab" in assumed:
            return "legacy_matlab_import"
        if has_track1:
            return "legacy_track1_import"
    if has_root_track2:
        return "legacy_track2_import"
    return "legacy_import"


def normalize_source_video(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    marker = "/Videos/"
    if marker in text:
        idx = text.rfind(marker)
        return f"Videos/{text[idx + len(marker):]}"
    if text.startswith("Videos/"):
        return text
    return text


def convert_legacy_params(
    dataset: str,
    legacy_params: dict[str, Any] | None,
) -> tuple[BottomTrackingParams | None, list[str]]:
    if legacy_params is None:
        return None, []

    notes: list[str] = []
    crop_top = int(legacy_params.get("crop_top", 0) or 0)
    crop_bottom = int(legacy_params.get("crop_bottom", 0) or 0)
    if crop_bottom != 0:
        notes.append(
            f"legacy crop_bottom={crop_bottom} could not be represented exactly in params_bottom.json"
        )

    max_area = legacy_params.get("max_area", None)
    if isinstance(max_area, float) and math.isinf(max_area):
        max_area = None

    params = BottomTrackingParams(
        rotation_deg=infer_rotation_deg(dataset),
        crop_rect=CropRect(x0=0, x1=None, y0=crop_top, y1=None),
        time_start_s=0.0,
        time_end_s=None,
        dark_max_val=90,
        blur_kernel=5,
        open_radius=int(
            legacy_params.get("colorOpenRadius", legacy_params.get("color_open_radius", 1)) or 1
        ),
        close_radius=int(
            legacy_params.get("colorCloseRadius", legacy_params.get("color_close_radius", 2)) or 2
        ),
        min_area=int(legacy_params.get("min_area", 90000) or 0),
        max_area=max_area,
        reject_near_image_border=bool(
            legacy_params.get(
                "rejectNearImageBorder",
                legacy_params.get("reject_near_image_border", True),
            )
        ),
        border_margin_px=int(
            legacy_params.get("borderMarginPx", legacy_params.get("border_margin_px", 3)) or 0
        ),
        cc_connectivity=int(
            legacy_params.get("ccConnectivity", legacy_params.get("cc_connectivity", 8)) or 8
        ),
    )
    return params, notes


def derive_default_labels(dataset: str, track2_path: Path) -> dict[str, Any]:
    data = load_msgpack(track2_path)
    x_positions = data.get("xPositions")
    if not isinstance(x_positions, list) or len(x_positions) == 0:
        return {
            "dataset": dataset,
            "version": 1,
            "bond_labels": {},
            "site_labels": {},
            "disabled": {"sites": [], "components": {}},
            "notes": "",
            "created_by": "convert_legacy_datasets.py",
        }

    n_sites = len(x_positions[0])
    return {
        "dataset": dataset,
        "version": 1,
        "bond_labels": {},
        "site_labels": {str(idx): str(idx) for idx in range(1, n_sites + 1)},
        "disabled": {"sites": [], "components": {}},
        "notes": "",
        "created_by": "convert_legacy_datasets.py",
    }


def copy_if_needed(src: Path, dst: Path, *, apply: bool, overwrite: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "exists"
        if apply:
            shutil.copy2(src, dst)
        return "overwritten"

    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "created"


def top_level_dirs() -> list[Path]:
    return sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_dir() and path.name not in {"old", ".legacy"} and not path.name.endswith("_BACKUP")
    )


def build_inventory() -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}

    for path in top_level_dirs():
        name = path.name
        suffix_info = try_component_suffix(name)
        track1 = path / "track1.msgpack"
        track2 = path / "track2_permanence.msgpack"
        manifest = path / "manifest.json"
        labels = path / "labels.json"
        params_black = path / "params_black.json"
        params_bottom = path / "params_bottom.json"

        if suffix_info is not None and track2.is_file():
            base, component = suffix_info
            entry = inventory.setdefault(
                base,
                {
                    "base": base,
                    "root_dir": DATA_DIR / base,
                    "legacy_component_dirs": {},
                    "root_track2_path": None,
                    "has_manifest": False,
                    "has_labels": False,
                    "has_track1": False,
                    "params_black_path": None,
                    "params_bottom_path": None,
                },
            )
            entry["legacy_component_dirs"][component] = path
            continue

        entry = inventory.setdefault(
            name,
            {
                "base": name,
                "root_dir": path,
                "legacy_component_dirs": {},
                "root_track2_path": None,
                "has_manifest": False,
                "has_labels": False,
                "has_track1": False,
                "params_black_path": None,
                "params_bottom_path": None,
            },
        )
        if track2.is_file():
            entry["root_track2_path"] = track2
        if track1.is_file():
            entry["has_track1"] = True
        if manifest.is_file():
            entry["has_manifest"] = True
        if labels.is_file():
            entry["has_labels"] = True
        if params_black.is_file():
            entry["params_black_path"] = params_black
        if params_bottom.is_file():
            entry["params_bottom_path"] = params_bottom

    return inventory


def legacy_track1_params(root_dir: Path, params_black_path: Path | None) -> dict[str, Any] | None:
    if params_black_path is not None and params_black_path.is_file():
        return json.loads(params_black_path.read_text(encoding="utf-8"))

    track1_path = root_dir / "track1.msgpack"
    if not track1_path.is_file():
        return None
    data = load_msgpack(track1_path)
    params = data.get("params")
    if isinstance(params, dict):
        return params
    return None


def track1_source_video(root_dir: Path) -> str:
    track1_path = root_dir / "track1.msgpack"
    if not track1_path.is_file():
        return ""
    data = load_msgpack(track1_path)
    return normalize_source_video(data.get("filepath"))


def track2_source_video(track2_path: Path | None) -> str:
    if track2_path is None or not track2_path.is_file():
        return ""
    data = load_msgpack(track2_path)
    return normalize_source_video(data.get("originalVideoPath"))


def choose_label_component(
    root_dir: Path,
    component_sources: dict[str, Path],
    root_track2_path: Path | None,
) -> Path | None:
    components = list(component_sources.keys())
    if "x" in components:
        candidate = root_dir / "components" / "x" / "track2_permanence.msgpack"
        if candidate.is_file():
            return candidate
        return component_sources["x"]
    if root_track2_path is not None and root_track2_path.is_file():
        return root_track2_path
    for component in components:
        candidate = root_dir / "components" / component / "track2_permanence.msgpack"
        if candidate.is_file():
            return candidate
        return component_sources[component]
    return None


def convert_one(entry: dict[str, Any], *, apply: bool, overwrite: bool) -> dict[str, Any]:
    dataset = str(entry["base"])
    root_dir = Path(entry["root_dir"])
    root_dir.mkdir(parents=True, exist_ok=True) if apply else None

    root_track2_path = entry["root_track2_path"]
    component_sources: dict[str, Path] = {
        component: component_dir / "track2_permanence.msgpack"
        for component, component_dir in sorted(entry["legacy_component_dirs"].items())
    }
    if root_track2_path is not None and len(component_sources) == 0:
        component_sources["x"] = root_track2_path

    component_results: dict[str, str] = {}
    for component, src in component_sources.items():
        dst = root_dir / "components" / component / "track2_permanence.msgpack"
        component_results[component] = copy_if_needed(src, dst, apply=apply, overwrite=overwrite)

    params_data = legacy_track1_params(root_dir, entry["params_black_path"])
    bottom_params, param_notes = convert_legacy_params(dataset, params_data)
    params_bottom_status = "skipped"
    if bottom_params is not None:
        params_bottom_path = root_dir / "params_bottom.json"
        if params_bottom_path.exists() and not overwrite:
            params_bottom_status = "exists"
        else:
            if apply:
                bottom_params.save(params_bottom_path)
            params_bottom_status = "created" if not params_bottom_path.exists() or overwrite else "exists"

    labels_status = "kept" if (root_dir / "labels.json").exists() else "skipped"
    label_component_path = choose_label_component(root_dir, component_sources, root_track2_path)
    if not (root_dir / "labels.json").exists() and label_component_path is not None:
        labels_payload = derive_default_labels(dataset, label_component_path)
        if apply:
            save_json(root_dir / "labels.json", labels_payload)
        labels_status = "created"

    track1_data = None
    if entry["has_track1"]:
        track1_data = load_msgpack(root_dir / "track1.msgpack")
    tracking_mode = infer_tracking_mode(
        track1_data,
        has_track1=bool(entry["has_track1"]),
        has_root_track2=root_track2_path is not None,
    )
    source_video = track1_source_video(root_dir) or track2_source_video(root_track2_path)
    if not source_video:
        for src in component_sources.values():
            source_video = track2_source_video(src)
            if source_video:
                break

    manifest_path = root_dir / "manifest.json"
    manifest_payload = DatasetManifest(
        dataset=dataset,
        tracking_mode=tracking_mode,
        version=1,
        source_video=source_video,
        params_file="params_bottom.json" if bottom_params is not None else "",
        track1_file="track1.msgpack" if entry["has_track1"] else "",
        labels_file="labels.json" if (root_dir / "labels.json").exists() or labels_status == "created" else "",
        components={
            component: f"components/{component}/track2_permanence.msgpack"
            for component in sorted(component_sources.keys())
        },
        rotation_deg=int(bottom_params.rotation_deg) if bottom_params is not None else infer_rotation_deg(dataset),
        crop_rect=(
            {
                "x0": int(bottom_params.crop_rect.x0),
                "x1": bottom_params.crop_rect.x1,
                "y0": int(bottom_params.crop_rect.y0),
                "y1": bottom_params.crop_rect.y1,
            }
            if bottom_params is not None
            else {"x0": 0, "x1": None, "y0": 0, "y1": None}
        ),
        notes="; ".join(
            note
            for note in (
                [f"converted from legacy layout on-disk for {dataset}"]
                + param_notes
            )
            if note
        ),
    )

    if root_track2_path is not None and len(entry["legacy_component_dirs"]) == 0:
        legacy_note = (
            f"legacy root track2 imported as component x from {root_track2_path.relative_to(DATA_DIR)}"
        )
        manifest_payload.notes = "; ".join(filter(None, [manifest_payload.notes, legacy_note]))

    manifest_status = "exists" if manifest_path.exists() and not overwrite else "created"
    if apply:
        save_json(manifest_path, asdict(manifest_payload))

    return {
        "dataset": dataset,
        "root_dir": str(root_dir),
        "tracking_mode": tracking_mode,
        "source_video": source_video,
        "components": sorted(component_sources.keys()),
        "has_track1": bool(entry["has_track1"]),
        "had_manifest": bool(entry["has_manifest"]),
        "copied_components": component_results,
        "params_bottom": params_bottom_status,
        "labels": labels_status,
        "manifest": manifest_status,
        "notes": manifest_payload.notes,
    }


def main() -> int:
    args = parse_args()
    inventory = build_inventory()
    report = {
        "apply": bool(args.apply),
        "overwrite": bool(args.overwrite),
        "datasets": [],
    }

    for dataset in sorted(inventory.keys()):
        report["datasets"].append(convert_one(inventory[dataset], apply=args.apply, overwrite=args.overwrite))

    save_json(Path(args.report), report)

    converted = len(report["datasets"])
    created_manifests = sum(1 for item in report["datasets"] if item["manifest"] == "created")
    print(f"Datasets inventoried: {converted}")
    print(f"Manifest files {'written' if args.apply else 'would be written'}: {created_manifests}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
