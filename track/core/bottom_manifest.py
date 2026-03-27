from __future__ import annotations

from pathlib import Path

from .bottom_params import BottomTrackingParams
from .io import find_video, load_json, save_json
from .layout import (
    VIDEOS_DIR,
    default_manifest,
    legacy_params_black_path,
    manifest_path,
    params_bottom_path,
    track1_path,
)
from .models import DatasetManifest


def load_manifest(name: str) -> DatasetManifest:
    data = load_json(manifest_path(name), default=None)
    if data is None:
        return default_manifest(name)
    return DatasetManifest.from_dict(data)


def enrich_manifest(name: str, manifest: DatasetManifest) -> DatasetManifest:
    dataset = Path(name).stem
    manifest.dataset = dataset

    video_path = find_video(dataset, VIDEOS_DIR)
    if video_path is not None and not manifest.source_video:
        manifest.source_video = str(video_path.relative_to(VIDEOS_DIR.parent))

    params_path = params_bottom_path(dataset)
    legacy_params_path = legacy_params_black_path(dataset)
    if params_path.exists():
        manifest.params_file = params_path.name
        params = BottomTrackingParams.load(params_path)
        manifest.rotation_deg = int(params.rotation_deg)
        manifest.crop_rect = {
            "x0": int(params.crop_rect.x0),
            "x1": None if params.crop_rect.x1 is None else int(params.crop_rect.x1),
            "y0": int(params.crop_rect.y0),
            "y1": None if params.crop_rect.y1 is None else int(params.crop_rect.y1),
        }
    elif legacy_params_path.exists():
        manifest.params_file = legacy_params_path.name
        params = BottomTrackingParams.load(legacy_params_path)
        manifest.rotation_deg = int(params.rotation_deg)
        manifest.crop_rect = {
            "x0": int(params.crop_rect.x0),
            "x1": None if params.crop_rect.x1 is None else int(params.crop_rect.x1),
            "y0": int(params.crop_rect.y0),
            "y1": None if params.crop_rect.y1 is None else int(params.crop_rect.y1),
        }

    t1_path = track1_path(dataset)
    if t1_path.exists():
        manifest.track1_file = t1_path.name

    return manifest


def save_manifest(name: str, manifest: DatasetManifest) -> Path:
    manifest = enrich_manifest(name, manifest)
    return save_json(manifest_path(name), manifest)

