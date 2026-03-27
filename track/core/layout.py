from __future__ import annotations

from pathlib import Path

from .io import video_name
from .models import DatasetManifest


TRACK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRACK_DIR / "data"
VIDEOS_DIR = TRACK_DIR / "Videos"


def dataset_name(name_or_path: str) -> str:
    return video_name(name_or_path)


def dataset_dir(name_or_path: str) -> Path:
    return DATA_DIR / dataset_name(name_or_path)


def ensure_dataset_dir(name_or_path: str) -> Path:
    path = dataset_dir(name_or_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(name_or_path: str) -> Path:
    return dataset_dir(name_or_path) / "manifest.json"


def params_bottom_path(name_or_path: str) -> Path:
    return dataset_dir(name_or_path) / "params_bottom.json"


def legacy_params_black_path(name_or_path: str) -> Path:
    return dataset_dir(name_or_path) / "params_black.json"


def labels_path(name_or_path: str) -> Path:
    return dataset_dir(name_or_path) / "labels.json"


def track1_path(name_or_path: str) -> Path:
    return dataset_dir(name_or_path) / "track1.msgpack"


def track1_backup_path(name_or_path: str) -> Path:
    return dataset_dir(name_or_path) / "track1.pre_manual_repair_backup.msgpack"


def component_dir(name_or_path: str, component: str) -> Path:
    return dataset_dir(name_or_path) / "components" / str(component)


def component_track2_path(name_or_path: str, component: str) -> Path:
    return component_dir(name_or_path, component) / "track2_permanence.msgpack"


def default_manifest(name_or_path: str) -> DatasetManifest:
    return DatasetManifest(dataset=dataset_name(name_or_path))
