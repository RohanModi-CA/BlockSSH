from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import msgpack


VIDEO_EXTENSIONS = [".mov", ".avi", ".mp4"]


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(data) if is_dataclass(data) else data
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return p


def load_msgpack(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False)


def save_msgpack(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(obj) if is_dataclass(obj) else obj
    with p.open("wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))
    return p


def video_name(video_path: str | Path) -> str:
    return Path(video_path).stem


def find_video(name_or_prefix: str, search_dir: str | Path) -> Path | None:
    search_root = Path(search_dir)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {search_root}")

    base = Path(name_or_prefix).stem
    ext_set = {ext.lower() for ext in VIDEO_EXTENSIONS}

    for ext in VIDEO_EXTENSIONS:
        for variant in (ext, ext.upper(), ext.capitalize()):
            candidate = search_root / f"{base}{variant}"
            if candidate.is_file():
                return candidate

    for candidate in sorted(search_root.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in ext_set:
            continue
        if candidate.stem == base or candidate.stem.endswith(base):
            return candidate

    return None

