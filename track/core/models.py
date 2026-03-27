from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CropRect:
    x0: int = 0
    x1: int | None = None
    y0: int = 0
    y1: int | None = None


@dataclass
class DatasetManifest:
    dataset: str
    tracking_mode: str = "bottom_black"
    version: int = 1
    source_video: str = ""
    params_file: str = "params_bottom.json"
    track1_file: str = "track1.msgpack"
    labels_file: str = "labels.json"
    components: dict[str, str] = field(default_factory=dict)
    rotation_deg: int = 0
    crop_rect: dict[str, int | None] = field(default_factory=lambda: {
        "x0": 0,
        "x1": None,
        "y0": 0,
        "y1": None,
    })
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        merged = {
            "dataset": str(data.get("dataset", "")),
            "tracking_mode": str(data.get("tracking_mode", "bottom_black")),
            "version": int(data.get("version", 1)),
            "source_video": str(data.get("source_video", "")),
            "params_file": str(data.get("params_file", "params_bottom.json")),
            "track1_file": str(data.get("track1_file", "track1.msgpack")),
            "labels_file": str(data.get("labels_file", "labels.json")),
            "components": dict(data.get("components", {}) or {}),
            "rotation_deg": int(data.get("rotation_deg", 0)),
            "crop_rect": dict(data.get("crop_rect", {}) or {}),
            "notes": str(data.get("notes", "")),
        }
        crop_rect = {
            "x0": int(merged["crop_rect"].get("x0", 0)),
            "x1": merged["crop_rect"].get("x1"),
            "y0": int(merged["crop_rect"].get("y0", 0)),
            "y1": merged["crop_rect"].get("y1"),
        }
        if crop_rect["x1"] is not None:
            crop_rect["x1"] = int(crop_rect["x1"])
        if crop_rect["y1"] is not None:
            crop_rect["y1"] = int(crop_rect["y1"])
        merged["crop_rect"] = crop_rect
        return cls(**merged)

