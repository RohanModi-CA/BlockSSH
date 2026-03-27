from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

from .io import load_json, save_json
from .models import CropRect


ALLOWED_ROTATIONS = (0, 90, 180, 270)


@dataclass
class BottomTrackingParams:
    rotation_deg: int = 0
    crop_rect: CropRect = field(default_factory=CropRect)
    time_start_s: float = 0.0
    time_end_s: float | None = None
    dark_max_val: int = 90
    blur_kernel: int = 5
    open_radius: int = 1
    close_radius: int = 2
    min_area: int = 90000
    max_area: float | None = None
    reject_near_image_border: bool = True
    border_margin_px: int = 3
    cc_connectivity: int = 8

    @property
    def effective_max_area(self) -> float:
        return float("inf") if self.max_area is None else float(self.max_area)

    @classmethod
    def defaults(cls) -> "BottomTrackingParams":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "BottomTrackingParams":
        data = load_json(path, default={}) or {}
        crop_data = dict(data.get("crop_rect", {}) or {})
        params = cls.defaults()
        for key, value in data.items():
            if key == "crop_rect":
                params.crop_rect = CropRect(
                    x0=int(crop_data.get("x0", 0)),
                    x1=None if crop_data.get("x1") is None else int(crop_data.get("x1")),
                    y0=int(crop_data.get("y0", 0)),
                    y1=None if crop_data.get("y1") is None else int(crop_data.get("y1")),
                )
            elif hasattr(params, key):
                setattr(params, key, value)
        params.rotation_deg = _normalize_rotation(params.rotation_deg)
        return params

    def save(self, path: str | Path) -> Path:
        payload = asdict(self)
        return save_json(path, payload)


def _normalize_rotation(rotation_deg: int) -> int:
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg not in ALLOWED_ROTATIONS:
        nearest = min(ALLOWED_ROTATIONS, key=lambda value: abs(value - rotation_deg))
        rotation_deg = nearest
    return rotation_deg
