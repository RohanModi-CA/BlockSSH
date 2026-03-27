from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .bottom_params import BottomTrackingParams, _normalize_rotation
from .models import CropRect


@dataclass(frozen=True)
class TransformSpec:
    rotation_deg: int
    crop_rect: CropRect
    rotated_shape: tuple[int, int]


def rotate_frame(frame: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation_deg = _normalize_rotation(rotation_deg)
    if rotation_deg == 0:
        return frame
    if rotation_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation_deg}")


def normalize_crop_rect(crop_rect: CropRect, width: int, height: int) -> CropRect:
    x0 = max(0, min(int(crop_rect.x0), max(0, width - 1)))
    y0 = max(0, min(int(crop_rect.y0), max(0, height - 1)))
    x1_raw = width if crop_rect.x1 is None else int(crop_rect.x1)
    y1_raw = height if crop_rect.y1 is None else int(crop_rect.y1)
    x1 = max(x0 + 1, min(x1_raw, width))
    y1 = max(y0 + 1, min(y1_raw, height))
    return CropRect(x0=x0, x1=x1, y0=y0, y1=y1)


def build_transform_spec(frame: np.ndarray, params: BottomTrackingParams) -> TransformSpec:
    rotated = rotate_frame(frame, params.rotation_deg)
    height, width = rotated.shape[:2]
    crop_rect = normalize_crop_rect(params.crop_rect, width, height)
    return TransformSpec(
        rotation_deg=_normalize_rotation(params.rotation_deg),
        crop_rect=crop_rect,
        rotated_shape=(height, width),
    )


def apply_transform(frame: np.ndarray, params: BottomTrackingParams) -> tuple[np.ndarray, TransformSpec]:
    spec = build_transform_spec(frame, params)
    rotated = rotate_frame(frame, spec.rotation_deg)
    cropped = rotated[spec.crop_rect.y0:spec.crop_rect.y1, spec.crop_rect.x0:spec.crop_rect.x1]
    return cropped, spec


def darken_outside_crop(rotated_frame: np.ndarray, crop_rect: CropRect) -> np.ndarray:
    disp = rotated_frame.copy()
    x0, x1, y0, y1 = crop_rect.x0, crop_rect.x1, crop_rect.y0, crop_rect.y1
    assert x1 is not None
    assert y1 is not None
    if y0 > 0:
        disp[:y0, :] = (disp[:y0, :].astype(np.float32) * 0.35).astype(np.uint8)
    if y1 < disp.shape[0]:
        disp[y1:, :] = (disp[y1:, :].astype(np.float32) * 0.35).astype(np.uint8)
    if x0 > 0:
        disp[:, :x0] = (disp[:, :x0].astype(np.float32) * 0.35).astype(np.uint8)
    if x1 < disp.shape[1]:
        disp[:, x1:] = (disp[:, x1:].astype(np.float32) * 0.35).astype(np.uint8)
    return disp

