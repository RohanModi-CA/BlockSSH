from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from track.tracking_classes import DetectionRecord

from .bottom_params import BottomTrackingParams


def disk_kernel(radius: int) -> Optional[np.ndarray]:
    if radius <= 0:
        return None
    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def build_dark_mask(gray: np.ndarray, params: BottomTrackingParams) -> np.ndarray:
    blur_k = max(1, int(params.blur_kernel))
    if blur_k % 2 == 0:
        blur_k += 1

    if blur_k > 1:
        gray = cv2.medianBlur(gray, blur_k)

    mask = (gray <= params.dark_max_val).astype(np.uint8) * 255

    k_open = disk_kernel(params.open_radius)
    if k_open is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    k_close = disk_kernel(params.close_radius)
    if k_close is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    return mask


def component_touches_border(stats_row, width: int, height: int, margin: int) -> bool:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x <= margin or y <= margin or (x + w) >= (width - margin) or (y + h) >= (height - margin)


def component_orientation_from_mask(component_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(component_mask)
    if len(xs) < 2:
        return float("nan")

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    x -= x.mean()
    y -= y.mean()

    mu20 = np.mean(x * x)
    mu02 = np.mean(y * y)
    mu11 = np.mean(x * y)
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)

    if theta >= math.pi / 2:
        theta -= math.pi
    elif theta < -math.pi / 2:
        theta += math.pi
    return float(theta)


def detect_frame_black(
    bgr_cropped: np.ndarray,
    params: BottomTrackingParams,
    x_offset: int = 0,
    y_offset: int = 0,
) -> List[DetectionRecord]:
    gray = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2GRAY)
    mask = build_dark_mask(gray, params)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=params.cc_connectivity,
        ltype=cv2.CV_32S,
    )

    height, width = mask.shape[:2]
    detections: List[DetectionRecord] = []
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < params.min_area or area > params.effective_max_area:
            continue
        if params.reject_near_image_border and component_touches_border(stats[label], width, height, params.border_margin_px):
            continue

        cx, cy = centroids[label]
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        local_labels = labels[y0:y0 + hh, x0:x0 + ww]
        component_mask = (local_labels == label).astype(np.uint8)
        angle = component_orientation_from_mask(component_mask)
        detections.append(
            DetectionRecord(
                x=float(cx + x_offset),
                y=float(cy + y_offset),
                color="b",
                area=area,
                angle=angle,
            )
        )

    detections.sort(key=lambda det: det.x)
    return detections


def draw_detections(
    bgr_cropped: np.ndarray,
    detections: List[DetectionRecord],
    frame_num: int,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    overlay = bgr_cropped.copy()
    for idx, det in enumerate(detections):
        x = int(round(det.x - x_offset))
        y = int(round(det.y - y_offset))
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), 2)
        if np.isfinite(det.angle):
            length = 40
            dx = int(round(math.cos(det.angle) * length))
            dy = int(round(math.sin(det.angle) * length))
            cv2.line(overlay, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 255), 2)
        cv2.putText(overlay, f"{idx}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"({det.x:.1f}, {det.y:.1f})", (x + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1, cv2.LINE_AA)

    cv2.putText(
        overlay,
        f"frame={frame_num}  n={len(detections)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay
