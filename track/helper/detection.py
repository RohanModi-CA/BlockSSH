"""
helper/detection.py
Core per-frame detection: HSV masking, morphology, ring-of-white validation.
The public interface is:
    kernels = DetectionKernels(params)      # build once per params set
    dets    = detect_frame(bgr, params, kernels, y_offset)
    overlay = draw_detections(bgr, dets, frame_num, y_offset)
"""

import math

import cv2
import numpy as np
from typing import List

from tracking_classes import DetectionRecord


# ---------------------------------------------------------------------------
# Morphology helpers
# ---------------------------------------------------------------------------

def get_disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def imfill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def cleanup_color_mask(bw: np.ndarray, se_open: np.ndarray, se_close: np.ndarray) -> np.ndarray:
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  se_open)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se_close)
    return imfill_holes(bw)


def is_bbox_near_border(x: int, y: int, w: int, h: int,
                         img_w: int, img_h: int, margin: int) -> bool:
    return (
        x < margin
        or y < margin
        or x + w >= img_w - margin
        or y + h >= img_h - margin
    )


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
    theta = 0.5 * math.atan2(-2.0 * mu11, mu02 - mu20)

    if theta >= math.pi / 2:
        theta -= math.pi
    elif theta < -math.pi / 2:
        theta += math.pi
    return float(theta)


# ---------------------------------------------------------------------------
# Kernel cache  (build once, reuse every frame)
# ---------------------------------------------------------------------------

class DetectionKernels:
    """
    Pre-built structuring elements and HSV bounds derived from TrackingParams.
    Construct once per params instance; pass into detect_frame().
    """

    def __init__(self, params: 'TrackingParams') -> None:  # type: ignore[name-defined]
        self.se_color_open  = get_disk_kernel(params.colorOpenRadius)
        self.se_color_close = get_disk_kernel(params.colorCloseRadius)
        self.se_white_close = get_disk_kernel(params.whiteCloseRadius)
        self.se_ring_inner  = get_disk_kernel(params.ringInnerRadius)
        self.se_ring_outer  = get_disk_kernel(params.ringOuterRadius)

        def _u8(v: float) -> int:
            return int(np.clip(v * 255, 0, 255))

        cv_min_sat       = _u8(params.minSat_color)
        cv_min_val       = _u8(params.minVal_color)
        cv_white_max_sat = _u8(params.whiteMaxSat)
        cv_white_min_val = _u8(params.whiteMinVal)

        self.lower_red1  = np.array([int(params.redHueLow1  * 179), cv_min_sat, cv_min_val])
        self.upper_red1  = np.array([int(params.redHueHigh1 * 179), 255,        255       ])
        self.lower_red2  = np.array([int(params.redHueLow2  * 179), cv_min_sat, cv_min_val])
        self.upper_red2  = np.array([int(params.redHueHigh2 * 179), 255,        255       ])
        self.lower_green = np.array([int(params.greenHueLow  * 179), cv_min_sat, cv_min_val])
        self.upper_green = np.array([int(params.greenHueHigh * 179), 255,        255       ])
        self.lower_white = np.array([0,   0,                cv_white_min_val])
        self.upper_white = np.array([179, cv_white_max_sat, 255             ])


# ---------------------------------------------------------------------------
# Single-colour blob detection
# ---------------------------------------------------------------------------

def _detect_color(
    color_mask:     np.ndarray,
    white_mask:     np.ndarray,
    color_char:     str,
    min_area:       float,
    max_area:       float,
    connectivity:   int,
    se_inner:       np.ndarray,
    se_outer:       np.ndarray,
    min_white_frac: float,
    reject_border:  bool,
    border_margin:  int,
    outer_radius:   int,
    y_offset:       int,
) -> List[DetectionRecord]:
    """
    Find blobs in color_mask that are surrounded by a sufficient white ring.
    Mirrors the extract_validated_detections logic from the original track1.py exactly.
    """
    if cv2.countNonZero(color_mask) == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_mask, connectivity=connectivity
    )
    img_h, img_w = color_mask.shape
    dets: List[DetectionRecord] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        if reject_border and is_bbox_near_border(x, y, w, h, img_w, img_h, border_margin):
            continue

        pad = outer_radius
        y1 = max(0, y - pad);      y2 = min(img_h, y + h + pad)
        x1 = max(0, x - pad);      x2 = min(img_w, x + w + pad)

        local_comp = (labels[y1:y2, x1:x2] == i).astype(np.uint8) * 255
        outer = cv2.dilate(local_comp, se_outer)
        inner = cv2.dilate(local_comp, se_inner)
        ring  = cv2.bitwise_and(outer, cv2.bitwise_not(inner))

        n_ring = cv2.countNonZero(ring)
        if n_ring == 0:
            continue

        ring_white = cv2.bitwise_and(ring, white_mask[y1:y2, x1:x2])
        if cv2.countNonZero(ring_white) / n_ring >= min_white_frac:
            angle = component_orientation_from_mask(local_comp)
            dets.append(DetectionRecord(
                x=float(cx),
                y=float(cy) + y_offset,
                color=color_char,
                area=float(area),
                angle=angle,
            ))

    return dets


# ---------------------------------------------------------------------------
# Public detection entry point
# ---------------------------------------------------------------------------

def detect_frame(
    bgr:      np.ndarray,
    params:   'TrackingParams',  # type: ignore[name-defined]
    kernels:  DetectionKernels,
    y_offset: int = 0,
) -> List[DetectionRecord]:
    """
    Run full red+green block detection on a single BGR frame.

    Parameters
    ----------
    bgr      : already-cropped BGR image (crop applied before calling)
    params   : TrackingParams instance
    kernels  : DetectionKernels built from the same params
    y_offset : added to every detection's y-coordinate to restore original-frame
               coordinates (pass params.crop_top)

    Returns
    -------
    Detections sorted left-to-right by x.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Build masks
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, kernels.lower_red1, kernels.upper_red1),
        cv2.inRange(hsv, kernels.lower_red2, kernels.upper_red2),
    )
    green_mask = cv2.inRange(hsv, kernels.lower_green, kernels.upper_green)
    white_mask = cv2.inRange(hsv, kernels.lower_white, kernels.upper_white)

    # Morphology
    red_mask   = cleanup_color_mask(red_mask,   kernels.se_color_open, kernels.se_color_close)
    green_mask = cleanup_color_mask(green_mask, kernels.se_color_open, kernels.se_color_close)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernels.se_white_close)
    white_mask = imfill_holes(white_mask)

    # Shared keyword args for both colours
    kw = dict(
        min_area=float(params.min_area),
        max_area=params.effective_max_area,
        connectivity=params.ccConnectivity,
        se_inner=kernels.se_ring_inner,
        se_outer=kernels.se_ring_outer,
        min_white_frac=params.minWhiteCoverageFraction,
        reject_border=params.rejectNearImageBorder,
        border_margin=params.border_margin_px,
        outer_radius=params.ringOuterRadius,
        y_offset=y_offset,
    )

    dets = (
        _detect_color(red_mask,   white_mask, 'r', **kw) +
        _detect_color(green_mask, white_mask, 'g', **kw)
    )
    return sorted(dets, key=lambda d: d.x)


# ---------------------------------------------------------------------------
# Preview overlay
# ---------------------------------------------------------------------------

def draw_detections(
    bgr:        np.ndarray,
    detections: List[DetectionRecord],
    frame_num:  int,
    y_offset:   int  = 0,
    label:      str  = "",
) -> np.ndarray:
    """
    Return a copy of bgr with cross markers on each detection and a status bar.
    Mirrors the display logic from the original track1.py.
    """
    out = bgr.copy()
    n_r = sum(1 for d in detections if d.color == 'r')
    n_g = sum(1 for d in detections if d.color == 'g')

    for d in detections:
        color = (0, 0, 255) if d.color == 'r' else (0, 255, 0)
        x = int(d.x)
        y = int(d.y - y_offset)
        cv2.drawMarker(
            out,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=3,
        )
        if np.isfinite(d.angle):
            length = 26
            dx = int(round(math.sin(d.angle) * length))
            dy = int(round(math.cos(d.angle) * length))
            cv2.line(out, (x - dx, y - dy), (x + dx, y + dy), (0, 255, 255), 2)

    text = f"Frame: {frame_num} | Det: {len(detections)} (R:{n_r}, G:{n_g})"
    if label:
        text += f"  [{label}]"

    box_w = min(len(text) * 15 + 20, out.shape[1] - 10)
    cv2.rectangle(out, (10, 10), (box_w, 70), (0, 0, 0), -1)
    cv2.putText(
        out, text, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out
