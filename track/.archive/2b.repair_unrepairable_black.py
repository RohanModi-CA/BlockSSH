#!/usr/bin/env python3
"""
2b.repair_unrepairable_black.py

Manual repair tool for black-track datasets that still fail after automatic
verification/repair.

Workflow
--------
1. Load data/<name>/track1.msgpack
2. Run the standard auto-repair in memory
3. Focus only on frames that still fail validation
4. Let the user manually edit detections in an OpenCV window
5. Save to a separate track1_manual_repaired.msgpack file and a JSON edit log
6. Optionally promote the repaired file into track1.msgpack for the main workflow

This tool never overwrites the original track1.msgpack unless you explicitly do
that yourself outside the tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import cv2
import msgpack
import numpy as np

from track.core.io import load_json, save_json
from track.core.layout import default_manifest, manifest_path
from track.core.models import DatasetManifest

from helper.video_io import dataset_dir, find_video, track1_output_path, video_name
import helper.verification_black as vb
from tracking_classes import DetectionRecord, VideoCentroids


WIN_NAME = "Manual Repair Black"
WIN_W = 1600
WIN_H = 900
SELECT_RADIUS_PX = 22.0
ANGLE_STEP_DEG = 5.0
AREA_STEP = 1.15


def load_vc(path: str) -> VideoCentroids:
    with open(path, "rb") as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def save_vc(path: str, vc: VideoCentroids) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(msgpack.packb(asdict(vc)))


def default_manual_paths(name: str) -> tuple[str, str]:
    d = dataset_dir(name)
    return (
        os.path.join(d, "track1_manual_repaired.msgpack"),
        os.path.join(d, "track1_manual_repair_log.json"),
    )


def default_backup_track1_path(name: str) -> str:
    d = dataset_dir(name)
    return os.path.join(d, "track1.pre_manual_repair_backup.msgpack")


def load_manifest(name: str) -> DatasetManifest:
    path = manifest_path(name)
    data = load_json(path, default=None)
    if data is None:
        return default_manifest(name)
    return DatasetManifest.from_dict(data)


def save_manifest(name: str, manifest: DatasetManifest) -> None:
    save_json(manifest_path(name), manifest)


def promote_repaired_track1(
    *,
    name: str,
    working_track1_path: str,
    backup_path: str,
    repaired_output_path: str,
    vc: VideoCentroids,
) -> None:
    if os.path.exists(backup_path):
        os.remove(backup_path)
    os.replace(working_track1_path, backup_path)
    save_vc(working_track1_path, vc)

    manifest = load_manifest(name)
    manifest.dataset = name
    manifest.track1_file = os.path.basename(working_track1_path)
    save_manifest(name, manifest)

    print("Promoted repaired result to working track1.msgpack.")
    print(f"Backup saved to: {backup_path}")
    print(f"Next: python3 track/Bottom/2.ProcessVerify.py {name}")


def resolve_reference_path(ref: str) -> str:
    if os.path.exists(ref):
        return ref
    candidate = track1_output_path(ref)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Reference track1 not found: {ref}")


def build_focus_indices(n_frames: int, bad_indices: list[int], context: int) -> list[int]:
    focus = set()
    for idx in bad_indices:
        lo = max(0, idx - context)
        hi = min(n_frames - 1, idx + context)
        for k in range(lo, hi + 1):
            focus.add(k)
    return sorted(focus)


def frame_run_label(frame_idx: int, bad_runs: list[tuple[int, int]]) -> Optional[str]:
    for run_idx, (start, end) in enumerate(bad_runs, start=1):
        if start <= frame_idx <= end:
            return f"{run_idx}/{len(bad_runs)} [{start},{end}]"
    return None


def detect_center_distance(det: DetectionRecord, x: float, y: float) -> float:
    return math.hypot(det.x - x, det.y - y)


def clone_detections(detections: list[DetectionRecord]) -> list[DetectionRecord]:
    return [DetectionRecord(x=d.x, y=d.y, color=d.color, area=d.area, angle=d.angle) for d in detections]


def normalize_frame_detections(vc: VideoCentroids, frame_idx: int) -> None:
    vc.frames[frame_idx].detections.sort(key=lambda d: d.x)


def median_area_from_neighbors(vc: VideoCentroids, frame_idx: int) -> float:
    areas = []
    for k in range(max(0, frame_idx - 2), min(len(vc.frames), frame_idx + 3)):
        areas.extend([float(d.area) for d in vc.frames[k].detections if np.isfinite(d.area) and d.area > 0])
    if areas:
        return float(np.median(np.array(areas, dtype=float)))
    return 100.0


def nearest_detection_index(detections: list[DetectionRecord], x: float, y: float, radius: float) -> Optional[int]:
    best_idx = None
    best_dist = float("inf")
    for idx, det in enumerate(detections):
        dist = detect_center_distance(det, x, y)
        if dist <= radius and dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def nearest_anchor_angle(dets: list[DetectionRecord], x: float) -> float:
    if not dets:
        return float("nan")
    best = min(dets, key=lambda d: abs(d.x - x))
    return float(best.angle)


def infer_angle(vc: VideoCentroids, frame_idx: int, x: float) -> float:
    prev_dets = vc.frames[max(0, frame_idx - 1)].detections if frame_idx > 0 else []
    next_dets = vc.frames[min(len(vc.frames) - 1, frame_idx + 1)].detections if frame_idx + 1 < len(vc.frames) else []
    candidates = [nearest_anchor_angle(prev_dets, x), nearest_anchor_angle(next_dets, x)]
    finite = [v for v in candidates if np.isfinite(v)]
    if finite:
        return float(np.mean(np.array(finite, dtype=float)))
    return 0.0


def find_prev_next_good(frame_idx: int, bad_index_set: set[int], n_frames: int) -> tuple[Optional[int], Optional[int]]:
    prev_good = None
    next_good = None
    for k in range(frame_idx - 1, -1, -1):
        if k not in bad_index_set:
            prev_good = k
            break
    for k in range(frame_idx + 1, n_frames):
        if k not in bad_index_set:
            next_good = k
            break
    return prev_good, next_good


def interpolate_from_anchors(
    vc: VideoCentroids,
    frame_idx: int,
    left_idx: Optional[int],
    right_idx: Optional[int],
) -> list[DetectionRecord]:
    if left_idx is None and right_idx is None:
        return []
    if left_idx is None:
        return clone_detections(vc.frames[right_idx].detections)
    if right_idx is None:
        return clone_detections(vc.frames[left_idx].detections)

    left_dets = vc.frames[left_idx].detections
    right_dets = vc.frames[right_idx].detections
    if not left_dets and not right_dets:
        return []

    tL = float(vc.frames[left_idx].frame_time_s)
    tR = float(vc.frames[right_idx].frame_time_s)
    tk = float(vc.frames[frame_idx].frame_time_s)
    if tR != tL:
        alpha = float(np.clip((tk - tL) / (tR - tL), 0.0, 1.0))
    else:
        alpha = float(frame_idx - left_idx) / float(max(1, right_idx - left_idx))

    n = int(round((1.0 - alpha) * len(left_dets) + alpha * len(right_dets)))
    n = max(0, n)
    if n == 0:
        return []

    out: list[DetectionRecord] = []
    for i in range(n):
        srcL = left_dets[min(i, len(left_dets) - 1)] if left_dets else None
        srcR = right_dets[min(i, len(right_dets) - 1)] if right_dets else None

        if srcL is not None and srcR is not None:
            x = float(srcL.x + alpha * (srcR.x - srcL.x))
            y = float(srcL.y + alpha * (srcR.y - srcL.y))
            area = float(max(1.0, srcL.area + alpha * (srcR.area - srcL.area)))
            if np.isfinite(srcL.angle) and np.isfinite(srcR.angle):
                cL, sL = math.cos(2.0 * srcL.angle), math.sin(2.0 * srcL.angle)
                cR, sR = math.cos(2.0 * srcR.angle), math.sin(2.0 * srcR.angle)
                angle = float(0.5 * math.atan2(sL + alpha * (sR - sL), cL + alpha * (cR - cL)))
            else:
                angle = srcL.angle if np.isfinite(srcL.angle) else srcR.angle
        elif srcL is not None:
            x = float(srcL.x)
            y = float(srcL.y)
            area = float(max(1.0, srcL.area))
            angle = float(srcL.angle)
        else:
            x = float(srcR.x)
            y = float(srcR.y)
            area = float(max(1.0, srcR.area))
            angle = float(srcR.angle)

        out.append(DetectionRecord(x=x, y=y, color="b", area=area, angle=angle))

    out.sort(key=lambda d: d.x)
    return out


def draw_detection(frame: np.ndarray, det: DetectionRecord, idx: int, color: tuple[int, int, int], selected: bool) -> None:
    x = int(round(det.x))
    y = int(round(det.y))
    radius = 12 if selected else 9
    thickness = 3 if selected else 2
    cv2.circle(frame, (x, y), radius, color, thickness)

    if np.isfinite(det.angle):
        length = 42 if selected else 34
        dx = int(round(math.cos(det.angle) * length))
        dy = int(round(math.sin(det.angle) * length))
        cv2.line(frame, (x - dx, y - dy), (x + dx, y + dy), color, thickness)

    cv2.putText(
        frame,
        f"{idx}",
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_reference_detections(frame: np.ndarray, detections: list[DetectionRecord], color: tuple[int, int, int], label: str) -> None:
    for idx, det in enumerate(detections):
        x = int(round(det.x))
        y = int(round(det.y))
        cv2.circle(frame, (x, y), 7, color, 1)
        if np.isfinite(det.angle):
            dx = int(round(math.cos(det.angle) * 24))
            dy = int(round(math.sin(det.angle) * 24))
            cv2.line(frame, (x - dx, y - dy), (x + dx, y + dy), color, 1)
    if detections:
        x0 = int(round(detections[0].x))
        y0 = int(round(detections[0].y))
        cv2.putText(frame, label, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def diagnostic_summary(diag: Optional[dict]) -> Optional[str]:
    if not diag:
        return None
    reason = diag.get("reason", "unknown")
    details = diag.get("details", {})
    if reason == "count_jump":
        return (
            f"count jump from prev good {diag.get('prev_good_idx')}: "
            f"{details.get('prev_good_count')} -> {diag.get('count')} "
            f"(delta={details.get('count_delta')})"
        )
    if reason == "transition_cost":
        return (
            f"transition cost {details.get('transition_cost', float('nan')):.3f} > "
            f"{details.get('transition_limit', float('nan')):.3f}"
        )
    if reason == "spacing_ratio":
        return (
            f"spacing ratio out of bounds: "
            f"{details.get('ratio_min', float('nan')):.2f} .. {details.get('ratio_max', float('nan')):.2f}"
        )
    if reason == "no_valid_transition":
        return f"no valid transition from prev good {diag.get('prev_good_idx')}"
    return reason


class RepairUI:
    def __init__(
        self,
        name: str,
        vc: VideoCentroids,
        video_path: str,
        out_path: str,
        log_path: str,
        bad_scan: dict,
        focus_indices: list[int],
        reference_vc: Optional[VideoCentroids] = None,
    ) -> None:
        self.name = name
        self.vc = vc
        self.video_path = video_path
        self.out_path = out_path
        self.log_path = log_path
        self.reference_vc = reference_vc

        self.focus_indices = focus_indices
        self.focus_pos = 0
        self.bad_scan = bad_scan
        self.bad_index_set = set(bad_scan["bad_indices"])
        self.diag_by_idx = {d["frame_idx"]: d for d in bad_scan["diagnostics"] if d["is_bad"]}
        self.bad_runs = bad_scan["bad_runs"]
        self.ref_spacing = bad_scan["ref_spacing_px"]
        self.context = 2

        self.selected_idx: Optional[int] = None
        self.dragging = False
        self.dirty = False
        self.last_message = "Edit detections, press v to validate, f to save if clean."
        self.undo_stack: dict[int, list[list[DetectionRecord]]] = {}
        self.edit_log: list[dict] = []

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

    def current_frame_idx(self) -> int:
        return self.focus_indices[self.focus_pos]

    def current_frame(self):
        return self.vc.frames[self.current_frame_idx()]

    def push_undo(self, frame_idx: int) -> None:
        self.undo_stack.setdefault(frame_idx, []).append(clone_detections(self.vc.frames[frame_idx].detections))
        self.undo_stack[frame_idx] = self.undo_stack[frame_idx][-20:]

    def mark_edit(self, action: str, frame_idx: int, extra: Optional[dict] = None) -> None:
        self.edit_log.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame_idx": int(frame_idx),
                "action": action,
                "detections": [
                    {
                        "x": float(d.x),
                        "y": float(d.y),
                        "area": float(d.area),
                        "angle": None if not np.isfinite(d.angle) else float(d.angle),
                    }
                    for d in self.vc.frames[frame_idx].detections
                ],
                "extra": extra or {},
            }
        )
        self.dirty = True

    def refresh_bad_scan(self) -> None:
        self.bad_scan = vb.scan_bad_frames_detailed(self.vc)
        self.bad_index_set = set(self.bad_scan["bad_indices"])
        self.diag_by_idx = {d["frame_idx"]: d for d in self.bad_scan["diagnostics"] if d["is_bad"]}
        self.bad_runs = self.bad_scan["bad_runs"]
        self.ref_spacing = self.bad_scan["ref_spacing_px"]

        current = self.current_frame_idx()
        if current in self.bad_index_set:
            self.focus_indices = build_focus_indices(len(self.vc.frames), self.bad_scan["bad_indices"], self.context)
            self.focus_pos = self.focus_indices.index(current) if current in self.focus_indices else 0
        elif self.bad_scan["bad_indices"]:
            self.focus_indices = build_focus_indices(len(self.vc.frames), self.bad_scan["bad_indices"], self.context)
            if current in self.focus_indices:
                self.focus_pos = self.focus_indices.index(current)
            else:
                self.focus_pos = 0
        else:
            self.focus_indices = [current]
            self.focus_pos = 0

    def prev_next_good(self, frame_idx: int) -> tuple[Optional[int], Optional[int]]:
        return find_prev_next_good(frame_idx, self.bad_index_set, len(self.vc.frames))

    def validate(self) -> None:
        self.refresh_bad_scan()
        if not self.bad_scan["bad_indices"]:
            self.last_message = "Validation clean. No bad frames remain."
        else:
            idx = self.current_frame_idx()
            diag = self.diag_by_idx.get(idx)
            summary = diagnostic_summary(diag)
            self.last_message = f"Validation: {self.bad_scan['n_bad']} bad frames remain. {summary or ''}".strip()

    def undo(self) -> None:
        frame_idx = self.current_frame_idx()
        stack = self.undo_stack.get(frame_idx)
        if not stack:
            self.last_message = "Undo stack empty for current frame."
            return
        self.vc.frames[frame_idx].detections = stack.pop()
        normalize_frame_detections(self.vc, frame_idx)
        self.selected_idx = None
        self.mark_edit("undo", frame_idx)
        self.last_message = f"Undid last edit on frame {frame_idx}."

    def save_log(self) -> None:
        payload = {
            "dataset": self.name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "output_path": self.out_path,
            "remaining_bad_indices": self.bad_scan["bad_indices"],
            "edits": self.edit_log,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def save(self, require_clean: bool) -> None:
        self.validate()
        if require_clean and self.bad_scan["bad_indices"]:
            self.last_message = "Refusing final save: bad frames remain. Use s for a draft save."
            return
        save_vc(self.out_path, self.vc)
        self.save_log()
        mode = "final" if require_clean else "draft"
        self.last_message = f"Saved {mode} repair file to {self.out_path}"

    def add_detection(self, x: float, y: float) -> None:
        frame_idx = self.current_frame_idx()
        self.push_undo(frame_idx)
        det = DetectionRecord(
            x=float(x),
            y=float(y),
            color="b",
            area=float(median_area_from_neighbors(self.vc, frame_idx)),
            angle=float(infer_angle(self.vc, frame_idx, x)),
        )
        self.vc.frames[frame_idx].detections.append(det)
        normalize_frame_detections(self.vc, frame_idx)
        self.selected_idx = nearest_detection_index(self.vc.frames[frame_idx].detections, x, y, 1e9)
        self.mark_edit("add_detection", frame_idx, {"x": float(x), "y": float(y)})
        self.last_message = f"Added detection at ({x:.1f}, {y:.1f}) on frame {frame_idx}."

    def delete_selected(self) -> None:
        frame_idx = self.current_frame_idx()
        dets = self.vc.frames[frame_idx].detections
        if self.selected_idx is None or self.selected_idx >= len(dets):
            self.last_message = "No selected detection to delete."
            return
        self.push_undo(frame_idx)
        deleted = dets.pop(self.selected_idx)
        normalize_frame_detections(self.vc, frame_idx)
        self.selected_idx = None
        self.mark_edit("delete_detection", frame_idx, {"x": float(deleted.x), "y": float(deleted.y)})
        self.last_message = f"Deleted detection from frame {frame_idx}."

    def set_selected_angle_delta(self, delta_deg: float) -> None:
        frame_idx = self.current_frame_idx()
        dets = self.vc.frames[frame_idx].detections
        if self.selected_idx is None or self.selected_idx >= len(dets):
            self.last_message = "No selected detection for angle edit."
            return
        self.push_undo(frame_idx)
        dets[self.selected_idx].angle = float(dets[self.selected_idx].angle + math.radians(delta_deg))
        self.mark_edit("rotate_angle", frame_idx, {"delta_deg": float(delta_deg)})
        self.last_message = f"Rotated selected detection by {delta_deg:+.1f} deg."

    def scale_selected_area(self, factor: float) -> None:
        frame_idx = self.current_frame_idx()
        dets = self.vc.frames[frame_idx].detections
        if self.selected_idx is None or self.selected_idx >= len(dets):
            self.last_message = "No selected detection for area edit."
            return
        self.push_undo(frame_idx)
        dets[self.selected_idx].area = float(max(1.0, dets[self.selected_idx].area * factor))
        self.mark_edit("scale_area", frame_idx, {"factor": float(factor)})
        self.last_message = f"Scaled selected area by {factor:.3f}."

    def copy_anchor(self, source_idx: Optional[int], label: str) -> None:
        frame_idx = self.current_frame_idx()
        if source_idx is None:
            self.last_message = f"No {label} good frame available."
            return
        self.push_undo(frame_idx)
        self.vc.frames[frame_idx].detections = clone_detections(self.vc.frames[source_idx].detections)
        normalize_frame_detections(self.vc, frame_idx)
        self.selected_idx = 0 if self.vc.frames[frame_idx].detections else None
        self.mark_edit(f"copy_{label}", frame_idx, {"source_frame": int(source_idx)})
        self.last_message = f"Copied detections from {label} good frame {source_idx}."

    def interpolate_current(self) -> None:
        frame_idx = self.current_frame_idx()
        prev_good, next_good = self.prev_next_good(frame_idx)
        self.push_undo(frame_idx)
        self.vc.frames[frame_idx].detections = interpolate_from_anchors(self.vc, frame_idx, prev_good, next_good)
        normalize_frame_detections(self.vc, frame_idx)
        self.selected_idx = 0 if self.vc.frames[frame_idx].detections else None
        self.mark_edit("interpolate", frame_idx, {"prev_good": prev_good, "next_good": next_good})
        self.last_message = f"Interpolated frame {frame_idx} from anchors {prev_good} / {next_good}."

    def select_next_detection(self, delta: int) -> None:
        dets = self.current_frame().detections
        if not dets:
            self.selected_idx = None
            self.last_message = "Current frame has no detections."
            return
        if self.selected_idx is None:
            self.selected_idx = 0
        else:
            self.selected_idx = min(max(0, self.selected_idx + delta), len(dets) - 1)

    def step_focus(self, delta: int) -> None:
        self.focus_pos = min(max(0, self.focus_pos + delta), len(self.focus_indices) - 1)
        self.selected_idx = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        frame_idx = self.current_frame_idx()
        dets = self.vc.frames[frame_idx].detections

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = nearest_detection_index(dets, x, y, SELECT_RADIUS_PX)
            if hit is not None:
                self.selected_idx = hit
                self.dragging = True
                self.push_undo(frame_idx)
                self.last_message = f"Selected detection {hit} on frame {frame_idx}."
            else:
                self.add_detection(float(x), float(y))

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.selected_idx is not None and self.selected_idx < len(dets):
                dets[self.selected_idx].x = float(x)
                dets[self.selected_idx].y = float(y)
                normalize_frame_detections(self.vc, frame_idx)
                self.selected_idx = nearest_detection_index(self.vc.frames[frame_idx].detections, x, y, 1e9)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.mark_edit("move_detection", frame_idx)
            self.last_message = f"Moved detection on frame {frame_idx}."

        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = nearest_detection_index(dets, x, y, SELECT_RADIUS_PX)
            if hit is not None:
                self.selected_idx = hit
                self.delete_selected()

    def draw(self) -> np.ndarray:
        frame_idx = self.current_frame_idx()
        frame_data = self.vc.frames[frame_idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data.frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read video frame {frame_data.frame_number}")

        overlay = frame.copy()
        prev_good, next_good = self.prev_next_good(frame_idx)
        if prev_good is not None:
            draw_reference_detections(overlay, self.vc.frames[prev_good].detections, (0, 180, 255), f"prev {prev_good}")
        if next_good is not None:
            draw_reference_detections(overlay, self.vc.frames[next_good].detections, (255, 180, 0), f"next {next_good}")
        if self.reference_vc is not None and frame_idx < len(self.reference_vc.frames):
            draw_reference_detections(overlay, self.reference_vc.frames[frame_idx].detections, (255, 255, 0), "ref")

        dets = self.vc.frames[frame_idx].detections
        for idx, det in enumerate(dets):
            draw_detection(overlay, det, idx, (0, 0, 255), idx == self.selected_idx)

        if len(dets) >= 2 and np.isfinite(self.ref_spacing):
            xs = np.array([d.x for d in sorted(dets, key=lambda d: d.x)], dtype=float)
            dx = np.diff(xs)
            ratio = dx / self.ref_spacing
            dx_text = f"dx med/min/max: {np.median(dx):.1f} / {dx.min():.1f} / {dx.max():.1f}"
            ratio_text = f"ratio min/max: {ratio.min():.2f} / {ratio.max():.2f}"
            cv2.putText(overlay, dx_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, ratio_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        status = "BAD" if frame_idx in self.bad_index_set else "context"
        run_label = frame_run_label(frame_idx, self.bad_runs)
        diag_text = diagnostic_summary(self.diag_by_idx.get(frame_idx))
        selected = dets[self.selected_idx] if self.selected_idx is not None and self.selected_idx < len(dets) else None

        cv2.putText(overlay, f"{self.name} frame={frame_idx} n={len(dets)} {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if status == "BAD" else (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"focus {self.focus_pos + 1}/{len(self.focus_indices)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
        if run_label is not None:
            cv2.putText(overlay, f"segment {run_label}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if diag_text is not None:
            cv2.putText(overlay, f"reason: {diag_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 220, 255), 2, cv2.LINE_AA)
        if selected is not None:
            selected_text = (
                f"selected {self.selected_idx}: x={selected.x:.1f} y={selected.y:.1f} "
                f"area={selected.area:.1f} angle={math.degrees(selected.angle):.1f}deg"
            )
            cv2.putText(overlay, selected_text, (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        help_lines = [
            "mouse: left select/drag or add, right delete",
            "a/d prev-next focus | z/x select det | ,/. angle | -/= area | i interpolate",
            "p copy prev good | n copy next good | u undo | v validate | s draft save | f final save | q quit",
            self.last_message,
        ]
        y0 = overlay.shape[0] - 95
        for i, text in enumerate(help_lines):
            cv2.putText(overlay, text, (20, y0 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay

    def close(self) -> None:
        self.cap.release()


def resolve_name_and_video(name_arg: str) -> tuple[str, str]:
    video_path = find_video(name_arg, "Videos")
    if video_path is None:
        raise FileNotFoundError(f"Could not find video for {name_arg} in Videos/")
    return video_name(video_path), video_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually repair black-track frames that remain bad after auto-repair."
    )
    parser.add_argument("name", help="Video name or suffix, e.g. IMG_9282 or 9282")
    parser.add_argument("--context", type=int, default=2, help="Extra context frames around unrepaired frames.")
    parser.add_argument("--out", help="Output path for manual-repaired track1 msgpack.")
    parser.add_argument("--log", help="Output path for manual repair JSON log.")
    parser.add_argument("--reference", help="Optional reference track1 path or dataset name to overlay.")
    args = parser.parse_args()

    try:
        name, video_path = resolve_name_and_video(args.name)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    track1_path = track1_output_path(name)
    if not os.path.exists(track1_path):
        print(f"Error: track1 output not found: {track1_path}")
        sys.exit(1)

    out_path, log_path = default_manual_paths(name)
    if args.out:
        out_path = args.out
    if args.log:
        log_path = args.log
    backup_path = default_backup_track1_path(name)

    print(f"Loading {track1_path}…")
    vc = load_vc(track1_path)
    print(f"  {len(vc.frames)} frames loaded.")
    print("Running auto-repair in memory…")

    try:
        vb.verify_and_sanitize(vc, repair=True, quiet=True)
    except RuntimeError as exc:
        print(f"  Auto-repair incomplete: {exc}")
    else:
        print("  Auto-repair succeeded with no remaining bad frames.")

    bad_scan = vb.scan_bad_frames_detailed(vc)
    if not bad_scan["bad_indices"]:
        print("No unrepaired bad frames remain after auto-repair.")
        ans = input(
            f"Promote the auto-repaired result to working track1.msgpack now?\n"
            f"  current : {track1_path}\n"
            f"  backup  : {backup_path}\n"
            f"[y/N] "
        ).strip().lower()
        if ans == "y":
            promote_repaired_track1(
                name=name,
                working_track1_path=track1_path,
                backup_path=backup_path,
                repaired_output_path=out_path,
                vc=vc,
            )
        else:
            print(f"You can still save/promote the repaired result manually later if desired: {out_path}")
        sys.exit(0)

    focus_indices = build_focus_indices(len(vc.frames), bad_scan["bad_indices"], max(0, args.context))
    print(
        f"Manual repair mode: {bad_scan['n_bad']} bad frames in {bad_scan['n_segments']} segments. "
        f"Showing {len(focus_indices)} frames including context={max(0, args.context)}."
    )

    reference_vc = None
    if args.reference:
        ref_path = resolve_reference_path(args.reference)
        print(f"Loading reference overlay: {ref_path}")
        reference_vc = load_vc(ref_path)

    ui = RepairUI(
        name=name,
        vc=vc,
        video_path=video_path,
        out_path=out_path,
        log_path=log_path,
        bad_scan=bad_scan,
        focus_indices=focus_indices,
        reference_vc=reference_vc,
    )
    ui.context = max(0, args.context)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    cv2.setMouseCallback(WIN_NAME, ui.on_mouse)

    try:
        while True:
            overlay = ui.draw()
            cv2.imshow(WIN_NAME, overlay)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("a"):
                ui.step_focus(-1)
            elif key == ord("d"):
                ui.step_focus(1)
            elif key == ord("z"):
                ui.select_next_detection(-1)
            elif key == ord("x"):
                ui.select_next_detection(1)
            elif key == ord("s"):
                ui.save(require_clean=False)
            elif key == ord("f"):
                ui.save(require_clean=True)
            elif key == ord("u"):
                ui.undo()
            elif key == ord("v"):
                ui.validate()
            elif key == ord("i"):
                ui.interpolate_current()
            elif key == ord("p"):
                prev_good, _ = ui.prev_next_good(ui.current_frame_idx())
                ui.copy_anchor(prev_good, "prev")
            elif key == ord("n"):
                _, next_good = ui.prev_next_good(ui.current_frame_idx())
                ui.copy_anchor(next_good, "next")
            elif key == ord(","):
                ui.set_selected_angle_delta(-ANGLE_STEP_DEG)
            elif key == ord("."):
                ui.set_selected_angle_delta(+ANGLE_STEP_DEG)
            elif key == ord("-"):
                ui.scale_selected_area(1.0 / AREA_STEP)
            elif key == ord("="):
                ui.scale_selected_area(AREA_STEP)
            elif key == 8 or key == 127:
                ui.delete_selected()

    finally:
        ui.close()
        cv2.destroyAllWindows()

    if os.path.exists(out_path):
        ui.validate()
        if not ui.bad_scan["bad_indices"]:
            ans = input(
                f"\nPromote repaired file to working track1.msgpack?\n"
                f"  current : {track1_path}\n"
                f"  repaired: {out_path}\n"
                f"  backup  : {backup_path}\n"
                f"[y/N] "
            ).strip().lower()
            if ans == "y":
                promote_repaired_track1(
                    name=name,
                    working_track1_path=track1_path,
                    backup_path=backup_path,
                    repaired_output_path=out_path,
                    vc=ui.vc,
                )
            else:
                print("Leaving repaired file separate; workflow not promoted.")


if __name__ == "__main__":
    main()
