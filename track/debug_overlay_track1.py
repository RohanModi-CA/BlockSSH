
#!/usr/bin/env python3
"""
debug_overlay_track1.py

Visualize Step 1 detections directly on the video.

Shows:
- centroids
- indices (left→right)
- spacing lines between neighbors
- optional live spacing stats

Usage
-----
python3 debug_overlay_track1.py IMG_0662
python3 debug_overlay_track1.py IMG_0662 --no-lines
python3 debug_overlay_track1.py IMG_0662 --print-spacing
"""

import os
import sys
import argparse
import msgpack
import cv2
import numpy as np

from tracking_classes import VideoCentroids
from helper.verification_black import scan_bad_frames_detailed, verify_and_sanitize


WIN_W = 1400
WIN_H = 800


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_video(name: str, video_dir="Videos"):
    name = os.path.splitext(os.path.basename(name))[0]

    for f in os.listdir(video_dir):
        base = os.path.splitext(f)[0]
        if base == name or base.endswith(name):
            return os.path.join(video_dir, f)

    return None


def track1_path(name: str):
    name = os.path.splitext(os.path.basename(name))[0]
    return os.path.join("data", name, "track1.msgpack")


def load_vc(path: str) -> VideoCentroids:
    with open(path, "rb") as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def spacing_stats(detections, ref_spacing=None):
    dets = sorted(detections, key=lambda d: d.x)
    if len(dets) < 2:
        return None

    xs = np.array([d.x for d in dets], dtype=float)
    dx = np.diff(xs)
    out = {
        "median_dx": float(np.median(dx)),
        "min_dx": float(dx.min()),
        "max_dx": float(dx.max()),
    }
    if ref_spacing is not None and np.isfinite(ref_spacing) and ref_spacing > 0:
        ratio = dx / ref_spacing
        out["min_ratio"] = float(ratio.min())
        out["max_ratio"] = float(ratio.max())
    return out


def build_focus_indices(n_frames, bad_indices, context):
    focus = set()
    for idx in bad_indices:
        lo = max(0, idx - context)
        hi = min(n_frames - 1, idx + context)
        for k in range(lo, hi + 1):
            focus.add(k)
    return sorted(focus)


def frame_run_label(frame_idx, bad_runs):
    for run_idx, (start, end) in enumerate(bad_runs, start=1):
        if start <= frame_idx <= end:
            return f"{run_idx}/{len(bad_runs)} [{start},{end}]"
    return None


def diagnostic_summary(diag):
    if not diag:
        return None
    reason = diag.get("reason", "unknown")
    details = diag.get("details", {})

    if reason == "count_jump":
        return (
            f"reason: count jump from prev good frame {diag.get('prev_good_idx')} "
            f"({details.get('prev_good_count')} -> {diag.get('count')}, "
            f"delta={details.get('count_delta')})"
        )
    if reason == "transition_cost":
        return (
            f"reason: transition cost {details.get('transition_cost', float('nan')):.3f} "
            f"> limit {details.get('transition_limit', float('nan')):.3f} "
            f"({details.get('transition_kind', 'unknown')})"
        )
    if reason == "spacing_ratio":
        return (
            f"reason: spacing ratio out of bounds "
            f"({details.get('ratio_min', float('nan')):.2f} .. {details.get('ratio_max', float('nan')):.2f})"
        )
    if reason == "no_valid_transition":
        return f"reason: no valid transition from prev good frame {diag.get('prev_good_idx')}"
    if reason == "empty_after_start":
        return "reason: empty frame after detections started"
    return f"reason: {reason}"


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

def draw_overlay(frame, detections, draw_lines=True):
    """
    Draw centroids + optional spacing lines.
    """
    h, w = frame.shape[:2]

    # sort just in case
    dets = sorted(detections, key=lambda d: d.x)

    xs = [d.x for d in dets]
    ys = [d.y for d in dets]

    # draw lines between neighbors
    if draw_lines and len(dets) >= 2:
        for i in range(len(dets) - 1):
            x1, y1 = int(xs[i]), int(ys[i])
            x2, y2 = int(xs[i+1]), int(ys[i+1])

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            dx = xs[i+1] - xs[i]
            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)

            cv2.putText(
                frame,
                f"{dx:.1f}",
                (mx, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    # draw points
    for i, d in enumerate(dets):
        x = int(round(d.x))
        y = int(round(d.y))

        cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)

        cv2.putText(
            frame,
            f"{i}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"({d.x:.0f},{d.y:.0f})",
            (x + 10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 255),
            1,
            cv2.LINE_AA
        )

    return frame


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Video name or suffix")
    parser.add_argument("--no-lines", action="store_true")
    parser.add_argument("--print-spacing", action="store_true")
    parser.add_argument(
        "--bad-only",
        action="store_true",
        help="Only step through frames flagged by black-track verification, with optional context.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Extra clean frames to include before/after each bad frame when using --bad-only.",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.50,
        help="Min spacing ratio used by the bad-frame scan.",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=1.50,
        help="Max spacing ratio used by the bad-frame scan.",
    )
    parser.add_argument(
        "--after-repair",
        action="store_true",
        help="Run the in-memory repair first, then show only frames that still fail final validation.",
    )
    args = parser.parse_args()

    video_path = find_video(args.name)
    if video_path is None:
        print("Video not found.")
        sys.exit(1)

    t1_path = track1_path(args.name)
    if not os.path.exists(t1_path):
        print("track1.msgpack not found.")
        sys.exit(1)

    print(f"Loading {t1_path}…")
    vc = load_vc(t1_path)

    print(f"{len(vc.frames)} frames")

    if args.after_repair:
        try:
            verify_and_sanitize(
                vc,
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
                repair=True,
                quiet=True,
            )
            print("Repair succeeded cleanly; no post-repair failures remain.")
        except RuntimeError as exc:
            print(f"Post-repair validation still failed: {exc}")

    bad_scan = None
    focus_indices = None
    focus_pos = {}
    bad_index_set = set()
    bad_run_lookup = []
    diag_by_idx = {}

    if args.bad_only or args.after_repair:
        bad_scan = scan_bad_frames_detailed(
            vc,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
        )
        bad_index_set = set(bad_scan["bad_indices"])
        bad_run_lookup = bad_scan["bad_runs"]
        diag_by_idx = {d["frame_idx"]: d for d in bad_scan["diagnostics"] if d["is_bad"]}

        if not bad_index_set:
            if args.after_repair:
                print("No bad frames remain after repair for the requested thresholds.")
            else:
                print("No bad frames found for the requested thresholds.")
            sys.exit(0)

        focus_indices = build_focus_indices(len(vc.frames), bad_scan["bad_indices"], max(0, args.context))
        focus_pos = {frame_idx: pos for pos, frame_idx in enumerate(focus_indices)}

        mode_label = "Post-repair failure mode" if args.after_repair else "Bad-frame focus mode"
        print(
            f"{mode_label}: {bad_scan['n_bad']} bad frames in "
            f"{bad_scan['n_segments']} segments, ref spacing={bad_scan['ref_spacing_px']:.4f}px"
        )
        print(f"Showing {len(focus_indices)} frames including context={max(0, args.context)}.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        sys.exit(1)

    win = "Track1 Debug Overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    idx = focus_indices[0] if focus_indices else 0

    while True:
        if idx >= len(vc.frames):
            break

        frame_data = vc.frames[idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data.frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        dets = frame_data.detections

        stats = spacing_stats(
            dets,
            ref_spacing=None if bad_scan is None else bad_scan["ref_spacing_px"],
        )

        if args.print_spacing and stats is not None:
            ratio_part = ""
            if "min_ratio" in stats and "max_ratio" in stats:
                ratio_part = (
                    f" | ratio min/max={stats['min_ratio']:.2f}/{stats['max_ratio']:.2f}"
                )
            print(
                f"frame {idx:5d} | n={len(dets):2d} | "
                f"median dx={stats['median_dx']:.1f} | "
                f"min={stats['min_dx']:.1f} | max={stats['max_dx']:.1f}"
                f"{ratio_part}"
            )

        overlay = draw_overlay(frame.copy(), dets, draw_lines=not args.no_lines)

        is_bad = idx in bad_index_set
        status = "BAD" if is_bad else "context" if focus_indices else "normal"
        color = (0, 0, 255) if is_bad else (0, 200, 255) if focus_indices else (255, 0, 255)

        cv2.putText(
            overlay,
            f"frame={idx}  n={len(dets)}  {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )

        if focus_indices:
            pos = focus_pos[idx] + 1
            cv2.putText(
                overlay,
                f"focus {pos}/{len(focus_indices)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

        run_label = frame_run_label(idx, bad_run_lookup)
        if run_label is not None:
            cv2.putText(
                overlay,
                f"segment {run_label}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        diag = diag_by_idx.get(idx)
        diag_text = diagnostic_summary(diag)
        if diag_text is not None:
            cv2.putText(
                overlay,
                diag_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 220, 255),
                2,
                cv2.LINE_AA
            )

        if stats is not None:
            cv2.putText(
                overlay,
                f"dx med/min/max = {stats['median_dx']:.1f} / {stats['min_dx']:.1f} / {stats['max_dx']:.1f}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            if "min_ratio" in stats and "max_ratio" in stats:
                cv2.putText(
                    overlay,
                    (
                        f"ratio min/max = {stats['min_ratio']:.2f} / {stats['max_ratio']:.2f}   "
                        f"bounds = [{args.ratio_min:.2f}, {args.ratio_max:.2f}]"
                    ),
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        cv2.putText(
            overlay,
            "a/d: prev/next   q: quit   space: skip ahead",
            (20, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(win, overlay)

        key = cv2.waitKey(30) & 0xFF

        def step(delta):
            nonlocal idx
            if focus_indices:
                pos = focus_pos[idx]
                pos = min(max(0, pos + delta), len(focus_indices) - 1)
                idx = focus_indices[pos]
            else:
                idx = min(max(0, idx + delta), len(vc.frames) - 1)

        if key == ord('q'):
            break
        elif key == ord('d'):
            step(1)
        elif key == ord('a'):
            step(-1)
        elif key == ord(' '):
            step(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
