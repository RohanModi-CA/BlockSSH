"""
helper/permanence_black.py
Build stable X, Y, and angle permanence matrices for black-on-white videos.

Compared with the original colour-based permanence builder, this module:
- ignores colour completely
- treats detections as an ordered 1-D chain in x
- uses dynamic programming over the visible interval offset to resolve all
  left-vs-right entry/exit choices globally rather than greedily
- emits one stable identity solution and then exports it three times:
    * X permanence file      (x stored in xPositions)
    * Y permanence file      (y stored in xPositions, for downstream compatibility)
    * angle permanence file  (angle stored in xPositions, for downstream compatibility)
- optionally trims unreliable columns, but only from the two ends

Public API
----------
build_permanence_xya(vc, tracking_results_path="", quiet=False, trim_weak_ends=True,
                     min_end_support=3)
    -> (track2_x, track2_y, track2_a, meta)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from tracking_classes import DetectionRecord, Track2XPermanence, VideoCentroids


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

_Y_WEIGHT = 0.35
_AREA_WEIGHT = 0.10

# Column-quality screening after the global interval path is chosen.
# Only end columns may be trimmed away. Any unreliable interior column is fatal.
_COL_MIN_SUPPORT = 3
_MAX_COLUMN_STEP_X_RATIO = 1.75
_MAX_COLUMN_STEP_Y_RATIO = 2.50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_arrays(dets: List[DetectionRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([d.x for d in dets], dtype=float)
    y = np.array([d.y for d in dets], dtype=float)
    a = np.array([d.area for d in dets], dtype=float)
    return x, y, a


def _estimate_reference_spacing(frames) -> float:
    pooled = []
    for f in frames:
        dets = f.detections
        if len(dets) < 2:
            continue
        x, _, _ = _frame_arrays(dets)
        if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
            continue
        pooled.extend(np.diff(x).tolist())
    if not pooled:
        raise RuntimeError("Could not estimate reference spacing for permanence building.")
    ref = float(np.median(np.array(pooled, dtype=float)))
    if not np.isfinite(ref) or ref <= 0:
        raise RuntimeError("Reference spacing for permanence building is invalid.")
    return ref


def _y_norm_from_spacing(ref_spacing: float) -> float:
    return max(12.0, 0.35 * float(ref_spacing))


def _transition_cost(
    prev_dets: List[DetectionRecord],
    curr_dets: List[DetectionRecord],
    prev_L: int,
    curr_L: int,
    ref_spacing: float,
    y_norm: float,
) -> float:
    """
    Cost of placing the current ordered visible interval at curr_L given that the
    previous ordered visible interval was at prev_L.

    The overlap is computed in global-column space and compared in observed order.
    """
    n_prev = len(prev_dets)
    n_curr = len(curr_dets)
    prev_R = prev_L + n_prev - 1
    curr_R = curr_L + n_curr - 1

    inter_L = max(prev_L, curr_L)
    inter_R = min(prev_R, curr_R)
    if inter_R < inter_L:
        return float("inf")

    prev_s = inter_L - prev_L
    curr_s = inter_L - curr_L
    overlap = inter_R - inter_L + 1

    p = prev_dets[prev_s: prev_s + overlap]
    c = curr_dets[curr_s: curr_s + overlap]

    px, py, pa = _frame_arrays(p)
    cx, cy, ca = _frame_arrays(c)

    area_norm = max(1.0, float(np.median(np.r_[pa, ca])))
    dx_term = np.abs(cx - px) / ref_spacing
    dy_term = np.abs(cy - py) / y_norm
    da_term = np.abs(ca - pa) / area_norm

    base_cost = float(np.mean(dx_term + _Y_WEIGHT * dy_term + _AREA_WEIGHT * da_term))

    # Small soft priors help break exact symmetries without dominating the global fit.
    delta = n_curr - n_prev
    prior = 0.0
    if delta == 1 and curr_L == prev_L - 1:
        # left entry: the new block should be at least roughly left of the previous leftmost
        prior += 0.10 * max(0.0, (curr_dets[0].x - prev_dets[0].x) / ref_spacing)
    elif delta == 1 and curr_L == prev_L:
        # right entry: the new block should be roughly right of the previous rightmost
        prior += 0.10 * max(0.0, (prev_dets[-1].x - curr_dets[-1].x) / ref_spacing)
    elif delta == -1 and curr_L == prev_L + 1 and n_prev >= 2:
        # left exit
        prior += 0.05 * max(0.0, (prev_dets[1].x - curr_dets[0].x) / ref_spacing)
    elif delta == -1 and curr_L == prev_L and n_prev >= 2:
        # right exit
        prior += 0.05 * max(0.0, (curr_dets[-1].x - prev_dets[-2].x) / ref_spacing)

    return base_cost + prior


def _next_left_positions(prev_L: int, n_prev: int, n_curr: int) -> List[int]:
    delta = n_curr - n_prev
    if delta == 0:
        return [prev_L]
    if delta == 1:
        return [prev_L - 1, prev_L]
    if delta == -1:
        return [prev_L + 1, prev_L]
    return []


def _viterbi_left_positions(frames, ref_spacing: float, quiet: bool) -> List[int]:
    """
    Solve the entire left/right entry-exit sequence globally.

    State at frame t is the leftmost global column index L_t of the visible run.
    Width is observed and therefore not part of the state.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    y_norm = _y_norm_from_spacing(ref_spacing)
    first_nz = next((k for k, f in enumerate(frames) if len(f.detections) > 0), -1)
    if first_nz == -1:
        raise RuntimeError("No frame contains detections — cannot initialise permanence.")

    active_frames = frames[first_nz:]
    if any(len(f.detections) == 0 for f in active_frames):
        bad = first_nz + next(i for i, f in enumerate(active_frames) if len(f.detections) == 0)
        raise RuntimeError(f"Zero-detection frame at k={bad} after tracking has started.")

    states: Dict[int, float] = {0: 0.0}
    backptrs: List[Dict[int, int]] = []

    log(f"  First non-empty frame: {first_nz}  ({len(active_frames[0].detections)} initial blocks)")

    for t in range(1, len(active_frames)):
        prev_dets = active_frames[t - 1].detections
        curr_dets = active_frames[t].detections

        if abs(len(curr_dets) - len(prev_dets)) > 1:
            raise RuntimeError(
                f"Count jump > 1 between active frames {first_nz + t - 1} and {first_nz + t}: "
                f"{len(prev_dets)} -> {len(curr_dets)}."
            )

        new_states: Dict[int, float] = {}
        new_back: Dict[int, int] = {}

        for prev_L, prev_cost in states.items():
            for curr_L in _next_left_positions(prev_L, len(prev_dets), len(curr_dets)):
                tr_cost = _transition_cost(prev_dets, curr_dets, prev_L, curr_L, ref_spacing, y_norm)
                if not np.isfinite(tr_cost):
                    continue
                cost = prev_cost + tr_cost
                if curr_L not in new_states or cost < new_states[curr_L]:
                    new_states[curr_L] = cost
                    new_back[curr_L] = prev_L

        if not new_states:
            raise RuntimeError(
                f"No valid interval placement exists at frame {first_nz + t}."
            )

        states = new_states
        backptrs.append(new_back)

    final_L = min(states, key=states.get)
    total_cost = float(states[final_L])

    Ls = [0] * len(active_frames)
    Ls[-1] = final_L
    for t in range(len(active_frames) - 2, -1, -1):
        Ls[t] = backptrs[t][Ls[t + 1]]

    log(f"  Global interval path solved. Final DP cost: {total_cost:.4f}")

    # Prepend dummy entries for leading all-NaN frames.
    return [None] * first_nz + Ls  # type: ignore[list-item]


def _materialize_xya_matrices(frames, left_positions: List[int]) -> Tuple[List[List[float]], List[List[float]], List[List[float]], int]:
    """Build full X, Y, and angle matrices from the chosen interval path."""
    active = [(k, L) for k, L in enumerate(left_positions) if L is not None]
    if not active:
        raise RuntimeError("No active interval positions were produced.")

    min_L = min(L for _, L in active)
    max_R = max(L + len(frames[k].detections) - 1 for k, L in active)
    shift = -min_L
    n_cols = max_R - min_L + 1

    X = []
    Y = []
    A = []

    for k, f in enumerate(frames):
        row_x = [float("nan")] * n_cols
        row_y = [float("nan")] * n_cols
        row_a = [float("nan")] * n_cols

        L = left_positions[k]
        if L is not None and len(f.detections) > 0:
            start = L + shift
            for i, d in enumerate(f.detections):
                j = start + i
                row_x[j] = float(d.x)
                row_y[j] = float(d.y)
                row_a[j] = float(d.angle)

        X.append(row_x)
        Y.append(row_y)
        A.append(row_a)

    return X, Y, A, n_cols


def _column_bad_flags(
    X: List[List[float]],
    Y: List[List[float]],
    ref_spacing: float,
    min_support: int,
) -> List[bool]:
    """
    Flag columns that are too weak or too discontinuous.

    A flagged interior column is fatal. Flagged end columns may be trimmed.
    """
    arr_x = np.array(X, dtype=float)
    arr_y = np.array(Y, dtype=float)
    n_cols = arr_x.shape[1]

    max_step_x = _MAX_COLUMN_STEP_X_RATIO * ref_spacing
    max_step_y = _MAX_COLUMN_STEP_Y_RATIO * _y_norm_from_spacing(ref_spacing)

    bad = []
    for j in range(n_cols):
        col_x = arr_x[:, j]
        col_y = arr_y[:, j]
        vis = np.where(np.isfinite(col_x))[0]

        if len(vis) < min_support:
            bad.append(True)
            continue

        # Only assess continuity where the column is visible in adjacent frames.
        consec_mask = np.diff(vis) == 1
        if np.any(consec_mask):
            left_rows = vis[:-1][consec_mask]
            right_rows = vis[1:][consec_mask]
            step_x = np.abs(col_x[right_rows] - col_x[left_rows])
            step_y = np.abs(col_y[right_rows] - col_y[left_rows])
            if np.any(step_x > max_step_x) or np.any(step_y > max_step_y):
                bad.append(True)
                continue

        bad.append(False)

    return bad


def _trim_end_columns_only(
    X: List[List[float]],
    Y: List[List[float]],
    A: List[List[float]],
    ref_spacing: float,
    min_end_support: int,
    quiet: bool,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], int, int, int]:
    """
    Trim unreliable columns, but only from the left and/or right ends.

    If any unreliable interior column remains, raise.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    min_support = max(_COL_MIN_SUPPORT, int(min_end_support))
    bad = _column_bad_flags(X, Y, ref_spacing, min_support=min_support)
    n_cols = len(bad)

    if not any(bad):
        return X, Y, A, 0, 0, n_cols

    ltrim = 0
    while ltrim < n_cols and bad[ltrim]:
        ltrim += 1

    rtrim = 0
    while rtrim < (n_cols - ltrim) and bad[n_cols - 1 - rtrim]:
        rtrim += 1

    keep_start = ltrim
    keep_end = n_cols - rtrim

    if keep_start >= keep_end:
        raise RuntimeError("All columns would need to be trimmed; nothing reliable remains.")

    if any(bad[keep_start:keep_end]):
        bad_inner = [i for i in range(keep_start, keep_end) if bad[i]]
        preview = ", ".join(map(str, bad_inner[:10])) + ("…" if len(bad_inner) > 10 else "")
        raise RuntimeError(
            "Unreliable interior columns remain after end-trimming logic. "
            f"Interior bad columns: {preview}"
        )

    if ltrim or rtrim:
        log(f"  Trimming unreliable end columns — left: {ltrim}, right: {rtrim}")
        X = [row[keep_start:keep_end] for row in X]
        Y = [row[keep_start:keep_end] for row in Y]
        A = [row[keep_start:keep_end] for row in A]

    return X, Y, A, ltrim, rtrim, keep_end - keep_start


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_permanence_xya(
    vc: VideoCentroids,
    tracking_results_path: str = "",
    quiet: bool = False,
    trim_weak_ends: bool = True,
    min_end_support: int = 3,
) -> Tuple[Track2XPermanence, Track2XPermanence, Track2XPermanence, dict]:
    """
    Build one persistent identity solution, then export it three times:
      - X permanence output
      - Y permanence output
      - angle permanence output

    The returned Track2XPermanence objects all use the legacy field name
    xPositions so downstream code can remain untouched.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    if not vc.passedVerification:
        raise RuntimeError("vc.passedVerification is False — run verification first.")

    frames = vc.frames
    ref_spacing = _estimate_reference_spacing(frames)

    log(f"  Permanence reference spacing: {ref_spacing:.4f} px")

    left_positions = _viterbi_left_positions(frames, ref_spacing, quiet=quiet)
    X, Y, A, n_cols_full = _materialize_xya_matrices(frames, left_positions)

    ltrim = rtrim = 0
    n_cols_kept = n_cols_full
    if trim_weak_ends:
        X, Y, A, ltrim, rtrim, n_cols_kept = _trim_end_columns_only(
            X, Y, A, ref_spacing, min_end_support=min_end_support, quiet=quiet
        )

    block_labels = ["b"] * n_cols_kept

    track2_x = Track2XPermanence(
        originalVideoPath=vc.filepath,
        trackingResultsPath=tracking_results_path,
        blockColors=block_labels,
        xPositions=X,
        frameTimes_s=[f.frame_time_s for f in frames],
        frameNumbers=[f.frame_number for f in frames],
    )

    track2_y = Track2XPermanence(
        originalVideoPath=vc.filepath,
        trackingResultsPath=tracking_results_path,
        blockColors=block_labels,
        xPositions=Y,
        frameTimes_s=[f.frame_time_s for f in frames],
        frameNumbers=[f.frame_number for f in frames],
    )

    track2_a = Track2XPermanence(
        originalVideoPath=vc.filepath,
        trackingResultsPath=tracking_results_path,
        blockColors=block_labels,
        xPositions=A,
        frameTimes_s=[f.frame_time_s for f in frames],
        frameNumbers=[f.frame_number for f in frames],
    )

    meta = {
        "reference_spacing_px": float(ref_spacing),
        "n_cols_full": int(n_cols_full),
        "n_cols_kept": int(n_cols_kept),
        "trimmed_left": int(ltrim),
        "trimmed_right": int(rtrim),
    }
    return track2_x, track2_y, track2_a, meta
