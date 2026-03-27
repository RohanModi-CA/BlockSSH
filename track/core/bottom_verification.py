"""
track/core/bottom_verification.py
Verification and interpolation-repair for black-on-white centroid detections.

This is the colour-free analogue of helper/verification.py.

Public API
----------
scan_bad_frames(vc, ratio_min, ratio_max)
    -> (n_bad, n_segments, ref_spacing_px)

verify_and_sanitize(vc, ratio_min, ratio_max, repair, quiet)
    -> (vc_modified, summary)

Design notes
------------
- colour is ignored entirely
- x ordering remains a hard invariant
- spacing in x remains the main geometric invariant
- y is used as a secondary temporal-consistency signal
- angle is carried through and repaired using doubled-angle interpolation
- repair is anchor-based interpolation across bad runs
- repair is conservative: if no plausible anchor alignment exists, the run is left
  untouched and final validation will fail
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from track.tracking_classes import VideoCentroids, DetectionRecord


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Transition-cost thresholds for marking bad frames.
# Costs are normalized by reference spacing and a y normalization scale.
_MAX_PAIR_COST_SAME = 2.50
_MAX_PAIR_COST_COUNT_CHANGE = 3.25

# Secondary weighting terms inside the normalized pair cost.
_Y_WEIGHT = 0.35
_AREA_WEIGHT = 0.10

# Repair guardrails
_MIN_OVERLAP_FOR_REPAIR = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarize_bad_runs(bad_indices: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start, end) pairs for contiguous bad runs."""
    if len(bad_indices) == 0:
        return []
    diff = np.diff(bad_indices)
    run_ends = np.where(diff > 1)[0]
    starts = np.insert(bad_indices[run_ends + 1], 0, bad_indices[0])
    ends = np.append(bad_indices[run_ends], bad_indices[-1])
    return list(zip(starts.tolist(), ends.tolist()))


def _frame_arrays(dets: List[DetectionRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([d.x for d in dets], dtype=float)
    y = np.array([d.y for d in dets], dtype=float)
    a = np.array([d.area for d in dets], dtype=float)
    return x, y, a


def _basic_frame_ok(dets: List[DetectionRecord]) -> bool:
    """
    Basic single-frame validity checks that do not use temporal context.
    """
    if len(dets) == 0:
        return True

    x, y, a = _frame_arrays(dets)

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(a)):
        return False

    if np.any(np.diff(x) <= 0):
        return False

    if np.any(a <= 0):
        return False

    return True


def _frame_spacing_ok(dets: List[DetectionRecord], ref_spacing: float,
                      ratio_min: float, ratio_max: float) -> bool:
    """Check x-spacing against the reference spacing."""
    if len(dets) < 2:
        return True

    x, _, _ = _frame_arrays(dets)
    dx = np.diff(x)
    if np.any(dx <= 0):
        return False

    sr = dx / ref_spacing
    return bool(np.all((sr >= ratio_min) & (sr <= ratio_max)))


def _collect_reference_spacings(
    frames,
    ratio_min: float,
    ratio_max: float,
    max_frames: int = 100,
) -> List[float]:
    """
    Pool spacing samples from geometrically clean frames.

    Compared with the old colour-based code, this does not require alternating
    colours; it only requires x ordering and internally self-consistent spacing.
    """
    out: List[float] = []
    n_used = 0

    for f in frames:
        dets = f.detections
        if len(dets) < 2:
            continue
        if not _basic_frame_ok(dets):
            continue

        x, _, _ = _frame_arrays(dets)
        dx = np.diff(x)
        med = float(np.median(dx))
        if med <= 0:
            continue

        ratios = dx / med
        if np.any((ratios < ratio_min) | (ratios > ratio_max)):
            continue

        out.extend(float(v) for v in dx)
        n_used += 1
        if n_used >= max_frames:
            break

    return out


def _find_reference_spacing(frames, ratio_min: float, ratio_max: float) -> Tuple[int, float]:
    """
    Find a usable reference frame and robust reference spacing.

    The reference spacing is the median pooled neighbour spacing over the first
    set of geometrically clean frames.
    """
    pooled = _collect_reference_spacings(frames, ratio_min, ratio_max)
    if not pooled:
        raise RuntimeError("Could not establish a reference x-spacing from the detections.")

    ref_spacing = float(np.median(np.array(pooled, dtype=float)))
    if not np.isfinite(ref_spacing) or ref_spacing <= 0:
        raise RuntimeError("Reference spacing is invalid.")

    for k, f in enumerate(frames):
        dets = f.detections
        if len(dets) < 2:
            continue
        if not _basic_frame_ok(dets):
            continue
        if not _frame_spacing_ok(dets, ref_spacing, ratio_min, ratio_max):
            continue
        return k, ref_spacing

    raise RuntimeError("Could not find a valid reference frame after estimating spacing.")


def _y_norm_from_spacing(ref_spacing: float) -> float:
    """
    Normalize y motion to a moderate physical scale.

    The black-only videos may still contain some y motion, but x spacing remains
    the primary invariant. This scale keeps y informative without overpowering x.
    """
    return max(12.0, 0.35 * float(ref_spacing))


def _pair_overlap_cost(
    prev_dets: List[DetectionRecord],
    curr_dets: List[DetectionRecord],
    prev_start: int,
    curr_start: int,
    overlap: int,
    ref_spacing: float,
    y_norm: float,
) -> float:
    """
    Cost for aligning two ordered slices of equal length.
    """
    if overlap <= 0:
        return float("inf")

    p = prev_dets[prev_start: prev_start + overlap]
    c = curr_dets[curr_start: curr_start + overlap]

    px, py, pa = _frame_arrays(p)
    cx, cy, ca = _frame_arrays(c)

    area_norm = max(1.0, float(np.median(np.r_[pa, ca])))

    dx_term = np.abs(cx - px) / ref_spacing
    dy_term = np.abs(cy - py) / y_norm
    da_term = np.abs(ca - pa) / area_norm

    return float(np.mean(dx_term + _Y_WEIGHT * dy_term + _AREA_WEIGHT * da_term))


def _best_transition(
    prev_dets: List[DetectionRecord],
    curr_dets: List[DetectionRecord],
    ref_spacing: float,
    y_norm: float,
) -> Optional[dict]:
    """
    Best local transition hypothesis between ordered detections.

    Allowed relations:
      - same count
      - left entry / right entry (count +1)
      - left exit  / right exit  (count -1)

    Returns a dict containing the best relation and its cost, or None if no
    relation is admissible.
    """
    n_prev = len(prev_dets)
    n_curr = len(curr_dets)
    delta = n_curr - n_prev

    if n_prev == 0 or n_curr == 0:
        return None

    candidates = []

    if delta == 0:
        cost = _pair_overlap_cost(prev_dets, curr_dets, 0, 0, n_prev, ref_spacing, y_norm)
        candidates.append({"kind": "same", "cost": cost, "overlap": n_prev})

    elif delta == 1:
        # new leftmost block
        cost_left = _pair_overlap_cost(prev_dets, curr_dets, 0, 1, n_prev, ref_spacing, y_norm)
        # new rightmost block
        cost_right = _pair_overlap_cost(prev_dets, curr_dets, 0, 0, n_prev, ref_spacing, y_norm)
        candidates.append({"kind": "left_entry", "cost": cost_left, "overlap": n_prev})
        candidates.append({"kind": "right_entry", "cost": cost_right, "overlap": n_prev})

    elif delta == -1:
        # old leftmost block disappeared
        cost_left = _pair_overlap_cost(prev_dets, curr_dets, 1, 0, n_curr, ref_spacing, y_norm)
        # old rightmost block disappeared
        cost_right = _pair_overlap_cost(prev_dets, curr_dets, 0, 0, n_curr, ref_spacing, y_norm)
        candidates.append({"kind": "left_exit", "cost": cost_left, "overlap": n_curr})
        candidates.append({"kind": "right_exit", "cost": cost_right, "overlap": n_curr})
    else:
        return None

    candidates = [c for c in candidates if np.isfinite(c["cost"])]
    if not candidates:
        return None

    candidates.sort(key=lambda d: (d["cost"], -d["overlap"]))
    best = candidates[0].copy()
    if len(candidates) > 1:
        best["second_best_cost"] = float(candidates[1]["cost"])
    else:
        best["second_best_cost"] = float("inf")
    return best


def _mark_bad_frames(frames, ref_spacing: float,
                     ratio_min: float, ratio_max: float) -> np.ndarray:
    """
    Mark bad frames using per-frame geometry and local temporal consistency.
    """
    n = len(frames)
    is_bad = np.zeros(n, dtype=bool)

    first_non_empty = -1
    prev_good_idx: Optional[int] = None
    y_norm = _y_norm_from_spacing(ref_spacing)

    for k, f in enumerate(frames):
        dets = f.detections

        if len(dets) > 0 and first_non_empty == -1:
            first_non_empty = k

        if len(dets) == 0:
            if first_non_empty != -1:
                is_bad[k] = True
            continue

        if not _basic_frame_ok(dets):
            is_bad[k] = True
            continue

        if not _frame_spacing_ok(dets, ref_spacing, ratio_min, ratio_max):
            is_bad[k] = True
            continue

        if prev_good_idx is not None:
            prev_dets = frames[prev_good_idx].detections
            delta = len(dets) - len(prev_dets)
            if abs(delta) > 1:
                is_bad[k] = True
                continue

            best = _best_transition(prev_dets, dets, ref_spacing, y_norm)
            if best is None:
                is_bad[k] = True
                continue

            limit = _MAX_PAIR_COST_SAME if delta == 0 else _MAX_PAIR_COST_COUNT_CHANGE
            if best["cost"] > limit:
                is_bad[k] = True
                continue

        prev_good_idx = k

    return is_bad


def _frame_failure_reason(dets: List[DetectionRecord]) -> Tuple[str, dict]:
    """
    Return a specific basic-frame failure reason and supporting details.
    """
    if len(dets) == 0:
        return "empty", {}

    x, y, a = _frame_arrays(dets)

    if np.any(~np.isfinite(x)):
        return "nonfinite_x", {}
    if np.any(~np.isfinite(y)):
        return "nonfinite_y", {}
    if np.any(~np.isfinite(a)):
        return "nonfinite_area", {}
    if np.any(np.diff(x) <= 0):
        return "nonincreasing_x", {}
    if np.any(a <= 0):
        return "nonpositive_area", {}

    return "unknown_basic_failure", {}


def _diagnose_frames(
    frames,
    ref_spacing: float,
    ratio_min: float,
    ratio_max: float,
) -> List[dict]:
    """
    Run the same marking logic as _mark_bad_frames, but retain reasons.
    """
    diagnostics: List[dict] = []
    first_non_empty = -1
    prev_good_idx: Optional[int] = None
    y_norm = _y_norm_from_spacing(ref_spacing)

    for k, f in enumerate(frames):
        dets = f.detections
        info = {
            "frame_idx": int(k),
            "frame_number": int(f.frame_number),
            "count": int(len(dets)),
            "is_bad": False,
            "reason": "ok",
            "details": {},
            "prev_good_idx": None if prev_good_idx is None else int(prev_good_idx),
        }

        if len(dets) > 0 and first_non_empty == -1:
            first_non_empty = k

        if len(dets) == 0:
            if first_non_empty != -1:
                info["is_bad"] = True
                info["reason"] = "empty_after_start"
            else:
                info["reason"] = "empty_before_start"
            diagnostics.append(info)
            continue

        if not _basic_frame_ok(dets):
            reason, details = _frame_failure_reason(dets)
            info["is_bad"] = True
            info["reason"] = reason
            info["details"] = details
            diagnostics.append(info)
            continue

        x, _, _ = _frame_arrays(dets)
        if len(dets) >= 2:
            dx = np.diff(x)
            ratios = dx / ref_spacing
            info["details"]["dx_min"] = float(dx.min())
            info["details"]["dx_max"] = float(dx.max())
            info["details"]["ratio_min"] = float(ratios.min())
            info["details"]["ratio_max"] = float(ratios.max())

        if not _frame_spacing_ok(dets, ref_spacing, ratio_min, ratio_max):
            info["is_bad"] = True
            info["reason"] = "spacing_ratio"
            diagnostics.append(info)
            continue

        if prev_good_idx is not None:
            prev_dets = frames[prev_good_idx].detections
            delta = len(dets) - len(prev_dets)
            info["details"]["prev_good_count"] = int(len(prev_dets))
            info["details"]["count_delta"] = int(delta)

            if abs(delta) > 1:
                info["is_bad"] = True
                info["reason"] = "count_jump"
                diagnostics.append(info)
                continue

            best = _best_transition(prev_dets, dets, ref_spacing, y_norm)
            if best is None:
                info["is_bad"] = True
                info["reason"] = "no_valid_transition"
                diagnostics.append(info)
                continue

            limit = _MAX_PAIR_COST_SAME if delta == 0 else _MAX_PAIR_COST_COUNT_CHANGE
            info["details"]["transition_kind"] = str(best["kind"])
            info["details"]["transition_cost"] = float(best["cost"])
            info["details"]["transition_limit"] = float(limit)
            info["details"]["transition_overlap"] = int(best["overlap"])
            info["details"]["transition_second_best_cost"] = float(best["second_best_cost"])

            if best["cost"] > limit:
                info["is_bad"] = True
                info["reason"] = "transition_cost"
                diagnostics.append(info)
                continue

        prev_good_idx = k
        diagnostics.append(info)

    return diagnostics


def _best_anchor_alignment(
    left_dets: List[DetectionRecord],
    right_dets: List[DetectionRecord],
    ref_spacing: float,
    y_norm: float,
) -> Optional[dict]:
    """
    Best geometry-only ordered alignment between the two anchor frames.

    Offset convention matches the old colour-based repair code:
      iR = iL - offset
    """
    if not left_dets or not right_dets:
        return None

    best = None

    for offset in range(-(len(right_dets) - 1), len(left_dets)):
        iL_s = max(0, offset)
        iL_e = min(len(left_dets), len(right_dets) + offset)
        if iL_s >= iL_e:
            continue
        iR_s = iL_s - offset
        iR_e = iL_e - offset

        overlap = iL_e - iL_s
        if overlap < _MIN_OVERLAP_FOR_REPAIR:
            continue

        cost = _pair_overlap_cost(
            left_dets, right_dets,
            prev_start=iL_s,
            curr_start=iR_s,
            overlap=overlap,
            ref_spacing=ref_spacing,
            y_norm=y_norm,
        )

        cand = {
            "offset": int(offset),
            "overlap": int(overlap),
            "cost": float(cost),
            "iL_s": int(iL_s),
            "iL_e": int(iL_e),
            "iR_s": int(iR_s),
            "iR_e": int(iR_e),
        }

        if best is None:
            best = cand
            continue

        if cand["overlap"] > best["overlap"]:
            best = cand
            continue

        if cand["overlap"] == best["overlap"] and cand["cost"] < best["cost"]:
            best = cand

    return best


def _ang_to_vec(theta: float) -> Tuple[float, float]:
    if not np.isfinite(theta):
        return float("nan"), float("nan")
    return float(np.cos(2.0 * theta)), float(np.sin(2.0 * theta))


def _repair_bad_run(
    frames,
    idx_start: int,
    idx_end: int,
    ref_spacing: float,
    quiet: bool,
) -> bool:
    """
    Repair one contiguous bad segment by interpolating between clean anchors.

    Returns True if repaired, False if the run could not be repaired.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    n_frames = len(frames)
    idxL = idx_start - 1
    idxR = idx_end + 1

    if idxL < 0 or idxR >= n_frames:
        log(f"    Skipping [{idx_start},{idx_end}]: touches the video boundary.")
        return False

    DL = frames[idxL].detections
    DR = frames[idxR].detections
    if not DL or not DR:
        log(f"    Skipping [{idx_start},{idx_end}]: one anchor frame is empty.")
        return False

    y_norm = _y_norm_from_spacing(ref_spacing)
    align = _best_anchor_alignment(DL, DR, ref_spacing, y_norm)
    if align is None:
        log(f"    Skipping [{idx_start},{idx_end}]: no plausible anchor alignment.")
        return False

    XL, YL, AL = _frame_arrays(DL)
    XR, YR, AR = _frame_arrays(DR)
    TL = np.array([d.angle for d in DL], dtype=float)
    TR = np.array([d.angle for d in DR], dtype=float)

    best_offset = align["offset"]
    overlap = align["overlap"]
    iL_s = align["iL_s"]
    iL_e = align["iL_e"]
    iR_s = align["iR_s"]
    iR_e = align["iR_e"]

    if overlap > 0:
        dx_avg = float(np.mean(XR[iR_s:iR_e] - XL[iL_s:iL_e]))
        dy_avg = float(np.mean(YR[iR_s:iR_e] - YL[iL_s:iL_e]))
        da_avg = float(np.mean(AR[iR_s:iR_e] - AL[iL_s:iL_e]))
    else:
        dx_avg = dy_avg = da_avg = 0.0

    # Union index space used for interpolation.
    shift = -min(0, best_offset)
    n_union = max(len(DL) + shift, len(DR) + best_offset + shift)

    U_XL = np.full(n_union, np.nan, dtype=float)
    U_XR = np.full(n_union, np.nan, dtype=float)
    U_YL = np.full(n_union, np.nan, dtype=float)
    U_YR = np.full(n_union, np.nan, dtype=float)
    U_AL = np.full(n_union, np.nan, dtype=float)
    U_AR = np.full(n_union, np.nan, dtype=float)

    U_CL = np.full(n_union, np.nan, dtype=float)
    U_SL = np.full(n_union, np.nan, dtype=float)
    U_CR = np.full(n_union, np.nan, dtype=float)
    U_SR = np.full(n_union, np.nan, dtype=float)

    for u in range(n_union):
        iL = u - shift
        iR = u - best_offset - shift
        inL = 0 <= iL < len(DL)
        inR = 0 <= iR < len(DR)

        if inL and inR:
            U_XL[u], U_XR[u] = XL[iL], XR[iR]
            U_YL[u], U_YR[u] = YL[iL], YR[iR]
            U_AL[u], U_AR[u] = AL[iL], AR[iR]

            cl, sl = _ang_to_vec(TL[iL])
            cr, sr = _ang_to_vec(TR[iR])
            U_CL[u], U_SL[u] = cl, sl
            U_CR[u], U_SR[u] = cr, sr

        elif inL:
            U_XL[u], U_XR[u] = XL[iL], XL[iL] + dx_avg
            U_YL[u], U_YR[u] = YL[iL], YL[iL] + dy_avg
            U_AL[u], U_AR[u] = AL[iL], max(1.0, AL[iL] + da_avg)

            cl, sl = _ang_to_vec(TL[iL])
            U_CL[u], U_SL[u] = cl, sl
            U_CR[u], U_SR[u] = cl, sl

        elif inR:
            U_XR[u], U_XL[u] = XR[iR], XR[iR] - dx_avg
            U_YR[u], U_YL[u] = YR[iR], YR[iR] - dy_avg
            U_AR[u], U_AL[u] = AR[iR], max(1.0, AR[iR] - da_avg)

            cr, sr = _ang_to_vec(TR[iR])
            U_CL[u], U_SL[u] = cr, sr
            U_CR[u], U_SR[u] = cr, sr

    tL = float(frames[idxL].frame_time_s)
    tR = float(frames[idxR].frame_time_s)
    denom = (tR - tL) if tR != tL else float(idxR - idxL)
    if denom == 0:
        log(f"    Skipping [{idx_start},{idx_end}]: anchor timestamps are degenerate.")
        return False

    for k in range(idx_start, idx_end + 1):
        if tR != tL:
            alpha = (float(frames[k].frame_time_s) - tL) / (tR - tL)
        else:
            alpha = (k - idxL) / (idxR - idxL)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        sk = int(round(shift + alpha * best_offset))
        ek = int(round(len(DL) + shift - 1 + alpha * (len(DR) - len(DL))))

        sk = max(0, sk)
        ek = min(n_union - 1, ek)
        if ek < sk:
            frames[k].detections = []
            continue

        new_dets = []
        for u in range(sk, ek + 1):
            x = float(U_XL[u] + alpha * (U_XR[u] - U_XL[u]))
            y = float(U_YL[u] + alpha * (U_YR[u] - U_YL[u]))
            a = float(U_AL[u] + alpha * (U_AR[u] - U_AL[u]))

            c2 = U_CL[u] + alpha * (U_CR[u] - U_CL[u])
            s2 = U_SL[u] + alpha * (U_SR[u] - U_SL[u])

            if np.isfinite(c2) and np.isfinite(s2) and (c2 != 0.0 or s2 != 0.0):
                angle = float(0.5 * np.arctan2(s2, c2))
            else:
                angle = float("nan")

            new_dets.append(DetectionRecord(
                x=x,
                y=y,
                color='b',
                area=max(1.0, a),
                angle=angle,
            ))

        new_dets.sort(key=lambda d: d.x)
        frames[k].detections = new_dets

    log(
        f"    Repaired [{idx_start},{idx_end}] with offset={best_offset}, "
        f"overlap={overlap}, cost={align['cost']:.4f}."
    )
    return True


# ---------------------------------------------------------------------------
# Public: non-destructive scan
# ---------------------------------------------------------------------------

def scan_bad_frames(
    vc: VideoCentroids,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
) -> Tuple[int, int, float]:
    """
    Quick non-destructive scan for bad frames.
    """
    frames = vc.frames
    if not frames:
        return 0, 0, float("nan")

    try:
        _, ref_spacing = _find_reference_spacing(frames, ratio_min, ratio_max)
    except RuntimeError:
        return 0, 0, float("nan")

    is_bad = _mark_bad_frames(frames, ref_spacing, ratio_min, ratio_max)
    bad_idx = np.where(is_bad)[0]
    bad_runs = _summarize_bad_runs(bad_idx)
    return int(len(bad_idx)), int(len(bad_runs)), float(ref_spacing)


def scan_bad_frames_detailed(
    vc: VideoCentroids,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
) -> dict:
    """
    Non-destructive scan that also returns the exact bad-frame indices and runs.
    """
    frames = vc.frames
    if not frames:
        return {
            "n_bad": 0,
            "n_segments": 0,
            "ref_spacing_px": float("nan"),
            "bad_indices": [],
            "bad_runs": [],
        }

    try:
        _, ref_spacing = _find_reference_spacing(frames, ratio_min, ratio_max)
    except RuntimeError:
        return {
            "n_bad": 0,
            "n_segments": 0,
            "ref_spacing_px": float("nan"),
            "bad_indices": [],
            "bad_runs": [],
        }

    diagnostics = _diagnose_frames(frames, ref_spacing, ratio_min, ratio_max)
    bad_idx = np.array([d["frame_idx"] for d in diagnostics if d["is_bad"]], dtype=int)
    bad_runs = _summarize_bad_runs(bad_idx)
    return {
        "n_bad": int(len(bad_idx)),
        "n_segments": int(len(bad_runs)),
        "ref_spacing_px": float(ref_spacing),
        "bad_indices": bad_idx.tolist(),
        "bad_runs": [(int(a), int(b)) for a, b in bad_runs],
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Public: full verify + optional sanitize
# ---------------------------------------------------------------------------

def verify_and_sanitize(
    vc: VideoCentroids,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
    repair: bool = True,
    quiet: bool = False,
) -> Tuple[VideoCentroids, dict]:
    """
    Full three-pass verification pipeline.

    Pass 0: establish reference spacing
    Pass 1: mark bad frames
    Pass 2: repair contiguous bad runs if requested
    Pass 3: final validation
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    frames = vc.frames
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("vc.frames is empty.")

    # ------------------------------------------------------------------
    # Pass 0: reference spacing
    # ------------------------------------------------------------------
    ref_idx, ref_spacing = _find_reference_spacing(frames, ratio_min, ratio_max)
    log(f"  Reference frame: {ref_idx}  |  spacing: {ref_spacing:.4f} px")

    # ------------------------------------------------------------------
    # Pass 1: initial marking
    # ------------------------------------------------------------------
    is_bad = _mark_bad_frames(frames, ref_spacing, ratio_min, ratio_max)
    bad_idx = np.where(is_bad)[0]
    bad_runs = _summarize_bad_runs(bad_idx)
    first_nz = next((k for k, f in enumerate(frames) if len(f.detections) > 0), -1)

    log(f"  First non-empty frame: {first_nz}")
    log(f"  Bad frames: {len(bad_idx)}  |  segments: {len(bad_runs)}")

    repaired_runs = 0
    repaired_frames = 0

    # ------------------------------------------------------------------
    # Pass 2: repair
    # ------------------------------------------------------------------
    if len(bad_runs) > 0 and repair:
        log(f"  Repairing {len(bad_idx)} frame(s) across {len(bad_runs)} segment(s)…")
        for idx_start, idx_end in bad_runs:
            ok = _repair_bad_run(frames, idx_start, idx_end, ref_spacing, quiet=quiet)
            if ok:
                repaired_runs += 1
                repaired_frames += (idx_end - idx_start + 1)

    # ------------------------------------------------------------------
    # Pass 3: final validation
    # ------------------------------------------------------------------
    log("  Final validation…")
    final_diag = _diagnose_frames(frames, ref_spacing, ratio_min, ratio_max)
    still_bad = np.array([d["frame_idx"] for d in final_diag if d["is_bad"]], dtype=int)
    if len(still_bad) > 0:
        diag_map = {d["frame_idx"]: d for d in final_diag if d["is_bad"]}
        preview_parts = []
        for idx in still_bad[:10]:
            reason = diag_map[int(idx)]["reason"]
            preview_parts.append(f"{int(idx)}({reason})")
        preview = ", ".join(preview_parts) + ("…" if len(still_bad) > 10 else "")
        raise RuntimeError(f"Verification failed — bad frames remain: {preview}")

    all_diffs = []
    for f in frames:
        if len(f.detections) >= 2:
            x, _, _ = _frame_arrays(f.detections)
            all_diffs.extend(np.diff(x).tolist())

    vc.passedVerification = True
    vc.meanBlockDistance = float(np.mean(all_diffs)) if all_diffs else 0.0

    summary = {
        "n_frames": n_frames,
        "reference_frame_idx": int(ref_idx),
        "reference_spacing_px": float(ref_spacing),
        "first_non_empty_idx": int(first_nz),
        "initial_bad_frames": int(len(bad_idx)),
        "bad_segments": int(len(bad_runs)),
        "sanitized_runs": int(repaired_runs),
        "sanitized_frames": int(repaired_frames),
        "final_mean_block_distance": float(vc.meanBlockDistance),
    }
    return vc, summary
