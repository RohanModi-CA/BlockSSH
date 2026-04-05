"""
helper/verification.py
Verification and interpolation-repair of Track1 centroid detections.

Public API
----------
scan_bad_frames(vc, ratio_min, ratio_max)
    → (n_bad: int, n_segments: int, ref_spacing_px: float)
    Non-destructive scan. Use this to ask the user before committing.

verify_and_sanitize(vc, ratio_min, ratio_max, repair, quiet)
    → (vc_modified: VideoCentroids, summary: dict)
    Full pipeline: passes 0-1 (detect), pass 2 (repair if requested), pass 3 (validate).
    Sets vc.passedVerification and vc.meanBlockDistance on success.
    Raises RuntimeError if final validation fails.
"""

import math
import numpy as np
from typing import List, Tuple
from tracking_classes import VideoCentroids, DetectionRecord


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarize_bad_runs(bad_indices: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive index pairs for contiguous bad runs."""
    if len(bad_indices) == 0:
        return []
    diff = np.diff(bad_indices)
    run_ends = np.where(diff > 1)[0]
    starts = np.insert(bad_indices[run_ends + 1], 0, bad_indices[0])
    ends   = np.append(bad_indices[run_ends], bad_indices[-1])
    return list(zip(starts.tolist(), ends.tolist()))


def _interp_angle(a_left: float, a_right: float, alpha: float) -> float:
    left_ok = np.isfinite(a_left)
    right_ok = np.isfinite(a_right)
    if left_ok and right_ok:
        z = (1.0 - alpha) * complex(math.cos(2.0 * a_left), math.sin(2.0 * a_left))
        z += alpha * complex(math.cos(2.0 * a_right), math.sin(2.0 * a_right))
        if abs(z) < 1e-12:
            return float("nan")
        return 0.5 * math.atan2(z.imag, z.real)
    if left_ok:
        return float(a_left)
    if right_ok:
        return float(a_right)
    return float("nan")


def _find_reference_spacing(frames, ratio_min, ratio_max) -> Tuple[int, float]:
    """
    Scan frames for the first one suitable as a spacing reference.
    Returns (ref_frame_idx, block_distance_px).
    Raises RuntimeError if no suitable frame is found.
    """
    for k, f in enumerate(frames):
        dets = f.detections
        if len(dets) < 2:
            continue
        x = np.array([d.x for d in dets], dtype=float)
        c = np.array([d.color for d in dets])
        if not np.all(np.isfinite(x)):     continue
        if np.any(np.diff(x) < 0):         continue
        if np.any(~np.isin(c, ['r', 'g'])): continue
        if np.any(c[:-1] == c[1:]):         continue
        return k, float(np.mean(np.diff(x)))
    raise RuntimeError("Could not find a valid reference frame to establish block spacing.")


def _mark_bad_frames(frames, ref_spacing: float, ratio_min: float, ratio_max: float) -> np.ndarray:
    """
    Return a boolean array of length len(frames).
    A frame is 'bad' if it is empty after detections have started, or if its
    detections fail the ordering / colour-alternation / spacing-ratio checks.
    """
    n = len(frames)
    is_bad = np.zeros(n, dtype=bool)
    first_nz = -1

    for k, f in enumerate(frames):
        dets = f.detections
        if len(dets) > 0 and first_nz == -1:
            first_nz = k

        if len(dets) == 0:
            if first_nz != -1:
                is_bad[k] = True
            continue

        x = np.array([d.x for d in dets], dtype=float)
        c = np.array([d.color for d in dets])

        if np.any(~np.isfinite(x)) or np.any(np.diff(x) < 0) or np.any(~np.isin(c, ['r', 'g'])):
            is_bad[k] = True
            continue

        if len(dets) >= 2:
            sr = np.diff(x) / ref_spacing
            if np.any(c[:-1] == c[1:]) or np.any((sr < ratio_min) | (sr > ratio_max)):
                is_bad[k] = True

    return is_bad


# ---------------------------------------------------------------------------
# Public: non-destructive scan
# ---------------------------------------------------------------------------

def scan_bad_frames(
    vc:        VideoCentroids,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
) -> Tuple[int, int, float]:
    """
    Quick scan for bad frames. Does NOT modify vc.

    Returns
    -------
    (n_bad, n_segments, ref_spacing_px)
    n_bad          — total number of bad frames found
    n_segments     — number of contiguous bad-frame runs
    ref_spacing_px — reference inter-block spacing used for ratio checks
    """
    frames = vc.frames
    if not frames:
        return 0, 0, float('nan')

    try:
        _, ref_spacing = _find_reference_spacing(frames, ratio_min, ratio_max)
    except RuntimeError:
        return 0, 0, float('nan')

    is_bad   = _mark_bad_frames(frames, ref_spacing, ratio_min, ratio_max)
    bad_idx  = np.where(is_bad)[0]
    bad_runs = _summarize_bad_runs(bad_idx)
    return int(len(bad_idx)), int(len(bad_runs)), ref_spacing


# ---------------------------------------------------------------------------
# Public: full verify + optional sanitize
# ---------------------------------------------------------------------------

def verify_and_sanitize(
    vc:        VideoCentroids,
    ratio_min: float = 0.50,
    ratio_max: float = 1.50,
    repair:    bool  = True,
    quiet:     bool  = False,
) -> Tuple[VideoCentroids, dict]:
    """
    Full three-pass verification pipeline.  Modifies vc.frames in-place when repair=True.

    Passes
    ------
    0 — Establish block-spacing reference from the first clean frame.
    1 — Mark every frame that fails ordering / colour / spacing checks.
    2 — (if repair=True) Interpolate across each bad segment.
    3 — Final re-check; raises RuntimeError if any bad frames remain.

    Sets vc.passedVerification = True and vc.meanBlockDistance on success.

    Returns (vc, summary_dict).
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    frames   = vc.frames
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("vc.frames is empty.")

    # ------------------------------------------------------------------
    # Pass 0: reference spacing
    # ------------------------------------------------------------------
    ref_idx, ref_spacing = _find_reference_spacing(frames, ratio_min, ratio_max)
    log(f"  Reference frame: {ref_idx}  |  spacing: {ref_spacing:.4f} px")

    # ------------------------------------------------------------------
    # Pass 1: mark bad frames
    # ------------------------------------------------------------------
    is_bad    = _mark_bad_frames(frames, ref_spacing, ratio_min, ratio_max)
    bad_idx   = np.where(is_bad)[0]
    bad_runs  = _summarize_bad_runs(bad_idx)
    first_nz  = next((k for k, f in enumerate(frames) if len(f.detections) > 0), -1)

    log(f"  First non-empty frame: {first_nz}")
    log(f"  Bad frames: {len(bad_idx)}  |  segments: {len(bad_runs)}")

    sanitized_runs   = 0
    sanitized_frames = 0

    # ------------------------------------------------------------------
    # Pass 2: repair
    # ------------------------------------------------------------------
    if bad_idx.size > 0 and repair:
        log(f"  Repairing {len(bad_idx)} frame(s) across {len(bad_runs)} segment(s)…")

        for idx_start, idx_end in bad_runs:
            idxL = idx_start - 1
            idxR = idx_end   + 1

            if idxL < 0 or idxR >= n_frames:
                log(f"    Skipping [{idx_start},{idx_end}]: touches boundary.")
                continue

            DL = frames[idxL].detections
            DR = frames[idxR].detections
            if not DL or not DR:
                log(f"    Skipping [{idx_start},{idx_end}]: neighbour frame is empty.")
                continue

            XL = np.array([d.x    for d in DL], dtype=float)
            YL = np.array([d.y    for d in DL], dtype=float)
            AL = np.array([d.area for d in DL], dtype=float)
            TL = np.array([d.angle for d in DL], dtype=float)
            CL = np.array([d.color for d in DL])

            XR = np.array([d.x    for d in DR], dtype=float)
            YR = np.array([d.y    for d in DR], dtype=float)
            AR = np.array([d.area for d in DR], dtype=float)
            TR = np.array([d.angle for d in DR], dtype=float)
            CR = np.array([d.color for d in DR])

            # Find the best colour-consistent alignment offset between L and R
            best_offset = None
            max_overlap = 0
            min_err     = float('inf')

            for offset in range(-(len(DR) - 1), len(DL)):
                iL_s = max(0, offset)
                iL_e = min(len(DL), len(DR) + offset)
                if iL_s >= iL_e:
                    continue
                iR_s = iL_s - offset
                iR_e = iL_e - offset
                if not np.array_equal(CL[iL_s:iL_e], CR[iR_s:iR_e]):
                    continue
                err     = np.mean(np.abs(XL[iL_s:iL_e] - XR[iR_s:iR_e]))
                overlap = iL_e - iL_s
                if overlap > max_overlap or (overlap == max_overlap and err < min_err):
                    max_overlap = overlap
                    min_err     = err
                    best_offset = offset

            if best_offset is None:
                log(f"    Skipping [{idx_start},{idx_end}]: no colour-consistent alignment found.")
                continue

            S_shift = -min(0, best_offset)
            N_union = max(len(DL) + S_shift, len(DR) + best_offset + S_shift)

            U_XL = np.full(N_union, np.nan);  U_XR = np.full(N_union, np.nan)
            U_YL = np.full(N_union, np.nan);  U_YR = np.full(N_union, np.nan)
            U_AL = np.full(N_union, np.nan);  U_AR = np.full(N_union, np.nan)
            U_TL = np.full(N_union, np.nan);  U_TR = np.full(N_union, np.nan)
            U_C  = ['?'] * N_union

            if max_overlap > 0:
                ol_s_L = max(0, best_offset)
                ol_e_L = min(len(DL), len(DR) + best_offset)
                ol_s_R = ol_s_L - best_offset
                ol_e_R = ol_e_L - best_offset
                dx_avg = float(np.mean(XR[ol_s_R:ol_e_R] - XL[ol_s_L:ol_e_L]))
                dy_avg = float(np.mean(YR[ol_s_R:ol_e_R] - YL[ol_s_L:ol_e_L]))
            else:
                dx_avg = dy_avg = 0.0

            for u in range(N_union):
                iL  = u - S_shift
                iR  = u - best_offset - S_shift
                inL = 0 <= iL < len(DL)
                inR = 0 <= iR < len(DR)
                if inL and inR:
                    U_XL[u], U_XR[u] = XL[iL], XR[iR]
                    U_YL[u], U_YR[u] = YL[iL], YR[iR]
                    U_AL[u], U_AR[u] = AL[iL], AR[iR]
                    U_TL[u], U_TR[u] = TL[iL], TR[iR]
                    U_C[u]            = CL[iL]
                elif inL:
                    U_XL[u], U_XR[u] = XL[iL], XL[iL] + dx_avg
                    U_YL[u], U_YR[u] = YL[iL], YL[iL] + dy_avg
                    U_AL[u], U_AR[u] = AL[iL], AL[iL]
                    U_TL[u], U_TR[u] = TL[iL], TL[iL]
                    U_C[u]            = CL[iL]
                elif inR:
                    U_XR[u], U_XL[u] = XR[iR], XR[iR] - dx_avg
                    U_YR[u], U_YL[u] = YR[iR], YR[iR] - dy_avg
                    U_AR[u], U_AL[u] = AR[iR], AR[iR]
                    U_TR[u], U_TL[u] = TR[iR], TR[iR]
                    U_C[u]            = CR[iR]

            tL = frames[idxL].frame_time_s
            tR = frames[idxR].frame_time_s
            if tR == tL:
                raise RuntimeError(
                    f"Anchor frames around segment [{idx_start},{idx_end}] have identical timestamps."
                )

            for k in range(idx_start, idx_end + 1):
                alpha = (frames[k].frame_time_s - tL) / (tR - tL)
                sk = int(round(S_shift + alpha * best_offset))
                ek = int(round(len(DL) + S_shift - 1 + alpha * (len(DR) - len(DL))))
                frames[k].detections = [
                    DetectionRecord(
                        x=float(U_XL[u] + alpha * (U_XR[u] - U_XL[u])),
                        y=float(U_YL[u] + alpha * (U_YR[u] - U_YL[u])),
                        color=U_C[u],
                        area=float(U_AL[u] + alpha * (U_AR[u] - U_AL[u])),
                        angle=float(_interp_angle(U_TL[u], U_TR[u], alpha)),
                    )
                    for u in range(sk, ek + 1)
                ]

            sanitized_runs   += 1
            sanitized_frames += idx_end - idx_start + 1

    # ------------------------------------------------------------------
    # Pass 3: final validation
    # ------------------------------------------------------------------
    log("  Final validation…")
    all_diffs  = []
    first_seen = False

    for k, f in enumerate(frames):
        if len(f.detections) > 0:
            first_seen = True
        elif first_seen:
            raise RuntimeError(f"Gap persists at frame {k} after repair.")
        if len(f.detections) >= 2:
            all_diffs.extend(np.diff([d.x for d in f.detections]))

    still_bad = []
    for k, f in enumerate(frames):
        dets = f.detections
        if len(dets) == 0:
            if first_nz != -1 and k >= first_nz:
                still_bad.append(k)
            continue
        x = np.array([d.x for d in dets], dtype=float)
        c = np.array([d.color for d in dets])
        if np.any(~np.isfinite(x)) or np.any(np.diff(x) < 0) or np.any(~np.isin(c, ['r', 'g'])):
            still_bad.append(k)
            continue
        if len(dets) >= 2:
            sr = np.diff(x) / ref_spacing
            if np.any(c[:-1] == c[1:]) or np.any((sr < ratio_min) | (sr > ratio_max)):
                still_bad.append(k)

    if still_bad:
        preview = ", ".join(map(str, still_bad[:10])) + ("…" if len(still_bad) > 10 else "")
        raise RuntimeError(f"Verification failed — bad frames remain: {preview}")

    vc.passedVerification = True
    vc.meanBlockDistance  = float(np.mean(all_diffs)) if all_diffs else 0.0

    summary = {
        'n_frames':                  n_frames,
        'reference_frame_idx':       ref_idx,
        'reference_spacing_px':      ref_spacing,
        'first_non_empty_idx':       first_nz,
        'initial_bad_frames':        int(len(bad_idx)),
        'bad_segments':              int(len(bad_runs)),
        'sanitized_runs':            sanitized_runs,
        'sanitized_frames':          sanitized_frames,
        'final_mean_block_distance': vc.meanBlockDistance,
    }
    return vc, summary
