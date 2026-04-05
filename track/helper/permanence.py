"""
helper/permanence.py
Build stable Track2 permanence matrices from a verified VideoCentroids object.

The front/non-black pipeline tracks full centroids in track1, but legacy step 2
only exported x positions. This module now builds one stable identity solution
and exports both x and y permanence matrices, while keeping the legacy
Track2XPermanence container for downstream compatibility.

Public API
----------
build_permanence(vc, quiet=False) → Track2XPermanence
build_permanence_xy(vc, quiet=False) → (Track2XPermanence, Track2XPermanence)
"""

import numpy as np
from tracking_classes import VideoCentroids, Track2XPermanence


_Y_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _opposite(c: str) -> str:
    return 'g' if c == 'r' else 'r'


def _alignment_penalty(
    this_x,
    this_y,
    prev_x,
    prev_y,
    ref_spacing: float,
    y_norm: float,
) -> float:
    dx = np.mean(np.abs(this_x - prev_x)) / ref_spacing
    dy = np.mean(np.abs(this_y - prev_y)) / y_norm
    return float(dx + _Y_WEIGHT * dy)


def _estimate_reference_spacing(frames) -> float:
    pooled = []
    for f in frames:
        x = np.array([d.x for d in f.detections], dtype=float)
        if len(x) < 2 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
            continue
        pooled.extend(np.diff(x).tolist())
    if not pooled:
        return 1.0
    return max(1.0, float(np.median(np.array(pooled, dtype=float))))


def _decide_increase_side(
    this_x,  this_colors,
    this_y,
    prev_x,  block_colors,
    prev_y,
    prev_L:  int,
    prev_R:  int,
    ref_spacing: float,
    y_norm: float,
):
    """
    When count increases by 1, decide whether the new block entered from the LEFT
    or the RIGHT side of the visible window.

    Returns (choose_left, left_ok, right_ok, pen_left, pen_right).
    """
    pen_left = _alignment_penalty(
        this_x[1:], this_y[1:], prev_x, prev_y, ref_spacing, y_norm
    )
    pen_right = _alignment_penalty(
        this_x[:-1], this_y[:-1], prev_x, prev_y, ref_spacing, y_norm
    )
    n = len(block_colors)

    # Expected colour sequences for each entry side
    if prev_L > 0:
        exp_left = block_colors[prev_L - 1 : prev_R + 1]
    else:
        exp_left = [_opposite(block_colors[0])] + block_colors[prev_L : prev_R + 1]

    if prev_R < n - 1:
        exp_right = block_colors[prev_L : prev_R + 2]
    else:
        exp_right = block_colors[prev_L : prev_R + 1] + [_opposite(block_colors[-1])]

    left_ok  = list(this_colors) == list(exp_left)
    right_ok = list(this_colors) == list(exp_right)

    if left_ok  and not right_ok: return True,  True,  False, pen_left, pen_right
    if right_ok and not left_ok:  return False, False, True,  pen_left, pen_right
    if left_ok  and right_ok:     return (pen_left <= pen_right), True, True, pen_left, pen_right
    return True, False, False, pen_left, pen_right  # ambiguous; default left


def _decide_decrease_side(
    this_x,  this_colors,
    this_y,
    prev_x,  block_colors,
    prev_y,
    prev_L:  int,
    prev_R:  int,
    ref_spacing: float,
    y_norm: float,
):
    """
    When count decreases by 1, decide whether the block exited from LEFT or RIGHT.

    Returns (choose_left_exit, left_ok, right_ok, pen_left, pen_right).
    """
    pen_left = _alignment_penalty(
        this_x, this_y, prev_x[1:], prev_y[1:], ref_spacing, y_norm
    )
    pen_right = _alignment_penalty(
        this_x, this_y, prev_x[:-1], prev_y[:-1], ref_spacing, y_norm
    )

    exp_left  = block_colors[prev_L + 1 : prev_R + 1]
    exp_right = block_colors[prev_L     : prev_R    ]

    left_ok  = list(this_colors) == list(exp_left)
    right_ok = list(this_colors) == list(exp_right)

    if left_ok  and not right_ok: return True,  True,  False, pen_left, pen_right
    if right_ok and not left_ok:  return False, False, True,  pen_left, pen_right
    if left_ok  and right_ok:     return (pen_left <= pen_right), True, True, pen_left, pen_right
    return True, False, False, pen_left, pen_right


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_permanence_xy(
    vc: VideoCentroids,
    quiet: bool = False,
) -> tuple[Track2XPermanence, Track2XPermanence]:
    """
    Build the permanence matrix from a *verified* VideoCentroids object.

    The returned pair contains x and y permanence matrices. Each matrix is
    [nFrames × nBlocks], and each column index is stable across the whole video.

    Raises RuntimeError on any tracking inconsistency (count jump > 1,
    colour mismatch, etc.).

    The returned object has trackingResultsPath = "" — the caller should set it
    to the path of the track1 msgpack after this function returns.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    if not vc.passedVerification:
        raise RuntimeError("vc.passedVerification is False — run verification first.")

    frames   = vc.frames
    n_frames = len(frames)

    # Find the first non-empty frame to initialise state
    first_nz = next((k for k, f in enumerate(frames) if len(f.detections) >= 1), -1)
    if first_nz == -1:
        raise RuntimeError("No frame contains detections — cannot initialise permanence.")

    init_frame  = frames[first_nz]
    block_colors = [d.color for d in init_frame.detections]
    prev_x       = np.array([d.x for d in init_frame.detections], dtype=float)
    prev_y       = np.array([d.y for d in init_frame.detections], dtype=float)
    prev_L       = 0
    prev_R       = len(prev_x) - 1
    prev_num     = len(prev_x)
    ref_spacing  = max(1.0, float(vc.meanBlockDistance or _estimate_reference_spacing(frames)))
    y_norm       = max(12.0, 0.35 * ref_spacing)

    matrix_x: list = [[float(v) for v in prev_x]]
    matrix_y: list = [[float(v) for v in prev_y]]

    max_vis        = prev_num
    count_same     = 0
    count_plus     = 0
    count_minus    = 0
    expand_left    = 0
    expand_right   = 0

    log(f"  First non-empty frame: {first_nz}  ({prev_num} initial blocks)")

    for k in range(first_nz + 1, n_frames):
        f           = frames[k]
        this_x      = np.array([d.x for d in f.detections], dtype=float)
        this_y      = np.array([d.y for d in f.detections], dtype=float)
        this_colors = [d.color for d in f.detections]
        this_num    = len(this_x)

        if this_num == 0:
            raise RuntimeError(f"Zero-detection frame at k={k} after tracking has started.")

        delta    = this_num - prev_num
        max_vis  = max(max_vis, this_num)

        if abs(delta) > 1:
            raise RuntimeError(f"Count jump > 1 at k={k}: {prev_num} → {this_num}.")

        curr_L = curr_R = -1

        # ---- delta == 0 ----
        if delta == 0:
            count_same += 1
            curr_L, curr_R = prev_L, prev_R

        # ---- delta == +1 : new block entered ----
        elif delta == 1:
            count_plus += 1
            choose_left, left_ok, right_ok, pen_l, pen_r = _decide_increase_side(
                this_x,
                this_colors,
                this_y,
                prev_x,
                block_colors,
                prev_y,
                prev_L,
                prev_R,
                ref_spacing,
                y_norm,
            )
            if not left_ok and not right_ok:
                raise RuntimeError(
                    f"Frame k={k}: count +1 but neither left nor right colour sequence matches.\n"
                    f"  Visible: {' '.join(this_colors)}\n"
                    f"  Global:  {' '.join(block_colors)}"
                )

            if choose_left:
                if prev_L > 0:
                    curr_L, curr_R = prev_L - 1, prev_R
                else:
                    # Extend the matrix leftward
                    block_colors.insert(0, this_colors[0])
                    for row in matrix_x:
                        row.insert(0, float('nan'))
                    for row in matrix_y:
                        row.insert(0, float('nan'))
                    prev_L += 1
                    prev_R += 1
                    curr_L  = prev_L - 1
                    curr_R  = prev_R
                    expand_left += 1
            else:
                if prev_R < len(block_colors) - 1:
                    curr_L, curr_R = prev_L, prev_R + 1
                else:
                    # Extend the matrix rightward
                    block_colors.append(this_colors[-1])
                    for row in matrix_x:
                        row.append(float('nan'))
                    for row in matrix_y:
                        row.append(float('nan'))
                    curr_L  = prev_L
                    curr_R  = prev_R + 1
                    expand_right += 1

        # ---- delta == -1 : block exited ----
        elif delta == -1:
            count_minus += 1
            choose_left, left_ok, right_ok, pen_l, pen_r = _decide_decrease_side(
                this_x,
                this_colors,
                this_y,
                prev_x,
                block_colors,
                prev_y,
                prev_L,
                prev_R,
                ref_spacing,
                y_norm,
            )
            if not left_ok and not right_ok:
                raise RuntimeError(
                    f"Frame k={k}: count -1 but neither exit side colour sequence matches."
                )
            if choose_left:
                curr_L, curr_R = prev_L + 1, prev_R
            else:
                curr_L, curr_R = prev_L, prev_R - 1

        # Sanity check: interval width must equal detection count
        if (curr_R - curr_L + 1) != this_num:
            raise RuntimeError(
                f"Interval width mismatch at frame k={k}: "
                f"calc={curr_R - curr_L + 1}, actual={this_num}."
            )

        row_x = [float('nan')] * len(block_colors)
        row_y = [float('nan')] * len(block_colors)
        for i, val in enumerate(this_x):
            row_x[curr_L + i] = float(val)
            row_y[curr_L + i] = float(this_y[i])
        matrix_x.append(row_x)
        matrix_y.append(row_y)

        prev_x, prev_y, prev_num, prev_L, prev_R = this_x, this_y, this_num, curr_L, curr_R

    # Prepend NaN rows for any leading empty frames
    n_cols = len(block_colors)
    if first_nz > 0:
        pad_rows = [[float('nan')] * n_cols for _ in range(first_nz)]
        matrix_x = pad_rows + matrix_x
        matrix_y = [[float('nan')] * n_cols for _ in range(first_nz)] + matrix_y

    if len(matrix_x) != n_frames or len(matrix_y) != n_frames:
        raise RuntimeError(
            f"Row count mismatch: x={len(matrix_x)}, y={len(matrix_y)}, expected {n_frames}."
        )

    log(f"  Total unique blocks: {n_cols}  |  max visible: {max_vis}")
    log(f"  Deltas — same: {count_same}, +1: {count_plus}, -1: {count_minus}")
    log(f"  Matrix expansions — left: {expand_left}, right: {expand_right}")
    log(f"  Block sequence: {' '.join(block_colors)}")

    t2_x = Track2XPermanence(
        originalVideoPath=vc.filepath,
        trackingResultsPath="",           # caller sets this after return
        blockColors=block_colors,
        xPositions=matrix_x,
        frameTimes_s=[f.frame_time_s  for f in frames],
        frameNumbers=[f.frame_number  for f in frames],
    )
    t2_y = Track2XPermanence(
        originalVideoPath=vc.filepath,
        trackingResultsPath="",
        blockColors=block_colors,
        xPositions=matrix_y,
        frameTimes_s=[f.frame_time_s for f in frames],
        frameNumbers=[f.frame_number for f in frames],
    )
    return t2_x, t2_y


def build_permanence(
    vc: VideoCentroids,
    quiet: bool = False,
) -> Track2XPermanence:
    t2_x, _ = build_permanence_xy(vc, quiet=quiet)
    return t2_x
