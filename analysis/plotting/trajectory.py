from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.models import Track2Dataset


def _valid_time_mask(time: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    if time.ndim != 1:
        raise ValueError("time must be 1D")
    if time.size == 0:
        return np.zeros(0, dtype=bool)

    valid = np.isfinite(time)
    if not np.any(valid):
        return valid

    prev = np.maximum.accumulate(np.where(valid, time, -np.inf))
    monotonic = np.ones_like(valid, dtype=bool)
    monotonic[1:] = valid[1:] & (time[1:] > prev[:-1])
    return valid & monotonic


def sample_video_frames(video_path: str, sample_times: np.ndarray):
    if not video_path:
        return [], [], "No originalVideoPath found in Track2."

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        return [], [], f"Original video not found: {video_path_obj}"

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        return [], [], f"Could not open original video: {video_path_obj}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        return [], [], f"Could not determine FPS for video: {video_path_obj}"

    if frame_count <= 0:
        cap.release()
        return [], [], f"Video appears to have zero frames: {video_path_obj}"

    images = []
    valid_times = []

    for t in sample_times:
        frame_idx = int(round(float(t) * fps))
        frame_idx = max(0, min(frame_idx, frame_count - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        images.append(frame_rgb)
        valid_times.append(float(t))

    cap.release()

    if len(images) == 0:
        return [], [], f"Could not decode any sampled frames from video: {video_path_obj}"

    return images, valid_times, None


def plot_track2_positions_overview(
    track2: Track2Dataset,
    *,
    framestrip: bool = False,
    nframestrip: int = 8,
    title: str | None = None,
):
    if nframestrip <= 0:
        raise ValueError("nframestrip must be positive")

    time = track2.frame_times_s
    x_positions = track2.x_positions
    block_colors = [str(c).lower() for c in track2.block_colors]
    time_mask = _valid_time_mask(time)

    if not np.any(time_mask):
        raise ValueError("Track2 frame_times_s contains no finite increasing timestamps")

    time = time[time_mask]
    x_positions = x_positions[time_mask]

    n_frames, n_blocks = x_positions.shape
    if n_blocks == 0:
        raise ValueError("No persistent blocks found to plot")

    visible_mask = np.isfinite(x_positions)

    if framestrip:
        fig = plt.figure(figsize=(14, 3 * n_blocks / 2 + 6))
        gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.2, 1.8], hspace=0.25)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_vis = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_strip = fig.add_subplot(gs[2, 0], sharex=ax_main)
    else:
        fig = plt.figure(figsize=(14, 3 * n_blocks / 2 + 3))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.20)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_vis = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_strip = None

    color_map = {"r": "#e74c3c", "g": "#2ecc71"}

    for j in range(n_blocks):
        c = block_colors[j] if j < len(block_colors) else "?"
        line_color = color_map.get(c, "#3498db")
        y = x_positions[:, j]

        ax_main.plot(time, y, color=line_color, linewidth=1.5, alpha=0.95)

        valid_idx = np.where(np.isfinite(y))[0]
        if valid_idx.size > 0:
            last_idx = valid_idx[-1]
            ax_main.text(
                time[last_idx],
                y[last_idx],
                f" {j}{c.upper()}",
                color=line_color,
                fontsize=9,
                va="center",
                ha="left",
                clip_on=True,
            )

    ax_main.set_title("Persistent Block X-Position Trajectories")
    ax_main.set_ylabel("X Position (px)")
    ax_main.grid(True, linestyle="--", alpha=0.3)

    vis_img = visible_mask.astype(float)
    ax_vis.imshow(
        vis_img,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[time[0], time[-1], n_blocks - 0.5, -0.5],
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    ax_vis.set_title("Visibility Mask (white = visible, black = hidden)")
    ax_vis.set_ylabel("Block Index")
    ax_vis.set_xlabel("Time (seconds)")

    if ax_strip is not None:
        sample_times = np.linspace(time[0], time[-1], nframestrip)
        images, valid_times, status_msg = sample_video_frames(track2.original_video_path, sample_times)

        if len(images) == 0:
            ax_strip.text(
                0.5,
                0.5,
                "Frame strip unavailable\n" + (status_msg if status_msg else ""),
                ha="center",
                va="center",
                transform=ax_strip.transAxes,
                fontsize=11,
            )
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            ax_strip.set_title("Sampled Video Frames")
            print(status_msg if status_msg else "Frame strip unavailable.")
        else:
            ax_strip.set_title("Sampled Video Frames")
            ax_strip.set_yticks([])
            ax_strip.set_xlabel("Time (seconds)")

            y0, y1 = 0.0, 1.0
            for img, t in zip(images, valid_times):
                if len(valid_times) == 1:
                    dt_half = 0.5 * (time[-1] - time[0]) if time[-1] > time[0] else 0.5
                else:
                    sample_step = np.median(np.diff(valid_times))
                    dt_half = 0.45 * sample_step

                x0 = t - dt_half
                x1 = t + dt_half

                ax_strip.imshow(
                    img,
                    extent=[x0, x1, y0, y1],
                    aspect="auto",
                    interpolation="nearest",
                )
                ax_strip.text(
                    t,
                    y1 + 0.02,
                    f"{t:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

                ax_main.axvline(t, color="k", alpha=0.12, linewidth=1)
                ax_vis.axvline(t, color="k", alpha=0.12, linewidth=1)

            ax_strip.set_xlim(time[0], time[-1])
            ax_strip.set_ylim(0, 1.15)

    fig.suptitle(
        title or f"Track2 Position Overview | Blocks: {n_blocks} | Frames: {n_frames}",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def plot_spacing_timeseries(
    time: np.ndarray,
    spacing: np.ndarray,
    pair_labels: list[str],
    *,
    title: str | None = None,
):
    if spacing.ndim != 2:
        raise ValueError("spacing must be 2D")

    n_frames, n_pairs = spacing.shape
    if n_pairs == 0:
        raise ValueError("No block pairs found to plot")
    if time.shape[0] != n_frames:
        raise ValueError("time length must match spacing rows")

    time_mask = _valid_time_mask(time)
    if not np.any(time_mask):
        raise ValueError("time contains no finite increasing timestamps")

    time = time[time_mask]
    spacing = spacing[time_mask]
    n_frames, n_pairs = spacing.shape

    fig, axes = plt.subplots(
        n_pairs,
        1,
        figsize=(12, 2 * n_pairs),
        sharex=True,
    )
    if n_pairs == 1:
        axes = [axes]

    color_map = {"rg": "#e74c3c", "gr": "#2ecc71"}
    global_avg = np.nanmean(spacing)

    for i in range(n_pairs):
        ax = axes[i]
        label = str(pair_labels[i]).lower() if i < len(pair_labels) else "?"
        color = color_map.get(label, "#3498db")

        ax.plot(time, spacing[:, i], color=color, linewidth=1)

        local_avg = np.nanmean(spacing[:, i])
        ax.set_ylabel("Dist (px)")
        ax.set_title(f"Pair {i + 1}: {label.upper()}  |  Avg: {local_avg:.1f}px")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelbottom=True)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(
        title or f"Block Spacing Time-Series (Global Avg: {global_avg:.1f}px)",
        fontsize=14,
    )
    plt.xlim(float(time[0]), float(time[-1]))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def plot_block_timeseries(
    time: np.ndarray,
    x_positions: np.ndarray,
    block_labels: list[str],
    *,
    title: str | None = None,
):
    if x_positions.ndim != 2:
        raise ValueError("x_positions must be 2D")

    n_frames, n_blocks = x_positions.shape
    if n_blocks == 0:
        raise ValueError("No blocks found to plot")
    if time.shape[0] != n_frames:
        raise ValueError("time length must match x_positions rows")

    time_mask = _valid_time_mask(time)
    if not np.any(time_mask):
        raise ValueError("time contains no finite increasing timestamps")

    time = time[time_mask]
    x_positions = x_positions[time_mask]
    n_frames, n_blocks = x_positions.shape

    fig, axes = plt.subplots(
        n_blocks,
        1,
        figsize=(12, 2 * n_blocks),
        sharex=True,
    )
    if n_blocks == 1:
        axes = [axes]

    color_map = {"r": "#e74c3c", "g": "#2ecc71"}
    global_avg = np.nanmean(x_positions)

    for i in range(n_blocks):
        ax = axes[i]
        label = str(block_labels[i]).lower() if i < len(block_labels) else "?"
        color = color_map.get(label, "#3498db")

        ax.plot(time, x_positions[:, i], color=color, linewidth=1)

        local_avg = np.nanmean(x_positions[:, i])
        ax.set_ylabel("X (px)")
        ax.set_title(f"Block {i}: {label.upper()}  |  Avg: {local_avg:.1f}px")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelbottom=True)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(
        title or f"Block X-Position Time-Series (Global Avg: {global_avg:.1f}px)",
        fontsize=14,
    )
    plt.xlim(float(time[0]), float(time[-1]))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig
