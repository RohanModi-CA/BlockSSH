from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .derived import derive_pair_labels, derive_spacing_dataset
from .io import (
    dataset_name_from_track2_path,
    join_dataset_component,
    load_track2_dataset,
    split_dataset_component,
)
from .models import Track2Dataset

BOND_SPACING_MODES = ("default", "comoving", "purecomoving")
BondSpacingMode = Literal["default", "comoving", "purecomoving"]


@dataclass(frozen=True)
class BondSignalDataset:
    dataset_name: str | None
    component: str
    mode: BondSpacingMode
    pair_labels: list[str]
    signal_matrix: np.ndarray
    frame_times_s: np.ndarray
    source_path: str


def _normalize_bond_spacing_mode(mode: str) -> BondSpacingMode:
    normalized = str(mode).strip().lower()
    if normalized not in BOND_SPACING_MODES:
        raise ValueError(
            f"Unsupported bond spacing mode '{mode}'. Expected one of {list(BOND_SPACING_MODES)}"
        )
    return normalized  # type: ignore[return-value]


def _resolve_requested_component(
    *,
    dataset: str | None,
    component: str | None,
    mode: BondSpacingMode,
) -> str:
    dataset_component = None
    if dataset is not None:
        _, dataset_component = split_dataset_component(dataset)

    resolved = str(component).strip().lower() if component is not None else dataset_component or "x"

    if mode == "default":
        if resolved not in {"x", "y", "a"}:
            raise ValueError(f"Default bond spacing mode does not support component '{resolved}'")
        return resolved

    if resolved not in {"x", "y", "a"}:
        raise ValueError(
            f"Comoving bond spacing mode does not support component '{resolved}'"
        )
    return resolved


def _validate_matching_track2(track2_x: Track2Dataset, track2_y: Track2Dataset) -> None:
    if track2_x.x_positions.shape != track2_y.x_positions.shape:
        raise ValueError(
            "Comoving bond spacing mode requires x and y datasets with matching xPositions shapes"
        )
    if track2_x.frame_times_s.shape != track2_y.frame_times_s.shape or not np.allclose(
        track2_x.frame_times_s,
        track2_y.frame_times_s,
        equal_nan=True,
    ):
        raise ValueError(
            "Comoving bond spacing mode requires x and y datasets with matching frame times"
        )
    if track2_x.frame_numbers.shape != track2_y.frame_numbers.shape or not np.array_equal(
        track2_x.frame_numbers,
        track2_y.frame_numbers,
    ):
        raise ValueError(
            "Comoving bond spacing mode requires x and y datasets with matching frame numbers"
        )
    if list(track2_x.block_colors) != list(track2_y.block_colors):
        raise ValueError(
            "Comoving bond spacing mode requires x and y datasets with matching block colors"
        )


def _derive_comoving_signal_matrices(
    track2_x: Track2Dataset,
    track2_y: Track2Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    dx = np.asarray(track2_x.x_positions[:, 1:] - track2_x.x_positions[:, :-1], dtype=float)
    dy = np.asarray(track2_y.x_positions[:, 1:] - track2_y.x_positions[:, :-1], dtype=float)
    n_frames, n_pairs = dx.shape

    longitudinal = np.full((n_frames, n_pairs), np.nan, dtype=float)
    transverse = np.full((n_frames, n_pairs), np.nan, dtype=float)

    for pair_idx in range(n_pairs):
        vx = np.asarray(dx[:, pair_idx], dtype=float)
        vy = np.asarray(dy[:, pair_idx], dtype=float)
        valid = np.isfinite(vx) & np.isfinite(vy)
        if not np.any(valid):
            continue

        mean_vx = float(np.nanmean(vx[valid]))
        mean_vy = float(np.nanmean(vy[valid]))
        if (not np.isfinite(mean_vx)) or (not np.isfinite(mean_vy)):
            continue
        reference = np.array([mean_vx, mean_vy], dtype=float)

        vectors = np.column_stack([vx, vy])
        norms = np.hypot(vx, vy)
        instant_valid = valid & np.isfinite(norms) & (norms > 0.0)
        if not np.any(instant_valid):
            continue

        xhat = np.full((n_frames, 2), np.nan, dtype=float)
        xhat[instant_valid, 0] = vx[instant_valid] / norms[instant_valid]
        xhat[instant_valid, 1] = vy[instant_valid] / norms[instant_valid]
        yhat = np.column_stack((-xhat[:, 1], xhat[:, 0]))

        delta = vectors - reference
        longitudinal[instant_valid, pair_idx] = np.sum(delta[instant_valid] * xhat[instant_valid], axis=1)
        transverse[instant_valid, pair_idx] = np.sum(delta[instant_valid] * yhat[instant_valid], axis=1)

    return longitudinal, transverse


def _derive_purecomoving_signal_matrices(
    track2_x: Track2Dataset,
    track2_y: Track2Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    x_pos = np.asarray(track2_x.x_positions, dtype=float)
    y_pos = np.asarray(track2_y.x_positions, dtype=float)
    n_frames, n_blocks = x_pos.shape
    n_pairs = max(0, n_blocks - 1)

    longitudinal = np.full((n_frames, n_pairs), np.nan, dtype=float)
    transverse = np.full((n_frames, n_pairs), np.nan, dtype=float)

    for pair_idx in range(n_pairs):
        left_block = pair_idx
        right_block = pair_idx + 1

        for frame_idx in range(1, n_frames):
            x_left = x_pos[frame_idx, left_block]
            y_left = y_pos[frame_idx, left_block]
            x_right = x_pos[frame_idx, right_block]
            y_right = y_pos[frame_idx, right_block]

            x_left_prev = x_pos[frame_idx - 1, left_block]
            y_left_prev = y_pos[frame_idx - 1, left_block]
            x_right_prev = x_pos[frame_idx - 1, right_block]
            y_right_prev = y_pos[frame_idx - 1, right_block]

            if not (np.isfinite(x_left) and np.isfinite(y_left) and
                    np.isfinite(x_right) and np.isfinite(y_right) and
                    np.isfinite(x_left_prev) and np.isfinite(y_left_prev) and
                    np.isfinite(x_right_prev) and np.isfinite(y_right_prev)):
                continue

            bond_x = x_right - x_left
            bond_y = y_right - y_left
            bond_norm = np.hypot(bond_x, bond_y)
            if not (np.isfinite(bond_norm) and bond_norm > 0):
                continue

            xhat_x = bond_x / bond_norm
            xhat_y = bond_y / bond_norm

            if xhat_x < 0:
                xhat_x = -xhat_x
                xhat_y = -xhat_y

            yhat_x = -xhat_y
            yhat_y = xhat_x

            dx_left = x_left - x_left_prev
            dy_left = y_left - y_left_prev
            dx_right = x_right - x_right_prev
            dy_right = y_right - y_right_prev

            dl = dx_left * xhat_x + dy_left * xhat_y + dx_right * xhat_x + dy_right * xhat_y
            dt = dx_left * yhat_x + dy_left * yhat_y + dx_right * yhat_x + dy_right * yhat_y

            longitudinal[frame_idx, pair_idx] = dl
            transverse[frame_idx, pair_idx] = dt

    return longitudinal, transverse


def load_bond_signal_dataset(
    *,
    dataset: str | None = None,
    track2_path: str | Path | None = None,
    track_data_root: str | Path | None = None,
    bond_spacing_mode: str = "default",
    component: str | None = None,
) -> BondSignalDataset:
    mode = _normalize_bond_spacing_mode(bond_spacing_mode)
    resolved_dataset_name = (
        str(dataset).strip()
        if dataset is not None
        else dataset_name_from_track2_path(track2_path) if track2_path is not None else None
    )
    requested_component = _resolve_requested_component(
        dataset=resolved_dataset_name,
        component=component,
        mode=mode,
    )

    if mode == "default":
        track2 = load_track2_dataset(
            dataset=dataset,
            track2_path=track2_path,
            track_data_root=track_data_root,
        )
        spacing = derive_spacing_dataset(track2)
        return BondSignalDataset(
            dataset_name=resolved_dataset_name,
            component=requested_component,
            mode=mode,
            pair_labels=list(spacing.pair_labels),
            signal_matrix=np.asarray(spacing.spacing_matrix, dtype=float),
            frame_times_s=np.asarray(track2.frame_times_s, dtype=float),
            source_path=str(track2.track2_path),
        )

    if resolved_dataset_name is None:
        raise ValueError("Comoving/purecomoving bond spacing mode requires a dataset name or track2 path")

    base_dataset, _ = split_dataset_component(resolved_dataset_name)
    if requested_component == "a":
        track2_a = load_track2_dataset(
            dataset=join_dataset_component(base_dataset, "a"),
            track_data_root=track_data_root,
        )
        spacing = derive_spacing_dataset(track2_a)
        return BondSignalDataset(
            dataset_name=resolved_dataset_name,
            component=requested_component,
            mode=mode,
            pair_labels=list(spacing.pair_labels),
            signal_matrix=np.asarray(spacing.spacing_matrix, dtype=float),
            frame_times_s=np.asarray(track2_a.frame_times_s, dtype=float),
            source_path=str(track2_a.track2_path),
        )

    track2_x = load_track2_dataset(
        dataset=join_dataset_component(base_dataset, "x"),
        track_data_root=track_data_root,
    )
    track2_y = load_track2_dataset(
        dataset=join_dataset_component(base_dataset, "y"),
        track_data_root=track_data_root,
    )
    _validate_matching_track2(track2_x, track2_y)

    if mode == "comoving":
        longitudinal, transverse = _derive_comoving_signal_matrices(track2_x, track2_y)
    else:
        longitudinal, transverse = _derive_purecomoving_signal_matrices(track2_x, track2_y)
    signal_matrix = longitudinal if requested_component == "x" else transverse

    return BondSignalDataset(
        dataset_name=resolved_dataset_name,
        component=requested_component,
        mode=mode,
        pair_labels=derive_pair_labels(track2_x.block_colors),
        signal_matrix=signal_matrix,
        frame_times_s=np.asarray(track2_x.frame_times_s, dtype=float),
        source_path=f"{track2_x.track2_path} | {track2_y.track2_path}",
    )
