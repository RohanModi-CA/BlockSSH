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

BOND_SPACING_MODES = ("default", "comoving")
BondSpacingMode = Literal["default", "comoving"]


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

    if resolved not in {"x", "y"}:
        raise ValueError(
            f"Comoving bond spacing mode requires component 'x' or 'y'; got '{resolved}'"
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
        prev_vector: tuple[float, float] | None = None
        prev_xhat: tuple[float, float] | None = None
        prev_yhat: tuple[float, float] | None = None
        prev_long = np.nan
        prev_trans = np.nan

        for frame_idx in range(n_frames):
            vx = float(dx[frame_idx, pair_idx])
            vy = float(dy[frame_idx, pair_idx])
            if (not np.isfinite(vx)) or (not np.isfinite(vy)):
                prev_vector = None
                prev_xhat = None
                prev_yhat = None
                prev_long = np.nan
                prev_trans = np.nan
                continue

            norm = float(np.hypot(vx, vy))
            if (not np.isfinite(norm)) or norm <= 0.0:
                prev_vector = None
                prev_xhat = None
                prev_yhat = None
                prev_long = np.nan
                prev_trans = np.nan
                continue

            curr_xhat = (vx / norm, vy / norm)
            curr_yhat = (-curr_xhat[1], curr_xhat[0])

            if prev_vector is None or prev_xhat is None or prev_yhat is None:
                long_value = norm
                trans_value = 0.0
            else:
                delta_x = vx - prev_vector[0]
                delta_y = vy - prev_vector[1]
                long_inc = delta_x * prev_xhat[0] + delta_y * prev_xhat[1]
                trans_inc = delta_x * prev_yhat[0] + delta_y * prev_yhat[1]
                long_value = float(prev_long + long_inc)
                trans_value = float(prev_trans + trans_inc)

            longitudinal[frame_idx, pair_idx] = long_value
            transverse[frame_idx, pair_idx] = trans_value

            prev_vector = (vx, vy)
            prev_xhat = curr_xhat
            prev_yhat = curr_yhat
            prev_long = long_value
            prev_trans = trans_value

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
        raise ValueError("Comoving bond spacing mode requires a dataset name or track2 path")

    base_dataset, _ = split_dataset_component(resolved_dataset_name)
    track2_x = load_track2_dataset(
        dataset=join_dataset_component(base_dataset, "x"),
        track_data_root=track_data_root,
    )
    track2_y = load_track2_dataset(
        dataset=join_dataset_component(base_dataset, "y"),
        track_data_root=track_data_root,
    )
    _validate_matching_track2(track2_x, track2_y)

    longitudinal, transverse = _derive_comoving_signal_matrices(track2_x, track2_y)
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
