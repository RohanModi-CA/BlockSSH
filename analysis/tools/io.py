from __future__ import annotations

from pathlib import Path

import msgpack
import numpy as np

from .models import Track2Dataset


DEFAULT_TRACK_DATA_ROOT = (Path(__file__).resolve().parents[2] / "track" / "data").resolve()
COMPONENT_SUFFIXES = ("x", "y", "a")


def get_default_track_data_root() -> Path:
    return DEFAULT_TRACK_DATA_ROOT


def split_dataset_component(dataset: str) -> tuple[str, str | None]:
    text = str(dataset).strip()
    for suffix in COMPONENT_SUFFIXES:
        token = f"_{suffix}"
        if text.endswith(token):
            return text[: -len(token)], suffix
    return text, None


def join_dataset_component(dataset: str, component: str) -> str:
    base, _ = split_dataset_component(dataset)
    return f"{base}_{str(component).strip().lower()}"


def _base_candidates(dataset: str) -> list[str]:
    base = str(dataset).strip()
    if base.startswith("IMG_"):
        return [base]
    return [base, f"IMG_{base}"]


def _candidate_dataset_dirs(root: Path, dataset: str) -> list[Path]:
    base_name, component = split_dataset_component(dataset)
    candidates: list[Path] = []
    for base in _base_candidates(base_name):
        if component is None:
            # New track layout defaults bare DATASET to x.
            candidates.append(root / base / "components" / "x")
            candidates.append(root / base)
        else:
            candidates.append(root / base / "components" / component)
            candidates.append(root / f"{base}_{component}")
    return candidates


def dataset_dir_from_name(dataset: str, track_data_root: str | Path | None = None) -> Path:
    root = Path(track_data_root) if track_data_root is not None else DEFAULT_TRACK_DATA_ROOT
    for dataset_path in _candidate_dataset_dirs(root, dataset):
        if dataset_path.exists():
            return dataset_path

    tried = ", ".join(str(path) for path in _candidate_dataset_dirs(root, dataset))
    raise FileNotFoundError(f"Could not resolve dataset '{dataset}'. Tried: {tried}")


def default_track2_path(dataset: str, track_data_root: str | Path | None = None) -> Path:
    root = Path(track_data_root) if track_data_root is not None else DEFAULT_TRACK_DATA_ROOT
    candidates = _candidate_dataset_dirs(root, dataset)
    for dataset_path in candidates:
        track2_path = dataset_path / "track2_permanence.msgpack"
        if track2_path.is_file():
            return track2_path

    if len(candidates) > 0:
        return candidates[0] / "track2_permanence.msgpack"
    return dataset_dir_from_name(dataset, track_data_root=track_data_root) / "track2_permanence.msgpack"


def resolve_track2_path(
    dataset: str | None = None,
    track2_path: str | Path | None = None,
    track_data_root: str | Path | None = None,
) -> Path:
    if dataset is None and track2_path is None:
        raise ValueError("Provide either DATASET or --track2")

    if track2_path is not None:
        return Path(track2_path)

    assert dataset is not None
    return default_track2_path(dataset, track_data_root=track_data_root)


def dataset_name_from_track2_path(track2_path: str | Path) -> str:
    path = Path(track2_path)
    if path.name != "track2_permanence.msgpack":
        raise ValueError(f"Expected a track2_permanence.msgpack path, got: {path}")

    parent = path.parent
    if parent.name in COMPONENT_SUFFIXES and parent.parent.name == "components":
        return f"{parent.parent.parent.name}_{parent.name}"
    if any(parent.name.endswith(f"_{suffix}") for suffix in COMPONENT_SUFFIXES):
        return parent.name
    return parent.name


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_msgpack(path: str | Path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_track2_dataset(
    dataset: str | None = None,
    track2_path: str | Path | None = None,
    track_data_root: str | Path | None = None,
) -> Track2Dataset:

    resolved = resolve_track2_path(
        dataset=dataset,
        track2_path=track2_path,
        track_data_root=track_data_root,
    )
    _require_file(resolved, "Track2 permanence file")

    data = load_msgpack(resolved)

    try:
        block_colors = [str(c).lower() for c in data["blockColors"]]
    except KeyError as exc:
        raise KeyError("Track2 permanence is missing key 'blockColors'") from exc

    try:
        x_positions = np.asarray(data["xPositions"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track2 permanence is missing key 'xPositions'") from exc

    try:
        frame_times_s = np.asarray(data["frameTimes_s"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track2 permanence is missing key 'frameTimes_s'") from exc

    frame_numbers_raw = data.get("frameNumbers", None)
    if frame_numbers_raw is None:
        frame_numbers = np.arange(frame_times_s.shape[0], dtype=int)
    else:
        frame_numbers = np.asarray(frame_numbers_raw, dtype=int)

    if x_positions.ndim != 2:
        raise ValueError(f"xPositions must be 2D. Got shape {x_positions.shape}")

    n_frames, _ = x_positions.shape
    if frame_times_s.shape[0] != n_frames:
        raise ValueError(
            f"time vector length ({frame_times_s.shape[0]}) does not match xPositions rows ({n_frames})"
        )
    if frame_numbers.shape[0] != n_frames:
        raise ValueError(
            f"frame number vector length ({frame_numbers.shape[0]}) does not match xPositions rows ({n_frames})"
        )

    purecomoving_raw = data.get("purecomovingSignal", None)
    purecomoving_signal = np.asarray(purecomoving_raw, dtype=float) if purecomoving_raw is not None else None

    return Track2Dataset(
        dataset_name=dataset,
        track2_path=str(resolved),
        original_video_path=str(data.get("originalVideoPath", "")),
        tracking_results_path=str(data.get("trackingResultsPath", "")),
        block_colors=block_colors,
        x_positions=x_positions,
        frame_times_s=frame_times_s,
        frame_numbers=frame_numbers,
        purecomoving_signal=purecomoving_signal,
    )
