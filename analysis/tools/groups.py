from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from pathlib import Path

from .catalog import DEFAULT_GROUPS_DIR
from .io import get_default_track_data_root, load_track2_dataset

CANONICAL_COMPONENTS = ("x", "y", "a")


def group_path_from_name(name: str, groups_dir: str | Path | None = None) -> Path:
    root = Path(groups_dir) if groups_dir is not None else DEFAULT_GROUPS_DIR
    path = root / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Group JSON not found: {path}")
    return path


def load_group_json(name: str, groups_dir: str | Path | None = None) -> dict:
    path = group_path_from_name(name, groups_dir=groups_dir)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Group JSON must be an object: {path}")
    return payload


def load_group_datasets(name: str, groups_dir: str | Path | None = None) -> list[str]:
    payload = load_group_json(name, groups_dir=groups_dir)
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError(f"Group '{name}' has no datasets")
    return [str(dataset) for dataset in datasets]


def _dataset_root(base_dataset: str, track_data_root: str | Path | None = None) -> Path:
    root = Path(track_data_root) if track_data_root is not None else get_default_track_data_root()
    return root / str(base_dataset)


def _manifest_path(base_dataset: str, track_data_root: str | Path | None = None) -> Path:
    return _dataset_root(base_dataset, track_data_root=track_data_root) / "manifest.json"


def _labels_path(base_dataset: str, track_data_root: str | Path | None = None) -> Path:
    return _dataset_root(base_dataset, track_data_root=track_data_root) / "labels.json"


def _available_components_from_manifest(
    base_dataset: str,
    track_data_root: str | Path | None = None,
) -> list[str]:
    manifest_path = _manifest_path(base_dataset, track_data_root=track_data_root)
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        components = payload.get("components", {})
        if isinstance(components, dict):
            ordered = [component for component in CANONICAL_COMPONENTS if component in components]
            if len(ordered) > 0:
                return ordered

    ordered: list[str] = []
    dataset_root = _dataset_root(base_dataset, track_data_root=track_data_root)
    for component in CANONICAL_COMPONENTS:
        if (dataset_root / "components" / component / "track2_permanence.msgpack").is_file():
            ordered.append(component)
            continue
        if (dataset_root.parent / f"{base_dataset}_{component}" / "track2_permanence.msgpack").is_file():
            ordered.append(component)
    return ordered


def _coerce_numeric_site_labels(values: list[str]) -> list[int] | None:
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError):
        return None


def _derive_dataset_selection_entry(
    base_dataset: str,
    *,
    track_data_root: str | Path | None = None,
) -> dict:
    track2 = load_track2_dataset(dataset=f"{base_dataset}_x", track_data_root=track_data_root)
    n_sites = int(track2.x_positions.shape[1])
    if n_sites < 2:
        raise ValueError(f"Dataset '{base_dataset}' has fewer than 2 tracked sites")

    n_bonds = n_sites - 1
    labels_path = _labels_path(base_dataset, track_data_root=track_data_root)
    disabled_site_indices: set[int] = set()
    site_label_values = [str(idx) for idx in range(1, n_sites + 1)]

    if labels_path.is_file():
        with labels_path.open("r", encoding="utf-8") as fh:
            labels_payload = json.load(fh)

        disabled = labels_payload.get("disabled", {})
        disabled_sites = disabled.get("sites", []) if isinstance(disabled, dict) else []
        disabled_site_indices = {int(value) - 1 for value in disabled_sites if int(value) >= 1}

        site_labels = labels_payload.get("site_labels", {})
        if isinstance(site_labels, dict):
            candidate_values: list[str] = []
            for idx in range(1, n_sites + 1):
                candidate_values.append(str(site_labels.get(str(idx), idx)))
            site_label_values = candidate_values

    numeric_site_labels = _coerce_numeric_site_labels(site_label_values)
    pair_ids: list[int] = []
    discards: list[int] = []

    for local_bond_idx in range(n_bonds):
        if local_bond_idx in disabled_site_indices or (local_bond_idx + 1) in disabled_site_indices:
            discards.append(local_bond_idx)
            continue

        if numeric_site_labels is None:
            pair_ids.append(local_bond_idx)
            continue

        left_site = int(numeric_site_labels[local_bond_idx])
        right_site = int(numeric_site_labels[local_bond_idx + 1])
        pair_ids.append(min(left_site, right_site) - 1)

    entry = {
        "contains": _available_components_from_manifest(base_dataset, track_data_root=track_data_root) or list(CANONICAL_COMPONENTS),
        "include": True,
        "discards": discards,
        "pair_ids": pair_ids,
    }
    return entry


def build_selection_config_payload(
    datasets: list[str],
    *,
    track_data_root: str | Path | None = None,
) -> OrderedDict[str, dict]:
    payload: OrderedDict[str, dict] = OrderedDict()
    for dataset in datasets:
        payload[str(dataset)] = _derive_dataset_selection_entry(str(dataset), track_data_root=track_data_root)
    return payload


def write_temp_selection_config(
    datasets: list[str],
    *,
    track_data_root: str | Path | None = None,
    prefix: str = "analysis_group_",
) -> tempfile.NamedTemporaryFile:
    payload = build_selection_config_payload(datasets, track_data_root=track_data_root)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=prefix,
        delete=False,
    )
    with handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return handle


def write_temp_component_selection_config(
    datasets: list[str],
    *,
    component: str,
    track_data_root: str | Path | None = None,
    prefix: str = "analysis_component_group_",
) -> tempfile.NamedTemporaryFile:
    component_name = str(component).strip()
    if component_name not in CANONICAL_COMPONENTS:
        raise ValueError(f"Unsupported component: {component_name}")

    base_payload = build_selection_config_payload(datasets, track_data_root=track_data_root)
    payload: OrderedDict[str, dict] = OrderedDict()
    for dataset, entry in base_payload.items():
        payload[f"{dataset}_{component_name}"] = {
            "contains": None,
            "include": bool(entry["include"]),
            "discards": list(entry["discards"]),
            "pair_ids": list(entry["pair_ids"]),
        }

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=prefix,
        delete=False,
    )
    with handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return handle
