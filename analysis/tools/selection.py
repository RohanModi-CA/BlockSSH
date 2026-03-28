from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .bonds import load_bond_signal_dataset
from .io import load_track2_dataset
from .models import DatasetSelection, SignalRecord, Track2Dataset

CANONICAL_COMPONENTS = ("x", "y", "a")


def _prompt_yes_no(prompt: str) -> bool:
    response = input(prompt).strip().lower()
    return response in {"y", "yes"}


def _prompt_component_choice(prompt: str, available: list[str]) -> str:
    allowed = set(available)
    default_choice = "x" if "x" in allowed else available[0]
    while True:
        response = input(prompt).strip().lower()
        if response == "":
            return default_choice
        if response in allowed:
            return response
        print(f"Please choose one of: {', '.join(available)} [default: {default_choice}]")


def _logical_component_to_physical_suffix(contains: list[str]) -> dict[str, str]:
    return {
        logical_component: CANONICAL_COMPONENTS[idx]
        for idx, logical_component in enumerate(contains)
    }


def _shared_logical_components(component_entries: list[tuple[str, dict[str, Any]]]) -> list[str]:
    if len(component_entries) == 0:
        return []

    shared = set(component_entries[0][1]["contains"])
    for _, entry in component_entries[1:]:
        shared &= set(entry["contains"])

    return [component for component in CANONICAL_COMPONENTS if component in shared]


def _resolve_component_dataset_names(
    raw_entries: OrderedDict[str, dict[str, Any]]
) -> list[tuple[str, bool, list[int], list[int]]]:
    component_entries = [
        (dataset_name, entry)
        for dataset_name, entry in raw_entries.items()
        if entry["include"] and entry["contains"] is not None
    ]
    component_choice_by_dataset: dict[str, str] = {}

    if component_entries:
        same_choice = _prompt_yes_no(
            "Use the same component for all datasets with 'contains'? [y/N] "
        )
        if same_choice:
            shared_available = _shared_logical_components(component_entries)
            if len(shared_available) == 0:
                raise ValueError("No overlapping logical components exist across the included datasets")
            shared_choice = _prompt_component_choice(
                f"Which component should be used for all of them ({'/'.join(shared_available)})? ",
                shared_available,
            )
            for dataset_name, entry in component_entries:
                component_choice_by_dataset[dataset_name] = shared_choice
        else:
            for dataset_name, entry in component_entries:
                component_choice_by_dataset[dataset_name] = _prompt_component_choice(
                    f"Dataset '{dataset_name}' component ({'/'.join(entry['contains'])})? ",
                    entry["contains"],
                )

    resolved: list[tuple[str, bool, list[int], list[int]]] = []
    for dataset_name, entry in raw_entries.items():
        include = entry["include"]
        discards = entry["discards"]
        pair_ids = entry["pair_ids"]
        contains = entry["contains"]
        if include and contains is not None:
            physical_suffix = _logical_component_to_physical_suffix(contains)[
                component_choice_by_dataset[dataset_name]
            ]
            resolved_name = f"{dataset_name}_{physical_suffix}"
        else:
            resolved_name = dataset_name
        resolved.append((resolved_name, include, discards, pair_ids))
    return resolved


def load_dataset_selection_entries(path: str | Path) -> OrderedDict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise ValueError("Top-level JSON must be a non-empty object keyed by dataset stem")

    validated: OrderedDict[str, dict[str, Any]] = OrderedDict()
    required = {"include", "discards", "pair_ids"}
    valid_contains = {"x", "y", "a"}

    for dataset_name, entry in cfg.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("Each top-level JSON key must be a non-empty dataset string")
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset '{dataset_name}' entry must be an object")

        missing = required.difference(entry.keys())
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required key(s): {sorted(missing)}"
            )

        include = entry["include"]
        discards = entry["discards"]
        pair_ids = entry["pair_ids"]

        if not isinstance(include, bool):
            raise ValueError(f"Dataset '{dataset_name}' field 'include' must be boolean")
        if not isinstance(discards, list):
            raise ValueError(f"Dataset '{dataset_name}' field 'discards' must be a list")
        if not isinstance(pair_ids, list):
            raise ValueError(f"Dataset '{dataset_name}' field 'pair_ids' must be a list")

        discards_out: list[int] = []
        for idx in discards:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid discard index {idx!r}; expected non-negative int"
                )
            discards_out.append(int(idx))

        pair_ids_out: list[int] = []
        for pair_id in pair_ids:
            if not isinstance(pair_id, int):
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid pair_id {pair_id!r}; expected int"
                )
            pair_ids_out.append(int(pair_id))

        contains = entry.get("contains")
        contains_out: list[str] | None = None
        if contains is not None:
            if not isinstance(contains, list) or len(contains) == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' field 'contains' must be a non-empty list when provided"
                )
            contains_out = []
            for axis in contains:
                if not isinstance(axis, str) or axis not in valid_contains:
                    raise ValueError(
                        f"Dataset '{dataset_name}' has invalid contains entry {axis!r}; expected one of {sorted(valid_contains)}"
                    )
                if axis in contains_out:
                    raise ValueError(
                        f"Dataset '{dataset_name}' field 'contains' may not contain duplicates"
                    )
                contains_out.append(axis)
            if len(contains_out) > len(CANONICAL_COMPONENTS):
                raise ValueError(
                    f"Dataset '{dataset_name}' field 'contains' has too many entries"
                )

        validated[dataset_name] = {
            "include": include,
            "discards": discards_out,
            "pair_ids": pair_ids_out,
            "contains": contains_out,
        }

    return validated


def load_dataset_selection(path: str | Path) -> OrderedDict[str, DatasetSelection]:
    raw_entries = load_dataset_selection_entries(path)
    validated: OrderedDict[str, DatasetSelection] = OrderedDict()

    for resolved_name, include, discards_out, pair_ids_out in _resolve_component_dataset_names(raw_entries):
        validated[resolved_name] = DatasetSelection(
            include=include,
            discards=discards_out,
            pair_ids=pair_ids_out,
        )

    return validated


def _build_bond_signal_records_for_dataset(
    dataset_name: str,
    selection: DatasetSelection,
    *,
    track_data_root: str | None = None,
    bond_spacing_mode: str = "default",
) -> list[SignalRecord]:
    bond_dataset = load_bond_signal_dataset(
        dataset=dataset_name,
        track_data_root=track_data_root,
        bond_spacing_mode=bond_spacing_mode,
    )
    n_pairs = int(bond_dataset.signal_matrix.shape[1])

    remaining_local_indices = [
        local_idx for local_idx in range(n_pairs) if local_idx not in set(selection.discards)
    ]

    if len(remaining_local_indices) != len(selection.pair_ids):
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(remaining_local_indices)} remaining local bonds after discards "
            f"but {len(selection.pair_ids)} config pair_ids were provided; these lengths must match exactly"
        )

    records: list[SignalRecord] = []
    for local_idx, requested_pair_id in zip(remaining_local_indices, selection.pair_ids):
        label = bond_dataset.pair_labels[local_idx] if local_idx < len(bond_dataset.pair_labels) else "?"
        records.append(
            SignalRecord(
                dataset_name=dataset_name,
                entity_id=int(requested_pair_id),
                local_index=int(local_idx),
                label=str(label).lower(),
                signal_kind="bond",
                source_path=bond_dataset.source_path,
                t=np.asarray(bond_dataset.frame_times_s, dtype=float),
                y=np.asarray(bond_dataset.signal_matrix[:, local_idx], dtype=float),
            )
        )
    return records


def _build_site_signal_records_for_dataset(
    dataset_name: str,
    selection: DatasetSelection,
    track2: Track2Dataset,
) -> list[SignalRecord]:
    x_positions = np.asarray(track2.x_positions, dtype=float)
    _, n_blocks = x_positions.shape
    n_pairs = max(0, n_blocks - 1)

    remaining_local_bonds = [i for i in range(n_pairs) if i not in set(selection.discards)]
    if len(remaining_local_bonds) != len(selection.pair_ids):
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(remaining_local_bonds)} remaining local bonds after discards "
            f"but {len(selection.pair_ids)} config pair_ids were provided; these lengths must match exactly"
        )

    site_mapping: dict[int, int] = {}
    for local_bond_idx, global_bond_id in zip(remaining_local_bonds, selection.pair_ids):
        left_site = int(global_bond_id)
        right_site = int(global_bond_id + 1)

        if local_bond_idx in site_mapping and site_mapping[local_bond_idx] != left_site:
            raise ValueError(
                f"Dataset '{dataset_name}' has conflicting site mapping for local block {local_bond_idx}"
            )
        if (local_bond_idx + 1) in site_mapping and site_mapping[local_bond_idx + 1] != right_site:
            raise ValueError(
                f"Dataset '{dataset_name}' has conflicting site mapping for local block {local_bond_idx + 1}"
            )

        site_mapping[local_bond_idx] = left_site
        site_mapping[local_bond_idx + 1] = right_site

    records: list[SignalRecord] = []
    for local_block_idx, site_id in sorted(site_mapping.items()):
        label = track2.block_colors[local_block_idx] if local_block_idx < len(track2.block_colors) else "?"
        records.append(
            SignalRecord(
                dataset_name=dataset_name,
                entity_id=int(site_id),
                local_index=int(local_block_idx),
                label=str(label).lower(),
                signal_kind="site",
                source_path=track2.track2_path,
                t=track2.frame_times_s,
                y=np.asarray(x_positions[:, local_block_idx], dtype=float),
            )
        )
    return records


def build_configured_bond_signals(
    config: OrderedDict[str, DatasetSelection],
    *,
    track_data_root: str | None = None,
    allow_duplicate_ids: bool = False,
    bond_spacing_mode: str = "default",
) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    seen_ids: set[int] = set()

    for dataset_name, selection in config.items():
        if not selection.include:
            continue

        dataset_records = _build_bond_signal_records_for_dataset(
            dataset_name,
            selection,
            track_data_root=track_data_root,
            bond_spacing_mode=bond_spacing_mode,
        )

        for record in dataset_records:
            if (not allow_duplicate_ids) and (record.entity_id in seen_ids):
                warnings.warn(
                    f"Skipping duplicate bond id {record.entity_id} from dataset '{dataset_name}' "
                    "because an earlier occurrence was already accepted"
                )
                continue
            records.append(record)
            seen_ids.add(record.entity_id)

    return records


def build_grouped_configured_bond_signals(
    config: OrderedDict[str, DatasetSelection],
    *,
    track_data_root: str | None = None,
    bond_spacing_mode: str = "default",
) -> OrderedDict[int, list[SignalRecord]]:
    grouped: OrderedDict[int, list[SignalRecord]] = OrderedDict()

    for dataset_name, selection in config.items():
        if not selection.include:
            continue

        dataset_records = _build_bond_signal_records_for_dataset(
            dataset_name,
            selection,
            track_data_root=track_data_root,
            bond_spacing_mode=bond_spacing_mode,
        )

        for record in dataset_records:
            grouped.setdefault(int(record.entity_id), []).append(record)

    return grouped


def build_configured_site_signals(
    config: OrderedDict[str, DatasetSelection],
    *,
    track_data_root: str | None = None,
    allow_duplicate_ids: bool = False,
) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    seen_ids: set[int] = set()

    for dataset_name, selection in config.items():
        if not selection.include:
            continue

        track2 = load_track2_dataset(dataset=dataset_name, track_data_root=track_data_root)
        dataset_records = _build_site_signal_records_for_dataset(dataset_name, selection, track2)

        for record in dataset_records:
            if (not allow_duplicate_ids) and (record.entity_id in seen_ids):
                warnings.warn(
                    f"Skipping duplicate site id {record.entity_id} from dataset '{dataset_name}' "
                    "because an earlier occurrence was already accepted"
                )
                continue
            records.append(record)
            seen_ids.add(record.entity_id)

    return records


def _normalize_display_bond_numbers(values: list[int] | None, *, arg_name: str) -> list[int] | None:
    if values is None:
        return None

    out: list[int] = []
    for value in values:
        if int(value) < 1:
            raise ValueError(f"{arg_name} values must be positive 1-based bond numbers")
        out.append(int(value))
    return sorted(set(out))


def collect_display_bond_numbers(records: list[SignalRecord]) -> list[int]:
    return sorted({int(record.entity_id) + 1 for record in records})


def filter_signal_records_by_display_bonds(
    records: list[SignalRecord],
    *,
    only_bonds: list[int] | None = None,
    exclude_bonds: list[int] | None = None,
    parity: str | None = None,
) -> list[SignalRecord]:
    only_set = set(_normalize_display_bond_numbers(only_bonds, arg_name="--only-bonds") or [])
    exclude_set = set(_normalize_display_bond_numbers(exclude_bonds, arg_name="--exclude-bonds") or [])

    if only_set and (only_set & exclude_set):
        overlap = sorted(only_set & exclude_set)
        raise ValueError(f"Bond numbers cannot appear in both --only-bonds and --exclude-bonds: {overlap}")

    if parity not in {None, "odd", "even"}:
        raise ValueError("parity must be one of None, 'odd', or 'even'")

    filtered: list[SignalRecord] = []
    for record in records:
        display_bond = int(record.entity_id) + 1
        if only_set and display_bond not in only_set:
            continue
        if display_bond in exclude_set:
            continue
        if parity == "odd" and display_bond % 2 == 0:
            continue
        if parity == "even" and display_bond % 2 != 0:
            continue
        filtered.append(record)

    return filtered
