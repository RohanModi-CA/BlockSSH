from __future__ import annotations

import argparse
from pathlib import Path

from .bottom_manifest import load_manifest as _load_manifest, save_manifest as _save_manifest
from .io import load_json, load_msgpack, save_json
from .layout import component_track2_path, labels_path


def _load_component_block_count(name: str, component: str = "x") -> int:
    path = component_track2_path(name, component)
    if not path.exists():
        raise FileNotFoundError(
            f"Component permanence not found: {path}\n"
            f"Run first: python3 Bottom/2.ProcessVerify.py {name}"
        )
    payload = load_msgpack(path)
    return len(payload.get("blockColors", []))


def _parse_disabled_list(raw: str, max_value: int) -> list[int]:
    raw = raw.strip()
    if raw == "":
        return []
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if token == "":
            continue
        value = int(token)
        if value < 1 or value > max_value:
            raise ValueError(f"Value {value} out of range 1..{max_value}")
        if value not in out:
            out.append(value)
    return sorted(out)


def _build_default_labels(prefix: str, n_items: int) -> dict[str, str]:
    return {str(idx): f"{prefix} {idx}" for idx in range(1, n_items + 1)}


def _parse_label_list(raw: str, n_items: int) -> list[str]:
    labels = [part.strip() for part in raw.split(",")]
    if len(labels) != n_items:
        raise ValueError(f"Expected exactly {n_items} labels, got {len(labels)}")
    if any(label == "" for label in labels):
        raise ValueError("Labels may not be empty")
    return labels


def _prompt_explicit_labels(kind: str, n_items: int) -> dict[str, str]:
    default_labels = [str(idx) for idx in range(1, n_items + 1)]
    if n_items == 0:
        return {}

    print(
        f"{kind.title()} labels left-to-right as comma-separated values "
        f"[{', '.join(default_labels)}]:"
    )
    while True:
        raw = input("> ").strip()
        if raw == "":
            labels = default_labels
            break
        try:
            labels = _parse_label_list(raw, n_items)
            break
        except ValueError as exc:
            print(f"  {exc}")
            print(f"  Please enter exactly {n_items} comma-separated {kind} labels.")

    return {str(idx): label for idx, label in enumerate(labels, start=1)}


def run_label(name: str) -> int:
    dataset = Path(name).stem
    manifest = _load_manifest(dataset)
    n_sites = _load_component_block_count(dataset, "x")
    if n_sites <= 0:
        raise RuntimeError("Could not infer any sites from the x component permanence output.")
    n_bonds = max(0, n_sites - 1)

    print(f"Dataset: {dataset}")
    print(f"Sites:   {n_sites}")
    print(f"Bonds:   {n_bonds}")
    print("Bond/site indexing is 1-based from left to right.")
    print("Press Enter to keep simple numeric defaults.")

    site_labels = _prompt_explicit_labels("site", n_sites)

    disabled_sites = _parse_disabled_list(
        input(f"Disabled sites as comma-separated 1..{n_sites} []: "),
        n_sites,
    )
    notes = input("Notes []: ").strip()

    payload = {
        "dataset": dataset,
        "version": 1,
        "bond_labels": {},
        "site_labels": site_labels,
        "disabled": {
            "sites": disabled_sites,
            "components": {},
        },
        "notes": notes,
        "created_by": "3.Label.py",
    }
    out_path = save_json(labels_path(dataset), payload)

    manifest.labels_file = out_path.name
    _save_manifest(dataset, manifest)

    print(f"\nLabels saved to: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or update dataset-local labels for the tracking workflow.")
    parser.add_argument("name", help="Dataset name, e.g. IMG_9282 or 9282")
    return parser
