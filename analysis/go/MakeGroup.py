#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.catalog import DEFAULT_GROUPS_DIR, list_base_datasets, save_group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a simple reusable dataset group definition.")
    parser.add_argument("items", nargs="+", help="Datasets followed by the group name as the final argument.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_GROUPS_DIR),
        help="Directory to store group JSON files. Default: analysis/configs/groups/",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if len(args.items) < 2:
        print("Error: provide at least one dataset and a final group name", file=sys.stderr)
        return 1

    datasets = [str(item) for item in args.items[:-1]]
    group_name = str(args.items[-1])
    known = set(list_base_datasets())
    missing = [dataset for dataset in datasets if dataset not in known]
    if missing:
        print(f"Warning: these datasets were not found in track/data right now: {missing}", file=sys.stderr)

    out_path = Path(args.out_dir) / f"{group_name}.json"
    saved = save_group(out_path, name=group_name, datasets=datasets)
    print(f"Group saved to: {saved}")
    print(f"Datasets: {datasets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
