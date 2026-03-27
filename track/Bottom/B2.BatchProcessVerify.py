#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track.core.layout import DATA_DIR, track1_path


def find_datasets() -> list[str]:
    names: list[str] = []
    if not DATA_DIR.is_dir():
        return names
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_dir() and track1_path(path.name).exists():
            names.append(path.name)
    return names


def _match_name(name: str, token: str) -> bool:
    token = Path(token).stem
    return token == name or name.endswith(token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bottom/2.ProcessVerify.py on multiple datasets.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Dataset names or numeric suffixes to exclude.")
    parser.add_argument("--ratio-min", type=float, default=0.50, help="Min spacing ratio relative to the reference spacing.")
    parser.add_argument("--ratio-max", type=float, default=1.50, help="Max spacing ratio relative to the reference spacing.")
    parser.add_argument("--no-trim-ends", action="store_true", help="Pass through to Bottom/2.ProcessVerify.py.")
    parser.add_argument("--min-end-support", type=int, default=3, help="Pass through to Bottom/2.ProcessVerify.py.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    datasets = [name for name in find_datasets() if not any(_match_name(name, token) for token in args.exclude)]
    if not datasets:
        print("No datasets selected for bottom verification.")
        return 0

    print(f"{len(datasets)} dataset(s) will be processed:\n")
    for name in datasets:
        print(f"  {name}")
    print()

    for idx, name in enumerate(datasets, start=1):
        print(f"[{idx}/{len(datasets)}] Verifying {name}...\n")
        cmd = [
            "python3",
            "track/Bottom/2.ProcessVerify.py",
            name,
            "--ratio-min",
            str(args.ratio_min),
            "--ratio-max",
            str(args.ratio_max),
            "--min-end-support",
            str(args.min_end_support),
        ]
        if args.no_trim_ends:
            cmd.append("--no-trim-ends")
        subprocess.run(cmd)
        print()

    print("Bottom batch verification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
