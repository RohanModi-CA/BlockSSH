#!/usr/bin/env python3
"""
B2.batch_verify_process_black.py  —  Run black-video verification and X / Y / angle
permanence construction on multiple datasets.

Scans data/ for datasets containing track1.msgpack and runs:
    python3 Bottom/2.ProcessVerify.py <name>

The interactive repair prompt inside step 2 is preserved.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from helper.video_io import track1_output_path


DATA_DIR = "data"


def find_datasets() -> list[str]:
    names: list[str] = []
    if not os.path.isdir(DATA_DIR):
        return names

    for d in sorted(os.listdir(DATA_DIR)):
        t1 = track1_output_path(d)
        if os.path.exists(t1):
            names.append(d)

    return names


def _match_name(name: str, token: str) -> bool:
    token = os.path.splitext(os.path.basename(token))[0]
    return token == name or name.endswith(token)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Bottom/2.ProcessVerify.py on multiple datasets."
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Dataset names or numeric suffixes to exclude.",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.50,
        help="Min spacing ratio relative to the reference spacing.",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=1.50,
        help="Max spacing ratio relative to the reference spacing.",
    )
    parser.add_argument(
        "--no-trim-ends",
        action="store_true",
        help="Pass through to Bottom/2.ProcessVerify.py.",
    )
    parser.add_argument(
        "--min-end-support",
        type=int,
        default=3,
        help="Pass through to Bottom/2.ProcessVerify.py.",
    )
    args = parser.parse_args()

    datasets = find_datasets()
    if not datasets:
        print("No datasets with track1.msgpack found in data/.")
        return

    tasks = []
    for name in datasets:
        if any(_match_name(name, tok) for tok in args.exclude):
            continue
        tasks.append(name)

    if not tasks:
        print("No datasets selected for verification.")
        return

    print(f"{len(tasks)} dataset(s) will be processed:\n")
    for name in tasks:
        print(f"  {name}")
    print()

    for i, name in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Verifying {name}...\n")

        cmd = [
            "python3",
            "Bottom/2.ProcessVerify.py",
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

    print("Batch black verification complete.")


if __name__ == "__main__":
    main()
