#!/usr/bin/env python3
"""
B1.batch_verify_process.py  —  Run verification + permanence construction
on multiple datasets.

Scans data/ for datasets containing track1.msgpack and runs:

    python3 .archive/2.verify_and_process.py <name>

The interactive repair prompt inside step 2 (y/N) is preserved.

Flags
-----
--exclude NAME [NAME ...]
    Skip particular datasets. Names can be full names (IMG_9282)
    or numeric suffixes (9282), using the same matching logic used
    elsewhere in the pipeline.

--ratio-min FLOAT
    Pass through to .archive/2.verify_and_process.py (default 0.50)

--ratio-max FLOAT
    Pass through to .archive/2.verify_and_process.py (default 1.50)

Usage
-----
    python3 B1.batch_verify_process.py
    python3 B1.batch_verify_process.py --exclude 9282 IMG_0537
    python3 B1.batch_verify_process.py --ratio-min 0.6 --ratio-max 1.4
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper.video_io import track1_output_path


DATA_DIR = "data"


def find_datasets():
    """Return dataset names that contain track1.msgpack."""
    names = []
    if not os.path.isdir(DATA_DIR):
        return names

    for d in sorted(os.listdir(DATA_DIR)):
        t1 = track1_output_path(d)
        if os.path.exists(t1):
            names.append(d)

    return names


def _match_name(name: str, token: str) -> bool:
    """
    Return True if token refers to this dataset name.

    Accepts either:
      • full name (IMG_9282)
      • numeric suffix (9282)
    """
    token = os.path.splitext(os.path.basename(token))[0]

    if token == name:
        return True

    if name.endswith(token):
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run .archive/2.verify_and_process.py on multiple datasets."
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Dataset names or numeric suffixes to exclude."
    )

    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.50,
        help="Min spacing ratio relative to reference."
    )

    parser.add_argument(
        "--ratio-max",
        type=float,
        default=1.50,
        help="Max spacing ratio relative to reference."
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
            ".archive/2.verify_and_process.py",
            name,
            "--ratio-min",
            str(args.ratio_min),
            "--ratio-max",
            str(args.ratio_max),
        ]

        # Important: do NOT suppress stdin so the y/N prompt still works
        subprocess.run(cmd)

        print()

    print("Batch verification complete.")


if __name__ == "__main__":
    main()
