#!/usr/bin/env python3
"""
Compatibility wrapper for the refactored bottom-tracking verify/process step.

The active user-facing path is:
    python3 track/Bottom/2.ProcessVerify.py DATASET
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track.core.bottom_verify import build_parser, run_process_verify


def main() -> int:
    print("Note: 2.verify_and_process_black.py is deprecated. Forwarding to Bottom/2.ProcessVerify.py")
    parser = build_parser()
    args = parser.parse_args()
    return run_process_verify(
        args.name,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        trim_weak_ends=not args.no_trim_ends,
        min_end_support=args.min_end_support,
    )


if __name__ == "__main__":
    raise SystemExit(main())
