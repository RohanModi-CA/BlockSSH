#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track.core.bottom_verify import build_parser, run_process_verify


def main() -> int:
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
