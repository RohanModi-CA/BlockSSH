#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track.core.bottom_label import build_parser, run_label


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_label(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
