#!/usr/bin/env python3
"""
Compatibility wrapper for the refactored bottom-tracking prepare step.

The active user-facing path is:
    python3 track/Bottom/0.VideoPrepareBottom.py DATASET
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track.core.bottom_prepare import build_parser, run_prepare


def main() -> int:
    print("Note: 0.video_prepare_black.py is deprecated. Forwarding to Bottom/0.VideoPrepareBottom.py")
    parser = build_parser()
    args = parser.parse_args()
    return run_prepare(args.name, no_preview=args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
