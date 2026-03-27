#!/usr/bin/env python3
"""
Compatibility wrapper for the refactored bottom-tracking run step.

The active user-facing path is:
    python3 track/Bottom/1.TrackRun.py DATASET
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track.core.bottom_run import build_parser, run_track


def main() -> int:
    print("Note: 1.track_run_black.py is deprecated. Forwarding to Bottom/1.TrackRun.py")
    parser = build_parser()
    args = parser.parse_args()
    return run_track(args.name, no_preview=args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
