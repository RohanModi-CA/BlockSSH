#!/usr/bin/env python3
"""
Compatibility wrapper for the refactored bottom-tracking manual repair step.

The active user-facing path is:
    python3 track/Bottom/2b.ManualRepair.py DATASET
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track.core.bottom_repair import load_legacy_manual_repair_module


def main() -> int:
    print("Note: 2b.repair_unrepairable_black.py is deprecated. Forwarding to Bottom/2b.ManualRepair.py")
    module = load_legacy_manual_repair_module()
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
