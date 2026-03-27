#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track.core.bottom_repair import load_legacy_manual_repair_module


def main() -> int:
    track_dir = Path(__file__).resolve().parents[1]
    os.chdir(track_dir)
    module = load_legacy_manual_repair_module()
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
