#!/usr/bin/env python3
"""
Compatibility wrapper for the refactored bottom batch verify/process step.

The active user-facing path is:
    python3 track/Bottom/B2.BatchProcessVerify.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_bottom_batch_module():
    module_path = Path(__file__).resolve().parent / "Bottom" / "B2.BatchProcessVerify.py"
    spec = importlib.util.spec_from_file_location("bottom_batch_process_verify", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    print("Note: B2.batch_verify_process_black.py is deprecated. Forwarding to Bottom/B2.BatchProcessVerify.py")
    module = _load_bottom_batch_module()
    raise SystemExit(module.main())
