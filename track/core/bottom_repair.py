from __future__ import annotations

import importlib.util
from pathlib import Path


def load_legacy_manual_repair_module():
    module_path = Path(__file__).resolve().parents[1] / ".archive" / "2b.repair_unrepairable_black.py"
    spec = importlib.util.spec_from_file_location("legacy_manual_repair_black", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load manual repair module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
