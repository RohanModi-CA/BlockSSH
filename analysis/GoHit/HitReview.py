#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_module():
    module_path = Path(__file__).resolve().parent / "viz" / "hit_review.py"
    spec = importlib.util.spec_from_file_location("analysis_gohit_hit_review_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GoHit review implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_module()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
