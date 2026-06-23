#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import msgpack


def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    return obj


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: export_track2_json.py INPUT.msgpack OUTPUT.json", file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    with src.open("rb") as fh:
        data = msgpack.unpackb(fh.read(), raw=False)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize(data), fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
