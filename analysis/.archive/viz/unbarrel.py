#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import msgpack
import numpy as np

from tools.io import get_default_track_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-off position-bias subtraction for an _area track2 dataset using the "
            "current fitted x-bias model."
        )
    )
    parser.add_argument("dataset", nargs="?", default="IMG_0681_rot270")
    parser.add_argument(
        "--track-data-root",
        default=str(get_default_track_data_root()),
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Optional explicit output dataset name. Default: <dataset>_area_unbarrel",
    )
    return parser


def load_msgpack(path: Path) -> dict:
    with open(path, "rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False)


def save_msgpack(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(msgpack.packb(payload, use_bin_type=True))


def bias_model_img_0681_rot270(x: np.ndarray) -> np.ndarray:
    x0 = 1854.06296
    c6 = 1.28708076e-15
    c4 = -3.54099248e-09
    c2 = 0.0085566775
    c0 = 310156.237
    dx = np.asarray(x, dtype=float) - x0
    return c6 * dx ** 6 + c4 * dx ** 4 + c2 * dx ** 2 + c0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.track_data_root)
    dataset = str(args.dataset)
    area_path = root / f"{dataset}_area" / "track2_permanence.msgpack"
    x_path = root / f"{dataset}_x" / "track2_permanence.msgpack"
    output_dataset = str(args.output_dataset) if args.output_dataset else f"{dataset}_area_unbarrel"
    out_path = root / output_dataset / "track2_permanence.msgpack"

    if not area_path.exists():
        raise FileNotFoundError(f"Area dataset not found: {area_path}")
    if not x_path.exists():
        raise FileNotFoundError(f"X dataset not found: {x_path}")
    if dataset != "IMG_0681_rot270":
        raise ValueError("This one-off script currently only has fitted parameters for IMG_0681_rot270.")

    area_payload = load_msgpack(area_path)
    x_payload = load_msgpack(x_path)

    area = np.asarray(area_payload["xPositions"], dtype=float)
    x = np.asarray(x_payload["xPositions"], dtype=float)
    if area.shape != x.shape:
        raise ValueError(f"Shape mismatch: area {area.shape} vs x {x.shape}")

    bias = bias_model_img_0681_rot270(x)
    corrected = area - bias

    finite_area = np.isfinite(area)
    corrected[~finite_area] = np.nan

    area_payload["xPositions"] = corrected.tolist()
    area_payload["unbarrelMetadata"] = {
        "sourceAreaTrack2Path": str(area_path),
        "sourceXTrack2Path": str(x_path),
        "mode": "subtract_even_sixth_bias_from_x",
        "parameters": {
            "x0": 1854.06296,
            "c6": 1.28708076e-15,
            "c4": -3.54099248e-09,
            "c2": 0.0085566775,
            "c0": 310156.237,
        },
    }

    save_msgpack(out_path, area_payload)

    finite_corrected = corrected[np.isfinite(corrected)]
    print(f"Input area: {area_path}")
    print(f"Input x   : {x_path}")
    print(f"Output    : {out_path}")
    print(f"Frames    : {area.shape[0]}")
    print(f"Columns   : {area.shape[1]}")
    print(
        "Corrected stats: "
        f"min={float(np.min(finite_corrected)):.3f} "
        f"median={float(np.median(finite_corrected)):.3f} "
        f"max={float(np.max(finite_corrected)):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
