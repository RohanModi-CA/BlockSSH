#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "viz" / "spacing_timeseries.py"
    spec = importlib.util.spec_from_file_location("analysis_go_timeseries_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Timeseries implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _component_track2_path(dataset: str, component: str, track_data_root: str | None) -> Path:
    root = Path(track_data_root) if track_data_root is not None else Path(__file__).resolve().parents[2] / "track" / "data"
    component = str(component).strip()
    dataset = str(dataset).strip()

    if component == "x":
        return root / dataset / "components" / "x" / "track2_permanence.msgpack"

    candidate = root / dataset / "components" / component / "track2_permanence.msgpack"
    if candidate.is_file():
        return candidate

    legacy = root / f"{dataset}_{component}" / "track2_permanence.msgpack"
    if legacy.is_file():
        return legacy

    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot timeseries for a dataset, defaulting to component x unless another component is requested.",
    )
    parser.add_argument("dataset", nargs="?", help="Dataset stem, e.g. IMG_0584. Ignored if --track2 is given.")
    parser.add_argument("--track2", default=None, help="Explicit path to track2_permanence.msgpack.")
    parser.add_argument(
        "--track-data-root",
        default=None,
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )
    parser.add_argument("--save", default=None, help="Optional path to save the figure.")
    parser.add_argument("--title", default=None, help="Optional title override.")
    parser.add_argument(
        "--timeseriesnorm",
        dest="timeseriesnorm",
        action="store_true",
        help="Rescale each displayed spacing time series to RMS 100 (default).",
    )
    parser.add_argument(
        "--no-timeseriesnorm",
        dest="timeseriesnorm",
        action="store_false",
        help="Disable RMS-100 timeseries normalization.",
    )
    parser.add_argument(
        "--not-bonds",
        action="store_true",
        help="Plot raw block traces instead of adjacent bond-spacing traces.",
    )
    parser.add_argument(
        "--component",
        "--show",
        default="x",
        help="Component to display. Default: x",
    )
    parser.set_defaults(timeseriesnorm=True)
    return parser


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(sys.argv[1:])

    delegated = [sys.argv[0]]
    if args.track2 is not None:
        delegated.extend(["--track2", str(args.track2)])
    elif args.dataset is not None:
        delegated.append(str(args.dataset))
        component = str(args.component).strip()
        if component != "x":
            delegated.extend(
                [
                    "--track2",
                    str(_component_track2_path(args.dataset, component, args.track_data_root)),
                ]
            )

    if args.track_data_root is not None:
        delegated.extend(["--track-data-root", str(args.track_data_root)])
    if args.save is not None:
        delegated.extend(["--save", str(args.save)])
    if args.title is not None:
        delegated.extend(["--title", str(args.title)])
    if args.timeseriesnorm:
        delegated.append("--timeseriesnorm")
    if args.not_bonds:
        delegated.append("--not-bonds")

    delegated.extend(passthrough)

    module = _load_module()
    original = list(sys.argv)
    try:
        sys.argv = delegated
        return int(module.main())
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
