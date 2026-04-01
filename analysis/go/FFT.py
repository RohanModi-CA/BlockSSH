#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.groups import load_group_datasets, write_temp_selection_config


def _load_module(relpath: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "viz" / relpath
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FFT implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--group", default=None)
    parser.add_argument("--groups-dir", default=None)
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--average", action="store_true")
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)
    return parser


@contextmanager
def _temporary_argv(argv: list[str]):
    original = list(sys.argv)
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def main() -> int:
    if "--group" not in sys.argv[1:] and "--groups-dir" not in sys.argv[1:]:
        module = _load_module("see_fft_xya.py", "analysis_go_fft_impl")
        return int(module.main())

    parser = _build_wrapper_parser()
    args, passthrough = parser.parse_known_args(sys.argv[1:])
    if args.group is None:
        raise ValueError("--groups-dir requires --group")

    passthrough = [arg for arg in passthrough if arg != "--average"]

    datasets = load_group_datasets(args.group, groups_dir=args.groups_dir)
    temp_config = write_temp_selection_config(
        datasets,
        track_data_root=args.track_data_root,
        prefix=f"analysis_group_fft_{args.group}_",
    )
    module = _load_module("avg_fft.py", "analysis_go_avg_fft_impl")
    delegated_argv = [sys.argv[0], temp_config.name] + passthrough
    if args.only is not None:
        delegated_argv.extend(["--only", args.only])
    with _temporary_argv(delegated_argv):
        return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
