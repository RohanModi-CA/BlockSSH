#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.GoHit.tools.cli import add_hit_mode_args
from analysis.tools.groups import load_group_datasets, write_temp_selection_config


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FFT implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--group", default=None)
    parser.add_argument("--groups-dir", default=None)
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--average", action="store_true")
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)
    add_hit_mode_args(parser)
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
    try:
        parser = _build_wrapper_parser()
        args, passthrough = parser.parse_known_args(sys.argv[1:])

        if args.hits:
            if args.group is not None or args.groups_dir is not None:
                raise ValueError("GoHit FFT --hits does not support --group yet")
            if args.dataset is None:
                raise ValueError("GoHit FFT --hits requires a dataset argument")
            from analysis.GoHit.tools.hits import load_catalog_if_available

            if load_catalog_if_available(args.dataset) is None:
                raise ValueError(
                    f"No GoHit confirmed hit catalog exists for dataset '{args.dataset}'. "
                    f"Run python3 analysis/GoHit/HitReview.py {args.dataset} first."
                )
            module = _load_module(
                Path(__file__).resolve().parent / "viz" / "see_fft_hits_xya.py",
                "analysis_gohit_fft_hits_impl",
            )
            delegated_argv = list(sys.argv)
            with _temporary_argv(delegated_argv):
                return int(module.main())

        if "--group" not in sys.argv[1:] and "--groups-dir" not in sys.argv[1:]:
            module = _load_module(
                Path(__file__).resolve().parents[1] / "go" / "FFT.py",
                "analysis_gohit_fft_legacy_impl",
            )
            return int(module.main())

        if args.group is None:
            raise ValueError("--groups-dir requires --group")

        passthrough = [arg for arg in passthrough if arg != "--average"]

        datasets = load_group_datasets(args.group, groups_dir=args.groups_dir)
        temp_config = write_temp_selection_config(
            datasets,
            track_data_root=args.track_data_root,
            prefix=f"analysis_group_fft_{args.group}_",
        )
        module = _load_module(
            Path(__file__).resolve().parents[1] / "viz" / "avg_fft.py",
            "analysis_gohit_avg_fft_impl",
        )
        delegated_argv = [sys.argv[0], temp_config.name] + passthrough
        if args.only is not None:
            delegated_argv.extend(["--only", args.only])
        with _temporary_argv(delegated_argv):
            return int(module.main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
