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


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "viz" / "see_config_subtract_xya.py"
    spec = importlib.util.spec_from_file_location("analysis_go_logsubtract_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_explicit_logsubtract(values: list[str]) -> bool:
    return any(value == "--logsubtract" for value in values)


def _rewrite_wrapper_aliases(values: list[str]) -> list[str]:
    return list(values)


def _build_dataset_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--config-json", default=None)
    return parser


def _build_group_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--group", default=None)
    parser.add_argument("--groups-dir", default=None)
    parser.add_argument("--track-data-root", default=None)
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
    module = _load_module()
    argv = list(sys.argv[1:])

    if "--group" in argv or "--groups-dir" in argv:
        parser = _build_group_wrapper_parser()
        args, passthrough = parser.parse_known_args(argv)
        dataset = None
        config_json = None
        group = args.group
        groups_dir = args.groups_dir
        track_data_root = args.track_data_root
    else:
        parser = _build_dataset_wrapper_parser()
        args, passthrough = parser.parse_known_args(argv)
        dataset = args.dataset
        config_json = args.config_json
        group = None
        groups_dir = None
        track_data_root = args.track_data_root

    passthrough = _rewrite_wrapper_aliases(list(passthrough))

    if config_json is not None:
        delegated_argv = [sys.argv[0], config_json]
    elif dataset is not None and dataset.endswith(".json") and Path(dataset).is_file():
        delegated_argv = [sys.argv[0], dataset]
    elif group is not None:
        datasets = load_group_datasets(group, groups_dir=groups_dir)
        temp_config = write_temp_selection_config(
            datasets,
            track_data_root=track_data_root,
            prefix=f"analysis_group_logsubtract_{group}_",
        )
        delegated_argv = [sys.argv[0], temp_config.name]
    elif dataset is not None:
        temp_config = write_temp_selection_config(
            [dataset],
            track_data_root=track_data_root,
            prefix=f"analysis_dataset_logsubtract_{dataset}_",
        )
        delegated_argv = [sys.argv[0], temp_config.name]
    else:
        delegated_argv = list(sys.argv)

    if (dataset is not None or group is not None or config_json is not None) and not _has_explicit_logsubtract(
        passthrough
    ):
        delegated_argv.extend(["--logsubtract", "x"])

    delegated_argv.extend(passthrough)
    with _temporary_argv(delegated_argv):
        return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
