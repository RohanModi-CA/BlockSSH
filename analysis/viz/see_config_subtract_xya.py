#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter

from plotting.common import render_figure
from plotting.frequency import COMPONENT_COLORS, _plot_frequency_image
from tools.cli import (
    add_average_domain_args,
    add_bond_filter_args,
    add_bond_spacing_mode_arg,
    add_colormap_arg,
    add_normalization_args,
    add_output_args,
    add_plot_scale_args,
    add_signal_processing_args,
    add_track_data_root_arg,
    resolve_normalization_mode,
)
from tools.io import default_track2_path
from tools.models import AverageSpectrumResult, DatasetSelection
from tools.selection import (
    build_configured_bond_signals,
    collect_display_bond_numbers,
    filter_signal_records_by_display_bonds,
    load_dataset_selection_entries,
)
from tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from tools.spectral import (
    ABSOLUTE_ZERO_TOL,
    compute_average_spectrum,
    compute_fft_contributions,
    compute_welch_contributions,
)

CANONICAL_COMPONENTS = ("x", "y", "a")
BASELINE_MATCH_MODES = ("none", "median-offset")
DEFAULT_BASELINE_WINDOW_BINS = 101


@dataclass
class SubtractSpec:
    component: str | None = None
    config_path: str | None = None
    scale: float = 1.0
    baseline_match: str = "none"
    baseline_window_bins: int = DEFAULT_BASELINE_WINDOW_BINS

    def is_component(self) -> bool:
        return self.component is not None

    def label(self) -> str:
        if self.component is not None:
            return self.component
        assert self.config_path is not None
        return Path(self.config_path).stem

    def display_label(self) -> str:
        base_label = self.label()
        if self.baseline_match == "none":
            return base_label
        return f"{base_label} [{self.baseline_match}:{self.baseline_window_bins}]"


@dataclass
class ShowGroup:
    target: str
    subtracts: list[SubtractSpec] = field(default_factory=list)


def _parse_subtract_tail(
    raw_values: list[str],
    *,
    option_name: str,
) -> tuple[float, str, int]:
    tail = list(raw_values)
    scale = 1.0
    baseline_match = "none"
    baseline_window_bins = DEFAULT_BASELINE_WINDOW_BINS

    if tail:
        try:
            scale = float(tail[0])
        except ValueError:
            pass
        else:
            tail = tail[1:]

    if tail:
        baseline_match = str(tail[0]).strip().lower()
        tail = tail[1:]
    if tail:
        try:
            baseline_window_bins = int(tail[0])
        except ValueError as exc:
            raise argparse.ArgumentError(
                None,
                f"{option_name} baseline window must be an integer number of bins; got {tail[0]!r}",
            ) from exc
        tail = tail[1:]
    if tail:
        raise argparse.ArgumentError(
            None,
            f"{option_name} expects TARGET [SCALE] [BASELINE_MATCH] [BASELINE_WINDOW_BINS]",
        )

    if baseline_match not in BASELINE_MATCH_MODES:
        raise argparse.ArgumentError(
            None,
            f"{option_name} baseline match must be one of {BASELINE_MATCH_MODES}; got {baseline_match!r}",
        )
    if baseline_window_bins < 3:
        raise argparse.ArgumentError(None, f"{option_name} baseline window must be at least 3 bins")
    if baseline_window_bins % 2 == 0:
        raise argparse.ArgumentError(None, f"{option_name} baseline window must be odd")

    return scale, baseline_match, baseline_window_bins


class _AppendShowAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        operations = list(getattr(namespace, self.dest, []) or [])
        operations.append(("show", str(values)))
        setattr(namespace, self.dest, operations)


class _AppendSubtractAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if not isinstance(values, list):
            values = [values]
        component = str(values[0])
        scale, baseline_match, baseline_window_bins = _parse_subtract_tail(
            [str(value) for value in values[1:]],
            option_name="--subtract",
        )
        operations = list(getattr(namespace, self.dest, []) or [])
        operations.append(("subtract", component, scale, baseline_match, baseline_window_bins))
        setattr(namespace, self.dest, operations)


class _AppendSubtractConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if not isinstance(values, list):
            values = [values]
        config_path = str(values[0])
        scale, baseline_match, baseline_window_bins = _parse_subtract_tail(
            [str(value) for value in values[1:]],
            option_name="--subtract-config",
        )
        operations = list(getattr(namespace, self.dest, []) or [])
        operations.append(("subtract_config", config_path, scale, baseline_match, baseline_window_bins))
        setattr(namespace, self.dest, operations)


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average config-selected X/Y/A spectra and show ordered subtraction panels.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    add_track_data_root_arg(parser)
    add_bond_spacing_mode_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_plot_scale_args(parser)
    add_signal_processing_args(parser)
    add_bond_filter_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)
    add_spectrasave_arg(parser)

    parser.add_argument("--freq-min-hz", type=float, default=None)
    parser.add_argument("--freq-max-hz", type=float, default=None)

    welch_group = parser.add_mutually_exclusive_group()
    welch_group.add_argument(
        "--welch",
        dest="welch",
        action="store_true",
        help="Average Welch spectra (default).",
    )
    welch_group.add_argument(
        "--no-welch",
        dest="welch",
        action="store_false",
        help="Average FFT spectra instead of Welch spectra.",
    )
    parser.set_defaults(welch=True)
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument(
        "--compare-xya",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Use logical x/y/a resolution from ordered config 'contains'. Default: true.",
    )
    parser.add_argument(
        "--allow-duplicate-bonds",
        action="store_true",
        help="Allow the same configured bond id to contribute multiple times.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Render each panel as the final-result frequency image using the existing repo image path.",
    )
    parser.add_argument(
        "--clipto",
        type=float,
        default=None,
        help="Display-only lower clip applied to all plotted values.",
    )
    parser.add_argument(
        "--showonlyresult",
        action="store_true",
        help="Only plot the final subtraction result trace in each non-image panel.",
    )
    parser.add_argument(
        "--spectrasave-mode",
        choices=["raw", "display"],
        default="display",
        help="When --spectrasave is used, save the raw subtraction result or the display-processed trace. Default: display",
    )
    parser.add_argument(
        "--savitzky",
        action="store_true",
        help="Apply Savitzky-Golay smoothing to plotted spectra only.",
    )
    parser.add_argument(
        "--gaussianblur",
        action="store_true",
        help="Apply Gaussian smoothing to plotted spectra only. Stackable with other smoothing options.",
    )
    parser.add_argument(
        "--gaussian-sigma-bins",
        type=float,
        default=1.5,
        help="Gaussian sigma in frequency bins for --gaussianblur. Default: 1.5",
    )
    parser.add_argument(
        "--gaussian-truncate",
        type=float,
        default=4.0,
        help="Gaussian kernel truncate radius in sigmas for --gaussianblur. Default: 4.0",
    )
    parser.add_argument(
        "--medianblur",
        action="store_true",
        help="Apply median filtering to plotted spectra only. Stackable with other smoothing options.",
    )
    parser.add_argument(
        "--median-window-bins",
        type=int,
        default=5,
        help="Median-filter window length in bins for --medianblur. Must be odd. Default: 5",
    )
    parser.add_argument(
        "--savitzky-window-bins",
        type=int,
        default=11,
        help="Savitzky-Golay window length in bins. Must be odd. Default: 11",
    )
    parser.add_argument(
        "--savitzky-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order. Default: 2",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show(); useful for headless runs with --save.",
    )
    parser.add_argument(
        "--show",
        dest="operations",
        action=_AppendShowAction,
        choices=CANONICAL_COMPONENTS,
        help="Start a new panel from this logical component. Repeatable.",
    )
    parser.add_argument(
        "--subtract",
        dest="operations",
        action=_AppendSubtractAction,
        nargs="+",
        metavar="ARG",
        help=(
            "Subtract COMPONENT from the current --show target as "
            "--subtract COMPONENT [SCALE] [BASELINE_MATCH] [BASELINE_WINDOW_BINS]. "
            "BASELINE_MATCH choices: none, median-offset."
        ),
    )
    parser.add_argument(
        "--subtract-config",
        "--subtract_config",
        dest="operations",
        action=_AppendSubtractConfigAction,
        nargs="+",
        metavar="ARG",
        help=(
            "Subtract the average spectrum from a plain config-selected bond set as "
            "--subtract-config CONFIG_JSON [SCALE] [BASELINE_MATCH] [BASELINE_WINDOW_BINS]. "
            "BASELINE_MATCH choices: none, median-offset."
        ),
    )
    return parser


def _format_bond_list(display_bonds: list[int]) -> str:
    if len(display_bonds) == 0:
        return "[]"
    if len(display_bonds) <= 12:
        return "[" + ", ".join(str(v) for v in display_bonds) + "]"
    head = ", ".join(str(v) for v in display_bonds[:10])
    return f"[{head}, ...] ({len(display_bonds)} total)"


def _logical_to_physical_suffix(contains: list[str]) -> dict[str, str]:
    return {
        logical_component: CANONICAL_COMPONENTS[idx]
        for idx, logical_component in enumerate(contains)
    }


def _component_overlap(raw_config: OrderedDict[str, dict]) -> list[str]:
    included = [entry for entry in raw_config.values() if entry["include"]]
    if len(included) == 0:
        return []
    shared = set(included[0]["contains"])
    for entry in included[1:]:
        shared &= set(entry["contains"])
    return [component for component in CANONICAL_COMPONENTS if component in shared]


def _resolved_component_config(
    raw_config: OrderedDict[str, dict],
    *,
    logical_component: str,
) -> OrderedDict[str, DatasetSelection]:
    resolved: OrderedDict[str, DatasetSelection] = OrderedDict()
    for dataset_name, entry in raw_config.items():
        if entry["include"] and entry["contains"] is not None:
            physical_suffix = _logical_to_physical_suffix(entry["contains"])[logical_component]
            resolved_name = f"{dataset_name}_{physical_suffix}"
        else:
            resolved_name = dataset_name
        resolved[resolved_name] = DatasetSelection(
            include=bool(entry["include"]),
            discards=list(entry["discards"]),
            pair_ids=list(entry["pair_ids"]),
        )
    return resolved


def _validate_compare_xya_inputs(
    raw_config: OrderedDict[str, dict],
    *,
    track_data_root: str | None,
) -> list[str]:
    included = [(dataset_name, entry) for dataset_name, entry in raw_config.items() if entry["include"]]
    if len(included) == 0:
        raise ValueError("No included datasets remain in the config")

    for dataset_name, entry in included:
        if entry["contains"] is None:
            raise ValueError(
                f"--compare-xya requires every included dataset to declare ordered 'contains'; dataset '{dataset_name}' does not"
            )
        suffix_map = _logical_to_physical_suffix(entry["contains"])
        for logical_component, physical_suffix in suffix_map.items():
            resolved_name = f"{dataset_name}_{physical_suffix}"
            track2_path = default_track2_path(resolved_name, track_data_root=track_data_root)
            if not track2_path.exists():
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' declares logical component '{logical_component}' via physical suffix "
                    f"'{physical_suffix}', but {track2_path} does not exist"
                )

    overlap = _component_overlap(raw_config)
    if len(overlap) == 0:
        raise ValueError("No overlapping logical components exist across the included datasets")
    return overlap


def _resolve_show_groups(operations: list[tuple]) -> list[ShowGroup]:
    if len(operations) == 0:
        raise ValueError("Provide at least one --show component")

    groups: list[ShowGroup] = []
    current: ShowGroup | None = None
    for operation in operations:
        kind = operation[0]
        if kind == "show":
            current = ShowGroup(target=str(operation[1]))
            groups.append(current)
            continue
        if kind == "subtract":
            if current is None:
                raise ValueError("--subtract must appear after a --show")
            current.subtracts.append(
                SubtractSpec(
                    component=str(operation[1]),
                    scale=float(operation[2]),
                    baseline_match=str(operation[3]),
                    baseline_window_bins=int(operation[4]),
                )
            )
            continue
        if kind == "subtract_config":
            if current is None:
                raise ValueError("--subtract-config must appear after a --show")
            current.subtracts.append(
                SubtractSpec(
                    config_path=str(operation[1]),
                    scale=float(operation[2]),
                    baseline_match=str(operation[3]),
                    baseline_window_bins=int(operation[4]),
                )
            )
            continue
        raise ValueError(f"Unsupported operation kind: {kind}")

    return groups


def _required_components(groups: list[ShowGroup]) -> list[str]:
    required = []
    seen = set()
    for group in groups:
        for component in [group.target] + [item.component for item in group.subtracts if item.component is not None]:
            if component not in seen:
                required.append(component)
                seen.add(component)
    return required


def _build_plain_config_selection(config_json: str | Path) -> OrderedDict[str, DatasetSelection]:
    raw_config = load_dataset_selection_entries(config_json)
    resolved: OrderedDict[str, DatasetSelection] = OrderedDict()
    included = False

    for dataset_name, entry in raw_config.items():
        include = bool(entry["include"])
        if include:
            included = True
        if include and entry["contains"] is not None:
            raise ValueError(
                f"--subtract-config requires a plain config without logical components; "
                f"dataset '{dataset_name}' in {config_json} declares 'contains'"
            )
        resolved[dataset_name] = DatasetSelection(
            include=include,
            discards=list(entry["discards"]),
            pair_ids=list(entry["pair_ids"]),
        )

    if not included:
        raise ValueError(f"--subtract-config file has no included datasets: {config_json}")
    return resolved


def _compute_average_result_from_records(args, records: list, *, label: str) -> AverageSpectrumResult:
    contributions = (
        compute_welch_contributions(
            records,
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )
        if args.welch
        else compute_fft_contributions(
            records,
            longest=args.longest,
            handlenan=args.handlenan,
            timeseriesnorm=args.timeseriesnorm,
        )
    )
    if len(contributions) == 0:
        raise ValueError(f"No spectra were accepted from the selected bond contributors for {label}")

    return compute_average_spectrum(
        contributions,
        normalize_mode=args.normalize,
        relative_range=tuple(args.relative_range),
        average_domain=args.average_domain,
        lowest_freq=args.freq_min_hz,
        highest_freq=args.freq_max_hz,
    )


def _compute_component_results(args, groups: list[ShowGroup]):
    if not args.compare_xya:
        raise ValueError("This script requires --compare-xya true because --show/--subtract use logical x/y/a components")

    raw_config = load_dataset_selection_entries(args.config_json)
    overlap = _validate_compare_xya_inputs(raw_config, track_data_root=args.track_data_root)
    required = _required_components(groups)
    missing = [component for component in required if component not in overlap]
    if missing:
        raise ValueError(f"Config does not expose overlapping logical components needed for plotting: {missing}")

    results_by_component = OrderedDict()
    available_display_bonds = None
    selected_display_bonds = None

    for logical_component in required:
        config = _resolved_component_config(raw_config, logical_component=logical_component)
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=args.allow_duplicate_bonds,
            bond_spacing_mode=args.bond_spacing_mode,
        )
        component_available = collect_display_bond_numbers(records)
        records = filter_signal_records_by_display_bonds(
            records,
            only_bonds=args.only_bonds,
            exclude_bonds=args.exclude_bonds,
            parity=args.bond_parity,
        )
        component_selected = collect_display_bond_numbers(records)
        if len(records) == 0:
            raise ValueError(
                f"Bond selection removed all configured bond contributors for component '{logical_component}'"
            )

        contributions = (
            compute_welch_contributions(
                records,
                welch_len_s=args.welch_len_s,
                welch_overlap_fraction=args.welch_overlap,
                longest=args.longest,
                handlenan=args.handlenan,
                timeseriesnorm=args.timeseriesnorm,
            )
            if args.welch
            else compute_fft_contributions(
                records,
                longest=args.longest,
                handlenan=args.handlenan,
                timeseriesnorm=args.timeseriesnorm,
            )
        )
        if len(contributions) == 0:
            raise ValueError(
                f"No spectra were accepted from the selected bond contributors for component '{logical_component}'"
            )

        result = compute_average_spectrum(
            contributions,
            normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            average_domain=args.average_domain,
            lowest_freq=args.freq_min_hz,
            highest_freq=args.freq_max_hz,
        )
        results_by_component[logical_component] = result

        if available_display_bonds is None:
            available_display_bonds = component_available
        if selected_display_bonds is None:
            selected_display_bonds = component_selected

    return raw_config, overlap, results_by_component, available_display_bonds or [], selected_display_bonds or []


def _compute_external_subtraction_results(args, groups: list[ShowGroup]) -> tuple[OrderedDict[str, AverageSpectrumResult], dict[str, dict[str, list[int]]]]:
    required_paths: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group.subtracts:
            if item.config_path is None:
                continue
            resolved_path = str(Path(item.config_path).resolve())
            if resolved_path in seen:
                continue
            required_paths.append(resolved_path)
            seen.add(resolved_path)

    results_by_path: OrderedDict[str, AverageSpectrumResult] = OrderedDict()
    bond_info_by_path: dict[str, dict[str, list[int]]] = {}
    for config_path in required_paths:
        config = _build_plain_config_selection(config_path)
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=args.allow_duplicate_bonds,
            bond_spacing_mode=args.bond_spacing_mode,
        )
        available_display_bonds = collect_display_bond_numbers(records)
        records = filter_signal_records_by_display_bonds(
            records,
            only_bonds=args.only_bonds,
            exclude_bonds=args.exclude_bonds,
            parity=args.bond_parity,
        )
        selected_display_bonds = collect_display_bond_numbers(records)
        if len(records) == 0:
            raise ValueError(
                f"Bond selection removed all configured bond contributors for subtraction config '{config_path}'"
            )

        results_by_path[config_path] = _compute_average_result_from_records(
            args,
            records,
            label=f"subtraction config '{config_path}'",
        )
        bond_info_by_path[config_path] = {
            "available": available_display_bonds,
            "selected": selected_display_bonds,
        }

    return results_by_path, bond_info_by_path


def _build_common_frequency_grid(results_by_component: OrderedDict[str, object]) -> np.ndarray:
    results = list(results_by_component.values())
    overlap_low = max(float(result.freq_grid[0]) for result in results)
    overlap_high = min(float(result.freq_grid[-1]) for result in results)
    if overlap_high <= overlap_low:
        raise ValueError("Component spectra do not share an overlapping frequency window")

    lead_grid = np.asarray(results[0].freq_grid, dtype=float)
    mask = (lead_grid >= overlap_low) & (lead_grid <= overlap_high)
    common = lead_grid[mask]
    if common.size < 2:
        raise ValueError("Shared component frequency grid has fewer than 2 points")
    return common


def _build_common_frequency_grid_from_results(results: list[AverageSpectrumResult]) -> np.ndarray:
    overlap_low = max(float(result.freq_grid[0]) for result in results)
    overlap_high = min(float(result.freq_grid[-1]) for result in results)
    if overlap_high <= overlap_low:
        raise ValueError("Component spectra do not share an overlapping frequency window")

    lead_grid = np.asarray(results[0].freq_grid, dtype=float)
    mask = (lead_grid >= overlap_low) & (lead_grid <= overlap_high)
    common = lead_grid[mask]
    if common.size < 2:
        raise ValueError("Shared component frequency grid has fewer than 2 points")
    return common


def _interpolate_component_amplitudes(
    results_by_component: OrderedDict[str, object],
    *,
    common_freq: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        component: np.interp(common_freq, result.freq_grid, result.avg_amp)
        for component, result in results_by_component.items()
    }


def _small_positive_floor(arrays: list[np.ndarray]) -> float:
    positives = []
    for values in arrays:
        arr = np.asarray(values, dtype=float)
        mask = np.isfinite(arr) & (arr > 0)
        if np.any(mask):
            positives.append(float(np.min(arr[mask])))
    if positives:
        return max(min(positives), np.finfo(float).eps)
    return np.finfo(float).eps


def _compute_median_baseline(values: np.ndarray, *, window_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Median baseline expects a 1D spectrum")
    if arr.size < 3:
        raise ValueError("Median baseline requires at least 3 spectrum bins")

    kernel = min(int(window_bins), int(arr.size))
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 3:
        raise ValueError(
            f"Median baseline window ({window_bins}) is incompatible with spectrum length ({arr.size})"
        )

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Median baseline cannot be computed from an all-NaN spectrum")

    work = np.asarray(arr, dtype=float).copy()
    fill_value = float(np.median(finite))
    work[~np.isfinite(work)] = fill_value
    return np.asarray(medfilt(work, kernel_size=kernel), dtype=float)


def _align_subtract_source_to_target(
    target_amp: np.ndarray,
    source_amp: np.ndarray,
    *,
    spec: SubtractSpec,
) -> np.ndarray:
    target_arr = np.asarray(target_amp, dtype=float)
    source_arr = np.asarray(source_amp, dtype=float)
    if spec.baseline_match == "none":
        return source_arr
    if spec.baseline_match != "median-offset":
        raise ValueError(f"Unsupported baseline match mode: {spec.baseline_match}")

    target_baseline = _compute_median_baseline(target_arr, window_bins=spec.baseline_window_bins)
    source_baseline = _compute_median_baseline(source_arr, window_bins=spec.baseline_window_bins)
    return source_arr + (target_baseline - source_baseline)


def _display_safe(values: np.ndarray, *, floor: float) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float).copy()
    invalid = (~np.isfinite(arr)) | (arr < floor)
    arr[invalid] = floor
    return arr, int(np.count_nonzero(invalid))


def _expr_label(group: ShowGroup) -> str:
    pieces = [group.target]
    for item in group.subtracts:
        source_label = item.display_label()
        if np.isclose(item.scale, 1.0):
            pieces.append(source_label)
        else:
            pieces.append(f"{item.scale:.6g}*{source_label}")
    if len(pieces) == 1:
        return pieces[0]
    return f"{pieces[0]} - " + " - ".join(pieces[1:])


def _maybe_smooth(values: np.ndarray, args) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Display smoothing expects a 1D spectrum")

    if args.medianblur:
        window = int(args.median_window_bins)
        if window < 3:
            raise ValueError("--median-window-bins must be at least 3")
        if window % 2 == 0:
            raise ValueError("--median-window-bins must be odd")
        kernel = min(window, int(arr.size))
        if kernel % 2 == 0:
            kernel -= 1
        if kernel < 3:
            raise ValueError(
                f"--median-window-bins ({window}) is incompatible with plotted spectrum length ({arr.size})"
            )
        arr = np.asarray(medfilt(arr, kernel_size=kernel), dtype=float)

    if args.gaussianblur:
        sigma = float(args.gaussian_sigma_bins)
        truncate = float(args.gaussian_truncate)
        if sigma <= 0:
            raise ValueError("--gaussian-sigma-bins must be > 0")
        if truncate <= 0:
            raise ValueError("--gaussian-truncate must be > 0")
        arr = np.asarray(
            gaussian_filter1d(arr, sigma=sigma, mode="nearest", truncate=truncate),
            dtype=float,
        )

    if args.savitzky:
        window = int(args.savitzky_window_bins)
        polyorder = int(args.savitzky_polyorder)
        if window < 3:
            raise ValueError("--savitzky-window-bins must be at least 3")
        if window % 2 == 0:
            raise ValueError("--savitzky-window-bins must be odd")
        if polyorder < 0:
            raise ValueError("--savitzky-polyorder must be >= 0")
        if polyorder >= window:
            raise ValueError("--savitzky-polyorder must be smaller than --savitzky-window-bins")
        if arr.size < window:
            raise ValueError(
                f"--savitzky-window-bins ({window}) exceeds the plotted spectrum length ({arr.size})"
            )
        arr = np.asarray(savgol_filter(arr, window_length=window, polyorder=polyorder, mode="interp"), dtype=float)

    return arr


def _compute_panel_outputs(
    groups: list[ShowGroup],
    *,
    amp_by_component: dict[str, np.ndarray],
    external_amp_by_path: dict[str, np.ndarray],
    args,
    clipto: float | None,
) -> list[dict[str, object]]:
    panels: list[dict[str, object]] = []
    for idx, group in enumerate(groups, start=1):
        target_raw = np.asarray(amp_by_component[group.target], dtype=float)
        subtract_terms_raw: list[tuple[SubtractSpec, np.ndarray]] = []
        for item in group.subtracts:
            if item.component is not None:
                source_raw = np.asarray(amp_by_component[item.component], dtype=float)
            else:
                assert item.config_path is not None
                source_raw = np.asarray(external_amp_by_path[str(Path(item.config_path).resolve())], dtype=float)
            aligned_source_raw = _align_subtract_source_to_target(target_raw, source_raw, spec=item)
            subtract_terms_raw.append((item, aligned_source_raw))

        result_raw = target_raw.copy()
        for item, source_raw in subtract_terms_raw:
            result_raw = result_raw - float(item.scale) * source_raw

        target_display = _maybe_smooth(target_raw, args)
        subtract_terms_display = [
            (item, _maybe_smooth(source_raw, args))
            for item, source_raw in subtract_terms_raw
        ]
        result_display = _maybe_smooth(result_raw, args)

        display_floor = _small_positive_floor(
            [target_display] + [float(item.scale) * source for item, source in subtract_terms_display] + [result_display]
        )
        if clipto is not None:
            display_floor = float(clipto)
        if display_floor <= 0:
            raise ValueError("--clipto must be > 0")

        target_plot, target_clipped = _display_safe(target_display, floor=display_floor)
        source_plots = []
        total_source_clipped = 0
        for item, source_display in subtract_terms_display:
            scaled_source = float(item.scale) * source_display
            plot_source, clipped = _display_safe(scaled_source, floor=display_floor)
            source_plots.append((item, plot_source))
            total_source_clipped += clipped
        result_plot, result_clipped = _display_safe(result_display, floor=display_floor)

        panels.append(
            {
                "index": idx,
                "group": group,
                "expr_label": _expr_label(group),
                "target_plot": target_plot,
                "source_plots": source_plots,
                "result_raw": result_raw,
                "result_plot": result_plot,
                "display_floor": display_floor,
                "clipped_total": target_clipped + total_source_clipped + result_clipped,
            }
        )

    return panels


def _plot_groups(
    groups: list[ShowGroup],
    *,
    common_freq: np.ndarray,
    results_by_component: OrderedDict[str, AverageSpectrumResult],
    amp_by_component: dict[str, np.ndarray],
    external_amp_by_path: dict[str, np.ndarray],
    panel_outputs: list[dict[str, object]],
    args,
    plot_scale: str,
    title: str,
    cmap_index: int,
    full_image: bool,
    clipto: float | None,
):
    external_colors = ["#7f7f7f", "#8c564b", "#17becf", "#bcbd22", "#9467bd", "#ff7f0e"]
    n_rows = len(groups)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(13, max(4.0, 3.8 * n_rows)),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, panel in zip(axes, panel_outputs):
        idx = int(panel["index"])
        group = panel["group"]
        assert isinstance(group, ShowGroup)
        target_plot = np.asarray(panel["target_plot"], dtype=float)
        source_plots = panel["source_plots"]
        result_plot = np.asarray(panel["result_plot"], dtype=float)
        display_floor = float(panel["display_floor"])
        clipped_total = int(panel["clipped_total"])
        expr_label = str(panel["expr_label"])

        if clipped_total > 0:
            warnings.warn(
                f"Panel {idx} ({expr_label}) required display clipping for {clipped_total} plotted bins; "
                f"display floor={display_floor:.3e}"
            )

        if full_image:
            lead_result = next(iter(results_by_component.values()))
            image_result = AverageSpectrumResult(
                freq_grid=common_freq,
                avg_amp=result_plot,
                norm_low=float(lead_result.norm_low),
                norm_high=float(lead_result.norm_high),
                freq_low=float(common_freq[0]),
                freq_high=float(common_freq[-1]),
                contributors=[],
            )
            _plot_frequency_image(
                fig,
                ax,
                freq=image_result.freq_grid,
                amp=image_result.avg_amp,
                plot_scale=plot_scale,
                cmap_index=cmap_index,
                y_min=float(image_result.freq_grid[0]),
                y_max=float(image_result.freq_grid[-1]),
                x_label="Arbitrary X",
                x_max=1.0,
                title=f"Panel {idx}: {expr_label}",
                linear_color_label="Normalized Amplitude",
                log_color_label="Amplitude (dB)",
            )
        elif plot_scale == "log":
            if not args.showonlyresult:
                ax.semilogy(common_freq, target_plot, linewidth=1.6, color=COMPONENT_COLORS[group.target], label=group.target)
                for item, plot_source in source_plots:
                    if item.component is not None:
                        color = COMPONENT_COLORS[item.component]
                    else:
                        color = external_colors[abs(hash(item.label())) % len(external_colors)]
                    source_label = (
                        item.display_label()
                        if np.isclose(item.scale, 1.0)
                        else f"{item.scale:.6g}*{item.display_label()}"
                    )
                    ax.semilogy(common_freq, plot_source, linewidth=1.2, linestyle="--", color=color, label=source_label)
            ax.semilogy(common_freq, result_plot, linewidth=2.2, color="black", label=expr_label)
        else:
            if not args.showonlyresult:
                ax.plot(common_freq, target_plot, linewidth=1.6, color=COMPONENT_COLORS[group.target], label=group.target)
                for item, plot_source in source_plots:
                    if item.component is not None:
                        color = COMPONENT_COLORS[item.component]
                    else:
                        color = external_colors[abs(hash(item.label())) % len(external_colors)]
                    source_label = (
                        item.display_label()
                        if np.isclose(item.scale, 1.0)
                        else f"{item.scale:.6g}*{item.display_label()}"
                    )
                    ax.plot(common_freq, plot_source, linewidth=1.2, linestyle="--", color=color, label=source_label)
            ax.plot(common_freq, result_plot, linewidth=2.2, color="black", label=expr_label)
            ax.axhline(display_floor, color="0.65", linewidth=0.8, alpha=0.8)

        if not full_image:
            ax.set_ylabel("Normalized Amplitude")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            ax.set_title(f"Panel {idx}: {expr_label}")

    if not full_image:
        axes[-1].set_xlabel("Frequency (Hz)")
        axes[-1].set_xlim(float(common_freq[0]), float(common_freq[-1]))
    fig.suptitle(title)
    return fig


def _render_figure(fig, *, save: str | None, show: bool) -> None:
    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _save_panel_spectrasaves(
    args,
    *,
    config_json: str,
    common_freq: np.ndarray,
    panel_outputs: list[dict[str, object]],
) -> None:
    if args.spectrasave is None:
        return

    config_stem = Path(config_json).stem
    for panel in panel_outputs:
        idx = int(panel["index"])
        expr_label = str(panel["expr_label"])
        group = panel["group"]
        assert isinstance(group, ShowGroup)

        if args.spectrasave_mode == "display":
            amplitude = np.asarray(panel["result_plot"], dtype=float)
        else:
            amplitude = np.asarray(panel["result_raw"], dtype=float)

        export_path = resolve_spectrasave_path(
            args.spectrasave,
            default_name=build_default_spectrasave_name(
                config_stem,
                "subtract-view",
                f"panel-{idx}",
                args.spectrasave_mode,
            ),
            multi_suffix=f"panel-{idx}",
        )
        assert export_path is not None

        metadata = {
            "sourceKind": "subtract-view",
            "configPath": config_json,
            "panelIndex": idx,
            "panelTarget": group.target,
            "panelExpression": expr_label,
            "spectraSaveMode": args.spectrasave_mode,
            "plotScale": args.plot_scale,
            "showOnlyResult": bool(args.showonlyresult),
            "fullImage": bool(args.full_image),
            "clipTo": None if args.clipto is None else float(args.clipto),
            "displayFloor": float(panel["display_floor"]),
            "baselineSubtracts": [
                {
                    "label": item.display_label(),
                    "scale": float(item.scale),
                    "baselineMatch": item.baseline_match,
                    "baselineWindowBins": int(item.baseline_window_bins),
                }
                for item in group.subtracts
            ],
            "displaySmoothing": {
                "medianBlur": bool(args.medianblur),
                "medianWindowBins": int(args.median_window_bins),
                "gaussianBlur": bool(args.gaussianblur),
                "gaussianSigmaBins": float(args.gaussian_sigma_bins),
                "gaussianTruncate": float(args.gaussian_truncate),
                "savitzky": bool(args.savitzky),
                "savitzkyWindowBins": int(args.savitzky_window_bins),
                "savitzkyPolyorder": int(args.savitzky_polyorder),
            },
        }

        saved = save_spectrum_msgpack(
            export_path,
            freq=common_freq,
            amplitude=amplitude,
            label=expr_label,
            metadata=metadata,
        )
        print(f"Spectrum saved to: {saved}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)

    if args.freq_min_hz is not None and args.freq_max_hz is not None and args.freq_max_hz <= args.freq_min_hz:
        print("Error: --freq-max-hz must be greater than --freq-min-hz", file=sys.stderr)
        return 1

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        groups = _resolve_show_groups(list(args.operations or []))
        _, overlap, results_by_component, available_display_bonds, selected_display_bonds = _compute_component_results(
            args,
            groups,
        )
        external_results_by_path, external_bond_info_by_path = _compute_external_subtraction_results(args, groups)
        all_results = list(results_by_component.values()) + list(external_results_by_path.values())
        common_freq = _build_common_frequency_grid_from_results(all_results)
        amp_by_component = _interpolate_component_amplitudes(results_by_component, common_freq=common_freq)
        external_amp_by_path = {
            config_path: np.interp(common_freq, result.freq_grid, result.avg_amp)
            for config_path, result in external_results_by_path.items()
        }
        panel_outputs = _compute_panel_outputs(
            groups,
            amp_by_component=amp_by_component,
            external_amp_by_path=external_amp_by_path,
            args=args,
            clipto=args.clipto,
        )

        accepted_display_bonds = sorted({
            int(contrib.record.entity_id) + 1
            for result in results_by_component.values()
            for contrib in result.contributors
        })
        total_contributors = sum(len(result.contributors) for result in results_by_component.values())
        n_datasets = len({
            contrib.record.dataset_name
            for result in results_by_component.values()
            for contrib in result.contributors
        })

        norm_desc = args.normalize
        if args.normalize == "relative":
            norm_desc = f"relative [{args.relative_range[0]}, {args.relative_range[1]}] Hz"

        panel_desc = " | ".join(_expr_label(group) for group in groups)
        title = args.title or (
            f"Config {'Welch' if args.welch else 'FFT'} subtraction view | "
            f"components={','.join(_required_components(groups))} | panels={len(groups)} | norm={norm_desc}"
        )

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds)}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds)}")
        for config_path, bond_info in external_bond_info_by_path.items():
            print(
                f"Subtract config {Path(config_path).name} available display bonds: "
                f"{_format_bond_list(bond_info['available'])}"
            )
            print(
                f"Subtract config {Path(config_path).name} selected display bonds: "
                f"{_format_bond_list(bond_info['selected'])}"
            )
        print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
        print(f"Requested component overlap: {overlap}")
        print(f"Plotted panels: {panel_desc}")
        print(f"Total accepted contributors: {total_contributors}")
        print(f"Unique datasets: {n_datasets}")
        print(f"Common frequency window: [{common_freq[0]:.6f}, {common_freq[-1]:.6f}] Hz")
        print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
        if args.welch:
            print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
        smoothing_steps: list[str] = []
        if args.medianblur:
            smoothing_steps.append(f"median window={args.median_window_bins}")
        if args.gaussianblur:
            smoothing_steps.append(
                f"gaussian sigma={args.gaussian_sigma_bins:.6g} truncate={args.gaussian_truncate:.6g}"
            )
        if args.savitzky:
            smoothing_steps.append(
                f"savgol window={args.savitzky_window_bins} polyorder={args.savitzky_polyorder}"
            )
        if smoothing_steps:
            print(f"Display smoothing pipeline: {' -> '.join(smoothing_steps)}")
        if args.spectrasave is not None:
            print(f"SpectraSave export mode: {args.spectrasave_mode}")
        print("Normalization band processing: linear detrend -> zero-floor -> integrate area")
        print(f"Near-zero denominator threshold: {ABSOLUTE_ZERO_TOL:.0e}")

        _save_panel_spectrasaves(
            args,
            config_json=args.config_json,
            common_freq=common_freq,
            panel_outputs=panel_outputs,
        )

        fig = _plot_groups(
            groups,
            common_freq=common_freq,
            results_by_component=results_by_component,
            amp_by_component=amp_by_component,
            external_amp_by_path=external_amp_by_path,
            panel_outputs=panel_outputs,
            args=args,
            plot_scale=args.plot_scale,
            title=title,
            cmap_index=args.cm,
            full_image=args.full_image,
            clipto=args.clipto,
        )
        if args.no_show:
            _render_figure(fig, save=args.save, show=False)
        else:
            render_figure(fig, save=args.save)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
