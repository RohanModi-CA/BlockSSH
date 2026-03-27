#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from plotting.common import ensure_parent_dir, render_figure
from plotting.frequency import _apply_compact_image_axis_style, _plot_frequency_image
from tools.cli import (
    add_average_domain_args,
    add_bond_filter_args,
    add_colormap_arg,
    add_normalization_args,
    add_output_args,
    add_plot_scale_args,
    resolve_normalization_mode,
    add_signal_processing_args,
    add_track_data_root_arg,
)
from tools.io import default_track2_path
from tools.models import DatasetSelection
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
DEFAULT_CLEAN_STEPS = ("x<-a", "y<-a", "x<-clean_y")
DEFAULT_TARGET_HZ_BY_COMPONENT = {
    "x": 4.8,
    "y": 1.85,
    "a": 4.8,
}


def _logical_to_physical_suffix(contains: list[str]) -> dict[str, str]:
    return {
        logical_component: CANONICAL_COMPONENTS[idx]
        for idx, logical_component in enumerate(contains)
    }


def _resolved_component_config(
    raw_config: OrderedDict[str, dict],
    *,
    logical_component: str,
) -> OrderedDict[str, DatasetSelection]:
    resolved: OrderedDict[str, DatasetSelection] = OrderedDict()
    for dataset_name, entry in raw_config.items():
        if not entry["include"]:
            resolved_name = dataset_name
        else:
            contains = entry["contains"]
            if contains is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not declare ordered 'contains'; cleantest requires component-aware configs"
                )
            if logical_component not in contains:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not provide logical component '{logical_component}'"
                )
            physical_suffix = _logical_to_physical_suffix(contains)[logical_component]
            resolved_name = f"{dataset_name}_{physical_suffix}"

        resolved[resolved_name] = DatasetSelection(
            include=bool(entry["include"]),
            discards=list(entry["discards"]),
            pair_ids=list(entry["pair_ids"]),
        )
    return resolved


def _validate_component_inputs(
    raw_config: OrderedDict[str, dict],
    *,
    track_data_root: str | None,
    required_components: tuple[str, ...],
) -> None:
    included = [(dataset_name, entry) for dataset_name, entry in raw_config.items() if entry["include"]]
    if len(included) == 0:
        raise ValueError("No included datasets remain in the config")

    for dataset_name, entry in included:
        contains = entry["contains"]
        if contains is None:
            raise ValueError(
                f"Dataset '{dataset_name}' does not declare ordered 'contains'; cleantest requires component-aware configs"
            )

        suffix_map = _logical_to_physical_suffix(contains)
        for logical_component in required_components:
            if logical_component not in contains:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not provide required logical component '{logical_component}'"
                )
            resolved_name = f"{dataset_name}_{suffix_map[logical_component]}"
            track2_path = default_track2_path(resolved_name, track_data_root=track_data_root)
            if not track2_path.exists():
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' declares logical component '{logical_component}', but {track2_path} does not exist"
                )


def _compute_contributions(args, records):
    return (
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


def _nearest_bin(freq: np.ndarray, target_hz: float) -> int:
    freq = np.asarray(freq, dtype=float)
    if freq.size == 0:
        raise ValueError("Frequency grid is empty")
    return int(np.argmin(np.abs(freq - float(target_hz))))


def _compute_local_baseline(amp: np.ndarray, window_bins: int) -> np.ndarray:
    amp = np.asarray(amp, dtype=float)
    if amp.size == 0:
        return amp.copy()

    kernel = max(3, int(window_bins))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > amp.size:
        kernel = amp.size if amp.size % 2 == 1 else max(1, amp.size - 1)
    if kernel < 3:
        finite = amp[np.isfinite(amp)]
        floor = np.nanpercentile(finite, 10.0) if finite.size > 0 else 0.0
        return np.full_like(amp, float(floor))

    baseline = medfilt(amp, kernel_size=kernel)
    finite = baseline[np.isfinite(baseline)]
    if finite.size == 0:
        return np.zeros_like(amp)
    floor = np.nanpercentile(finite, 10.0)
    return np.maximum(baseline, floor)


def _compute_global_baseline(amp: np.ndarray) -> float:
    amp = np.asarray(amp, dtype=float)
    finite = amp[np.isfinite(amp)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, 10.0))


def _apply_display_floor(
    clean_amp: np.ndarray,
    reference_amp: np.ndarray,
    *,
    mode: str,
    baseline_window_bins: int,
) -> np.ndarray:
    clean_amp = np.asarray(clean_amp, dtype=float)
    reference_amp = np.asarray(reference_amp, dtype=float)

    if mode == "none":
        return clean_amp
    if mode == "zero":
        return np.maximum(clean_amp, 0.0)
    if mode == "global-baseline":
        floor = _compute_global_baseline(reference_amp)
        return np.maximum(clean_amp, floor)
    if mode == "local-baseline":
        baseline = _compute_local_baseline(reference_amp, baseline_window_bins)
        return np.maximum(clean_amp, baseline)
    raise ValueError(f"Unsupported display floor mode: {mode}")


def _plot_side_by_side_images(
    freq: np.ndarray,
    payloads: list[tuple[str, np.ndarray]],
    *,
    cmap_index: int,
    plot_scale: str,
    title: str,
    marker_hz: list[float],
):
    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(4.1 * len(payloads), 5.0),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        bottom=0.09,
        top=0.92,
        wspace=0.02,
    )

    lead_ax = axes[0]
    for ax in axes[1:]:
        ax.sharex(lead_ax)
        ax.sharey(lead_ax)

    for idx, (label, amp) in enumerate(payloads):
        _plot_frequency_image(
            fig,
            axes[idx],
            freq=np.asarray(freq, dtype=float),
            amp=np.asarray(amp, dtype=float),
            plot_scale=plot_scale,
            cmap_index=cmap_index,
            y_min=float(freq[0]),
            y_max=float(freq[-1]),
            title=label,
            linear_color_label="Normalized Amplitude",
            log_color_label="Amplitude (dB)",
            show_colorbar=False,
            annotate_range=True,
        )
        for target_hz in marker_hz:
            axes[idx].axhline(target_hz, color="white", linestyle="--", linewidth=0.9, alpha=0.65)
        _apply_compact_image_axis_style(
            axes[idx],
            show_right_ylabel=idx == len(payloads) - 1,
        )

    fig.suptitle(title, fontsize=14)
    return fig


def _format_state_label(component: str, removed_components: list[str]) -> str:
    if len(removed_components) == 0:
        return component
    suffix = " and ".join(f"no {removed}" for removed in removed_components)
    return f"{component} {suffix}"


def _display_state_amp(
    *,
    component: str,
    amp: np.ndarray,
    removed_components: list[str],
    raw_amp_by_component: dict[str, np.ndarray],
    args,
) -> np.ndarray:
    amp = np.asarray(amp, dtype=float)
    if len(removed_components) == 0:
        return amp
    return _apply_display_floor(
        amp,
        raw_amp_by_component[component],
        mode=args.display_floor,
        baseline_window_bins=args.baseline_window_bins,
    )


def _plot_clean_step_curves(
    freq: np.ndarray,
    step_payloads: list[dict[str, object]],
    *,
    plot_scale: str,
    title: str,
):
    nrows = len(step_payloads)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(13.0, 3.3 * nrows),
        squeeze=False,
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.92,
        hspace=0.36,
        wspace=0.18,
    )

    for row_idx, payload in enumerate(step_payloads):
        row_axes = axes[row_idx]
        step_total = int(payload["step_total"])
        row_axes[0].text(
            -0.34,
            0.5,
            f"Step {payload['step_index']}/{step_total}",
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

        panels = [
            ("source", payload["source_label"], payload["source_amp"]),
            ("before", payload["before_label"], payload["before_amp"]),
            ("after", payload["after_label"], payload["after_amp"]),
        ]
        selected_hz = float(payload["selected_hz"])

        for col_idx, (panel_kind, panel_label, panel_amp) in enumerate(panels):
            ax = row_axes[col_idx]
            amp = np.asarray(panel_amp, dtype=float)
            if plot_scale == "log":
                positive_amp = np.where(amp > 0, amp, np.nan)
                ax.semilogy(freq, positive_amp, linewidth=1.25)
            else:
                ax.plot(freq, amp, linewidth=1.25)
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)

            ax.axvline(selected_hz, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
            ax.set_title(f"{panel_label} {panel_kind}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(freq[0], freq[-1])
            if col_idx == 0:
                ax.set_ylabel("Normalized Amplitude")
            if row_idx == nrows - 1:
                ax.set_xlabel("Frequency (Hz)")

    fig.suptitle(title, fontsize=14)
    return fig


def _plot_clean_step_images(
    freq: np.ndarray,
    step_payloads: list[dict[str, object]],
    *,
    cmap_index: int,
    plot_scale: str,
    title: str,
):
    nrows = len(step_payloads)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(12.3, 3.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.92,
        hspace=0.26,
        wspace=0.05,
    )

    lead_ax = axes[0, 0]
    for ax in axes.ravel()[1:]:
        ax.sharex(lead_ax)
        ax.sharey(lead_ax)

    for row_idx, payload in enumerate(step_payloads):
        row_axes = axes[row_idx]
        step_total = int(payload["step_total"])
        row_axes[0].text(
            -0.34,
            0.5,
            f"Step {payload['step_index']}/{step_total}",
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

        panels = [
            ("source", payload["source_label"], payload["source_amp"]),
            ("before", payload["before_label"], payload["before_amp"]),
            ("after", payload["after_label"], payload["after_amp"]),
        ]
        selected_hz = float(payload["selected_hz"])

        for col_idx, (panel_kind, panel_label, panel_amp) in enumerate(panels):
            ax = row_axes[col_idx]
            _plot_frequency_image(
                fig,
                ax,
                freq=np.asarray(freq, dtype=float),
                amp=np.asarray(panel_amp, dtype=float),
                plot_scale=plot_scale,
                cmap_index=cmap_index,
                y_min=float(freq[0]),
                y_max=float(freq[-1]),
                title=f"{panel_label} {panel_kind}",
                linear_color_label="Normalized Amplitude",
                log_color_label="Amplitude (dB)",
                show_colorbar=False,
                annotate_range=True,
            )
            ax.axhline(selected_hz, color="white", linestyle="--", linewidth=0.9, alpha=0.65)
            _apply_compact_image_axis_style(
                ax,
                show_right_ylabel=col_idx == 2,
            )

    fig.suptitle(title, fontsize=14)
    return fig


def _save_path_for_step(save_path: str | None, step_index: int, total_steps: int) -> str | None:
    if save_path is None:
        return None
    if total_steps <= 1:
        return save_path
    path = Path(save_path)
    return str(path.with_name(f"{path.stem}.step{step_index:02d}{path.suffix}"))


def _average_component_result(args, raw_config: OrderedDict[str, dict], logical_component: str):
    config = _resolved_component_config(raw_config, logical_component=logical_component)
    records = build_configured_bond_signals(
        config,
        track_data_root=args.track_data_root,
        allow_duplicate_ids=args.allow_duplicate_bonds,
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
            f"Bond selection removed all configured bond contributors for component '{logical_component}'"
        )

    contributions = _compute_contributions(args, records)
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
    return result, available_display_bonds, selected_display_bonds


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_clean_step(value: str) -> tuple[str, str]:
    raw = str(value).strip().replace(" ", "")
    if "<-" not in raw:
        raise argparse.ArgumentTypeError(
            f"Invalid --clean step {value!r}; expected TARGET<-REMOVE, e.g. x<-a or x<-clean_y"
        )
    target, remove = raw.split("<-", 1)
    valid_remove = set(CANONICAL_COMPONENTS) | {f"clean_{component}" for component in CANONICAL_COMPONENTS}
    if target not in CANONICAL_COMPONENTS:
        raise argparse.ArgumentTypeError(
            f"Invalid clean target {target!r}; expected one of {CANONICAL_COMPONENTS}"
        )
    if remove not in valid_remove:
        raise argparse.ArgumentTypeError(
            f"Invalid clean source {remove!r}; expected one of {sorted(valid_remove)}"
        )
    return target, remove


def _resolved_clean_steps(values: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
    if values:
        return list(values)
    return [_parse_clean_step(step) for step in DEFAULT_CLEAN_STEPS]


def _resolved_show_steps(values: list[int] | None, *, total_steps: int) -> list[int]:
    if values is None:
        return list(range(1, total_steps + 1))

    resolved: list[int] = []
    seen: set[int] = set()
    for raw_step in values:
        step_index = int(raw_step)
        if step_index < 1 or step_index > total_steps:
            raise ValueError(
                f"--show-steps entry {step_index} is out of range; expected values between 1 and {total_steps}"
            )
        if step_index not in seen:
            resolved.append(step_index)
            seen.add(step_index)
    return resolved


def _source_component(remove_name: str) -> str:
    return remove_name[6:] if remove_name.startswith("clean_") else remove_name


def _prompt_select_step(step_payloads: list[dict[str, object]]) -> dict[str, object]:
    print("Select cleantest step to export:")
    for payload in step_payloads:
        print(
            f"  {int(payload['step_index'])}: "
            f"{payload['target']} <- {payload['remove']} "
            f"@ {float(payload['selected_hz']):.6g} Hz"
        )

    valid = {int(payload["step_index"]): payload for payload in step_payloads}
    while True:
        raw = input("Step number: ").strip()
        try:
            step_index = int(raw)
        except ValueError:
            print("Enter a valid integer step number.")
            continue
        if step_index in valid:
            return valid[step_index]
        print(f"Choose one of: {sorted(valid)}")


def _prompt_select_panel(payload: dict[str, object]) -> tuple[str, str, np.ndarray]:
    panel_map = {
        "source": (str(payload["source_label"]), np.asarray(payload["source_amp"], dtype=float)),
        "before": (str(payload["before_label"]), np.asarray(payload["before_amp"], dtype=float)),
        "after": (str(payload["after_label"]), np.asarray(payload["after_amp"], dtype=float)),
    }
    print("Select panel to export:")
    for panel_kind, (label, _) in panel_map.items():
        print(f"  {panel_kind}: {label}")

    while True:
        raw = input("Panel [source/before/after]: ").strip().lower()
        if raw in panel_map:
            label, amp = panel_map[raw]
            return raw, label, amp
        print("Choose one of: source, before, after")


def _resolve_target_hz(args, source_name: str) -> float:
    source_component = _source_component(source_name)
    specific_value = getattr(args, f"target_hz_{source_component}")
    if specific_value is not None:
        return float(specific_value)
    if args.target_hz is not None:
        return float(args.target_hz)
    return float(DEFAULT_TARGET_HZ_BY_COMPONENT[source_component])


def _run_clean_pipeline(
    *,
    freq: np.ndarray,
    raw_amp_by_component: dict[str, np.ndarray],
    clean_steps: list[tuple[str, str]],
    power_mode: bool,
    args,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, float | int | str]],
    list[dict[str, object]],
]:
    current_amp_by_component = {
        component: np.asarray(amp, dtype=float).copy()
        for component, amp in raw_amp_by_component.items()
    }
    derived_amp_by_name: dict[str, np.ndarray] = {}
    summaries: list[dict[str, float | int | str]] = []
    step_payloads: list[dict[str, object]] = []
    removed_components_by_component = {
        component: []
        for component in raw_amp_by_component
    }

    for step_index, (target, remove_name) in enumerate(clean_steps, start=1):
        if target not in current_amp_by_component:
            raise ValueError(f"Clean step {step_index} targets unavailable component '{target}'")

        if remove_name.startswith("clean_"):
            if remove_name not in derived_amp_by_name:
                raise ValueError(
                    f"Clean step {step_index} requires '{remove_name}', but it has not been produced yet"
                )
            remove_amp = derived_amp_by_name[remove_name]
            source_removed_components = list(removed_components_by_component[_source_component(remove_name)])
        else:
            if remove_name not in raw_amp_by_component:
                raise ValueError(f"Clean step {step_index} removes unavailable component '{remove_name}'")
            remove_amp = raw_amp_by_component[remove_name]
            source_removed_components = []

        target_hz = _resolve_target_hz(args, remove_name)
        idx = _nearest_bin(freq, target_hz)
        selected_hz = float(freq[idx])

        target_amp = np.asarray(current_amp_by_component[target], dtype=float)
        target_removed_components_before = list(removed_components_by_component[target])
        remove_amp = np.asarray(remove_amp, dtype=float)
        if power_mode:
            target_spec = np.maximum(target_amp, 0.0) ** 2
            remove_spec = np.maximum(remove_amp, 0.0) ** 2
        else:
            target_spec = target_amp
            remove_spec = remove_amp

        ref_remove = float(remove_spec[idx])
        ref_target = float(target_spec[idx])
        if not np.isfinite(ref_remove) or ref_remove <= 0:
            raise ValueError(
                f"Step {step_index} source '{remove_name}' reference value at {selected_hz:.6f} Hz "
                f"was non-positive; cannot form a scale factor"
            )

        scale = ref_target / ref_remove
        cleaned_target_spec = target_spec - scale * remove_spec
        if power_mode:
            cleaned_target_amp = np.sqrt(np.maximum(cleaned_target_spec, 0.0))
        else:
            cleaned_target_amp = cleaned_target_spec

        target_removed_components_after = list(target_removed_components_before)
        source_component = _source_component(remove_name)
        if source_component not in target_removed_components_after:
            target_removed_components_after.append(source_component)

        current_amp_by_component[target] = cleaned_target_amp
        derived_amp_by_name[f"clean_{target}"] = cleaned_target_amp.copy()
        removed_components_by_component[target] = target_removed_components_after
        summaries.append(
            {
                "step_index": int(step_index),
                "target": target,
                "remove": remove_name,
                "requested_hz": float(target_hz),
                "selected_hz": float(selected_hz),
                "bin_index": int(idx),
                "reference_target": float(ref_target),
                "reference_remove": float(ref_remove),
                "scale": float(scale),
            }
        )
        step_payloads.append(
            {
                "step_index": int(step_index),
                "target": target,
                "remove": remove_name,
                "selected_hz": float(selected_hz),
                "source_label": _format_state_label(_source_component(remove_name), source_removed_components),
                "before_label": _format_state_label(target, target_removed_components_before),
                "after_label": _format_state_label(target, target_removed_components_after),
                "source_amp": _display_state_amp(
                    component=_source_component(remove_name),
                    amp=remove_amp,
                    removed_components=source_removed_components,
                    raw_amp_by_component=raw_amp_by_component,
                    args=args,
                ),
                "before_amp": _display_state_amp(
                    component=target,
                    amp=target_amp,
                    removed_components=target_removed_components_before,
                    raw_amp_by_component=raw_amp_by_component,
                    args=args,
                ),
                "after_amp": _display_state_amp(
                    component=target,
                    amp=cleaned_target_amp,
                    removed_components=target_removed_components_after,
                    raw_amp_by_component=raw_amp_by_component,
                    args=args,
                ),
            }
        )

    return current_amp_by_component, derived_amp_by_name, summaries, step_payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply ordered scalar cleaning steps to averaged normalized spectra.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    add_track_data_root_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_plot_scale_args(parser)
    add_signal_processing_args(parser)
    add_bond_filter_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument("--freq-min-hz", type=float, default=None)
    parser.add_argument("--freq-max-hz", type=float, default=None)
    parser.add_argument("--welch", action="store_true", default=True, help="Use Welch spectra instead of FFT.")
    parser.add_argument("--welch-len-s", type=float, default=100.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--allow-duplicate-bonds", action="store_true")
    parser.add_argument("--full-image", action="store_true", help="Show the enabled spectra as side-by-side frequency images.")
    parser.add_argument(
        "--one-fig",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Keep all cleaning rows in one figure. Pass false to open one figure per row. Default: true",
    )
    parser.add_argument(
        "--clean",
        dest="clean_steps",
        action="append",
        type=_parse_clean_step,
        default=None,
        metavar="TARGET<-REMOVE",
        help=(
            "Ordered cleaning step. May be repeated. Defaults to: "
            "x<-a, y<-a, x<-clean_y"
        ),
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=None,
        help="Global reference frequency override for all cleaning steps. Default: source-specific peaks.",
    )
    parser.add_argument(
        "--target-hz-x",
        type=float,
        default=None,
        help="Reference frequency override when the removed source is x. Default: 4.8",
    )
    parser.add_argument(
        "--target-hz-y",
        type=float,
        default=None,
        help="Reference frequency override when the removed source is y or clean_y. Default: 1.85",
    )
    parser.add_argument(
        "--target-hz-a",
        type=float,
        default=None,
        help="Reference frequency override when the removed source is a or clean_a. Default: 4.8",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Estimate and subtract in power space, then convert back to amplitude for plotting.",
    )
    parser.add_argument(
        "--show-steps",
        type=int,
        nargs="+",
        default=None,
        metavar="STEP",
        help="Only plot these 1-based cleaning steps. All steps are still computed.",
    )
    parser.add_argument(
        "--display-floor",
        choices=["local-baseline", "global-baseline", "none", "zero"],
        default="local-baseline",
        help="How to floor the cleaned spectrum for display only. Default: local-baseline",
    )
    parser.add_argument(
        "--baseline-window-bins",
        type=int,
        default=101,
        help="Odd median-filter window in bins for the local-baseline display floor. Default: 101",
    )
    parser.add_argument(
        "--show-x",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the original X spectrum. Default: true",
    )
    parser.add_argument(
        "--show-a",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the original A spectrum. Default: true",
    )
    parser.add_argument(
        "--show-y",
        "--show-scaled-a",
        dest="show_y",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the original Y spectrum. Default: true",
    )
    parser.add_argument(
        "--show-cleaned-x",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the final cleaned X spectrum. Default: true",
    )
    parser.add_argument(
        "--show-cleaned-y",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Show the final cleaned Y spectrum. Default: false",
    )
    add_spectrasave_arg(parser)
    return parser


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
        config_stem = Path(args.config_json).stem
        clean_steps = _resolved_clean_steps(args.clean_steps)

        raw_config = load_dataset_selection_entries(args.config_json)
        required_components = {
            target
            for target, _ in clean_steps
        } | {
            _source_component(remove_name)
            for _, remove_name in clean_steps
        }
        if args.show_x:
            required_components.add("x")
        if args.show_a:
            required_components.add("a")
        if args.show_y or args.show_cleaned_y:
            required_components.add("y")
        _validate_component_inputs(
            raw_config,
            track_data_root=args.track_data_root,
            required_components=tuple(sorted(required_components)),
        )

        component_results: dict[str, object] = {}
        available_display_bonds: list[int] | None = None
        selected_display_bonds: list[int] | None = None
        for component in sorted(required_components):
            result, available_bonds, selected_bonds = _average_component_result(args, raw_config, component)
            component_results[component] = result
            if available_display_bonds is None:
                available_display_bonds = available_bonds
                selected_display_bonds = selected_bonds

        reference_component = clean_steps[0][0]
        result_ref = component_results[reference_component]
        freq = result_ref.freq_grid
        for component, result in component_results.items():
            if not np.allclose(freq, result.freq_grid, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Average spectra for '{reference_component}' and '{component}' do not share the same frequency grid"
                )

        raw_amp_by_component = {
            component: np.asarray(component_results[component].avg_amp, dtype=float)
            for component in required_components
        }
        current_amp_by_component, derived_amp_by_name, clean_summaries, step_payloads = _run_clean_pipeline(
            freq=freq,
            raw_amp_by_component=raw_amp_by_component,
            clean_steps=clean_steps,
            power_mode=bool(args.power),
            args=args,
        )
        total_steps = len(step_payloads)
        for payload in step_payloads:
            payload["step_total"] = int(total_steps)
        show_steps = _resolved_show_steps(args.show_steps, total_steps=total_steps)
        visible_step_numbers = set(show_steps)
        visible_step_payloads = [
            payload for payload in step_payloads
            if int(payload["step_index"]) in visible_step_numbers
        ]

        accepted_display_bonds = sorted({
            int(contrib.record.entity_id) + 1
            for contrib in result_ref.contributors
        })
        n_datasets = len({contrib.record.dataset_name for contrib in result_ref.contributors})
        norm_desc = args.normalize
        if args.normalize == "relative":
            norm_desc = f"relative [{rel_low}, {rel_high}] Hz"
        pipeline_desc = ", ".join(f"{target}<-{remove_name}" for target, remove_name in clean_steps)

        title = args.title or (
            f"Clean Test | {'Welch' if args.welch else 'FFT'} | pipeline={pipeline_desc} | "
            f"mode={'power' if args.power else 'amplitude'} | datasets={n_datasets} | "
            f"bonds={len(accepted_display_bonds)} | norm={norm_desc}"
        )

        print(f"Available configured display bonds: {_format_bond_list(available_display_bonds)}")
        print(f"Selected display bonds: {_format_bond_list(selected_display_bonds)}")
        print(f"Accepted display bonds: {_format_bond_list(accepted_display_bonds)}")
        print(f"Unique datasets: {n_datasets}")
        print(f"Frequency window: [{result_ref.freq_low:.6f}, {result_ref.freq_high:.6f}] Hz")
        print(f"Normalization window: [{result_ref.norm_low:.6f}, {result_ref.norm_high:.6f}] Hz")
        print("Normalization band processing: linear detrend -> zero-floor -> integrate area")
        print(f"Near-zero denominator threshold: {ABSOLUTE_ZERO_TOL:.0e}")
        print(f"Spectrum type: {'Welch' if args.welch else 'FFT'}")
        if args.welch:
            print(f"Welch parameters: len={args.welch_len_s:.6g}s overlap={args.welch_overlap:.6g}")
        print(f"Cleaning pipeline: {pipeline_desc}")
        if args.target_hz is not None:
            print(f"Global target frequency override: {args.target_hz:.6f} Hz")
        print(
            "Default source target frequencies: "
            + ", ".join(f"{component}={hz:.6f} Hz" for component, hz in DEFAULT_TARGET_HZ_BY_COMPONENT.items())
        )
        for summary in clean_summaries:
            print(
                f"Step {summary['step_index']}: {summary['target']} <- {summary['remove']} | "
                f"requested={summary['requested_hz']:.6f} Hz | used={summary['selected_hz']:.6f} Hz "
                f"(bin {summary['bin_index']}) | target_ref={summary['reference_target']:.12g} | "
                f"source_ref={summary['reference_remove']:.12g} | scale={summary['scale']:.12g}"
            )
        print(f"Plotted steps: {show_steps}")
        print(f"Subtraction mode: {'power' if args.power else 'amplitude'}")
        print(f"Display floor mode: {args.display_floor}")

        if args.spectrasave is not None:
            selected_payload = _prompt_select_step(step_payloads)
            panel_kind, panel_label, panel_amp = _prompt_select_panel(selected_payload)
            default_name = build_default_spectrasave_name(
                config_stem,
                "cleantest",
                f"step-{int(selected_payload['step_index']):02d}",
                panel_kind,
                f"target-{selected_payload['target']}",
                f"remove-{selected_payload['remove']}",
            )
            export_path = resolve_spectrasave_path(
                args.spectrasave,
                default_name=default_name,
            )
            assert export_path is not None
            saved = save_spectrum_msgpack(
                export_path,
                freq=freq,
                amplitude=panel_amp,
                label=f"{config_stem} cleantest step {int(selected_payload['step_index'])} {panel_kind}",
                metadata={
                    "sourceKind": "cleantest",
                    "spectrumKind": "welch" if args.welch else "fft",
                    "configPath": args.config_json,
                    "cleanSteps": [f"{target}<-{remove_name}" for target, remove_name in clean_steps],
                    "selectedStep": int(selected_payload["step_index"]),
                    "selectedPanel": panel_kind,
                    "panelLabel": panel_label,
                    "stepTarget": str(selected_payload["target"]),
                    "stepRemove": str(selected_payload["remove"]),
                    "stepSelectedHz": float(selected_payload["selected_hz"]),
                    "normalize": args.normalize,
                    "relativeRange": list(map(float, args.relative_range)),
                    "averageDomain": args.average_domain,
                    "freqLowHz": float(result_ref.freq_low),
                    "freqHighHz": float(result_ref.freq_high),
                    "normLowHz": float(result_ref.norm_low),
                    "normHighHz": float(result_ref.norm_high),
                    "datasets": sorted({contrib.record.dataset_name for contrib in result_ref.contributors}),
                    "bondIds": accepted_display_bonds,
                    "powerMode": bool(args.power),
                },
            )
            print(f"Spectrum saved to: {saved}")

        if args.one_fig:
            if args.full_image:
                fig = _plot_clean_step_images(
                    freq,
                    visible_step_payloads,
                    cmap_index=args.cm,
                    plot_scale=args.plot_scale,
                    title=title,
                )
            else:
                fig = _plot_clean_step_curves(
                    freq,
                    visible_step_payloads,
                    plot_scale=args.plot_scale,
                    title=title,
                )

            render_figure(fig, save=args.save)
        else:
            figures_to_show = []
            for payload in visible_step_payloads:
                step_title = (
                    f"{title} | Step {payload['step_index']}/{total_steps}: "
                    f"{payload['target']}<-{payload['remove']}"
                )
                if args.full_image:
                    fig = _plot_clean_step_images(
                        freq,
                        [payload],
                        cmap_index=args.cm,
                        plot_scale=args.plot_scale,
                        title=step_title,
                    )
                else:
                    fig = _plot_clean_step_curves(
                        freq,
                        [payload],
                        plot_scale=args.plot_scale,
                        title=step_title,
                    )
                step_save_path = _save_path_for_step(args.save, int(payload["step_index"]), total_steps)
                if step_save_path is not None:
                    save_path = ensure_parent_dir(step_save_path)
                    fig.savefig(save_path, dpi=300)
                    print(f"Plot saved to: {save_path}")
                figures_to_show.append(fig)
            plt.show()
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _format_bond_list(display_bonds: list[int]) -> str:
    if len(display_bonds) == 0:
        return "[]"
    if len(display_bonds) <= 12:
        return "[" + ", ".join(str(v) for v in display_bonds) + "]"
    head = ", ".join(str(v) for v in display_bonds[:10])
    return f"[{head}, ...] ({len(display_bonds)} total)"


if __name__ == "__main__":
    raise SystemExit(main())
