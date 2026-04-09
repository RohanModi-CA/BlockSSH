#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import collections
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from plotting.common import apply_major_tick_spacing, render_figure, resolve_clipped_window
from plotting.frequency import _link_fft_frequency_to_image_frequency, _plot_frequency_image, _plot_fft_curve_panel
from tools.cli import (
    add_colormap_arg,
    add_flattening_args,
    add_frequency_window_args,
    add_normalization_args,
    add_output_args,
    resolve_normalization_mode,
    add_tickspace_arg,
    validate_frequency_window_args,
    validate_tickspace_arg,
)
from tools.flattening import FlatteningResult, apply_global_baseline_processing_to_results, plot_flattening_diagnostic
from tools.models import AverageSpectrumResult, FFTResult, ProcessedSignal, SignalRecord, SpectrumContribution
from tools.peaks import load_peaks_csv
from tools.spectrasave import add_spectrasave_arg, build_default_spectrasave_name, get_default_spectrasave_dir, load_spectrum_msgpack, resolve_existing_spectrasave_path, resolve_spectrasave_path, save_spectrum_msgpack
from tools.spectral import build_common_grid, choose_frequency_window, interp_amplitude, normalize_spectrum, resolve_normalization_window


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _default_peaks_csv_path(spectrasave_path: Path) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "peaks" / f"{spectrasave_path.stem}.csv"


def _load_preload_peaks(csv_path: Path, *, require_exists: bool) -> list[float]:
    if not csv_path.exists():
        if require_exists:
            raise FileNotFoundError(f"Peaks CSV not found: {csv_path}")
        return []
    peaks = [float(peak) for peak in load_peaks_csv(csv_path) if np.isfinite(peak) and peak > 0]
    return sorted(set(peaks))


def _validate_flatten_args(args) -> str | None:
    flat_low, flat_high = map(float, args.flatten_reference_band)
    if flat_high <= flat_low:
        return "Error: --flatten-reference-band STOP_HZ must be greater than START_HZ"
    return None


def _validate_relative_range(args) -> str | None:
    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        return "Error: --relative-range STOP_HZ must be greater than START_HZ"
    return None


def _spectrasave_key(path: Path, spectrum: dict) -> str:
    metadata = spectrum.get("metadata") or {}
    component = metadata.get("component")
    if component is not None and str(component).strip():
        return str(component).strip()
    label = spectrum.get("label")
    if label is not None and str(label).strip():
        return str(label).strip()
    return path.stem


def _index_to_role(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"S{index + 1}"


def _compare_label(mode: str) -> str:
    labels = {
        "shared-geom": "Shared Spectrum (Geometric Mean)",
        "shared-min": "Shared Spectrum (Pointwise Min)",
        "a-only-diff": "A-Only Spectrum (Positive Difference)",
        "a-only-ratio": "A-Only Enrichment Score (Ratio)",
    }
    return labels[mode]


def _compare_label_with_params(mode: str, *, shared_geom_power: float) -> str:
    if mode == "shared-geom":
        if np.isclose(shared_geom_power, 0.5):
            return _compare_label(mode)
        return f"Shared Spectrum ((A*B)^{shared_geom_power:g})"
    return _compare_label(mode)


def _compare_y_label(mode: str) -> str:
    return "Normalized Value" if mode == "a-only-ratio" else "Normalized Amplitude"


def _resolve_compare_flatten_enabled(args) -> bool:
    if args.compare is not None:
        return not args.no_flatten
    return bool(args.flatten)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display one or more SpectraSave spectra as curves with optional image panels.",
    )
    parser.add_argument("spectrasave_inputs", nargs="+", metavar="spectrasave", help="Path to one or more SpectraSave msgpack artifacts.")
    parser.add_argument(
        "--use-subplots",
        action="store_true",
        help="When multiple spectra are provided, plot them in separate subplots instead of overlaying them.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional custom labels for the legend/subplots. Must match the number of spectrasave files.",
    )
    parser.add_argument(
        "--peaks-csv",
        default=None,
        help="Optional peaks CSV to preload. Defaults to configs/peaks/<spectrasave-stem>.csv when preload is enabled.",
    )
    parser.add_argument(
        "--no-preload-csv",
        dest="preload_csv",
        action="store_false",
        help="Disable automatic preloading of a corresponding peaks CSV.",
    )
    parser.set_defaults(preload_csv=True)
    add_flattening_args(parser)
    for action in parser._actions:
        if "--baseline-match" in getattr(action, "option_strings", ()):
            action.nargs = "?"
            action.const = "A"
            action.metavar = "TARGET"
            action.help = (
                "Baseline-match target by input role. Pass no value to target A, or pass A/B/C/... explicitly."
            )
            break
    parser.set_defaults(flatten=None)
    parser.add_argument(
        "--no-flatten",
        dest="no_flatten",
        action="store_true",
        help="Disable flattening in comparison mode, where flattening is otherwise enabled by default.",
    )
    parser.set_defaults(no_flatten=False)
    add_normalization_args(parser)
    parser.set_defaults(normalize="relative", relative_range=(3.0, 28.0))
    add_spectrasave_arg(parser, help_text="Save the derived comparison spectrum as msgpack. Pass no value to use an auto-generated path under spectrasave/.")
    parser.add_argument(
        "--no-peaks",
        dest="show_peaks",
        action="store_false",
        help="Disable drawing peak markers entirely.",
    )
    parser.set_defaults(show_peaks=True)
    parser.add_argument(
        "--no-lines",
        dest="show_lines",
        action="store_false",
        help="Disable connecting lines between preloaded peak markers.",
    )
    parser.set_defaults(show_lines=True)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    fft_group = parser.add_mutually_exclusive_group()
    fft_group.add_argument(
        "--fft-log",
        dest="fft_log",
        action="store_true",
        help="Use a log y-axis for the FFT curve panel (default).",
    )
    fft_group.add_argument(
        "--fft-linear",
        dest="fft_log",
        action="store_false",
        help="Use a linear y-axis for the FFT curve panel.",
    )
    parser.set_defaults(fft_log=True)

    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-log",
        dest="image_plot_scale",
        action="store_const",
        const="log",
        help="Use dB display for the full-image panel (default).",
    )
    image_group.add_argument(
        "--image-linear",
        dest="image_plot_scale",
        action="store_const",
        const="linear",
        help="Use linear amplitude display for the full-image panel.",
    )
    parser.set_defaults(image_plot_scale="log")
    parser.add_argument(
        "--no-image",
        dest="show_image",
        action="store_false",
        help="Hide the right-hand full-image panel and show only the FFT curve.",
    )
    parser.set_defaults(show_image=True)

    add_frequency_window_args(parser, help_scope="displayed frequency range for both panels")
    add_tickspace_arg(parser)
    parser.add_argument(
        "--image-cols",
        type=int,
        default=64,
        help="Nominal width of the repeated-image panel in arbitrary x units. Default: 64",
    )
    parser.add_argument(
        "--full-couple",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Couple FFT x-zoom and image y-zoom. Pass --full-couple false to disable.",
    )
    parser.add_argument(
        "--show-metadata",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Print artifact label and metadata summary to stdout. Pass --show-metadata false to disable.",
    )
    parser.add_argument(
        "--offset",
        nargs=2,
        type=float,
        metavar=("K", "O"),
        help="Apply offset O to the Y-axis amplitude of the K-th spectrum (1-indexed). Only affects that single spectrum.",
    )
    parser.add_argument(
        "--scale",
        nargs=2,
        type=float,
        metavar=("K", "S"),
        help="Multiply the Y-axis amplitude of the K-th spectrum by S (1-indexed). Only affects that single spectrum.",
    )
    parser.add_argument(
        "--savitzky",
        action="store_true",
        help="Apply Savitzky-Golay smoothing to all spectra.",
    )
    parser.add_argument(
        "--gaussianblur",
        action="store_true",
        help="Apply Gaussian smoothing to all spectra. Stackable with --savitzky.",
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
        "--compare",
        choices=["shared-geom", "shared-min", "a-only-diff", "a-only-ratio"],
        default=None,
        help="Compare exactly two spectra and derive a common or A-only result.",
    )
    parser.add_argument(
        "--switch",
        action="store_true",
        help="Swap logical A/B roles before comparison.",
    )
    parser.add_argument(
        "--only-result",
        action="store_true",
        help="In comparison mode, display only the derived result.",
    )
    parser.add_argument(
        "--compare-subplots",
        action="store_true",
        help="In comparison mode, show A, B, and result as aligned subplots instead of a single overlay axes.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Small positive stabilizer used by ratio-based comparison modes. Default: 1e-12",
    )
    parser.add_argument(
        "--shared-geom-power",
        type=float,
        default=0.5,
        help="Exponent applied to the A*B product in --compare shared-geom. Default: 0.5",
    )
    parser.add_argument(
        "--interp-kind",
        default="cubic",
        choices=["linear", "quadratic", "cubic"],
        help="Interpolation kind for the common frequency grid in comparison mode. Default: cubic",
    )
    parser.add_argument(
        "--coarsest",
        action="store_true",
        help="Use the coarsest common comparison grid instead of the finest (default).",
    )
    parser.add_argument(
        "--normalize-last",
        action="store_true",
        help="In comparison mode, apply normalization after flattening/baseline matching instead of before it.",
    )
    return parser


def _compute_freq_bounds(freq: np.ndarray, requested_min: float | None, requested_max: float | None) -> tuple[float, float]:
    finite_freq = np.asarray(freq, dtype=float)
    finite_freq = finite_freq[np.isfinite(finite_freq)]
    if finite_freq.size == 0:
        raise ValueError("Spectrum frequency axis is empty or non-finite")

    return resolve_clipped_window(
        float(np.min(finite_freq)),
        float(np.max(finite_freq)),
        requested_min,
        requested_max,
    )


def _maybe_smooth(amp: np.ndarray, args) -> np.ndarray:
    arr = np.asarray(amp, dtype=float)
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
                f"--savitzky-window-bins ({window}) exceeds the spectrum length ({arr.size})"
            )
        arr = np.asarray(savgol_filter(arr, window_length=window, polyorder=polyorder, mode="interp"), dtype=float)
    return arr


def _print_metadata(path: Path, spectrum: dict) -> None:
    print(f"SpectraSave: {path}")
    print(f"Label: {spectrum.get('label') or '<none>'}")

    metadata = spectrum.get("metadata") or {}
    if not metadata:
        print("Metadata: <none>")
        return

    print("Metadata:")
    for key in sorted(metadata):
        print(f"  {key}: {metadata[key]}")


def _make_spectrum_contribution(path: Path, spectrum: dict) -> SpectrumContribution:
    freq = np.asarray(spectrum["freq"], dtype=float)
    amp = np.asarray(spectrum["amplitude"], dtype=float)
    t_dummy = np.array([0.0, 1.0], dtype=float)
    y_dummy = np.array([0.0, 0.0], dtype=float)
    return SpectrumContribution(
        record=SignalRecord(
            dataset_name=path.stem,
            entity_id=0,
            local_index=0,
            label=spectrum.get("label") or path.stem,
            signal_kind="bond",
            source_path=str(path),
            t=t_dummy,
            y=y_dummy,
        ),
        processed=ProcessedSignal(
            t=t_dummy,
            y=y_dummy,
            dt=1.0,
            Fs=1.0,
            nyquist=0.5,
            proc_msg="spectrasave",
        ),
        fft_result=FFTResult(freq=freq, amplitude=amp),
    )


def _normalize_compare_spectrum(
    freq: np.ndarray,
    amp: np.ndarray,
    *,
    normalize_mode: str,
    relative_range: tuple[float, float],
) -> tuple[np.ndarray, float, float]:
    freq_low = float(np.min(freq))
    freq_high = float(np.max(freq))
    norm_low, norm_high = resolve_normalization_window(
        freq_low,
        freq_high,
        normalize_mode=normalize_mode,
        relative_range=relative_range,
    )
    normalized = normalize_spectrum(freq, amp, norm_low=norm_low, norm_high=norm_high)
    if normalized is None:
        raise ValueError(
            f"Comparison normalization denominator in [{norm_low:.6g}, {norm_high:.6g}] Hz was too small"
        )
    return np.asarray(normalized, dtype=float), float(norm_low), float(norm_high)


def _build_average_result(freq: np.ndarray, amp: np.ndarray) -> AverageSpectrumResult:
    freq_arr = np.asarray(freq, dtype=float)
    amp_arr = np.asarray(amp, dtype=float)
    finite_freq = freq_arr[np.isfinite(freq_arr)]
    if finite_freq.size == 0:
        raise ValueError("Spectrum frequency axis is empty or non-finite")
    return AverageSpectrumResult(
        freq_grid=freq_arr,
        avg_amp=amp_arr,
        norm_low=float(np.min(finite_freq)),
        norm_high=float(np.max(finite_freq)),
        freq_low=float(np.min(finite_freq)),
        freq_high=float(np.max(finite_freq)),
        contributors=[],
    )


def _build_spectrum_keys(spectra: list[dict], paths: list[Path]) -> list[str]:
    return [_index_to_role(index) for index, _ in enumerate(spectra)]


def _maybe_apply_baseline_processing(
    spectra: list[dict],
    paths: list[Path],
    args,
) -> tuple[list[dict], list[str], dict[str, FlatteningResult]]:
    return _apply_baseline_processing_with_keys(
        spectra,
        keys_by_index=_build_spectrum_keys(spectra, paths),
        args=args,
    )


def _apply_baseline_processing_with_keys(
    spectra: list[dict],
    *,
    keys_by_index: list[str],
    args,
) -> tuple[list[dict], list[str], dict[str, FlatteningResult]]:
    if not args.flatten and getattr(args, "baseline_match", None) is None:
        return spectra, list(keys_by_index), {}

    ordered: collections.OrderedDict[str, AverageSpectrumResult] = collections.OrderedDict()
    for key, spectrum in zip(keys_by_index, spectra, strict=True):
        ordered[key] = _build_average_result(spectrum["freq"], spectrum["amplitude"])

    processed, flattening_by_key = apply_global_baseline_processing_to_results(
        ordered,
        flatten=args.flatten,
        baseline_match=getattr(args, "baseline_match", None),
        reference_band=tuple(float(value) for value in args.flatten_reference_band),
    )

    processed_spectra: list[dict] = []
    for key, spectrum in zip(keys_by_index, spectra, strict=True):
        updated = dict(spectrum)
        updated["amplitude"] = np.asarray(processed[key].avg_amp, dtype=float)
        processed_spectra.append(updated)
    return processed_spectra, keys_by_index, flattening_by_key


def _maybe_emit_flatten_plot(
    args,
    *,
    spectra: list[dict],
    paths: list[Path],
    keys_by_index: list[str],
    flattening_by_key: dict[str, FlatteningResult],
) -> None:
    if not flattening_by_key or (args.flatten_plot is None and not args.flatten_show_plot):
        return

    target_key = None
    if args.baseline_match is not None:
        requested_key = str(args.baseline_match).strip().upper()
        if requested_key in flattening_by_key:
            target_key = requested_key
    elif len(spectra) == 1:
        target_key = next(iter(flattening_by_key))
    elif "A" in flattening_by_key:
        target_key = "A"
    else:
        target_key = next(iter(flattening_by_key))

    index = next((i for i, key in enumerate(keys_by_index) if key == target_key), 0)
    raw_spectrum = spectra[index]
    fig = plot_flattening_diagnostic(
        np.asarray(raw_spectrum["freq"], dtype=float),
        np.asarray(raw_spectrum["amplitude"], dtype=float),
        flattening_by_key[target_key],
        title=f"{paths[index].stem} flattening diagnostic",
    )
    if args.flatten_plot is not None:
        save_path = Path(args.flatten_plot)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Flattening plot saved to: {save_path}")
    if args.flatten_show_plot:
        plt.show()
    else:
        plt.close(fig)


def _build_compare_title(mode: str, *, switch_used: bool, custom_title: str | None, shared_geom_power: float) -> str:
    if custom_title is not None:
        return custom_title
    suffix = "B vs A" if switch_used else "A vs B"
    if mode.startswith("shared-"):
        return f"{_compare_label_with_params(mode, shared_geom_power=shared_geom_power)}: {suffix}"
    return _compare_label_with_params(mode, shared_geom_power=shared_geom_power)


def _build_compare_metadata(
    *,
    mode: str,
    switch_used: bool,
    epsilon: float,
    shared_geom_power: float,
    interp_kind: str,
    grid_mode: str,
    normalize_mode: str,
    relative_range: tuple[float, float],
    norm_low_hz: float,
    norm_high_hz: float,
    flatten_enabled: bool,
    baseline_match_target: str | None,
    flatten_reference_band: tuple[float, float],
    path_a: Path,
    path_b: Path,
    spectrum_a: dict,
    spectrum_b: dict,
) -> dict[str, object]:
    metadata_a = spectrum_a.get("metadata") or {}
    metadata_b = spectrum_b.get("metadata") or {}
    return {
        "sourceKind": "spectrasave-compare",
        "compareMode": mode,
        "resultType": "dimensionless" if mode == "a-only-ratio" else "amplitude",
        "switchUsed": bool(switch_used),
        "sourcePathA": str(path_a),
        "sourcePathB": str(path_b),
        "sourceLabelA": spectrum_a.get("label"),
        "sourceLabelB": spectrum_b.get("label"),
        "sourceComponentA": metadata_a.get("component"),
        "sourceComponentB": metadata_b.get("component"),
        "interpKind": interp_kind,
        "gridMode": grid_mode,
        "normalize": normalize_mode,
        "relativeRange": [float(relative_range[0]), float(relative_range[1])],
        "normLowHz": float(norm_low_hz),
        "normHighHz": float(norm_high_hz),
        "flattenEnabled": bool(flatten_enabled),
        "baselineMatchTarget": baseline_match_target,
        "flattenReferenceBandHz": [float(flatten_reference_band[0]), float(flatten_reference_band[1])],
        "epsilon": float(epsilon),
        "sharedGeomPower": float(shared_geom_power),
    }


def _next_ssvc_index() -> int:
    base_dir = get_default_spectrasave_dir()
    max_index = 0
    pattern = re.compile(r"^SSVC_(\d{4,})(?:__.*)?\.msgpack$")
    if base_dir.exists():
        for path in base_dir.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match is None:
                continue
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _build_compare_default_name(
    *,
    path_a: Path,
    path_b: Path,
    mode: str,
    shared_geom_power: float,
) -> str:
    index = _next_ssvc_index()
    mode_token = mode
    if mode == "shared-geom" and not np.isclose(shared_geom_power, 0.5):
        mode_token = f"{mode}-p{shared_geom_power:g}"
    suffix_name = build_default_spectrasave_name(
        path_a.stem,
        "vs",
        path_b.stem,
        mode_token,
    )
    return f"SSVC_{index:04d}__{suffix_name}"


def _save_compare_result(
    args,
    *,
    mode: str,
    path_a: Path,
    path_b: Path,
    result_freq: np.ndarray,
    result_amp: np.ndarray,
    title: str,
    metadata: dict[str, object],
    shared_geom_power: float,
) -> Path | None:
    export_path = resolve_spectrasave_path(
        args.spectrasave,
        default_name=_build_compare_default_name(
            path_a=path_a,
            path_b=path_b,
            mode=mode,
            shared_geom_power=shared_geom_power,
        ),
    )
    if export_path is None:
        return None
    saved = save_spectrum_msgpack(
        export_path,
        freq=result_freq,
        amplitude=result_amp,
        label=title,
        metadata=metadata,
    )
    print(f"Spectrum saved to: {saved}")
    return saved


def _compare_spectra(
    path_a: Path,
    spectrum_a: dict,
    path_b: Path,
    spectrum_b: dict,
    *,
    mode: str,
    interp_kind: str,
    grid_mode: str,
    flatten_enabled: bool,
    flatten_reference_band: tuple[float, float],
    baseline_match_target: str | None,
    normalize_mode: str,
    relative_range: tuple[float, float],
    epsilon: float,
    shared_geom_power: float,
    normalize_last: bool,
) -> tuple[dict, dict, dict, dict[str, FlatteningResult], tuple[float, float]]:
    contributions = [
        _make_spectrum_contribution(path_a, spectrum_a),
        _make_spectrum_contribution(path_b, spectrum_b),
    ]
    freq_low, freq_high = choose_frequency_window(contributions)
    freq_grid = build_common_grid(contributions, freq_low, freq_high, grid_mode=grid_mode)

    aligned_a = dict(spectrum_a)
    aligned_b = dict(spectrum_b)
    aligned_a["freq"] = freq_grid
    aligned_b["freq"] = freq_grid
    aligned_a["amplitude"] = interp_amplitude(
        np.asarray(spectrum_a["freq"], dtype=float),
        np.asarray(spectrum_a["amplitude"], dtype=float),
        freq_grid,
        kind=interp_kind,
    )
    aligned_b["amplitude"] = interp_amplitude(
        np.asarray(spectrum_b["freq"], dtype=float),
        np.asarray(spectrum_b["amplitude"], dtype=float),
        freq_grid,
        kind=interp_kind,
    )

    ordered = [
        {"path": path_a, "key": "A", "spectrum": aligned_a},
        {"path": path_b, "key": "B", "spectrum": aligned_b},
    ]

    flattenings: dict[str, FlatteningResult] = {}
    processed_aligned = [dict(entry["spectrum"]) for entry in ordered]
    if normalize_last:
        if flatten_enabled or baseline_match_target is not None:
            temp_args = argparse.Namespace(
                flatten=flatten_enabled,
                baseline_match=baseline_match_target,
                flatten_reference_band=flatten_reference_band,
            )
            processed_aligned, _, flattenings = _apply_baseline_processing_with_keys(
                [entry["spectrum"] for entry in ordered],
                keys_by_index=["A", "B"],
                args=temp_args,
            )

        proc_a = dict(processed_aligned[0])
        proc_b = dict(processed_aligned[1])
        proc_a["amplitude"], norm_low, norm_high = _normalize_compare_spectrum(
            np.asarray(proc_a["freq"], dtype=float),
            np.asarray(proc_a["amplitude"], dtype=float),
            normalize_mode=normalize_mode,
            relative_range=relative_range,
        )
        proc_b["amplitude"], _, _ = _normalize_compare_spectrum(
            np.asarray(proc_b["freq"], dtype=float),
            np.asarray(proc_b["amplitude"], dtype=float),
            normalize_mode=normalize_mode,
            relative_range=relative_range,
        )
    else:
        proc_a = dict(processed_aligned[0])
        proc_b = dict(processed_aligned[1])
        proc_a["amplitude"], norm_low, norm_high = _normalize_compare_spectrum(
            np.asarray(proc_a["freq"], dtype=float),
            np.asarray(proc_a["amplitude"], dtype=float),
            normalize_mode=normalize_mode,
            relative_range=relative_range,
        )
        proc_b["amplitude"], _, _ = _normalize_compare_spectrum(
            np.asarray(proc_b["freq"], dtype=float),
            np.asarray(proc_b["amplitude"], dtype=float),
            normalize_mode=normalize_mode,
            relative_range=relative_range,
        )
        normalized_inputs = [proc_a, proc_b]
        if flatten_enabled or baseline_match_target is not None:
            temp_args = argparse.Namespace(
                flatten=flatten_enabled,
                baseline_match=baseline_match_target,
                flatten_reference_band=flatten_reference_band,
            )
            processed_aligned, _, flattenings = _apply_baseline_processing_with_keys(
                normalized_inputs,
                keys_by_index=["A", "B"],
                args=temp_args,
            )
            proc_a = dict(processed_aligned[0])
            proc_b = dict(processed_aligned[1])

    amp_a = np.asarray(proc_a["amplitude"], dtype=float)
    amp_b = np.asarray(proc_b["amplitude"], dtype=float)
    if mode == "shared-geom":
        result_amp = np.power(np.maximum(amp_a, 0.0) * np.maximum(amp_b, 0.0), float(shared_geom_power))
    elif mode == "shared-min":
        result_amp = np.minimum(amp_a, amp_b)
    elif mode == "a-only-diff":
        result_amp = np.maximum(amp_a - amp_b, 0.0)
    elif mode == "a-only-ratio":
        result_amp = amp_a / (amp_b + float(epsilon))
    else:
        raise ValueError(f"Unsupported comparison mode: {mode}")

    result = {
        "label": _compare_label_with_params(mode, shared_geom_power=shared_geom_power),
        "metadata": {},
        "freq": np.asarray(proc_a["freq"], dtype=float),
        "amplitude": np.asarray(result_amp, dtype=float),
    }
    return proc_a, proc_b, result, flattenings, (norm_low, norm_high)


def plot_multi_spectrasave(
    spectra: list[dict],
    *,
    labels: list[str] | None,
    filenames: list[str] | None,
    use_subplots: bool,
    fft_log: bool,
    freq_min_hz: float | None,
    freq_max_hz: float | None,
    title: str | None,
    peaks_hz_list: list[list[float]] | None = None,
    tickspace_hz: float | None = None,
    y_label: str = "Amplitude",
    offset_k: int | None = None,
    offset_o: float | None = None,
    scale_k: int | None = None,
    scale_s: float | None = None,
):
    n = len(spectra)
    if labels is None:
        if filenames:
            labels = [f"{filenames[i]}: {s.get('label') or ''}" if s.get('label') else filenames[i] for i, s in enumerate(spectra)]
        else:
            labels = [s.get("label") or f"Spectrum {i}" for i, s in enumerate(spectra)]

    # Compute global freq bounds across all spectra
    all_x_min = []
    all_x_max = []
    for s in spectra:
        x_min, x_max = _compute_freq_bounds(s["freq"], freq_min_hz, freq_max_hz)
        all_x_min.append(x_min)
        all_x_max.append(x_max)
    global_x_min = min(all_x_min)
    global_x_max = max(all_x_max)

    if use_subplots:
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True, constrained_layout=True)
        if n == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        axes = [ax] * n

    for i in range(n):
        s = spectra[i]
        ax = axes[i]
        freq = np.asarray(s["freq"], dtype=float)
        amp = np.asarray(s["amplitude"], dtype=float)
        if offset_k is not None and offset_o is not None and i == offset_k - 1:
            amp = amp + offset_o
        if scale_k is not None and scale_s is not None and i == scale_k - 1:
            amp = amp * scale_s
        label = labels[i]

        if fft_log:
            ax.semilogy(freq, amp, linewidth=1.2, label=label)
            positive = amp[np.isfinite(amp) & (amp > 0)]
            if positive.size > 0:
                ymin = np.percentile(positive, 0.1) * 0.7
                ymax = np.max(positive) * 1.3
                # For overlay, we only adjust once or keep track of global ymin/ymax
                # For subplots, we adjust per-axis.
                if use_subplots:
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                        ax.set_ylim(ymin, ymax)
        else:
            ax.plot(freq, amp, linewidth=1.2, label=label)

        if use_subplots:
            ax.set_title(label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            if peaks_hz_list and peaks_hz_list[i]:
                for peak in peaks_hz_list[i]:
                    if global_x_min <= peak <= global_x_max:
                        ax.axvline(peak, color="tab:red", linewidth=1.1, alpha=0.6, zorder=3)

    if not use_subplots:
        ax = axes[0]
        ax.set_title(title or "Spectra Comparison")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(y_label)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        if peaks_hz_list:
            # For overlay, show all unique peaks
            unique_peaks = sorted(set(p for sublist in peaks_hz_list for p in sublist))
            for peak in unique_peaks:
                if global_x_min <= peak <= global_x_max:
                    ax.axvline(peak, color="tab:red", linewidth=1.1, alpha=0.4, zorder=3)
    else:
        axes[-1].set_xlabel("Frequency (Hz)")

    for ax in (axes if use_subplots else [axes[0]]):
        ax.set_xlim(global_x_min, global_x_max)
        apply_major_tick_spacing(ax, tickspace_hz, axis="x")

    if title and use_subplots:
        fig.suptitle(title, fontsize=14)

    return fig


def plot_spectrasave_dual_panel(
    spectrum: dict,
    *,
    fft_log: bool,
    image_plot_scale: str,
    show_image: bool,
    cmap_index: int,
    freq_min_hz: float | None,
    freq_max_hz: float | None,
    image_cols: int,
    full_couple: bool,
    title: str | None,
    peaks_hz: list[float],
    show_lines: bool,
    tickspace_hz: float | None,
    offset_k: int | None = None,
    offset_o: float | None = None,
    scale_k: int | None = None,
    scale_s: float | None = None,
):
    freq = np.asarray(spectrum["freq"], dtype=float)
    amp = np.asarray(spectrum["amplitude"], dtype=float)
    if offset_k is not None and offset_o is not None and offset_k == 1:
        amp = amp + offset_o
    if scale_k is not None and scale_s is not None and scale_k == 1:
        amp = amp * scale_s

    finite_freq = freq[np.isfinite(freq)]
    if finite_freq.size == 0:
        raise ValueError("Spectrum frequency axis is empty or non-finite")

    x_min, x_max = resolve_clipped_window(
        float(np.min(finite_freq)),
        float(np.max(finite_freq)),
        freq_min_hz,
        freq_max_hz,
    )

    label = spectrum.get("label") or "Spectrum"
    if show_image:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        ax_fft = axes[0]
        ax_img = axes[1]
    else:
        fig, ax_fft = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax_img = None

    _plot_fft_curve_panel(
        ax_fft,
        freq=freq,
        amp=amp,
        log_scale=fft_log,
        x_min=x_min,
        x_max=x_max,
        title=f"{label} FFT",
        x_tickspace_hz=tickspace_hz,
    )

    for peak in peaks_hz:
        if x_min <= peak <= x_max:
            ax_fft.axvline(peak, color="tab:red", linewidth=1.1, alpha=0.6, zorder=3)

    if show_image and ax_img is not None:
        _plot_frequency_image(
            fig,
            ax_img,
            freq=freq,
            amp=amp,
            plot_scale=image_plot_scale,
            cmap_index=cmap_index,
            y_min=x_min,
            y_max=x_max,
            x_max=float(image_cols),
            title=f"{label} Full Image",
            y_tickspace_hz=tickspace_hz,
        )

    if show_image and full_couple and ax_img is not None:
        _link_fft_frequency_to_image_frequency([ax_fft], [ax_img])

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.normalize = resolve_normalization_mode(args)
    if args.baseline_match is not None:
        args.baseline_match = str(args.baseline_match).strip().upper()
    freq_window_error = validate_frequency_window_args(args)
    if freq_window_error is not None:
        print(freq_window_error, file=sys.stderr)
        return 1
    tickspace_error = validate_tickspace_arg(args)
    if tickspace_error is not None:
        print(tickspace_error, file=sys.stderr)
        return 1
    flatten_error = _validate_flatten_args(args)
    if flatten_error is not None:
        print(flatten_error, file=sys.stderr)
        return 1
    relative_range_error = _validate_relative_range(args)
    if relative_range_error is not None:
        print(relative_range_error, file=sys.stderr)
        return 1

    if args.labels and len(args.labels) != len(args.spectrasave_inputs):
        print(f"Error: Number of labels ({len(args.labels)}) must match number of spectrasave files ({len(args.spectrasave_inputs)})", file=sys.stderr)
        return 1
    if args.compare is not None and len(args.spectrasave_inputs) != 2:
        print("Error: --compare requires exactly two spectrasave inputs", file=sys.stderr)
        return 1
    if args.epsilon <= 0:
        print("Error: --epsilon must be greater than 0", file=sys.stderr)
        return 1
    if args.shared_geom_power <= 0:
        print("Error: --shared-geom-power must be greater than 0", file=sys.stderr)
        return 1
    if args.compare is not None and args.baseline_match is not None:
        lowered = str(args.baseline_match).strip().lower()
        if lowered not in {"a", "b", "none"}:
            print("Error: In comparison mode, --baseline-match must be omitted, passed with no value, or set to A/B/none", file=sys.stderr)
            return 1

    flatten_enabled = _resolve_compare_flatten_enabled(args)

    if not flatten_enabled and args.baseline_match is not None and args.baseline_match != "none":
        import warnings

        warnings.warn(
            f"--baseline-match={args.baseline_match} is active without --flatten; "
            "warping component baselines multiplicatively to match the target spectrum's curved response envelope. "
            "Pass --flatten to also flatten to a horizontal reference line.",
            UserWarning,
        )

    try:
        spectra = []
        spectrasave_paths: list[Path] = []
        all_peaks_hz = []
        filenames = []
        for path_str in args.spectrasave_inputs:
            path = resolve_existing_spectrasave_path(path_str)
            spectrum = load_spectrum_msgpack(path)
            spectra.append(spectrum)
            spectrasave_paths.append(path)
            filenames.append(path.name)
            
            peaks_hz: list[float] = []
            if args.preload_csv and args.show_peaks:
                # If a single --peaks-csv was given, it might not make sense for multiple files
                # but we'll try to resolve it per-file if not explicitly given.
                p_csv = Path(args.peaks_csv) if args.peaks_csv is not None else _default_peaks_csv_path(path)
                peaks_hz = _load_preload_peaks(p_csv, require_exists=args.peaks_csv is not None)
            all_peaks_hz.append(peaks_hz)

            if args.show_metadata:
                _print_metadata(path, spectrum)
                if peaks_hz:
                    print(f"  Preloaded peaks: {len(peaks_hz)}")

        if args.baseline_match is not None and args.baseline_match != "NONE":
            available_targets = {_index_to_role(index) for index in range(len(spectra))}
            if args.baseline_match not in available_targets:
                printable = ", ".join(sorted(available_targets))
                print(f"Error: --baseline-match target must be one of {printable} for the provided inputs", file=sys.stderr)
                return 1

        if not args.show_peaks:
            all_peaks_hz = [[] for _ in all_peaks_hz]

        offset_k = int(args.offset[0]) if args.offset else None
        offset_o = float(args.offset[1]) if args.offset else None
        scale_k = int(args.scale[0]) if args.scale else None
        scale_s = float(args.scale[1]) if args.scale else None

        if args.compare is not None:
            logical = [
                (spectrasave_paths[1], spectra[1], all_peaks_hz[1], filenames[1]),
                (spectrasave_paths[0], spectra[0], all_peaks_hz[0], filenames[0]),
            ] if args.switch else [
                (spectrasave_paths[0], spectra[0], all_peaks_hz[0], filenames[0]),
                (spectrasave_paths[1], spectra[1], all_peaks_hz[1], filenames[1]),
            ]
            path_a, spectrum_a, peaks_a, filename_a = logical[0]
            path_b, spectrum_b, peaks_b, filename_b = logical[1]
            baseline_match_target = args.baseline_match
            if baseline_match_target is None and flatten_enabled:
                baseline_match_target = "A"
            elif baseline_match_target is not None:
                lowered = str(baseline_match_target).strip().lower()
                if lowered == "a":
                    baseline_match_target = "A"
                elif lowered == "b":
                    baseline_match_target = "B"
                elif lowered == "none":
                    baseline_match_target = None

            raw_compare_inputs = [
                {
                    **dict(spectrum_a),
                    "freq": np.asarray(spectrum_a["freq"], dtype=float).copy(),
                    "amplitude": np.asarray(spectrum_a["amplitude"], dtype=float).copy(),
                },
                {
                    **dict(spectrum_b),
                    "freq": np.asarray(spectrum_b["freq"], dtype=float).copy(),
                    "amplitude": np.asarray(spectrum_b["amplitude"], dtype=float).copy(),
                },
            ]
            proc_a, proc_b, result_spectrum, flattening_by_key, (norm_low, norm_high) = _compare_spectra(
                path_a,
                spectrum_a,
                path_b,
                spectrum_b,
                mode=args.compare,
                interp_kind=args.interp_kind,
                grid_mode="coarsest" if args.coarsest else "finest",
                flatten_enabled=flatten_enabled,
                flatten_reference_band=tuple(float(value) for value in args.flatten_reference_band),
                baseline_match_target=baseline_match_target,
                normalize_mode=args.normalize,
                relative_range=tuple(float(value) for value in args.relative_range),
                epsilon=float(args.epsilon),
                shared_geom_power=float(args.shared_geom_power),
                normalize_last=bool(args.normalize_last),
            )
            if flattening_by_key:
                _maybe_emit_flatten_plot(
                    argparse.Namespace(
                        flatten_plot=args.flatten_plot,
                        flatten_show_plot=args.flatten_show_plot,
                        baseline_match=baseline_match_target,
                    ),
                    spectra=raw_compare_inputs,
                    paths=[path_a, path_b],
                    keys_by_index=["A", "B"],
                    flattening_by_key=flattening_by_key,
                )

            display_a = dict(proc_a)
            display_b = dict(proc_b)
            display_result = dict(result_spectrum)
            display_a["amplitude"] = _maybe_smooth(np.asarray(display_a["amplitude"], dtype=float), args)
            display_b["amplitude"] = _maybe_smooth(np.asarray(display_b["amplitude"], dtype=float), args)
            display_result["amplitude"] = _maybe_smooth(np.asarray(display_result["amplitude"], dtype=float), args)
            compare_title = _build_compare_title(
                args.compare,
                switch_used=args.switch,
                custom_title=args.title,
                shared_geom_power=float(args.shared_geom_power),
            )
            compare_metadata = _build_compare_metadata(
                mode=args.compare,
                switch_used=args.switch,
                epsilon=float(args.epsilon),
                shared_geom_power=float(args.shared_geom_power),
                interp_kind=args.interp_kind,
                grid_mode="coarsest" if args.coarsest else "finest",
                normalize_mode=args.normalize,
                relative_range=tuple(float(value) for value in args.relative_range),
                norm_low_hz=norm_low,
                norm_high_hz=norm_high,
                flatten_enabled=flatten_enabled,
                baseline_match_target=baseline_match_target,
                flatten_reference_band=tuple(float(value) for value in args.flatten_reference_band),
                path_a=path_a,
                path_b=path_b,
                spectrum_a=spectrum_a,
                spectrum_b=spectrum_b,
            )
            _save_compare_result(
                args,
                mode=args.compare,
                path_a=path_a,
                path_b=path_b,
                result_freq=np.asarray(result_spectrum["freq"], dtype=float),
                result_amp=np.asarray(result_spectrum["amplitude"], dtype=float),
                title=compare_title,
                metadata=compare_metadata,
                shared_geom_power=float(args.shared_geom_power),
            )
            if args.only_result:
                fig = plot_multi_spectrasave(
                    [display_result],
                    labels=[_compare_label_with_params(args.compare, shared_geom_power=float(args.shared_geom_power))],
                    filenames=None,
                    use_subplots=False,
                    fft_log=args.fft_log,
                    freq_min_hz=args.freq_min_hz,
                    freq_max_hz=args.freq_max_hz,
                    title=compare_title,
                    peaks_hz_list=[[]],
                    tickspace_hz=args.tickspace_hz,
                    y_label=_compare_y_label(args.compare),
                )
            else:
                fig = plot_multi_spectrasave(
                    [display_a, display_b, display_result],
                    labels=["A", "B", _compare_label_with_params(args.compare, shared_geom_power=float(args.shared_geom_power))],
                    filenames=[filename_a, filename_b, "derived"],
                    use_subplots=args.compare_subplots,
                    fft_log=args.fft_log,
                    freq_min_hz=args.freq_min_hz,
                    freq_max_hz=args.freq_max_hz,
                    title=compare_title,
                    peaks_hz_list=[peaks_a, peaks_b, []],
                    tickspace_hz=args.tickspace_hz,
                    y_label=_compare_y_label(args.compare),
                )
            render_figure(fig, save=args.save)
            return 0

        for spectrum in spectra:
            spectrum["amplitude"] = _maybe_smooth(np.asarray(spectrum["amplitude"], dtype=float), args)

        raw_spectra = [
            {
                **spectrum,
                "freq": np.asarray(spectrum["freq"], dtype=float).copy(),
                "amplitude": np.asarray(spectrum["amplitude"], dtype=float).copy(),
            }
            for spectrum in spectra
        ]
        baseline_args = argparse.Namespace(
            flatten=flatten_enabled,
            baseline_match=args.baseline_match,
            flatten_reference_band=args.flatten_reference_band,
        )
        spectra, spectrum_keys, flattening_by_key = _maybe_apply_baseline_processing(spectra, spectrasave_paths, baseline_args)
        _maybe_emit_flatten_plot(
            args,
            spectra=raw_spectra,
            paths=spectrasave_paths,
            keys_by_index=spectrum_keys,
            flattening_by_key=flattening_by_key,
        )

        if len(spectra) == 1:
            fig = plot_spectrasave_dual_panel(
                spectra[0],
                fft_log=args.fft_log,
                image_plot_scale=args.image_plot_scale,
                show_image=args.show_image,
                cmap_index=args.cm,
                freq_min_hz=args.freq_min_hz,
                freq_max_hz=args.freq_max_hz,
                image_cols=args.image_cols,
                full_couple=args.full_couple,
                title=args.title,
                peaks_hz=all_peaks_hz[0],
                show_lines=args.show_lines,
                tickspace_hz=args.tickspace_hz,
                offset_k=offset_k,
                offset_o=offset_o,
                scale_k=scale_k,
                scale_s=scale_s,
            )
        else:
            fig = plot_multi_spectrasave(
                spectra,
                labels=args.labels,
                filenames=filenames,
                use_subplots=args.use_subplots,
                fft_log=args.fft_log,
                freq_min_hz=args.freq_min_hz,
                freq_max_hz=args.freq_max_hz,
                title=args.title,
                peaks_hz_list=all_peaks_hz,
                tickspace_hz=args.tickspace_hz,
                offset_k=offset_k,
                offset_o=offset_o,
                scale_k=scale_k,
                scale_s=scale_s,
            )
            
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
