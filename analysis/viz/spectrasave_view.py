#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
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
    add_frequency_window_args,
    add_output_args,
    add_tickspace_arg,
    validate_frequency_window_args,
    validate_tickspace_arg,
)
from tools.peaks import load_peaks_csv
from tools.spectrasave import load_spectrum_msgpack, resolve_existing_spectrasave_path


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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display one or more SpectraSave spectra as curves with optional image panels.",
    )
    parser.add_argument("spectrasave", nargs="+", help="Path to one or more SpectraSave msgpack artifacts.")
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
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            if peaks_hz_list and peaks_hz_list[i]:
                for peak in peaks_hz_list[i]:
                    if global_x_min <= peak <= global_x_max:
                        ax.axvline(peak, color="tab:red", linewidth=1.1, alpha=0.6, zorder=3)

    if not use_subplots:
        ax = axes[0]
        ax.set_title(title or "Spectra Comparison")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
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
):
    freq = np.asarray(spectrum["freq"], dtype=float)
    amp = np.asarray(spectrum["amplitude"], dtype=float)
    if offset_k is not None and offset_o is not None and offset_k == 1:
        amp = amp + offset_o

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax_fft = axes[0]
    ax_img = axes[1]

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

    if full_couple:
        _link_fft_frequency_to_image_frequency([ax_fft], [ax_img])

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    freq_window_error = validate_frequency_window_args(args)
    if freq_window_error is not None:
        print(freq_window_error, file=sys.stderr)
        return 1
    tickspace_error = validate_tickspace_arg(args)
    if tickspace_error is not None:
        print(tickspace_error, file=sys.stderr)
        return 1

    if args.labels and len(args.labels) != len(args.spectrasave):
        print(f"Error: Number of labels ({len(args.labels)}) must match number of spectrasave files ({len(args.spectrasave)})", file=sys.stderr)
        return 1

    try:
        spectra = []
        all_peaks_hz = []
        filenames = []
        for path_str in args.spectrasave:
            path = resolve_existing_spectrasave_path(path_str)
            spectrum = load_spectrum_msgpack(path)
            spectra.append(spectrum)
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

        if not args.show_peaks:
            all_peaks_hz = [[] for _ in all_peaks_hz]

        offset_k = int(args.offset[0]) if args.offset else None
        offset_o = float(args.offset[1]) if args.offset else None
        scale_k = int(args.scale[0]) if args.scale else None
        scale_s = float(args.scale[1]) if args.scale else None

        for spectrum in spectra:
            spectrum["amplitude"] = _maybe_smooth(np.asarray(spectrum["amplitude"], dtype=float), args).tolist()

        if len(spectra) == 1:
            fig = plot_spectrasave_dual_panel(
                spectra[0],
                fft_log=args.fft_log,
                image_plot_scale=args.image_plot_scale,
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
