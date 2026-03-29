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

from plotting.common import apply_major_tick_spacing, render_figure, resolve_clipped_window
from plotting.frequency import _link_fft_frequency_to_image_frequency, _plot_frequency_image
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
        description="Display a saved SpectraSave spectrum as a curve plus full-image panel with coupled zoom.",
    )
    parser.add_argument("spectrasave", help="Path to a SpectraSave msgpack artifact.")
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
    peaks_hz: list[float] | None = None,
    show_lines: bool = True,
    tickspace_hz: float | None = None,
):
    freq = np.asarray(spectrum["freq"], dtype=float)
    amp = np.asarray(spectrum["amplitude"], dtype=float)
    if freq.ndim != 1 or amp.ndim != 1 or freq.size != amp.size:
        raise ValueError("SpectraSave payload must contain equal-length 1D freq and amplitude arrays")

    x_min, x_max = _compute_freq_bounds(freq, freq_min_hz, freq_max_hz)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.5),
        gridspec_kw={"width_ratios": [1.35, 0.85]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.11, top=0.90 if title else 0.94, wspace=0.18)

    ax_fft, ax_image = axes

    if fft_log:
        ax_fft.semilogy(freq, amp, linewidth=1.2)
        positive = amp[np.isfinite(amp) & (amp > 0)]
        if positive.size > 0:
            ymin = np.percentile(positive, 0.1) * 0.7
            ymax = np.max(positive) * 1.3
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                ax_fft.set_ylim(ymin, ymax)
    else:
        ax_fft.plot(freq, amp, linewidth=1.2)

    curve_title = spectrum.get("label") or "SpectraSave FFT"
    ax_fft.set_title(curve_title)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_xlim(x_min, x_max)
    apply_major_tick_spacing(ax_fft, tickspace_hz, axis="x")
    ax_fft.grid(True, alpha=0.3)

    _plot_frequency_image(
        fig,
        ax_image,
        freq=freq,
        amp=amp,
        plot_scale=image_plot_scale,
        cmap_index=cmap_index,
        y_min=x_min,
        y_max=x_max,
        x_label="Arbitrary X",
        x_max=float(max(1, image_cols)),
        title="Full FFT Image",
        linear_color_label="Amplitude",
        log_color_label="Amplitude (dB)",
        y_tickspace_hz=tickspace_hz,
    )
    ax_image.set_xticks([])

    peaks_hz = [] if peaks_hz is None else [float(peak) for peak in peaks_hz]
    visible_peaks = [peak for peak in peaks_hz if x_min <= peak <= x_max]
    if visible_peaks:
        overlay_color = "tab:red"
        peak_x = np.linspace(0.15, max(0.15, float(max(1, image_cols)) - 0.15), len(visible_peaks), dtype=float)

        for peak in visible_peaks:
            ax_fft.axvline(peak, color=overlay_color, linewidth=1.1, alpha=0.9, zorder=3)

        ax_image.scatter(
            peak_x,
            visible_peaks,
            s=30,
            color=overlay_color,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        if show_lines and len(visible_peaks) > 1:
            ax_image.plot(peak_x, visible_peaks, color=overlay_color, linewidth=1.1, alpha=0.9, zorder=3)

    if full_couple:
        _link_fft_frequency_to_image_frequency([ax_fft], [ax_image])

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

    try:
        path = resolve_existing_spectrasave_path(args.spectrasave)
        spectrum = load_spectrum_msgpack(path)
        peaks_csv = None
        peaks_hz: list[float] = []

        if args.preload_csv:
            peaks_csv = Path(args.peaks_csv) if args.peaks_csv is not None else _default_peaks_csv_path(path)
            peaks_hz = _load_preload_peaks(peaks_csv, require_exists=args.peaks_csv is not None)

        if args.show_metadata:
            _print_metadata(path, spectrum)
            if peaks_csv is not None and peaks_hz:
                print(f"Preloaded peaks: {len(peaks_hz)} from {peaks_csv}")
            elif peaks_csv is not None and args.peaks_csv is not None:
                print(f"Preloaded peaks: 0 from {peaks_csv}")

        fig = plot_spectrasave_dual_panel(
            spectrum,
            fft_log=args.fft_log,
            image_plot_scale=args.image_plot_scale,
            cmap_index=args.cm,
            freq_min_hz=args.freq_min_hz,
            freq_max_hz=args.freq_max_hz,
            image_cols=args.image_cols,
            full_couple=args.full_couple,
            title=args.title,
            peaks_hz=peaks_hz,
            show_lines=args.show_lines,
            tickspace_hz=args.tickspace,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
