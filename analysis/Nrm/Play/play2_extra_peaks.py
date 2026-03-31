#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    analysis_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(analysis_root))

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting.common import apply_major_tick_spacing, render_figure, resolve_clipped_window
from analysis.plotting.frequency import _link_fft_frequency_to_image_frequency, _plot_frequency_image
from analysis.tools.cli import (
    add_colormap_arg,
    add_frequency_window_args,
    add_output_args,
    add_tickspace_arg,
    validate_frequency_window_args,
    validate_tickspace_arg,
)
from analysis.tools.peaks import load_peaks_csv, resolve_peaks_csv
from analysis.tools.spectrasave import load_spectrum_msgpack, resolve_existing_spectrasave_path


@dataclass(frozen=True)
class OverlayPeak:
    freq_hz: float
    kind: str
    color: str
    linestyle: str
    linewidth: float
    source_text: str


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _default_peaks_csv_path(spectrasave_path: Path) -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "peaks" / f"{spectrasave_path.stem}.csv"


def _resolve_peaks_path(raw_value: str | None, spectrasave_path: Path, *, preload_csv: bool) -> Path | None:
    if raw_value is not None:
        return resolve_peaks_csv(raw_value)
    if not preload_csv:
        return None
    default_path = _default_peaks_csv_path(spectrasave_path)
    return default_path if default_path.exists() else None


def _load_peaks(peaks_path: Path | None) -> np.ndarray:
    if peaks_path is None:
        return np.array([], dtype=float)
    peaks = [float(value) for value in load_peaks_csv(peaks_path) if np.isfinite(value) and value > 0]
    if not peaks:
        return np.array([], dtype=float)
    return np.unique(np.asarray(peaks, dtype=float))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay base, harmonic, sum, and difference peaks on a SpectraSave spectrum.",
    )
    parser.add_argument("spectrasave", help="Path to a SpectraSave msgpack artifact.")
    parser.add_argument(
        "peaks_csv",
        nargs="?",
        default=None,
        help="Optional peaks CSV name or path. Uses the standard peaks resolver. Defaults to configs/peaks/<spectrasave-stem>.csv when omitted and preload is enabled.",
    )
    parser.add_argument(
        "--no-preload-csv",
        dest="preload_csv",
        action="store_false",
        help="Disable automatic loading of configs/peaks/<spectrasave-stem>.csv.",
    )
    parser.set_defaults(preload_csv=True)
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


def _build_overlays(peaks_hz: np.ndarray, freq_min: float, freq_max: float) -> list[OverlayPeak]:
    overlays: list[OverlayPeak] = []

    for peak in peaks_hz:
        if freq_min <= peak <= freq_max:
            overlays.append(
                OverlayPeak(
                    freq_hz=float(peak),
                    kind="peak",
                    color="tab:red",
                    linestyle="-",
                    linewidth=1.2,
                    source_text=f"{peak:.6g} Hz peak",
                )
            )

        harmonic = 2.0 * float(peak)
        if freq_min <= harmonic <= freq_max:
            overlays.append(
                OverlayPeak(
                    freq_hz=harmonic,
                    kind="harmonic",
                    color="tab:red",
                    linestyle="--",
                    linewidth=0.8,
                    source_text=f"{harmonic:.6g} Hz harmonic from 2*{peak:.6g}",
                )
            )

    for idx, left in enumerate(peaks_hz):
        for right in peaks_hz[idx + 1 :]:
            sum_hz = float(left + right)
            if freq_min <= sum_hz <= freq_max:
                overlays.append(
                    OverlayPeak(
                        freq_hz=sum_hz,
                        kind="sum",
                        color="tab:green",
                        linestyle="--",
                        linewidth=0.8,
                        source_text=f"{sum_hz:.6g} Hz sum from {left:.6g}+{right:.6g}",
                    )
                )

            diff_hz = float(abs(right - left))
            if diff_hz > 0 and freq_min <= diff_hz <= freq_max:
                overlays.append(
                    OverlayPeak(
                        freq_hz=diff_hz,
                        kind="difference",
                        color="0.5",
                        linestyle="--",
                        linewidth=0.8,
                        source_text=f"{diff_hz:.6g} Hz diff from {right:.6g}-{left:.6g}",
                    )
                )
    return overlays


def _annotate_overlay_labels(ax, overlays: list[OverlayPeak], freq: np.ndarray, amp: np.ndarray, *, fft_log: bool) -> None:
    if not overlays:
        return

    eps = np.finfo(float).tiny
    positive = amp[np.isfinite(amp) & (amp > 0)]
    y_min, y_max = ax.get_ylim()

    for idx, overlay in enumerate(overlays):
        y_here = float(np.interp(overlay.freq_hz, freq, amp))
        if fft_log and positive.size > 0:
            y_low = max(float(np.min(positive)), eps)
            y_high = max(float(np.max(positive)), y_low * 10.0)
            fraction = 0.90 - 0.10 * (idx % 4)
            y_text = np.exp(np.log(y_low) + fraction * (np.log(y_high) - np.log(y_low)))
        else:
            span = y_max - y_min
            y_text = y_min + (0.92 - 0.10 * (idx % 4)) * span
        label = overlay.source_text
        ax.text(
            overlay.freq_hz,
            y_text,
            label,
            rotation=90,
            rotation_mode="anchor",
            ha="left",
            va="top",
            fontsize=8,
            color=overlay.color,
            alpha=0.95,
            clip_on=True,
        )


def plot_spectrasave_with_extra_peaks(
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
    peaks_hz: np.ndarray,
    tickspace_hz: float | None = None,
):
    freq = np.asarray(spectrum["freq"], dtype=float)
    amp = np.asarray(spectrum["amplitude"], dtype=float)
    if freq.ndim != 1 or amp.ndim != 1 or freq.size != amp.size:
        raise ValueError("SpectraSave payload must contain equal-length 1D freq and amplitude arrays")

    supported_min = float(np.min(freq))
    supported_max = float(np.max(freq))
    overlays = _build_overlays(peaks_hz, supported_min, supported_max)
    x_min, x_max = _compute_freq_bounds(freq, freq_min_hz, freq_max_hz)
    visible_overlays = [overlay for overlay in overlays if x_min <= overlay.freq_hz <= x_max]

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

    for overlay in visible_overlays:
        ax_fft.axvline(
            overlay.freq_hz,
            color=overlay.color,
            linestyle=overlay.linestyle,
            linewidth=overlay.linewidth,
            alpha=0.9,
            zorder=3,
        )
        ax_image.axhline(
            overlay.freq_hz,
            color=overlay.color,
            linestyle=overlay.linestyle,
            linewidth=overlay.linewidth,
            alpha=0.7,
            zorder=3,
        )

    _annotate_overlay_labels(ax_fft, visible_overlays, freq, amp, fft_log=fft_log)

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

    if full_couple:
        _link_fft_frequency_to_image_frequency([ax_fft], [ax_image])

    if title:
        fig.suptitle(title)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    error = validate_frequency_window_args(args) or validate_tickspace_arg(args)
    if error is not None:
        raise SystemExit(error)

    spectrasave_path = resolve_existing_spectrasave_path(args.spectrasave)
    spectrum = load_spectrum_msgpack(spectrasave_path)
    peaks_path = _resolve_peaks_path(args.peaks_csv, spectrasave_path, preload_csv=args.preload_csv)
    peaks_hz = _load_peaks(peaks_path)

    if args.show_metadata:
        _print_metadata(spectrasave_path, spectrum)
        if peaks_path is None:
            print("Peaks CSV: <none>")
        else:
            print(f"Peaks CSV: {peaks_path}")
            print(f"Loaded peaks: {len(peaks_hz)}")

    fig = plot_spectrasave_with_extra_peaks(
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
        tickspace_hz=args.tickspace,
    )
    render_figure(fig, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
