#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from plotting.common import (
    apply_major_tick_spacing,
    centers_to_edges,
    colormap_name,
    ensure_parent_dir,
    resolve_clipped_window,
)
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


DEFAULT_PEAKS_DIR = Path(__file__).resolve().parents[1] / "configs" / "peaks"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively pick peak frequencies from a spectrasave spectrum.",
    )
    parser.add_argument(
        "spectrasave",
        help="Path to a spectrasave spectrum msgpack file.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV path. Defaults to configs/peaks/<spectrasave-stem>.csv.",
    )
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
        dest="image_log",
        action="store_true",
        help="Use a log / dB display for the image panel (default).",
    )
    image_group.add_argument(
        "--image-linear",
        dest="image_log",
        action="store_false",
        help="Use a linear display for the image panel.",
    )
    parser.set_defaults(image_log=True)

    add_frequency_window_args(parser)
    add_tickspace_arg(parser)
    parser.add_argument(
        "--image-columns",
        type=int,
        default=64,
        help="Number of repeated columns to render in the right-hand image. Default: 64",
    )
    parser.add_argument(
        "--marker-x",
        type=float,
        default=0.5,
        help="Horizontal placement for picked points on the image panel, as a fraction in [0, 1]. Default: 0.5",
    )
    return parser


def default_csv_path(spectrasave_path: Path) -> Path:
    return DEFAULT_PEAKS_DIR / f"{spectrasave_path.stem}.csv"


def save_peaks_csv(path: str | Path, peaks: list[float]) -> Path:
    output = ensure_parent_dir(path)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{peak:.12g}" for peak in peaks])
    return output


def load_existing_peaks(path: str | Path) -> list[float]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    peaks = [float(peak) for peak in load_peaks_csv(csv_path) if np.isfinite(peak) and peak > 0]
    return sorted(set(peaks))


class PeakPicker:
    def __init__(
        self,
        *,
        freq: np.ndarray,
        amplitude: np.ndarray,
        output_csv: Path,
        title: str | None,
        fft_log: bool,
        image_log: bool,
        cmap_index: int,
        freq_min_hz: float | None,
        freq_max_hz: float | None,
        image_columns: int,
        marker_x: float,
        tickspace_hz: float | None,
        initial_peaks: list[float] | None = None,
    ) -> None:
        self.freq = np.asarray(freq, dtype=float)
        self.amplitude = np.asarray(amplitude, dtype=float)
        self.output_csv = Path(output_csv)
        self.title = title or "Click Peak Finder"
        self.fft_log = bool(fft_log)
        self.image_log = bool(image_log)
        self.cmap_index = int(cmap_index)
        self.image_columns = max(int(image_columns), 2)
        self.marker_x = float(np.clip(marker_x, 0.0, 1.0))
        self.tickspace_hz = tickspace_hz
        self.mode = "neutral"
        self.peaks: list[float] = []
        self._syncing = False
        self._status_text = None
        self._mode_text = None
        self._fft_lines: list = []
        self._image_scatter = None

        finite_mask = np.isfinite(self.freq) & np.isfinite(self.amplitude) & (self.freq > 0)
        self.freq = self.freq[finite_mask]
        self.amplitude = self.amplitude[finite_mask]
        if self.freq.size == 0:
            raise ValueError("Spectrum contains no positive finite frequencies")

        self.freq_min, self.freq_max = resolve_clipped_window(
            float(self.freq[0]),
            float(self.freq[-1]),
            freq_min_hz,
            freq_max_hz,
        )

        self.fig = plt.figure(figsize=(13.5, 7.2))
        self.ax_fft = self.fig.add_axes([0.07, 0.18, 0.43, 0.72])
        self.ax_img = self.fig.add_axes([0.56, 0.18, 0.32, 0.72])
        self.ax_save = self.fig.add_axes([0.07, 0.05, 0.11, 0.06])
        self.btn_save = Button(self.ax_save, "Save CSV")

        self._build_plot()
        if initial_peaks:
            self.peaks = [
                float(np.clip(peak, self.freq[0], self.freq[-1]))
                for peak in sorted(set(float(peak) for peak in initial_peaks))
            ]
            self._refresh_overlays()
        self._connect()

    def _build_plot(self) -> None:
        if self.fft_log:
            self.ax_fft.semilogy(self.freq, self.amplitude, linewidth=1.2, color="tab:blue")
            positive_vals = self.amplitude[self.amplitude > 0]
            if positive_vals.size > 0:
                ymin = np.percentile(positive_vals, 0.1) * 0.7
                ymax = np.max(positive_vals) * 1.3
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                    self.ax_fft.set_ylim(ymin, ymax)
        else:
            self.ax_fft.plot(self.freq, self.amplitude, linewidth=1.2, color="tab:blue")

        self.ax_fft.set_xlim(self.freq_min, self.freq_max)
        apply_major_tick_spacing(self.ax_fft, self.tickspace_hz, axis="x")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Amplitude")
        self.ax_fft.set_title(self.title)
        self.ax_fft.grid(True, alpha=0.3)

        self._draw_image()

        self._status_text = self.fig.text(
            0.21,
            0.08,
            "",
            ha="left",
            va="center",
            fontsize=10,
        )
        self._mode_text = self.fig.text(
            0.07,
            0.92,
            "",
            ha="left",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "0.5",
                "alpha": 0.9,
            },
        )
        self._set_mode("neutral")

    def _draw_image(self) -> None:
        mask = (self.freq >= self.freq_min) & (self.freq <= self.freq_max)
        freq_img = self.freq[mask]
        amp_img = self.amplitude[mask]
        if freq_img.size == 0:
            raise ValueError("No frequencies remain inside the requested display range")

        image_2d = np.tile(amp_img[:, None], (1, self.image_columns))
        eps = np.finfo(float).eps
        if self.image_log:
            image_plot = 20.0 * np.log10(image_2d + eps)
            norm = None
            color_label = "Amplitude (dB)"
        else:
            image_plot = image_2d
            norm = None
            color_label = "Amplitude"

        x_edges = np.linspace(0.0, 1.0, self.image_columns + 1)
        fallback_step = float(np.median(np.diff(freq_img))) if freq_img.size > 1 else max(1e-6, self.freq_max - self.freq_min)
        y_edges = centers_to_edges(freq_img, fallback_step=fallback_step)

        pcm = self.ax_img.pcolormesh(
            x_edges,
            y_edges,
            image_plot,
            shading="flat",
            cmap=colormap_name(self.cmap_index),
            norm=norm,
        )
        self.fig.colorbar(pcm, ax=self.ax_img, label=color_label)
        self.ax_img.set_xlim(0.0, 1.0)
        self.ax_img.set_ylim(self.freq_min, self.freq_max)
        apply_major_tick_spacing(self.ax_img, self.tickspace_hz, axis="y")
        self.ax_img.set_xlabel("Image X")
        self.ax_img.set_ylabel("Frequency (Hz)")
        self.ax_img.set_title("FFT Image")
        self.ax_img.set_xticks([])

    def _connect(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.ax_fft.callbacks.connect("xlim_changed", self._on_fft_xlim_changed)
        self.ax_img.callbacks.connect("ylim_changed", self._on_img_ylim_changed)
        self.btn_save.on_clicked(lambda _event: self._save())

    def _update_status(self, text: str) -> None:
        if self._status_text is not None:
            self._status_text.set_text(text)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if self.mode != "neutral":
            self._deactivate_toolbar_mode()
        if self._mode_text is not None:
            self._mode_text.set_text(f"Mode: {self.mode.upper()}")
        self._update_status(
            f"Mode: {self.mode.upper()} | a add | r remove | esc neutral | s save | CSV: {self.output_csv}"
        )
        print(f"Mode set to: {self.mode}")
        self.fig.canvas.draw_idle()

    def _deactivate_toolbar_mode(self) -> None:
        manager = getattr(self.fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is None:
            return

        mode = str(getattr(toolbar, "mode", "")).lower()
        if "zoom" in mode and hasattr(toolbar, "zoom"):
            toolbar.zoom()
        elif "pan" in mode and hasattr(toolbar, "pan"):
            toolbar.pan()

    def _toolbar_has_active_mode(self) -> bool:
        manager = getattr(self.fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        mode = getattr(toolbar, "mode", "")
        return bool(mode)

    def _on_fft_xlim_changed(self, ax) -> None:
        if self._syncing:
            return
        left, right = ax.get_xlim()
        self._syncing = True
        try:
            self.ax_img.set_ylim(left, right)
        finally:
            self._syncing = False

    def _on_img_ylim_changed(self, ax) -> None:
        if self._syncing:
            return
        low, high = ax.get_ylim()
        self._syncing = True
        try:
            self.ax_fft.set_xlim(low, high)
        finally:
            self._syncing = False

    def _pick_frequency(self, event) -> float | None:
        if event.inaxes is self.ax_fft:
            return None if event.xdata is None else float(event.xdata)
        if event.inaxes is self.ax_img:
            return None if event.ydata is None else float(event.ydata)
        return None

    def _removal_tolerance_hz(self) -> float:
        low, high = self.ax_fft.get_xlim()
        span = abs(high - low)
        return max(span * 0.01, 1e-9)

    def _add_peak(self, freq_hz: float) -> None:
        clamped = float(np.clip(freq_hz, self.freq[0], self.freq[-1]))
        if any(np.isclose(clamped, existing, rtol=0.0, atol=self._removal_tolerance_hz() * 0.25) for existing in self.peaks):
            return
        self.peaks.append(clamped)
        self.peaks.sort()
        self._refresh_overlays()
        self._update_status(f"Added {clamped:.6g} Hz | {len(self.peaks)} peaks selected | CSV: {self.output_csv}")

    def _remove_peak_near(self, freq_hz: float) -> None:
        if not self.peaks:
            self._update_status("No peaks selected to remove")
            return
        peaks_arr = np.asarray(self.peaks, dtype=float)
        idx = int(np.argmin(np.abs(peaks_arr - freq_hz)))
        nearest = float(peaks_arr[idx])
        if abs(nearest - freq_hz) > self._removal_tolerance_hz():
            self._update_status(f"No selected peak within {self._removal_tolerance_hz():.4g} Hz of {freq_hz:.6g} Hz")
            return
        del self.peaks[idx]
        self._refresh_overlays()
        self._update_status(f"Removed {nearest:.6g} Hz | {len(self.peaks)} peaks selected | CSV: {self.output_csv}")

    def _refresh_overlays(self) -> None:
        for line in self._fft_lines:
            line.remove()
        self._fft_lines = [
            self.ax_fft.axvline(peak, color="red", linewidth=1.2, alpha=0.9)
            for peak in self.peaks
        ]

        if self._image_scatter is not None:
            self._image_scatter.remove()
            self._image_scatter = None

        if self.peaks:
            if len(self.peaks) == 1:
                x_vals = np.array([self.marker_x], dtype=float)
            else:
                x_vals = np.linspace(0.15, 0.85, len(self.peaks), dtype=float)
            y_vals = np.asarray(self.peaks, dtype=float)
            self._image_scatter = self.ax_img.scatter(
                x_vals,
                y_vals,
                s=36,
                c="red",
                marker="o",
                edgecolors="white",
                linewidths=0.4,
                zorder=4,
            )
        self.fig.canvas.draw_idle()

    def _save(self) -> None:
        saved = save_peaks_csv(self.output_csv, self.peaks)
        self._update_status(f"Saved {len(self.peaks)} peaks to {saved}")
        print(f"Saved {len(self.peaks)} peaks to: {saved}")

    def _on_key_press(self, event) -> None:
        key = "" if event.key is None else str(event.key).lower()
        if key == "a":
            self._set_mode("neutral" if self.mode == "add" else "add")
        elif key == "r":
            self._set_mode("neutral" if self.mode == "remove" else "remove")
        elif key == "escape":
            self._set_mode("neutral")
        elif key == "s":
            self._save()

    def _on_click(self, event) -> None:
        if event.inaxes not in {self.ax_fft, self.ax_img}:
            return
        if self._toolbar_has_active_mode():
            return
        if event.button != 1:
            return
        if self.mode == "neutral":
            return

        freq_hz = self._pick_frequency(event)
        if freq_hz is None or not np.isfinite(freq_hz):
            return

        if self.mode == "add":
            self._add_peak(freq_hz)
        elif self.mode == "remove":
            self._remove_peak_near(freq_hz)


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
        spectrasave_path = resolve_existing_spectrasave_path(args.spectrasave)
        payload = load_spectrum_msgpack(spectrasave_path)
        if payload.get("artifactType") != "spectrum":
            raise ValueError("Spectrasave input was not a spectrum artifact")

        output_csv = Path(args.csv) if args.csv is not None else default_csv_path(spectrasave_path)
        initial_peaks = load_existing_peaks(output_csv)

        picker = PeakPicker(
            freq=payload["freq"],
            amplitude=payload["amplitude"],
            output_csv=output_csv,
            title=args.title or payload.get("label") or spectrasave_path.stem,
            fft_log=args.fft_log,
            image_log=args.image_log,
            cmap_index=args.cm,
            freq_min_hz=args.freq_min_hz,
            freq_max_hz=args.freq_max_hz,
            image_columns=args.image_columns,
            marker_x=args.marker_x,
            tickspace_hz=args.tickspace_hz,
            initial_peaks=initial_peaks,
        )

        if initial_peaks:
            picker._update_status(
                f"Preloaded {len(initial_peaks)} peaks from {output_csv} | "
                f"a add | r remove | esc neutral | s save"
            )
            print(f"Preloaded {len(initial_peaks)} peaks from: {output_csv}")

        if args.save is not None:
            save_path = ensure_parent_dir(args.save)
            picker.fig.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")

        plt.show()
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
