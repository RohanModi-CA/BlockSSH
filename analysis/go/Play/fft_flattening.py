#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = ROOT / "analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from analysis.viz import see_fft_xya


DEFAULT_TRACK_DATA_ROOT = ROOT / "track" / "data"


@dataclass(frozen=True)
class FlattenedSpectrumResult:
    dataset: str
    component: str
    freq_hz: np.ndarray
    amplitude: np.ndarray
    baseline: np.ndarray
    baseline_smooth: np.ndarray
    residual: np.ndarray
    transfer: np.ndarray
    flattened: np.ndarray
    reference_level: float


def load_component_averages(
    *,
    dataset: str,
    track_data_root: Path = DEFAULT_TRACK_DATA_ROOT,
    bond_spacing_mode: str = "comoving",
    use_welch: bool = True,
    longest: bool = False,
    handlenan: bool = False,
    timeseriesnorm: bool = True,
    welch_len_s: float = 100.0,
    welch_overlap: float = 0.5,
    sliding_len_s: float = 20.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    args = SimpleNamespace(
        dataset=dataset,
        track2=None,
        track_data_root=str(track_data_root),
        bond_spacing_mode=bond_spacing_mode,
        disable_component=[],
        disable=[],
        welch=use_welch,
        longest=longest,
        handlenan=handlenan,
        timeseriesnorm=timeseriesnorm,
        welch_len_s=welch_len_s,
        welch_overlap=welch_overlap,
        sliding_len_s=sliding_len_s,
    )
    component_results, component_track2 = see_fft_xya._load_component_results(args)
    averages: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for component, results in component_results.items():
        dataset_name = component_track2[component].dataset_name
        averaged = see_fft_xya._average_result_from_pair_results(
            results,
            dataset_name=dataset_name,
            use_welch=use_welch,
        )
        averages[component] = (averaged.freq_grid, averaged.avg_amp)
    return averages


def rolling_quantile_baseline(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    *,
    quantile: float = 0.15,
    envelope_hz: float = 1.5,
    smooth_hz: float = 1.2,
) -> np.ndarray:
    step = float(np.median(np.diff(freq_hz)))
    half_window = max(1, int(round(envelope_hz / step / 2.0)))
    envelope = np.empty_like(amplitude)
    for idx in range(amplitude.size):
        lo = max(0, idx - half_window)
        hi = min(amplitude.size, idx + half_window + 1)
        envelope[idx] = np.quantile(amplitude[lo:hi], quantile)

    smooth_points = max(5, int(round(smooth_hz / step)))
    if smooth_points % 2 == 0:
        smooth_points += 1
    baseline = savgol_filter(envelope, smooth_points, polyorder=2, mode="interp")
    return np.minimum(baseline, amplitude)


def smooth_log_envelope(
    freq_hz: np.ndarray,
    values: np.ndarray,
    *,
    smooth_hz: float = 4.0,
) -> np.ndarray:
    step = float(np.median(np.diff(freq_hz)))
    smooth_points = max(7, int(round(smooth_hz / step)))
    if smooth_points % 2 == 0:
        smooth_points += 1
    log_values = np.log(np.maximum(values, np.finfo(float).tiny))
    smoothed = savgol_filter(log_values, smooth_points, polyorder=2, mode="interp")
    return np.exp(smoothed)


def flattening_transfer(
    freq_hz: np.ndarray,
    baseline: np.ndarray,
    *,
    reference_band: tuple[float, float] = (20.0, 30.0),
    response_smooth_hz: float = 4.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    smooth_baseline = smooth_log_envelope(freq_hz, baseline, smooth_hz=response_smooth_hz)
    mask = (freq_hz >= reference_band[0]) & (freq_hz <= reference_band[1])
    if not np.any(mask):
        raise ValueError("Reference band does not overlap frequency grid")
    reference_level = float(
        np.exp(np.mean(np.log(np.maximum(smooth_baseline[mask], np.finfo(float).tiny))))
    )
    transfer = reference_level / np.maximum(smooth_baseline, np.finfo(float).tiny)
    return transfer, reference_level, smooth_baseline


def compute_flattened_component_spectra(
    *,
    dataset: str,
    track_data_root: Path = DEFAULT_TRACK_DATA_ROOT,
    bond_spacing_mode: str = "comoving",
    components: tuple[str, ...] = ("x", "y", "a"),
    baseline_quantile: float = 0.15,
    baseline_envelope_hz: float = 1.5,
    baseline_smooth_hz: float = 1.2,
    response_smooth_hz: float = 4.0,
    reference_band: tuple[float, float] = (20.0, 30.0),
    use_welch: bool = True,
) -> dict[str, FlattenedSpectrumResult]:
    spectra = load_component_averages(
        dataset=dataset,
        track_data_root=track_data_root,
        bond_spacing_mode=bond_spacing_mode,
        use_welch=use_welch,
    )
    results: dict[str, FlattenedSpectrumResult] = {}
    for component in components:
        if component not in spectra:
            continue
        freq_hz, amplitude = spectra[component]
        baseline = rolling_quantile_baseline(
            freq_hz,
            amplitude,
            quantile=baseline_quantile,
            envelope_hz=baseline_envelope_hz,
            smooth_hz=baseline_smooth_hz,
        )
        transfer, reference_level, baseline_smooth = flattening_transfer(
            freq_hz,
            baseline,
            reference_band=reference_band,
            response_smooth_hz=response_smooth_hz,
        )
        results[component] = FlattenedSpectrumResult(
            dataset=dataset,
            component=component,
            freq_hz=freq_hz,
            amplitude=amplitude,
            baseline=baseline,
            baseline_smooth=baseline_smooth,
            residual=amplitude - baseline,
            transfer=transfer,
            flattened=amplitude * transfer,
            reference_level=reference_level,
        )
    return results


def load_reference_peaks(peaks_path: Path | None = None) -> np.ndarray:
    if peaks_path is None:
        peaks_path = Path(__file__).resolve().parents[1] / "realpeak0681.csv"
    return np.loadtxt(peaks_path, delimiter=",", ndmin=1)


def save_flattened_csv(path: Path, results: dict[str, FlattenedSpectrumResult]) -> Path:
    if not results:
        raise ValueError("No flattened spectra were available to save")
    first = next(iter(results.values()))
    fields = ["freq_hz"]
    for component in ("x", "y", "a"):
        if component in results:
            fields.extend(
                [
                    f"{component}_amplitude",
                    f"{component}_baseline",
                    f"{component}_baseline_smooth",
                    f"{component}_residual",
                    f"{component}_transfer",
                    f"{component}_flattened",
                ]
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for idx, hz in enumerate(first.freq_hz):
            row: list[float] = [float(hz)]
            for component in ("x", "y", "a"):
                if component not in results:
                    continue
                result = results[component]
                row.extend(
                    [
                        float(result.amplitude[idx]),
                        float(result.baseline[idx]),
                        float(result.baseline_smooth[idx]),
                        float(result.residual[idx]),
                        float(result.transfer[idx]),
                        float(result.flattened[idx]),
                    ]
                )
            writer.writerow(row)
    return path


def plot_flattened_results(
    results: dict[str, FlattenedSpectrumResult],
    *,
    peak_freqs: np.ndarray | None = None,
    title: str | None = None,
):
    components = [component for component in ("x", "y", "a") if component in results]
    if not components:
        raise ValueError("No flattened spectra were available to plot")

    fig, axes = plt.subplots(len(components), 2, figsize=(14, 10), sharex="col")
    if len(components) == 1:
        axes = np.array([axes])

    colors = {"x": "#1f77b4", "y": "#2ca02c", "a": "#d62728"}
    for row_axes, component in zip(axes, components):
        result = results[component]
        ax = row_axes[0]
        ax_tf = row_axes[1]

        ax.plot(result.freq_hz, result.amplitude, color=colors[component], lw=1.2, label=f"{component} average Welch amplitude")
        ax.plot(result.freq_hz, result.baseline, color="black", lw=2.0, label="baseline")
        ax.plot(result.freq_hz, result.baseline_smooth, color="#ff7f0e", lw=1.6, label="smoothed response envelope")
        ax.fill_between(
            result.freq_hz,
            result.baseline,
            result.amplitude,
            where=result.amplitude >= result.baseline,
            color=colors[component],
            alpha=0.18,
        )
        if peak_freqs is not None:
            for peak in peak_freqs:
                if result.freq_hz[0] <= peak <= result.freq_hz[-1]:
                    ax.axvline(peak, color="0.85", lw=0.7, zorder=0)
        ax.set_yscale("log")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{result.dataset} component {component}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", frameon=False)

        inset = ax.inset_axes([0.66, 0.13, 0.31, 0.3])
        inset.plot(result.freq_hz, result.residual, color=colors[component], lw=0.9)
        inset.set_xlim(0.0, 24.0)
        inset.set_title("residual", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(True, alpha=0.2)

        ax_tf.plot(result.freq_hz, result.transfer, color="#9467bd", lw=1.4, label="flattening transfer H(f)")
        ax_tf.plot(result.freq_hz, result.flattened, color=colors[component], lw=1.0, alpha=0.9, label="flattened spectrum")
        ax_tf.axhline(result.reference_level, color="0.6", lw=1.0, ls="--")
        if peak_freqs is not None:
            for peak in peak_freqs:
                if result.freq_hz[0] <= peak <= result.freq_hz[-1]:
                    ax_tf.axvline(peak, color="0.9", lw=0.7, zorder=0)
        ax_tf.set_yscale("log")
        ax_tf.set_ylabel("Gain / Flattened amp.")
        ax_tf.set_title(f"{component} transfer and flattened spectrum")
        ax_tf.grid(True, which="both", alpha=0.25)
        ax_tf.legend(loc="upper right", frameon=False)

    axes[-1, 0].set_xlim(0.0, 30.0)
    axes[-1, 1].set_xlim(0.0, 30.0)
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(title or f"{first_dataset(results)} flattened spectral response", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def first_dataset(results: dict[str, FlattenedSpectrumResult]) -> str:
    return next(iter(results.values())).dataset


def save_plot(path: Path, fig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute flattening transfer functions and flattened average spectra.")
    parser.add_argument("dataset", nargs="?", default="IMG_0681_rot270")
    parser.add_argument("--track-data-root", default=str(DEFAULT_TRACK_DATA_ROOT))
    parser.add_argument("--bond-spacing-mode", default="comoving")
    parser.add_argument("--components", nargs="+", default=["x", "y", "a"])
    parser.add_argument("--save-csv", default=None, help="Optional output CSV path.")
    parser.add_argument("--save-plot", default=None, help="Optional output plot path.")
    parser.add_argument("--show-plot", action="store_true", help="Display the plot interactively.")
    parser.add_argument("--no-peaks", action="store_true", help="Do not overlay reference peaks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    results = compute_flattened_component_spectra(
        dataset=args.dataset,
        track_data_root=Path(args.track_data_root),
        bond_spacing_mode=args.bond_spacing_mode,
        components=tuple(args.components),
    )

    if args.save_csv:
        saved_csv = save_flattened_csv(Path(args.save_csv), results)
        print(f"Wrote {saved_csv}")

    fig = None
    if args.save_plot or args.show_plot:
        peak_freqs = None if args.no_peaks else load_reference_peaks()
        fig = plot_flattened_results(results, peak_freqs=peak_freqs)
        if args.save_plot:
            saved_plot = save_plot(Path(args.save_plot), fig)
            print(f"Wrote {saved_plot}")
        if args.show_plot:
            plt.show()
        else:
            plt.close(fig)

    for component, result in results.items():
        print(f"{component} reference flattened-baseline level over 20-30 Hz: {result.reference_level:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
