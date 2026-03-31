from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks


matplotlib.use("Agg")


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    bond_spacing_mode: str = "comoving"
    bond_index: int = 0
    sliding_len_s: float = 20.0
    components: tuple[str, ...] = ("x", "y", "a")

    # Hit finding from spectrogram broadband energy
    peak_finder_mode: str = "all"
    peak_height_frac: float = 0.10
    peak_prominence_frac: float = 0.05
    peak_min_distance_s: float = 0.5
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)

    # Segment selection between hits
    min_segment_len_s: float = 25.0
    ignore_beginning_len_s: float = 0.0
    ignore_end_len_s: float = 4.0
    ignore_first_segment: bool = True
    ignore_last_segment: bool = True

    # Target FFT peak analysis
    peak_of_interest_hz: float = 3.35
    integration_window_hz: float = 0.05


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")


REPO_ROOT = add_repo_root_to_path()

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.io import split_dataset_component
from analysis.tools.signal import compute_complex_spectrogram, compute_one_sided_fft, preprocess_signal


class PlotWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 1

    def reset(self) -> None:
        for pattern in ("[0-9][0-9][0-9]_*.png", "[0-9][0-9][0-9]_*.pdf", "summary.txt"):
            for path in self.output_dir.glob(pattern):
                path.unlink()

    def save(self, fig: plt.Figure, slug: str) -> Path:
        path = self.output_dir / f"{self.counter:03d}_{slug}.png"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"[saved] {path.name}")
        self.counter += 1
        return path


def summarize_array(name: str, values: np.ndarray) -> str:
    return f"{name}: [{', '.join(f'{x:.6g}' for x in values)}]"


def save_summary(output_dir: Path, text: str) -> Path:
    path = output_dir / "summary.txt"
    path.write_text(text)
    return path


def load_base_dataset(cfg: Config):
    ds = load_bond_signal_dataset(
        dataset=cfg.dataset,
        bond_spacing_mode=cfg.bond_spacing_mode,
    )
    return ds


def plot_timeseries(writer: PlotWriter, frame_times_s: np.ndarray, signal_matrix: np.ndarray, pair_labels: list[str], cfg: Config) -> Path:
    n_frames, n_bonds = signal_matrix.shape
    fig, axes = plt.subplots(
        n_bonds,
        1,
        figsize=(12, max(2.2 * n_bonds, 4)),
        sharex=True,
        constrained_layout=True,
    )
    if n_bonds == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(frame_times_s, signal_matrix[:, i], lw=1.0)
        label = pair_labels[i] if i < len(pair_labels) else f"bond {i}"
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{cfg.dataset} bond spacing signals ({cfg.bond_spacing_mode})")
    return writer.save(fig, "timeseries")


def build_x_component_analysis(cfg: Config):
    base_dataset, _ = split_dataset_component(cfg.dataset)
    ds_x = load_bond_signal_dataset(
        dataset=f"{base_dataset}_x",
        bond_spacing_mode=cfg.bond_spacing_mode,
        component="x",
    )
    y = ds_x.signal_matrix[:, cfg.bond_index]
    processed, err = preprocess_signal(ds_x.frame_times_s, y, longest=False, handlenan=False)
    if processed is None:
        raise ValueError(f"Preprocess failed: {err}")

    spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.sliding_len_s)
    if spec is None:
        raise ValueError("Spectrogram window too short")

    s_db = 20.0 * np.log10(np.abs(spec.S_complex) + np.finfo(float).eps)
    t_global = spec.t + processed.t[0]
    broadband_energy = np.sum(np.abs(spec.S_complex), axis=0)
    return base_dataset, ds_x, processed, spec, t_global, s_db, broadband_energy


def plot_x_spectrogram_and_energy(writer: PlotWriter, base_dataset: str, ds_x, processed, spec, t_global: np.ndarray, s_db: np.ndarray, broadband_energy: np.ndarray, pair_labels: list[str], cfg: Config) -> Path:
    fig, (ax_spec, ax_energy) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    pcm = ax_spec.pcolormesh(t_global, spec.f, s_db, shading="auto", cmap="turbo", rasterized=True)
    fig.colorbar(pcm, ax=ax_spec, label="dB")
    label = pair_labels[cfg.bond_index] if cfg.bond_index < len(pair_labels) else f"bond {cfg.bond_index}"
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_title(f"X component spectrogram | bond {cfg.bond_index} ({label})")
    ax_spec.set_ylim(0.01, processed.nyquist)

    ax_energy.plot(t_global, broadband_energy, lw=1.5, color="C1")
    ax_energy.set_ylabel("broadband energy")
    ax_energy.set_xlabel("time (s)")
    ax_energy.grid(alpha=0.3)
    ax_energy.set_title("Integrated broadband energy (sum across frequencies)")

    fig.suptitle(f"{base_dataset} | mode={cfg.bond_spacing_mode}")
    return writer.save(fig, "x_spectrogram_broadband_energy")


def detect_peak_times(t_global: np.ndarray, broadband_energy: np.ndarray, processed, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    if cfg.peak_finder_mode == "all":
        peak_indices, _ = find_peaks(broadband_energy)
    elif cfg.peak_finder_mode == "thresholded":
        height = float(np.max(broadband_energy) * cfg.peak_height_frac)
        prominence = float(np.max(broadband_energy) * cfg.peak_prominence_frac)
        distance = max(1, int(round(cfg.peak_min_distance_s * processed.Fs)))
        peak_indices, _ = find_peaks(
            broadband_energy,
            height=height,
            distance=distance,
            prominence=prominence,
        )
    else:
        raise ValueError(f"Unsupported peak_finder_mode: {cfg.peak_finder_mode}")

    peak_times = t_global[peak_indices]
    if cfg.manual_peak_times_s:
        peak_times = np.sort(np.unique(np.concatenate([peak_times, np.asarray(cfg.manual_peak_times_s, dtype=float)])))
    return peak_indices, peak_times


def plot_peak_detection(writer: PlotWriter, base_dataset: str, spec, s_db: np.ndarray, t_global: np.ndarray, broadband_energy: np.ndarray, peak_indices: np.ndarray, peak_times: np.ndarray, processed, pair_labels: list[str], cfg: Config) -> Path:
    fig, (ax_spec, ax_energy) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    pcm = ax_spec.pcolormesh(t_global, spec.f, s_db, shading="auto", cmap="turbo", rasterized=True)
    fig.colorbar(pcm, ax=ax_spec, label="dB")
    label = pair_labels[cfg.bond_index] if cfg.bond_index < len(pair_labels) else f"bond {cfg.bond_index}"
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_title(f"X component spectrogram | bond {cfg.bond_index} ({label})")
    ax_spec.set_ylim(0.01, processed.nyquist)

    for t_peak in peak_times:
        ax_spec.axvline(t_peak, color="red", linestyle="--", linewidth=1.2, alpha=0.65)

    ax_energy.plot(t_global, broadband_energy, lw=1.5, color="C1", label="broadband energy")
    ax_energy.plot(t_global[peak_indices], broadband_energy[peak_indices], "r^", markersize=7, label=f"peaks (n={len(peak_times)})")
    for t_peak in peak_times:
        ax_energy.axvline(t_peak, color="red", linestyle="--", linewidth=1.2, alpha=0.65)
    ax_energy.set_ylabel("broadband energy")
    ax_energy.set_xlabel("time (s)")
    ax_energy.grid(alpha=0.3)
    ax_energy.legend(loc="upper right")
    ax_energy.set_title("Integrated broadband energy with detected peaks")

    fig.suptitle(f"{base_dataset} | mode={cfg.bond_spacing_mode} | {len(peak_times)} peaks used")
    return writer.save(fig, "peak_detection")


def build_enabled_regions(ds_x, t_global: np.ndarray, broadband_energy: np.ndarray, peak_times: np.ndarray, cfg: Config):
    peak_times_sorted = np.sort(np.asarray(peak_times, dtype=float))
    segment_edges = np.concatenate(([float(t_global[0])], peak_times_sorted, [float(t_global[-1])]))
    durations = np.diff(segment_edges)
    usable = durations >= float(cfg.min_segment_len_s)
    if usable.size > 0 and cfg.ignore_first_segment:
        usable[0] = False
    if usable.size > 0 and cfg.ignore_last_segment:
        usable[-1] = False

    t_pos = np.asarray(ds_x.frame_times_s, dtype=float)
    y_pos = np.asarray(ds_x.signal_matrix[:, cfg.bond_index], dtype=float)
    enabled_region_bounds: list[tuple[float, float]] = []
    enabled_region_position_time_arrays: list[np.ndarray] = []

    for i in range(len(durations)):
        if not usable[i]:
            continue
        left = float(segment_edges[i] + cfg.ignore_beginning_len_s)
        right = float(segment_edges[i + 1] - cfg.ignore_end_len_s)
        if right <= left:
            continue
        enabled_region_bounds.append((left, right))
        mask = (t_pos >= left) & (t_pos <= right) & np.isfinite(y_pos)
        arr = np.column_stack([t_pos[mask], y_pos[mask]])
        if arr.size > 0:
            enabled_region_position_time_arrays.append(arr)

    return segment_edges, usable, enabled_region_bounds, enabled_region_position_time_arrays, t_pos, y_pos


def plot_enabled_regions(writer: PlotWriter, base_dataset: str, spec, s_db: np.ndarray, t_global: np.ndarray, broadband_energy: np.ndarray, peak_times: np.ndarray, processed, t_pos: np.ndarray, y_pos: np.ndarray, enabled_region_bounds: list[tuple[float, float]], pair_labels: list[str], cfg: Config) -> Path:
    fig, (ax_spec, ax_ts, ax_energy) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    pcm = ax_spec.pcolormesh(t_global, spec.f, s_db, shading="auto", cmap="turbo", rasterized=True)
    fig.colorbar(pcm, ax=ax_spec, label="dB")
    label = pair_labels[cfg.bond_index] if cfg.bond_index < len(pair_labels) else f"bond {cfg.bond_index}"
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_ylim(0.01, processed.nyquist)
    ax_spec.set_title(f"X spectrogram | bond {cfg.bond_index} ({label})")

    ax_ts.plot(t_pos, y_pos, lw=1.0, color="tab:blue")
    ax_ts.set_ylabel("position")
    ax_ts.set_title("Timeseries (positions)")
    ax_ts.grid(alpha=0.25)

    ax_energy.plot(t_global, broadband_energy, lw=1.4, color="tab:orange")
    ax_energy.set_ylabel("broadband energy")
    ax_energy.set_xlabel("time (s)")
    ax_energy.set_title("Integrated broadband energy")
    ax_energy.grid(alpha=0.25)

    for t_peak in peak_times:
        ax_spec.axvline(t_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.45)
        ax_ts.axvline(t_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.35)
        ax_energy.axvline(t_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.35)

    for left, right in enabled_region_bounds:
        ax_ts.axvspan(left, right, color="limegreen", alpha=0.18, linewidth=0)
        ax_energy.axvspan(left, right, color="limegreen", alpha=0.18, linewidth=0)

    fig.suptitle(f"{base_dataset} | enabled regions={len(enabled_region_bounds)} | mode={cfg.bond_spacing_mode}")
    return writer.save(fig, "enabled_regions")


def compute_region_ffts(enabled_region_position_time_arrays: list[np.ndarray]):
    fft_results = []
    valid_region_indices = []
    processed_regions = []

    for region_idx, arr in enumerate(enabled_region_position_time_arrays):
        t_region = np.asarray(arr[:, 0], dtype=float)
        y_region = np.asarray(arr[:, 1], dtype=float)
        processed_region, err = preprocess_signal(t_region, y_region, longest=False, handlenan=False)
        if processed_region is None:
            print(f"Skipping region {region_idx}: preprocess failed ({err})")
            continue
        fft_results.append(compute_one_sided_fft(processed_region.y, processed_region.dt))
        valid_region_indices.append(region_idx)
        processed_regions.append(processed_region)

    if not fft_results:
        raise ValueError("No enabled regions produced valid FFTs")

    return fft_results, valid_region_indices, processed_regions


def plot_region_ffts(writer: PlotWriter, base_dataset: str, fft_results, valid_region_indices: list[int], enabled_region_bounds: list[tuple[float, float]]) -> Path:
    n_regions = len(fft_results)
    fig, axes = plt.subplots(
        n_regions,
        1,
        figsize=(12, max(2.4 * n_regions, 4)),
        sharex=True,
        constrained_layout=True,
    )
    if n_regions == 1:
        axes = [axes]

    for ax, region_idx, fft_region in zip(axes, valid_region_indices, fft_results):
        amp = np.maximum(np.asarray(fft_region.amplitude, dtype=float), np.finfo(float).eps)
        ax.semilogy(fft_region.freq, amp, lw=1.2, color="tab:purple")
        left, right = enabled_region_bounds[region_idx]
        ax.set_title(f"Region {region_idx}: t in [{left:.2f}, {right:.2f}] s")
        ax.set_ylabel("FFT amp")
        ax.grid(alpha=0.25, which="both")

    axes[-1].set_xlabel("frequency (Hz)")
    fig.suptitle(f"{base_dataset} | enabled-region FFT amplitudes (log scale)")
    return writer.save(fig, "region_ffts")


def compute_peak_scaling_metrics(fft_results, valid_region_indices: list[int], enabled_region_bounds: list[tuple[float, float]], t_global: np.ndarray, broadband_energy: np.ndarray, enabled_region_position_time_arrays: list[np.ndarray], cfg: Config):
    peak_amplitudes = []
    region_broadband_energy = []
    used_bin_counts = []
    used_freq_ranges = []
    region_mean_abs = []
    region_proc_series = []

    for fft_region, region_idx in zip(fft_results, valid_region_indices):
        f = np.asarray(fft_region.freq, dtype=float)
        amp = np.asarray(fft_region.amplitude, dtype=float)
        in_window = np.where(np.abs(f - cfg.peak_of_interest_hz) <= cfg.integration_window_hz)[0]
        if in_window.size < 3:
            idx_use = np.sort(np.argsort(np.abs(f - cfg.peak_of_interest_hz))[:3])
            warnings.warn(
                f"Region {region_idx}: only {in_window.size} bins inside ±{cfg.integration_window_hz} Hz around "
                f"{cfg.peak_of_interest_hz} Hz; using nearest {idx_use.size} bins instead."
            )
        else:
            idx_use = in_window

        peak_amplitudes.append(float(np.mean(amp[idx_use])))
        used_bin_counts.append(int(idx_use.size))
        used_freq_ranges.append((float(f[idx_use[0]]), float(f[idx_use[-1]])))

        left, right = enabled_region_bounds[region_idx]
        mask_e = (t_global >= left) & (t_global <= right)
        region_broadband_energy.append(float(np.mean(broadband_energy[mask_e])) if np.any(mask_e) else np.nan)

        arr = enabled_region_position_time_arrays[region_idx]
        t_region = np.asarray(arr[:, 0], dtype=float)
        y_raw = np.asarray(arr[:, 1], dtype=float)
        finite = np.isfinite(t_region) & np.isfinite(y_raw)
        t_region = t_region[finite]
        y_raw = y_raw[finite]
        if y_raw.size < 3:
            region_mean_abs.append(np.nan)
            region_proc_series.append((t_region, y_raw, np.full_like(y_raw, np.nan)))
            continue
        y_centered = y_raw - np.mean(y_raw)
        y_proc = sp_signal.detrend(y_centered, type="linear")
        region_mean_abs.append(float(np.mean(np.abs(y_proc))))
        region_proc_series.append((t_region, y_raw, y_proc))

    return {
        "peak_amplitudes": np.asarray(peak_amplitudes, dtype=float),
        "region_broadband_energy": np.asarray(region_broadband_energy, dtype=float),
        "used_bin_counts": np.asarray(used_bin_counts, dtype=int),
        "used_freq_ranges": used_freq_ranges,
        "region_mean_abs": np.asarray(region_mean_abs, dtype=float),
        "region_proc_series": region_proc_series,
    }


def plot_peak_vs_broadband(writer: PlotWriter, peak_amplitudes: np.ndarray, region_broadband_energy: np.ndarray, valid_region_indices: list[int], cfg: Config) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(region_broadband_energy, peak_amplitudes, s=60, color="tab:purple", alpha=0.9)
    for x, y, ridx in zip(region_broadband_energy, peak_amplitudes, valid_region_indices):
        ax.annotate(str(ridx), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.set_xlabel("region broadband energy (mean)")
    ax.set_ylabel(f"mean FFT amplitude near {cfg.peak_of_interest_hz:.4g} Hz")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Region FFT amplitude vs broadband energy")
    return writer.save(fig, "peak_vs_broadband_energy")


def plot_fft_bands(writer: PlotWriter, fft_results, valid_region_indices: list[int], enabled_region_bounds: list[tuple[float, float]], peak_amplitudes: np.ndarray, used_bin_counts: np.ndarray, used_freq_ranges: list[tuple[float, float]], cfg: Config) -> Path:
    n_regions = len(fft_results)
    fig, axes = plt.subplots(
        n_regions,
        1,
        figsize=(12, max(2.6 * n_regions, 4)),
        sharex=True,
        constrained_layout=True,
    )
    if n_regions == 1:
        axes = [axes]

    for ax, fft_region, region_idx, mean_amp, n_bins, (f_lo, f_hi) in zip(
        axes,
        fft_results,
        valid_region_indices,
        peak_amplitudes,
        used_bin_counts,
        used_freq_ranges,
    ):
        f = np.asarray(fft_region.freq, dtype=float)
        amp = np.maximum(np.asarray(fft_region.amplitude, dtype=float), np.finfo(float).eps)
        ax.semilogy(f, amp, lw=1.2, color="tab:purple")
        in_band = (f >= f_lo) & (f <= f_hi)
        if np.any(in_band):
            ax.fill_between(f[in_band], np.full(np.sum(in_band), np.finfo(float).eps), amp[in_band], color="tab:orange", alpha=0.25, label=f"integrated band ({n_bins} bins)")
        ax.axvline(cfg.peak_of_interest_hz, color="tab:red", linestyle="--", lw=1.0, alpha=0.7)
        left, right = enabled_region_bounds[region_idx]
        ax.set_title(f"Region {region_idx}: t in [{left:.2f}, {right:.2f}] s")
        ax.text(
            0.99,
            0.96,
            f"mean={mean_amp:.3e}\nrange=[{f_lo:.3f}, {f_hi:.3f}] Hz",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.6"),
        )
        ax.set_ylabel("FFT amp")
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="lower left")

    axes[-1].set_xlabel("frequency (Hz)")
    fig.suptitle(f"Per-region FFTs with integrated band around {cfg.peak_of_interest_hz:.4g} Hz")
    return writer.save(fig, "region_fft_peak_bands")


def plot_peak_vs_amplitude(writer: PlotWriter, region_mean_abs: np.ndarray, peak_amplitudes: np.ndarray, valid_region_indices: list[int], cfg: Config) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.scatter(region_mean_abs, peak_amplitudes, s=60, color="tab:green", alpha=0.9)
    for x, y, ridx in zip(region_mean_abs, peak_amplitudes, valid_region_indices):
        ax.annotate(str(ridx), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax.set_xlabel("mean |x| of centered+detrended region timeseries")
    ax.set_ylabel(f"mean FFT amplitude near {cfg.peak_of_interest_hz:.4g} Hz")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Region FFT amplitude vs mean |x| (detrended timeseries)")
    return writer.save(fig, "peak_vs_region_amplitude")


def plot_region_timeseries_and_fft(writer: PlotWriter, fft_results, valid_region_indices: list[int], enabled_region_bounds: list[tuple[float, float]], metrics: dict, cfg: Config) -> Path:
    n_regions = len(fft_results)
    fig, axes = plt.subplots(
        n_regions,
        2,
        figsize=(15, max(2.8 * n_regions, 4.5)),
        constrained_layout=True,
    )
    if n_regions == 1:
        axes = np.asarray([axes])

    for (ax_ts, ax_fft), fft_region, region_idx, mean_abs_val, mean_amp, n_bins, (f_lo, f_hi), series in zip(
        axes,
        fft_results,
        valid_region_indices,
        metrics["region_mean_abs"],
        metrics["peak_amplitudes"],
        metrics["used_bin_counts"],
        metrics["used_freq_ranges"],
        metrics["region_proc_series"],
    ):
        t_region, y_raw, y_proc = series
        ax_ts.plot(t_region, y_proc, lw=1.2, color="tab:green", label="centered+detrended")
        ax_ts.axhline(0.0, color="0.35", lw=0.8, alpha=0.6)
        left, right = enabled_region_bounds[region_idx]
        ax_ts.set_title(f"Region {region_idx}: t in [{left:.2f}, {right:.2f}] s")
        ax_ts.set_ylabel("position")
        ax_ts.grid(alpha=0.25)
        ax_ts.legend(loc="upper right")
        ax_ts.text(
            0.99,
            0.96,
            f"mean|x|={mean_abs_val:.3e}",
            transform=ax_ts.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.6"),
        )

        f = np.asarray(fft_region.freq, dtype=float)
        amp = np.maximum(np.asarray(fft_region.amplitude, dtype=float), np.finfo(float).eps)
        ax_fft.semilogy(f, amp, lw=1.2, color="tab:purple")
        in_band = (f >= f_lo) & (f <= f_hi)
        if np.any(in_band):
            ax_fft.fill_between(f[in_band], np.full(np.sum(in_band), np.finfo(float).eps), amp[in_band], color="tab:orange", alpha=0.25, label=f"integrated band ({n_bins} bins)")
        ax_fft.axvline(cfg.peak_of_interest_hz, color="tab:red", linestyle="--", lw=1.0, alpha=0.7)
        ax_fft.set_ylabel("FFT amp")
        ax_fft.grid(alpha=0.25, which="both")
        ax_fft.legend(loc="lower left")
        ax_fft.text(
            0.99,
            0.96,
            f"mean FFT={mean_amp:.3e}\nrange=[{f_lo:.3f}, {f_hi:.3f}] Hz",
            transform=ax_fft.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.6"),
        )

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("frequency (Hz)")
    fig.suptitle(f"Per-region timeseries and FFT near {cfg.peak_of_interest_hz:.4g} Hz", fontsize=12)
    return writer.save(fig, "region_timeseries_and_fft")


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    ds = load_base_dataset(CONFIG)
    timeseries_path = plot_timeseries(writer, ds.frame_times_s, ds.signal_matrix, ds.pair_labels, CONFIG)

    base_dataset, ds_x, processed, spec, t_global, s_db, broadband_energy = build_x_component_analysis(CONFIG)

    peak_indices, peak_times = detect_peak_times(t_global, broadband_energy, processed, CONFIG)
    peak_plot_path = plot_peak_detection(writer, base_dataset, spec, s_db, t_global, broadband_energy, peak_indices, peak_times, processed, ds.pair_labels, CONFIG)

    _, usable, enabled_region_bounds, enabled_region_position_time_arrays, t_pos, y_pos = build_enabled_regions(ds_x, t_global, broadband_energy, peak_times, CONFIG)
    enabled_regions_path = plot_enabled_regions(writer, base_dataset, spec, s_db, t_global, broadband_energy, peak_times, processed, t_pos, y_pos, enabled_region_bounds, ds.pair_labels, CONFIG)

    fft_results, valid_region_indices, _processed_regions = compute_region_ffts(enabled_region_position_time_arrays)
    region_ffts_path = plot_region_ffts(writer, base_dataset, fft_results, valid_region_indices, enabled_region_bounds)

    metrics = compute_peak_scaling_metrics(
        fft_results,
        valid_region_indices,
        enabled_region_bounds,
        t_global,
        broadband_energy,
        enabled_region_position_time_arrays,
        CONFIG,
    )
    fft_bands_path = plot_fft_bands(writer, fft_results, valid_region_indices, enabled_region_bounds, metrics["peak_amplitudes"], metrics["used_bin_counts"], metrics["used_freq_ranges"], CONFIG)
    peak_amp_path = plot_peak_vs_amplitude(writer, metrics["region_mean_abs"], metrics["peak_amplitudes"], valid_region_indices, CONFIG)
    region_combo_path = plot_region_timeseries_and_fft(writer, fft_results, valid_region_indices, enabled_region_bounds, metrics, CONFIG)

    summary_lines = [
        f"repo_root: {REPO_ROOT}",
        f"dataset: {CONFIG.dataset}",
        f"bond_spacing_mode: {CONFIG.bond_spacing_mode}",
        f"bond_index: {CONFIG.bond_index}",
        f"peak_finder_mode: {CONFIG.peak_finder_mode}",
        f"peak_of_interest_hz: {CONFIG.peak_of_interest_hz}",
        f"integration_window_hz: {CONFIG.integration_window_hz}",
        f"detected_peak_times_s: {np.array2string(peak_times, precision=3)}",
        f"usable_segment_flags: {np.array2string(usable.astype(int))}",
        f"enabled_region_bounds: {enabled_region_bounds}",
        summarize_array("peak_amplitudes", metrics["peak_amplitudes"]),
        summarize_array("region_mean_abs", metrics["region_mean_abs"]),
        summarize_array("used_bin_counts", metrics["used_bin_counts"].astype(float)),
        "",
        "saved_plots:",
        str(timeseries_path),
        str(peak_plot_path),
        str(enabled_regions_path),
        str(region_ffts_path),
        str(fft_bands_path),
        str(peak_amp_path),
        str(region_combo_path),
    ]
    summary_path = save_summary(OUTPUT_DIR, "\n".join(summary_lines) + "\n")

    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
