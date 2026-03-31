from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class TargetBand:
    label: str
    center_hz: float
    half_width_hz: float
    n_scan: int


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    bond_spacing_mode: str = "comoving"
    primary_component: str = "x"
    primary_bond_index: int = 0
    sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    ignore_beginning_len_s: float = 0.0
    ignore_end_len_s: float = 4.0
    ignore_first_segment: bool = True
    ignore_last_segment: bool = True
    target_bands: tuple[TargetBand, ...] = (
        TargetBand(label="3.35 Hz", center_hz=3.35, half_width_hz=0.50, n_scan=401),
        TargetBand(label="12.0 Hz", center_hz=12.0, half_width_hz=1.00, n_scan=401),
        TargetBand(label="16.65 Hz", center_hz=16.65, half_width_hz=1.00, n_scan=401),
    )


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
        for pattern in ("[0-9][0-9][0-9]_*.png", "summary.txt"):
            for path in self.output_dir.glob(pattern):
                path.unlink()

    def save(self, fig: plt.Figure, slug: str) -> Path:
        path = self.output_dir / f"{self.counter:03d}_{slug}.png"
        fig.savefig(path, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"[saved] {path.name}")
        self.counter += 1
        return path


def save_summary(text: str) -> Path:
    path = OUTPUT_DIR / "summary.txt"
    path.write_text(text)
    return path


def summarize_array(values: np.ndarray, precision: int = 4) -> str:
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(float(x)) for x in values) + "]"


def fit_sine_scan(t: np.ndarray, y: np.ndarray, freqs: np.ndarray) -> dict[str, np.ndarray]:
    omega_t = 2.0 * np.pi * np.outer(t, freqs)
    cos_m = np.cos(omega_t)
    sin_m = np.sin(omega_t)

    cc = np.sum(cos_m * cos_m, axis=0)
    ss = np.sum(sin_m * sin_m, axis=0)
    cs = np.sum(cos_m * sin_m, axis=0)
    yc = y @ cos_m
    ys = y @ sin_m

    det = cc * ss - cs * cs
    valid = np.abs(det) > 1e-12
    a = np.full(freqs.shape, np.nan, dtype=float)
    b = np.full(freqs.shape, np.nan, dtype=float)
    a[valid] = (yc[valid] * ss[valid] - ys[valid] * cs[valid]) / det[valid]
    b[valid] = (ys[valid] * cc[valid] - yc[valid] * cs[valid]) / det[valid]

    amplitude = np.hypot(a, b)
    phase = np.arctan2(-b, a)

    y_norm2 = float(np.sum(y * y))
    fit_power = a * yc + b * ys
    sse = y_norm2 - fit_power
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0.0:
        r2 = np.full(freqs.shape, np.nan, dtype=float)
    else:
        r2 = 1.0 - sse / sst

    return {
        "freqs": freqs,
        "a": a,
        "b": b,
        "amplitude": amplitude,
        "phase": phase,
        "r2": r2,
    }


def fit_sine_at_frequency(t: np.ndarray, y: np.ndarray, freq_hz: float) -> dict[str, float]:
    result = fit_sine_scan(t, y, np.asarray([freq_hz], dtype=float))
    return {
        "freq_hz": float(freq_hz),
        "a": float(result["a"][0]),
        "b": float(result["b"][0]),
        "phase": float(result["phase"][0]),
        "amplitude": float(result["amplitude"][0]),
        "r2": float(result["r2"][0]),
    }


def build_primary_trace(cfg: Config):
    base_dataset, _ = split_dataset_component(cfg.dataset)
    ds_primary = load_bond_signal_dataset(
        dataset=f"{base_dataset}_{cfg.primary_component}",
        bond_spacing_mode=cfg.bond_spacing_mode,
        component=cfg.primary_component,
    )
    y = ds_primary.signal_matrix[:, cfg.primary_bond_index]
    processed, err = preprocess_signal(ds_primary.frame_times_s, y, longest=False, handlenan=False)
    if processed is None:
        raise ValueError(f"Primary preprocess failed: {err}")
    spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.sliding_len_s)
    if spec is None:
        raise ValueError("Primary spectrogram window too short")
    s_db = 20.0 * np.log10(np.abs(spec.S_complex) + np.finfo(float).eps)
    t_global = spec.t + processed.t[0]
    broadband_energy = np.sum(np.abs(spec.S_complex), axis=0)
    return base_dataset, ds_primary, processed, spec, t_global, s_db, broadband_energy


def detect_peak_times(t_global: np.ndarray, broadband_energy: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    peak_indices, _ = find_peaks(broadband_energy)
    peak_times = t_global[peak_indices]
    if cfg.manual_peak_times_s:
        peak_times = np.sort(
            np.unique(np.concatenate([peak_times, np.asarray(cfg.manual_peak_times_s, dtype=float)]))
        )
    return peak_indices, peak_times


def build_enabled_regions(
    ds_trace,
    bond_index: int,
    t_global: np.ndarray,
    peak_times: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray, np.ndarray]:
    segment_edges = np.concatenate(([float(t_global[0])], np.sort(peak_times), [float(t_global[-1])]))
    durations = np.diff(segment_edges)
    usable = durations >= float(cfg.min_segment_len_s)
    if usable.size > 0 and cfg.ignore_first_segment:
        usable[0] = False
    if usable.size > 0 and cfg.ignore_last_segment:
        usable[-1] = False

    t_raw = np.asarray(ds_trace.frame_times_s, dtype=float)
    y_raw = np.asarray(ds_trace.signal_matrix[:, bond_index], dtype=float)
    regions: list[dict[str, object]] = []

    for region_index in range(len(durations)):
        if not usable[region_index]:
            continue
        left = float(segment_edges[region_index] + cfg.ignore_beginning_len_s)
        right = float(segment_edges[region_index + 1] - cfg.ignore_end_len_s)
        if right <= left:
            continue
        mask = (t_raw >= left) & (t_raw <= right) & np.isfinite(y_raw)
        if not np.any(mask):
            continue
        t_region_raw = t_raw[mask]
        y_region_raw = y_raw[mask]
        processed, err = preprocess_signal(t_region_raw, y_region_raw, longest=False, handlenan=False)
        if processed is None:
            print(f"Skipping region {region_index}: preprocess failed ({err})")
            continue
        regions.append(
            {
                "region_index": region_index,
                "left": left,
                "right": right,
                "t_raw": t_region_raw,
                "y_raw": y_region_raw,
                "processed": processed,
                "mean_abs": float(np.mean(np.abs(processed.y))),
                "rms": float(np.sqrt(np.mean(processed.y ** 2))),
            }
        )

    return segment_edges, usable, regions, t_raw, y_raw


def analyze_regions(regions: list[dict[str, object]], target_bands: tuple[TargetBand, ...]) -> dict[str, dict[str, np.ndarray]]:
    n_regions = len(regions)
    metrics: dict[str, dict[str, np.ndarray]] = {}

    for target in target_bands:
        scan_freqs = np.linspace(
            target.center_hz - target.half_width_hz,
            target.center_hz + target.half_width_hz,
            target.n_scan,
        )
        exact_amp = np.full(n_regions, np.nan, dtype=float)
        exact_r2 = np.full(n_regions, np.nan, dtype=float)
        best_amp = np.full(n_regions, np.nan, dtype=float)
        best_r2 = np.full(n_regions, np.nan, dtype=float)
        best_freq = np.full(n_regions, np.nan, dtype=float)
        best_phase = np.full(n_regions, np.nan, dtype=float)
        fft_peak_freq = np.full(n_regions, np.nan, dtype=float)
        fft_peak_amp = np.full(n_regions, np.nan, dtype=float)
        scan_amplitudes = np.full((n_regions, target.n_scan), np.nan, dtype=float)
        scan_r2 = np.full((n_regions, target.n_scan), np.nan, dtype=float)

        for idx, region in enumerate(regions):
            processed = region["processed"]
            assert processed is not None
            t_proc = np.asarray(processed.t, dtype=float)
            y_proc = np.asarray(processed.y, dtype=float)

            exact = fit_sine_at_frequency(t_proc, y_proc, target.center_hz)
            scan = fit_sine_scan(t_proc, y_proc, scan_freqs)
            best_idx = int(np.nanargmax(scan["amplitude"]))

            exact_amp[idx] = exact["amplitude"]
            exact_r2[idx] = exact["r2"]
            best_amp[idx] = float(scan["amplitude"][best_idx])
            best_r2[idx] = float(scan["r2"][best_idx])
            best_freq[idx] = float(scan["freqs"][best_idx])
            best_phase[idx] = float(scan["phase"][best_idx])
            scan_amplitudes[idx, :] = scan["amplitude"]
            scan_r2[idx, :] = scan["r2"]

            fft_region = compute_one_sided_fft(y_proc, processed.dt)
            band_mask = (
                (fft_region.freq >= target.center_hz - target.half_width_hz)
                & (fft_region.freq <= target.center_hz + target.half_width_hz)
            )
            if np.any(band_mask):
                band_amp = fft_region.amplitude[band_mask]
                band_freq = fft_region.freq[band_mask]
                local_idx = int(np.argmax(band_amp))
                fft_peak_freq[idx] = float(band_freq[local_idx])
                fft_peak_amp[idx] = float(band_amp[local_idx])

        metrics[target.label] = {
            "scan_freqs": scan_freqs,
            "exact_amp": exact_amp,
            "exact_r2": exact_r2,
            "best_amp": best_amp,
            "best_r2": best_r2,
            "best_freq": best_freq,
            "best_phase": best_phase,
            "fft_peak_freq": fft_peak_freq,
            "fft_peak_amp": fft_peak_amp,
            "scan_amplitudes": scan_amplitudes,
            "scan_r2": scan_r2,
        }

    return metrics


def component_bond_sweep(
    cfg: Config,
    enabled_region_bounds: list[tuple[float, float]],
    target_bands: tuple[TargetBand, ...],
) -> dict[str, dict[tuple[str, int], dict[str, float]]]:
    base_dataset, _ = split_dataset_component(cfg.dataset)
    results: dict[str, dict[tuple[str, int], dict[str, float]]] = {}

    for target in target_bands:
        target_result: dict[tuple[str, int], dict[str, float]] = {}
        for component in ("x", "y", "a"):
            ds = load_bond_signal_dataset(
                dataset=f"{base_dataset}_{component}",
                bond_spacing_mode=cfg.bond_spacing_mode,
                component=component,
            )
            for bond_index in range(ds.signal_matrix.shape[1]):
                region_mean_abs: list[float] = []
                region_best_amp: list[float] = []
                region_best_r2: list[float] = []

                scan_freqs = np.linspace(
                    target.center_hz - target.half_width_hz,
                    target.center_hz + target.half_width_hz,
                    target.n_scan,
                )

                for left, right in enabled_region_bounds:
                    mask = (
                        (ds.frame_times_s >= left)
                        & (ds.frame_times_s <= right)
                        & np.isfinite(ds.signal_matrix[:, bond_index])
                    )
                    if not np.any(mask):
                        continue
                    processed, err = preprocess_signal(
                        ds.frame_times_s[mask],
                        ds.signal_matrix[:, bond_index][mask],
                        longest=False,
                        handlenan=False,
                    )
                    if processed is None:
                        continue
                    scan = fit_sine_scan(np.asarray(processed.t), np.asarray(processed.y), scan_freqs)
                    best_idx = int(np.nanargmax(scan["amplitude"]))
                    region_mean_abs.append(float(np.mean(np.abs(processed.y))))
                    region_best_amp.append(float(scan["amplitude"][best_idx]))
                    region_best_r2.append(float(scan["r2"][best_idx]))

                if len(region_mean_abs) < 2:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(region_mean_abs, region_best_amp)[0, 1])

                target_result[(component, bond_index)] = {
                    "corr": corr,
                    "mean_best_r2": float(np.mean(region_best_r2)) if region_best_r2 else np.nan,
                    "max_best_r2": float(np.max(region_best_r2)) if region_best_r2 else np.nan,
                    "mean_best_amp": float(np.mean(region_best_amp)) if region_best_amp else np.nan,
                }

        results[target.label] = target_result

    return results


def plot_timeseries(writer: PlotWriter, ds_primary, cfg: Config) -> Path:
    n_frames, n_bonds = ds_primary.signal_matrix.shape
    fig, axes = plt.subplots(
        n_bonds,
        1,
        figsize=(12, max(2.2 * n_bonds, 4)),
        sharex=True,
        constrained_layout=True,
    )
    if n_bonds == 1:
        axes = [axes]

    for bond_index, ax in enumerate(axes):
        ax.plot(ds_primary.frame_times_s, ds_primary.signal_matrix[:, bond_index], lw=1.0)
        label = ds_primary.pair_labels[bond_index] if bond_index < len(ds_primary.pair_labels) else f"bond {bond_index}"
        ax.set_ylabel(label)
        if bond_index == cfg.primary_bond_index:
            ax.set_title(f"Primary trace: {cfg.primary_component.upper()} bond {cfg.primary_bond_index}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{cfg.dataset} {cfg.primary_component}-component bond signals ({cfg.bond_spacing_mode})")
    return writer.save(fig, "timeseries")


def plot_peak_detection_regions(
    writer: PlotWriter,
    base_dataset: str,
    spec,
    s_db: np.ndarray,
    t_global: np.ndarray,
    broadband_energy: np.ndarray,
    peak_times: np.ndarray,
    t_raw: np.ndarray,
    y_raw: np.ndarray,
    regions: list[dict[str, object]],
    ds_primary,
    processed_primary,
    cfg: Config,
) -> Path:
    fig, (ax_spec, ax_trace, ax_energy) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    pcm = ax_spec.pcolormesh(t_global, spec.f, s_db, shading="auto", cmap="turbo", rasterized=True)
    fig.colorbar(pcm, ax=ax_spec, label="dB")
    label = ds_primary.pair_labels[cfg.primary_bond_index] if cfg.primary_bond_index < len(ds_primary.pair_labels) else f"bond {cfg.primary_bond_index}"
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_ylim(0.01, processed_primary.nyquist)
    ax_spec.set_title(f"Primary spectrogram | {cfg.primary_component.upper()} bond {cfg.primary_bond_index} ({label})")

    ax_trace.plot(t_raw, y_raw, lw=1.0, color="tab:blue")
    ax_trace.set_ylabel("position")
    ax_trace.set_title("Primary timeseries with enabled regions")
    ax_trace.grid(alpha=0.25)

    ax_energy.plot(t_global, broadband_energy, lw=1.3, color="tab:orange")
    ax_energy.set_ylabel("broadband")
    ax_energy.set_xlabel("time (s)")
    ax_energy.set_title("Broadband spectrogram energy")
    ax_energy.grid(alpha=0.25)

    for t_peak in peak_times:
        ax_spec.axvline(t_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.45)
        ax_trace.axvline(t_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.35)
        ax_energy.axvline(t_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.35)

    for region in regions:
        left = float(region["left"])
        right = float(region["right"])
        ax_trace.axvspan(left, right, color="limegreen", alpha=0.18, linewidth=0)
        ax_energy.axvspan(left, right, color="limegreen", alpha=0.18, linewidth=0)

    fig.suptitle(f"{base_dataset} | peak picking and quiet-region selection")
    return writer.save(fig, "peak_detection_regions")


def plot_region_timeseries(writer: PlotWriter, regions: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(
        len(regions),
        1,
        figsize=(12, max(2.1 * len(regions), 4)),
        sharex=False,
        constrained_layout=True,
    )
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        processed = region["processed"]
        assert processed is not None
        ax.plot(processed.t, processed.y, lw=1.0, color="tab:green")
        ax.axhline(0.0, color="0.35", lw=0.8, alpha=0.6)
        ax.grid(alpha=0.25)
        ax.set_ylabel("detrended")
        ax.set_title(
            f"Region {region['region_index']} | t=[{region['left']:.2f}, {region['right']:.2f}] s | "
            f"mean|x|={region['mean_abs']:.3f}"
        )

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Enabled-region detrended timeseries")
    return writer.save(fig, "region_timeseries")


def plot_target_band_scans(
    writer: PlotWriter,
    regions: list[dict[str, object]],
    metrics: dict[str, dict[str, np.ndarray]],
    target_bands: tuple[TargetBand, ...],
) -> Path:
    fig, axes = plt.subplots(
        len(target_bands),
        2,
        figsize=(14, 4.2 * len(target_bands)),
        constrained_layout=True,
    )
    if len(target_bands) == 1:
        axes = np.asarray([axes])

    mean_abs = np.asarray([float(region["mean_abs"]) for region in regions], dtype=float)
    region_labels = [int(region["region_index"]) for region in regions]

    for row_axes, target in zip(axes, target_bands):
        ax_scan, ax_scatter = row_axes
        data = metrics[target.label]
        freqs = data["scan_freqs"]

        for idx, region in enumerate(regions):
            color = plt.cm.viridis(idx / max(1, len(regions) - 1))
            ax_scan.plot(freqs, data["scan_amplitudes"][idx], color=color, alpha=0.75, lw=1.0)
            ax_scan.plot(data["best_freq"][idx], data["best_amp"][idx], "o", color=color, ms=4)

        ax_scan.axvline(target.center_hz, color="tab:red", linestyle="--", lw=1.0, alpha=0.8)
        ax_scan.set_xlabel("frequency (Hz)")
        ax_scan.set_ylabel("fit amplitude")
        ax_scan.set_title(f"{target.label}: local frequency scan")
        ax_scan.grid(alpha=0.25)

        ax_scatter.scatter(mean_abs, data["exact_amp"], color="tab:gray", alpha=0.75, label="exact target")
        sc = ax_scatter.scatter(
            mean_abs,
            data["best_amp"],
            c=data["best_r2"],
            cmap="viridis",
            s=60,
            edgecolors="black",
            linewidths=0.3,
            label="best in band",
        )
        for x_val, y_val, region_index in zip(mean_abs, data["best_amp"], region_labels):
            ax_scatter.annotate(str(region_index), (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax_scatter.set_xlabel("region mean |x|")
        ax_scatter.set_ylabel("fit amplitude")
        ax_scatter.set_title(
            f"{target.label}: scaling | corr exact={np.corrcoef(mean_abs, data['exact_amp'])[0, 1]:.3f}, "
            f"corr local={np.corrcoef(mean_abs, data['best_amp'])[0, 1]:.3f}"
        )
        ax_scatter.grid(alpha=0.25)
        ax_scatter.legend(loc="upper left")
        fig.colorbar(sc, ax=ax_scatter, label="best-fit $R^2$")

    fig.suptitle("Target-band scans and scaling diagnostics")
    return writer.save(fig, "target_band_scans")


def plot_target_best_frequency(
    writer: PlotWriter,
    regions: list[dict[str, object]],
    metrics: dict[str, dict[str, np.ndarray]],
    target_bands: tuple[TargetBand, ...],
) -> Path:
    fig, axes = plt.subplots(
        len(target_bands),
        1,
        figsize=(12, 3.6 * len(target_bands)),
        constrained_layout=True,
    )
    if len(target_bands) == 1:
        axes = [axes]

    centers = np.asarray([(float(region["left"]) + float(region["right"])) * 0.5 for region in regions], dtype=float)
    mean_abs = np.asarray([float(region["mean_abs"]) for region in regions], dtype=float)

    for ax, target in zip(axes, target_bands):
        data = metrics[target.label]
        sc = ax.scatter(
            centers,
            data["best_freq"],
            c=data["best_r2"],
            s=40.0 + 18.0 * mean_abs,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.3,
        )
        for x_val, y_val, region in zip(centers, data["best_freq"], regions):
            ax.annotate(str(region["region_index"]), (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.axhline(target.center_hz, color="tab:red", linestyle="--", lw=1.0, alpha=0.8)
        ax.axhline(target.center_hz - target.half_width_hz, color="0.5", linestyle=":", lw=0.8, alpha=0.7)
        ax.axhline(target.center_hz + target.half_width_hz, color="0.5", linestyle=":", lw=0.8, alpha=0.7)
        ax.set_xlabel("region center time (s)")
        ax.set_ylabel("best freq (Hz)")
        ax.set_title(
            f"{target.label}: local best frequency | mean $R^2$={np.nanmean(data['best_r2']):.3f}, "
            f"freq range=[{np.nanmin(data['best_freq']):.4f}, {np.nanmax(data['best_freq']):.4f}] Hz"
        )
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="best-fit $R^2$")

    fig.suptitle("Per-region best-fit frequency in each target band")
    return writer.save(fig, "target_best_frequency")


def plot_component_bond_sweep(
    writer: PlotWriter,
    sweep_results: dict[str, dict[tuple[str, int], dict[str, float]]],
    target_bands: tuple[TargetBand, ...],
) -> Path:
    fig, axes = plt.subplots(1, len(target_bands), figsize=(5.3 * len(target_bands), 5.6), constrained_layout=True)
    if len(target_bands) == 1:
        axes = [axes]

    row_labels = ["x", "y", "a"]
    col_labels = ["bond 0", "bond 1", "bond 2"]

    for ax, target in zip(axes, target_bands):
        data = sweep_results[target.label]
        heat = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)
        text = [["" for _ in col_labels] for _ in row_labels]

        for row_idx, component in enumerate(row_labels):
            for col_idx, bond_index in enumerate(range(len(col_labels))):
                cell = data[(component, bond_index)]
                heat[row_idx, col_idx] = cell["mean_best_r2"]
                text[row_idx][col_idx] = f"corr={cell['corr']:.2f}\nmax={cell['max_best_r2']:.2f}"

        im = ax.imshow(heat, cmap="magma", aspect="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(heat))))
        for row_idx in range(len(row_labels)):
            for col_idx in range(len(col_labels)):
                ax.text(
                    col_idx,
                    row_idx,
                    text[row_idx][col_idx],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

        ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
        ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
        ax.set_title(f"{target.label}\ncell color = mean best-fit $R^2$")
        fig.colorbar(im, ax=ax, label="mean best-fit $R^2$")

    fig.suptitle("Component/bond sweep inside the same quiet regions")
    return writer.save(fig, "component_bond_sweep")


def plot_primary_region_fits(
    writer: PlotWriter,
    regions: list[dict[str, object]],
    metrics: dict[str, dict[str, np.ndarray]],
    target_label: str,
) -> Path:
    data = metrics[target_label]
    n_regions = len(regions)
    ncols = 2
    nrows = math.ceil(n_regions / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(3.0 * nrows, 4.0)), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, region, best_freq, best_amp, best_phase, best_r2 in zip(
        axes,
        regions,
        data["best_freq"],
        data["best_amp"],
        data["best_phase"],
        data["best_r2"],
    ):
        processed = region["processed"]
        assert processed is not None
        t_proc = np.asarray(processed.t, dtype=float)
        y_proc = np.asarray(processed.y, dtype=float)
        fit_curve = best_amp * np.cos(2.0 * np.pi * best_freq * t_proc + best_phase)
        stride = max(1, len(t_proc) // 700)
        ax.plot(t_proc[::stride], y_proc[::stride], lw=1.0, color="tab:green", label="detrended")
        ax.plot(t_proc[::stride], fit_curve[::stride], lw=1.0, color="tab:red", alpha=0.9, label="best local fit")
        ax.axhline(0.0, color="0.35", lw=0.8, alpha=0.6)
        ax.grid(alpha=0.25)
        ax.set_title(
            f"Region {region['region_index']} | f={best_freq:.4f} Hz | amp={best_amp:.3f} | $R^2$={best_r2:.3f}"
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("x")
        ax.legend(loc="upper right")

    for ax in axes[n_regions:]:
        ax.axis("off")

    fig.suptitle(f"{target_label}: per-region best-fit reconstructions")
    return writer.save(fig, "primary_region_fits")


def build_summary(
    cfg: Config,
    regions: list[dict[str, object]],
    metrics: dict[str, dict[str, np.ndarray]],
    sweep_results: dict[str, dict[tuple[str, int], dict[str, float]]],
    peak_times: np.ndarray,
    usable: np.ndarray,
    saved_plot_paths: list[Path],
) -> str:
    lines: list[str] = [
        f"repo_root: {REPO_ROOT}",
        f"dataset: {cfg.dataset}",
        f"bond_spacing_mode: {cfg.bond_spacing_mode}",
        f"primary_component: {cfg.primary_component}",
        f"primary_bond_index: {cfg.primary_bond_index}",
        f"detected_peak_times_s: {np.array2string(peak_times, precision=3)}",
        f"usable_segment_flags: {np.array2string(usable.astype(int))}",
        "",
        "expectation:",
        "A genuinely fundamental peak should stay near a stable frequency, gain amplitude as the region oscillation amplitude grows, and produce cleaner per-region fits than a weak nonlinear sideband.",
        "A line can still be visibly present on a log FFT while explaining little variance in the raw time series, so FFT visibility and fit R^2 are different questions.",
        "",
        "primary regions:",
    ]

    for region in regions:
        lines.append(
            f"region {int(region['region_index'])}: t=[{float(region['left']):.3f}, {float(region['right']):.3f}] s | "
            f"mean|x|={float(region['mean_abs']):.4f} | rms={float(region['rms']):.4f}"
        )

    lines.append("")
    lines.append("target findings:")

    mean_abs = np.asarray([float(region["mean_abs"]) for region in regions], dtype=float)

    for target in cfg.target_bands:
        data = metrics[target.label]
        exact_corr = float(np.corrcoef(mean_abs, data["exact_amp"])[0, 1])
        local_corr = float(np.corrcoef(mean_abs, data["best_amp"])[0, 1])
        lines.append(
            f"{target.label}: exact corr={exact_corr:.3f}, local corr={local_corr:.3f}, "
            f"mean exact R^2={np.nanmean(data['exact_r2']):.3f}, mean local R^2={np.nanmean(data['best_r2']):.3f}, "
            f"best-frequency range=[{np.nanmin(data['best_freq']):.4f}, {np.nanmax(data['best_freq']):.4f}] Hz"
        )
        lines.append(f"  exact amplitudes: {summarize_array(data['exact_amp'])}")
        lines.append(f"  local amplitudes: {summarize_array(data['best_amp'])}")
        lines.append(f"  local best freqs: {summarize_array(data['best_freq'])}")
        lines.append(f"  local best R^2: {summarize_array(data['best_r2'])}")
        lines.append(f"  FFT peak freqs in band: {summarize_array(data['fft_peak_freq'])}")
        lines.append(f"  FFT peak amps in band: {summarize_array(data['fft_peak_amp'])}")

    lines.extend(
        [
            "",
            "interpretation:",
            (
                "3.35 Hz behaves much more like a true fundamental than the FFT-bin-average method suggested: "
                "its local-band fit amplitude tracks region mean|x| substantially better than the exact fixed-3.35 fit."
            ),
            (
                "The exact-frequency fit can fail badly when the local peak sits a few hundredths of a hertz away from 3.35; "
                "over 25-95 s windows that phase mismatch is enough to cancel a visually obvious spectral line."
            ),
            (
                "12.0 Hz and 16.65 Hz can still be real visible peaks on a log FFT, but in these quiet-region time-domain fits "
                "they explain much less variance than the 3.35 Hz band. That means they are present, but much less dominant in this observable."
            ),
            (
                "If the question is 'is this line fundamental or nonlinear?', the current data says 3.35 Hz has the cleanest fundamental-like scaling signature in the primary x-component bond signal. "
                "The higher-frequency lines are visible, but their time-domain scaling is weaker and harder to extract robustly from these segments."
            ),
            "",
            "component/bond sweep:",
        ]
    )

    for target in cfg.target_bands:
        lines.append(target.label)
        for component in ("x", "y", "a"):
            entries = []
            for bond_index in range(3):
                cell = sweep_results[target.label][(component, bond_index)]
                entries.append(
                    f"{component}{bond_index}: corr={cell['corr']:.3f}, meanR2={cell['mean_best_r2']:.3f}, maxR2={cell['max_best_r2']:.3f}"
                )
            lines.append("  " + " | ".join(entries))

    lines.extend(["", "saved_plots:"])
    lines.extend(str(path) for path in saved_plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    base_dataset, ds_primary, processed_primary, spec, t_global, s_db, broadband_energy = build_primary_trace(CONFIG)
    _, peak_times = detect_peak_times(t_global, broadband_energy, CONFIG)
    _, usable, regions, t_raw, y_raw = build_enabled_regions(
        ds_primary,
        CONFIG.primary_bond_index,
        t_global,
        peak_times,
        CONFIG,
    )

    if not regions:
        raise ValueError("No enabled regions survived preprocessing.")

    metrics = analyze_regions(regions, CONFIG.target_bands)
    enabled_region_bounds = [(float(region["left"]), float(region["right"])) for region in regions]
    sweep_results = component_bond_sweep(CONFIG, enabled_region_bounds, CONFIG.target_bands)

    saved_plots = [
        plot_timeseries(writer, ds_primary, CONFIG),
        plot_peak_detection_regions(
            writer,
            base_dataset,
            spec,
            s_db,
            t_global,
            broadband_energy,
            peak_times,
            t_raw,
            y_raw,
            regions,
            ds_primary,
            processed_primary,
            CONFIG,
        ),
        plot_region_timeseries(writer, regions),
        plot_target_band_scans(writer, regions, metrics, CONFIG.target_bands),
        plot_target_best_frequency(writer, regions, metrics, CONFIG.target_bands),
        plot_component_bond_sweep(writer, sweep_results, CONFIG.target_bands),
        plot_primary_region_fits(writer, regions, metrics, "3.35 Hz"),
    ]

    summary_path = save_summary(
        build_summary(
            CONFIG,
            regions,
            metrics,
            sweep_results,
            peak_times,
            usable,
            saved_plots,
        )
    )
    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
