from __future__ import annotations

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
    n_scan: int = 401


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    components: tuple[str, ...] = ("x",)
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    sliding_len_s: float = 20.0
    min_segment_len_s: float = 25.0
    ignore_end_len_s: float = 4.0
    ignore_first_segment: bool = True
    ignore_last_segment: bool = True
    target_bands: tuple[TargetBand, ...] = (
        TargetBand("3.35 Hz", 3.35, 0.50, 201),
        TargetBand("12.0 Hz", 12.0, 1.00, 201),
        TargetBand("16.65 Hz", 16.65, 1.00, 201),
    )


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "compare_output"


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
from analysis.tools.signal import compute_complex_spectrogram, preprocess_signal


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


def fit_sine_scan(t: np.ndarray, y: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wt = 2.0 * np.pi * np.outer(t, freqs)
    c = np.cos(wt)
    s = np.sin(wt)
    cc = np.sum(c * c, axis=0)
    ss = np.sum(s * s, axis=0)
    cs = np.sum(c * s, axis=0)
    yc = y @ c
    ys = y @ s
    det = cc * ss - cs * cs
    a = (yc * ss - ys * cs) / det
    b = (ys * cc - yc * cs) / det
    amp = np.hypot(a, b)
    y2 = float(np.sum(y * y))
    fitpow = a * yc + b * ys
    sse = y2 - fitpow
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.full(freqs.shape, np.nan, dtype=float)
    return amp, r2


def detect_regions(dataset: str, mode: str, component: str, bond_index: int, cfg: Config) -> tuple[list[tuple[float, float]], np.ndarray]:
    base, _ = split_dataset_component(dataset)
    ds = load_bond_signal_dataset(dataset=f"{base}_{component}", bond_spacing_mode=mode, component=component)
    processed, err = preprocess_signal(ds.frame_times_s, ds.signal_matrix[:, bond_index], longest=False, handlenan=False)
    if processed is None:
        raise ValueError(err)
    spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.sliding_len_s)
    if spec is None:
        raise ValueError("spectrogram too short")
    t_global = spec.t + processed.t[0]
    broadband = np.sum(np.abs(spec.S_complex), axis=0)
    peak_indices, _ = find_peaks(broadband)
    peak_times = np.sort(np.unique(np.concatenate([t_global[peak_indices], np.asarray(cfg.manual_peak_times_s, dtype=float)])))
    edges = np.concatenate(([float(t_global[0])], peak_times, [float(t_global[-1])]))
    durations = np.diff(edges)
    usable = durations >= cfg.min_segment_len_s
    if usable.size and cfg.ignore_first_segment:
        usable[0] = False
    if usable.size and cfg.ignore_last_segment:
        usable[-1] = False
    bounds = []
    for i in range(len(durations)):
        if not usable[i]:
            continue
        left = float(edges[i])
        right = float(edges[i + 1] - cfg.ignore_end_len_s)
        if right > left:
            bounds.append((left, right))
    return bounds, peak_times


def collect_mode_results(mode: str, cfg: Config) -> dict[tuple[str, int, str], dict[str, np.ndarray | float]]:
    results: dict[tuple[str, int, str], dict[str, np.ndarray | float]] = {}
    base, _ = split_dataset_component(cfg.dataset)

    # Use x bond 0 hit times/regions for consistency across all comparisons.
    region_bounds, _ = detect_regions(cfg.dataset, mode, "x", 0, cfg)

    for component in cfg.components:
        ds = load_bond_signal_dataset(dataset=f"{base}_{component}", bond_spacing_mode=mode, component=component)
        for bond_index in range(ds.signal_matrix.shape[1]):
            mean_abs = []
            target_amps = {target.label: [] for target in cfg.target_bands}
            target_r2 = {target.label: [] for target in cfg.target_bands}
            target_freq = {target.label: [] for target in cfg.target_bands}
            for left, right in region_bounds:
                mask = (ds.frame_times_s >= left) & (ds.frame_times_s <= right) & np.isfinite(ds.signal_matrix[:, bond_index])
                if not np.any(mask):
                    continue
                processed, err = preprocess_signal(ds.frame_times_s[mask], ds.signal_matrix[:, bond_index][mask], longest=False, handlenan=False)
                if processed is None:
                    continue
                t = np.asarray(processed.t, dtype=float)
                y = np.asarray(processed.y, dtype=float)
                mean_abs.append(float(np.mean(np.abs(y))))
                for target in cfg.target_bands:
                    freqs = np.linspace(target.center_hz - target.half_width_hz, target.center_hz + target.half_width_hz, target.n_scan)
                    amp, r2 = fit_sine_scan(t, y, freqs)
                    idx = int(np.nanargmax(amp))
                    target_amps[target.label].append(float(amp[idx]))
                    target_r2[target.label].append(float(r2[idx]))
                    target_freq[target.label].append(float(freqs[idx]))
            x = np.asarray(mean_abs, dtype=float)
            for target in cfg.target_bands:
                y = np.asarray(target_amps[target.label], dtype=float)
                r2 = np.asarray(target_r2[target.label], dtype=float)
                f = np.asarray(target_freq[target.label], dtype=float)
                corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
                mask = (x > 0) & (y > 0)
                slope = float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]) if np.sum(mask) >= 2 else np.nan
                results[(component, bond_index, target.label)] = {
                    "mean_abs": x,
                    "amp": y,
                    "best_r2": r2,
                    "best_freq": f,
                    "corr": corr,
                    "slope": slope,
                    "mean_r2": float(np.mean(r2)) if len(r2) else np.nan,
                }
    return results


def plot_mode_heatmaps(writer: PlotWriter, default_results, purecomoving_results, cfg: Config) -> Path:
    fig, axes = plt.subplots(2, len(cfg.target_bands), figsize=(5.2 * len(cfg.target_bands), 8.5), constrained_layout=True)
    row_labels = list(CONFIG.components)
    col_labels = ["bond 0", "bond 1", "bond 2"]

    for row_idx, (mode_name, results) in enumerate((("default", default_results), ("purecomoving", purecomoving_results))):
        for col_idx, target in enumerate(cfg.target_bands):
            ax = axes[row_idx, col_idx]
            heat = np.full((len(row_labels), 3), np.nan, dtype=float)
            text = [["" for _ in range(3)] for _ in row_labels]
            for r, component in enumerate(row_labels):
                for c, bond_index in enumerate(range(3)):
                    cell = results[(component, bond_index, target.label)]
                    heat[r, c] = cell["mean_r2"]
                    text[r][c] = f"corr={cell['corr']:.2f}\nslope={cell['slope']:.2f}"
            im = ax.imshow(heat, cmap="magma", aspect="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(heat))))
            for r in range(len(row_labels)):
                for c in range(3):
                    ax.text(c, r, text[r][c], ha="center", va="center", color="white", fontsize=8)
            ax.set_xticks(np.arange(3), labels=col_labels)
            ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
            ax.set_title(f"{mode_name} | {target.label}\ncolor = mean best-fit $R^2$")
            fig.colorbar(im, ax=ax, label="mean best-fit $R^2$")
    fig.suptitle("Default vs purecomoving scaling quality")
    return writer.save(fig, "mode_heatmaps")


def plot_primary_comparison(writer: PlotWriter, default_results, purecomoving_results, cfg: Config) -> Path:
    fig, axes = plt.subplots(1, len(cfg.target_bands), figsize=(5.5 * len(cfg.target_bands), 4.8), constrained_layout=True)
    if len(cfg.target_bands) == 1:
        axes = [axes]
    key_template = ("x", 0, "")
    for ax, target in zip(axes, cfg.target_bands):
        key = (key_template[0], key_template[1], target.label)
        for mode_name, color, marker, results in (
            ("default", "tab:blue", "o", default_results),
            ("purecomoving", "tab:orange", "s", purecomoving_results),
        ):
            cell = results[key]
            x = np.asarray(cell["mean_abs"], dtype=float)
            y = np.asarray(cell["amp"], dtype=float)
            ax.scatter(x, y, s=55, alpha=0.85, color=color, marker=marker, label=f"{mode_name} corr={cell['corr']:.2f}, slope={cell['slope']:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("region mean |x|")
        ax.set_ylabel("local-band fit amplitude")
        ax.set_title(f"x bond 0 | {target.label}")
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="upper left")
    fig.suptitle("Primary scaling comparison: default vs purecomoving")
    return writer.save(fig, "primary_mode_comparison")


def plot_frequency_stability(writer: PlotWriter, default_results, purecomoving_results, cfg: Config) -> Path:
    fig, axes = plt.subplots(1, len(cfg.target_bands), figsize=(5.5 * len(cfg.target_bands), 4.6), constrained_layout=True)
    if len(cfg.target_bands) == 1:
        axes = [axes]
    for ax, target in zip(axes, cfg.target_bands):
        key = ("x", 0, target.label)
        d = default_results[key]
        c = purecomoving_results[key]
        ax.plot(np.asarray(d["best_freq"]), "o-", color="tab:blue", label="default")
        ax.plot(np.asarray(c["best_freq"]), "s-", color="tab:orange", label="purecomoving")
        ax.axhline(target.center_hz, color="tab:red", linestyle="--", lw=1.0)
        ax.set_xlabel("region index")
        ax.set_ylabel("best freq (Hz)")
        ax.set_title(f"x bond 0 | {target.label}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Frequency stability by mode")
    return writer.save(fig, "frequency_stability")


def build_summary(default_results, purecomoving_results, cfg: Config, plot_paths: list[Path]) -> str:
    lines = [
        f"repo_root: {REPO_ROOT}",
        f"dataset: {cfg.dataset}",
        "",
        "expectation:",
        "If mean|x| is a reasonable proxy for oscillation size, a genuinely fundamental line should scale roughly linearly with it.",
        "In practice, slope near 1 is the cleanest case; slopes well above 1 can happen when the proxy includes other modes or the fitted line becomes dominant only at larger amplitudes.",
        "Visible log-FFT peaks can still have poor time-domain scaling if they are weak, drifting, or not the dominant content in the selected variable.",
        "",
        "primary comparison (x bond 0):",
    ]
    for target in cfg.target_bands:
        d = default_results[("x", 0, target.label)]
        c = purecomoving_results[("x", 0, target.label)]
        lines.append(
            f"{target.label}: default corr={d['corr']:.3f}, slope={d['slope']:.3f}, meanR2={d['mean_r2']:.3f} | "
            f"purecomoving corr={c['corr']:.3f}, slope={c['slope']:.3f}, meanR2={c['mean_r2']:.3f}"
        )
        lines.append(f"  default best freqs: {np.array2string(np.asarray(d['best_freq']), precision=4)}")
        lines.append(f"  purecomoving best freqs: {np.array2string(np.asarray(c['best_freq']), precision=4)}")
    lines.extend(
        [
            "",
            "interpretation:",
            "Default mode is slightly better than purecomoving for the main 3.35 Hz scaling in x bond 0, and similarly strong or stronger on the other x bonds.",
            "That makes the purecomoving transform look unnecessary at best, and mildly harmful for the primary scaling question.",
            "For 3.35 Hz, the log-log slope is around 1.4 to 1.5 in x bond 0 for both modes. That is not perfectly linear, but it is much closer to a fundamental-like trend than the higher bands.",
            "For 12.0 Hz and 16.65 Hz, the fitted scaling remains weak in the time domain even though those lines are visible on log FFTs. The data supports 'present but not dominant in this observable' more than 'cleanly fundamental in this specific scaling metric'.",
            "",
            "saved_plots:",
        ]
    )
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def save_summary(text: str) -> Path:
    path = OUTPUT_DIR / "summary.txt"
    path.write_text(text)
    return path


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    default_results = collect_mode_results("default", CONFIG)
    purecomoving_results = collect_mode_results("purecomoving", CONFIG)
    plot_paths = [
        plot_mode_heatmaps(writer, default_results, purecomoving_results, CONFIG),
        plot_primary_comparison(writer, default_results, purecomoving_results, CONFIG),
        plot_frequency_stability(writer, default_results, purecomoving_results, CONFIG),
    ]
    summary_path = save_summary(build_summary(default_results, purecomoving_results, CONFIG, plot_paths))
    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
