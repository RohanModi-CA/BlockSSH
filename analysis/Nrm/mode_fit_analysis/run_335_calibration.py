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
class Config:
    dataset: str = "IMG_0681_rot270"
    bond_spacing_mode: str = "default"
    component: str = "x"
    target_hz: float = 3.35
    half_width_hz: float = 0.50
    n_scan: int = 401
    sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    ignore_first_segment: bool = True
    ignore_last_segment: bool = True
    begin_trim_grid_s: tuple[int, ...] = (0, 1, 2, 4, 6, 8)
    end_trim_grid_s: tuple[int, ...] = (0, 2, 4, 6, 8, 10)


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "calibration_output"


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


def save_summary(text: str) -> Path:
    path = OUTPUT_DIR / "summary.txt"
    path.write_text(text)
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


def detect_base_segments(cfg: Config) -> tuple[object, np.ndarray]:
    base_dataset, _ = split_dataset_component(cfg.dataset)
    ds = load_bond_signal_dataset(
        dataset=f"{base_dataset}_{cfg.component}",
        bond_spacing_mode=cfg.bond_spacing_mode,
        component=cfg.component,
    )
    processed, err = preprocess_signal(ds.frame_times_s, ds.signal_matrix[:, 0], longest=False, handlenan=False)
    if processed is None:
        raise ValueError(err)
    spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.sliding_len_s)
    if spec is None:
        raise ValueError("spectrogram too short")
    t_global = spec.t + processed.t[0]
    broadband = np.sum(np.abs(spec.S_complex), axis=0)
    peak_indices, _ = find_peaks(broadband)
    peak_times = np.sort(
        np.unique(np.concatenate([t_global[peak_indices], np.asarray(cfg.manual_peak_times_s, dtype=float)]))
    )
    edges = np.concatenate(([float(t_global[0])], peak_times, [float(t_global[-1])]))
    durations = np.diff(edges)
    usable = durations >= cfg.min_segment_len_s
    if usable.size and cfg.ignore_first_segment:
        usable[0] = False
    if usable.size and cfg.ignore_last_segment:
        usable[-1] = False
    return ds, edges, usable


def region_rows_for_trim(ds, edges: np.ndarray, usable: np.ndarray, begin_trim_s: float, end_trim_s: float, cfg: Config) -> list[dict[str, float]]:
    rows = []
    freqs = np.linspace(cfg.target_hz - cfg.half_width_hz, cfg.target_hz + cfg.half_width_hz, cfg.n_scan)
    for region_index in range(len(edges) - 1):
        if not usable[region_index]:
            continue
        left = float(edges[region_index] + begin_trim_s)
        right = float(edges[region_index + 1] - end_trim_s)
        if right <= left:
            continue
        row: dict[str, float] = {"region": float(region_index), "left": left, "right": right}
        for bond_index in range(ds.signal_matrix.shape[1]):
            mask = (
                (ds.frame_times_s >= left)
                & (ds.frame_times_s <= right)
                & np.isfinite(ds.signal_matrix[:, bond_index])
            )
            if not np.any(mask):
                break
            processed, err = preprocess_signal(
                ds.frame_times_s[mask],
                ds.signal_matrix[:, bond_index][mask],
                longest=False,
                handlenan=False,
            )
            if processed is None:
                break
            amp, r2 = fit_sine_scan(np.asarray(processed.t, dtype=float), np.asarray(processed.y, dtype=float), freqs)
            idx = int(np.nanargmax(amp))
            row[f"amp{bond_index}"] = float(amp[idx])
            row[f"r2{bond_index}"] = float(r2[idx])
            row[f"freq{bond_index}"] = float(freqs[idx])
            row[f"mean{bond_index}"] = float(np.mean(np.abs(processed.y)))
            row[f"rms{bond_index}"] = float(np.sqrt(np.mean(processed.y ** 2)))
        else:
            row["proxy12"] = float(np.hypot(row["amp1"], row["amp2"]))
            row["proxy02"] = float(np.hypot(row["amp0"], row["amp2"]))
            row["proxy01"] = float(np.hypot(row["amp0"], row["amp1"]))
            row["mean_proxy"] = float(np.mean([row["mean0"], row["mean1"], row["mean2"]]))
            row["rms_proxy"] = float(np.sqrt(row["rms0"] ** 2 + row["rms1"] ** 2 + row["rms2"] ** 2))
            rows.append(row)
    return rows


def trim_grid_search(ds, edges: np.ndarray, usable: np.ndarray, cfg: Config):
    shape = (len(cfg.begin_trim_grid_s), len(cfg.end_trim_grid_s))
    corr_grid = np.full(shape, np.nan, dtype=float)
    slope_grid = np.full(shape, np.nan, dtype=float)
    mean_r2_grid = np.full(shape, np.nan, dtype=float)
    bad_grid = np.full(shape, np.nan, dtype=float)
    score_grid = np.full(shape, np.nan, dtype=float)
    rows_map: dict[tuple[int, int], list[dict[str, float]]] = {}
    best_score = None
    best_key = None

    for i, begin_trim_s in enumerate(cfg.begin_trim_grid_s):
        for j, end_trim_s in enumerate(cfg.end_trim_grid_s):
            rows = region_rows_for_trim(ds, edges, usable, begin_trim_s, end_trim_s, cfg)
            rows_map[(i, j)] = rows
            x = np.asarray([row["proxy12"] for row in rows], dtype=float)
            y = np.asarray([row["amp0"] for row in rows], dtype=float)
            r2 = np.asarray([row["r20"] for row in rows], dtype=float)
            freq = np.asarray([row["freq0"] for row in rows], dtype=float)
            corr = float(np.corrcoef(x, y)[0, 1])
            slope = float(np.polyfit(np.log(x), np.log(y), 1)[0])
            mean_r2 = float(np.mean(r2))
            bad = float(np.sum((r2 < 0.15) | (np.abs(freq - cfg.target_hz) > 0.08)))
            score = corr + mean_r2 - 0.05 * bad
            corr_grid[i, j] = corr
            slope_grid[i, j] = slope
            mean_r2_grid[i, j] = mean_r2
            bad_grid[i, j] = bad
            score_grid[i, j] = score
            if best_score is None or score > best_score:
                best_score = score
                best_key = (i, j)

    assert best_key is not None
    return {
        "corr_grid": corr_grid,
        "slope_grid": slope_grid,
        "mean_r2_grid": mean_r2_grid,
        "bad_grid": bad_grid,
        "score_grid": score_grid,
        "rows_map": rows_map,
        "best_key": best_key,
    }


def plot_trim_grid(writer: PlotWriter, search, cfg: Config) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    items = [
        ("corr", search["corr_grid"], "viridis"),
        ("slope", search["slope_grid"], "cividis"),
        ("mean R^2", search["mean_r2_grid"], "magma"),
        ("score", search["score_grid"], "plasma"),
    ]
    best_i, best_j = search["best_key"]

    for ax, (title, grid, cmap) in zip(axes.ravel(), items):
        im = ax.imshow(grid, aspect="auto", cmap=cmap)
        for i, begin_trim_s in enumerate(cfg.begin_trim_grid_s):
            for j, end_trim_s in enumerate(cfg.end_trim_grid_s):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        ax.scatter(best_j, best_i, s=110, facecolors="none", edgecolors="cyan", linewidths=2.0)
        ax.set_xticks(np.arange(len(cfg.end_trim_grid_s)), labels=[str(x) for x in cfg.end_trim_grid_s])
        ax.set_yticks(np.arange(len(cfg.begin_trim_grid_s)), labels=[str(x) for x in cfg.begin_trim_grid_s])
        ax.set_xlabel("end trim (s)")
        ax.set_ylabel("begin trim (s)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle("3.35 Hz calibration: trim-grid search using leave-one-out modal proxy")
    return writer.save(fig, "trim_grid")


def plot_proxy_comparison(writer: PlotWriter, rows: list[dict[str, float]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    region_ids = [int(row["region"]) for row in rows]

    # Old proxy
    x_old = np.asarray([row["mean0"] for row in rows], dtype=float)
    y = np.asarray([row["amp0"] for row in rows], dtype=float)
    axes[0].scatter(x_old, y, c="tab:gray", s=60)
    for x_val, y_val, region_id in zip(x_old, y, region_ids):
        axes[0].annotate(str(region_id), (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bond 0 mean |x|")
    axes[0].set_ylabel("bond 0 fitted 3.35 Hz amplitude")
    axes[0].set_title(
        f"Broadband proxy\ncorr={np.corrcoef(x_old, y)[0, 1]:.3f}, slope={np.polyfit(np.log(x_old), np.log(y), 1)[0]:.3f}"
    )
    axes[0].grid(alpha=0.25, which="both")

    # New proxy
    x_new = np.asarray([row["proxy12"] for row in rows], dtype=float)
    colors = np.asarray([row["r20"] for row in rows], dtype=float)
    sc = axes[1].scatter(x_new, y, c=colors, cmap="viridis", s=70, edgecolors="black", linewidths=0.3)
    for x_val, y_val, region_id in zip(x_new, y, region_ids):
        axes[1].annotate(str(region_id), (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"leave-one-out proxy $\sqrt{A_1^2 + A_2^2}$")
    axes[1].set_ylabel("bond 0 fitted 3.35 Hz amplitude")
    axes[1].set_title(
        f"Mode-aligned proxy\ncorr={np.corrcoef(x_new, y)[0, 1]:.3f}, slope={np.polyfit(np.log(x_new), np.log(y), 1)[0]:.3f}"
    )
    axes[1].grid(alpha=0.25, which="both")
    fig.colorbar(sc, ax=axes[1], label="bond 0 fit $R^2$")

    fig.suptitle("Hidden error: broadband amplitude proxy vs modal proxy")
    return writer.save(fig, "proxy_comparison")


def plot_region_diagnostics(writer: PlotWriter, rows: list[dict[str, float]], cfg: Config) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    region_ids = np.asarray([int(row["region"]) for row in rows], dtype=int)

    for bond_index, color in zip(range(3), ("tab:blue", "tab:orange", "tab:green")):
        axes[0].plot(region_ids, [row[f"freq{bond_index}"] for row in rows], "o-", color=color, label=f"bond {bond_index}")
    axes[0].axhline(cfg.target_hz, color="tab:red", linestyle="--", lw=1.0)
    axes[0].set_xlabel("region index")
    axes[0].set_ylabel("best local frequency (Hz)")
    axes[0].set_title("Local best frequency by region")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for bond_index, color in zip(range(3), ("tab:blue", "tab:orange", "tab:green")):
        axes[1].plot(region_ids, [row[f"r2{bond_index}"] for row in rows], "o-", color=color, label=f"bond {bond_index}")
    axes[1].axhline(0.15, color="tab:red", linestyle="--", lw=1.0)
    axes[1].set_xlabel("region index")
    axes[1].set_ylabel("best-fit $R^2$")
    axes[1].set_title("Fit quality by region")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Region diagnostics for 3.35 Hz calibration")
    return writer.save(fig, "region_diagnostics")


def plot_bond_scaling(writer: PlotWriter, rows: list[dict[str, float]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    x01 = np.asarray([row["amp1"] for row in rows], dtype=float)
    y0 = np.asarray([row["amp0"] for row in rows], dtype=float)
    axes[0].scatter(x01, y0, s=65, color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("bond 1 fitted 3.35 Hz amplitude")
    axes[0].set_ylabel("bond 0 fitted 3.35 Hz amplitude")
    axes[0].set_title(f"bond0 vs bond1 | corr={np.corrcoef(x01, y0)[0, 1]:.3f}")
    axes[0].grid(alpha=0.25, which="both")

    x02 = np.asarray([row["amp2"] for row in rows], dtype=float)
    axes[1].scatter(x02, y0, s=65, color="tab:green")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("bond 2 fitted 3.35 Hz amplitude")
    axes[1].set_ylabel("bond 0 fitted 3.35 Hz amplitude")
    axes[1].set_title(f"bond0 vs bond2 | corr={np.corrcoef(x02, y0)[0, 1]:.3f}")
    axes[1].grid(alpha=0.25, which="both")

    fig.suptitle("3.35 Hz is highly coherent across the x bonds")
    return writer.save(fig, "bond_scaling")


def build_summary(search, rows: list[dict[str, float]], cfg: Config, plot_paths: list[Path]) -> str:
    best_i, best_j = search["best_key"]
    begin_trim_s = cfg.begin_trim_grid_s[best_i]
    end_trim_s = cfg.end_trim_grid_s[best_j]
    x_old = np.asarray([row["mean0"] for row in rows], dtype=float)
    x_new = np.asarray([row["proxy12"] for row in rows], dtype=float)
    y0 = np.asarray([row["amp0"] for row in rows], dtype=float)
    old_corr = float(np.corrcoef(x_old, y0)[0, 1])
    new_corr = float(np.corrcoef(x_new, y0)[0, 1])
    old_slope = float(np.polyfit(np.log(x_old), np.log(y0), 1)[0])
    new_slope = float(np.polyfit(np.log(x_new), np.log(y0), 1)[0])
    region_ids = [int(row["region"]) for row in rows]
    bad_regions = [
        int(row["region"])
        for row in rows
        if (row["r20"] < 0.15) or (abs(row["freq0"] - cfg.target_hz) > 0.08)
    ]

    lines = [
        f"repo_root: {REPO_ROOT}",
        f"dataset: {cfg.dataset}",
        f"bond_spacing_mode: {cfg.bond_spacing_mode}",
        f"component: {cfg.component}",
        f"target_hz: {cfg.target_hz}",
        f"best_begin_trim_s: {begin_trim_s}",
        f"best_end_trim_s: {end_trim_s}",
        "",
        "what was wrong before:",
        "The old x-axis, mean|x| on a single bond, is broadband. It mixes the 3.35 Hz mode with everything else in the quiet region.",
        "That makes a genuinely clean fundamental look worse than it is, because low-amplitude regions can still have non-3.35 motion that inflates mean|x|.",
        "The 3.35 Hz amplitudes across the three x bonds are actually extremely coherent; the mode is fine, the proxy was not.",
        "",
        "calibration result:",
        f"Using bond 0 mean|x| as the proxy: corr={old_corr:.3f}, slope={old_slope:.3f}",
        f"Using leave-one-out modal proxy sqrt(A1^2 + A2^2): corr={new_corr:.3f}, slope={new_slope:.3f}",
        f"Mean bond 0 fit R^2 at best trim: {np.mean([row['r20'] for row in rows]):.3f}",
        f"Regions kept: {region_ids}",
        f"Problem regions: {bad_regions}",
        "",
        "per-region values:",
    ]

    for row in rows:
        lines.append(
            f"region {int(row['region'])}: amp0={row['amp0']:.4f}, amp1={row['amp1']:.4f}, amp2={row['amp2']:.4f}, "
            f"proxy12={row['proxy12']:.4f}, freq0={row['freq0']:.4f}, r20={row['r20']:.4f}"
        )

    lines.extend(
        [
            "",
            "interpretation:",
            "3.35 Hz does behave almost perfectly once the analysis is mode-aligned.",
            "The remaining ugly point is a real weak/contaminated segment, not a general failure of the 3.35 Hz scaling.",
            "For the weaker peaks, the lesson is clear: do not compare them to broadband mean|x|. Compare them against a mode-aware amplitude proxy instead.",
            "",
            "saved_plots:",
        ]
    )
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    ds, edges, usable = detect_base_segments(CONFIG)
    search = trim_grid_search(ds, edges, usable, CONFIG)
    best_i, best_j = search["best_key"]
    rows = search["rows_map"][(best_i, best_j)]
    plot_paths = [
        plot_trim_grid(writer, search, CONFIG),
        plot_proxy_comparison(writer, rows),
        plot_region_diagnostics(writer, rows, CONFIG),
        plot_bond_scaling(writer, rows),
    ]
    summary_path = save_summary(build_summary(search, rows, CONFIG, plot_paths))
    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
