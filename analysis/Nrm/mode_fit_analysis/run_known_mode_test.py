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
    known_kind: str


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    bond_spacing_mode: str = "default"
    component: str = "x"
    sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    begin_trim_s: float = 8.0
    end_trim_s: float = 10.0
    n_scan: int = 401
    target_bands: tuple[TargetBand, ...] = (
        TargetBand("3.35 Hz", 3.35, 0.50, "fundamental"),
        TargetBand("6.35 Hz", 6.35, 0.60, "fundamental"),
        TargetBand("8.96 Hz", 8.96, 0.70, "fundamental"),
        TargetBand("18.0 Hz", 18.00, 1.00, "non-fundamental"),
    )


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "known_mode_output"


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


def detect_regions(cfg: Config) -> tuple[object, list[tuple[int, float, float]]]:
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
    if usable.size:
        usable[0] = False
        usable[-1] = False

    regions: list[tuple[int, float, float]] = []
    for i in range(len(durations)):
        if not usable[i]:
            continue
        left = float(edges[i] + cfg.begin_trim_s)
        right = float(edges[i + 1] - cfg.end_trim_s)
        if right > left:
            regions.append((i, left, right))
    return ds, regions


def analyze_targets(ds, regions: list[tuple[int, float, float]], cfg: Config) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}

    for target in cfg.target_bands:
        freqs = np.linspace(target.center_hz - target.half_width_hz, target.center_hz + target.half_width_hz, cfg.n_scan)
        rows = []
        for region_index, left, right in regions:
            row = {"region": float(region_index), "left": left, "right": right}
            for bond_index in range(3):
                mask = (
                    (ds.frame_times_s >= left)
                    & (ds.frame_times_s <= right)
                    & np.isfinite(ds.signal_matrix[:, bond_index])
                )
                if not np.any(mask):
                    break
                processed, err = preprocess_signal(ds.frame_times_s[mask], ds.signal_matrix[:, bond_index][mask], longest=False, handlenan=False)
                if processed is None:
                    break
                amp, r2 = fit_sine_scan(np.asarray(processed.t, dtype=float), np.asarray(processed.y, dtype=float), freqs)
                idx = int(np.nanargmax(amp))
                row[f"amp{bond_index}"] = float(amp[idx])
                row[f"r2{bond_index}"] = float(r2[idx])
                row[f"freq{bond_index}"] = float(freqs[idx])
            else:
                row["proxy12"] = float(np.hypot(row["amp1"], row["amp2"]))
                rows.append(row)

        results[target.label] = {
            "region": np.asarray([row["region"] for row in rows], dtype=int),
            "proxy12": np.asarray([row["proxy12"] for row in rows], dtype=float),
            "amp0": np.asarray([row["amp0"] for row in rows], dtype=float),
            "r20": np.asarray([row["r20"] for row in rows], dtype=float),
            "freq0": np.asarray([row["freq0"] for row in rows], dtype=float),
            "amp1": np.asarray([row["amp1"] for row in rows], dtype=float),
            "amp2": np.asarray([row["amp2"] for row in rows], dtype=float),
        }
    return results


def plot_scaling_panel(writer: PlotWriter, results, cfg: Config) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, target in zip(axes.ravel(), cfg.target_bands):
        data = results[target.label]
        sc = ax.scatter(data["proxy12"], data["amp0"], c=data["r20"], cmap="viridis", s=70, edgecolors="black", linewidths=0.3)
        for x_val, y_val, region_id in zip(data["proxy12"], data["amp0"], data["region"]):
            ax.annotate(str(int(region_id)), (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
        corr = float(np.corrcoef(data["proxy12"], data["amp0"])[0, 1])
        slope = float(np.polyfit(np.log(data["proxy12"]), np.log(data["amp0"]), 1)[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"leave-one-out proxy $\sqrt{A_1^2 + A_2^2}$")
        ax.set_ylabel("bond 0 amplitude")
        ax.set_title(f"{target.label} | {target.known_kind}\ncorr={corr:.3f}, slope={slope:.3f}")
        ax.grid(alpha=0.25, which="both")
        fig.colorbar(sc, ax=ax, label="bond 0 fit $R^2$")
    fig.suptitle("Known-mode test: scaling with the calibrated modal proxy")
    return writer.save(fig, "scaling_panel")


def plot_frequency_panel(writer: PlotWriter, results, cfg: Config) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, target in zip(axes.ravel(), cfg.target_bands):
        data = results[target.label]
        ax.plot(data["region"], data["freq0"], "o-", color="tab:blue")
        ax.axhline(target.center_hz, color="tab:red", linestyle="--", lw=1.0)
        ax.axhline(target.center_hz - target.half_width_hz, color="0.5", linestyle=":", lw=0.8)
        ax.axhline(target.center_hz + target.half_width_hz, color="0.5", linestyle=":", lw=0.8)
        ax.set_xlabel("region index")
        ax.set_ylabel("best frequency (Hz)")
        ax.set_title(
            f"{target.label}\nrange=[{np.min(data['freq0']):.4f}, {np.max(data['freq0']):.4f}] Hz"
        )
        ax.grid(alpha=0.25)
    fig.suptitle("Frequency stability inside each target band")
    return writer.save(fig, "frequency_panel")


def plot_quality_panel(writer: PlotWriter, results, cfg: Config) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for ax, target in zip(axes.ravel(), cfg.target_bands):
        data = results[target.label]
        ax.plot(data["region"], data["r20"], "o-", color="tab:green")
        ax.axhline(0.15, color="tab:red", linestyle="--", lw=1.0)
        ax.set_xlabel("region index")
        ax.set_ylabel("bond 0 fit $R^2$")
        ax.set_title(f"{target.label}\nmean $R^2$={np.mean(data['r20']):.4f}")
        ax.grid(alpha=0.25)
    fig.suptitle("Time-domain fit quality for each target")
    return writer.save(fig, "quality_panel")


def build_summary(results, cfg: Config, plot_paths: list[Path]) -> str:
    lines = [
        f"repo_root: {REPO_ROOT}",
        f"dataset: {cfg.dataset}",
        f"bond_spacing_mode: {cfg.bond_spacing_mode}",
        f"component: {cfg.component}",
        f"begin_trim_s: {cfg.begin_trim_s}",
        f"end_trim_s: {cfg.end_trim_s}",
        "",
        "interpretive rule:",
        "A fundamental should look more like 3.35 Hz: stable best-fit frequency, strong bond-to-bond coherence, and near-linear scaling against the leave-one-out modal proxy.",
        "A non-fundamental can still be visible in a log FFT, but it should look less stable and less clean under this calibrated time-domain test.",
        "",
        "results:",
    ]
    for target in cfg.target_bands:
        data = results[target.label]
        corr = float(np.corrcoef(data["proxy12"], data["amp0"])[0, 1])
        slope = float(np.polyfit(np.log(data["proxy12"]), np.log(data["amp0"]), 1)[0])
        lines.append(
            f"{target.label} ({target.known_kind}): corr={corr:.3f}, slope={slope:.3f}, "
            f"meanR2={np.mean(data['r20']):.5f}, freq-range=[{np.min(data['freq0']):.4f}, {np.max(data['freq0']):.4f}] Hz"
        )
        lines.append(
            "  "
            + str(
                [
                    (
                        int(region),
                        round(float(amp0), 3),
                        round(float(proxy), 3),
                        round(float(r20), 4),
                        round(float(freq0), 4),
                    )
                    for region, amp0, proxy, r20, freq0 in zip(
                        data["region"], data["amp0"], data["proxy12"], data["r20"], data["freq0"]
                    )
                ]
            )
        )
    lines.extend(
        [
            "",
            "reading this table:",
            "3.35 Hz is the reference case and should look best.",
            "6.35 Hz and 8.96 Hz can still qualify as fundamental here if they inherit the same scaling pattern and frequency stability, even if their absolute R^2 is smaller because the lines are weaker.",
            "18 Hz should fail at least one of these tests more obviously.",
            "",
            "saved_plots:",
        ]
    )
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    ds, regions = detect_regions(CONFIG)
    results = analyze_targets(ds, regions, CONFIG)
    plot_paths = [
        plot_scaling_panel(writer, results, CONFIG),
        plot_frequency_panel(writer, results, CONFIG),
        plot_quality_panel(writer, results, CONFIG),
    ]
    summary_path = save_summary(build_summary(results, CONFIG, plot_paths))
    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
