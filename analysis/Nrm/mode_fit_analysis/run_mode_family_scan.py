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
    datasets: tuple[str, ...] = ("IMG_0681_rot270", "IMG_0680_rot270", "CDX_10IC")
    components: tuple[str, ...] = ("x", "y", "a")
    bond_spacing_mode: str = "default"
    sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    begin_trim_s: float = 8.0
    end_trim_s: float = 10.0
    scan_range_hz: tuple[float, float] = (2.0, 20.0)
    n_targets: int = 181
    half_width_hz: float = 0.35
    n_local_scan: int = 101
    smooth_len_targets: int = 7
    family_min_distance_hz: float = 0.7
    family_prominence: float = 0.12
    reference_hz: tuple[float, ...] = (3.35, 6.35, 8.96, 12.0, 16.65, 18.0)


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "family_scan_output"


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


def detect_regions(dataset: str, component: str, cfg: Config):
    base_dataset, _ = split_dataset_component(dataset)
    ds = load_bond_signal_dataset(
        dataset=f"{base_dataset}_{component}",
        bond_spacing_mode=cfg.bond_spacing_mode,
        component=component,
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
    regions = []
    for i in range(len(durations)):
        if not usable[i]:
            continue
        left = float(edges[i] + cfg.begin_trim_s)
        right = float(edges[i + 1] - cfg.end_trim_s)
        if right > left:
            regions.append((i, left, right))
    return ds, regions


def collect_processed_regions(ds, regions):
    processed_by_region = []
    for region_index, left, right in regions:
        region_entry = {"region": region_index, "left": left, "right": right, "bonds": []}
        for bond_index in range(ds.signal_matrix.shape[1]):
            mask = (
                (ds.frame_times_s >= left)
                & (ds.frame_times_s <= right)
                & np.isfinite(ds.signal_matrix[:, bond_index])
            )
            processed, err = preprocess_signal(
                ds.frame_times_s[mask],
                ds.signal_matrix[:, bond_index][mask],
                longest=False,
                handlenan=False,
            )
            if processed is None:
                raise ValueError(f"Region {region_index}, bond {bond_index} preprocess failed: {err}")
            region_entry["bonds"].append(
                {
                    "processed": processed,
                    "fft": compute_one_sided_fft(np.asarray(processed.y), processed.dt),
                }
            )
        processed_by_region.append(region_entry)
    return processed_by_region


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_log_slope(x: np.ndarray, y: np.ndarray) -> float:
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def local_fft_prominence(fft, target_hz: float, half_width_hz: float) -> float:
    mask = (fft.freq >= target_hz - half_width_hz) & (fft.freq <= target_hz + half_width_hz)
    local_amp = fft.amplitude[mask]
    if local_amp.size < 3:
        return float("nan")
    peak_idx = int(np.argmax(local_amp))
    peak_amp = float(local_amp[peak_idx])
    side = np.delete(local_amp, peak_idx)
    baseline = float(np.median(side)) if side.size else float("nan")
    if not np.isfinite(baseline) or baseline <= 0.0:
        return float("nan")
    return peak_amp / baseline


def summarize_target(processed_regions, target_hz: float, cfg: Config) -> dict:
    scan_freqs = np.linspace(target_hz - cfg.half_width_hz, target_hz + cfg.half_width_hz, cfg.n_local_scan)
    n_regions = len(processed_regions)
    n_bonds = len(processed_regions[0]["bonds"])
    amps = np.full((n_regions, n_bonds), np.nan, dtype=float)
    r2s = np.full((n_regions, n_bonds), np.nan, dtype=float)
    best_freqs = np.full((n_regions, n_bonds), np.nan, dtype=float)
    fft_prom = np.full((n_regions, n_bonds), np.nan, dtype=float)

    for region_idx, region in enumerate(processed_regions):
        for bond_idx, bond_entry in enumerate(region["bonds"]):
            processed = bond_entry["processed"]
            amp, r2 = fit_sine_scan(np.asarray(processed.t), np.asarray(processed.y), scan_freqs)
            best_idx = int(np.nanargmax(amp))
            amps[region_idx, bond_idx] = float(amp[best_idx])
            r2s[region_idx, bond_idx] = float(r2[best_idx])
            best_freqs[region_idx, bond_idx] = float(scan_freqs[best_idx])
            fft_prom[region_idx, bond_idx] = local_fft_prominence(bond_entry["fft"], target_hz, cfg.half_width_hz)

    bond_records = []
    for response_bond in range(n_bonds):
        other_idx = [idx for idx in range(n_bonds) if idx != response_bond]
        proxy = np.sqrt(np.nansum(amps[:, other_idx] ** 2, axis=1))
        response = amps[:, response_bond]
        corr = safe_corr(proxy, response)
        slope = safe_log_slope(proxy, response)
        freq_rms = float(np.sqrt(np.nanmean((best_freqs[:, response_bond] - np.nanmean(best_freqs[:, response_bond])) ** 2)))
        mean_freq_gap = float(np.nanmean(np.abs(best_freqs[:, response_bond][:, None] - best_freqs[:, other_idx])))
        mean_r2 = float(np.nanmean(r2s[:, response_bond]))
        prom = float(np.nanmean(fft_prom[:, response_bond]))
        prom_score = np.log10(prom) if np.isfinite(prom) and prom > 0.0 else -1.0
        bond_score = (
            (corr if np.isfinite(corr) else -1.0)
            + 0.25 * mean_r2
            + 0.15 * prom_score
            - 0.8 * abs((slope if np.isfinite(slope) else 3.0) - 1.0)
            - 1.2 * freq_rms
            - 0.8 * mean_freq_gap
        )
        bond_records.append(
            {
                "response_bond": response_bond,
                "corr": corr,
                "slope": slope,
                "mean_r2": mean_r2,
                "freq_rms": freq_rms,
                "mean_freq_gap": mean_freq_gap,
                "mean_fft_prom": prom,
                "bond_score": bond_score,
            }
        )

    best_bond_record = max(bond_records, key=lambda item: item["bond_score"])
    corr_values = np.asarray([item["corr"] for item in bond_records], dtype=float)
    median_corr = float(np.nanmedian(corr_values))
    consensus_freq_gap = float(np.nanmean(np.nanstd(best_freqs, axis=1)))
    mean_fft_prom_all = float(np.nanmean(fft_prom))
    prom_all_score = np.log10(mean_fft_prom_all) if np.isfinite(mean_fft_prom_all) and mean_fft_prom_all > 0.0 else -1.0
    overall_score = (
        best_bond_record["bond_score"]
        + 0.35 * median_corr
        + 0.2 * prom_all_score
        - 1.0 * consensus_freq_gap
    )
    return {
        "target_hz": target_hz,
        "score": overall_score,
        "best_response_bond": best_bond_record["response_bond"],
        "best_corr": best_bond_record["corr"],
        "best_slope": best_bond_record["slope"],
        "best_mean_r2": best_bond_record["mean_r2"],
        "best_freq_rms": best_bond_record["freq_rms"],
        "best_mean_freq_gap": best_bond_record["mean_freq_gap"],
        "best_mean_fft_prom": best_bond_record["mean_fft_prom"],
        "median_corr": median_corr,
        "consensus_freq_gap": consensus_freq_gap,
        "mean_fft_prom_all": mean_fft_prom_all,
        "bond_records": bond_records,
    }


def smooth_values(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.copy()
    kernel = np.ones(length, dtype=float) / float(length)
    return np.convolve(values, kernel, mode="same")


def cluster_candidates(records, cfg: Config) -> list[dict]:
    target_hz = np.asarray([record["target_hz"] for record in records], dtype=float)
    score = np.asarray([record["score"] for record in records], dtype=float)
    smooth_score = smooth_values(score, cfg.smooth_len_targets)
    spacing_hz = float(target_hz[1] - target_hz[0])
    distance = max(1, int(round(cfg.family_min_distance_hz / spacing_hz)))
    peak_indices, props = find_peaks(smooth_score, distance=distance, prominence=cfg.family_prominence)
    candidates = []
    for peak_idx in peak_indices:
        record = records[int(peak_idx)]
        candidates.append(
            {
                "family_center_hz": float(record["target_hz"]),
                "score": float(record["score"]),
                "smooth_score": float(smooth_score[peak_idx]),
                "best_response_bond": int(record["best_response_bond"]),
                "best_corr": float(record["best_corr"]),
                "best_slope": float(record["best_slope"]),
                "best_mean_r2": float(record["best_mean_r2"]),
                "best_freq_rms": float(record["best_freq_rms"]),
                "best_mean_freq_gap": float(record["best_mean_freq_gap"]),
                "best_mean_fft_prom": float(record["best_mean_fft_prom"]),
                "median_corr": float(record["median_corr"]),
                "consensus_freq_gap": float(record["consensus_freq_gap"]),
            }
        )
    candidates.sort(key=lambda item: item["smooth_score"], reverse=True)
    return candidates


def run_scan_for_view(dataset: str, component: str, cfg: Config) -> dict:
    ds, regions = detect_regions(dataset, component, cfg)
    processed_regions = collect_processed_regions(ds, regions)
    targets = np.linspace(cfg.scan_range_hz[0], cfg.scan_range_hz[1], cfg.n_targets)
    records = [summarize_target(processed_regions, float(target_hz), cfg) for target_hz in targets]
    candidates = cluster_candidates(records, cfg)
    return {
        "dataset": dataset,
        "component": component,
        "n_bonds": ds.signal_matrix.shape[1],
        "n_regions": len(processed_regions),
        "records": records,
        "candidates": candidates,
    }


def plot_score_grid(writer: PlotWriter, results: list[dict], cfg: Config) -> Path:
    fig, axes = plt.subplots(len(cfg.datasets), len(cfg.components), figsize=(15, 10), sharex=True, sharey=True, constrained_layout=True)
    target_hz = np.linspace(cfg.scan_range_hz[0], cfg.scan_range_hz[1], cfg.n_targets)
    for row_idx, dataset in enumerate(cfg.datasets):
        for col_idx, component in enumerate(cfg.components):
            ax = axes[row_idx, col_idx]
            result = next(item for item in results if item["dataset"] == dataset and item["component"] == component)
            score = np.asarray([record["score"] for record in result["records"]], dtype=float)
            smooth_score = smooth_values(score, cfg.smooth_len_targets)
            ax.plot(target_hz, score, color="0.75", lw=1.0, label="raw")
            ax.plot(target_hz, smooth_score, color="tab:blue", lw=1.7, label="smoothed")
            for candidate in result["candidates"][:6]:
                ax.axvline(candidate["family_center_hz"], color="tab:red", alpha=0.25, lw=1.0)
            for ref_hz in cfg.reference_hz:
                ax.axvline(ref_hz, color="0.85", lw=0.6, linestyle="--")
            ax.set_title(f"{dataset} | {component}")
            ax.grid(alpha=0.25)
            if row_idx == len(cfg.datasets) - 1:
                ax.set_xlabel("frequency (Hz)")
            if col_idx == 0:
                ax.set_ylabel("family score")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Bond-agnostic family scan across datasets and observables")
    return writer.save(fig, "score_grid")


def plot_reference_summary(writer: PlotWriter, results: list[dict], cfg: Config) -> Path:
    fig, axes = plt.subplots(len(cfg.reference_hz), 1, figsize=(13, 2.4 * len(cfg.reference_hz)), constrained_layout=True)
    if len(cfg.reference_hz) == 1:
        axes = np.asarray([axes])
    labels = [f"{result['dataset']}:{result['component']}" for result in results]
    x = np.arange(len(labels))
    for ax, ref_hz in zip(axes, cfg.reference_hz):
        nearest = []
        for result in results:
            record = min(result["records"], key=lambda item: abs(item["target_hz"] - ref_hz))
            nearest.append(record)
        score = np.asarray([record["score"] for record in nearest], dtype=float)
        corr = np.asarray([record["best_corr"] for record in nearest], dtype=float)
        ax.bar(x - 0.18, score, width=0.36, label="score", color="tab:blue")
        ax.bar(x + 0.18, corr, width=0.36, label="best corr", color="tab:green")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(f"Nearest scan point to {ref_hz:.2f} Hz")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(loc="upper right")
    return writer.save(fig, "reference_summary")


def plot_best_candidates(writer: PlotWriter, results: list[dict]) -> Path:
    rows = []
    for result in results:
        for candidate in result["candidates"][:4]:
            rows.append(
                {
                    "label": f"{result['dataset']}:{result['component']}",
                    **candidate,
                }
            )
    rows.sort(key=lambda item: item["smooth_score"], reverse=True)
    top_rows = rows[:18]
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    y = np.arange(len(top_rows))
    ax.barh(y, [row["smooth_score"] for row in top_rows], color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['family_center_hz']:.2f} Hz | {row['label']} | b{row['best_response_bond']}" for row in top_rows])
    ax.invert_yaxis()
    ax.set_xlabel("smoothed family score")
    ax.set_title("Top candidate families across all scans")
    ax.grid(alpha=0.25, axis="x")
    return writer.save(fig, "best_candidates")


def build_summary(results: list[dict], plot_paths: list[Path], cfg: Config) -> str:
    lines = [
        f"repo_root: {REPO_ROOT}",
        f"bond_spacing_mode: {cfg.bond_spacing_mode}",
        f"datasets: {cfg.datasets}",
        f"components: {cfg.components}",
        f"scan_range_hz: {cfg.scan_range_hz}",
        f"n_targets: {cfg.n_targets}",
        "",
        "idea:",
        "The response bond should be allowed to vary with frequency. A higher mode can place a near-node on bond 0, so the scan now lets the strongest leave-one-out response bond emerge from the data at each candidate frequency.",
        "",
    ]
    for result in results:
        lines.append(f"{result['dataset']} | {result['component']} | n_regions={result['n_regions']} | n_bonds={result['n_bonds']}")
        if not result["candidates"]:
            lines.append("  no candidate families passed the current peak criteria")
        for candidate in result["candidates"][:8]:
            lines.append(
                f"  family {candidate['family_center_hz']:.3f} Hz | score={candidate['smooth_score']:.3f} | "
                f"best_bond={candidate['best_response_bond']} | corr={candidate['best_corr']:.3f} | "
                f"slope={candidate['best_slope']:.3f} | prom={candidate['best_mean_fft_prom']:.3f} | "
                f"freq_rms={candidate['best_freq_rms']:.4f} | consensus_gap={candidate['consensus_freq_gap']:.4f}"
            )
        lines.append("")

    lines.append("reference neighborhoods:")
    for ref_hz in cfg.reference_hz:
        lines.append(f"near {ref_hz:.2f} Hz")
        rows = []
        for result in results:
            nearest = min(result["records"], key=lambda item: abs(item["target_hz"] - ref_hz))
            rows.append(
                (
                    nearest["score"],
                    result["dataset"],
                    result["component"],
                    nearest["target_hz"],
                    nearest["best_response_bond"],
                    nearest["best_corr"],
                    nearest["best_slope"],
                    nearest["best_mean_r2"],
                    nearest["best_freq_rms"],
                    nearest["best_mean_fft_prom"],
                )
            )
        rows.sort(reverse=True)
        for row in rows:
            lines.append(
                f"  {row[1]} | {row[2]} | target={row[3]:.3f} | score={row[0]:.3f} | "
                f"best_bond={row[4]} | corr={row[5]:.3f} | slope={row[6]:.3f} | "
                f"meanR2={row[7]:.5f} | freq_rms={row[8]:.4f} | prom={row[9]:.3f}"
            )
        lines.append("")

    lines.append("saved_plots:")
    lines.extend(str(path) for path in plot_paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    results = []
    for dataset in CONFIG.datasets:
        for component in CONFIG.components:
            print(f"[scan] {dataset} | {component}")
            results.append(run_scan_for_view(dataset, component, CONFIG))
    plot_paths = [
        plot_score_grid(writer, results, CONFIG),
        plot_reference_summary(writer, results, CONFIG),
        plot_best_candidates(writer, results),
    ]
    summary_path = save_summary(build_summary(results, plot_paths, CONFIG))
    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
