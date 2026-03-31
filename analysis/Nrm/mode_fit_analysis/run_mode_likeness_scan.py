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
    sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    begin_trim_s: float = 8.0
    end_trim_s: float = 10.0
    scan_range_hz: tuple[float, float] = (2.0, 20.0)
    n_targets: int = 91
    half_width_hz: float = 0.5
    n_local_scan: int = 151
    good_r2_threshold: float = 0.01


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "scan_output"


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


def detect_regions(cfg: Config):
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
            region_entry["bonds"].append(processed)
        processed_by_region.append(region_entry)
    return processed_by_region


def evaluate_targets(processed_regions, cfg: Config):
    targets = np.linspace(cfg.scan_range_hz[0], cfg.scan_range_hz[1], cfg.n_targets)
    records = []

    for target_hz in targets:
        rows = []
        freqs = np.linspace(target_hz - cfg.half_width_hz, target_hz + cfg.half_width_hz, cfg.n_local_scan)
        for region in processed_regions:
            row = {"region": region["region"]}
            for bond_index, processed in enumerate(region["bonds"]):
                amp, r2 = fit_sine_scan(np.asarray(processed.t), np.asarray(processed.y), freqs)
                idx = int(np.nanargmax(amp))
                row[f"amp{bond_index}"] = float(amp[idx])
                row[f"r2{bond_index}"] = float(r2[idx])
                row[f"freq{bond_index}"] = float(freqs[idx])
            row["proxy12"] = float(np.hypot(row["amp1"], row["amp2"]))
            rows.append(row)

        amp0 = np.asarray([row["amp0"] for row in rows], dtype=float)
        amp1 = np.asarray([row["amp1"] for row in rows], dtype=float)
        amp2 = np.asarray([row["amp2"] for row in rows], dtype=float)
        proxy12 = np.asarray([row["proxy12"] for row in rows], dtype=float)
        r20 = np.asarray([row["r20"] for row in rows], dtype=float)
        f0 = np.asarray([row["freq0"] for row in rows], dtype=float)
        f1 = np.asarray([row["freq1"] for row in rows], dtype=float)
        f2 = np.asarray([row["freq2"] for row in rows], dtype=float)

        corr_proxy = float(np.corrcoef(proxy12, amp0)[0, 1])
        slope_proxy = float(np.polyfit(np.log(proxy12), np.log(amp0), 1)[0])
        corr01 = float(np.corrcoef(amp0, amp1)[0, 1])
        corr02 = float(np.corrcoef(amp0, amp2)[0, 1])
        corr12 = float(np.corrcoef(amp1, amp2)[0, 1])
        mean_r2 = float(np.mean(r20))
        good_frac = float(np.mean(r20 > cfg.good_r2_threshold))
        freq_span = float(np.max(f0) - np.min(f0))
        freq_rms = float(np.sqrt(np.mean((f0 - np.mean(f0)) ** 2)))
        bond_freq_spread = float(
            np.mean(
                [
                    np.mean(np.abs(f0 - f1)),
                    np.mean(np.abs(f0 - f2)),
                    np.mean(np.abs(f1 - f2)),
                ]
            )
        )

        # Score favors coherence, linear scaling, and stability; penalizes wandering frequencies.
        score = (
            corr_proxy
            + 0.5 * (corr01 + corr02 + corr12) / 3.0
            + mean_r2
            + 0.4 * good_frac
            - 0.8 * abs(slope_proxy - 1.0)
            - 1.2 * freq_rms
            - 0.8 * bond_freq_spread
        )

        records.append(
            {
                "target_hz": target_hz,
                "corr_proxy": corr_proxy,
                "slope_proxy": slope_proxy,
                "corr01": corr01,
                "corr02": corr02,
                "corr12": corr12,
                "mean_r2": mean_r2,
                "good_frac": good_frac,
                "freq_span": freq_span,
                "freq_rms": freq_rms,
                "bond_freq_spread": bond_freq_spread,
                "score": score,
                "amp0": amp0,
                "proxy12": proxy12,
                "r20": r20,
                "freq0": f0,
            }
        )

    return records


def plot_score_scan(writer: PlotWriter, records) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    target_hz = np.asarray([record["target_hz"] for record in records], dtype=float)
    score = np.asarray([record["score"] for record in records], dtype=float)
    corr_proxy = np.asarray([record["corr_proxy"] for record in records], dtype=float)
    slope_proxy = np.asarray([record["slope_proxy"] for record in records], dtype=float)
    mean_r2 = np.asarray([record["mean_r2"] for record in records], dtype=float)
    freq_rms = np.asarray([record["freq_rms"] for record in records], dtype=float)

    axes[0].plot(target_hz, score, color="tab:blue")
    axes[0].set_ylabel("score")
    axes[0].set_title("Mode-likeness score")
    axes[0].grid(alpha=0.25)

    axes[1].plot(target_hz, corr_proxy, color="tab:green")
    axes[1].axhline(0.9, color="0.5", linestyle="--", lw=0.8)
    axes[1].set_ylabel("corr")
    axes[1].set_title("Scaling correlation against leave-one-out proxy")
    axes[1].grid(alpha=0.25)

    axes[2].plot(target_hz, slope_proxy, color="tab:orange")
    axes[2].axhline(1.0, color="tab:red", linestyle="--", lw=0.8)
    axes[2].set_ylabel("slope")
    axes[2].set_title("Log-log scaling slope")
    axes[2].grid(alpha=0.25)

    axes[3].plot(target_hz, mean_r2, color="tab:purple", label="mean R^2")
    axes[3].plot(target_hz, freq_rms, color="tab:brown", label="freq rms")
    axes[3].set_ylabel("quality")
    axes[3].set_xlabel("target frequency (Hz)")
    axes[3].set_title("Fit quality and frequency stability")
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc="best")

    fig.suptitle("Label-free scan for mode-like frequencies")
    return writer.save(fig, "score_scan")


def plot_top_candidates(writer: PlotWriter, top_records) -> Path:
    fig, axes = plt.subplots(len(top_records), 2, figsize=(12, 4.2 * len(top_records)), constrained_layout=True)
    if len(top_records) == 1:
        axes = np.asarray([axes])
    for (ax_scaling, ax_freq), record in zip(axes, top_records):
        ax_scaling.scatter(record["proxy12"], record["amp0"], c=record["r20"], cmap="viridis", s=65, edgecolors="black", linewidths=0.3)
        ax_scaling.set_xscale("log")
        ax_scaling.set_yscale("log")
        ax_scaling.set_xlabel(r"leave-one-out proxy $\sqrt{A_1^2 + A_2^2}$")
        ax_scaling.set_ylabel("bond 0 amp")
        ax_scaling.set_title(
            f"{record['target_hz']:.3f} Hz | score={record['score']:.3f}\n"
            f"corr={record['corr_proxy']:.3f}, slope={record['slope_proxy']:.3f}, meanR2={record['mean_r2']:.4f}"
        )
        ax_scaling.grid(alpha=0.25, which="both")

        ax_freq.plot(record["freq0"], "o-", color="tab:blue")
        ax_freq.axhline(record["target_hz"], color="tab:red", linestyle="--", lw=1.0)
        ax_freq.set_xlabel("region index")
        ax_freq.set_ylabel("best freq (Hz)")
        ax_freq.set_title(
            f"{record['target_hz']:.3f} Hz | freq rms={record['freq_rms']:.4f}\n"
            f"bond-spread={record['bond_freq_spread']:.4f}"
        )
        ax_freq.grid(alpha=0.25)
    fig.suptitle("Top mode-like candidates from the label-free scan")
    return writer.save(fig, "top_candidates")


def build_summary(records, cfg: Config, plot_paths: list[Path]) -> str:
    sorted_records = sorted(records, key=lambda item: item["score"], reverse=True)
    lines = [
        f"repo_root: {REPO_ROOT}",
        f"dataset: {cfg.dataset}",
        f"bond_spacing_mode: {cfg.bond_spacing_mode}",
        f"component: {cfg.component}",
        f"scan_range_hz: {cfg.scan_range_hz}",
        f"n_targets: {cfg.n_targets}",
        "",
        "idea:",
        "A mode-like line should emerge from the data as one that is stable in frequency across quiet regions, coherent across bonds, and close to linear when scaled against a leave-one-out modal proxy.",
        "",
        "top candidates by score:",
    ]
    for record in sorted_records[:12]:
        lines.append(
            f"{record['target_hz']:.3f} Hz | score={record['score']:.3f} | corr={record['corr_proxy']:.3f} | "
            f"slope={record['slope_proxy']:.3f} | meanR2={record['mean_r2']:.5f} | "
            f"freq_rms={record['freq_rms']:.5f} | bond_freq_spread={record['bond_freq_spread']:.5f}"
        )
    lines.extend(
        [
            "",
            "reference frequencies of interest:",
        ]
    )
    for f_ref in (3.35, 6.35, 8.96, 18.0):
        nearest = min(records, key=lambda item: abs(item["target_hz"] - f_ref))
        lines.append(
            f"{nearest['target_hz']:.3f} Hz near {f_ref:.2f} | score={nearest['score']:.3f} | corr={nearest['corr_proxy']:.3f} | "
            f"slope={nearest['slope_proxy']:.3f} | meanR2={nearest['mean_r2']:.5f} | freq_rms={nearest['freq_rms']:.5f}"
        )
    lines.extend(
        [
            "",
            "what to look for:",
            "If the scan is doing something real, known fundamentals should rise toward the top and known non-fundamentals should not.",
            "Weak but real fundamentals may still have tiny absolute R^2, so cross-bond coherence and frequency stability matter at least as much as raw fit strength.",
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
    processed_regions = collect_processed_regions(ds, regions)
    records = evaluate_targets(processed_regions, CONFIG)
    sorted_records = sorted(records, key=lambda item: item["score"], reverse=True)
    plot_paths = [
        plot_score_scan(writer, records),
        plot_top_candidates(writer, sorted_records[:6]),
    ]
    summary_path = save_summary(build_summary(records, CONFIG, plot_paths))
    print(f"Saved plots to {OUTPUT_DIR}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
