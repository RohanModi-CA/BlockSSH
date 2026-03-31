from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Triad:
    label: str
    f1_hz: float
    f2_hz: float
    f3_hz: float


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    component: str = "x"
    bond_index: int = 0
    bond_spacing_mode: str = "default"

    broadband_sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    ignore_beginning_len_s: float = 0.0
    ignore_end_len_s: float = 4.0
    ignore_first_segment: bool = True
    ignore_last_segment: bool = True

    welch_len_s: float = 10.0
    welch_overlap_fraction: float = 0.5
    search_half_width_hz: float = 0.25
    min_windows_per_region: int = 3
    amplitude_gate_fraction: float = 0.20

    null_trials: int = 400
    null_seed: int = 0

    scan_lines_hz: tuple[float, ...] = (0.41, 3.35, 6.35, 8.96, 12.0, 16.65, 18.0)
    target_sum_hz: float = 18.0
    scan_half_width_hz: float = 0.30
    scan_f1_grid_hz: tuple[float, ...] = (
        0.41,
        3.35,
        6.35,
        8.96,
    )
    scan_f2_grid_hz: tuple[float, ...] = (
        0.41,
        3.35,
        6.35,
        8.96,
    )

    triads: tuple[Triad, ...] = (
        Triad(label="8.96 + 8.96 -> 18.0", f1_hz=8.96, f2_hz=8.96, f3_hz=18.0),
        Triad(label="8.96 + 8.96 -> 17.5 (control)", f1_hz=8.96, f2_hz=8.96, f3_hz=17.5),
        Triad(label="8.96 + 8.96 -> 18.5 (control)", f1_hz=8.96, f2_hz=8.96, f3_hz=18.5),
        Triad(label="6.35 + 8.96 -> 15.31", f1_hz=6.35, f2_hz=8.96, f3_hz=15.31),
        Triad(label="6.35 + 6.35 -> 12.70", f1_hz=6.35, f2_hz=6.35, f3_hz=12.70),
        Triad(label="3.35 + 8.96 -> 12.31", f1_hz=3.35, f2_hz=8.96, f3_hz=12.31),
        Triad(label="3.35 + 6.35 -> 9.70", f1_hz=3.35, f2_hz=6.35, f3_hz=9.70),
        Triad(label="0.41 + 8.96 -> 9.37", f1_hz=0.41, f2_hz=8.96, f3_hz=9.37),
    )


CONFIG = Config()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "bicoherence_output"


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
from analysis.tools.signal import hann_window_periodic, next_power_of_two, preprocess_signal


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


def compute_complex_spectrogram_with_overlap(y: np.ndarray, fs: float, welch_len_s: float, overlap_fraction: float):
    n = len(y)
    nperseg = max(8, int(round(welch_len_s * fs)))
    nperseg = min(nperseg, n)
    if nperseg < 8:
        return None
    noverlap = min(int(round(overlap_fraction * nperseg)), nperseg - 1)
    nfft = max(nperseg, next_power_of_two(nperseg))
    window = hann_window_periodic(nperseg)
    freq, time, s_complex = sp_signal.spectrogram(
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
        mode="complex",
    )
    return np.asarray(freq, dtype=float), np.asarray(time, dtype=float), np.asarray(s_complex, dtype=np.complex128)


def load_primary_processed(cfg: Config):
    base_dataset, _ = split_dataset_component(cfg.dataset)
    ds = load_bond_signal_dataset(
        dataset=f"{base_dataset}_{cfg.component}",
        bond_spacing_mode=cfg.bond_spacing_mode,
        component=cfg.component,
    )
    processed, err = preprocess_signal(
        ds.frame_times_s,
        ds.signal_matrix[:, cfg.bond_index],
        longest=False,
        handlenan=False,
    )
    if processed is None:
        raise ValueError(f"Preprocess failed: {err}")
    return base_dataset, ds, processed


def detect_regions(ds, processed, cfg: Config) -> list[dict[str, object]]:
    spec = compute_complex_spectrogram_with_overlap(
        processed.y,
        processed.Fs,
        cfg.broadband_sliding_len_s,
        0.9,
    )
    if spec is None:
        raise ValueError("Could not compute broadband spectrogram.")
    freq, time, s_complex = spec
    broadband_energy = np.sum(np.abs(s_complex), axis=0)
    t_global = time + processed.t[0]
    peak_indices, _ = find_peaks(broadband_energy)
    peak_times = t_global[peak_indices]
    if cfg.manual_peak_times_s:
        peak_times = np.sort(
            np.unique(np.concatenate([peak_times, np.asarray(cfg.manual_peak_times_s, dtype=float)]))
        )

    segment_edges = np.concatenate(([float(t_global[0])], peak_times, [float(t_global[-1])]))
    durations = np.diff(segment_edges)
    usable = durations >= float(cfg.min_segment_len_s)
    if usable.size > 0 and cfg.ignore_first_segment:
        usable[0] = False
    if usable.size > 0 and cfg.ignore_last_segment:
        usable[-1] = False

    t_raw = np.asarray(ds.frame_times_s, dtype=float)
    y_raw = np.asarray(ds.signal_matrix[:, cfg.bond_index], dtype=float)
    regions: list[dict[str, object]] = []
    for region_index in range(len(durations)):
        if not usable[region_index]:
            continue
        left = float(segment_edges[region_index] + cfg.ignore_beginning_len_s)
        right = float(segment_edges[region_index + 1] - cfg.ignore_end_len_s)
        if right <= left:
            continue
        mask = (t_raw >= left) & (t_raw <= right) & np.isfinite(y_raw)
        proc, err = preprocess_signal(t_raw[mask], y_raw[mask], longest=False, handlenan=False)
        if proc is None:
            continue
        regions.append(
            {
                "region_index": region_index,
                "left": left,
                "right": right,
                "processed": proc,
            }
        )
    return regions


def extract_local_component(freq: np.ndarray, s_complex: np.ndarray, center_hz: float, half_width_hz: float):
    mask = np.abs(freq - float(center_hz)) <= float(half_width_hz)
    if not np.any(mask):
        raise ValueError(f"No bins near {center_hz:.4f} Hz")
    band_freq = freq[mask]
    band_spec = s_complex[mask, :]
    amp = np.abs(band_spec)
    idx = np.argmax(amp, axis=0)
    chosen = band_spec[idx, np.arange(band_spec.shape[1])]
    return {
        "freq_hz": band_freq[idx],
        "complex": chosen,
        "amplitude": np.abs(chosen),
        "phase": np.angle(chosen),
    }


def pooled_bicoherence_from_pairs(z1: list[np.ndarray], z2: list[np.ndarray], z3: list[np.ndarray]) -> float:
    if not z1:
        return np.nan
    x1 = np.concatenate(z1)
    x2 = np.concatenate(z2)
    x3 = np.concatenate(z3)
    num = np.abs(np.sum(x1 * x2 * np.conj(x3))) ** 2
    den = np.sum(np.abs(x1 * x2) ** 2) * np.sum(np.abs(x3) ** 2)
    if den <= 0:
        return np.nan
    return float(num / den)


def evaluate_triad(regions: list[dict[str, object]], triad: Triad, cfg: Config) -> dict[str, object]:
    rng = np.random.default_rng(cfg.null_seed)
    z1_regions: list[np.ndarray] = []
    z2_regions: list[np.ndarray] = []
    z3_regions: list[np.ndarray] = []
    region_rows: list[dict[str, float]] = []

    for region in regions:
        processed = region["processed"]
        spec = compute_complex_spectrogram_with_overlap(
            processed.y,
            processed.Fs,
            cfg.welch_len_s,
            cfg.welch_overlap_fraction,
        )
        if spec is None:
            continue
        freq, _, s_complex = spec
        c1 = extract_local_component(freq, s_complex, triad.f1_hz, cfg.search_half_width_hz)
        c2 = extract_local_component(freq, s_complex, triad.f2_hz, cfg.search_half_width_hz)
        c3 = extract_local_component(freq, s_complex, triad.f3_hz, cfg.search_half_width_hz)

        gate1 = float(np.median(c1["amplitude"])) * cfg.amplitude_gate_fraction
        gate2 = float(np.median(c2["amplitude"])) * cfg.amplitude_gate_fraction
        gate3 = float(np.median(c3["amplitude"])) * cfg.amplitude_gate_fraction
        keep = (c1["amplitude"] >= gate1) & (c2["amplitude"] >= gate2) & (c3["amplitude"] >= gate3)
        if int(np.sum(keep)) < cfg.min_windows_per_region:
            continue

        z1 = np.asarray(c1["complex"][keep], dtype=np.complex128)
        z2 = np.asarray(c2["complex"][keep], dtype=np.complex128)
        z3 = np.asarray(c3["complex"][keep], dtype=np.complex128)
        z1_regions.append(z1)
        z2_regions.append(z2)
        z3_regions.append(z3)

        num = np.abs(np.sum(z1 * z2 * np.conj(z3))) ** 2
        den = np.sum(np.abs(z1 * z2) ** 2) * np.sum(np.abs(z3) ** 2)
        bic = float(num / den) if den > 0 else np.nan
        phase_resid = np.angle(z1) + np.angle(z2) - np.angle(z3)
        plv = float(np.abs(np.mean(np.exp(1j * phase_resid))))
        region_rows.append(
            {
                "region_index": int(region["region_index"]),
                "n_windows": int(np.sum(keep)),
                "bicoherence": bic,
                "phase_plv": plv,
                "f1_mean_hz": float(np.mean(c1["freq_hz"][keep])),
                "f2_mean_hz": float(np.mean(c2["freq_hz"][keep])),
                "f3_mean_hz": float(np.mean(c3["freq_hz"][keep])),
            }
        )

    observed = pooled_bicoherence_from_pairs(z1_regions, z2_regions, z3_regions)
    null_values: list[float] = []
    for _ in range(cfg.null_trials):
        sh1: list[np.ndarray] = []
        sh2: list[np.ndarray] = []
        sh3: list[np.ndarray] = []
        for a, b, c in zip(z1_regions, z2_regions, z3_regions):
            shift = int(rng.integers(1, len(c)))
            sh1.append(a)
            sh2.append(b)
            sh3.append(np.roll(c, shift))
        null_values.append(pooled_bicoherence_from_pairs(sh1, sh2, sh3))
    null_arr = np.asarray(null_values, dtype=float)

    mean_amp_3 = []
    mean_amp_prod = []
    for a, b, c in zip(z1_regions, z2_regions, z3_regions):
        mean_amp_3.append(float(np.mean(np.abs(c))))
        mean_amp_prod.append(float(np.mean(np.abs(a) * np.abs(b))))
    if len(mean_amp_3) >= 3:
        amp_corr = float(
            np.corrcoef(
                np.log(np.maximum(mean_amp_prod, 1e-12)),
                np.log(np.maximum(mean_amp_3, 1e-12)),
            )[0, 1]
        )
    else:
        amp_corr = np.nan

    return {
        "label": triad.label,
        "f1_hz": triad.f1_hz,
        "f2_hz": triad.f2_hz,
        "f3_hz": triad.f3_hz,
        "observed_bicoherence": observed,
        "null_mean_bicoherence": float(np.nanmean(null_arr)) if null_arr.size else np.nan,
        "null_q95_bicoherence": float(np.nanquantile(null_arr, 0.95)) if null_arr.size else np.nan,
        "n_regions_used": len(region_rows),
        "n_windows_total": int(sum(int(row["n_windows"]) for row in region_rows)),
        "amp_prod_corr": amp_corr,
        "region_rows": region_rows,
    }


def scan_target18_bicoherence(regions: list[dict[str, object]], cfg: Config) -> dict[str, np.ndarray]:
    f1_grid = np.asarray(cfg.scan_f1_grid_hz, dtype=float)
    f2_grid = np.asarray(cfg.scan_f2_grid_hz, dtype=float)
    out = np.full((len(f1_grid), len(f2_grid)), np.nan, dtype=float)
    freq_err = np.full_like(out, np.nan)

    for i, f1 in enumerate(f1_grid):
        for j, f2 in enumerate(f2_grid):
            triad = Triad(label="", f1_hz=float(f1), f2_hz=float(f2), f3_hz=cfg.target_sum_hz)
            result = evaluate_triad(regions, triad, cfg)
            out[i, j] = float(result["observed_bicoherence"])
            freq_err[i, j] = abs((f1 + f2) - cfg.target_sum_hz)
    return {
        "f1_grid_hz": f1_grid,
        "f2_grid_hz": f2_grid,
        "bicoherence": out,
        "sum_error_hz": freq_err,
    }


def plot_triad_scores(writer: PlotWriter, triad_results: list[dict[str, object]]) -> Path:
    labels = [str(row["label"]) for row in triad_results]
    observed = np.asarray([float(row["observed_bicoherence"]) for row in triad_results], dtype=float)
    null_mean = np.asarray([float(row["null_mean_bicoherence"]) for row in triad_results], dtype=float)
    null_q95 = np.asarray([float(row["null_q95_bicoherence"]) for row in triad_results], dtype=float)

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(x - 0.18, observed, width=0.36, color="C0", label="observed")
    ax.bar(x + 0.18, null_mean, width=0.36, color="C1", label="null mean")
    ax.scatter(x + 0.18, null_q95, marker="_", s=700, linewidths=2.0, color="black", label="null 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("pooled normalized bicoherence")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    ax.set_title("Triad tests from Welch-style complex windows")
    return writer.save(fig, "triad_bicoherence_scores")


def plot_regionwise_bicoherence(writer: PlotWriter, triad_results: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(len(triad_results), 1, figsize=(11, 2.7 * len(triad_results)), sharex=True, constrained_layout=True)
    if len(triad_results) == 1:
        axes = [axes]
    for ax, result in zip(axes, triad_results):
        rows = result["region_rows"]
        region_idx = np.asarray([int(row["region_index"]) for row in rows], dtype=int)
        bic = np.asarray([float(row["bicoherence"]) for row in rows], dtype=float)
        nwin = np.asarray([int(row["n_windows"]) for row in rows], dtype=float)
        if region_idx.size:
            ax.scatter(region_idx, bic, s=18.0 + 4.0 * nwin, color="C0", edgecolor="black", linewidth=0.4, alpha=0.75)
        ax.axhline(float(result["null_q95_bicoherence"]), color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_ylabel("bic")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.set_title(str(result["label"]))
    axes[-1].set_xlabel("region index")
    return writer.save(fig, "regionwise_bicoherence")


def plot_target18_scan(writer: PlotWriter, scan: dict[str, np.ndarray]) -> Path:
    f1_grid = np.asarray(scan["f1_grid_hz"], dtype=float)
    f2_grid = np.asarray(scan["f2_grid_hz"], dtype=float)
    bic = np.asarray(scan["bicoherence"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    image = ax.imshow(bic, origin="lower", cmap="magma", aspect="auto")
    fig.colorbar(image, ax=ax, label="observed bicoherence to 18 Hz")
    ax.set_xticks(np.arange(len(f2_grid)))
    ax.set_xticklabels([f"{x:.2f}" for x in f2_grid])
    ax.set_yticks(np.arange(len(f1_grid)))
    ax.set_yticklabels([f"{x:.2f}" for x in f1_grid])
    ax.set_xlabel("f2 (Hz)")
    ax.set_ylabel("f1 (Hz)")
    ax.set_title("Which fundamental pairs best predict 18 Hz?")
    return writer.save(fig, "target18_pair_scan")


def build_summary(base_dataset: str, regions: list[dict[str, object]], triad_results: list[dict[str, object]], scan: dict[str, np.ndarray], saved_paths: list[Path], cfg: Config) -> str:
    lines: list[str] = []
    lines.append(f"repo_root: {REPO_ROOT}")
    lines.append(f"dataset: {base_dataset}")
    lines.append(f"bond_spacing_mode: {cfg.bond_spacing_mode}")
    lines.append(f"component: {cfg.component}")
    lines.append(f"bond_index: {cfg.bond_index}")
    lines.append(f"welch_len_s: {cfg.welch_len_s}")
    lines.append(f"welch_overlap_fraction: {cfg.welch_overlap_fraction}")
    lines.append("")
    lines.append("quiet regions:")
    for region in regions:
        lines.append(
            f"region {int(region['region_index'])}: "
            f"t=[{float(region['left']):.3f}, {float(region['right']):.3f}] s"
        )
    lines.append("")
    lines.append("triad results:")
    for row in triad_results:
        verdict = "above null" if float(row["observed_bicoherence"]) > float(row["null_q95_bicoherence"]) else "not above null"
        lines.append(
            f"{row['label']}: observed={float(row['observed_bicoherence']):.5f}, "
            f"null mean={float(row['null_mean_bicoherence']):.5f}, "
            f"null 95%={float(row['null_q95_bicoherence']):.5f}, "
            f"regions={int(row['n_regions_used'])}, windows={int(row['n_windows_total'])}, "
            f"corr(log |X1X2|, log |X3|)={float(row['amp_prod_corr']):.3f} | {verdict}"
        )
        for region_row in row["region_rows"]:
            lines.append(
                f"  region {int(region_row['region_index'])}: bic={float(region_row['bicoherence']):.4f}, "
                f"phasePLV={float(region_row['phase_plv']):.3f}, windows={int(region_row['n_windows'])}, "
                f"f1={float(region_row['f1_mean_hz']):.3f}, f2={float(region_row['f2_mean_hz']):.3f}, f3={float(region_row['f3_mean_hz']):.3f}"
            )
    lines.append("")
    bic = np.asarray(scan["bicoherence"], dtype=float)
    f1_grid = np.asarray(scan["f1_grid_hz"], dtype=float)
    f2_grid = np.asarray(scan["f2_grid_hz"], dtype=float)
    lines.append("target-18 pair scan:")
    if np.any(np.isfinite(bic)):
        best_idx = np.unravel_index(int(np.nanargmax(bic)), bic.shape)
        lines.append(
            f"best pair among scanned fundamentals: f1={float(f1_grid[best_idx[0]]):.2f} Hz, "
            f"f2={float(f2_grid[best_idx[1]]):.2f} Hz, observed bic={float(bic[best_idx]):.5f}"
        )
    else:
        lines.append("no scanned pairs had enough usable windows")
    lines.append("")
    lines.append("interpretation:")
    lines.append("Bicoherence is a stronger test than ordinary coherence here because it directly asks whether two frequencies predict the phase and amplitude of a third.")
    lines.append("If a line is a true independent mode, we usually do not expect a large, repeatable bicoherence tying it to other fundamentals by a simple sum relation.")
    lines.append("If a line is a nonlinear follower, a relation like f3 = f1 + f2 should create an elevated normalized bispectral statistic.")
    lines.append("In this first pass, 8.96 + 8.96 -> 18.0 is the key triad of interest. Compare its observed value to the circular-shift null and to the nearby control triads above rather than looking at the raw number alone.")
    lines.append("")
    lines.append("saved_plots:")
    for path in saved_paths:
        lines.append(str(path))
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()
    base_dataset, ds, processed = load_primary_processed(CONFIG)
    regions = detect_regions(ds, processed, CONFIG)
    triad_results = [evaluate_triad(regions, triad, CONFIG) for triad in CONFIG.triads]
    scan = scan_target18_bicoherence(regions, CONFIG)
    saved_paths = [
        plot_triad_scores(writer, triad_results),
        plot_regionwise_bicoherence(writer, triad_results),
        plot_target18_scan(writer, scan),
    ]
    summary = build_summary(base_dataset, regions, triad_results, scan, saved_paths, CONFIG)
    summary_path = save_summary(summary)
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
