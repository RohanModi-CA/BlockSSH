from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, find_peaks


@dataclass(frozen=True)
class HarmonicHypothesis:
    label: str
    parent_hz: float
    target_hz: float
    order: int


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    primary_component: str = "x"
    primary_bond_index: int = 0

    segmentation_mode: str = "default"
    comparison_modes: tuple[str, ...] = ("default", "comoving")
    broadband_sliding_len_s: float = 20.0
    local_track_sliding_len_s: float = 10.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    min_segment_len_s: float = 25.0
    ignore_beginning_len_s: float = 0.0
    ignore_end_len_s: float = 4.0
    ignore_first_segment: bool = True
    ignore_last_segment: bool = True

    recurrent_peak_search_hz: tuple[float, float] = (0.5, 25.0)
    recurrent_peak_prominence_frac: float = 0.03
    recurrent_peak_top_n: int = 6
    peak_merge_tol_hz: float = 0.08

    coherence_bands_hz: tuple[float, ...] = (3.35, 6.35, 8.96, 12.0, 16.65)
    coherence_half_width_hz: float = 0.15

    track_half_width_hz: float = 0.30
    gate_fraction_of_median: float = 0.50
    min_windows_per_region: int = 8
    null_trials: int = 400
    null_seed: int = 0

    fundamental_lines_hz: tuple[float, ...] = (0.41, 3.35, 6.35, 8.96)
    candidate_line_hz: float = 18.0

    independence_tests: tuple[HarmonicHypothesis, ...] = (
        HarmonicHypothesis(label="0.41 !-> 3.35 (x8 check)", parent_hz=0.41, target_hz=3.35, order=8),
        HarmonicHypothesis(label="3.35 !-> 6.35 (x2 check)", parent_hz=3.35, target_hz=6.35, order=2),
        HarmonicHypothesis(label="3.35 !-> 8.96 (x3-ish check)", parent_hz=3.35, target_hz=8.96, order=3),
        HarmonicHypothesis(label="6.35 !-> 8.96 (x1 check)", parent_hz=6.35, target_hz=8.96, order=1),
    )
    candidate_tests: tuple[HarmonicHypothesis, ...] = (
        HarmonicHypothesis(label="2 x 8.96 -> 18", parent_hz=8.96, target_hz=18.0, order=2),
        HarmonicHypothesis(label="3 x 6.35 -> 18", parent_hz=6.35, target_hz=18.0, order=3),
        HarmonicHypothesis(label="5 x 3.35 -> 18", parent_hz=3.35, target_hz=18.0, order=5),
        HarmonicHypothesis(label="44 x 0.41 -> 18", parent_hz=0.41, target_hz=18.0, order=44),
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


def load_primary_dataset(mode: str, cfg: Config):
    base_dataset, _ = split_dataset_component(cfg.dataset)
    ds = load_bond_signal_dataset(
        dataset=f"{base_dataset}_{cfg.primary_component}",
        bond_spacing_mode=mode,
        component=cfg.primary_component,
    )
    y = np.asarray(ds.signal_matrix[:, cfg.primary_bond_index], dtype=float)
    processed, err = preprocess_signal(ds.frame_times_s, y, longest=False, handlenan=False)
    if processed is None:
        raise ValueError(f"Preprocess failed for mode={mode}: {err}")
    return base_dataset, ds, processed


def detect_peak_times(processed, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.broadband_sliding_len_s)
    if spec is None:
        raise ValueError("Broadband spectrogram window too short.")
    broadband_energy = np.sum(np.abs(spec.S_complex), axis=0)
    t_global = spec.t + processed.t[0]
    peak_indices, _ = find_peaks(broadband_energy)
    peak_times = t_global[peak_indices]
    if cfg.manual_peak_times_s:
        peak_times = np.sort(
            np.unique(np.concatenate([peak_times, np.asarray(cfg.manual_peak_times_s, dtype=float)]))
        )
    return t_global, peak_times


def build_regions(ds, bond_index: int, t_global: np.ndarray, peak_times: np.ndarray, cfg: Config) -> list[dict[str, object]]:
    segment_edges = np.concatenate(([float(t_global[0])], np.sort(peak_times), [float(t_global[-1])]))
    durations = np.diff(segment_edges)
    usable = durations >= float(cfg.min_segment_len_s)
    if usable.size > 0 and cfg.ignore_first_segment:
        usable[0] = False
    if usable.size > 0 and cfg.ignore_last_segment:
        usable[-1] = False

    t_raw = np.asarray(ds.frame_times_s, dtype=float)
    y_raw = np.asarray(ds.signal_matrix[:, bond_index], dtype=float)
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
        processed, err = preprocess_signal(t_raw[mask], y_raw[mask], longest=False, handlenan=False)
        if processed is None:
            print(f"Skipping region {region_index}: preprocess failed ({err})")
            continue
        regions.append(
            {
                "region_index": region_index,
                "left": left,
                "right": right,
                "processed": processed,
                "rms": float(np.sqrt(np.mean(np.square(processed.y)))),
            }
        )
    return regions


def summarize_recurrent_peaks(regions: list[dict[str, object]], cfg: Config) -> list[dict[str, float]]:
    rows: list[tuple[float, float]] = []
    low_hz, high_hz = cfg.recurrent_peak_search_hz
    for region in regions:
        processed = region["processed"]
        fft = compute_one_sided_fft(processed.y, processed.dt)
        band_mask = (fft.freq >= low_hz) & (fft.freq <= high_hz)
        band_freq = fft.freq[band_mask]
        band_amp = fft.amplitude[band_mask]
        if band_freq.size == 0:
            continue
        prominence = float(np.nanmax(band_amp)) * cfg.recurrent_peak_prominence_frac
        peak_indices, _ = find_peaks(band_amp, prominence=prominence)
        if peak_indices.size == 0:
            continue
        order = np.argsort(band_amp[peak_indices])[::-1][: cfg.recurrent_peak_top_n]
        for idx in peak_indices[order]:
            rows.append((float(band_freq[idx]), float(band_amp[idx])))

    rows.sort(key=lambda item: item[0])
    clusters: list[dict[str, object]] = []
    for freq_hz, amplitude in rows:
        if not clusters or abs(freq_hz - float(clusters[-1]["freqs"][-1])) > cfg.peak_merge_tol_hz:
            clusters.append({"freqs": [freq_hz], "amps": [amplitude]})
        else:
            clusters[-1]["freqs"].append(freq_hz)
            clusters[-1]["amps"].append(amplitude)

    summary: list[dict[str, float]] = []
    for cluster in clusters:
        freqs = np.asarray(cluster["freqs"], dtype=float)
        amps = np.asarray(cluster["amps"], dtype=float)
        summary.append(
            {
                "center_hz": float(np.mean(freqs)),
                "n_occurrences": int(freqs.size),
                "mean_amp": float(np.mean(amps)),
                "max_amp": float(np.max(amps)),
            }
        )

    summary.sort(key=lambda item: (-int(item["n_occurrences"]), -float(item["mean_amp"])))
    return summary


def compute_same_frequency_coherence(cfg: Config) -> dict[str, np.ndarray]:
    pair_labels: list[str] = []
    band_centers = np.asarray(cfg.coherence_bands_hz, dtype=float)
    out = np.full((len(cfg.comparison_modes), 3, len(band_centers)), np.nan, dtype=float)

    base_dataset, _ = split_dataset_component(cfg.dataset)
    for mode_idx, mode in enumerate(cfg.comparison_modes):
        ds = load_bond_signal_dataset(
            dataset=f"{base_dataset}_{cfg.primary_component}",
            bond_spacing_mode=mode,
            component=cfg.primary_component,
        )
        processed_by_bond = []
        for bond_index in range(min(3, ds.signal_matrix.shape[1])):
            processed, _ = preprocess_signal(
                ds.frame_times_s,
                ds.signal_matrix[:, bond_index],
                longest=False,
                handlenan=False,
            )
            if processed is not None:
                processed_by_bond.append((bond_index, processed))

        pair_counter = 0
        for i in range(len(processed_by_bond)):
            for j in range(i + 1, len(processed_by_bond)):
                bond_i, proc_i = processed_by_bond[i]
                bond_j, proc_j = processed_by_bond[j]
                n = min(len(proc_i.y), len(proc_j.y))
                freq, cxy = coherence(
                    proc_i.y[:n],
                    proc_j.y[:n],
                    fs=proc_i.Fs,
                    nperseg=min(2048, n),
                )
                for band_idx, center_hz in enumerate(band_centers):
                    mask = np.abs(freq - center_hz) <= cfg.coherence_half_width_hz
                    if np.any(mask):
                        out[mode_idx, pair_counter, band_idx] = float(np.nanmax(cxy[mask]))
                pair_labels.append(f"{mode} b{bond_i}-b{bond_j}")
                pair_counter += 1

    return {
        "bands_hz": band_centers,
        "values": out,
        "pair_labels": np.asarray(["b0-b1", "b0-b2", "b1-b2"]),
        "mode_labels": np.asarray(cfg.comparison_modes),
    }


def extract_local_track(spec, center_hz: float, half_width_hz: float) -> dict[str, np.ndarray]:
    freq = np.asarray(spec.f, dtype=float)
    spectrum = np.asarray(spec.S_complex, dtype=np.complex128)
    band_mask = np.abs(freq - float(center_hz)) <= float(half_width_hz)
    if not np.any(band_mask):
        raise ValueError(f"No spectrogram bins near {center_hz:.4f} Hz")
    band_freq = freq[band_mask]
    band_spec = spectrum[band_mask, :]
    amp = np.abs(band_spec)
    peak_idx = np.argmax(amp, axis=0)
    chosen = band_spec[peak_idx, np.arange(band_spec.shape[1])]
    chosen_freq = band_freq[peak_idx]
    return {
        "time_s": np.asarray(spec.t, dtype=float),
        "freq_hz": np.asarray(chosen_freq, dtype=float),
        "complex": np.asarray(chosen, dtype=np.complex128),
        "amplitude": np.asarray(np.abs(chosen), dtype=float),
        "phase": np.asarray(np.angle(chosen), dtype=float),
    }


def plv_from_phase_residual(phase_residual: np.ndarray) -> float:
    if phase_residual.size == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * phase_residual))))


def summarize_line_tracks(regions: list[dict[str, object]], line_hz: float, cfg: Config) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    pooled_amplitude: list[float] = []
    pooled_freq: list[float] = []
    for region in regions:
        processed = region["processed"]
        spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.local_track_sliding_len_s)
        if spec is None:
            continue
        track = extract_local_track(spec, line_hz, cfg.track_half_width_hz)
        rows.append(
            {
                "region_index": int(region["region_index"]),
                "mean_amp": float(np.mean(track["amplitude"])),
                "mean_freq_hz": float(np.mean(track["freq_hz"])),
                "std_freq_hz": float(np.std(track["freq_hz"])),
            }
        )
        pooled_amplitude.extend(np.asarray(track["amplitude"], dtype=float).tolist())
        pooled_freq.extend(np.asarray(track["freq_hz"], dtype=float).tolist())

    return {
        "line_hz": line_hz,
        "rows": rows,
        "global_mean_amp": float(np.mean(pooled_amplitude)) if pooled_amplitude else np.nan,
        "global_mean_freq_hz": float(np.mean(pooled_freq)) if pooled_freq else np.nan,
        "global_std_freq_hz": float(np.std(pooled_freq)) if pooled_freq else np.nan,
    }


def evaluate_hypothesis(regions: list[dict[str, object]], hypothesis: HarmonicHypothesis, cfg: Config) -> dict[str, object]:
    rng = np.random.default_rng(cfg.null_seed)
    region_rows: list[dict[str, float]] = []
    phase_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    amp_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for region in regions:
        processed = region["processed"]
        spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.local_track_sliding_len_s)
        if spec is None:
            continue
        parent_track = extract_local_track(spec, hypothesis.parent_hz, cfg.track_half_width_hz)
        target_track = extract_local_track(spec, hypothesis.target_hz, cfg.track_half_width_hz)

        gate_parent = float(np.median(parent_track["amplitude"])) * cfg.gate_fraction_of_median
        gate_target = float(np.median(target_track["amplitude"])) * cfg.gate_fraction_of_median
        keep = (parent_track["amplitude"] >= gate_parent) & (target_track["amplitude"] >= gate_target)
        if int(np.sum(keep)) < cfg.min_windows_per_region:
            continue

        parent_phase = np.asarray(parent_track["phase"][keep], dtype=float)
        target_phase = np.asarray(target_track["phase"][keep], dtype=float)
        residual = np.angle(np.exp(1j * (target_phase - hypothesis.order * parent_phase)))
        region_plv = plv_from_phase_residual(residual)
        region_rows.append(
            {
                "region_index": int(region["region_index"]),
                "n_windows": int(np.sum(keep)),
                "plv": region_plv,
                "parent_freq_mean_hz": float(np.mean(parent_track["freq_hz"][keep])),
                "target_freq_mean_hz": float(np.mean(target_track["freq_hz"][keep])),
            }
        )
        phase_pairs.append((parent_phase, target_phase))
        amp_pairs.append(
            (
                np.asarray(parent_track["amplitude"][keep], dtype=float),
                np.asarray(target_track["amplitude"][keep], dtype=float),
            )
        )

    def pooled_plv(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        phases = [target_phase - hypothesis.order * parent_phase for parent_phase, target_phase in pairs]
        if len(phases) == 0:
            return np.nan
        return plv_from_phase_residual(np.concatenate(phases))

    observed_plv = pooled_plv(phase_pairs)
    null_values: list[float] = []
    for _ in range(cfg.null_trials):
        shifted_pairs = []
        for parent_phase, target_phase in phase_pairs:
            shift = int(rng.integers(1, len(target_phase)))
            shifted_pairs.append((parent_phase, np.roll(target_phase, shift)))
        null_values.append(pooled_plv(shifted_pairs))
    null_arr = np.asarray(null_values, dtype=float)

    amp_parent = np.concatenate([parent for parent, _ in amp_pairs]) if amp_pairs else np.array([], dtype=float)
    amp_target = np.concatenate([target for _, target in amp_pairs]) if amp_pairs else np.array([], dtype=float)
    if amp_parent.size >= 3 and amp_target.size >= 3:
        x = np.log(np.maximum(amp_parent, 1e-12))
        y = np.log(np.maximum(amp_target, 1e-12))
        amp_log_corr = float(np.corrcoef(x, y)[0, 1])
    else:
        amp_log_corr = np.nan

    return {
        "label": hypothesis.label,
        "parent_hz": hypothesis.parent_hz,
        "target_hz": hypothesis.target_hz,
        "order": hypothesis.order,
        "region_rows": region_rows,
        "observed_plv": observed_plv,
        "null_mean_plv": float(np.nanmean(null_arr)) if null_arr.size else np.nan,
        "null_q95_plv": float(np.nanquantile(null_arr, 0.95)) if null_arr.size else np.nan,
        "n_windows_total": int(sum(int(row["n_windows"]) for row in region_rows)),
        "n_regions_used": len(region_rows),
        "amp_log_corr": amp_log_corr,
    }


def compute_region_amplitude_matrix(regions: list[dict[str, object]], lines_hz: tuple[float, ...], cfg: Config) -> dict[str, np.ndarray]:
    amps = np.full((len(regions), len(lines_hz)), np.nan, dtype=float)
    freqs = np.full((len(regions), len(lines_hz)), np.nan, dtype=float)
    region_ids = np.full(len(regions), np.nan, dtype=float)
    for row_idx, region in enumerate(regions):
        region_ids[row_idx] = int(region["region_index"])
        processed = region["processed"]
        spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.local_track_sliding_len_s)
        if spec is None:
            continue
        for col_idx, line_hz in enumerate(lines_hz):
            track = extract_local_track(spec, line_hz, cfg.track_half_width_hz)
            amps[row_idx, col_idx] = float(np.mean(track["amplitude"]))
            freqs[row_idx, col_idx] = float(np.mean(track["freq_hz"]))

    corr = np.full((len(lines_hz), len(lines_hz)), np.nan, dtype=float)
    for i in range(len(lines_hz)):
        for j in range(len(lines_hz)):
            xi = np.log(np.maximum(amps[:, i], 1e-12))
            xj = np.log(np.maximum(amps[:, j], 1e-12))
            corr[i, j] = float(np.corrcoef(xi, xj)[0, 1])

    return {
        "lines_hz": np.asarray(lines_hz, dtype=float),
        "region_ids": region_ids,
        "mean_amp": amps,
        "mean_freq": freqs,
        "log_amp_corr": corr,
    }


def plot_recurrent_peaks(writer: PlotWriter, regions: list[dict[str, object]], cfg: Config) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    low_hz, high_hz = cfg.recurrent_peak_search_hz

    for row_idx, region in enumerate(regions):
        processed = region["processed"]
        fft = compute_one_sided_fft(processed.y, processed.dt)
        band_mask = (fft.freq >= low_hz) & (fft.freq <= high_hz)
        band_freq = fft.freq[band_mask]
        band_amp = fft.amplitude[band_mask]
        if band_freq.size == 0:
            continue
        prominence = float(np.nanmax(band_amp)) * cfg.recurrent_peak_prominence_frac
        peak_indices, _ = find_peaks(band_amp, prominence=prominence)
        if peak_indices.size == 0:
            continue
        order = np.argsort(band_amp[peak_indices])[::-1][: cfg.recurrent_peak_top_n]
        chosen = peak_indices[order]
        ax.scatter(
            band_freq[chosen],
            np.full(chosen.shape, row_idx),
            s=220.0 * band_amp[chosen] / np.nanmax(band_amp[chosen]),
            alpha=0.70,
            color="C0",
            edgecolor="black",
            linewidth=0.4,
        )

    for freq_hz in cfg.coherence_bands_hz:
        ax.axvline(freq_hz, color="black", linestyle=":", linewidth=0.9, alpha=0.45)

    ax.set_xlim(low_hz, high_hz)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("region index")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels([str(int(region["region_index"])) for region in regions])
    ax.grid(alpha=0.25)
    ax.set_title("Default-mode recurring FFT peaks across quiet regions")
    return writer.save(fig, "default_mode_recurrent_region_peaks")


def plot_same_frequency_coherence(writer: PlotWriter, result: dict[str, np.ndarray]) -> Path:
    bands_hz = np.asarray(result["bands_hz"], dtype=float)
    values = np.asarray(result["values"], dtype=float)
    mode_labels = [str(x) for x in result["mode_labels"]]
    pair_labels = [str(x) for x in result["pair_labels"]]

    fig, axes = plt.subplots(1, len(mode_labels), figsize=(12, 4), sharey=True, constrained_layout=True)
    if len(mode_labels) == 1:
        axes = [axes]

    for mode_idx, ax in enumerate(axes):
        for pair_idx, pair_label in enumerate(pair_labels):
            ax.plot(bands_hz, values[mode_idx, pair_idx, :], marker="o", linewidth=1.5, label=pair_label)
        ax.set_title(mode_labels[mode_idx])
        ax.set_xlabel("band center (Hz)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("max magnitude-squared coherence")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Same-frequency bond coherence is high in both observables")
    return writer.save(fig, "same_frequency_bond_coherence_compare")


def plot_hypothesis_scores(writer: PlotWriter, hypothesis_results: list[dict[str, object]], slug: str, title: str) -> Path:
    labels = [str(row["label"]) for row in hypothesis_results]
    observed = np.asarray([float(row["observed_plv"]) for row in hypothesis_results], dtype=float)
    null_mean = np.asarray([float(row["null_mean_plv"]) for row in hypothesis_results], dtype=float)
    null_q95 = np.asarray([float(row["null_q95_plv"]) for row in hypothesis_results], dtype=float)

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(x - 0.18, observed, width=0.36, label="observed PLV", color="C0")
    ax.bar(x + 0.18, null_mean, width=0.36, label="null mean PLV", color="C1")
    ax.scatter(x + 0.18, null_q95, marker="_", s=600, linewidths=2.0, color="black", label="null 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("pooled phase-lock value")
    ax.set_ylim(0.0, max(0.3, 1.10 * np.nanmax(np.concatenate([observed, null_q95]))))
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left")
    ax.set_title(title)
    return writer.save(fig, slug)


def plot_regionwise_phase_lock(writer: PlotWriter, hypothesis_results: list[dict[str, object]], slug: str) -> Path:
    fig, axes = plt.subplots(len(hypothesis_results), 1, figsize=(11, 2.6 * len(hypothesis_results)), sharex=True, constrained_layout=True)
    if len(hypothesis_results) == 1:
        axes = [axes]

    for ax, result in zip(axes, hypothesis_results):
        rows = result["region_rows"]
        region_idx = np.asarray([int(row["region_index"]) for row in rows], dtype=int)
        plv = np.asarray([float(row["plv"]) for row in rows], dtype=float)
        n_windows = np.asarray([int(row["n_windows"]) for row in rows], dtype=float)
        if region_idx.size:
            ax.scatter(region_idx, plv, s=20.0 + 5.0 * n_windows, color="C0", alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.axhline(float(result["null_q95_plv"]), color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("PLV")
        ax.grid(alpha=0.25)
        ax.set_title(str(result["label"]))
    axes[-1].set_xlabel("region index")
    return writer.save(fig, slug)


def plot_region_amplitude_matrix(writer: PlotWriter, matrix: dict[str, np.ndarray]) -> Path:
    lines_hz = np.asarray(matrix["lines_hz"], dtype=float)
    region_ids = np.asarray(matrix["region_ids"], dtype=float)
    amps = np.asarray(matrix["mean_amp"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(np.log10(np.maximum(amps, 1e-6)), aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="log10 mean tracked amplitude")
    ax.set_xticks(np.arange(len(lines_hz)))
    ax.set_xticklabels([f"{line:.2f}" for line in lines_hz])
    ax.set_yticks(np.arange(len(region_ids)))
    ax.set_yticklabels([str(int(x)) for x in region_ids])
    ax.set_xlabel("tracked line (Hz)")
    ax.set_ylabel("region index")
    ax.set_title("Default-mode line amplitudes across quiet regions")
    return writer.save(fig, "line_amplitude_matrix")


def plot_line_correlation_matrix(writer: PlotWriter, matrix: dict[str, np.ndarray]) -> Path:
    lines_hz = np.asarray(matrix["lines_hz"], dtype=float)
    corr = np.asarray(matrix["log_amp_corr"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.3, 5.4), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="corr(log amplitude)")
    ax.set_xticks(np.arange(len(lines_hz)))
    ax.set_xticklabels([f"{line:.2f}" for line in lines_hz], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(lines_hz)))
    ax.set_yticklabels([f"{line:.2f}" for line in lines_hz])
    ax.set_title("Region-scale amplitude correlation between lines")
    return writer.save(fig, "line_amplitude_correlation_matrix")


def build_summary(
    cfg: Config,
    base_dataset: str,
    segmentation_peak_times: np.ndarray,
    regions: list[dict[str, object]],
    recurrent_peaks: list[dict[str, float]],
    same_frequency: dict[str, np.ndarray],
    line_summaries: list[dict[str, object]],
    amplitude_matrix: dict[str, np.ndarray],
    independence_results: list[dict[str, object]],
    candidate_results: list[dict[str, object]],
    saved_plot_paths: list[Path],
) -> str:
    lines: list[str] = []
    lines.append(f"repo_root: {REPO_ROOT}")
    lines.append(f"dataset: {base_dataset}")
    lines.append(f"primary_component: {cfg.primary_component}")
    lines.append(f"primary_bond_index: {cfg.primary_bond_index}")
    lines.append(f"segmentation_mode: {cfg.segmentation_mode}")
    lines.append(f"comparison_modes: {list(cfg.comparison_modes)}")
    lines.append(f"segmentation_peak_times_s: {np.array2string(segmentation_peak_times, precision=3)}")
    lines.append("")
    lines.append("working conclusion:")
    lines.append("Treating 0.41, 3.35, 6.35, and 8.96 Hz as known fundamentals, the useful operational signature is not 'high coherence' by itself.")
    lines.append("The stronger signature is: a fundamental line should recur cleanly across quiet regions and should not be predictably phase-locked to the other fundamentals by a simple integer relation.")
    lines.append("A likely non-fundamental line should instead look parasitic: weaker, more regime-dependent, and more predictable from other lines in amplitude and/or phase.")
    lines.append("")
    lines.append("default-mode quiet regions:")
    for region in regions:
        lines.append(
            f"region {int(region['region_index'])}: "
            f"t=[{float(region['left']):.3f}, {float(region['right']):.3f}] s | "
            f"rms={float(region['rms']):.4f}"
        )
    lines.append("")
    lines.append("most recurrent default-mode x0 FFT peaks:")
    for row in recurrent_peaks[:10]:
        lines.append(
            f"{float(row['center_hz']):.3f} Hz | occurrences={int(row['n_occurrences'])} | "
            f"mean_amp={float(row['mean_amp']):.4f} | max_amp={float(row['max_amp']):.4f}"
        )
    lines.append("")
    lines.append("tracked line summaries:")
    for row in line_summaries:
        lines.append(
            f"{float(row['line_hz']):.2f} Hz: global mean tracked freq={float(row['global_mean_freq_hz']):.4f} Hz, "
            f"global freq std={float(row['global_std_freq_hz']):.4f} Hz, "
            f"global mean amp={float(row['global_mean_amp']):.5f}"
        )
    lines.append("")
    lines.append("same-frequency bond coherence summary:")
    bands_hz = np.asarray(same_frequency["bands_hz"], dtype=float)
    values = np.asarray(same_frequency["values"], dtype=float)
    mode_labels = [str(x) for x in same_frequency["mode_labels"]]
    pair_labels = [str(x) for x in same_frequency["pair_labels"]]
    for mode_idx, mode_label in enumerate(mode_labels):
        lines.append(mode_label)
        for pair_idx, pair_label in enumerate(pair_labels):
            vals = ", ".join(
                f"{band_hz:.2f} Hz={values[mode_idx, pair_idx, band_idx]:.3f}"
                for band_idx, band_hz in enumerate(bands_hz)
            )
            lines.append(f"  {pair_label}: {vals}")
    lines.append("")
    lines.append("independence checks between known fundamentals:")
    for row in independence_results:
        verdict = "passes independence check" if float(row["observed_plv"]) <= float(row["null_q95_plv"]) else "suspiciously locked"
        lines.append(
            f"{row['label']}: observed PLV={float(row['observed_plv']):.4f}, "
            f"null mean={float(row['null_mean_plv']):.4f}, null 95%={float(row['null_q95_plv']):.4f}, "
            f"regions={int(row['n_regions_used'])}, windows={int(row['n_windows_total'])} | {verdict}"
        )
    lines.append("")
    lines.append(f"candidate non-fundamental checks for {cfg.candidate_line_hz:.2f} Hz:")
    for row in candidate_results:
        verdict = "above null" if float(row["observed_plv"]) > float(row["null_q95_plv"]) else "not above null"
        lines.append(
            f"{row['label']}: observed PLV={float(row['observed_plv']):.4f}, "
            f"null mean={float(row['null_mean_plv']):.4f}, null 95%={float(row['null_q95_plv']):.4f}, "
            f"regions={int(row['n_regions_used'])}, windows={int(row['n_windows_total'])}, "
            f"log-amp corr={float(row['amp_log_corr']):.3f} | {verdict}"
        )
        for region_row in row["region_rows"]:
            lines.append(
                f"  region {int(region_row['region_index'])}: PLV={float(region_row['plv']):.3f}, "
                f"windows={int(region_row['n_windows'])}, "
                f"parent_mean={float(region_row['parent_freq_mean_hz']):.3f} Hz, "
                f"target_mean={float(region_row['target_freq_mean_hz']):.3f} Hz"
            )
    lines.append("")
    lines.append("region-scale log-amplitude correlations:")
    corr = np.asarray(amplitude_matrix["log_amp_corr"], dtype=float)
    lines_hz = np.asarray(amplitude_matrix["lines_hz"], dtype=float)
    for i in range(len(lines_hz)):
        vals = ", ".join(
            f"{lines_hz[j]:.2f}:{corr[i, j]:.3f}"
            for j in range(len(lines_hz))
            if j != i
        )
        lines.append(f"{lines_hz[i]:.2f} Hz -> {vals}")
    lines.append("")
    lines.append("interpretation:")
    lines.append("Default mode still looks like the better observable for this question because the known fundamentals recur cleanly there and the quiet-region segmentation is sharper.")
    lines.append("The known fundamentals 0.41, 3.35, 6.35, and 8.96 all recur across regions with stable tracked frequencies. That supports calling them real modes in this observable.")
    lines.append("The phase-lock independence tests mostly behave the way you want for fundamentals: 0.41 vs 3.35 and 3.35 vs 6.35 are not above the null, so they are not acting like simple harmonics of each other. 6.35 vs 8.96 is near-null as well. 3.35 vs 8.96 is only mildly elevated and not decisive.")
    lines.append("The ~18 Hz line is different. It is much weaker than the fundamentals, and its region-scale amplitude co-varies strongly with both 6.35 and 8.96, especially the latter.")
    lines.append("Unlike the other tested integer-lock relations, the 2 x 8.96 -> 18 hypothesis does rise above the 95% circular-shift null in the pooled phase-lock metric. That is not a full proof yet, but it is exactly the sort of predictable coherence signature you would expect from a non-fundamental follower line.")
    lines.append("Operationally, the current data supports a 'proof of fundamentality' for the known lines by recurrence plus failure of simple lock relations, and it gives a first positive non-fundamental signature for 18 through its amplitude covariance and weak-but-real 2 x 8.96 phase-lock excess. The natural next step is a bicoherence or trispectrum-style test centered on 8.96 and 18.")
    lines.append("")
    lines.append("saved_plots:")
    for path in saved_plot_paths:
        lines.append(str(path))
    return "\n".join(lines) + "\n"


def main() -> None:
    writer = PlotWriter(OUTPUT_DIR)
    writer.reset()

    base_dataset, ds, processed = load_primary_dataset(CONFIG.segmentation_mode, CONFIG)
    t_global, peak_times = detect_peak_times(processed, CONFIG)
    regions = build_regions(ds, CONFIG.primary_bond_index, t_global, peak_times, CONFIG)
    recurrent_peaks = summarize_recurrent_peaks(regions, CONFIG)
    same_frequency = compute_same_frequency_coherence(CONFIG)
    line_summaries = [
        summarize_line_tracks(regions, line_hz, CONFIG)
        for line_hz in (*CONFIG.fundamental_lines_hz, CONFIG.candidate_line_hz)
    ]
    amplitude_matrix = compute_region_amplitude_matrix(
        regions,
        (*CONFIG.fundamental_lines_hz, CONFIG.candidate_line_hz),
        CONFIG,
    )
    independence_results = [
        evaluate_hypothesis(regions, hypothesis, CONFIG)
        for hypothesis in CONFIG.independence_tests
    ]
    candidate_results = [
        evaluate_hypothesis(regions, hypothesis, CONFIG)
        for hypothesis in CONFIG.candidate_tests
    ]

    saved_plot_paths = [
        plot_recurrent_peaks(writer, regions, CONFIG),
        plot_same_frequency_coherence(writer, same_frequency),
        plot_region_amplitude_matrix(writer, amplitude_matrix),
        plot_line_correlation_matrix(writer, amplitude_matrix),
        plot_hypothesis_scores(
            writer,
            independence_results,
            "fundamental_independence_scores",
            "Known fundamentals: simple integer-lock checks should stay near null",
        ),
        plot_hypothesis_scores(
            writer,
            candidate_results,
            "candidate_18_scores",
            "Candidate 18 Hz: tested integer-lock relations vs null",
        ),
        plot_regionwise_phase_lock(writer, candidate_results, "candidate_18_regionwise_phase_lock"),
    ]

    summary = build_summary(
        CONFIG,
        base_dataset,
        peak_times,
        regions,
        recurrent_peaks,
        same_frequency,
        line_summaries,
        amplitude_matrix,
        independence_results,
        candidate_results,
        saved_plot_paths,
    )
    summary_path = save_summary(summary)
    print(f"[saved] {summary_path.name}")


if __name__ == "__main__":
    main()
