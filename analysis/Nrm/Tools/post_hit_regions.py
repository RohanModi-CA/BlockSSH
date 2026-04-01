from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


SCRIPT_DIR = Path(__file__).resolve().parent


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


@dataclass(frozen=True)
class EnabledRegionConfig:
    bond_spacing_mode: str = "default"
    sliding_len_s: float = 20.0
    manual_peak_times_s: tuple[float, ...] = (400.0, 494.0)
    peak_finder_mode: str = "all"
    prominence: float = 0.0
    distance_samples: int | None = None
    min_segment_len_s: float = 25.0
    begin_trim_s: float = 0.0
    end_trim_s: float = 4.0
    enable_mode: str = "default"
    keep_edge_regions: bool = False


@dataclass(frozen=True)
class EnabledRegion:
    region_index: int
    left_s: float
    right_s: float
    raw_t: np.ndarray
    raw_y: np.ndarray
    processed_t: np.ndarray
    processed_y: np.ndarray


@dataclass(frozen=True)
class ExtractionResult:
    dataset: str
    component: str
    bond_id: int
    config: EnabledRegionConfig
    frame_times_s: np.ndarray
    signal: np.ndarray
    spectrogram_freq_hz: np.ndarray
    spectrogram_time_s: np.ndarray
    spectrogram_power: np.ndarray
    broadband_time_s: np.ndarray
    broadband_energy: np.ndarray
    auto_peak_times_s: np.ndarray
    peak_times_s: np.ndarray
    enabled_bounds_s: np.ndarray
    regions: tuple[EnabledRegion, ...]


def _validate_args(dataset: str, component: str, bond_id: int) -> None:
    if component not in {"x", "y", "a"}:
        raise ValueError(f"component must be one of x, y, a; got {component!r}")
    if bond_id < 0:
        raise ValueError("bond_id (0-based) must be non-negative")
    if not dataset:
        raise ValueError("dataset must be non-empty")


def _load_signal(dataset: str, component: str, bond_id: int, bond_spacing_mode: str):
    _validate_args(dataset, component, bond_id)
    base_dataset, _ = split_dataset_component(dataset)
    ds = load_bond_signal_dataset(
        dataset=f"{base_dataset}_{component}",
        bond_spacing_mode=bond_spacing_mode,
        component=component,
    )
    if bond_id >= ds.signal_matrix.shape[1]:
        raise ValueError(f"bond_id {bond_id} out of range for dataset with {ds.signal_matrix.shape[1]} bonds")
    signal = np.asarray(ds.signal_matrix[:, bond_id], dtype=float)
    frame_times_s = np.asarray(ds.frame_times_s, dtype=float)
    return ds, frame_times_s, signal


def _find_peak_indices(broadband_energy: np.ndarray, cfg: EnabledRegionConfig) -> np.ndarray:
    kwargs: dict[str, object] = {}
    if cfg.peak_finder_mode == "prominence":
        kwargs["prominence"] = cfg.prominence
        if cfg.distance_samples is not None:
            kwargs["distance"] = cfg.distance_samples
    elif cfg.peak_finder_mode != "all":
        raise ValueError(f"Unsupported peak_finder_mode {cfg.peak_finder_mode!r}")
    peak_indices, _ = find_peaks(broadband_energy, **kwargs)
    return np.asarray(peak_indices, dtype=int)


def _region_is_enabled(
    region_position: int,
    n_regions: int,
    duration_s: float,
    trimmed_left_s: float,
    trimmed_right_s: float,
    cfg: EnabledRegionConfig,
) -> bool:
    if trimmed_right_s <= trimmed_left_s:
        return False
    if cfg.enable_mode == "all":
        pass
    elif cfg.enable_mode == "duration_only":
        if duration_s < cfg.min_segment_len_s:
            return False
    elif cfg.enable_mode == "default":
        if duration_s < cfg.min_segment_len_s:
            return False
        if not cfg.keep_edge_regions and (region_position == 0 or region_position == n_regions - 1):
            return False
    else:
        raise ValueError(f"Unsupported enable_mode {cfg.enable_mode!r}")
    return True


def extract_post_hit_regions(
    *,
    dataset: str,
    component: str,
    bond_id: int,
    config: EnabledRegionConfig | None = None,
) -> ExtractionResult:
    cfg = config or EnabledRegionConfig()
    _, frame_times_s, signal = _load_signal(dataset, component, bond_id, cfg.bond_spacing_mode)
    processed, err = preprocess_signal(frame_times_s, signal, longest=False, handlenan=False)
    if processed is None:
        raise ValueError(err)
    spec = compute_complex_spectrogram(processed.y, processed.Fs, cfg.sliding_len_s)
    if spec is None:
        raise ValueError("Signal is too short to build the reference spectrogram.")

    broadband_time_s = np.asarray(spec.t + processed.t[0], dtype=float)
    spectrogram_power = np.abs(spec.S_complex)
    broadband_energy = np.sum(spectrogram_power, axis=0)
    auto_peak_indices = _find_peak_indices(broadband_energy, cfg)
    auto_peak_times_s = broadband_time_s[auto_peak_indices]
    manual_peak_times_s = np.asarray(cfg.manual_peak_times_s, dtype=float)
    peak_times_s = np.sort(np.unique(np.concatenate([auto_peak_times_s, manual_peak_times_s])))

    edges = np.concatenate(([float(broadband_time_s[0])], peak_times_s, [float(broadband_time_s[-1])]))
    enabled_regions: list[EnabledRegion] = []
    enabled_bounds: list[tuple[float, float]] = []
    durations = np.diff(edges)

    for region_position, duration_s in enumerate(durations):
        left_edge_s = float(edges[region_position])
        right_edge_s = float(edges[region_position + 1])
        left_s = left_edge_s + cfg.begin_trim_s
        right_s = right_edge_s - cfg.end_trim_s
        if not _region_is_enabled(region_position, len(durations), float(duration_s), left_s, right_s, cfg):
            continue

        mask = (
            (frame_times_s >= left_s)
            & (frame_times_s <= right_s)
            & np.isfinite(signal)
        )
        raw_t = frame_times_s[mask]
        raw_y = signal[mask]
        if raw_t.size < 10:
            continue
        proc_region, err = preprocess_signal(raw_t, raw_y, longest=False, handlenan=False)
        if proc_region is None:
            raise ValueError(f"Region {region_position} preprocess failed: {err}")
        enabled_regions.append(
            EnabledRegion(
                region_index=region_position,
                left_s=float(left_s),
                right_s=float(right_s),
                raw_t=np.asarray(raw_t, dtype=float),
                raw_y=np.asarray(raw_y, dtype=float),
                processed_t=np.asarray(proc_region.t, dtype=float),
                processed_y=np.asarray(proc_region.y, dtype=float),
            )
        )
        enabled_bounds.append((float(left_s), float(right_s)))

    return ExtractionResult(
        dataset=dataset,
        component=component,
        bond_id=bond_id,
        config=cfg,
        frame_times_s=frame_times_s,
        signal=signal,
        spectrogram_freq_hz=np.asarray(spec.f, dtype=float),
        spectrogram_time_s=broadband_time_s,
        spectrogram_power=spectrogram_power,
        broadband_time_s=broadband_time_s,
        broadband_energy=np.asarray(broadband_energy, dtype=float),
        auto_peak_times_s=np.asarray(auto_peak_times_s, dtype=float),
        peak_times_s=np.asarray(peak_times_s, dtype=float),
        enabled_bounds_s=np.asarray(enabled_bounds, dtype=float) if enabled_bounds else np.empty((0, 2), dtype=float),
        regions=tuple(enabled_regions),
    )


def save_regions_npz(result: ExtractionResult, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region_times = np.asarray([region.raw_t for region in result.regions], dtype=object)
    region_values = np.asarray([region.raw_y for region in result.regions], dtype=object)
    processed_times = np.asarray([region.processed_t for region in result.regions], dtype=object)
    processed_values = np.asarray([region.processed_y for region in result.regions], dtype=object)
    np.savez(
        out_path,
        dataset=result.dataset,
        component=result.component,
        bond_id=result.bond_id,
        bond_spacing_mode=result.config.bond_spacing_mode,
        enabled_bounds_s=result.enabled_bounds_s,
        region_times_s=region_times,
        region_values=region_values,
        processed_times_s=processed_times,
        processed_values=processed_values,
        peak_times_s=result.peak_times_s,
        auto_peak_times_s=result.auto_peak_times_s,
    )
    return out_path


def _preview_figure(result: ExtractionResult) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    ymax_signal = float(np.nanmax(result.signal)) if result.signal.size else 1.0
    ymin_signal = float(np.nanmin(result.signal)) if result.signal.size else -1.0

    axes[0].plot(result.frame_times_s, result.signal, color="tab:blue", linewidth=0.9)
    for region in result.regions:
        axes[0].axvspan(region.left_s, region.right_s, color="tab:green", alpha=0.12)
        axes[0].axvline(region.left_s, color="tab:green", alpha=0.45, linewidth=0.9)
        axes[0].axvline(region.right_s, color="tab:green", alpha=0.45, linewidth=0.9)
        axes[0].text(
            0.5 * (region.left_s + region.right_s),
            ymin_signal + 0.92 * (ymax_signal - ymin_signal),
            str(region.region_index),
            ha="center",
            va="top",
            fontsize=8,
            color="tab:green",
        )
    axes[0].set_ylabel("signal")
    axes[0].set_title(f"{result.dataset} | {result.component} bond {result.bond_id + 1} | timeseries")
    axes[0].grid(alpha=0.25)

    spec_power = np.maximum(result.spectrogram_power, np.finfo(float).tiny)
    mesh = axes[1].pcolormesh(
        result.spectrogram_time_s,
        result.spectrogram_freq_hz,
        np.log10(spec_power),
        shading="auto",
        cmap="turbo",
    )
    fig.colorbar(mesh, ax=axes[1], label="log10 |S|")
    ymax_freq = float(np.max(result.spectrogram_freq_hz)) if result.spectrogram_freq_hz.size else 1.0
    for region in result.regions:
        axes[1].axvspan(region.left_s, region.right_s, color="white", alpha=0.08)
        axes[1].axvline(region.left_s, color="white", alpha=0.55, linewidth=0.9)
        axes[1].axvline(region.right_s, color="white", alpha=0.55, linewidth=0.9)
        axes[1].text(
            0.5 * (region.left_s + region.right_s),
            0.96 * ymax_freq,
            str(region.region_index),
            ha="center",
            va="top",
            fontsize=8,
            color="white",
        )
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].set_title("Reference spectrogram")

    axes[2].plot(result.broadband_time_s, result.broadband_energy, color="black", linewidth=1.2, label="broadband energy")
    axes[2].plot(
        result.auto_peak_times_s,
        np.interp(result.auto_peak_times_s, result.broadband_time_s, result.broadband_energy),
        "o",
        color="tab:red",
        label="auto peaks",
    )
    if result.config.manual_peak_times_s:
        manual = np.asarray(result.config.manual_peak_times_s, dtype=float)
        axes[2].plot(
            manual,
            np.interp(manual, result.broadband_time_s, result.broadband_energy),
            "s",
            color="tab:orange",
            label="manual peaks",
        )
    ymax = float(np.max(result.broadband_energy)) if result.broadband_energy.size else 1.0
    for region in result.regions:
        axes[2].axvspan(region.left_s, region.right_s, color="tab:green", alpha=0.12)
        axes[2].axvline(region.left_s, color="tab:green", alpha=0.45, linewidth=0.9)
        axes[2].axvline(region.right_s, color="tab:green", alpha=0.45, linewidth=0.9)
        axes[2].text(
            0.5 * (region.left_s + region.right_s),
            0.96 * ymax,
            str(region.region_index),
            ha="center",
            va="top",
            fontsize=8,
        )
    axes[2].set_ylabel("BB energy")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title(
        "Broadband energy with hit peaks and enabled post-hit regions\n"
        f"mode={result.config.bond_spacing_mode}, enable={result.config.enable_mode}, trim=({result.config.begin_trim_s:.1f}, {result.config.end_trim_s:.1f}) s"
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")
    return fig


def save_preview(result: ExtractionResult, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = _preview_figure(result)
    path = out_dir / f"{result.dataset}_{result.component}_bond{result.bond_id}_enabled_regions_preview.png"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return path


def show_preview(result: ExtractionResult) -> None:
    fig = _preview_figure(result)
    plt.show()
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract enabled post-hit quiet regions for one dataset/component/bond."
    )
    parser.add_argument("--dataset", required=True, help="Base dataset name, e.g. IMG_0681_rot270")
    parser.add_argument("--component", required=True, choices=("x", "y", "a"))
    parser.add_argument("--bond-id", required=True, type=int, help="1-based bond index")
    parser.add_argument("--bond-spacing-mode", default="default", choices=("default", "comoving", "purecomoving"))
    parser.add_argument("--sliding-len-s", type=float, default=20.0)
    parser.add_argument("--manual-peak-times-s", type=float, nargs="*", default=[400.0, 494.0])
    parser.add_argument("--peak-finder-mode", default="all", choices=("all", "prominence"))
    parser.add_argument("--prominence", type=float, default=0.0)
    parser.add_argument("--distance-samples", type=int, default=None)
    parser.add_argument("--min-segment-len-s", type=float, default=25.0)
    parser.add_argument("--begin-trim-s", type=float, default=0.0)
    parser.add_argument("--end-trim-s", type=float, default=4.0)
    parser.add_argument("--enable-mode", default="default", choices=("default", "duration_only", "all"))
    parser.add_argument("--keep-edge-regions", action="store_true")
    parser.add_argument("--save-npz", default=None, help="Optional output .npz path for extracted region arrays")
    parser.add_argument("--preview-save", action="store_true", help="Save the reference preview plot")
    parser.add_argument("--preview-show", action="store_true", help="Show the reference preview plot")
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "output"),
        help="Directory used for preview output and default .npz output",
    )
    parser.add_argument("--print-json", action="store_true", help="Print machine-readable summary JSON")
    return parser


def _config_from_args(args: argparse.Namespace) -> EnabledRegionConfig:
    return EnabledRegionConfig(
        bond_spacing_mode=args.bond_spacing_mode,
        sliding_len_s=args.sliding_len_s,
        manual_peak_times_s=tuple(args.manual_peak_times_s),
        peak_finder_mode=args.peak_finder_mode,
        prominence=args.prominence,
        distance_samples=args.distance_samples,
        min_segment_len_s=args.min_segment_len_s,
        begin_trim_s=args.begin_trim_s,
        end_trim_s=args.end_trim_s,
        enable_mode=args.enable_mode,
        keep_edge_regions=args.keep_edge_regions,
    )


def _summary_payload(result: ExtractionResult) -> dict[str, object]:
    return {
        "dataset": result.dataset,
        "component": result.component,
        "bond_id": result.bond_id,
        "bond_spacing_mode": result.config.bond_spacing_mode,
        "n_regions": len(result.regions),
        "peak_times_s": result.peak_times_s.tolist(),
        "enabled_bounds_s": result.enabled_bounds_s.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    bond_id = args.bond_id - 1
    cfg = _config_from_args(args)
    result = extract_post_hit_regions(
        dataset=args.dataset,
        component=args.component,
        bond_id=bond_id,
        config=cfg,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path: Path | None = None
    if args.save_npz:
        npz_path = save_regions_npz(result, args.save_npz)
    elif args.preview_save:
        npz_path = save_regions_npz(
            result,
            output_dir / f"{args.dataset}_{args.component}_bond{args.bond_id}_enabled_regions.npz",
        )

    preview_path: Path | None = None
    if args.preview_save:
        preview_path = save_preview(result, output_dir)
    if args.preview_show:
        show_preview(result)

    payload = _summary_payload(result)
    if npz_path is not None:
        payload["npz_path"] = str(npz_path)
    if preview_path is not None:
        payload["preview_path"] = str(preview_path)

    if args.print_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"dataset={result.dataset} component={result.component} bond_id={result.bond_id + 1}")
        print(f"n_regions={len(result.regions)}")
        print(f"enabled_bounds_s={payload['enabled_bounds_s']}")
        if npz_path is not None:
            print(f"saved_npz={npz_path}")
        if preview_path is not None:
            print(f"saved_preview={preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
