from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


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

from analysis.Nrm.Tools.post_hit_regions import extract_post_hit_regions, save_preview
from analysis.tools.signal import compute_one_sided_fft
from analysis.tools.spectrasave import (
    add_spectrasave_arg,
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)


@dataclass(frozen=True)
class Config:
    dataset: str = "IMG_0681_rot270"
    component: str = "x"
    bond_ids: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class RegionSpectrum:
    bond_id: int
    region_index: int
    left_s: float
    right_s: float
    dt_s: float
    n_samples: int
    freq_hz: np.ndarray
    amplitude: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot and average enabled-window FFTs for IMG_0681_rot270 component x bonds 0-2."
    )
    add_spectrasave_arg(parser)
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show the final FFT figure on screen with matplotlib.",
    )
    parser.add_argument(
        "--noshow",
        dest="show",
        action="store_false",
        help="Do not show the final FFT figure on screen.",
    )
    parser.set_defaults(show=False)
    parser.add_argument(
        "--savitzky",
        action="store_true",
        help="Apply Savitzky-Golay smoothing to plotted spectra only.",
    )
    parser.add_argument(
        "--gaussianblur",
        action="store_true",
        help="Apply Gaussian smoothing to plotted spectra only. Stackable with --savitzky.",
    )
    parser.add_argument(
        "--gaussian-sigma-bins",
        type=float,
        default=1.5,
        help="Gaussian sigma in frequency bins for --gaussianblur. Default: 1.5",
    )
    parser.add_argument(
        "--gaussian-truncate",
        type=float,
        default=4.0,
        help="Gaussian kernel truncate radius in sigmas for --gaussianblur. Default: 4.0",
    )
    parser.add_argument(
        "--savitzky-window-bins",
        type=int,
        default=11,
        help="Savitzky-Golay window length in bins. Must be odd. Default: 11",
    )
    parser.add_argument(
        "--savitzky-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order. Default: 2",
    )
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def build_finest_common_grid(spectra: list[RegionSpectrum]) -> np.ndarray:
    if not spectra:
        raise ValueError("No spectra available to build a common grid.")

    freq_low = max(float(spec.freq_hz[0]) for spec in spectra)
    freq_high = min(float(spec.freq_hz[-1]) for spec in spectra)
    if not np.isfinite(freq_low) or not np.isfinite(freq_high) or freq_high <= freq_low:
        raise ValueError("No overlapping frequency window across the enabled-window FFTs.")

    steps: list[float] = []
    for spec in spectra:
        local = spec.freq_hz[(spec.freq_hz >= freq_low) & (spec.freq_hz <= freq_high)]
        if local.size < 2:
            continue
        dx = np.diff(local)
        dx = dx[np.isfinite(dx) & (dx > 0)]
        if dx.size:
            steps.append(float(np.min(dx)))

    if not steps:
        raise ValueError("Could not determine a positive FFT spacing for the common grid.")

    df = min(steps)
    grid = np.arange(freq_low, freq_high + 0.5 * df, df, dtype=float)
    grid = grid[(grid >= freq_low - 1e-12) & (grid <= freq_high + 1e-12)]
    if grid.size < 2:
        grid = np.array([freq_low, freq_high], dtype=float)
    return grid


def collect_region_spectra(cfg: Config) -> list[RegionSpectrum]:
    spectra: list[RegionSpectrum] = []
    for bond_id in cfg.bond_ids:
        result = extract_post_hit_regions(
            dataset=cfg.dataset,
            component=cfg.component,
            bond_id=bond_id,
        )
        save_preview(result, OUTPUT_DIR)
        for region in result.regions:
            dt_s = float(region.processed_t[1] - region.processed_t[0])
            fft = compute_one_sided_fft(region.processed_y, dt_s)
            spectra.append(
                RegionSpectrum(
                    bond_id=bond_id,
                    region_index=region.region_index,
                    left_s=region.left_s,
                    right_s=region.right_s,
                    dt_s=dt_s,
                    n_samples=int(region.processed_y.size),
                    freq_hz=np.asarray(fft.freq, dtype=float),
                    amplitude=np.asarray(fft.amplitude, dtype=float),
                )
            )
    if not spectra:
        raise ValueError("No enabled regions were found for the requested bonds.")
    return spectra


def smooth_for_plot(arr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = np.asarray(arr, dtype=float)

    if args.gaussianblur:
        sigma = float(args.gaussian_sigma_bins)
        truncate = float(args.gaussian_truncate)
        if sigma <= 0:
            raise ValueError("--gaussian-sigma-bins must be > 0")
        if truncate <= 0:
            raise ValueError("--gaussian-truncate must be > 0")
        out = np.asarray(
            gaussian_filter1d(out, sigma=sigma, mode="nearest", truncate=truncate),
            dtype=float,
        )

    if args.savitzky:
        window = int(args.savitzky_window_bins)
        polyorder = int(args.savitzky_polyorder)
        if window < 3:
            raise ValueError("--savitzky-window-bins must be at least 3")
        if window % 2 == 0:
            raise ValueError("--savitzky-window-bins must be odd")
        if polyorder < 0:
            raise ValueError("--savitzky-polyorder must be >= 0")
        if polyorder >= window:
            raise ValueError("--savitzky-polyorder must be smaller than --savitzky-window-bins")
        if window > out.size:
            raise ValueError(
                f"--savitzky-window-bins ({window}) exceeds the plotted spectrum length ({out.size})"
            )
        out = np.asarray(
            savgol_filter(out, window_length=window, polyorder=polyorder, mode="interp"),
            dtype=float,
        )

    return out


def make_fft_figure(
    spectra: list[RegionSpectrum],
    freq_grid_hz: np.ndarray,
    interp_amplitudes: np.ndarray,
    mean_amplitude: np.ndarray,
    args: argparse.Namespace,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    ax_avg, ax_all = axes
    eps = np.finfo(float).tiny

    mean_plot = np.maximum(smooth_for_plot(mean_amplitude, args), eps)
    ax_avg.plot(freq_grid_hz, mean_plot, color="black", linewidth=2.4)
    ax_avg.set_title("Average enabled-window FFT")
    ax_avg.set_ylabel("amplitude")
    ax_avg.set_yscale("log")
    ax_avg.grid(alpha=0.25)

    for row, spec in zip(interp_amplitudes, spectra):
        ax_all.plot(
            freq_grid_hz,
            np.maximum(row, eps),
            linewidth=0.9,
            alpha=0.45,
            label=f"bond {spec.bond_id} region {spec.region_index}",
        )

    ax_all.set_title(
        f"{CONFIG.dataset} | {CONFIG.component} | bonds {CONFIG.bond_ids[0]}-{CONFIG.bond_ids[-1]} | all enabled-window FFTs"
    )
    ax_all.set_xlabel("frequency (Hz)")
    ax_all.set_ylabel("amplitude")
    ax_all.set_yscale("log")
    ax_all.grid(alpha=0.25)
    ax_all.legend(loc="upper right", fontsize=7, ncol=2)
    return fig


def save_combined_dataset(
    spectra: list[RegionSpectrum],
    freq_grid_hz: np.ndarray,
    interp_amplitudes: np.ndarray,
    mean_amplitude: np.ndarray,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dataset=CONFIG.dataset,
        component=CONFIG.component,
        bond_ids=np.asarray(CONFIG.bond_ids, dtype=int),
        freq_grid_hz=freq_grid_hz,
        mean_amplitude=mean_amplitude,
        interpolated_amplitudes=interp_amplitudes,
        source_bond_ids=np.asarray([spec.bond_id for spec in spectra], dtype=int),
        source_region_indices=np.asarray([spec.region_index for spec in spectra], dtype=int),
        source_bounds_s=np.asarray([(spec.left_s, spec.right_s) for spec in spectra], dtype=float),
        source_dt_s=np.asarray([spec.dt_s for spec in spectra], dtype=float),
        source_n_samples=np.asarray([spec.n_samples for spec in spectra], dtype=int),
        raw_freq_hz=np.asarray([spec.freq_hz for spec in spectra], dtype=object),
        raw_amplitude=np.asarray([spec.amplitude for spec in spectra], dtype=object),
    )
    return path


CONFIG = Config()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_matplotlib(args.show)

    import matplotlib.pyplot as plt

    spectra = collect_region_spectra(CONFIG)
    freq_grid_hz = build_finest_common_grid(spectra)
    interp_amplitudes = np.vstack(
        [np.interp(freq_grid_hz, spec.freq_hz, spec.amplitude) for spec in spectra]
    )
    mean_amplitude = np.mean(interp_amplitudes, axis=0)

    fig = make_fft_figure(
        spectra,
        freq_grid_hz,
        interp_amplitudes,
        mean_amplitude,
        args,
    )

    figure_path = OUTPUT_DIR / "play1_enabled_window_ffts.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", dpi=180)

    dataset_path = save_combined_dataset(
        spectra,
        freq_grid_hz,
        interp_amplitudes,
        mean_amplitude,
        OUTPUT_DIR / "play1_enabled_window_fft_average.npz",
    )

    spectrasave_path = None
    if args.spectrasave is not None:
        spectrasave_path = resolve_spectrasave_path(
            args.spectrasave,
            default_name=build_default_spectrasave_name(
                CONFIG.dataset,
                CONFIG.component,
                f"bonds-{CONFIG.bond_ids[0]}-{CONFIG.bond_ids[-1]}",
                "enabled-window-average-fft",
            ),
        )
        spectrasave_path = save_spectrum_msgpack(
            spectrasave_path,
            freq=freq_grid_hz,
            amplitude=mean_amplitude,
            label=f"{CONFIG.dataset} {CONFIG.component} bonds {CONFIG.bond_ids[0]}-{CONFIG.bond_ids[-1]} enabled-window average FFT",
            metadata={
                "dataset": CONFIG.dataset,
                "component": CONFIG.component,
                "bondIds": list(CONFIG.bond_ids),
                "nEnabledWindows": len(spectra),
                "gaussianBlur": bool(args.gaussianblur),
                "gaussianSigmaBins": float(args.gaussian_sigma_bins),
                "gaussianTruncate": float(args.gaussian_truncate),
                "savitzky": bool(args.savitzky),
                "savitzkyWindowBins": int(args.savitzky_window_bins),
                "savitzkyPolyorder": int(args.savitzky_polyorder),
                "sourceNpz": str(dataset_path),
                "sourceFigure": str(figure_path),
            },
        )

    if args.show:
        plt.show()
    plt.close(fig)

    payload = {
        "dataset": CONFIG.dataset,
        "component": CONFIG.component,
        "bond_ids": list(CONFIG.bond_ids),
        "n_enabled_windows": len(spectra),
        "freq_low_hz": float(freq_grid_hz[0]),
        "freq_high_hz": float(freq_grid_hz[-1]),
        "n_freq_samples": int(freq_grid_hz.size),
        "show": bool(args.show),
        "gaussianblur": bool(args.gaussianblur),
        "gaussian_sigma_bins": float(args.gaussian_sigma_bins),
        "gaussian_truncate": float(args.gaussian_truncate),
        "savitzky": bool(args.savitzky),
        "savitzky_window_bins": int(args.savitzky_window_bins),
        "savitzky_polyorder": int(args.savitzky_polyorder),
        "figure_path": str(figure_path),
        "dataset_path": str(dataset_path),
        "spectrasave_path": None if spectrasave_path is None else str(spectrasave_path),
        "tool_preview_paths": [
            str(OUTPUT_DIR / f"{CONFIG.dataset}_{CONFIG.component}_bond{bond_id}_enabled_regions_preview.png")
            for bond_id in CONFIG.bond_ids
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
