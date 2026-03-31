#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
PLAY_DIR = SCRIPT_DIR.parent


def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")


REPO_ROOT = add_repo_root_to_path()
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from play5 import _bicoherence_from_indices
from play7 import build_windows
from PlayInt2.int3 import flattened_welch_spectrum, detect_observed_peaks, collect_segments


@dataclass(frozen=True)
class TriadEstimate:
    label: str
    relation: str
    f1_hz: float
    f2_hz: float
    f3_hz: float
    best_start_s: float
    best_stop_s: float
    b_sq: float
    phase_lock: float
    alpha_real: float
    alpha_imag: float
    alpha_abs: float
    alpha_phase_rad: float
    beta_abs: float
    child_rms: float
    predictor_rms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate localized complex quadratic coupling coefficients for strong bicoherence triads.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--segment-len-s", type=float, default=100.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--analysis-window-s", type=float, default=100.0)
    parser.add_argument("--analysis-step-s", type=float, default=25.0)
    parser.add_argument("--min-window-segments", type=int, default=6)
    parser.add_argument("--bond-spacing-mode", choices=("default", "comoving"), default="comoving")
    parser.add_argument("--snap-bins", type=int, default=3)
    parser.add_argument("--peak-prominence", type=float, default=0.03)
    parser.add_argument("--merge-hz", type=float, default=0.08)
    parser.add_argument("--min-bicoherence", type=float, default=0.6)
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def _snap_index(index: int, amplitude: np.ndarray, radius: int) -> int:
    if radius <= 0:
        return int(index)
    lo = max(0, int(index) - int(radius))
    hi = min(amplitude.size, int(index) + int(radius) + 1)
    return int(lo + np.argmax(amplitude[lo:hi]))


def _resolve_peak(freq_target: float, observed_peaks) -> float:
    return float(min(observed_peaks, key=lambda peak: abs(peak.freq_hz - freq_target)).freq_hz)


def _best_window_and_bins(
    *,
    freqs: np.ndarray,
    mean_amplitude: np.ndarray,
    windows,
    X: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    relation: str,
    snap_bins: int,
) -> tuple[int, int, int, object, float, float]:
    df = float(freqs[1] - freqs[0])
    ia = _snap_index(int(round(f1_hz / df)), mean_amplitude, snap_bins)
    ib = _snap_index(int(round(f2_hz / df)), mean_amplitude, snap_bins)

    if relation == "sum":
        i1, i2, i3 = int(ia), int(ib), int(ia + ib)
    elif relation == "2x":
        i1, i2, i3 = int(ia), int(ia), int(2 * ia)
    elif relation == "diff":
        ih, il = max(int(ia), int(ib)), min(int(ia), int(ib))
        i1, i2, i3 = int(ih - il), int(il), int(ih)
    else:
        raise ValueError(relation)
    if i3 >= freqs.size:
        raise ValueError("Triad exceeded frequency grid")

    best_score = -np.inf
    best_window = None
    best_phase_lock = 0.0
    for window in windows:
        subset = X[window.segment_indices]
        b_sq, phase_lock = _bicoherence_from_indices(i1, i2, i3, subset)
        if b_sq > best_score:
            best_score = float(b_sq)
            best_phase_lock = float(phase_lock)
            best_window = window
    if best_window is None:
        raise ValueError("No best window found")
    return i1, i2, i3, best_window, float(best_score), float(best_phase_lock)


def estimate_alpha(X_subset: np.ndarray, i1: int, i2: int, i3: int) -> tuple[complex, float, float]:
    z = X_subset[:, i1] * X_subset[:, i2]
    y = X_subset[:, i3]
    denom = np.vdot(z, z)
    if np.abs(denom) <= 1e-18:
        return 0.0 + 0.0j, float(np.sqrt(np.mean(np.abs(z) ** 2))), float(np.sqrt(np.mean(np.abs(y) ** 2)))
    alpha = np.vdot(z, y) / denom
    return complex(alpha), float(np.sqrt(np.mean(np.abs(z) ** 2))), float(np.sqrt(np.mean(np.abs(y) ** 2)))


def collect_estimates(args: argparse.Namespace) -> list[TriadEstimate]:
    freqs_flat, amp_flat = flattened_welch_spectrum(bond_spacing_mode=str(args.bond_spacing_mode))
    observed_peaks = detect_observed_peaks(
        freqs_flat,
        amp_flat,
        peak_prominence=float(args.peak_prominence),
        merge_hz=float(args.merge_hz),
        min_freq_hz=0.2,
        max_freq_hz=28.0,
    )
    freqs, records, mean_amplitude = collect_segments(
        float(args.segment_len_s),
        float(args.overlap),
        bond_spacing_mode=str(args.bond_spacing_mode),
    )
    X = np.vstack([record.spectrum for record in records])
    mids = np.asarray([record.mid_s for record in records], dtype=float)
    windows = build_windows(
        mids,
        analysis_window_s=float(args.analysis_window_s),
        analysis_step_s=float(args.analysis_step_s),
        min_segments=int(args.min_window_segments),
    )

    triads = [
        ("0.417 + 8.965 -> 9.419", "sum", _resolve_peak(0.42, observed_peaks), _resolve_peak(8.96, observed_peaks)),
        ("8.965 + 9.419 -> 18.354", "sum", _resolve_peak(8.96, observed_peaks), _resolve_peak(9.42, observed_peaks)),
        ("2 * 8.965 -> 17.966", "2x", _resolve_peak(8.96, observed_peaks), _resolve_peak(8.96, observed_peaks)),
        ("0.417 + 16.626 -> 17.014", "sum", _resolve_peak(0.42, observed_peaks), _resolve_peak(16.63, observed_peaks)),
        ("2 * 0.417 -> 0.857", "2x", _resolve_peak(0.42, observed_peaks), _resolve_peak(0.42, observed_peaks)),
        ("23.137 - 8.965 -> 14.238", "diff", _resolve_peak(23.10, observed_peaks), _resolve_peak(8.96, observed_peaks)),
    ]

    estimates: list[TriadEstimate] = []
    for label, relation, f1_hz, f2_hz in triads:
        i1, i2, i3, best_window, best_b2, best_pl = _best_window_and_bins(
            freqs=freqs,
            mean_amplitude=mean_amplitude,
            windows=windows,
            X=X,
            f1_hz=f1_hz,
            f2_hz=f2_hz,
            relation=relation,
            snap_bins=int(args.snap_bins),
        )
        if best_b2 < float(args.min_bicoherence):
            continue
        subset = X[best_window.segment_indices]
        alpha, predictor_rms, child_rms = estimate_alpha(subset, i1, i2, i3)
        estimates.append(
            TriadEstimate(
                label=label,
                relation=relation,
                f1_hz=float(freqs[i1]),
                f2_hz=float(freqs[i2]),
                f3_hz=float(freqs[i3]),
                best_start_s=float(best_window.start_s),
                best_stop_s=float(best_window.stop_s),
                b_sq=float(best_b2),
                phase_lock=float(best_pl),
                alpha_real=float(np.real(alpha)),
                alpha_imag=float(np.imag(alpha)),
                alpha_abs=float(np.abs(alpha)),
                alpha_phase_rad=float(np.angle(alpha)),
                beta_abs=float(float(np.abs(alpha)) * predictor_rms / child_rms if child_rms > 1e-18 else 0.0),
                child_rms=float(child_rms),
                predictor_rms=float(predictor_rms),
            )
        )
    estimates.sort(key=lambda row: row.b_sq, reverse=True)
    return estimates


def write_csv(path: Path, estimates: list[TriadEstimate]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "relation",
                "f1_hz",
                "f2_hz",
                "f3_hz",
                "best_start_s",
                "best_stop_s",
                "b_sq",
                "phase_lock",
                "alpha_real",
                "alpha_imag",
                "alpha_abs",
                "alpha_phase_rad",
                "beta_abs",
                "child_rms",
                "predictor_rms",
            ]
        )
        for row in estimates:
            writer.writerow(
                [
                    row.label,
                    row.relation,
                    f"{row.f1_hz:.6f}",
                    f"{row.f2_hz:.6f}",
                    f"{row.f3_hz:.6f}",
                    f"{row.best_start_s:.3f}",
                    f"{row.best_stop_s:.3f}",
                    f"{row.b_sq:.6f}",
                    f"{row.phase_lock:.6f}",
                    f"{row.alpha_real:.6e}",
                    f"{row.alpha_imag:.6e}",
                    f"{row.alpha_abs:.6e}",
                    f"{row.alpha_phase_rad:.6f}",
                    f"{row.beta_abs:.6f}",
                    f"{row.child_rms:.6e}",
                    f"{row.predictor_rms:.6e}",
                ]
            )
    return path


def plot_results(estimates: list[TriadEstimate]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    labels = [row.label for row in estimates]
    y = np.arange(len(estimates))

    ax = axes[0]
    b2 = np.array([row.b_sq for row in estimates], dtype=float)
    beta = np.array([row.beta_abs for row in estimates], dtype=float)
    ax.barh(y, b2, color="#7570b3", alpha=0.85)
    for yi, row in zip(y, estimates):
        ax.text(row.b_sq + 0.01, yi, f"|alpha|={row.alpha_abs:.2e}, |beta|={row.beta_abs:.2f}", fontsize=8, va="center")
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Best localized bicoherence")
    ax.set_title("Strong Triads: Localized Bicoherence and Fitted Quadratic Strength")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    alpha_real = np.array([row.alpha_real for row in estimates], dtype=float)
    alpha_imag = np.array([row.alpha_imag for row in estimates], dtype=float)
    ax.scatter(alpha_real, alpha_imag, s=70, color="#d95f02")
    for row in estimates:
        ax.text(row.alpha_real, row.alpha_imag, f"  {row.label}", fontsize=8, va="center")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Re(alpha)")
    ax.set_ylabel("Im(alpha)")
    ax.set_title("Localized Complex Quadratic Coefficient")
    ax.grid(alpha=0.25)

    path = OUTPUT_DIR / "int4_quadratic_coefficients.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(path: Path, estimates: list[TriadEstimate], fig_path: Path, csv_path: Path) -> Path:
    lines: list[str] = []
    lines.append("Localized quadratic coefficient estimates")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  alpha is the least-squares complex coefficient in X3 ~= alpha * X1 * X2 over the best localized window.")
    lines.append("  |alpha| depends on the current FFT normalization and channel scaling, so compare it relatively within this pipeline.")
    lines.append("  |beta| = |alpha| * rms(X1X2) / rms(X3) is a dimensionless relative-strength measure.")
    lines.append("")
    for row in estimates:
        lines.append(
            f"{row.label}: b2={row.b_sq:.3f}, phase_lock={row.phase_lock:.3f}, "
            f"|alpha|={row.alpha_abs:.3e}, phase(alpha)={row.alpha_phase_rad:.3f} rad, "
            f"|beta|={row.beta_abs:.3f}, window={row.best_start_s:.1f}-{row.best_stop_s:.1f}s"
        )
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(fig_path))
    lines.append(str(csv_path))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)
    estimates = collect_estimates(args)
    if not estimates:
        raise ValueError("No triads met the minimum bicoherence threshold.")
    csv_path = write_csv(OUTPUT_DIR / "int4_quadratic_coefficients.csv", estimates)
    fig_path = plot_results(estimates)
    summary_path = write_summary(OUTPUT_DIR / "int4_quadratic_coefficients_summary.txt", estimates, fig_path, csv_path)

    for row in estimates:
        print(
            f"{row.label}: b2={row.b_sq:.3f} | |alpha|={row.alpha_abs:.3e} | |beta|={row.beta_abs:.3f} | "
            f"phase(alpha)={row.alpha_phase_rad:.3f} | window={row.best_start_s:.1f}-{row.best_stop_s:.1f}s"
        )
    print(f"Saved figure to {fig_path}")
    print(f"Saved summary to {summary_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
