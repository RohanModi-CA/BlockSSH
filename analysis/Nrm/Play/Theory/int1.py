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
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))


def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")


REPO_ROOT = add_repo_root_to_path()
THEORY_DIR = REPO_ROOT / "theory"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from helpers.dgnic import solve_chain
from analysis.go.Play.fft_flattening import compute_flattened_component_spectra
from PlayInt2.int1 import detect_observed_peaks
from analysis.tools.signal import compute_welch_spectrum


@dataclass(frozen=True)
class DataTargets:
    f1_hz: float
    f2_hz: float
    amp1: float
    amp2: float
    ratio: float


@dataclass(frozen=True)
class SimulationResult:
    beta: float
    times: np.ndarray
    disp: np.ndarray
    bond: np.ndarray
    freqs: np.ndarray
    avg_amp: np.ndarray
    f1_amp: float
    f2_amp: float
    ratio: float
    linear_modes_hz: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-model quadratic chain comparison against current data; beta->0 verifies the linear dgnic limit.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--m1-g", type=float, default=28.04, help="Light mass in grams.")
    parser.add_argument("--m2-g", type=float, default=60.04, help="Heavy mass in grams.")
    parser.add_argument("--k", type=float, default=194.0)
    parser.add_argument("--L", type=float, default=59 * 0.0254)
    parser.add_argument("--drive-site", type=int, default=0)
    parser.add_argument("--drive-freq-hz", type=float, default=0.41748)
    parser.add_argument("--drive-accel", type=float, default=0.16, help="Sinusoidal drive acceleration amplitude in m/s^2.")
    parser.add_argument("--beta-values", type=float, nargs="+", default=(0.0, 250.0, 500.0, 750.0, 1000.0))
    parser.add_argument("--tmax", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sample-rate", type=float, default=60.0)
    parser.add_argument("--damping-gamma", type=float, default=0.12)
    parser.add_argument("--welch-segment-s", type=float, default=20.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--analysis-start-s", type=float, default=15.0, help="Discard earlier simulation time before Welch.")
    parser.add_argument("--peak-prominence", type=float, default=0.02)
    parser.add_argument("--merge-hz", type=float, default=0.06)
    parser.add_argument("--bond-spacing-mode", choices=("default", "comoving"), default="comoving")
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def resolve_data_targets(bond_spacing_mode: str, peak_prominence: float, merge_hz: float) -> DataTargets:
    results = compute_flattened_component_spectra(
        dataset="IMG_0681_rot270",
        bond_spacing_mode=bond_spacing_mode,
        components=("x",),
        use_welch=True,
    )
    result = results["x"]
    freqs = np.asarray(result.freq_hz, dtype=float)
    amp = np.asarray(result.flattened, dtype=float)
    peaks = detect_observed_peaks(
        freqs,
        amp,
        peak_prominence=peak_prominence,
        merge_hz=merge_hz,
        min_freq_hz=0.2,
        max_freq_hz=3.0,
    )
    p1 = min(peaks, key=lambda peak: abs(peak.freq_hz - 0.42))
    p2 = min(peaks, key=lambda peak: abs(peak.freq_hz - 0.91))
    return DataTargets(
        f1_hz=float(p1.freq_hz),
        f2_hz=float(p2.freq_hz),
        amp1=float(p1.amplitude),
        amp2=float(p2.amplitude),
        ratio=float(p2.amplitude / p1.amplitude) if p1.amplitude > 0 else np.nan,
    )


def quadratic_contact_accel(pos: np.ndarray, masses: np.ndarray, k: float, beta: float) -> np.ndarray:
    if beta == 0.0 or pos.size < 2:
        return np.zeros_like(pos)
    delta = pos[1:] - pos[:-1]
    bond_force = beta * k * delta**2
    accel = np.zeros_like(pos)
    accel[:-1] += bond_force / masses[:-1]
    accel[1:] -= bond_force / masses[1:]
    return accel


def rk4_step(y: np.ndarray, v: np.ndarray, dt: float, accel_fn) -> tuple[np.ndarray, np.ndarray]:
    k1y = v
    k1v = accel_fn(y, v)
    y2 = y + 0.5 * dt * k1y
    v2 = v + 0.5 * dt * k1v
    k2y = v2
    k2v = accel_fn(y2, v2)
    y3 = y + 0.5 * dt * k2y
    v3 = v + 0.5 * dt * k2v
    k3y = v3
    k3v = accel_fn(y3, v3)
    y4 = y + dt * k3y
    v4 = v + dt * k3v
    k4y = v4
    k4v = accel_fn(y4, v4)
    y_next = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_next = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_next, v_next


def run_single_simulation(
    *,
    beta: float,
    N: int,
    m1_kg: float,
    m2_kg: float,
    k: float,
    L: float,
    drive_site: int,
    drive_freq_hz: float,
    drive_accel: float,
    tmax: float,
    dt: float,
    sample_rate: float,
    damping_gamma: float,
    welch_segment_s: float,
    welch_overlap: float,
    analysis_start_s: float,
) -> SimulationResult:
    chain = solve_chain(N=N, m1=m1_kg, m2=m2_kg, k=k, L=L)
    H = chain.H
    masses = chain.masses

    n_steps = int(round(tmax / dt)) + 1
    sample_step = max(1, int(round((1.0 / sample_rate) / dt)))
    n_samples = (n_steps - 1) // sample_step + 1
    times = np.zeros(n_samples, dtype=float)
    disp = np.zeros((n_samples, N), dtype=float)

    y = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)
    omega = 2.0 * np.pi * drive_freq_hz

    def accel_fn(pos: np.ndarray, vel: np.ndarray, t: float) -> np.ndarray:
        drive = np.zeros(N, dtype=float)
        drive[drive_site] = drive_accel * np.sin(omega * t)
        linear = -(H @ pos)
        nonlinear = quadratic_contact_accel(pos, masses, k, beta)
        return linear - damping_gamma * vel + nonlinear + drive

    sample_idx = 0
    times[sample_idx] = 0.0
    disp[sample_idx] = y
    sample_idx += 1

    for step in range(1, n_steps):
        t = (step - 1) * dt
        y, v = rk4_step(y, v, dt, lambda pos, vel: accel_fn(pos, vel, t))
        if step % sample_step == 0:
            times[sample_idx] = step * dt
            disp[sample_idx] = y
            sample_idx += 1

    times = times[:sample_idx]
    disp = disp[:sample_idx]
    keep = times >= analysis_start_s
    times_keep = times[keep]
    disp_keep = disp[keep]
    bond = disp_keep[:, :-1] - disp_keep[:, 1:]

    amps = []
    freqs = None
    for bond_id in range(bond.shape[1]):
        welch = compute_welch_spectrum(
            bond[:, bond_id] - np.mean(bond[:, bond_id]),
            sample_rate,
            welch_segment_s,
            overlap_fraction=welch_overlap,
        )
        if welch is None:
            raise ValueError(f"Welch failed for beta={beta} bond {bond_id}")
        freqs = np.asarray(welch.freq, dtype=float)
        amps.append(np.asarray(welch.amplitude, dtype=float))
    if freqs is None:
        raise ValueError("No theory Welch spectra produced")
    avg_amp = np.mean(np.vstack(amps), axis=0)
    f1_amp = float(np.interp(drive_freq_hz, freqs, avg_amp))
    f2_amp = float(np.interp(2.0 * drive_freq_hz, freqs, avg_amp))
    ratio = f2_amp / f1_amp if f1_amp > 0 else np.nan

    return SimulationResult(
        beta=float(beta),
        times=times_keep,
        disp=disp_keep,
        bond=bond,
        freqs=freqs,
        avg_amp=avg_amp,
        f1_amp=f1_amp,
        f2_amp=f2_amp,
        ratio=float(ratio),
        linear_modes_hz=np.asarray(chain.freq, dtype=float),
    )


def write_csv(path: Path, rows: list[SimulationResult], data_targets: DataTargets) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta", "theory_f1_amp", "theory_f2_amp", "theory_ratio", "data_f1_hz", "data_f2_hz", "data_ratio"])
        for row in rows:
            writer.writerow(
                [
                    f"{row.beta:.6f}",
                    f"{row.f1_amp:.6e}",
                    f"{row.f2_amp:.6e}",
                    f"{row.ratio:.6f}",
                    f"{data_targets.f1_hz:.6f}",
                    f"{data_targets.f2_hz:.6f}",
                    f"{data_targets.ratio:.6f}",
                ]
            )
    return path


def plot_results(rows: list[SimulationResult], data_targets: DataTargets) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    results = compute_flattened_component_spectra(
        dataset="IMG_0681_rot270",
        bond_spacing_mode="comoving",
        components=("x",),
        use_welch=True,
    )
    result = results["x"]
    freqs_data = np.asarray(result.freq_hz, dtype=float)
    amp_data = np.asarray(result.flattened, dtype=float)
    mask_data = (freqs_data >= 0.2) & (freqs_data <= 3.0)
    ax = axes[0]
    ax.plot(freqs_data[mask_data], amp_data[mask_data], color="black", lw=1.1, label="data flattened spectrum")
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(rows)))
    for color, row in zip(palette, rows):
        mask = (row.freqs >= 0.2) & (row.freqs <= 3.0)
        ax.plot(row.freqs[mask], row.avg_amp[mask], color=color, alpha=0.95, label=f"theory beta={row.beta:g}")
    ax.axvline(data_targets.f1_hz, color="#1b9e77", ls="--", alpha=0.8)
    ax.axvline(data_targets.f2_hz, color="#d95f02", ls="--", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Low-Frequency Data vs Single-Model Quadratic Theory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    betas = np.array([row.beta for row in rows], dtype=float)
    ratios = np.array([row.ratio for row in rows], dtype=float)
    ax.plot(betas, ratios, "o-", color="#7570b3", label="theory harmonic ratio")
    ax.axhline(data_targets.ratio, color="#d95f02", ls="--", label=f"data ratio = {data_targets.ratio:.3f}")
    ax.set_xlabel("Quadratic coefficient beta")
    ax.set_ylabel(f"A(2f)/A(f),  f={rows[0].f1_amp and data_targets.f1_hz:.3f} Hz")
    ax.set_title("Second-Harmonic Ratio vs Beta")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2]
    linear_modes = rows[0].linear_modes_hz
    ax.scatter(np.arange(1, linear_modes.size + 1), linear_modes, color="#1b9e77", s=40, label="same-model beta=0 linear modes")
    for idx, f in enumerate(linear_modes, start=1):
        ax.text(idx, f, f"{f:.2f}", fontsize=8, ha="left", va="bottom")
    ax.axhline(rows[0].freqs[np.argmin(np.abs(rows[0].freqs - data_targets.f1_hz))], color="#1f78b4", ls=":", alpha=0.7)
    ax.axhline(data_targets.f1_hz, color="#1b9e77", ls="--", alpha=0.8, label=f"data fundamental {data_targets.f1_hz:.3f}")
    ax.axhline(data_targets.f2_hz, color="#d95f02", ls="--", alpha=0.8, label=f"data 2f-like {data_targets.f2_hz:.3f}")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Beta->0 Verification Limit Against dgnic Spectrum")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    path = OUTPUT_DIR / "int1_quadratic_theory_compare.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(path: Path, rows: list[SimulationResult], data_targets: DataTargets, fig_path: Path, csv_path: Path) -> Path:
    best = min(rows, key=lambda row: abs(row.ratio - data_targets.ratio))
    lines = []
    lines.append("Single-model quadratic theory comparison")
    lines.append("")
    lines.append("Model:")
    lines.append("  Uses the dgnic linear matrix H as the exact beta=0 limit.")
    lines.append("  Adds one asymmetric quadratic bond-force term: F_quad = beta * k * (delta)^2.")
    lines.append("  This is intentionally a single approach; beta->0 is the verification limit, not a separate model.")
    lines.append("")
    lines.append(f"Data targets: f={data_targets.f1_hz:.6f} Hz, 2f-like={data_targets.f2_hz:.6f} Hz, ratio={data_targets.ratio:.6f}")
    lines.append("")
    lines.append("Beta sweep:")
    for row in rows:
        lines.append(
            f"  beta={row.beta:8.3f} | theory ratio={row.ratio:.6f} | "
            f"A(f)={row.f1_amp:.3e} | A(2f)={row.f2_amp:.3e}"
        )
    lines.append("")
    lines.append(
        f"Closest match to data ratio: beta={best.beta:.3f} with theory ratio={best.ratio:.6f} "
        f"(data={data_targets.ratio:.6f})"
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

    data_targets = resolve_data_targets(
        bond_spacing_mode=str(args.bond_spacing_mode),
        peak_prominence=float(args.peak_prominence),
        merge_hz=float(args.merge_hz),
    )
    rows = [
        run_single_simulation(
            beta=float(beta),
            N=int(args.N),
            m1_kg=float(args.m1_g) * 1e-3,
            m2_kg=float(args.m2_g) * 1e-3,
            k=float(args.k),
            L=float(args.L),
            drive_site=int(args.drive_site),
            drive_freq_hz=float(args.drive_freq_hz),
            drive_accel=float(args.drive_accel),
            tmax=float(args.tmax),
            dt=float(args.dt),
            sample_rate=float(args.sample_rate),
            damping_gamma=float(args.damping_gamma),
            welch_segment_s=float(args.welch_segment_s),
            welch_overlap=float(args.welch_overlap),
            analysis_start_s=float(args.analysis_start_s),
        )
        for beta in args.beta_values
    ]
    csv_path = write_csv(OUTPUT_DIR / "int1_quadratic_theory_compare.csv", rows, data_targets)
    fig_path = plot_results(rows, data_targets)
    summary_path = write_summary(OUTPUT_DIR / "int1_quadratic_theory_compare_summary.txt", rows, data_targets, fig_path, csv_path)

    print(
        f"Resolved data targets: f={data_targets.f1_hz:.6f} Hz, 2f-like={data_targets.f2_hz:.6f} Hz, "
        f"ratio={data_targets.ratio:.6f}"
    )
    for row in rows:
        print(
            f"beta={row.beta:8.3f} | theory ratio={row.ratio:.6f} | "
            f"A(f)={row.f1_amp:.3e} | A(2f)={row.f2_amp:.3e}"
        )
    print(f"Saved figure to {fig_path}")
    print(f"Saved summary to {summary_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
