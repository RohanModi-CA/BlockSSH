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
from play5 import _bicoherence_from_indices
from play7 import build_windows
from PlayInt2.int1 import detect_observed_peaks
from analysis.go.Play.fft_flattening import compute_flattened_component_spectra
from analysis.tools.signal import hann_window_periodic, next_power_of_two
from play6 import collect_segments as collect_data_segments


@dataclass(frozen=True)
class TriadFit:
    f1_hz: float
    f2_hz: float
    f3_hz: float
    b_sq: float
    phase_lock: float
    beta_abs: float
    alpha_abs: float
    start_s: float
    stop_s: float


@dataclass(frozen=True)
class SweepRow:
    beta_model: float
    harmonic_ratio: float
    triad_b_sq: float
    triad_beta_abs: float
    triad_phase_lock: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalized single-model quadratic theory calibrated against the data-extracted beta.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--m1-g", type=float, default=28.04)
    parser.add_argument("--m2-g", type=float, default=60.04)
    parser.add_argument("--k", type=float, default=194.0)
    parser.add_argument("--L", type=float, default=59 * 0.0254)
    parser.add_argument("--drive-site", type=int, default=0)
    parser.add_argument("--drive-freq-hz", type=float, default=0.41748)
    parser.add_argument("--drive-accel", type=float, default=1.6)
    parser.add_argument("--beta-values", type=float, nargs="+", default=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10))
    parser.add_argument("--tmax", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sample-rate", type=float, default=60.0)
    parser.add_argument("--damping-gamma", type=float, default=0.12)
    parser.add_argument("--welch-segment-s", type=float, default=20.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--analysis-start-s", type=float, default=20.0)
    parser.add_argument("--analysis-window-s", type=float, default=20.0)
    parser.add_argument("--analysis-step-s", type=float, default=5.0)
    parser.add_argument("--min-window-segments", type=int, default=4)
    parser.add_argument("--peak-prominence", type=float, default=0.02)
    parser.add_argument("--merge-hz", type=float, default=0.06)
    parser.add_argument("--bond-spacing-mode", choices=("default", "purecomoving"), default="purecomoving")
    parser.add_argument("--snap-bins", type=int, default=2)
    parser.add_argument("--target-beta", type=float, default=0.91, help="Data-side dimensionless beta target to compare against.")
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def load_data_lowfreq_targets(bond_spacing_mode: str, peak_prominence: float, merge_hz: float) -> tuple[float, float]:
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
    return float(p1.freq_hz), float(p2.freq_hz)


def _snap_index(index: int, amplitude: np.ndarray, radius: int) -> int:
    if radius <= 0:
        return int(index)
    lo = max(0, int(index) - int(radius))
    hi = min(amplitude.size, int(index) + int(radius) + 1)
    return int(lo + np.argmax(amplitude[lo:hi]))


def fit_local_triad(
    freqs: np.ndarray,
    X: np.ndarray,
    mids: np.ndarray,
    *,
    f1_target: float,
    f2_target: float,
    analysis_window_s: float,
    analysis_step_s: float,
    min_window_segments: int,
    snap_bins: int,
) -> TriadFit:
    mean_amp = np.mean(np.abs(X), axis=0)
    df = float(freqs[1] - freqs[0])
    i1 = _snap_index(int(round(f1_target / df)), mean_amp, snap_bins)
    i2 = _snap_index(int(round(f2_target / df)), mean_amp, snap_bins)
    i3 = i1 + i2
    windows = build_windows(
        mids,
        analysis_window_s=analysis_window_s,
        analysis_step_s=analysis_step_s,
        min_segments=min_window_segments,
    )
    best_score = -np.inf
    best_phase = 0.0
    best_window = None
    for window in windows:
        subset = X[window.segment_indices]
        b_sq, phase_lock = _bicoherence_from_indices(i1, i2, i3, subset)
        if b_sq > best_score:
            best_score = float(b_sq)
            best_phase = float(phase_lock)
            best_window = window
    if best_window is None:
        raise ValueError("No analysis window available")
    subset = X[best_window.segment_indices]
    z = subset[:, i1] * subset[:, i2]
    y = subset[:, i3]
    denom = np.vdot(z, z)
    alpha = 0.0 + 0.0j if np.abs(denom) <= 1e-18 else np.vdot(z, y) / denom
    predictor_rms = float(np.sqrt(np.mean(np.abs(z) ** 2)))
    child_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    beta_abs = float(np.abs(alpha) * predictor_rms / child_rms) if child_rms > 1e-18 else 0.0
    return TriadFit(
        f1_hz=float(freqs[i1]),
        f2_hz=float(freqs[i2]),
        f3_hz=float(freqs[i3]),
        b_sq=float(best_score),
        phase_lock=float(best_phase),
        beta_abs=float(beta_abs),
        alpha_abs=float(np.abs(alpha)),
        start_s=float(best_window.start_s),
        stop_s=float(best_window.stop_s),
    )


def build_sim_segments(times: np.ndarray, bond: np.ndarray, segment_len_s: float, overlap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(times)))
    fs = 1.0 / dt
    nperseg = max(16, int(round(segment_len_s * fs)))
    nperseg = min(nperseg, bond.shape[0])
    noverlap = min(int(round(overlap * nperseg)), nperseg - 1)
    step = nperseg - noverlap
    nfft = max(nperseg, next_power_of_two(nperseg))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    window = hann_window_periodic(nperseg)
    window_norm = float(np.sum(window))

    records = []
    mids = []
    for bond_id in range(bond.shape[1]):
        for start in range(0, bond.shape[0] - nperseg + 1, step):
            stop = start + nperseg
            segment = (bond[start:stop, bond_id] - np.mean(bond[start:stop, bond_id])) * window
            spec = np.fft.rfft(segment, n=nfft) / window_norm
            if spec.size > 2:
                spec = spec.copy()
                spec[1:-1] *= 2.0
            records.append(spec)
            mids.append(float(np.mean(times[start:stop])))
    if not records:
        raise ValueError("No theory segments built")
    X = np.vstack(records)
    mids = np.asarray(mids, dtype=float)
    avg_amp = np.mean(np.abs(X), axis=0)
    return freqs, X, mids, avg_amp


def quadratic_contact_accel(pos: np.ndarray, masses: np.ndarray, k: float, beta_model: float, ref_scale: float) -> np.ndarray:
    if beta_model == 0.0 or pos.size < 2:
        return np.zeros_like(pos)
    delta = pos[1:] - pos[:-1]
    bond_force = k * beta_model * (delta**2) / ref_scale
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


def simulate(
    *,
    beta_model: float,
    ref_scale: float,
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
    analysis_start_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    chain = solve_chain(N=N, m1=m1_kg, m2=m2_kg, k=k, L=L)
    H = chain.H
    masses = chain.masses
    omega = 2.0 * np.pi * drive_freq_hz

    n_steps = int(round(tmax / dt)) + 1
    sample_step = max(1, int(round((1.0 / sample_rate) / dt)))
    n_samples = (n_steps - 1) // sample_step + 1
    times = np.zeros(n_samples, dtype=float)
    disp = np.zeros((n_samples, N), dtype=float)
    y = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)

    def accel_fn(pos: np.ndarray, vel: np.ndarray, t: float) -> np.ndarray:
        drive = np.zeros(N, dtype=float)
        drive[drive_site] = drive_accel * np.sin(omega * t)
        return (
            -(H @ pos)
            - damping_gamma * vel
            + quadratic_contact_accel(pos, masses, k, beta_model, ref_scale)
            + drive
        )

    sample_idx = 0
    times[sample_idx] = 0.0
    disp[sample_idx] = y
    sample_idx += 1
    for step in range(1, n_steps):
        t = (step - 1) * dt
        y, v = rk4_step(y, v, dt, lambda pos, vel: accel_fn(pos, vel, t))
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(v)):
            raise FloatingPointError("Simulation became non-finite")
        if step % sample_step == 0:
            times[sample_idx] = step * dt
            disp[sample_idx] = y
            sample_idx += 1

    times = times[:sample_idx]
    disp = disp[:sample_idx]
    keep = times >= analysis_start_s
    return times[keep], disp[keep]


def harmonic_ratio(freqs: np.ndarray, avg_amp: np.ndarray, f1_target: float, f2_target: float) -> float:
    a1 = float(np.interp(f1_target, freqs, avg_amp))
    a2 = float(np.interp(f2_target, freqs, avg_amp))
    return float(a2 / a1) if a1 > 0 else np.nan


def write_csv(path: Path, rows: list[SweepRow], data_fit: TriadFit, data_ratio: float, ref_scale: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta_model", "harmonic_ratio", "triad_b_sq", "triad_beta_abs", "triad_phase_lock", "data_beta_abs", "data_ratio", "ref_scale_m"])
        for row in rows:
            writer.writerow(
                [
                    f"{row.beta_model:.6f}",
                    f"{row.harmonic_ratio:.6f}",
                    f"{row.triad_b_sq:.6f}",
                    f"{row.triad_beta_abs:.6f}",
                    f"{row.triad_phase_lock:.6f}",
                    f"{data_fit.beta_abs:.6f}",
                    f"{data_ratio:.6f}",
                    f"{ref_scale:.6e}",
                ]
            )
    return path


def plot_results(
    rows: list[SweepRow],
    data_fit: TriadFit,
    data_ratio: float,
    data_freqs: np.ndarray,
    data_amp: np.ndarray,
    spectra_rows: list[tuple[float, np.ndarray, np.ndarray]],
    target_beta: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    ax = axes[0]
    mask = (data_freqs >= 0.2) & (data_freqs <= 1.5)
    ax.plot(data_freqs[mask], data_amp[mask], color="black", lw=1.2, label="data flattened")
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(spectra_rows)))
    for color, (beta_model, freqs, avg_amp) in zip(palette, spectra_rows):
        m = (freqs >= 0.2) & (freqs <= 1.5)
        ax.plot(freqs[m], avg_amp[m], color=color, alpha=0.9, label=f"theory beta={beta_model:g}")
    ax.axvline(data_fit.f1_hz, color="#1b9e77", ls="--", alpha=0.7)
    ax.axvline(data_fit.f3_hz, color="#d95f02", ls="--", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Low-Frequency Data vs Normalized Quadratic Model")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    betas = np.array([row.beta_model for row in rows], dtype=float)
    beta_fit = np.array([row.triad_beta_abs for row in rows], dtype=float)
    ax.plot(betas, beta_fit, "o-", color="#7570b3", label="theory extracted beta")
    ax.axhline(data_fit.beta_abs, color="#d95f02", ls="--", label=f"data beta ≈ {data_fit.beta_abs:.3f}")
    ax.axvline(target_beta, color="#1b9e77", ls=":", label=f"target input beta ≈ {target_beta:.2f}")
    ax.set_xlabel("Model beta")
    ax.set_ylabel("Extracted beta_abs")
    ax.set_title("Extracted Quadratic Strength vs Input Beta")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2]
    ratios = np.array([row.harmonic_ratio for row in rows], dtype=float)
    b2 = np.array([row.triad_b_sq for row in rows], dtype=float)
    ax.plot(betas, ratios, "o-", color="#1b9e77", label="theory A(2f)/A(f)")
    ax.axhline(data_ratio, color="#d95f02", ls="--", label=f"data ratio = {data_ratio:.3f}")
    ax.axvline(target_beta, color="#1b9e77", ls=":", alpha=0.8, label=f"target input beta ≈ {target_beta:.2f}")
    ax2 = ax.twinx()
    ax2.plot(betas, b2, "s--", color="#7570b3", label="theory bicoherence")
    ax.set_xlabel("Model beta")
    ax.set_ylabel("Harmonic ratio")
    ax2.set_ylabel("Bicoherence")
    ax.set_title("Low-Frequency Harmonic Response vs Input Beta")
    ax.grid(alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    path = OUTPUT_DIR / "int2_beta_calibrated_compare.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(path: Path, rows: list[SweepRow], data_fit: TriadFit, data_ratio: float, ref_scale: float, target_beta: float, fig_path: Path, csv_path: Path) -> Path:
    best = min(rows, key=lambda row: abs(row.triad_beta_abs - data_fit.beta_abs))
    lines = []
    lines.append("Normalized single-model quadratic theory")
    lines.append("")
    lines.append("Model:")
    lines.append("  Same dgnic H matrix for all runs.")
    lines.append("  Quadratic bond force = k * beta * delta^2 / delta_ref")
    lines.append("  delta_ref is the characteristic bond-displacement scale from the beta=0 run.")
    lines.append("  This makes the model beta dimensionless; target comparison is against the data-side beta ~ 0.91.")
    lines.append("")
    lines.append(f"delta_ref = {ref_scale:.6e} m")
    lines.append(f"Target model beta from data-side fit: {target_beta:.6f}")
    lines.append(
        f"Data low-frequency triad: {data_fit.f1_hz:.6f} + {data_fit.f2_hz:.6f} -> {data_fit.f3_hz:.6f} | "
        f"beta_abs={data_fit.beta_abs:.6f} | b2={data_fit.b_sq:.6f} | ratio={data_ratio:.6f}"
    )
    lines.append("")
    lines.append("Sweep:")
    for row in rows:
        lines.append(
            f"  beta_model={row.beta_model:.3f} | extracted_beta={row.triad_beta_abs:.6f} | "
            f"b2={row.triad_b_sq:.6f} | ratio={row.harmonic_ratio:.6f}"
        )
    lines.append("")
    lines.append(
        f"Closest extracted-beta match: beta_model={best.beta_model:.3f}, "
        f"extracted_beta={best.triad_beta_abs:.6f}, data_beta={data_fit.beta_abs:.6f}"
    )
    lines.append("Stable range observed in this sweep is only beta_model <= ~0.10; beta ~ 0.91 lies far outside it.")
    lines.append("")
    lines.append("saved_files:")
    lines.append(str(fig_path))
    lines.append(str(csv_path))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)

    data_f1, data_f3 = load_data_lowfreq_targets(str(args.bond_spacing_mode), float(args.peak_prominence), float(args.merge_hz))
    data_freqs, data_records, _ = collect_data_segments(
        float(args.welch_segment_s),
        float(args.welch_overlap),
        bond_spacing_mode=str(args.bond_spacing_mode),
    )
    X_data = np.vstack([record.spectrum for record in data_records])
    mids_data = np.asarray([record.mid_s for record in data_records], dtype=float)
    data_fit = fit_local_triad(
        data_freqs,
        X_data,
        mids_data,
        f1_target=data_f1,
        f2_target=data_f1,
        analysis_window_s=float(args.analysis_window_s),
        analysis_step_s=float(args.analysis_step_s),
        min_window_segments=int(args.min_window_segments),
        snap_bins=int(args.snap_bins),
    )
    results = compute_flattened_component_spectra(
        dataset="IMG_0681_rot270",
        bond_spacing_mode=str(args.bond_spacing_mode),
        components=("x",),
        use_welch=True,
    )["x"]
    data_flat_freqs = np.asarray(results.freq_hz, dtype=float)
    data_flat_amp = np.asarray(results.flattened, dtype=float)
    data_ratio = float(np.interp(data_f3, data_flat_freqs, data_flat_amp) / np.interp(data_f1, data_flat_freqs, data_flat_amp))

    times0, disp0 = simulate(
        beta_model=0.0,
        ref_scale=1.0,
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
        analysis_start_s=float(args.analysis_start_s),
    )
    bond0 = disp0[:, :-1] - disp0[:, 1:]
    ref_scale = float(np.sqrt(np.mean(bond0**2)))
    if ref_scale <= 1e-12:
        raise ValueError("Reference scale from beta=0 run is too small.")

    rows: list[SweepRow] = []
    spectra_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    for beta_model in args.beta_values:
        times, disp = simulate(
            beta_model=float(beta_model),
            ref_scale=ref_scale,
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
            analysis_start_s=float(args.analysis_start_s),
        )
        bond = disp[:, :-1] - disp[:, 1:]
        freqs_sim, X_sim, mids_sim, avg_amp = build_sim_segments(times, bond, float(args.welch_segment_s), float(args.welch_overlap))
        fit = fit_local_triad(
            freqs_sim,
            X_sim,
            mids_sim,
            f1_target=float(args.drive_freq_hz),
            f2_target=float(args.drive_freq_hz),
            analysis_window_s=float(args.analysis_window_s),
            analysis_step_s=float(args.analysis_step_s),
            min_window_segments=int(args.min_window_segments),
            snap_bins=int(args.snap_bins),
        )
        rows.append(
            SweepRow(
                beta_model=float(beta_model),
                harmonic_ratio=harmonic_ratio(freqs_sim, avg_amp, float(args.drive_freq_hz), 2.0 * float(args.drive_freq_hz)),
                triad_b_sq=float(fit.b_sq),
                triad_beta_abs=float(fit.beta_abs),
                triad_phase_lock=float(fit.phase_lock),
            )
        )
        spectra_rows.append((float(beta_model), freqs_sim, avg_amp))

    csv_path = write_csv(OUTPUT_DIR / "int2_beta_calibrated_compare.csv", rows, data_fit, data_ratio, ref_scale)
    fig_path = plot_results(rows, data_fit, data_ratio, data_flat_freqs, data_flat_amp, spectra_rows, float(args.target_beta))
    summary_path = write_summary(OUTPUT_DIR / "int2_beta_calibrated_compare_summary.txt", rows, data_fit, data_ratio, ref_scale, float(args.target_beta), fig_path, csv_path)

    print(
        f"Data target: {data_fit.f1_hz:.6f} + {data_fit.f2_hz:.6f} -> {data_fit.f3_hz:.6f} | "
        f"data beta_abs={data_fit.beta_abs:.6f} | data ratio={data_ratio:.6f}"
    )
    print(f"Target model beta to compare against: {float(args.target_beta):.6f}")
    print(f"delta_ref(beta=0) = {ref_scale:.6e} m")
    for row in rows:
        print(
            f"beta_model={row.beta_model:.3f} | extracted_beta={row.triad_beta_abs:.6f} | "
            f"b2={row.triad_b_sq:.6f} | ratio={row.harmonic_ratio:.6f}"
        )
    print(f"Saved figure to {fig_path}")
    print(f"Saved summary to {summary_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
