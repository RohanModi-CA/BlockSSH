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

from analysis.go.Play.fft_flattening import compute_flattened_component_spectra
from play5 import _bicoherence_from_indices
from play6 import collect_segments
from play7 import build_windows, empirical_pvalue, score_windows_surrogate
from PlayInt2.int1 import detect_observed_peaks


DATASET = "IMG_0681_rot270"
COMPONENT = "x"


@dataclass(frozen=True)
class Peak:
    freq_hz: float
    amplitude: float
    prominence: float
    rank: int


@dataclass(frozen=True)
class Candidate:
    relation: str
    parent_a_hz: float
    parent_b_hz: float
    child_target_hz: float
    child_peak_hz: float
    parent_a_amp: float
    parent_b_amp: float
    child_amp: float
    child_prom: float
    match_error_hz: float
    i1: int
    i2: int
    i3: int
    best_b_sq: float
    best_phase_lock: float
    surrogate_p: float
    alpha_real: float
    alpha_imag: float
    alpha_abs: float
    beta_abs: float
    best_start_s: float
    best_stop_s: float
    window_index: int
    score: float


@dataclass(frozen=True)
class Accepted:
    step: int
    relation: str
    parent_a_hz: float
    parent_b_hz: float
    child_target_hz: float
    child_peak_hz: float
    best_b_sq: float
    surrogate_p: float
    beta_abs: float
    alpha_real: float
    alpha_imag: float
    best_start_s: float
    best_stop_s: float
    global_target_drop_db: float
    global_target_drop_ratio: float
    active_target_drop_db: float
    active_target_drop_ratio: float
    global_peak_drop_db: float
    global_peak_drop_ratio: float
    active_peak_drop_db: float
    active_peak_drop_ratio: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Residual child-subtraction workflow on flattened peak families using localized bicoherence and fitted complex coupling.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--segment-len-s", type=float, default=100.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--analysis-window-s", type=float, default=100.0)
    parser.add_argument("--analysis-step-s", type=float, default=25.0)
    parser.add_argument("--min-window-segments", type=int, default=6)
    parser.add_argument("--bond-spacing-mode", choices=("default", "comoving"), default="comoving")
    parser.add_argument("--peak-prominence", type=float, default=0.03)
    parser.add_argument("--merge-hz", type=float, default=0.08)
    parser.add_argument("--peak-match-tol-hz", type=float, default=0.12)
    parser.add_argument("--top-parent-count", type=int, default=10)
    parser.add_argument("--n-surrogates", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snap-bins", type=int, default=3)
    parser.add_argument("--min-bicoherence", type=float, default=0.35)
    parser.add_argument("--max-surrogate-p", type=float, default=0.20)
    parser.add_argument("--max-child-over-parent", type=float, default=1.0)
    parser.add_argument("--max-beta-abs", type=float, default=1.25)
    parser.add_argument("--max-accepted", type=int, default=6)
    parser.add_argument("--zoom-halfwidth-hz", type=float, default=0.35)
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def load_flattened_result(*, bond_spacing_mode: str):
    results = compute_flattened_component_spectra(
        dataset=DATASET,
        bond_spacing_mode=bond_spacing_mode,
        components=(COMPONENT,),
        use_welch=True,
    )
    return results[COMPONENT]


def to_peak_list(observed) -> list[Peak]:
    return [Peak(float(p.freq_hz), float(p.amplitude), float(p.prominence), int(p.generator_rank)) for p in observed]


def _snap_index(index: int, amplitude: np.ndarray, radius: int) -> int:
    if radius <= 0:
        return int(index)
    lo = max(0, int(index) - int(radius))
    hi = min(amplitude.size, int(index) + int(radius) + 1)
    return int(lo + np.argmax(amplitude[lo:hi]))


def _fit_alpha(X_subset: np.ndarray, i1: int, i2: int, i3: int) -> tuple[complex, float]:
    z = X_subset[:, i1] * X_subset[:, i2]
    y = X_subset[:, i3]
    denom = np.vdot(z, z)
    if np.abs(denom) <= 1e-18:
        return 0.0 + 0.0j, 0.0
    alpha = np.vdot(z, y) / denom
    predictor_rms = float(np.sqrt(np.mean(np.abs(z) ** 2)))
    child_rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    beta_abs = float(np.abs(alpha) * predictor_rms / child_rms) if child_rms > 1e-18 else 0.0
    return complex(alpha), beta_abs


def _best_window(
    X: np.ndarray,
    windows,
    *,
    i1: int,
    i2: int,
    i3: int,
    n_surrogates: int,
    rng: np.random.Generator,
) -> tuple[int, float, float, float, complex, float]:
    scores = np.array([_bicoherence_from_indices(i1, i2, i3, X[w.segment_indices])[0] for w in windows], dtype=float)
    if scores.size == 0:
        raise ValueError("No windows available")
    best_idx = int(np.argmax(scores))
    best_subset = X[windows[best_idx].segment_indices]
    best_b_sq, best_phase_lock = _bicoherence_from_indices(i1, i2, i3, best_subset)
    alpha, beta_abs = _fit_alpha(best_subset, i1, i2, i3)
    null_max = []
    for _ in range(n_surrogates):
        null_scores = score_windows_surrogate(X, windows, i1=i1, i2=i2, i3=i3, rng=rng)
        null_max.append(float(np.max(null_scores)))
    surrogate_p = empirical_pvalue(float(best_b_sq), np.asarray(null_max, dtype=float))
    return best_idx, float(best_b_sq), float(best_phase_lock), float(surrogate_p), alpha, float(beta_abs)


def build_candidates(
    peaks: list[Peak],
    freqs: np.ndarray,
    mean_amp: np.ndarray,
    X: np.ndarray,
    windows,
    *,
    top_parent_count: int,
    peak_match_tol_hz: float,
    snap_bins: int,
    max_child_over_parent: float,
    n_surrogates: int,
    rng: np.random.Generator,
) -> list[Candidate]:
    parents = sorted(peaks, key=lambda peak: peak.amplitude, reverse=True)[: min(top_parent_count, len(peaks))]
    seen: set[tuple[str, float, float, float]] = set()
    out: list[Candidate] = []
    df = float(freqs[1] - freqs[0])

    for idx, peak_a in enumerate(parents):
        for relation, peak_b in [("2x", peak_a)] + [("sum", peak_b) for peak_b in parents[idx + 1 :]]:
            if relation == "2x":
                child_target = 2.0 * peak_a.freq_hz
                pa, pb = peak_a, peak_a
            else:
                pa, pb = peak_a, peak_b
                child_target = pa.freq_hz + pb.freq_hz
            nearest_child = min(peaks, key=lambda peak: abs(peak.freq_hz - child_target))
            match_error = abs(nearest_child.freq_hz - child_target)
            if match_error > peak_match_tol_hz:
                continue
            if abs(nearest_child.freq_hz - pa.freq_hz) < 0.05 or abs(nearest_child.freq_hz - pb.freq_hz) < 0.05:
                continue
            if nearest_child.amplitude > max(pa.amplitude, pb.amplitude) * max_child_over_parent:
                continue

            i1 = _snap_index(int(round(pa.freq_hz / df)), mean_amp, snap_bins)
            i2 = _snap_index(int(round(pb.freq_hz / df)), mean_amp, snap_bins)
            if relation == "2x":
                i2 = i1
            i3 = i1 + i2
            if i3 >= freqs.size:
                continue

            key = (relation, round(freqs[i1], 3), round(freqs[i2], 3), round(nearest_child.freq_hz, 3))
            if key in seen:
                continue
            seen.add(key)

            best_idx, best_b_sq, best_phase_lock, surrogate_p, alpha, beta_abs = _best_window(
                X,
                windows,
                i1=i1,
                i2=i2,
                i3=i3,
                n_surrogates=n_surrogates,
                rng=rng,
            )
            score = float(nearest_child.amplitude * best_b_sq * max(0.0, 1.0 - surrogate_p))
            out.append(
                Candidate(
                    relation=relation,
                    parent_a_hz=float(freqs[i1]),
                    parent_b_hz=float(freqs[i2]),
                    child_target_hz=float(freqs[i3]),
                    child_peak_hz=float(nearest_child.freq_hz),
                    parent_a_amp=float(pa.amplitude),
                    parent_b_amp=float(pb.amplitude),
                    child_amp=float(nearest_child.amplitude),
                    child_prom=float(nearest_child.prominence),
                    match_error_hz=float(match_error),
                    i1=int(i1),
                    i2=int(i2),
                    i3=int(i3),
                    best_b_sq=float(best_b_sq),
                    best_phase_lock=float(best_phase_lock),
                    surrogate_p=float(surrogate_p),
                    alpha_real=float(np.real(alpha)),
                    alpha_imag=float(np.imag(alpha)),
                    alpha_abs=float(np.abs(alpha)),
                    beta_abs=float(beta_abs),
                    best_start_s=float(windows[best_idx].start_s),
                    best_stop_s=float(windows[best_idx].stop_s),
                    window_index=int(best_idx),
                    score=float(score),
                )
            )
    return sorted(out, key=lambda row: row.score, reverse=True)


def local_flattened_amplitude(avg_abs: np.ndarray, transfer_interp: np.ndarray) -> np.ndarray:
    return np.asarray(avg_abs, dtype=float) * np.asarray(transfer_interp, dtype=float)


def local_drop_metric(flat_before: np.ndarray, flat_after: np.ndarray, freqs: np.ndarray, center_hz: float, halfwidth_hz: float) -> tuple[float, float]:
    mask = (freqs >= center_hz - halfwidth_hz) & (freqs <= center_hz + halfwidth_hz)
    if not np.any(mask):
        return 0.0, 1.0
    local_before = float(np.max(flat_before[mask]))
    local_after = float(np.max(flat_after[mask]))
    ratio = local_after / local_before if local_before > 1e-18 else 1.0
    drop_db = float(20.0 * np.log10(max(ratio, 1e-12)))
    return drop_db, float(ratio)


def exact_bin_drop_metric(flat_before: np.ndarray, flat_after: np.ndarray, freqs: np.ndarray, center_hz: float) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(freqs - center_hz)))
    before = float(flat_before[idx])
    after = float(flat_after[idx])
    ratio = after / before if before > 1e-18 else 1.0
    drop_db = float(20.0 * np.log10(max(ratio, 1e-12)))
    return drop_db, float(ratio)


def subtract_candidates(
    candidates: list[Candidate],
    X_initial: np.ndarray,
    windows,
    freqs: np.ndarray,
    transfer_interp: np.ndarray,
    *,
    min_bicoherence: float,
    max_surrogate_p: float,
    max_beta_abs: float,
    max_accepted: int,
    zoom_halfwidth_hz: float,
) -> tuple[np.ndarray, list[Accepted], list[np.ndarray]]:
    X_res = X_initial.copy()
    accepted: list[Accepted] = []
    spectra_steps = [local_flattened_amplitude(np.mean(np.abs(X_res), axis=0), transfer_interp)]
    used_childs: set[float] = set()

    for cand in candidates:
        if len(accepted) >= max_accepted:
            break
        if cand.best_b_sq < min_bicoherence or cand.surrogate_p > max_surrogate_p or cand.beta_abs > max_beta_abs:
            continue
        child_key = round(cand.child_peak_hz, 3)
        if child_key in used_childs:
            continue

        window = windows[cand.window_index]
        subset_idx = window.segment_indices
        X_before = X_res.copy()
        modeled = (cand.alpha_real + 1j * cand.alpha_imag) * X_res[subset_idx, cand.i1] * X_res[subset_idx, cand.i2]
        X_res[subset_idx, cand.i3] = X_res[subset_idx, cand.i3] - modeled

        flat_before = local_flattened_amplitude(np.mean(np.abs(X_before), axis=0), transfer_interp)
        flat_after = local_flattened_amplitude(np.mean(np.abs(X_res), axis=0), transfer_interp)
        global_target_drop_db, global_target_ratio = exact_bin_drop_metric(flat_before, flat_after, freqs, cand.child_target_hz)
        global_peak_drop_db, global_peak_ratio = exact_bin_drop_metric(flat_before, flat_after, freqs, cand.child_peak_hz)
        flat_active_before = local_flattened_amplitude(np.mean(np.abs(X_before[subset_idx]), axis=0), transfer_interp)
        flat_active_after = local_flattened_amplitude(np.mean(np.abs(X_res[subset_idx]), axis=0), transfer_interp)
        active_target_drop_db, active_target_ratio = exact_bin_drop_metric(flat_active_before, flat_active_after, freqs, cand.child_target_hz)
        active_peak_drop_db, active_peak_ratio = exact_bin_drop_metric(flat_active_before, flat_active_after, freqs, cand.child_peak_hz)

        accepted.append(
            Accepted(
                step=len(accepted) + 1,
                relation=cand.relation,
                parent_a_hz=float(cand.parent_a_hz),
                parent_b_hz=float(cand.parent_b_hz),
                child_target_hz=float(cand.child_target_hz),
                child_peak_hz=float(cand.child_peak_hz),
                best_b_sq=float(cand.best_b_sq),
                surrogate_p=float(cand.surrogate_p),
                beta_abs=float(cand.beta_abs),
                alpha_real=float(cand.alpha_real),
                alpha_imag=float(cand.alpha_imag),
                best_start_s=float(cand.best_start_s),
                best_stop_s=float(cand.best_stop_s),
                global_target_drop_db=float(global_target_drop_db),
                global_target_drop_ratio=float(global_target_ratio),
                active_target_drop_db=float(active_target_drop_db),
                active_target_drop_ratio=float(active_target_ratio),
                global_peak_drop_db=float(global_peak_drop_db),
                global_peak_drop_ratio=float(global_peak_ratio),
                active_peak_drop_db=float(active_peak_drop_db),
                active_peak_drop_ratio=float(active_peak_ratio),
            )
        )
        spectra_steps.append(flat_after)
        used_childs.add(child_key)
    return X_res, accepted, spectra_steps


def save_candidate_csv(path: Path, rows: list[Candidate]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relation", "parent_a_hz", "parent_b_hz", "child_target_hz", "child_peak_hz",
                "parent_a_amp", "parent_b_amp", "child_amp", "child_prom", "match_error_hz",
                "b_sq", "phase_lock", "surrogate_p", "alpha_abs", "beta_abs", "start_s", "stop_s", "score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.relation,
                    f"{row.parent_a_hz:.6f}",
                    f"{row.parent_b_hz:.6f}",
                    f"{row.child_target_hz:.6f}",
                    f"{row.child_peak_hz:.6f}",
                    f"{row.parent_a_amp:.6f}",
                    f"{row.parent_b_amp:.6f}",
                    f"{row.child_amp:.6f}",
                    f"{row.child_prom:.6f}",
                    f"{row.match_error_hz:.6f}",
                    f"{row.best_b_sq:.6f}",
                    f"{row.best_phase_lock:.6f}",
                    f"{row.surrogate_p:.6f}",
                    f"{row.alpha_abs:.6e}",
                    f"{row.beta_abs:.6f}",
                    f"{row.best_start_s:.3f}",
                    f"{row.best_stop_s:.3f}",
                    f"{row.score:.6f}",
                ]
            )
    return path


def save_accepted_csv(path: Path, rows: list[Accepted]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step", "relation", "parent_a_hz", "parent_b_hz", "child_target_hz", "child_peak_hz", "b_sq",
                "surrogate_p", "beta_abs", "alpha_real", "alpha_imag",
                "best_start_s", "best_stop_s",
                "global_target_drop_db", "global_target_drop_ratio",
                "active_target_drop_db", "active_target_drop_ratio",
                "global_peak_drop_db", "global_peak_drop_ratio",
                "active_peak_drop_db", "active_peak_drop_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.step, row.relation,
                    f"{row.parent_a_hz:.6f}", f"{row.parent_b_hz:.6f}", f"{row.child_target_hz:.6f}", f"{row.child_peak_hz:.6f}",
                    f"{row.best_b_sq:.6f}", f"{row.surrogate_p:.6f}", f"{row.beta_abs:.6f}",
                    f"{row.alpha_real:.6e}", f"{row.alpha_imag:.6e}",
                    f"{row.best_start_s:.3f}", f"{row.best_stop_s:.3f}",
                    f"{row.global_target_drop_db:.6f}", f"{row.global_target_drop_ratio:.6f}",
                    f"{row.active_target_drop_db:.6f}", f"{row.active_target_drop_ratio:.6f}",
                    f"{row.global_peak_drop_db:.6f}", f"{row.global_peak_drop_ratio:.6f}",
                    f"{row.active_peak_drop_db:.6f}", f"{row.active_peak_drop_ratio:.6f}",
                ]
            )
    return path


def plot_all(
    *,
    freqs_flat: np.ndarray,
    flat_full: np.ndarray,
    candidates: list[Candidate],
    accepted: list[Accepted],
    spectra_steps: list[np.ndarray],
    X_initial: np.ndarray,
    X_residual: np.ndarray,
    transfer_interp: np.ndarray,
    zoom_halfwidth_hz: float,
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    ax = axes[0]
    top = candidates[: min(12, len(candidates))]
    y = np.arange(len(top))
    ax.barh(y, [row.score for row in top], color="#7570b3", alpha=0.85)
    ax.set_yticks(y, [f"{row.parent_a_hz:.2f}{'*2' if row.relation=='2x' else '+'+f'{row.parent_b_hz:.2f}'} -> {row.child_peak_hz:.2f}" for row in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate score")
    ax.set_title("Top Candidate Children Before Subtraction")
    ax.grid(axis="x", alpha=0.25)
    for yi, row in zip(y, top):
        ax.text(row.score, yi, f"  b2={row.best_b_sq:.2f}, p={row.surrogate_p:.2f}, beta={row.beta_abs:.2f}", fontsize=7, va="center")

    ax = axes[1]
    mask = (freqs_flat >= 0.2) & (freqs_flat <= 26.0)
    ax.plot(freqs_flat[mask], flat_full[mask], color="black", lw=1.2, label="initial")
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(1, len(spectra_steps) - 1)))
    for idx, (spec, color) in enumerate(zip(spectra_steps[1:], colors), start=1):
        ax.plot(freqs_flat[mask], spec[mask], color=color, lw=1.0, alpha=0.9, label=f"after step {idx}")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Flattened amplitude")
    ax.set_title("Stepwise Residual Spectrum")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path = OUTPUT_DIR / "int1_candidates_and_steps.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    if accepted:
        n = len(accepted)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(3.5, 2.6 * n)), constrained_layout=True)
        axes = np.atleast_1d(axes)
        flat_initial = local_flattened_amplitude(np.mean(np.abs(X_initial), axis=0), transfer_interp)
        flat_res = local_flattened_amplitude(np.mean(np.abs(X_residual), axis=0), transfer_interp)
        for ax, row in zip(axes, accepted):
            mask = (freqs_flat >= row.child_peak_hz - zoom_halfwidth_hz) & (freqs_flat <= row.child_peak_hz + zoom_halfwidth_hz)
            ax.plot(freqs_flat[mask], flat_initial[mask], color="black", lw=1.2, label="initial")
            ax.plot(freqs_flat[mask], flat_res[mask], color="#d95f02", lw=1.1, label="final residual")
            ax.axvline(row.child_peak_hz, color="#7570b3", ls="--", alpha=0.8)
            ax.axvline(row.parent_a_hz, color="#1b9e77", ls=":", alpha=0.7)
            if row.relation != "2x":
                ax.axvline(row.parent_b_hz, color="#1b9e77", ls=":", alpha=0.7)
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(
                f"step {row.step}: {row.parent_a_hz:.3f} {row.relation} {row.parent_b_hz:.3f} -> target {row.child_target_hz:.3f}, peak {row.child_peak_hz:.3f} | "
                f"b2={row.best_b_sq:.2f}, p={row.surrogate_p:.2f}, target drop={row.active_target_drop_db:.2f} dB, peak drop={row.active_peak_drop_db:.2f} dB"
            )
        axes[0].legend(fontsize=8)
        path = OUTPUT_DIR / "int1_local_child_zooms.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    initial_bin = local_flattened_amplitude(np.mean(np.abs(X_initial), axis=0), transfer_interp)
    residual_bin = local_flattened_amplitude(np.mean(np.abs(X_residual), axis=0), transfer_interp)
    change = residual_bin - initial_bin
    mask = (freqs_flat >= 0.2) & (freqs_flat <= 26.0)
    ax.plot(freqs_flat[mask], change[mask], color="#d95f02", lw=1.0)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    for row in accepted:
        ax.axvline(row.child_peak_hz, color="#7570b3", ls="--", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual - initial flattened amplitude")
    ax.set_title("Where Subtraction Actually Changed the Spectrum")
    ax.grid(alpha=0.25)
    path = OUTPUT_DIR / "int1_global_change.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    return paths


def write_summary(path: Path, candidates: list[Candidate], accepted: list[Accepted], figure_paths: list[Path], csv_paths: list[Path], args: argparse.Namespace) -> Path:
    lines: list[str] = []
    lines.append("PlayInt3 residual subtraction prototype")
    lines.append("")
    lines.append("Workflow:")
    lines.append("  1. detect strongest peaks on transfer-flattened spectrum")
    lines.append("  2. generate sum/2x child candidates from top parents")
    lines.append("  3. validate candidates with localized bicoherence + surrogate null")
    lines.append("  4. fit complex alpha in the best window")
    lines.append("  5. subtract modeled child contribution only in active windows")
    lines.append("")
    lines.append(f"top_parent_count = {args.top_parent_count}")
    lines.append(f"candidate_count = {len(candidates)}")
    lines.append(f"accepted_count = {len(accepted)}")
    lines.append("")
    lines.append("Top candidates:")
    for row in candidates[:12]:
        lines.append(
            f"  {row.parent_a_hz:.3f} {row.relation} {row.parent_b_hz:.3f} -> {row.child_peak_hz:.3f} | "
            f"b2={row.best_b_sq:.3f} | p={row.surrogate_p:.3f} | beta={row.beta_abs:.3f} | score={row.score:.3f}"
        )
    lines.append("")
    lines.append("Accepted subtractions:")
    for row in accepted:
        lines.append(
                f"  step {row.step}: {row.parent_a_hz:.3f} {row.relation} {row.parent_b_hz:.3f} -> {row.child_peak_hz:.3f} | "
                f"b2={row.best_b_sq:.3f} | p={row.surrogate_p:.3f} | beta={row.beta_abs:.3f} | "
                f"window={row.best_start_s:.1f}-{row.best_stop_s:.1f}s | "
                f"target drop={row.active_target_drop_db:.2f} dB | peak drop={row.active_peak_drop_db:.2f} dB"
            )
    lines.append("")
    lines.append("saved_files:")
    for p in [*figure_paths, *csv_paths]:
        lines.append(str(p))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)
    rng = np.random.default_rng(args.seed)

    flat_result = load_flattened_result(bond_spacing_mode=str(args.bond_spacing_mode))
    freqs_flat = np.asarray(flat_result.freq_hz, dtype=float)
    flat_full = np.asarray(flat_result.flattened, dtype=float)
    transfer = np.asarray(flat_result.transfer, dtype=float)
    observed = detect_observed_peaks(
        freqs_flat,
        flat_full,
        peak_prominence=float(args.peak_prominence),
        merge_hz=float(args.merge_hz),
        min_freq_hz=0.2,
        max_freq_hz=26.0,
    )
    peaks = to_peak_list(observed)

    freqs_seg, records, mean_amplitude = collect_segments(
        float(args.segment_len_s),
        float(args.overlap),
        bond_spacing_mode=str(args.bond_spacing_mode),
    )
    transfer_interp = np.interp(freqs_seg, freqs_flat, transfer)
    X = np.vstack([record.spectrum for record in records])
    mids = np.asarray([record.mid_s for record in records], dtype=float)
    windows = build_windows(
        mids,
        analysis_window_s=float(args.analysis_window_s),
        analysis_step_s=float(args.analysis_step_s),
        min_segments=int(args.min_window_segments),
    )
    candidates = build_candidates(
        peaks,
        freqs_seg,
        mean_amplitude,
        X,
        windows,
        top_parent_count=int(args.top_parent_count),
        peak_match_tol_hz=float(args.peak_match_told_hz) if hasattr(args, "peak_match_told_hz") else float(args.peak_match_tol_hz),
        snap_bins=int(args.snap_bins),
        max_child_over_parent=float(args.max_child_over_parent),
        n_surrogates=int(args.n_surrogates),
        rng=rng,
    )
    X_res, accepted, spectra_steps = subtract_candidates(
        candidates,
        X,
        windows,
        freqs_seg,
        transfer_interp,
        min_bicoherence=float(args.min_bicoherence),
        max_surrogate_p=float(args.max_surrogate_p),
        max_beta_abs=float(args.max_beta_abs),
        max_accepted=int(args.max_accepted),
        zoom_halfwidth_hz=float(args.zoom_halfwidth_hz),
    )

    figure_paths = plot_all(
        freqs_flat=freqs_seg,
        flat_full=local_flattened_amplitude(np.mean(np.abs(X), axis=0), transfer_interp),
        candidates=candidates,
        accepted=accepted,
        spectra_steps=spectra_steps,
        X_initial=X,
        X_residual=X_res,
        transfer_interp=transfer_interp,
        zoom_halfwidth_hz=float(args.zoom_halfwidth_hz),
    )
    cand_csv = save_candidate_csv(OUTPUT_DIR / "int1_candidates.csv", candidates)
    acc_csv = save_accepted_csv(OUTPUT_DIR / "int1_accepted.csv", accepted)
    summary_path = write_summary(OUTPUT_DIR / "int1_summary.txt", candidates, accepted, figure_paths, [cand_csv, acc_csv], args)

    print(f"Detected {len(peaks)} peaks; built {len(candidates)} candidates; accepted {len(accepted)} subtractions.")
    print("Top accepted:")
    for row in accepted:
        print(
            f"  step {row.step}: {row.parent_a_hz:.3f} {row.relation} {row.parent_b_hz:.3f} -> {row.child_peak_hz:.3f} | "
            f"b2={row.best_b_sq:.3f} | p={row.surrogate_p:.3f} | beta={row.beta_abs:.3f} | "
            f"target drop={row.active_target_drop_db:.2f} dB | peak drop={row.active_peak_drop_db:.2f} dB"
        )
    print(f"Saved summary to {summary_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
