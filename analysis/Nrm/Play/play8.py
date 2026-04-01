#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

from analysis.tools.bonds import _derive_comoving_signal_matrices, load_bond_signal_dataset
from analysis.tools.io import join_dataset_component, load_track2_dataset
from analysis.tools.signal import hann_window_periodic, next_power_of_two, preprocess_signal
from play1 import CONFIG


@dataclass(frozen=True)
class SignalSet:
    label: str
    frame_times_s: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class DriftStats:
    label: str
    mean: float
    std: float
    vmin: float
    vmax: float
    diff_std: float
    diff_max: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose comoving bond-signal runaway and test stable alternatives.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure.")
    parser.add_argument("--segment-len-s", type=float, default=100.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--snap-bins", type=int, default=3)
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def load_raw_bond_vectors(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = dataset
    track2_x = load_track2_dataset(dataset=join_dataset_component(base, "x"))
    track2_y = load_track2_dataset(dataset=join_dataset_component(base, "y"))
    dx = np.asarray(track2_x.x_positions[:, 1:] - track2_x.x_positions[:, :-1], dtype=float)
    dy = np.asarray(track2_y.x_positions[:, 1:] - track2_y.x_positions[:, :-1], dtype=float)
    return np.asarray(track2_x.frame_times_s, dtype=float), dx, dy


def derive_increment_comoving(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames, n_pairs = dx.shape
    long_inc = np.full((n_frames, n_pairs), np.nan, dtype=float)
    trans_inc = np.full((n_frames, n_pairs), np.nan, dtype=float)

    for pair_idx in range(n_pairs):
        prev_vector: tuple[float, float] | None = None
        prev_xhat: tuple[float, float] | None = None
        prev_yhat: tuple[float, float] | None = None

        for frame_idx in range(n_frames):
            vx = float(dx[frame_idx, pair_idx])
            vy = float(dy[frame_idx, pair_idx])
            if (not np.isfinite(vx)) or (not np.isfinite(vy)):
                prev_vector = None
                prev_xhat = None
                prev_yhat = None
                continue

            norm = float(np.hypot(vx, vy))
            if (not np.isfinite(norm)) or norm <= 0.0:
                prev_vector = None
                prev_xhat = None
                prev_yhat = None
                continue

            curr_xhat = (vx / norm, vy / norm)
            curr_yhat = (-curr_xhat[1], curr_xhat[0])
            if prev_vector is None or prev_xhat is None or prev_yhat is None:
                long_inc[frame_idx, pair_idx] = 0.0
                trans_inc[frame_idx, pair_idx] = 0.0
            else:
                delta_x = vx - prev_vector[0]
                delta_y = vy - prev_vector[1]
                long_inc[frame_idx, pair_idx] = delta_x * prev_xhat[0] + delta_y * prev_xhat[1]
                trans_inc[frame_idx, pair_idx] = delta_x * prev_yhat[0] + delta_y * prev_yhat[1]

            prev_vector = (vx, vy)
            prev_xhat = curr_xhat
            prev_yhat = curr_yhat

    return long_inc, trans_inc


def collect_signal_sets() -> list[SignalSet]:
    default_ds = load_bond_signal_dataset(dataset=f"{CONFIG.dataset}_{CONFIG.component}", bond_spacing_mode="default", component=CONFIG.component)
    purecomoving_ds = load_bond_signal_dataset(dataset=f"{CONFIG.dataset}_{CONFIG.component}", bond_spacing_mode="purecomoving", component=CONFIG.component)

    frame_times_s, dx, dy = load_raw_bond_vectors(CONFIG.dataset)
    current_long, current_trans = _derive_comoving_signal_matrices(
        load_track2_dataset(dataset=join_dataset_component(CONFIG.dataset, "x")),
        load_track2_dataset(dataset=join_dataset_component(CONFIG.dataset, "y")),
    )
    norm = np.hypot(dx, dy)
    long_inc, trans_inc = derive_increment_comoving(dx, dy)

    return [
        SignalSet("default_x_spacing", np.asarray(default_ds.frame_times_s, dtype=float), np.asarray(default_ds.signal_matrix, dtype=float)),
        SignalSet("current_purecomoving_long", np.asarray(purecomoving_ds.frame_times_s, dtype=float), np.asarray(current_long, dtype=float)),
        SignalSet("stable_length", frame_times_s, np.asarray(norm, dtype=float)),
        SignalSet("increment_long", frame_times_s, np.asarray(long_inc, dtype=float)),
        SignalSet("increment_trans", frame_times_s, np.asarray(trans_inc, dtype=float)),
        SignalSet("current_purecomoving_trans", np.asarray(purecomoving_ds.frame_times_s, dtype=float), np.asarray(current_trans, dtype=float)),
    ]


def drift_stats(signal_set: SignalSet, bond_id: int) -> DriftStats:
    y = np.asarray(signal_set.matrix[:, bond_id], dtype=float)
    diff = np.diff(y)
    return DriftStats(
        label=signal_set.label,
        mean=float(np.nanmean(y)),
        std=float(np.nanstd(y)),
        vmin=float(np.nanmin(y)),
        vmax=float(np.nanmax(y)),
        diff_std=float(np.nanstd(diff)),
        diff_max=float(np.nanmax(np.abs(diff))),
    )


def estimate_bicoherence(
    frame_times_s: np.ndarray,
    signal: np.ndarray,
    f1_target: float,
    f2_target: float,
    *,
    segment_len_s: float,
    overlap: float,
    snap_bins: int,
) -> tuple[float, float, float, float]:
    processed, err = preprocess_signal(frame_times_s, signal, longest=False, handlenan=False)
    if processed is None:
        raise ValueError(err)
    nperseg = max(8, int(round(segment_len_s * processed.Fs)))
    if nperseg > processed.y.size:
        raise ValueError("segment length too long for signal")
    step = max(1, nperseg - int(round(overlap * nperseg)))
    nfft = max(nperseg, next_power_of_two(nperseg))
    freqs = np.fft.rfftfreq(nfft, d=processed.dt)
    window = hann_window_periodic(nperseg)
    window_norm = float(np.sum(window))

    spectra: list[np.ndarray] = []
    for start in range(0, processed.y.size - nperseg + 1, step):
        stop = start + nperseg
        seg = processed.y[start:stop] * window
        X = np.fft.rfft(seg, n=nfft) / window_norm
        if X.size > 2:
            X = X.copy()
            X[1:-1] *= 2.0
        spectra.append(X)
    X = np.vstack(spectra)
    mean_amp = np.mean(np.abs(X), axis=0)
    df = float(freqs[1] - freqs[0])

    def snap(target: float) -> int:
        idx = int(round(target / df))
        lo = max(0, idx - snap_bins)
        hi = min(freqs.size, idx + snap_bins + 1)
        return int(lo + np.argmax(mean_amp[lo:hi]))

    i1 = snap(f1_target)
    i2 = snap(f2_target)
    i3 = i1 + i2
    z = X[:, i1] * X[:, i2] * np.conj(X[:, i3])
    denom = np.mean(np.abs(X[:, i1] * X[:, i2]) ** 2) * np.mean(np.abs(X[:, i3]) ** 2)
    b = 0.0 if denom <= 1e-18 else float(np.abs(np.mean(z)) ** 2 / denom)
    return float(freqs[i1]), float(freqs[i2]), float(freqs[i3]), b


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)

    signal_sets = collect_signal_sets()
    bond_id = CONFIG.bond_ids[0]

    print(f"--- Play 8: Comoving Diagnostics ({CONFIG.dataset}) ---")
    print("Bond-0 drift/jump stats:")
    for signal_set in signal_sets:
        stats = drift_stats(signal_set, bond_id)
        print(
            f"{stats.label:<22} mean={stats.mean:9.3f} std={stats.std:8.3f} "
            f"min={stats.vmin:9.3f} max={stats.vmax:9.3f} diff_std={stats.diff_std:7.3f} diff_max={stats.diff_max:7.3f}"
        )

    triads = [
        ("Main Nonlinear Sum", 6.34, 12.053),
        ("Possible 2*8.97 Peak", 8.97, 8.97),
        ("Possible 8.97 Sideband", 8.97, 0.416),
    ]
    compare_sets = [
        next(signal_set for signal_set in signal_sets if signal_set.label == "default_x_spacing"),
        next(signal_set for signal_set in signal_sets if signal_set.label == "current_comoving_long"),
        next(signal_set for signal_set in signal_sets if signal_set.label == "stable_length"),
        next(signal_set for signal_set in signal_sets if signal_set.label == "increment_long"),
    ]

    print("\nBicoherence comparison on bond 0:")
    for signal_set in compare_sets:
        print(f"\n{signal_set.label}:")
        for label, f1, f2 in triads:
            f1_sel, f2_sel, f3_sel, b = estimate_bicoherence(
                signal_set.frame_times_s,
                signal_set.matrix[:, bond_id],
                f1,
                f2,
                segment_len_s=float(args.segment_len_s),
                overlap=float(args.overlap),
                snap_bins=int(args.snap_bins),
            )
            print(f"  {label:<22} {f1_sel:.3f} + {f2_sel:.3f} -> {f3_sel:.3f} | {b:.4f}")

    t = signal_sets[0].frame_times_s
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    plot_names = [
        ("default_x_spacing", "current_comoving_long", "stable_length"),
        ("current_comoving_trans", "increment_trans"),
        ("current_comoving_long", "increment_long"),
    ]
    by_name = {signal_set.label: signal_set for signal_set in signal_sets}

    for ax, names in zip(axes, plot_names):
        for name in names:
            signal_set = by_name[name]
            ax.plot(signal_set.frame_times_s, signal_set.matrix[:, bond_id], linewidth=0.9, label=name)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)

    axes[0].set_title("Bond 0: default vs current comoving vs stable length")
    axes[1].set_title("Bond 0: current vs increment transverse")
    axes[2].set_title("Bond 0: current vs increment longitudinal")
    axes[2].set_xlabel("Time (s)")

    output_path = OUTPUT_DIR / "play8_comoving_diagnostics.png"
    fig.savefig(output_path)
    print(f"\nResults saved to {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
