#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import compute_complex_spectrogram, preprocess_signal


@dataclass(frozen=True)
class CoherenceHit:
    index: int
    time_s: float
    coherence_loss: float
    coherence_value: float
    energy_value: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect candidate impacts from sudden loss of sliding-FFT phase coherence.",
    )
    parser.add_argument("dataset", help="Dataset stem, e.g. IMG_0681_rot270")
    parser.add_argument("--component", default="x", choices=["x", "y", "a"])
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument("--bond-spacing-mode", default="purecomoving", choices=["default", "purecomoving"])
    parser.add_argument("--sliding-len-s", type=float, default=1.5)
    parser.add_argument("--coherence-min-hz", type=float, default=2.0)
    parser.add_argument("--coherence-max-hz", type=float, default=25.0)
    parser.add_argument("--energy-min-hz", type=float, default=2.0)
    parser.add_argument("--energy-max-hz", type=float, default=25.0)
    parser.add_argument("--coherence-smooth-s", type=float, default=0.5)
    parser.add_argument("--loss-threshold-sigma", type=float, default=2.5)
    parser.add_argument("--min-hit-separation-s", type=float, default=8.0)
    parser.add_argument(
        "--energy-gate-quantile",
        type=float,
        default=0.25,
        help="Only keep coherence-loss peaks where the broadband energy is at least this quantile. Default: 0.25",
    )
    parser.add_argument("--focus-hit", type=int, action="append", default=[])
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


def _repair_time_vector(frame_times_s: np.ndarray, frame_numbers: np.ndarray) -> tuple[np.ndarray, float, bool]:
    t = np.asarray(frame_times_s, dtype=float)
    frame_numbers = np.asarray(frame_numbers, dtype=float)
    positive_dt = np.diff(t)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if positive_dt.size == 0:
        raise ValueError("Could not infer a positive frame interval from frameTimes_s")
    dt = float(np.median(positive_dt))
    rebuilt = (frame_numbers - float(frame_numbers[0])) * dt
    invalid_tail = bool(np.any(t[-max(1, min(32, t.size)) :] == 0.0))
    nonmonotone_fraction = float(np.mean(np.diff(t) <= 0)) if t.size > 1 else 0.0
    clearly_bad = invalid_tail or nonmonotone_fraction > 0.001 or (not np.all(np.isfinite(t)))
    return (rebuilt if clearly_bad else t), dt, clearly_bad


def _moving_average(y: np.ndarray, window_samples: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.size == 0 or window_samples <= 1:
        return arr.copy()
    kernel = np.ones(int(window_samples), dtype=float) / float(window_samples)
    return np.convolve(arr, kernel, mode="same")


def _robust_threshold(y: np.ndarray, sigma: float) -> tuple[float, float, float]:
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot build threshold from all-NaN input")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_std = 1.4826 * mad
    if robust_std <= 0.0 or not np.isfinite(robust_std):
        robust_std = float(np.std(finite))
    return median + sigma * robust_std, median, robust_std


def _aggregate(processed_signals: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(processed_signals), axis=0)


def _prepare_processed_bonds(
    dataset: str,
    *,
    component: str,
    track_data_root: str | None,
    bond_spacing_mode: str,
) -> tuple[np.ndarray, list[np.ndarray], float, bool]:
    dataset_name = f"{dataset}_{component}"
    bond_dataset = load_bond_signal_dataset(
        dataset=dataset_name,
        track_data_root=track_data_root,
        bond_spacing_mode=bond_spacing_mode,
        component=component,
    )

    from analysis.tools.io import load_track2_dataset

    track2 = load_track2_dataset(dataset=dataset_name, track_data_root=track_data_root)
    repaired_t, _, repaired_flag = _repair_time_vector(track2.frame_times_s, track2.frame_numbers)

    processed_t: np.ndarray | None = None
    processed_signals: list[np.ndarray] = []
    for bond_idx in range(bond_dataset.signal_matrix.shape[1]):
        processed, error = preprocess_signal(repaired_t, bond_dataset.signal_matrix[:, bond_idx])
        if processed is None:
            print(f"Skipping bond {bond_idx + 1}: {error}", file=sys.stderr)
            continue
        if processed_t is None:
            processed_t = np.asarray(processed.t, dtype=float)
        processed_signals.append(np.asarray(processed.y, dtype=float))

    if processed_t is None or not processed_signals:
        raise ValueError("No bond signals were successfully preprocessed")

    fs = 1.0 / float(np.median(np.diff(processed_t)))
    return processed_t, processed_signals, fs, repaired_flag


def _build_spectrogram_stack(
    processed_signals: list[np.ndarray],
    fs: float,
    sliding_len_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    specs = [compute_complex_spectrogram(y, fs, sliding_len_s) for y in processed_signals]
    if any(spec is None for spec in specs):
        raise ValueError("Sliding FFT window is too short for at least one processed bond signal")
    ref = specs[0]
    assert ref is not None
    stack = np.stack([spec.S_complex for spec in specs if spec is not None], axis=0)
    return np.asarray(ref.t, dtype=float), np.asarray(ref.f, dtype=float), np.asarray(stack)


def _band_mask(freq: np.ndarray, low: float, high: float, *, label: str) -> np.ndarray:
    mask = (freq >= float(low)) & (freq <= float(high))
    if not np.any(mask):
        raise ValueError(f"No frequency bins fall in {label} range [{low:.3g}, {high:.3g}] Hz")
    return mask


def _compute_energy_trace(spec_stack: np.ndarray, freq_mask: np.ndarray) -> np.ndarray:
    power = np.abs(spec_stack[:, freq_mask, :]) ** 2
    return np.sqrt(np.mean(power, axis=(0, 1)))


def _compute_phase_coherence_trace(spec_stack: np.ndarray, freq_mask: np.ndarray) -> np.ndarray:
    band_stack = np.asarray(spec_stack[:, freq_mask, :], dtype=np.complex128)
    n_times = band_stack.shape[2]
    coherence = np.full(n_times, np.nan, dtype=float)
    coherence[0] = 1.0

    prev_vector = band_stack[:, :, 0].reshape(-1)
    prev_norm = float(np.linalg.norm(prev_vector))
    for time_idx in range(1, n_times):
        curr_vector = band_stack[:, :, time_idx].reshape(-1)
        curr_norm = float(np.linalg.norm(curr_vector))
        if prev_norm <= 0.0 or curr_norm <= 0.0 or (not np.isfinite(prev_norm)) or (not np.isfinite(curr_norm)):
            coherence[time_idx] = np.nan
        else:
            overlap = np.vdot(prev_vector, curr_vector)
            coherence[time_idx] = float(np.abs(overlap) / (prev_norm * curr_norm))
        prev_vector = curr_vector
        prev_norm = curr_norm
    return coherence


def _detect_coherence_hits(
    t_spec: np.ndarray,
    coherence_loss: np.ndarray,
    energy_trace: np.ndarray,
    *,
    loss_threshold_sigma: float,
    min_hit_separation_s: float,
    energy_gate_quantile: float,
) -> tuple[list[CoherenceHit], float, float, float, float]:
    threshold, loss_median, loss_std = _robust_threshold(coherence_loss, loss_threshold_sigma)
    energy_gate = float(np.quantile(energy_trace[np.isfinite(energy_trace)], energy_gate_quantile))
    dt_spec = float(np.median(np.diff(t_spec)))
    min_distance = max(1, int(round(min_hit_separation_s / dt_spec)))
    peak_indices, _ = find_peaks(
        coherence_loss,
        height=threshold,
        distance=min_distance,
        prominence=max(loss_std, 1e-12),
    )

    hits: list[CoherenceHit] = []
    for peak_idx in peak_indices:
        if float(energy_trace[peak_idx]) < energy_gate:
            continue
        hits.append(
            CoherenceHit(
                index=len(hits) + 1,
                time_s=float(t_spec[peak_idx]),
                coherence_loss=float(coherence_loss[peak_idx]),
                coherence_value=float(1.0 - coherence_loss[peak_idx]),
                energy_value=float(energy_trace[peak_idx]),
            )
        )
    return hits, threshold, loss_median, loss_std, energy_gate


def _choose_focus_hits(hits: list[CoherenceHit], explicit: list[int]) -> list[CoherenceHit]:
    by_id = {hit.index: hit for hit in hits}
    chosen: list[CoherenceHit] = []
    for hit_id in explicit:
        if hit_id in by_id:
            chosen.append(by_id[hit_id])
    if chosen:
        return chosen[:4]
    if not hits:
        return []
    losses = np.asarray([hit.coherence_loss for hit in hits], dtype=float)
    order = np.argsort(losses)
    picks = [order[0], order[len(order) // 2], order[-1]]
    for idx in picks:
        hit = hits[int(idx)]
        if all(existing.index != hit.index for existing in chosen):
            chosen.append(hit)
    return chosen[:4]


def _save_hits_csv(path: Path, dataset: str, component: str, hits: list[CoherenceHit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "component", "hit_index", "time_s", "coherence_loss", "coherence_value", "energy_value"])
        for hit in hits:
            writer.writerow(
                [
                    dataset,
                    component,
                    int(hit.index),
                    f"{hit.time_s:.9f}",
                    f"{hit.coherence_loss:.9f}",
                    f"{hit.coherence_value:.9f}",
                    f"{hit.energy_value:.9f}",
                ]
            )


def main() -> int:
    args = build_parser().parse_args()
    processed_t, processed_signals, fs, repaired_flag = _prepare_processed_bonds(
        args.dataset,
        component=args.component,
        track_data_root=args.track_data_root,
        bond_spacing_mode=args.bond_spacing_mode,
    )
    aggregate_signal = _aggregate(processed_signals)
    t_rel, freq, spec_stack = _build_spectrogram_stack(processed_signals, fs, args.sliding_len_s)
    t_spec = t_rel + float(processed_t[0])

    coherence_mask = _band_mask(freq, args.coherence_min_hz, args.coherence_max_hz, label="coherence")
    energy_mask = _band_mask(freq, args.energy_min_hz, args.energy_max_hz, label="energy")

    raw_coherence = _compute_phase_coherence_trace(spec_stack, coherence_mask)
    coherence_smooth_samples = max(1, int(round(args.coherence_smooth_s / float(np.median(np.diff(t_spec))))))
    coherence = _moving_average(np.nan_to_num(raw_coherence, nan=np.nanmedian(raw_coherence)), coherence_smooth_samples)
    coherence = np.clip(coherence, 0.0, 1.0)
    coherence_loss = np.maximum(0.0, 1.0 - coherence)
    energy_trace = _compute_energy_trace(spec_stack, energy_mask)

    hits, loss_threshold, loss_median, loss_std, energy_gate = _detect_coherence_hits(
        t_spec,
        coherence_loss,
        energy_trace,
        loss_threshold_sigma=args.loss_threshold_sigma,
        min_hit_separation_s=args.min_hit_separation_s,
        energy_gate_quantile=args.energy_gate_quantile,
    )
    focus_hits = _choose_focus_hits(hits, args.focus_hit)

    aggregate_spec = compute_complex_spectrogram(aggregate_signal, fs, args.sliding_len_s)
    if aggregate_spec is None:
        raise ValueError("Could not build aggregate sliding FFT")
    s_db = 20.0 * np.log10(np.abs(aggregate_spec.S_complex) + np.finfo(float).eps)
    t_edges = np.empty(aggregate_spec.t.size + 1, dtype=float)
    f_edges = np.empty(aggregate_spec.f.size + 1, dtype=float)
    t_centers = aggregate_spec.t + float(processed_t[0])
    t_step = float(np.median(np.diff(t_centers))) if t_centers.size > 1 else args.sliding_len_s
    f_step = float(np.median(np.diff(aggregate_spec.f))) if aggregate_spec.f.size > 1 else 1.0
    t_edges[1:-1] = 0.5 * (t_centers[:-1] + t_centers[1:])
    t_edges[0] = t_centers[0] - 0.5 * t_step
    t_edges[-1] = t_centers[-1] + 0.5 * t_step
    f_edges[1:-1] = 0.5 * (aggregate_spec.f[:-1] + aggregate_spec.f[1:])
    f_edges[0] = aggregate_spec.f[0] - 0.5 * f_step
    f_edges[-1] = aggregate_spec.f[-1] + 0.5 * f_step

    overview_fig, overview_axes = plt.subplots(5, 1, figsize=(16, 18), constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4 = overview_axes

    ax0.plot(processed_t, aggregate_signal, color="black", linewidth=0.8)
    for hit in hits:
        ax0.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
        ax0.text(hit.time_s, ax0.get_ylim()[1], f"H{hit.index}", fontsize=8, color="crimson", ha="left", va="top")
    ax0.set_title(f"{args.dataset} aggregate signal with phase-coherence hit candidates")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True, alpha=0.25)

    ax1.plot(t_spec, energy_trace, color="navy", linewidth=1.2)
    ax1.axhline(energy_gate, color="darkorange", linestyle="--", linewidth=1.1, label=f"energy gate q={args.energy_gate_quantile:.2f}")
    for hit in hits:
        ax1.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.65)
    ax1.set_title("Broadband energy trace used as a gate")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Energy trace")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(t_spec, coherence, color="black", linewidth=1.1)
    for hit in hits:
        ax2.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.65)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Sliding-FFT phase coherence between adjacent time columns")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Coherence")
    ax2.grid(True, alpha=0.25)

    ax3.plot(t_spec, coherence_loss, color="darkorange", linewidth=1.2)
    ax3.axhline(loss_median, color="0.5", linestyle=":", linewidth=1.0, label="median")
    ax3.axhline(loss_threshold, color="crimson", linestyle="--", linewidth=1.1, label="threshold")
    if hits:
        ax3.scatter([hit.time_s for hit in hits], [hit.coherence_loss for hit in hits], color="crimson", s=24, zorder=3)
    for hit in hits:
        ax3.text(hit.time_s, hit.coherence_loss, f"H{hit.index}", fontsize=8, color="crimson", ha="left", va="bottom")
    ax3.set_title("Coherence loss = 1 - coherence")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Coherence loss")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right", fontsize=8)

    pcm = ax4.pcolormesh(t_edges, f_edges, s_db, shading="flat", cmap="viridis")
    overview_fig.colorbar(pcm, ax=ax4, label="Amplitude (dB)")
    for hit in hits:
        ax4.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0, alpha=0.65)
    ax4.set_ylim(0.0, min(args.energy_max_hz * 1.2, 0.5 * fs))
    ax4.set_title("Aggregate sliding FFT")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")

    n_focus = max(1, len(focus_hits))
    focus_fig, focus_axes = plt.subplots(n_focus, 4, figsize=(20, 4.5 * n_focus), constrained_layout=True)
    if n_focus == 1:
        focus_axes = np.asarray([focus_axes], dtype=object)

    for row_idx, hit in enumerate(focus_hits):
        ax_sig, ax_energy, ax_coh, ax_spec = focus_axes[row_idx]
        local_low = max(float(processed_t[0]), hit.time_s - 15.0)
        local_high = min(float(processed_t[-1]), hit.time_s + 15.0)

        sig_mask = (processed_t >= local_low) & (processed_t <= local_high)
        energy_mask_local = (t_spec >= local_low) & (t_spec <= local_high)

        ax_sig.plot(processed_t[sig_mask], aggregate_signal[sig_mask], color="black", linewidth=0.8)
        ax_sig.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_sig.set_title(f"Hit {hit.index}: local signal")
        ax_sig.set_xlabel("Time (s)")
        ax_sig.set_ylabel("Amplitude")
        ax_sig.grid(True, alpha=0.25)

        ax_energy.plot(t_spec[energy_mask_local], energy_trace[energy_mask_local], color="navy", linewidth=1.1)
        ax_energy.axhline(energy_gate, color="darkorange", linestyle="--", linewidth=1.0)
        ax_energy.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_energy.set_title(f"Hit {hit.index}: energy gate | E={hit.energy_value:.3g}")
        ax_energy.set_xlabel("Time (s)")
        ax_energy.set_ylabel("Energy")
        ax_energy.grid(True, alpha=0.25)

        ax_coh.plot(t_spec[energy_mask_local], coherence[energy_mask_local], color="black", linewidth=1.0, label="coherence")
        ax_coh.plot(t_spec[energy_mask_local], coherence_loss[energy_mask_local], color="darkorange", linewidth=1.0, label="loss")
        ax_coh.axhline(loss_threshold, color="crimson", linestyle="--", linewidth=1.0)
        ax_coh.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_coh.set_title(f"Hit {hit.index}: coherence drop | loss={hit.coherence_loss:.3g}")
        ax_coh.set_xlabel("Time (s)")
        ax_coh.set_ylabel("Coherence / loss")
        ax_coh.grid(True, alpha=0.25)
        ax_coh.legend(fontsize=7, loc="upper right")

        spec_mask_t = (t_centers >= local_low) & (t_centers <= local_high)
        local_t_edges = t_edges[np.where(spec_mask_t)[0][0] : np.where(spec_mask_t)[0][-1] + 2] if np.any(spec_mask_t) else t_edges
        local_s_db = s_db[:, spec_mask_t] if np.any(spec_mask_t) else s_db
        ax_spec.pcolormesh(local_t_edges, f_edges, local_s_db, shading="flat", cmap="viridis")
        ax_spec.axvline(hit.time_s, color="crimson", linestyle="--", linewidth=1.0)
        ax_spec.set_ylim(0.0, min(args.energy_max_hz * 1.2, 0.5 * fs))
        ax_spec.set_title(f"Hit {hit.index}: local spectrogram")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")

    print(f"Dataset: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Bond spacing mode: {args.bond_spacing_mode}")
    print(f"Frame-time repaired: {'yes' if repaired_flag else 'no'}")
    print(f"Processed bonds: {len(processed_signals)}")
    print(f"Sliding window: {args.sliding_len_s:.6g} s")
    print(f"Coherence band: [{args.coherence_min_hz:.3g}, {args.coherence_max_hz:.3g}] Hz")
    print(f"Energy band: [{args.energy_min_hz:.3g}, {args.energy_max_hz:.3g}] Hz")
    print(f"Detected coherence-loss hits: {len(hits)}")
    print(f"Coherence-loss threshold: median={loss_median:.6g} std={loss_std:.6g} threshold={loss_threshold:.6g}")
    print(f"Energy gate: {energy_gate:.6g}")
    print(f"Focus hits: {[hit.index for hit in focus_hits]}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.component}__phase_coherence_hits"
        overview_path = save_dir / f"{stem}__overview.png"
        focus_path = save_dir / f"{stem}__focus.png"
        csv_path = save_dir / f"{stem}.csv"
        overview_fig.savefig(overview_path, dpi=160)
        focus_fig.savefig(focus_path, dpi=160)
        _save_hits_csv(csv_path, args.dataset, args.component, hits)
        print(f"Saved overview figure: {overview_path}")
        print(f"Saved focus figure: {focus_path}")
        print(f"Saved hits CSV: {csv_path}")

    if args.no_show:
        plt.close(overview_fig)
        plt.close(focus_fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
