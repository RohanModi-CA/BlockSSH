#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from play1 import CONFIG, OUTPUT_DIR
from analysis.Nrm.Tools.post_hit_regions import EnabledRegionConfig, extract_post_hit_regions
from analysis.tools.signal import hann_window_periodic, next_power_of_two, preprocess_signal


@dataclass(frozen=True)
class SegmentCollection:
    freqs: np.ndarray
    spectra: np.ndarray
    mean_amplitude: np.ndarray
    segment_labels: list[str]
    nperseg: int
    noverlap: int
    excluded_fraction: float


@dataclass(frozen=True)
class BicoherenceResult:
    label: str
    f1_target: float
    f2_target: float
    f1_selected: float
    f2_selected: float
    f3_selected: float
    b_sq: float
    phase_lock: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Welch-style bicoherence on full traces using equal-length overlapping complex FFT segments.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the final figure with matplotlib.",
    )
    parser.add_argument(
        "--segment-len-s",
        type=float,
        default=20.0,
        help="Segment length in seconds. Default: 20.0",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Segment overlap fraction in [0, 1). Default: 0.5",
    )
    parser.add_argument(
        "--mask-hit-window-s",
        type=float,
        default=6.0,
        help="Half-width in seconds masked around each detected/manual hit time. Default: 6.0",
    )
    parser.add_argument(
        "--no-mask-hits",
        action="store_true",
        help="Disable hit masking and use all segments from the full traces.",
    )
    parser.add_argument(
        "--bond-spacing-mode",
        choices=("default", "comoving"),
        default="default",
        help="Bond signal representation to analyze. Default: default",
    )
    parser.add_argument(
        "--search-bins",
        type=int,
        default=0,
        help="Search radius in bins around each target pair while preserving exact sum-bin closure. Default: 0",
    )
    parser.add_argument(
        "--snap-bins",
        type=int,
        default=0,
        help="Snap each target frequency to the strongest local mean-amplitude bin within this radius before bicoherence. Default: 0",
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


def _build_hit_mask(times: np.ndarray, peak_times_s: np.ndarray, half_width_s: float) -> np.ndarray:
    mask = np.ones(times.shape, dtype=bool)
    if half_width_s <= 0 or peak_times_s.size == 0:
        return mask
    for hit_time in peak_times_s:
        mask &= np.abs(times - float(hit_time)) > half_width_s
    return mask


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    bounded = np.concatenate(([False], mask, [False]))
    changes = np.diff(bounded.astype(int))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def collect_segment_spectra(cfg, args: argparse.Namespace) -> SegmentCollection:
    all_spectra: list[np.ndarray] = []
    segment_labels: list[str] = []
    excluded_samples = 0
    total_samples = 0
    nperseg: int | None = None
    noverlap: int | None = None
    freqs: np.ndarray | None = None

    for bond_id in cfg.bond_ids:
        region_config = EnabledRegionConfig(bond_spacing_mode=str(args.bond_spacing_mode))
        result = extract_post_hit_regions(
            dataset=cfg.dataset,
            component=cfg.component,
            bond_id=bond_id,
            config=region_config,
        )
        processed, err = preprocess_signal(result.frame_times_s, result.signal, longest=False, handlenan=False)
        if processed is None:
            raise ValueError(f"Failed to preprocess full trace for bond {bond_id}: {err}")

        local_nperseg = max(8, int(round(float(args.segment_len_s) * processed.Fs)))
        local_nperseg = min(local_nperseg, processed.y.size)
        if local_nperseg < 8:
            raise ValueError(f"Segment length is too short for bond {bond_id}")
        local_noverlap = min(int(round(float(args.overlap) * local_nperseg)), local_nperseg - 1)
        local_step = local_nperseg - local_noverlap
        if local_step <= 0:
            raise ValueError("Segment overlap leaves no forward step; use overlap < 1")

        if nperseg is None:
            nperseg = local_nperseg
            noverlap = local_noverlap
            nfft = max(nperseg, next_power_of_two(nperseg))
            freqs = np.fft.rfftfreq(nfft, d=processed.dt)
            window = hann_window_periodic(nperseg)
            window_norm = float(np.sum(window))
        else:
            if local_nperseg != nperseg or local_noverlap != noverlap:
                raise ValueError("Inconsistent segment parameters across bonds")

        if args.no_mask_hits:
            usable_mask = np.ones(processed.t.shape, dtype=bool)
        else:
            usable_mask = _build_hit_mask(
                processed.t,
                result.peak_times_s,
                half_width_s=float(args.mask_hit_window_s),
            )

        total_samples += int(usable_mask.size)
        excluded_samples += int((~usable_mask).sum())

        nfft = max(nperseg, next_power_of_two(nperseg))
        window = hann_window_periodic(nperseg)
        window_norm = float(np.sum(window))

        for run_start, run_stop in _true_runs(usable_mask):
            if run_stop - run_start < nperseg:
                continue
            for start in range(run_start, run_stop - nperseg + 1, local_step):
                stop = start + nperseg
                segment = processed.y[start:stop] * window
                spectrum = np.fft.rfft(segment, n=nfft) / window_norm
                if spectrum.size > 2:
                    spectrum = spectrum.copy()
                    spectrum[1:-1] *= 2.0
                all_spectra.append(spectrum)
                segment_labels.append(f"bond{bond_id}:{start}:{stop}")

    if not all_spectra or freqs is None or nperseg is None or noverlap is None:
        raise ValueError("No valid segments were collected. Reduce masking or segment length.")

    excluded_fraction = float(excluded_samples) / float(total_samples) if total_samples > 0 else 0.0
    return SegmentCollection(
        freqs=np.asarray(freqs, dtype=float),
        spectra=np.vstack(all_spectra),
        mean_amplitude=np.mean(np.abs(np.vstack(all_spectra)), axis=0),
        segment_labels=segment_labels,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        excluded_fraction=excluded_fraction,
    )


def _bicoherence_from_indices(i: int, j: int, k: int, X: np.ndarray) -> tuple[float, float]:
    X1 = X[:, i]
    X2 = X[:, j]
    X3 = X[:, k]
    bispec_terms = X1 * X2 * np.conj(X3)
    bispec = np.mean(bispec_terms)
    denom = np.mean(np.abs(X1 * X2) ** 2) * np.mean(np.abs(X3) ** 2)
    b_sq = float((np.abs(bispec) ** 2) / denom) if denom > 1e-18 else 0.0
    phase_lock = float(np.abs(np.mean(np.exp(1j * np.angle(bispec_terms)))))
    return b_sq, phase_lock


def _snap_index_to_local_peak(index: int, amplitude: np.ndarray, radius: int) -> int:
    if radius <= 0:
        return int(index)
    start = max(0, int(index) - int(radius))
    stop = min(amplitude.size, int(index) + int(radius) + 1)
    local = amplitude[start:stop]
    if local.size == 0:
        return int(index)
    return int(start + np.argmax(local))


def estimate_bicoherence(
    f1_target: float,
    f2_target: float,
    freqs: np.ndarray,
    X: np.ndarray,
    *,
    mean_amplitude: np.ndarray | None = None,
    snap_bins: int = 0,
    search_bins: int = 0,
) -> BicoherenceResult:
    df = float(freqs[1] - freqs[0])
    i_center = int(round(f1_target / df))
    j_center = int(round(f2_target / df))
    if mean_amplitude is not None and snap_bins > 0:
        i_center = _snap_index_to_local_peak(i_center, mean_amplitude, snap_bins)
        j_center = _snap_index_to_local_peak(j_center, mean_amplitude, snap_bins)
    best: BicoherenceResult | None = None

    for i in range(max(0, i_center - search_bins), min(X.shape[1], i_center + search_bins + 1)):
        for j in range(max(0, j_center - search_bins), min(X.shape[1], j_center + search_bins + 1)):
            k = i + j
            if k >= X.shape[1]:
                continue
            b_sq, phase_lock = _bicoherence_from_indices(i, j, k, X)
            candidate = BicoherenceResult(
                label="",
                f1_target=float(f1_target),
                f2_target=float(f2_target),
                f1_selected=float(freqs[i]),
                f2_selected=float(freqs[j]),
                f3_selected=float(freqs[k]),
                b_sq=b_sq,
                phase_lock=phase_lock,
            )
            if best is None or candidate.b_sq > best.b_sq:
                best = candidate

    if best is None:
        raise ValueError("Could not evaluate any valid frequency triplets")
    return best


def main() -> int:
    args = build_parser().parse_args()
    if args.segment_len_s <= 0:
        raise ValueError("--segment-len-s must be > 0")
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("--overlap must satisfy 0 <= value < 1")
    if args.mask_hit_window_s < 0:
        raise ValueError("--mask-hit-window-s must be >= 0")
    if args.search_bins < 0:
        raise ValueError("--search-bins must be >= 0")
    if args.snap_bins < 0:
        raise ValueError("--snap-bins must be >= 0")

    configure_matplotlib(args.show)

    print(f"--- Play 5: Welch-Style Bicoherence on Full Traces ({CONFIG.dataset}) ---")
    collection = collect_segment_spectra(CONFIG, args)
    freqs = collection.freqs
    X = collection.spectra
    n_segments = X.shape[0]
    df = float(freqs[1] - freqs[0])
    noise_floor = 1.0 / n_segments

    print(
        f"Segments: {n_segments} | df={df:.4f} Hz | nperseg={collection.nperseg} "
        f"| overlap={collection.noverlap / collection.nperseg:.2f} | mode={args.bond_spacing_mode}"
    )
    if args.no_mask_hits:
        print("Hit masking: disabled")
    else:
        print(
            f"Hit masking: +/- {args.mask_hit_window_s:.2f} s around peak times "
            f"| excluded samples ~ {100.0 * collection.excluded_fraction:.1f}%"
        )

    triplets = [
        ("Main Nonlinear Sum", 6.34, 12.053),
        ("Possible 2*8.97 Peak", 8.97, 8.97),
        ("Pendulum Sideband", 18.393, 0.41),
        ("Negative Control", 3.74, 14.653),
        ("Second Harmonic", 6.34, 6.34),
    ]

    results: list[BicoherenceResult] = []
    print(
        f"{'Relationship':<22} | {'f1_sel':<7} | {'f2_sel':<7} | {'f3_sel':<7} | "
        f"{'Bicoherence^2':<15} | {'Phase Lock':<10}"
    )
    print("-" * 96)
    for label, f1, f2 in triplets:
        result = estimate_bicoherence(
            f1,
            f2,
            freqs,
            X,
            mean_amplitude=collection.mean_amplitude,
            snap_bins=int(args.snap_bins),
            search_bins=int(args.search_bins),
        )
        results.append(
            BicoherenceResult(
                label=label,
                f1_target=result.f1_target,
                f2_target=result.f2_target,
                f1_selected=result.f1_selected,
                f2_selected=result.f2_selected,
                f3_selected=result.f3_selected,
                b_sq=result.b_sq,
                phase_lock=result.phase_lock,
            )
        )
        print(
            f"{label:<22} | {result.f1_selected:<7.3f} | {result.f2_selected:<7.3f} | {result.f3_selected:<7.3f} | "
            f"{result.b_sq:<15.4f} | {result.phase_lock:<10.4f}"
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r.label for r in results]
    values = [r.b_sq for r in results]
    colors = ["#2ca02c" if value > 3 * noise_floor else "#7f7f7f" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(noise_floor, color="red", linestyle="--", label=f"Noise floor (1/N={noise_floor:.3f})")
    ax.set_ylabel("Squared Bicoherence")
    ax.set_title(f"Welch-Style Bicoherence | {CONFIG.dataset} | {n_segments} segments")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    output_path = OUTPUT_DIR / "play5_bicoherence_results.png"
    fig.savefig(output_path)
    print(f"\nResults saved to {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
