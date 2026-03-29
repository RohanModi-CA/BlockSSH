#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from play1 import CONFIG, OUTPUT_DIR
from analysis.Nrm.Tools.post_hit_regions import extract_post_hit_regions
from analysis.tools.signal import hann_window_periodic, next_power_of_two, preprocess_signal


@dataclass(frozen=True)
class SegmentRecord:
    bond_id: int
    start_s: float
    stop_s: float
    mid_s: float
    spectrum: np.ndarray


@dataclass(frozen=True)
class TripletSpec:
    label: str
    f1: float
    f2: float


@dataclass(frozen=True)
class TripletBins:
    label: str
    i1: int
    i2: int
    i3: int
    f1_sel: float
    f2_sel: float
    f3_sel: float


@dataclass(frozen=True)
class SubsetResult:
    mode: str
    score: float
    segment_count: int
    start_s: float
    stop_s: float
    segment_indices: np.ndarray
    metrics: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Welch-style segment subsets for localized bicoherence in the verified main triad.",
    )
    parser.add_argument("--show", action="store_true", help="Show the final figure.")
    parser.add_argument("--segment-len-s", type=float, default=100.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--snap-bins",
        type=int,
        default=3,
        help="Snap target frequencies to local mean-amplitude peaks within this many bins. Default: 3",
    )
    parser.add_argument(
        "--min-contiguous-segments",
        type=int,
        default=4,
        help="Minimum segment count for contiguous interval search. Default: 4",
    )
    parser.add_argument(
        "--min-topk-segments",
        type=int,
        default=4,
        help="Minimum segment count for top-K amplitude subset search. Default: 4",
    )
    parser.add_argument(
        "--max-topk-segments",
        type=int,
        default=12,
        help="Maximum segment count for top-K amplitude subset search. Default: 12",
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


def collect_segments(segment_len_s: float, overlap: float) -> tuple[np.ndarray, list[SegmentRecord], np.ndarray]:
    records: list[SegmentRecord] = []
    nperseg: int | None = None
    freqs: np.ndarray | None = None

    for bond_id in CONFIG.bond_ids:
        result = extract_post_hit_regions(dataset=CONFIG.dataset, component=CONFIG.component, bond_id=bond_id)
        processed, err = preprocess_signal(result.frame_times_s, result.signal, longest=False, handlenan=False)
        if processed is None:
            raise ValueError(f"Failed preprocessing bond {bond_id}: {err}")

        local_nperseg = max(8, int(round(segment_len_s * processed.Fs)))
        if local_nperseg > processed.y.size:
            continue
        step = max(1, local_nperseg - int(round(overlap * local_nperseg)))
        local_nfft = max(local_nperseg, next_power_of_two(local_nperseg))
        window = hann_window_periodic(local_nperseg)
        window_norm = float(np.sum(window))

        if nperseg is None:
            nperseg = local_nperseg
            freqs = np.fft.rfftfreq(local_nfft, d=processed.dt)
        elif local_nperseg != nperseg:
            raise ValueError("Segment length resolved differently across bonds")

        for start in range(0, processed.y.size - local_nperseg + 1, step):
            stop = start + local_nperseg
            segment = processed.y[start:stop] * window
            spectrum = np.fft.rfft(segment, n=local_nfft) / window_norm
            if spectrum.size > 2:
                spectrum = spectrum.copy()
                spectrum[1:-1] *= 2.0
            records.append(
                SegmentRecord(
                    bond_id=bond_id,
                    start_s=float(processed.t[start]),
                    stop_s=float(processed.t[stop - 1]),
                    mid_s=float(np.mean(processed.t[start:stop])),
                    spectrum=spectrum,
                )
            )

    if not records or freqs is None:
        raise ValueError("No valid segments collected")

    records.sort(key=lambda record: (record.mid_s, record.bond_id))
    X = np.vstack([record.spectrum for record in records])
    mean_amplitude = np.mean(np.abs(X), axis=0)
    return freqs, records, mean_amplitude


def _snap_index(index: int, amplitude: np.ndarray, radius: int) -> int:
    if radius <= 0:
        return int(index)
    lo = max(0, int(index) - int(radius))
    hi = min(amplitude.size, int(index) + int(radius) + 1)
    return int(lo + np.argmax(amplitude[lo:hi]))


def resolve_triplet_bins(
    freqs: np.ndarray,
    mean_amplitude: np.ndarray,
    triplets: list[TripletSpec],
    snap_bins: int,
) -> dict[str, TripletBins]:
    df = float(freqs[1] - freqs[0])
    out: dict[str, TripletBins] = {}
    for spec in triplets:
        i1 = int(round(spec.f1 / df))
        i2 = int(round(spec.f2 / df))
        i1 = _snap_index(i1, mean_amplitude, snap_bins)
        i2 = _snap_index(i2, mean_amplitude, snap_bins)
        i3 = i1 + i2
        if i3 >= freqs.size:
            raise ValueError(f"Triplet {spec.label} exceeded frequency grid")
        out[spec.label] = TripletBins(
            label=spec.label,
            i1=i1,
            i2=i2,
            i3=i3,
            f1_sel=float(freqs[i1]),
            f2_sel=float(freqs[i2]),
            f3_sel=float(freqs[i3]),
        )
    return out


def bicoherence_for_subset(X: np.ndarray, bins: TripletBins) -> float:
    X1 = X[:, bins.i1]
    X2 = X[:, bins.i2]
    X3 = X[:, bins.i3]
    z = X1 * X2 * np.conj(X3)
    denom = np.mean(np.abs(X1 * X2) ** 2) * np.mean(np.abs(X3) ** 2)
    return 0.0 if denom <= 1e-18 else float(np.abs(np.mean(z)) ** 2 / denom)


def segment_score(X: np.ndarray, bins: TripletBins) -> np.ndarray:
    return np.abs(X[:, bins.i1]) * np.abs(X[:, bins.i2]) * np.abs(X[:, bins.i3])


def evaluate_subset(X: np.ndarray, bins_by_label: dict[str, TripletBins]) -> dict[str, float]:
    return {label: bicoherence_for_subset(X, bins) for label, bins in bins_by_label.items()}


def subset_priority(metrics: dict[str, float]) -> float:
    return float(metrics["Main Nonlinear Sum"] - max(
        metrics["Possible 2*8.97 Peak"],
        metrics["Negative Control"],
        metrics["Pendulum Sideband"],
        metrics["Second Harmonic"],
    ))


def search_best_contiguous(
    records: list[SegmentRecord],
    X: np.ndarray,
    bins_by_label: dict[str, TripletBins],
    min_segments: int,
) -> SubsetResult:
    best: SubsetResult | None = None
    for start in range(len(records)):
        for stop in range(start + min_segments - 1, len(records)):
            subset_idx = np.arange(start, stop + 1, dtype=int)
            metrics = evaluate_subset(X[subset_idx], bins_by_label)
            result = SubsetResult(
                mode="best_contiguous",
                score=subset_priority(metrics),
                segment_count=int(subset_idx.size),
                start_s=float(records[start].mid_s),
                stop_s=float(records[stop].mid_s),
                segment_indices=subset_idx,
                metrics=metrics,
            )
            if best is None or result.score > best.score:
                best = result
    if best is None:
        raise ValueError("No contiguous subset found")
    return best


def search_best_topk(
    records: list[SegmentRecord],
    X: np.ndarray,
    bins_by_label: dict[str, TripletBins],
    min_segments: int,
    max_segments: int,
) -> SubsetResult:
    main_bins = bins_by_label["Main Nonlinear Sum"]
    ranking = np.argsort(segment_score(X, main_bins))[::-1]
    best: SubsetResult | None = None
    upper = min(max_segments, ranking.size)
    for k in range(min_segments, upper + 1):
        subset_idx = np.sort(ranking[:k])
        metrics = evaluate_subset(X[subset_idx], bins_by_label)
        result = SubsetResult(
            mode="best_topk_main_amplitude",
            score=subset_priority(metrics),
            segment_count=int(subset_idx.size),
            start_s=float(min(records[i].mid_s for i in subset_idx)),
            stop_s=float(max(records[i].mid_s for i in subset_idx)),
            segment_indices=subset_idx,
            metrics=metrics,
        )
        if best is None or result.score > best.score:
            best = result
    if best is None:
        raise ValueError("No top-k subset found")
    return best


def format_segments(records: list[SegmentRecord], indices: np.ndarray) -> str:
    parts = []
    for idx in indices:
        record = records[int(idx)]
        parts.append(f"{record.mid_s:.1f}s[b{record.bond_id}]")
    return ", ".join(parts)


def main() -> int:
    args = build_parser().parse_args()
    if args.segment_len_s <= 0:
        raise ValueError("--segment-len-s must be > 0")
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1)")
    if args.snap_bins < 0:
        raise ValueError("--snap-bins must be >= 0")

    configure_matplotlib(args.show)

    triplets = [
        TripletSpec("Main Nonlinear Sum", 6.34, 12.053),
        TripletSpec("Possible 2*8.97 Peak", 8.97, 8.97),
        TripletSpec("Pendulum Sideband", 18.393, 0.41),
        TripletSpec("Negative Control", 3.74, 14.653),
        TripletSpec("Second Harmonic", 6.34, 6.34),
    ]

    freqs, records, mean_amplitude = collect_segments(args.segment_len_s, args.overlap)
    X = np.vstack([record.spectrum for record in records])
    bins_by_label = resolve_triplet_bins(freqs, mean_amplitude, triplets, args.snap_bins)
    overall_metrics = evaluate_subset(X, bins_by_label)
    contiguous = search_best_contiguous(records, X, bins_by_label, args.min_contiguous_segments)
    topk = search_best_topk(records, X, bins_by_label, args.min_topk_segments, args.max_topk_segments)

    print(f"--- Play 6: Localized Bicoherence Search ({CONFIG.dataset}) ---")
    print(
        f"Segments: {len(records)} | segment_len={args.segment_len_s:.1f}s | "
        f"df={freqs[1]-freqs[0]:.4f} Hz | snap_bins={args.snap_bins}"
    )
    print("Resolved bins:")
    for label, bins in bins_by_label.items():
        print(f"  {label:<20} {bins.f1_sel:.3f} + {bins.f2_sel:.3f} -> {bins.f3_sel:.3f}")

    print("\nWhole-record bicoherence:")
    for label, value in overall_metrics.items():
        print(f"  {label:<20} {value:.4f}")

    for result in (contiguous, topk):
        print(f"\n{result.mode}:")
        print(
            f"  score={result.score:.4f} | n={result.segment_count} | "
            f"time={result.start_s:.1f}s to {result.stop_s:.1f}s"
        )
        for label, value in result.metrics.items():
            print(f"  {label:<20} {value:.4f}")
        print(f"  segments: {format_segments(records, result.segment_indices)}")

    main_bins = bins_by_label["Main Nonlinear Sum"]
    main_z = X[:, main_bins.i1] * X[:, main_bins.i2] * np.conj(X[:, main_bins.i3])
    main_score = np.abs(main_z)
    mids = np.array([record.mid_s for record in records], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(mids, main_score, c=[record.bond_id for record in records], cmap="tab10", s=34)
    ax.set_ylabel("|Main Triad Product|")
    ax.set_title(f"Main Triad Segment Strength | {CONFIG.dataset}")
    ax.axvspan(contiguous.start_s, contiguous.stop_s, color="tab:green", alpha=0.16, label="Best contiguous")
    ax.axvspan(topk.start_s, topk.stop_s, color="tab:orange", alpha=0.10, label="Best top-k span")
    ax.legend()

    ax = axes[1]
    labels = ["Whole", "Contiguous", "Top-K"]
    data = np.array(
        [
            [overall_metrics[t.label] for t in triplets],
            [contiguous.metrics[t.label] for t in triplets],
            [topk.metrics[t.label] for t in triplets],
        ],
        dtype=float,
    )
    x = np.arange(len(labels), dtype=float)
    width = 0.18
    for idx, triplet in enumerate(triplets):
        ax.bar(x + (idx - 1.5) * width, data[:, idx], width=width, label=triplet.label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Squared Bicoherence")
    ax.set_title("Whole Record vs Localized Subsets")
    ax.legend()

    output_path = OUTPUT_DIR / "play6_localized_bicoherence.png"
    fig.savefig(output_path)
    print(f"\nResults saved to {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
