#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.common import render_figure
from tools.bond_phase import estimate_bond_peak_phases, write_bond_peak_phase_csv
from tools.cli import add_output_args, add_signal_processing_args, add_track_data_root_arg
from tools.peaks import load_peaks_csv
from tools.selection import build_configured_bond_signals, load_dataset_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary local-phase estimator on bond signals: use short-time complex spectra, "
            "gauge-fix each window from a reference bond and peak, then plot average phase per bond for each peak."
        ),
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing strictly increasing peak frequencies.")
    add_track_data_root_arg(parser)
    add_signal_processing_args(parser)
    add_output_args(parser, include_title=True)
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path for the per-bond phase table.",
    )
    parser.add_argument(
        "--search-width-hz",
        type=float,
        default=0.25,
        help="Half-width of the frequency search window around each target peak. Default: 0.25",
    )
    parser.add_argument(
        "--welch-len-s",
        type=float,
        default=20.0,
        help="Segment length in seconds for the complex short-time spectrum. Default: 20",
    )
    parser.add_argument(
        "--welch-overlap",
        type=float,
        default=0.5,
        help="Overlap fraction for the short-time segmentation. Default: 0.5",
    )
    parser.add_argument(
        "--min-reference-fraction",
        type=float,
        default=0.05,
        help=(
            "Reject windows where the reference bond amplitude at the reference peak is below "
            "this fraction of that bond/peak's median window amplitude. Default: 0.05"
        ),
    )
    parser.add_argument(
        "--reference-bond",
        type=int,
        default=5,
        help="Bond id used to define the per-window gauge rotation. Default: 5",
    )
    parser.add_argument(
        "--reference-peak-index",
        type=int,
        default=1,
        help="1-based peak index used as the gauge reference. Default: 1",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help=(
            "Display wrapped phases in [-pi, pi] as points without a connecting line. "
            "Default behavior is to show an unwrapped guide line across bonds."
        ),
    )
    return parser


def _configured_bond_ids(config) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for _, selection in config.items():
        if not selection.include:
            continue
        for bond_id in selection.pair_ids:
            bond_id = int(bond_id)
            if bond_id not in seen:
                seen.add(bond_id)
                out.append(bond_id)
    return out


def default_csv_path(config_json: str | Path, peaks_csv: str | Path) -> Path:
    config_stem = Path(config_json).stem
    peaks_stem = Path(peaks_csv).stem
    return Path("configs/phases_local") / f"{config_stem}__{peaks_stem}.csv"


def plot_phase_per_bond_by_peak(result, *, title: str | None = None, wrap: bool = False):
    peaks = np.asarray(result.peaks_hz, dtype=float)
    bond_ids = np.asarray(result.bond_ids, dtype=int)
    phase = np.asarray(result.relative_phase_rad, dtype=float)
    coherence = np.asarray(result.bond_coherence, dtype=float)

    n_peaks = len(peaks)
    ncols = min(3, max(1, n_peaks))
    nrows = int(np.ceil(n_peaks / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.7 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for peak_idx, peak_hz in enumerate(peaks):
        ax = axes[peak_idx // ncols][peak_idx % ncols]
        y = phase[:, peak_idx]
        c = coherence[:, peak_idx]
        valid = np.isfinite(y)
        scatter = ax.scatter(
            bond_ids[valid],
            y[valid],
            c=c[valid],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=45,
            zorder=3,
        )
        if wrap:
            ax.set_ylim(-np.pi, np.pi)
        else:
            if np.any(valid):
                y_unwrapped = np.unwrap(y[valid])
                ax.plot(bond_ids[valid], y_unwrapped, color="black", linewidth=1.2, alpha=0.7)
                y_min = float(np.min(y_unwrapped))
                y_max = float(np.max(y_unwrapped))
                pad = max(0.2, 0.08 * max(1e-12, y_max - y_min))
                ax.set_ylim(y_min - pad, y_max + pad)
            else:
                ax.set_ylim(-np.pi, np.pi)
        ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle=":")
        ax.axhline(np.pi, color="0.75", linewidth=0.8, linestyle="--")
        ax.axhline(-np.pi, color="0.75", linewidth=0.8, linestyle="--")
        ax.set_xticks(bond_ids)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Bond ID")
        ax.set_ylabel("Phase (rad)")
        ax.set_title(
            f"Peak {peak_hz:.3f} Hz\n"
            f"mean f={result.mean_selected_freq_hz[peak_idx]:.3f}, "
            f"n={int(result.n_windows_used[peak_idx])}"
        )
        fig.colorbar(scatter, ax=ax, pad=0.01, label="Coherence")

    for unused_idx in range(n_peaks, nrows * ncols):
        axes[unused_idx // ncols][unused_idx % ncols].set_visible(False)

    fig.suptitle(
        title
        or (
            f"Local Gauge Bond Phase | Dataset: {result.dataset_name} | "
            f"Ref bond: {result.reference_bond_id} @ {result.reference_peak_hz:.3f} Hz"
        ),
        fontsize=14,
    )
    return fig


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.welch_len_s <= 0:
        print("Error: --welch-len-s must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= args.welch_overlap < 1.0):
        print("Error: --welch-overlap must satisfy 0 <= value < 1", file=sys.stderr)
        return 1
    if args.search_width_hz <= 0:
        print("Error: --search-width-hz must be > 0", file=sys.stderr)
        return 1
    if args.min_reference_fraction < 0:
        print("Error: --min-reference-fraction must be >= 0", file=sys.stderr)
        return 1
    if args.reference_bond < 1:
        print("Error: --reference-bond must be >= 1", file=sys.stderr)
        return 1
    if args.reference_peak_index < 1:
        print("Error: --reference-peak-index must be >= 1", file=sys.stderr)
        return 1

    try:
        peaks = load_peaks_csv(args.peaks_csv)
        config = load_dataset_selection(args.config_json)
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=False,
        )
        configured_bond_ids = _configured_bond_ids(config)
        if args.reference_bond not in configured_bond_ids:
            raise ValueError(
                f"Reference bond {args.reference_bond} is not in the configured enabled bonds: {configured_bond_ids}"
            )

        result = estimate_bond_peak_phases(
            records,
            peaks,
            reference_bond_id=args.reference_bond,
            reference_peak_index=args.reference_peak_index,
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            search_width_hz=args.search_width_hz,
            min_reference_fraction=args.min_reference_fraction,
            longest=args.longest,
            handlenan=args.handlenan,
        )

        output_csv = Path(args.csv) if args.csv is not None else default_csv_path(args.config_json, args.peaks_csv)
        saved_csv = write_bond_peak_phase_csv(result, output_csv)

        print(f"Dataset: {result.dataset_name}")
        print(f"Reference bond id: {result.reference_bond_id}")
        print(f"Reference peak (Hz): {result.reference_peak_hz}")
        print(f"Bonds: {result.bond_ids.tolist()}")
        print(f"Peaks (Hz): {result.peaks_hz.tolist()}")
        print(f"Mean selected frequencies (Hz): {result.mean_selected_freq_hz.tolist()}")
        print(f"Windows used per peak: {result.n_windows_used.tolist()}")
        print(f"CSV saved to: {saved_csv}")

        fig = plot_phase_per_bond_by_peak(result, title=args.title, wrap=args.wrap)
        render_figure(fig, save=args.save)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
