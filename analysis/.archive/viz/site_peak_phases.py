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

from plotting.common import centers_to_edges, render_figure
from tools.cli import add_output_args, add_signal_processing_args, add_track_data_root_arg
from tools.peaks import load_peaks_csv
from tools.selection import build_configured_site_signals, load_dataset_selection
from tools.site_phase import estimate_site_peak_phases, write_site_peak_phase_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate site-relative phases at selected peaks for one configured dataset.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing strictly increasing peak frequencies.")
    add_track_data_root_arg(parser)
    add_signal_processing_args(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path. Defaults to configs/phases/<config-stem>__<peaks-stem>.csv",
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
        default=100.0,
        help="Segment length in seconds for the complex Welch-style phase estimate. Default: 100",
    )
    parser.add_argument(
        "--welch-overlap",
        type=float,
        default=0.5,
        help="Overlap fraction for the complex Welch-style segmentation. Default: 0.5",
    )
    return parser


def default_csv_path(config_json: str | Path, peaks_csv: str | Path) -> Path:
    config_stem = Path(config_json).stem
    peaks_stem = Path(peaks_csv).stem
    return Path("configs/phases") / f"{config_stem}__{peaks_stem}.csv"


def plot_site_peak_phases(result, *, title: str | None = None):
    peaks = np.asarray(result.peaks_hz, dtype=float)
    site_ids = np.asarray(result.site_ids, dtype=int)
    phase = np.asarray(result.relative_phase_rad, dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8.0, 1.4 * len(peaks)), 8.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    x_edges = centers_to_edges(peaks, fallback_step=1.0)
    y_edges = centers_to_edges(site_ids.astype(float), fallback_step=1.0)

    mesh = axes[0].pcolormesh(
        x_edges,
        y_edges,
        phase,
        cmap="twilight",
        shading="auto",
        vmin=-np.pi,
        vmax=np.pi,
    )
    cbar = fig.colorbar(mesh, ax=axes[0], pad=0.01)
    cbar.set_label("Relative Phase (rad)")
    axes[0].set_ylabel("Site ID")
    axes[0].set_xlabel("Peak Frequency (Hz)")
    axes[0].set_title(
        title
        or f"Site Relative Phase | Dataset: {result.dataset_name} | Ref site: {result.reference_site_id}"
    )

    for row_idx, site_id in enumerate(site_ids):
        axes[1].plot(peaks, phase[row_idx, :], marker="o", linewidth=1.0, label=f"Site {int(site_id)}")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].axhline(np.pi, color="black", linewidth=0.8, alpha=0.25, linestyle="--")
    axes[1].axhline(-np.pi, color="black", linewidth=0.8, alpha=0.25, linestyle="--")
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_ylabel("Relative Phase (rad)")
    axes[1].set_xlabel("Peak Frequency (Hz)")
    axes[1].grid(True, alpha=0.3)
    if len(site_ids) <= 12:
        axes[1].legend(ncols=min(4, len(site_ids)))

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

    try:
        peaks = load_peaks_csv(args.peaks_csv)
        config = load_dataset_selection(args.config_json)
        records = build_configured_site_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=False,
        )
        result = estimate_site_peak_phases(
            records,
            peaks,
            welch_len_s=args.welch_len_s,
            welch_overlap_fraction=args.welch_overlap,
            search_width_hz=args.search_width_hz,
            longest=args.longest,
            handlenan=args.handlenan,
        )

        output_csv = Path(args.csv) if args.csv is not None else default_csv_path(args.config_json, args.peaks_csv)
        saved_csv = write_site_peak_phase_csv(result, output_csv)

        print(f"Dataset: {result.dataset_name}")
        print(f"Reference site id: {result.reference_site_id}")
        print(f"Sites: {result.site_ids.tolist()}")
        print(f"Peaks (Hz): {result.peaks_hz.tolist()}")
        print(f"Selected frequencies (Hz): {result.selected_freq_hz.tolist()}")
        print(f"Windows used per peak: {result.n_windows_used.tolist()}")
        print(f"Coherence fraction per peak: {result.coherence_fraction.tolist()}")
        print(f"CSV saved to: {saved_csv}")

        fig = plot_site_peak_phases(result, title=args.title)
        render_figure(fig, save=args.save)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
