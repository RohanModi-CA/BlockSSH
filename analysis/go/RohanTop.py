#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.plotting.common import render_figure
from analysis.tools.groups import load_group_datasets
from analysis.tools.rohan_top import (
    DEFAULT_TARGET_FREQUENCIES,
    RohanTopResult,
    analyze_rohan_top,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze bond-frequency response and chirality for x, y, or a component data.",
    )
    parser.add_argument("dataset", nargs="?", help="Dataset stem, e.g. 11topo.")
    parser.add_argument("--group", default=None, help="Optional saved group name from MakeGroup.py.")
    parser.add_argument("--groups-dir", default=None, help="Directory containing group JSON files.")
    parser.add_argument(
        "--component",
        required=True,
        choices=["x", "y", "a"],
        help="Component to analyze.",
    )
    parser.add_argument("--track-data-root", default=None, help="Root directory containing track datasets.")
    parser.add_argument("--tmin", type=float, default=5.0, help="Lower time bound for the hit window.")
    parser.add_argument("--tmax", type=float, default=47.0, help="Upper time bound for the hit window.")
    parser.add_argument("--fs", type=float, default=120.0, help="Sampling rate in Hz.")
    parser.add_argument("--win", type=int, default=400, help="FFT window length in samples.")
    parser.add_argument("--av", type=int, default=2000, help="Number of overlapping windows to average.")
    parser.add_argument("--offset", type=int, default=0, help="Window start offset in samples.")
    parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=None,
        help="Target frequencies in Hz. Default: the script's ten reference frequencies.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1000,
        help="Moving-average window for the chirality trace.",
    )
    parser.add_argument("--save", default=None, help="Optional figure path. Group mode uses it as a prefix.")
    parser.add_argument("--title", default=None, help="Optional title override.")
    return parser


def _resolve_save_path(save: str | None, dataset_name: str, *, grouped: bool) -> str | None:
    if save is None:
        return None
    path = Path(save)
    if not grouped:
        return str(path)
    suffix = path.suffix or ".png"
    base = path.with_suffix("") if path.suffix else path
    return str(base.parent / f"{base.name}_{dataset_name}{suffix}")


def _format_targets(result: RohanTopResult) -> list[str]:
    lines: list[str] = []
    for idx, target in enumerate(result.target_frequencies):
        signed = result.target_signed[idx]
        amp = result.target_amplitude[idx]
        if signed.size == 0:
            continue
        dominant = int(np.nanargmax(np.abs(signed)))
        lines.append(
            f"  {target:.3f} Hz: bond {dominant + 1}, signed={signed[dominant]:.6g}, amplitude={amp[dominant]:.6g}"
        )
    return lines


def _print_summary(result: RohanTopResult) -> None:
    print(
        f"Dataset {result.dataset_name} | component {result.component} | window {result.tmin:g} to {result.tmax:g} s"
    )
    print(f"  Source: {result.source_path}")
    print(
        f"  Samples: {result.time_s.size} | bonds: {result.bond_signal.shape[1]} | FFT windows: {result.window_count}"
    )
    if result.chirality_smoothed.size > 0:
        finite = result.chirality_smoothed[np.isfinite(result.chirality_smoothed)]
        if finite.size > 0:
            print(
                f"  Chirality: min={np.min(finite):.6g} max={np.max(finite):.6g} mean={np.mean(finite):.6g}"
            )
    for line in _format_targets(result):
        print(line)


def _plot_result(result: RohanTopResult, *, title: str | None = None) -> plt.Figure:
    freq_mask = (result.freq_hz >= 0) & (result.freq_hz <= min(25.0, 0.5 * result.fs))
    plot_freq = result.freq_hz[freq_mask]
    plot_amp = result.avg_amplitude[:, freq_mask]
    n_bonds = result.bond_signal.shape[1]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 10.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.6, 1.2]},
    )

    ax_fft, ax_targets, ax_chiral = axes
    cmap = plt.get_cmap("turbo", max(1, n_bonds))
    for bond_idx in range(n_bonds):
        ax_fft.plot(
            plot_freq,
            plot_amp[bond_idx] + bond_idx / 20.0,
            color=cmap(bond_idx),
            linewidth=1.0,
        )
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude + offset")
    ax_fft.set_title(title or f"{result.dataset_name} component {result.component}")
    ax_fft.grid(True, alpha=0.25)
    for target in result.target_frequencies:
        ax_fft.axvline(float(target), color="black", linewidth=0.6, alpha=0.2)

    colors = plt.get_cmap("turbo", max(1, result.target_frequencies.size))
    bond_axis = np.arange(1, n_bonds + 1, dtype=float)
    for idx, target in enumerate(result.target_frequencies):
        row = result.target_signed[idx]
        norm = float(np.linalg.norm(row))
        if norm > 0 and np.isfinite(norm):
            lead = row[0] if row.size > 0 and np.isfinite(row[0]) and row[0] != 0 else 1.0
            display = np.sign(lead) * row / norm + (idx + 1)
        else:
            display = np.full_like(row, idx + 1, dtype=float)
        ax_targets.plot(bond_axis, display, color=colors(idx), linewidth=0.8)
        ax_targets.axhline(idx + 1, color="gray", linewidth=0.3, alpha=0.3)
    ax_targets.set_xlabel("Bond index")
    ax_targets.set_ylabel("Target bands")
    ax_targets.set_title("Target-frequency participation")
    ax_targets.set_yticks(np.arange(1, result.target_frequencies.size + 1))
    ax_targets.set_yticklabels([f"{freq:g}" for freq in result.target_frequencies])
    ax_targets.grid(True, alpha=0.2)

    ax_chiral.plot(result.time_s, result.chirality_smoothed, color="tab:blue", linewidth=1.4, label="smoothed")
    ax_chiral.plot(result.time_s, result.chirality, color="tab:gray", linewidth=0.8, alpha=0.4, label="raw")
    ax_chiral.set_xlabel("Time (s)")
    ax_chiral.set_ylabel("Chirality")
    ax_chiral.set_title("Chirality diagnostic")
    ax_chiral.grid(True, alpha=0.25)
    ax_chiral.legend(loc="upper right", frameon=False)

    return fig


def _run_single(dataset: str, args: argparse.Namespace) -> RohanTopResult:
    return analyze_rohan_top(
        dataset,
        component=str(args.component),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        track_data_root=args.track_data_root,
        fs=float(args.fs),
        win=int(args.win),
        av=int(args.av),
        offset=int(args.offset),
        target_frequencies=np.asarray(args.targets if args.targets is not None else DEFAULT_TARGET_FREQUENCIES, dtype=float),
        smooth_window=int(args.smooth_window),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])

    datasets = [str(args.dataset)] if args.dataset is not None else []
    grouped = False
    if args.group is not None:
        if args.dataset is not None:
            raise ValueError("Provide either DATASET or --group, not both")
        datasets = load_group_datasets(args.group, groups_dir=args.groups_dir)
        grouped = True
    elif args.dataset is None:
        raise ValueError("Provide a dataset or --group")

    for dataset in datasets:
        result = _run_single(dataset, args)
        _print_summary(result)
        fig = _plot_result(result, title=args.title)
        save_path = _resolve_save_path(args.save, result.dataset_name, grouped=grouped)
        render_figure(fig, save=save_path)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
