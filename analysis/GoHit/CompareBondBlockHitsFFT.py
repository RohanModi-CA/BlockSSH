#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.GoHit.tools.hits import (
    build_interhit_regions,
    build_posthit_regions,
    load_catalog_if_available,
)
from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.flattening import (
    apply_global_baseline_processing_to_results,
    plot_flattening_diagnostic,
)
from analysis.tools.io import load_track2_dataset
from analysis.tools.models import AverageSpectrumResult
from analysis.tools.signal import compute_one_sided_fft, preprocess_signal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare averaged hit-window FFTs for default bond-x spacing versus raw block-x positions.",
    )
    parser.add_argument("dataset", help="Dataset stem, e.g. 11triv")
    parser.add_argument("--track-data-root", default=None)
    parser.add_argument(
        "--region-mode",
        choices=["interhit", "posthit"],
        default="interhit",
        help="Which GoHit regions to average. Default: interhit",
    )
    parser.add_argument("--delay", type=float, default=10.0, help="Seconds to exclude after each hit for interhit mode.")
    parser.add_argument(
        "--exclude-before",
        type=float,
        default=3.0,
        help="Seconds to exclude before the next hit for interhit mode.",
    )
    parser.add_argument("--hit-window", type=float, default=5.0, help="Window size in seconds for posthit mode.")
    parser.add_argument("--freq-max-hz", type=float, default=30.0, help="Maximum plotted frequency. Default: 30")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten both spectra using the shared baseline flattening helper.",
    )
    parser.add_argument(
        "--baseline-match",
        choices=["none", "bond", "block"],
        default="none",
        help="After computing baselines, warp one spectrum to match the other's baseline envelope/reference.",
    )
    parser.add_argument(
        "--flatten-reference-band",
        nargs=2,
        type=float,
        default=(20.0, 30.0),
        metavar=("START_HZ", "STOP_HZ"),
        help="Reference band used by the flattening helper. Default: 20 30",
    )
    parser.add_argument("--save", default=None, help="Optional output image path.")
    parser.add_argument("--save-csv", default=None, help="Optional CSV path for the compared spectra.")
    parser.add_argument(
        "--save-flatten-diagnostics",
        action="store_true",
        help="Save one flattening diagnostic plot per spectrum when flattening or baseline matching is active.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def _preprocess_matrix(t: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    processed_cols: list[np.ndarray] = []
    processed_t: np.ndarray | None = None
    for idx in range(matrix.shape[1]):
        processed, error = preprocess_signal(t, matrix[:, idx])
        if processed is None:
            print(f"Skipping column {idx + 1}: {error}", file=sys.stderr)
            continue
        if processed_t is None:
            processed_t = np.asarray(processed.t, dtype=float)
        processed_cols.append(np.asarray(processed.y, dtype=float))

    if processed_t is None or not processed_cols:
        raise ValueError("No usable signals remained after preprocessing")
    return processed_t, np.column_stack(processed_cols)


def _average_region_spectra(pt: np.ndarray, sig_matrix: np.ndarray, regions) -> tuple[np.ndarray, np.ndarray]:
    all_region_means: list[np.ndarray] = []
    common_freq: np.ndarray | None = None
    dt = float(np.median(np.diff(pt)))

    for region in regions:
        mask = (pt >= float(region.start_s)) & (pt <= float(region.stop_s))
        series: list[tuple[np.ndarray, np.ndarray]] = []
        for col in range(sig_matrix.shape[1]):
            seg = np.asarray(sig_matrix[mask, col], dtype=float)
            if seg.size < 16:
                continue
            fft_res = compute_one_sided_fft(seg, dt)
            freq = np.asarray(fft_res.freq, dtype=float)
            amp = np.asarray(fft_res.amplitude, dtype=float)
            if freq.size < 2 or amp.size != freq.size:
                continue
            if common_freq is None:
                common_freq = freq
            series.append((freq, amp))

        if not series:
            continue
        assert common_freq is not None
        region_mean = np.mean([np.interp(common_freq, f, a) for f, a in series], axis=0)
        all_region_means.append(region_mean)

    if common_freq is None or not all_region_means:
        raise ValueError("No usable FFT segments were available for the chosen regions")
    return common_freq, np.mean(np.vstack(all_region_means), axis=0)


def _to_average_result(freq: np.ndarray, amp: np.ndarray) -> AverageSpectrumResult:
    return AverageSpectrumResult(
        freq_grid=np.asarray(freq, dtype=float),
        avg_amp=np.asarray(amp, dtype=float),
        norm_low=float(freq[0]),
        norm_high=float(freq[-1]),
        freq_low=float(freq[0]),
        freq_high=float(freq[-1]),
        contributors=[],
    )


def _build_regions(args, *, hit_times_s, t_stop_s: float):
    if args.region_mode == "posthit":
        return build_posthit_regions(
            hit_times_s,
            t_stop_s=t_stop_s,
            window_s=float(args.hit_window),
        )
    return build_interhit_regions(
        hit_times_s,
        t_stop_s=t_stop_s,
        exclude_after_s=float(args.delay),
        exclude_before_s=float(args.exclude_before),
    )


def _save_csv(path: Path, *, bond_freq, bond_amp, block_freq, block_amp) -> None:
    common_freq = np.asarray(bond_freq, dtype=float)
    block_interp = np.interp(common_freq, block_freq, block_amp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq_hz", "bond_x_fft", "block_x_fft"])
        for hz, bond_val, block_val in zip(common_freq, bond_amp, block_interp):
            writer.writerow([float(hz), float(bond_val), float(block_val)])


def main() -> int:
    args = build_parser().parse_args()

    if not args.no_show:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if args.flatten_reference_band[1] <= args.flatten_reference_band[0]:
        print("Error: --flatten-reference-band STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        catalog = load_catalog_if_available(args.dataset)
        if catalog is None:
            raise ValueError(
                f"No GoHit confirmed hit catalog exists for dataset '{args.dataset}'. "
                f"Run python3 analysis/GoHit/HitReview.py {args.dataset} first."
            )

        bond_dataset = load_bond_signal_dataset(
            dataset=f"{args.dataset}_x",
            track_data_root=args.track_data_root,
            bond_spacing_mode="default",
            component="x",
        )
        track2 = load_track2_dataset(
            dataset=f"{args.dataset}_x",
            track_data_root=args.track_data_root,
        )

        bond_t, bond_matrix = _preprocess_matrix(
            np.asarray(bond_dataset.frame_times_s, dtype=float),
            np.asarray(bond_dataset.signal_matrix, dtype=float),
        )
        block_t, block_matrix = _preprocess_matrix(
            np.asarray(track2.frame_times_s, dtype=float),
            np.asarray(track2.x_positions, dtype=float),
        )

        start_t = max(float(bond_t[0]), float(block_t[0]))
        stop_t = min(float(bond_t[-1]), float(block_t[-1]))
        if stop_t <= start_t:
            raise ValueError("Bond and block traces have no overlapping valid time span")

        bond_keep = (bond_t >= start_t) & (bond_t <= stop_t)
        block_keep = (block_t >= start_t) & (block_t <= stop_t)
        bond_t = bond_t[bond_keep]
        bond_matrix = bond_matrix[bond_keep, :]
        block_t = block_t[block_keep]
        block_matrix = block_matrix[block_keep, :]

        regions = _build_regions(
            args,
            hit_times_s=catalog.hit_times_s,
            t_stop_s=min(float(bond_t[-1]), float(block_t[-1])),
        )
        if len(regions) == 0:
            raise ValueError(f"No usable {args.region_mode} regions were available")

        bond_freq, bond_amp_raw = _average_region_spectra(bond_t, bond_matrix, regions)
        block_freq, block_amp_raw = _average_region_spectra(block_t, block_matrix, regions)

        raw_results = OrderedDict(
            bond=_to_average_result(bond_freq, bond_amp_raw),
            block=_to_average_result(block_freq, block_amp_raw),
        )
        processed_results, flattenings = apply_global_baseline_processing_to_results(
            raw_results,
            flatten=bool(args.flatten),
            baseline_match=str(args.baseline_match),
            reference_band=(float(args.flatten_reference_band[0]), float(args.flatten_reference_band[1])),
        )
        bond_amp = np.asarray(processed_results["bond"].avg_amp, dtype=float)
        block_amp = np.asarray(processed_results["block"].avg_amp, dtype=float)

        out_path = (
            Path(args.save)
            if args.save is not None
            else Path(__file__).resolve().parent / "out" / f"{args.dataset}_bond_vs_block_x_{args.region_mode}_fft.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
        ax0.plot(bond_t, np.nanmean(bond_matrix, axis=1), color="tab:blue", lw=0.7, label="mean bond x")
        ax0.plot(block_t, np.nanmean(block_matrix, axis=1), color="black", lw=0.7, alpha=0.8, label="mean block x")
        for region in regions:
            ax0.axvspan(region.start_s, region.stop_s, color="tab:green", alpha=0.14)
        for hit_time in catalog.hit_times_s:
            ax0.axvline(hit_time, color="crimson", linestyle="--", linewidth=0.7, alpha=0.45)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Mean signal")
        ax0.set_title(f"{args.dataset} {args.region_mode} windows")
        ax0.grid(True, alpha=0.25)
        ax0.legend()

        mask_bond = (bond_freq >= 0.0) & (bond_freq <= float(args.freq_max_hz))
        mask_block = (block_freq >= 0.0) & (block_freq <= float(args.freq_max_hz))
        ax1.plot(bond_freq[mask_bond], bond_amp[mask_bond], color="tab:blue", lw=1.6, label="Bond x FFT (default spacing)")
        ax1.plot(block_freq[mask_block], block_amp[mask_block], color="black", lw=1.6, label="Block x FFT")
        ax1.set_yscale("log")
        ax1.set_xlim(0.0, float(args.freq_max_hz))
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("FFT amplitude")
        title_suffix = []
        if args.flatten:
            title_suffix.append("flattened")
        if args.baseline_match != "none":
            title_suffix.append(f"baseline-matched to {args.baseline_match}")
        suffix_text = " | ".join(title_suffix) if title_suffix else "raw"
        ax1.set_title(
            f"Bond x default vs block x | {args.region_mode} | delay={args.delay:g}s | exclude-before={args.exclude_before:g}s | {suffix_text}"
        )
        ax1.grid(True, which="both", alpha=0.25)
        ax1.legend()
        fig.savefig(out_path, dpi=180)

        if args.save_csv is not None:
            _save_csv(
                Path(args.save_csv),
                bond_freq=bond_freq,
                bond_amp=bond_amp,
                block_freq=block_freq,
                block_amp=block_amp,
            )

        if args.save_flatten_diagnostics and flattenings:
            for key, color_label in (("bond", "bond_x_default"), ("block", "block_x")):
                flat = flattenings.get(key)
                if flat is None:
                    continue
                diag_fig = plot_flattening_diagnostic(
                    raw_results[key].freq_grid,
                    raw_results[key].avg_amp,
                    flat,
                    title=f"{args.dataset} {color_label} flattening diagnostic",
                )
                diag_path = out_path.with_name(f"{out_path.stem}_{key}_flatten.png")
                diag_fig.savefig(diag_path, dpi=180)
                plt.close(diag_fig)
                print(f"Saved flatten diagnostic: {diag_path}")

        print(f"Dataset: {args.dataset}")
        print(f"Catalog hits: {len(catalog.hit_times_s)}")
        print(f"Region mode: {args.region_mode}")
        print(f"Delay: {args.delay:g} s")
        print(f"Exclude before: {args.exclude_before:g} s")
        print(f"Hit window: {args.hit_window:g} s")
        print(f"Usable regions: {len(regions)}")
        print(f"Bonds used: {bond_matrix.shape[1]}")
        print(f"Blocks used: {block_matrix.shape[1]}")
        print(f"Flatten: {'on' if args.flatten else 'off'}")
        print(f"Baseline match: {args.baseline_match}")
        print(f"Saved plot: {out_path}")
        if args.save_csv is not None:
            print(f"Saved CSV: {Path(args.save_csv)}")

        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
