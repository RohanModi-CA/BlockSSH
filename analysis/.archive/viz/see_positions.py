#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys

from plotting.common import render_figure
from plotting.trajectory import plot_track2_positions_overview
from tools.cli import add_output_args, add_track2_input_args
from tools.derived import summarize_track2_positions
from tools.io import load_track2_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Track2 persistent block trajectories and visibility.",
    )
    add_track2_input_args(parser)
    add_output_args(parser, include_title=True)
    parser.add_argument(
        "--framestrip",
        action="store_true",
        help="Show a sampled strip of original video frames.",
    )
    parser.add_argument(
        "--nframestrip",
        type=int,
        default=8,
        help="Number of sampled frames to show when --framestrip is used. Default: 8",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        track2 = load_track2_dataset(
            dataset=args.dataset,
            track2_path=args.track2,
            track_data_root=args.track_data_root,
        )
        summary = summarize_track2_positions(track2)

        print(f"Track2: {track2.track2_path}")
        print(f"Frames: {summary['n_frames']}")
        print(f"Persistent blocks: {summary['n_blocks']}")
        print(f"Time range: [{summary['time_start_s']:.6f}, {summary['time_end_s']:.6f}] s")
        print(f"Overall NaN fraction: {summary['nan_fraction']:.3f}")
        print("Visible samples per block:")
        counts = summary["visible_counts"]
        print("  " + ", ".join(f"{i}:{int(c)}" for i, c in enumerate(counts)))
        print(f"Frames with non-increasing visible x-order: {summary['bad_order_frames']}")

        fig = plot_track2_positions_overview(
            track2,
            framestrip=args.framestrip,
            nframestrip=args.nframestrip,
            title=args.title,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
