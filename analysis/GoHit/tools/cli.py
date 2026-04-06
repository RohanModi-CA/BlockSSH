from __future__ import annotations

import argparse


def add_hit_mode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hits",
        action="store_true",
        help="Use the GoHit confirmed-hit pipeline instead of the default full-trace analysis.",
    )
    parser.add_argument(
        "--posthit",
        action="store_true",
        help="Use post-hit fixed windows instead of interhit windows when --hits is enabled.",
    )
    parser.add_argument(
        "--hit-component",
        choices=["x", "y", "a"],
        default="x",
        help="Component used to detect and confirm the shared dataset hit catalog. Default: x",
    )


def add_hit_region_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to exclude after each hit for interhit regions when --hits is enabled. Default: 1.0",
    )
    parser.add_argument(
        "--exclude-before",
        type=float,
        default=1.0,
        help="Seconds to exclude before the next hit for interhit regions when --hits is enabled. Default: 1.0",
    )
    parser.add_argument(
        "--hit-window",
        type=float,
        default=5.0,
        help="Window length in seconds for posthit regions when --hits is enabled. Default: 5.0",
    )


def describe_hit_region_settings(*, posthit: bool, delay: float, exclude_before: float, hit_window: float) -> list[str]:
    lines = [f"Region mode: {'posthit' if posthit else 'interhit'}"]
    if posthit:
        lines.append(f"Hit window (s): {float(hit_window):g}")
    else:
        lines.append(f"Delay (s): {float(delay):g}")
        lines.append(f"Exclude before (s): {float(exclude_before):g}")
    return lines
