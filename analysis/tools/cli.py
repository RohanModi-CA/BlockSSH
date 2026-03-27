from __future__ import annotations

import argparse

from .io import get_default_track_data_root


class _StoreNormalizeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, "normalize_explicit", True)


class _StoreRelativeRangeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, tuple(float(value) for value in values))
        setattr(namespace, "relative_range_explicit", True)


def add_track2_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset stem, e.g. IMG_0584. Ignored if --track2 is given.",
    )
    parser.add_argument(
        "--track2",
        default=None,
        help="Explicit path to track2_permanence.msgpack.",
    )
    parser.add_argument(
        "--track-data-root",
        default=str(get_default_track_data_root()),
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )


def add_track_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track-data-root",
        default=str(get_default_track_data_root()),
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )


def add_signal_processing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--longest",
        action="store_true",
        help="Use the longest contiguous valid segment instead of interpolating through gaps.",
    )
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help="Discard invalid time/value samples before uniform resampling.",
    )
    parser.add_argument(
        "--timeseriesnorm",
        dest="timeseriesnorm",
        action="store_true",
        help="Rescale each processed bond time series to RMS 100 before spectral analysis (default).",
    )
    parser.add_argument(
        "--no-timeseriesnorm",
        dest="timeseriesnorm",
        action="store_false",
        help="Disable RMS-100 timeseries normalization before analysis.",
    )
    parser.set_defaults(timeseriesnorm=True)


def add_output_args(parser: argparse.ArgumentParser, *, include_title: bool = False) -> None:
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the figure.",
    )
    if include_title:
        parser.add_argument(
            "--title",
            default=None,
            help="Optional title override.",
        )


def add_colormap_arg(parser: argparse.ArgumentParser, *, default: int = 6) -> None:
    parser.add_argument(
        "--cm",
        type=int,
        choices=range(1, 11),
        default=default,
        help=f"Colormap index. Default: {default}",
    )


def add_normalization_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        normalize="absolute",
        normalize_explicit=False,
        relative_range=(2.0, 8.0),
        relative_range_explicit=False,
    )
    parser.add_argument(
        "--normalize",
        action=_StoreNormalizeAction,
        choices=["absolute", "relative"],
        help="Normalization mode. Default: absolute",
    )
    parser.add_argument(
        "--relative-range",
        nargs=2,
        type=float,
        action=_StoreRelativeRangeAction,
        metavar=("START_HZ", "STOP_HZ"),
        help="Relative-normalization band in Hz. Default: 2 8. Passing this without --normalize switches normalization to relative automatically.",
    )


def resolve_normalization_mode(args: argparse.Namespace) -> str:
    if getattr(args, "normalize_explicit", False):
        return str(args.normalize)
    if getattr(args, "relative_range_explicit", False):
        return "relative"
    return str(getattr(args, "normalize", "absolute"))


def add_average_domain_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--average-linear",
        dest="average_domain",
        action="store_const",
        const="linear",
        help="Average spectra in linear amplitude space (default).",
    )
    group.add_argument(
        "--average-log",
        dest="average_domain",
        action="store_const",
        const="log",
        help="Average spectra in dB amplitude space.",
    )
    parser.set_defaults(average_domain="linear")


def add_plot_scale_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-linear",
        dest="plot_scale",
        action="store_const",
        const="linear",
        help="Use a linear display scale.",
    )
    group.add_argument(
        "--plot-log",
        dest="plot_scale",
        action="store_const",
        const="log",
        help="Use a log / dB display scale (default).",
    )
    parser.set_defaults(plot_scale="log")


def add_peak_integration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--integration-window-width",
        type=float,
        default=0.1,
        help="Half-width in Hz for the peak-integration windows. Default: 0.1",
    )
    parser.add_argument(
        "--normalization-multiplier",
        type=float,
        default=4.0,
        help="ROI padding multiplier applied to the integration window width. Default: 4",
    )


def add_bond_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only-bonds",
        type=int,
        nargs="+",
        default=None,
        help="Keep only these 1-based global bond numbers, e.g. --only-bonds 1 2 9.",
    )
    parser.add_argument(
        "--exclude-bonds",
        type=int,
        nargs="+",
        default=[],
        help="Exclude these 1-based global bond numbers after any other bond filtering.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--odd-bonds",
        dest="bond_parity",
        action="store_const",
        const="odd",
        help="Keep only odd 1-based bond numbers.",
    )
    group.add_argument(
        "--even-bonds",
        dest="bond_parity",
        action="store_const",
        const="even",
        help="Keep only even 1-based bond numbers.",
    )
    parser.set_defaults(bond_parity=None)
