#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import msgpack
import numpy as np
from scipy.optimize import curve_fit

from tools.io import get_default_track_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporary diagnostic plot: area versus x with simple camera-bias fits."
    )
    parser.add_argument("dataset", nargs="?", default="IMG_0681_rot270")
    parser.add_argument(
        "--track-data-root",
        default=str(get_default_track_data_root()),
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional save path. By default the plot is shown interactively.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of bins for the median trend line. Default: 60",
    )
    return parser


def load_msgpack(path: Path) -> dict:
    with open(path, "rb") as fh:
        return msgpack.unpackb(fh.read(), raw=False)


def dataset_paths(dataset: str, track_data_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(track_data_root)
    track1 = root / dataset / "track1.msgpack"
    x_track2 = root / f"{dataset}_x" / "track2_permanence.msgpack"
    area_track2 = root / f"{dataset}_area" / "track2_permanence.msgpack"
    return track1, x_track2, area_track2


def build_xya(track1_payload: dict, x_track2_payload: dict, area_track2_payload: dict) -> tuple[np.ndarray, np.ndarray]:
    frames = track1_payload["frames"]
    x_positions = np.asarray(x_track2_payload["xPositions"], dtype=float)
    area_positions = np.asarray(area_track2_payload["xPositions"], dtype=float)

    if len(frames) != x_positions.shape[0] or x_positions.shape != area_positions.shape:
        raise ValueError("track1, _x, and _area shapes do not align.")

    mask = np.isfinite(x_positions) & np.isfinite(area_positions)
    return x_positions[mask], area_positions[mask]


def binned_median(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.array([]), np.array([])

    edges = np.linspace(float(np.min(x)), float(np.max(x)), int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(int(bins), np.nan, dtype=float)

    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, int(bins) - 1)
    for b in range(int(bins)):
        vals = y[idx == b]
        if vals.size > 0:
            med[b] = float(np.median(vals))
    return centers, med


def quadratic_model_centered(xc: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (xc + b) ** 2 + c


def eval_shifted_quadratic(x: np.ndarray, center: float, a: float, shift: float, c: float) -> np.ndarray:
    xc = x - center
    return quadratic_model_centered(xc, a, shift, c)


def even_quartic_model(x: np.ndarray, x0: float, c4: float, c2: float, c0: float) -> np.ndarray:
    xc = x - x0
    return c4 * xc ** 4 + c2 * xc ** 2 + c0


def eval_even_quartic(x: np.ndarray, center: float, c4: float, c2: float, c0: float) -> np.ndarray:
    return even_quartic_model(x, center, c4, c2, c0)


def even_sixth_model(x: np.ndarray, x0: float, c6: float, c4: float, c2: float, c0: float) -> np.ndarray:
    xc = x - x0
    return c6 * xc ** 6 + c4 * xc ** 4 + c2 * xc ** 2 + c0


def even_eighth_model(
    x: np.ndarray,
    x0: float,
    c8: float,
    c6: float,
    c4: float,
    c2: float,
    c0: float,
) -> np.ndarray:
    xc = x - x0
    return c8 * xc ** 8 + c6 * xc ** 6 + c4 * xc ** 4 + c2 * xc ** 2 + c0


def fit_shifted_quadratic(x: np.ndarray, y: np.ndarray, center: float) -> tuple[float, float, float]:
    xc = x - center
    amp = float(np.max(y) - np.min(y)) if y.size else 1.0
    span = float(np.max(np.abs(xc))) if xc.size else 1.0
    a0 = amp / max(span ** 2, 1.0)
    b0 = 0.0
    c0 = float(np.median(y)) if y.size else 0.0
    popt, _ = curve_fit(
        quadratic_model_centered,
        xc,
        y,
        p0=(a0, b0, c0),
        maxfev=20000,
    )
    a, b, c = popt
    return float(a), float(b), float(c)


def fit_even_quartic(x: np.ndarray, y: np.ndarray, center: float) -> tuple[float, float, float, float]:
    span = float(np.max(np.abs(x - center))) if x.size else 1.0
    amp = float(np.max(y) - np.min(y)) if y.size else 1.0
    p0 = (
        center,
        0.0,
        amp / max(span ** 2, 1.0),
        float(np.median(y)) if y.size else 0.0,
    )
    bounds = (
        (float(np.min(x)), -np.inf, -np.inf, -np.inf),
        (float(np.max(x)), np.inf, np.inf, np.inf),
    )
    popt, _ = curve_fit(
        even_quartic_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    x0, c4, c2, c0 = popt
    return float(x0), float(c4), float(c2), float(c0)


def fit_even_sixth(x: np.ndarray, y: np.ndarray, center: float) -> tuple[float, float, float, float, float]:
    span = float(np.max(np.abs(x - center))) if x.size else 1.0
    amp = float(np.max(y) - np.min(y)) if y.size else 1.0
    p0 = (
        center,
        0.0,
        0.0,
        amp / max(span ** 2, 1.0),
        float(np.median(y)) if y.size else 0.0,
    )
    bounds = (
        (float(np.min(x)), -np.inf, -np.inf, -np.inf, -np.inf),
        (float(np.max(x)), np.inf, np.inf, np.inf, np.inf),
    )
    popt, _ = curve_fit(
        even_sixth_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    x0, c6, c4, c2, c0 = popt
    return float(x0), float(c6), float(c4), float(c2), float(c0)


def fit_even_eighth(x: np.ndarray, y: np.ndarray, center: float) -> tuple[float, float, float, float, float, float]:
    span = float(np.max(np.abs(x - center))) if x.size else 1.0
    amp = float(np.max(y) - np.min(y)) if y.size else 1.0
    p0 = (
        center,
        0.0,
        0.0,
        0.0,
        amp / max(span ** 2, 1.0),
        float(np.median(y)) if y.size else 0.0,
    )
    bounds = (
        (float(np.min(x)), -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),
        (float(np.max(x)), np.inf, np.inf, np.inf, np.inf, np.inf),
    )
    popt, _ = curve_fit(
        even_eighth_model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=80000,
    )
    x0, c8, c6, c4, c2, c0 = popt
    return float(x0), float(c8), float(c6), float(c4), float(c2), float(c0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    track1_path, x_track2_path, area_track2_path = dataset_paths(args.dataset, args.track_data_root)
    if not track1_path.exists():
        raise FileNotFoundError(f"track1 not found: {track1_path}")
    if not x_track2_path.exists():
        raise FileNotFoundError(f"X permanence not found: {x_track2_path}")
    if not area_track2_path.exists():
        raise FileNotFoundError(f"Area permanence not found: {area_track2_path}")

    track1 = load_msgpack(track1_path)
    x_track2 = load_msgpack(x_track2_path)
    area_track2 = load_msgpack(area_track2_path)
    x, area = build_xya(track1, x_track2, area_track2)

    bx, bm = binned_median(x, area, args.bins)
    fit_mask = np.isfinite(bx) & np.isfinite(bm)
    fit_x = bx[fit_mask]
    fit_y = bm[fit_mask]
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 800)
    center = 0.5 * (float(np.min(x)) + float(np.max(x)))

    qa, qshift, qc = fit_shifted_quadratic(fit_x, fit_y, center)
    q_curve = eval_shifted_quadratic(x_grid, center, qa, qshift, qc)

    q6_center, c6, c6_4, c6_2, c6_0 = fit_even_sixth(fit_x, fit_y, center)
    q6_curve = even_sixth_model(x_grid, q6_center, c6, c6_4, c6_2, c6_0)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.scatter(x, area, s=4, alpha=0.08, linewidths=0, color="tab:blue", label="raw")
    ax.plot(bx, bm, color="black", linewidth=2.5, label="binned median")
    ax.plot(x_grid, q_curve, color="tab:red", linewidth=2, label=r"$a(x+b)^2 + C$")
    ax.plot(x_grid, q6_curve, color="tab:orange", linewidth=2, label=r"6th even")
    ax.set_title(f"{args.dataset}: area vs x")
    ax.set_xlabel("Centroid x [px]")
    ax.set_ylabel("Area [px^2]")
    ax.legend(loc="best")

    txt = "\n".join([
        rf"$a={qa:.6g}$",
        rf"$x_{{0,\mathrm{{quad}}}}={center:.6g}$",
        rf"$b={qshift:.6g}$",
        rf"$C={qc:.6g}$",
        rf"$x_{{0,\mathrm{{6th}}}}={q6_center:.6g}$",
        rf"$c_6={c6:.6g}$",
        rf"$c_4={c6_4:.6g}$",
        rf"$c_2={c6_2:.6g}$",
        rf"$c_0={c6_0:.6g}$",
    ])
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    if args.save:
        save_path = Path(args.save)
        fig.savefig(save_path, dpi=160)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    print(f"Samples: {x.size}")
    print(f"Quadratic fit: a={qa:.9g}, x0={center:.9g}, b={qshift:.9g}, C={qc:.9g}")
    print(f"Even sixth center: x0={q6_center:.9g}")
    print(f"Even sixth fit: c6={c6:.9g}, c4={c6_4:.9g}, c2={c6_2:.9g}, c0={c6_0:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
