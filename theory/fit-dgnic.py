import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from helpers.dgnic import (
    DEFAULT_K,
    DEFAULT_L,
    DEFAULT_M1,
    DEFAULT_M2,
    DEFAULT_N,
    DISORDER_SEED,
    load_peaks_csv,
    select_frequencies,
    solve_chain,
)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from plotting.frequency import _plot_frequency_image
from tools.spectrasave import load_spectrum_msgpack


def parse_index_list(text):
    indices = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        indices.append(int(item))
    return indices


def validate_drop_indices(name, indices, size):
    unique = sorted(set(indices))
    if len(unique) != len(indices):
        raise ValueError(f"{name} contains duplicates: {indices}")
    for idx in unique:
        if idx < 1 or idx > size:
            raise ValueError(f"{name} index {idx} is outside 1..{size}")
    return unique


def build_lineplot_image(freq, fmax, n_cols, n_rows=800):
    img = np.zeros((n_rows, n_cols), dtype=float)

    if fmax <= 0:
        return img

    for f in np.asarray(freq, dtype=float):
        row = int(round((f / fmax) * (n_rows - 1)))
        row = max(0, min(n_rows - 1, row))
        r0 = max(0, row - 1)
        r1 = min(n_rows, row + 2)
        img[r0:r1, :] = 1.0

    return img


def plot_lineplot_panel(ax, freq, fmax, title):
    n_cols = max(len(freq), 1)
    img = build_lineplot_image(freq, fmax, n_cols)
    extent = [0.5, n_cols + 0.5, 0.0, fmax]
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="jet",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Frequency [Hz]")


def resolve_spectrasave_path(raw_path):
    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.suffix:
        candidates.append(path.with_suffix(".msgpack"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"SpectraSave file not found. Tried: {tried}")


def link_image_frequency_axes(axes):
    if len(axes) < 2:
        return

    sync_state = {"active": False}

    def _sync(changed_ax):
        if sync_state["active"]:
            return
        sync_state["active"] = True
        try:
            ylim = tuple(changed_ax.get_ylim())
            for ax in axes:
                if ax is not changed_ax and tuple(ax.get_ylim()) != ylim:
                    ax.set_ylim(ylim)
        finally:
            sync_state["active"] = False

    for ax in axes:
        ax.callbacks.connect("ylim_changed", _sync)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the dgnic mass-spring model to a CSV of target peak frequencies."
    )
    parser.add_argument("peaks_csv", help="CSV containing strictly increasing target peaks in Hz.")
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="Chain length. Default: 10.")
    parser.add_argument(
        "--skip-lowest",
        type=int,
        default=1,
        help="Ignore this many lowest-frequency model modes before matching. Default: 1.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of model modes to compare after skipping. Default: use all peaks from the CSV.",
    )
    parser.add_argument("--switch", action="store_true", help="Swap m1 and m2 before solving.")
    parser.add_argument(
        "--disorder",
        type=float,
        default=0.0,
        help="Fixed percent disorder to include during fitting. Default: 0.",
    )
    parser.add_argument(
        "--mass-disorder-ratio",
        type=float,
        default=1.0,
        help="Multiply the spring disorder fraction by this factor for the masses. Default: 1.",
    )
    parser.add_argument("--seed", type=int, default=DISORDER_SEED, help="Disorder seed. Default: 7.")
    parser.add_argument(
        "--fit-seed",
        action="store_true",
        help="Scan seeds from --min-seed to --max-seed and keep the best fit. Default: false.",
    )
    parser.add_argument(
        "--min-seed",
        type=int,
        default=1,
        help="Minimum disorder seed to test when --fit-seed is enabled. Default: 1.",
    )
    parser.add_argument(
        "--max-seed",
        type=int,
        default=100,
        help="Maximum disorder seed to test when --fit-seed is enabled. Default: 100.",
    )
    parser.add_argument("--m1", type=float, default=DEFAULT_M1, help="Initial guess for m1 in kg.")
    parser.add_argument("--m2", type=float, default=DEFAULT_M2, help="Initial guess for m2 in kg.")
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="Initial guess for spring constant.")
    parser.add_argument("--L", type=float, default=DEFAULT_L, help="Initial guess for pendulum length in m.")
    parser.add_argument("--fix-m1", action="store_true", help="Hold m1 fixed at the supplied initial value.")
    parser.add_argument("--fix-m2", action="store_true", help="Hold m2 fixed at the supplied initial value.")
    parser.add_argument("--fix-k", action="store_true", help="Hold k fixed at the supplied initial value.")
    parser.add_argument("--fix-L", action="store_true", help="Hold L fixed at the supplied initial value.")
    parser.add_argument(
        "--lower-m1",
        type=float,
        default=1e-4,
        help="Lower bound for m1 in kg. Default: 1e-4.",
    )
    parser.add_argument(
        "--upper-m1",
        type=float,
        default=0.5,
        help="Upper bound for m1 in kg. Default: 0.5.",
    )
    parser.add_argument(
        "--lower-m2",
        type=float,
        default=1e-4,
        help="Lower bound for m2 in kg. Default: 1e-4.",
    )
    parser.add_argument(
        "--upper-m2",
        type=float,
        default=0.5,
        help="Upper bound for m2 in kg. Default: 0.5.",
    )
    parser.add_argument(
        "--lower-k",
        type=float,
        default=1.0,
        help="Lower bound for k. Default: 1.",
    )
    parser.add_argument(
        "--upper-k",
        type=float,
        default=1e4,
        help="Upper bound for k. Default: 1e4.",
    )
    parser.add_argument(
        "--lower-L",
        type=float,
        default=0.05,
        help="Lower bound for L in m. Default: 0.05.",
    )
    parser.add_argument(
        "--upper-L",
        type=float,
        default=5.0,
        help="Upper bound for L in m. Default: 5.",
    )
    parser.add_argument(
        "--method",
        choices=["trf", "dogbox", "lm"],
        default="trf",
        help="least_squares method. Default: trf.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=2000,
        help="Maximum optimizer function evaluations. Default: 2000.",
    )
    parser.add_argument(
        "--drop-target-indices",
        type=parse_index_list,
        default=[],
        help=(
            "Comma-separated 1-based indices of target peaks to exclude before fitting. "
            "Indices refer to the selected target peak list."
        ),
    )
    parser.add_argument(
        "--drop-model-indices",
        type=parse_index_list,
        default=[],
        help=(
            "Comma-separated 1-based indices of the selected model mode list to exclude before fitting. "
            "Indices refer to the model frequencies after --skip-lowest and --count are applied."
        ),
    )
    parser.add_argument(
        "--search-drop-target",
        action="store_true",
        help="Try all single target-peak omissions and keep the best fit.",
    )
    parser.add_argument(
        "--search-drop-model",
        action="store_true",
        help="Try all single model-mode omissions and keep the best fit.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Show lineplot-style images for target peaks and fitted model peaks.",
    )
    parser.add_argument(
        "--loadspectra",
        default=None,
        metavar="PATH",
        help=(
            "Load a SpectraSave msgpack artifact and, with --full-image, compare its full-image panel "
            "side by side against the selected model peaks image using coupled zoom."
        ),
    )
    return parser.parse_args()


def build_fit_spec(args):
    names = []
    x0 = []
    lower = []
    upper = []
    fixed = {
        "m1": float(args.m1),
        "m2": float(args.m2),
        "k": float(args.k),
        "L": float(args.L),
    }

    specs = [
        ("m1", args.fix_m1, args.lower_m1, args.upper_m1),
        ("m2", args.fix_m2, args.lower_m2, args.upper_m2),
        ("k", args.fix_k, args.lower_k, args.upper_k),
        ("L", args.fix_L, args.lower_L, args.upper_L),
    ]

    for name, is_fixed, lo, hi in specs:
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError(f"Invalid bounds for {name}: [{lo}, {hi}]")
        value = fixed[name]
        if not (lo <= value <= hi):
            raise ValueError(f"Initial {name}={value} lies outside [{lo}, {hi}]")
        if not is_fixed:
            names.append(name)
            x0.append(value)
            lower.append(lo)
            upper.append(hi)

    return fixed, names, np.asarray(x0, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def unpack_params(x, fixed, names):
    params = dict(fixed)
    for idx, name in enumerate(names):
        params[name] = float(x[idx])
    return params


def apply_drops(values, drop_indices):
    kept = np.asarray(values, dtype=float)
    if not drop_indices:
        return kept
    mask = np.ones(kept.size, dtype=bool)
    mask[np.asarray(drop_indices, dtype=int) - 1] = False
    return kept[mask]


def build_model_selection(result, skip_lowest, count):
    return select_frequencies(result.freq, skip_lowest=skip_lowest, count=count)


def fit_for_seed(args, target, count, fixed, names, x0, lower, upper, seed, drop_target_indices, drop_model_indices):
    target_used = apply_drops(target, drop_target_indices)

    def residuals(x):
        params = unpack_params(x, fixed, names)
        result = solve_chain(
            N=args.N,
            m1=params["m1"],
            m2=params["m2"],
            k=params["k"],
            L=params["L"],
            switch=args.switch,
            disorder=args.disorder / 100.0,
            mass_disorder_ratio=args.mass_disorder_ratio,
            seed=seed,
        )
        model = build_model_selection(result, args.skip_lowest, count)
        model = apply_drops(model, drop_model_indices)
        if model.shape != target_used.shape:
            raise ValueError(
                "Model/target size mismatch after dropping peaks: "
                f"model has {model.size} selected modes, target has {target_used.size}"
            )
        return model - target_used

    if args.method == "lm":
        optimize_result = least_squares(
            residuals,
            x0,
            method="lm",
            max_nfev=args.max_nfev,
        )
    elif names:
        optimize_result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method=args.method,
            max_nfev=args.max_nfev,
        )
    else:
        class StaticResult:
            success = True
            status = 0
            message = "All parameters fixed; no optimization performed."
            x = np.asarray([], dtype=float)
            nfev = 0
            cost = 0.5 * float(np.sum(residuals(np.asarray([], dtype=float)) ** 2))

        optimize_result = StaticResult()

    best = unpack_params(optimize_result.x, fixed, names)
    final_result = solve_chain(
        N=args.N,
        m1=best["m1"],
        m2=best["m2"],
        k=best["k"],
        L=best["L"],
        switch=args.switch,
        disorder=args.disorder / 100.0,
        mass_disorder_ratio=args.mass_disorder_ratio,
        seed=seed,
    )
    fitted_full = build_model_selection(final_result, args.skip_lowest, count)
    fitted = apply_drops(fitted_full, drop_model_indices)
    resid = fitted - target_used
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return optimize_result, best, final_result, fitted_full, fitted, target_used, resid, rmse


def main():
    args = parse_args()
    if args.disorder < 0:
        raise ValueError("--disorder must be non-negative")
    if args.mass_disorder_ratio < 0:
        raise ValueError("--mass-disorder-ratio must be non-negative")
    if args.fit_seed and args.min_seed > args.max_seed:
        raise ValueError("--min-seed must be <= --max-seed")
    if args.count is not None and args.count < 1:
        raise ValueError("--count must be at least 1")

    target = load_peaks_csv(args.peaks_csv)
    count = len(target) if args.count is None else args.count
    if count > args.N - args.skip_lowest:
        raise ValueError("--count exceeds available model modes after --skip-lowest")

    drop_target_choices = [validate_drop_indices("--drop-target-indices", args.drop_target_indices, len(target))]
    drop_model_choices = [validate_drop_indices("--drop-model-indices", args.drop_model_indices, count)]

    if args.search_drop_target:
        drop_target_choices = [[idx] for idx in range(1, len(target) + 1)]
    if args.search_drop_model:
        drop_model_choices = [[idx] for idx in range(1, count + 1)]

    fixed, names, x0, lower, upper = build_fit_spec(args)
    seeds = [args.seed]
    if args.fit_seed:
        seeds = list(range(args.min_seed, args.max_seed + 1))

    trials = []
    for drop_target_indices in drop_target_choices:
        for drop_model_indices in drop_model_choices:
            for seed in seeds:
                optimize_result, best, final_result, fitted_full, fitted, target_used, resid, rmse = fit_for_seed(
                    args,
                    target,
                    count,
                    fixed,
                    names,
                    x0,
                    lower,
                    upper,
                    seed,
                    drop_target_indices,
                    drop_model_indices,
                )
                trials.append(
                    (
                        rmse,
                        seed,
                        optimize_result,
                        best,
                        final_result,
                        fitted_full,
                        fitted,
                        target_used,
                        resid,
                        tuple(drop_target_indices),
                        tuple(drop_model_indices),
                    )
                )

    rmse, best_seed, optimize_result, best, final_result, fitted_full, fitted, target_used, resid, drop_target_indices, drop_model_indices = min(
        trials,
        key=lambda item: (item[0], not bool(item[2].success), item[1]),
    )

    print(f"Target peaks CSV: {args.peaks_csv}")
    print(f"Selected target peaks [Hz]: {target}")
    print(f"skip_lowest={args.skip_lowest}, count={count}, N={args.N}, switch={args.switch}")
    print(f"dropped target indices (1-based within selected target): {list(drop_target_indices)}")
    print(f"dropped model indices (1-based within selected model list): {list(drop_model_indices)}")
    if args.fit_seed:
        print(
            f"fixed disorder: +/-{args.disorder:.3f}% with best seed {best_seed} "
            f"(searched {args.min_seed}-{args.max_seed})"
        )
    else:
        print(f"fixed disorder: +/-{args.disorder:.3f}% with seed {best_seed}")
    print()
    print("Fit status:")
    print(f"success={optimize_result.success}")
    print(f"status={optimize_result.status}")
    print(f"message={optimize_result.message}")
    print(f"nfev={optimize_result.nfev}")
    print()
    print("Best-fit parameters:")
    print(f"m1 = {best['m1']:.12g} kg")
    print(f"m2 = {best['m2']:.12g} kg")
    print(f"k  = {best['k']:.12g}")
    print(f"L  = {best['L']:.12g} m")
    print(f"mass_disorder_ratio = {args.mass_disorder_ratio:.12g}")
    print()
    print("Disordered masses:")
    print(final_result.masses)
    print("Disordered springs:")
    print(final_result.springs)
    print()
    print("Selected model peaks before drops [Hz]:")
    print(fitted_full)
    print("Selected target peaks after drops [Hz]:")
    print(target_used)
    print("Matched model peaks [Hz]:")
    print(fitted)
    print("Residuals model-target [Hz]:")
    print(resid)
    print(f"RMSE = {rmse:.12g} Hz")

    if args.full_image:
        fmax = 1.05 * max(
            float(np.max(target)) if len(target) else 0.0,
            float(np.max(fitted_full)) if len(fitted_full) else 0.0,
        )
        if args.loadspectra is not None:
            spectrasave_path = resolve_spectrasave_path(args.loadspectra)
            spectrum = load_spectrum_msgpack(spectrasave_path)
            freq = np.asarray(spectrum["freq"], dtype=float)
            amp = np.asarray(spectrum["amplitude"], dtype=float)
            if freq.ndim != 1 or amp.ndim != 1 or freq.size != amp.size:
                raise ValueError("Loaded SpectraSave payload must contain equal-length 1D freq and amplitude arrays")
            spectrasave_fmax = float(np.max(freq)) if freq.size else 0.0
            shared_fmax = max(fmax, spectrasave_fmax)
            if shared_fmax <= 0:
                shared_fmax = 1.0
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 7), constrained_layout=False)
            fig.subplots_adjust(left=0.07, right=0.96, bottom=0.10, top=0.92, wspace=0.22)
            plot_lineplot_panel(axes[0], fitted_full, shared_fmax, "Selected Model Peaks")
            _plot_frequency_image(
                fig,
                axes[1],
                freq=freq,
                amp=amp,
                plot_scale="log",
                cmap_index=6,
                y_min=0.0,
                y_max=shared_fmax,
                x_label="Arbitrary X",
                x_max=64.0,
                title=f"Loaded SpectraSave Full Image\n{spectrasave_path.name}",
                linear_color_label="Amplitude",
                log_color_label="Amplitude (dB)",
            )
            link_image_frequency_axes([axes[0], axes[1]])
        else:
            fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True)
            plot_lineplot_panel(axes[0], target, fmax, "Target Peaks")
            plot_lineplot_panel(axes[1], fitted_full, fmax, "Selected Model Peaks")
            plot_lineplot_panel(axes[2], fitted, fmax, "Matched Model Peaks")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
