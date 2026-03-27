import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from helpers.dgnic import (
    DEFAULT_G,
    DEFAULT_K,
    DEFAULT_M1,
    DEFAULT_M2,
    DEFAULT_N,
    DEFAULT_L,
    DISORDER_SEED,
    difference_operator,
)


PANEL_CHOICES = ["freqs", "eigenvectors", "qspace", "thesis_transformed", "lineplot"]
ONLY_CHOICES = PANEL_CHOICES + ["none"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mass-spring / transformed SSH-like chain visualizer."
    )
    parser.add_argument("--m1", type=float, default=None)
    parser.add_argument("--m2", type=float, default=None)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument(
        "--left-end-mass-g",
        type=float,
        default=1.0,
        help="Mass of the extra left endpoint site in grams.",
    )
    parser.add_argument(
        "--right-end-mass-g",
        type=float,
        default=1.5,
        help="Mass of the extra right endpoint site in grams.",
    )
    parser.add_argument("--switch", action="store_true")
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DISORDER_SEED)
    parser.add_argument(
        "--disorder",
        type=float,
        default=100.0 * 0,
        help="Percent disorder applied independently to all masses and springs.",
    )
    parser.add_argument(
        "--mass-disorder-ratio",
        type=float,
        default=1.0,
        help="Multiply the spring disorder fraction by this factor for the masses. Default: 1.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ONLY_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--lineplot_max_freq",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--show-zero",
        action="store_true",
        help="Draw a light dotted zero line for each vertically shifted q-space mode.",
    )
    parser.add_argument(
        "--abs-amplitude",
        action="store_true",
        help="Plot abs(q) in the q-space panel instead of the signed bond stretch.",
    )
    parser.add_argument(
        "--neg",
        action="store_true",
        help="Multiply displayed q-space amplitudes by -1 before plotting.",
    )
    return parser.parse_args()


def make_axes(n_panels):
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    return fig, list(axes[:n_panels])


def build_lineplot_image(freq, fmax, n_cols, n_rows=800):
    img = np.zeros((n_rows, n_cols), dtype=float)

    if fmax <= 0:
        return img

    for f in freq:
        row = int(round((f / fmax) * (n_rows - 1)))
        row = max(0, min(n_rows - 1, row))

        r0 = max(0, row - 1)
        r1 = min(n_rows, row + 2)

        img[r0:r1, :] = 1.0

    return img


def sort_eigensystem(A):
    vals, vecs = np.linalg.eig(A)
    vals = np.real_if_close(vals, tol=1000)
    vecs = np.real_if_close(vecs, tol=1000)

    idx = np.argsort(np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    return vals, vecs


def apply_uniform_disorder(values, strength, rng):
    delta = rng.uniform(-strength, strength, size=np.shape(values))
    return np.asarray(values, dtype=float) * (1.0 + delta)


def build_custom_chain(
    *,
    interior_N,
    m1,
    m2,
    k,
    L,
    switch,
    disorder,
    mass_disorder_ratio,
    seed,
    left_end_mass,
    right_end_mass,
):
    if interior_N < 1:
        raise ValueError("N must be at least 1 for a chain with two added end sites")
    if min(m1, m2, k, L, left_end_mass, right_end_mass) <= 0:
        raise ValueError("All masses, k, and L must be positive")
    if disorder < 0:
        raise ValueError("disorder must be non-negative")
    if mass_disorder_ratio < 0:
        raise ValueError("mass_disorder_ratio must be non-negative")

    if switch:
        m1, m2 = m2, m1

    interior_masses = np.full(interior_N, m1, dtype=float)
    interior_masses[1::2] = m2
    base_masses = np.concatenate(
        ([left_end_mass], interior_masses, [right_end_mass])
    )
    total_N = base_masses.size
    base_springs = np.full(total_N - 1, k, dtype=float)

    rng = np.random.default_rng(seed)
    if disorder > 0:
        masses = apply_uniform_disorder(
            base_masses, disorder * mass_disorder_ratio, rng
        )
        springs = apply_uniform_disorder(base_springs, disorder, rng)
    else:
        masses = base_masses.copy()
        springs = base_springs.copy()

    H = np.zeros((total_N, total_N), dtype=float)
    omega_p_sq = DEFAULT_G / L

    for i in range(total_N):
        k_left = springs[i - 1] if i > 0 else 0.0
        k_right = springs[i] if i < total_N - 1 else 0.0
        H[i, i] = (k_left + k_right) / masses[i] + omega_p_sq

        if i > 0:
            H[i, i - 1] = -k_left / masses[i]
        if i < total_N - 1:
            H[i, i + 1] = -k_right / masses[i]

    lam, V = sort_eigensystem(H)
    lam[np.abs(lam) < 1e-14] = 0.0
    omega = np.sqrt(np.clip(lam, 0.0, None))
    freq = omega / (2.0 * np.pi)

    return SimpleNamespace(
        H=H,
        masses=masses,
        springs=springs,
        lam=lam,
        V=V,
        freq=freq,
        E_full=None,
        E_red=None,
        m1=m1,
        m2=m2,
        k=k,
        L=L,
        N=total_N,
        interior_N=interior_N,
        left_end_mass=left_end_mass,
        right_end_mass=right_end_mass,
        seed=seed,
        disorder=disorder,
        switched=switch,
    )


def main():
    args = parse_args()
    disorder_strength = args.disorder / 100.0
    interior_N = args.N
    m1 = DEFAULT_M1 if args.m1 is None else args.m1
    m2 = DEFAULT_M2 if args.m2 is None else args.m2
    k = DEFAULT_K if args.k is None else args.k
    left_end_mass = 1e-3 * args.left_end_mass_g
    right_end_mass = 1e-3 * args.right_end_mass_g

    if args.only is None:
        selected_panels = [p for p in PANEL_CHOICES if p != "eigenvectors"]
    elif "none" in args.only:
        selected_panels = []
    else:
        selected_panels = args.only

    result = build_custom_chain(
        interior_N=interior_N,
        m1=m1,
        m2=m2,
        k=k,
        L=args.L,
        switch=args.switch,
        disorder=disorder_strength,
        mass_disorder_ratio=args.mass_disorder_ratio,
        seed=args.seed,
        left_end_mass=left_end_mass,
        right_end_mass=right_end_mass,
    )
    N = result.N
    lam = result.lam
    V = result.V
    freq = result.freq
    masses = result.masses
    springs = result.springs
    E_full = result.E_full
    E_red = result.E_red

    print(f"Disorder seed: {args.seed}")
    print(f"Disorder strength: +/-{args.disorder:.1f}%")
    print(
        "Endpoint masses [g]: "
        f"left={1000.0 * result.left_end_mass:.6g}, "
        f"right={1000.0 * result.right_end_mass:.6g}"
    )
    print(f"Interior sites: {result.interior_N}")
    print(f"Total sites including endpoints: {N}")
    print("Disordered masses:")
    print(masses)
    print("Disordered springs:")
    print(springs)

    print("Disordered frequencies [Hz]:")
    for i, (lam_i, f_i) in enumerate(zip(lam, freq), start=1):
        print(f"mode {i:2d}: lambda = {lam_i:.12g},  f = {f_i:.12g} Hz")

    print(
        "\nTransformed spectrum comparison skipped: "
        "the extra endpoint masses break the pure alternating m1/m2 chain assumption."
    )

    lineplot_max_freq = args.lineplot_max_freq
    if lineplot_max_freq is None:
        max_freq = float(np.max(freq)) if len(freq) else 0.0
        lineplot_max_freq = 1.05 * max_freq if max_freq > 0 else 1.0

    if selected_panels:
        fig, axes = make_axes(len(selected_panels))

        for ax, panel in zip(axes, selected_panels):

            if panel == "freqs":
                ax.scatter(np.arange(1, N + 1), freq, s=40, c='k')
                ax.set_xlabel("Mode number")
                ax.set_ylabel("Frequency [Hz]")
                ax.set_title("Disordered mass-spring chain with custom end masses")

            elif panel == "eigenvectors":

                spacing = 1.5
                colors = ["black", "maroon"]

                for i in range(N):
                    v = V[:, i].copy()
                    vmax = np.max(np.abs(v))
                    if vmax > 0:
                        v = v / vmax

                    offset = (i + 1) * spacing
                    color = colors[i % 2]

                    ax.plot(np.arange(1, N + 1), v + offset,
                            '.-', color=color, linewidth=1)

                ax.set_xlabel("Site")
                ax.set_ylabel("Normalized mode + offset")
                ax.set_title("Disordered eigenvectors with extra endpoint sites")

            elif panel == "thesis_transformed":
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "Transformed-spectrum comparison\nis not defined for the\ncustom endpoint-mass chain.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            elif panel == "qspace":
                        # Keep adjacent normalized q-space traces from touching while
                        # avoiding more vertical space than necessary.
                        spacing = 1.05 if args.abs_amplitude else 2.05
                        colors = ["steelblue", "gray"]
                        q_indices = np.arange(1, N)
                        
                        # The matrix that converts displacements (y) to bond stretches (q)
                        D = difference_operator(N)
                        
                        for i in range(N):
                            # 1. Get the physical displacement eigenvector
                            y_vec = V[:, i].copy()
                            
                            # 2. Convert to bond stretches (q_n = y_n - y_{n+1})
                            q_vec = D @ y_vec
                            
                            # 3. Normalize for visualization
                            qmax = np.max(np.abs(q_vec))
                            if qmax > 1e-12:
                                q_vec = q_vec / qmax
                            else:
                                q_vec = np.zeros_like(q_vec)

                            if args.abs_amplitude:
                                q_vec = np.abs(q_vec)

                            if args.neg:
                                q_vec = -q_vec
                                
                            offset = (i + 1) * spacing

                            if args.show_zero:
                                ax.plot(
                                    q_indices,
                                    np.full(N - 1, offset),
                                    linestyle=':',
                                    color='lightgray',
                                    linewidth=0.8,
                                    zorder=0,
                                )
                            
                            ax.plot(q_indices, q_vec + offset, '.-', color=colors[i % 2], linewidth=1)

                        ax.set_xlabel("Bond index $n$")
                        ax.set_ylabel("Normalized Bond Stretch + offset")
                        ax.set_title("All Modes in q-space (Bond Stretches)")

            elif panel == "lineplot":

                img = build_lineplot_image(freq, lineplot_max_freq, N)

                extent = [0.5, N + 0.5, 0.0, lineplot_max_freq]

                ax.imshow(
                    img,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap="jet",
                    interpolation="nearest",
                    vmin=-1,
                    vmax=1
                )

                ax.set_xlabel("Arbitrary index")
                ax.set_ylabel("Frequency [Hz]")
                ax.set_title("Eigenfrequency image plot")

        plt.tight_layout()
        plt.show()

    print("\nInterpretation:")
    if result.m1 < result.m2:
        print("m1 < m2: this is the thesis' topological ordering if the chain starts with m1.")
    elif result.m1 > result.m2:
        print("m1 > m2: this is the thesis' non-topological ordering if the chain starts with m1.")
    else:
        print("m1 == m2: gap closes; this is the critical case.")


if __name__ == "__main__":
    main()
