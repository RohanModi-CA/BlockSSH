import argparse

import matplotlib.pyplot as plt
import numpy as np

from helpers.dgnic import (
    DEFAULT_K,
    DEFAULT_M1,
    DEFAULT_M2,
    DEFAULT_N,
    DEFAULT_L,
    DISORDER_SEED,
    difference_operator,
    solve_chain,
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


def main():
    args = parse_args()
    disorder_strength = args.disorder / 100.0
    N = args.N
    m1 = DEFAULT_M1 if args.m1 is None else args.m1
    m2 = DEFAULT_M2 if args.m2 is None else args.m2
    k = DEFAULT_K if args.k is None else args.k

    if args.only is None:
        selected_panels = [p for p in PANEL_CHOICES if p != "eigenvectors"]
    elif "none" in args.only:
        selected_panels = []
    else:
        selected_panels = args.only

    result = solve_chain(
        N=N,
        m1=m1,
        m2=m2,
        k=k,
        L=args.L,
        switch=args.switch,
        disorder=disorder_strength,
        mass_disorder_ratio=args.mass_disorder_ratio,
        seed=args.seed,
    )
    lam = result.lam
    V = result.V
    freq = result.freq
    masses = result.masses
    springs = result.springs
    E_full = result.E_full
    E_red = result.E_red

    print(f"Disorder seed: {args.seed}")
    print(f"Disorder strength: +/-{args.disorder:.1f}%")
    print("Disordered masses:")
    print(masses)
    print("Disordered springs:")
    print(springs)

    print("Disordered frequencies [Hz]:")
    for i, (lam_i, f_i) in enumerate(zip(lam, freq), start=1):
        print(f"mode {i:2d}: lambda = {lam_i:.12g},  f = {f_i:.12g} Hz")

    print("\nTransformed full-chain eigenvalues E_full:")
    print(E_full)

    print("\nReduced SSH-like eigenvalues E_red:")
    print(E_red)

    zero_idx = np.argmin(np.abs(E_full))
    E_full_nozero = np.delete(E_full, zero_idx)

    print("\nMax |E_full_nozero - E_red|:",
          np.max(np.abs(np.sort(E_full_nozero) - np.sort(E_red))))

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
                ax.set_title("Disordered mass-spring chain")

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
                ax.set_title("Disordered eigenvectors")

            elif panel == "thesis_transformed":
                ax.plot(np.arange(1, N + 1), E_full, 'o', label='full transformed')
                ax.plot(np.arange(1, N), E_red, 'x', label='reduced SSH-like')
                ax.set_xlabel("Sorted eigenvalue index")
                ax.set_ylabel("E")
                ax.set_title("Thesis transformed spectrum")
                ax.legend()

            elif panel == "qspace":
                        # Keep adjacent normalized q-space traces from touching while
                        # avoiding more vertical space than necessary.
                        spacing = 1.05 if args.abs_amplitude else 2.05
                        colors = ["steelblue", "gray"]
                        q_indices = np.arange(1, N)
                        
                        # The matrix that converts displacements (y) to bond stretches (q)
                        D = difference_operator(N)
                        
                        # Theoretical mechanical eigenvalue for the edge state: k(m1+m2)/(m1*m2)
                        target_lam = result.k * (result.m1 + result.m2) / (result.m1 * result.m2)
                        
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
                            
                            # Highlight the edge state(s) in red! 
                            # (If its eigenvalue is very close to the theoretical mid-gap value)
                            is_edge = np.abs(lam[i] - target_lam) < (0.05 * target_lam)
                            
                            if is_edge:
                                ax.plot(q_indices, q_vec + offset, 'o-', color='red', linewidth=2.5, zorder=10)
                            else:
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
