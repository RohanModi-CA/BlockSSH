import argparse
import numpy as np
import matplotlib.pyplot as plt


PANEL_CHOICES = ["freqs", "eigenvectors", "qspace", "thesis_transformed", "lineplot"]


def build_mass_pattern(N, m1, m2):
    """
    Mass pattern [m1, m2, m1, m2, ...] of length N.
    """
    m = np.full(N, m1, dtype=float)
    m[1::2] = m2
    return m


def build_original_dynamical_matrix_old(N, k, m1, m2):
    """
    Original displacement-space dynamical matrix H from the mass-spring chain:
        y_ddot = -H y
    """
    m = build_mass_pattern(N, m1, m2)
    H = np.zeros((N, N), dtype=float)

    for i in range(N):
        if i == 0 or i == N - 1:
            H[i, i] = k / m[i]
        else:
            H[i, i] = 2.0 * k / m[i]

        if i > 0:
            H[i, i - 1] = -k / m[i]
        if i < N - 1:
            H[i, i + 1] = -k / m[i]

    return H, m

def build_original_dynamical_matrix(N, k, m1, m2, g=9.81, L=59*.0254):
    """
    H_new = H_old + (g/L)*I
    """
    m = build_mass_pattern(N, m1, m2)
    H = np.zeros((N, N), dtype=float)
    
    omega_p_sq = g / L  # Pendulum contribution

    for i in range(N):
        # Standard Spring terms
        if i == 0 or i == N - 1:
            diag_spring = k / m[i]
        else:
            diag_spring = 2.0 * k / m[i]
        
        # Add the pendulum term to the diagonal
        H[i, i] = diag_spring + omega_p_sq

        if i > 0:
            H[i, i - 1] = -k / m[i]
        if i < N - 1:
            H[i, i + 1] = -k / m[i]

    return H, m

def build_transformed_matrix_from_H(H, k, m1, m2):
    """
    Thesis finite-chain transformed matrix:
        H_tilde = (m1*m2/k) * H - (m1 + m2) * I
    """
    N = H.shape[0]
    return (m1 * m2 / k) * H - (m1 + m2) * np.eye(N)


def build_ssh_like_reduced_matrix(N, m1, m2):
    """
    Reduced SSH-like matrix acting on q_n = y_n - y_{n+1}, size (N-1)x(N-1).
    """
    M = np.zeros((N - 1, N - 1), dtype=float)

    for i in range(N - 2):
        hop = m1 if (i % 2 == 0) else m2
        M[i, i + 1] = hop
        M[i + 1, i] = hop

    return M


def difference_operator(N):
    D = np.zeros((N - 1, N), dtype=float)
    for i in range(N - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    return D


def sort_eigensystem(A):
    vals, vecs = np.linalg.eig(A)
    vals = np.real_if_close(vals, tol=1000)
    vecs = np.real_if_close(vecs, tol=1000)

    idx = np.argsort(np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])

    return vals, vecs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mass-spring / transformed SSH-like chain visualizer."
    )
    parser.add_argument("--m1", type=float, default=None)
    parser.add_argument("--m2", type=float, default=None)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--switch", action="store_true")
    parser.add_argument("--N", type=int, default=11)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PANEL_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--lineplot_max_freq",
        type=float,
        default=None,
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

    k = 194

    m1 = 12.5 / (1000.0 ** 2)
    m2 = 20.4 / (1000.0 ** 2)
    m1=37e-3
    m2=74e-3

    if args.N is not None:
        N = args.N
    else:
        raise ValueError()

    if args.m1 is not None:
        m1 = args.m1
    if args.m2 is not None:
        m2 = args.m2
    if args.k is not None:
        k = args.k

    if args.switch:
        m1, m2 = m2, m1

    selected_panels = PANEL_CHOICES if args.only is None else args.only

    H, masses = build_original_dynamical_matrix(N, k, m1, m2)
    lam, V = sort_eigensystem(H)

    lam[np.abs(lam) < 1e-14] = 0.0

    omega = np.sqrt(np.clip(lam, 0.0, None))
    freq = omega / (2.0 * np.pi)

    H_tilde = build_transformed_matrix_from_H(H, k, m1, m2)
    E_full, W_full = sort_eigensystem(H_tilde)

    SSH_like = build_ssh_like_reduced_matrix(N, m1, m2)
    E_red, U_red = sort_eigensystem(SSH_like)

    print("Original frequencies [Hz]:")
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

    fig, axes = make_axes(len(selected_panels))

    lineplot_max_freq = args.lineplot_max_freq
    if lineplot_max_freq is None:
        max_freq = float(np.max(freq)) if len(freq) else 0.0
        lineplot_max_freq = 1.05 * max_freq if max_freq > 0 else 1.0

    for ax, panel in zip(axes, selected_panels):

        if panel == "freqs":
            ax.scatter(np.arange(1, N + 1), freq, s=40, c='k')
            ax.set_xlabel("Mode number")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_title("Original mass-spring chain")

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
            ax.set_title("Original eigenvectors")

        elif panel == "thesis_transformed":
            ax.plot(np.arange(1, N + 1), E_full, 'o', label='full transformed')
            ax.plot(np.arange(1, N), E_red, 'x', label='reduced SSH-like')
            ax.set_xlabel("Sorted eigenvalue index")
            ax.set_ylabel("E")
            ax.set_title("Thesis transformed spectrum")
            ax.legend()

        elif panel == "qspace":
                    spacing = 1.5
                    colors = ["steelblue", "gray"]
                    
                    # The matrix that converts displacements (y) to bond stretches (q)
                    D = difference_operator(N)
                    
                    # Theoretical mechanical eigenvalue for the edge state: k(m1+m2)/(m1*m2)
                    target_lam = k * (m1 + m2) / (m1 * m2)
                    
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
                            
                        offset = (i + 1) * spacing
                        
                        # Highlight the edge state(s) in red! 
                        # (If its eigenvalue is very close to the theoretical mid-gap value)
                        is_edge = np.abs(lam[i] - target_lam) < (0.05 * target_lam)
                        
                        if is_edge:
                            ax.plot(np.arange(1, N), q_vec + offset, 'o-', color='red', linewidth=2.5, zorder=10)
                        else:
                            ax.plot(np.arange(1, N), q_vec + offset, '.-', color=colors[i % 2], linewidth=1)

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
    if m1 < m2:
        print("m1 < m2: this is the thesis' topological ordering if the chain starts with m1.")
    elif m1 > m2:
        print("m1 > m2: this is the thesis' non-topological ordering if the chain starts with m1.")
    else:
        print("m1 == m2: gap closes; this is the critical case.")


if __name__ == "__main__":
    main()
