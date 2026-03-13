import numpy as np
import matplotlib.pyplot as plt


def build_mass_pattern(N, m1, m2):
    """
    Mass pattern [m1, m2, m1, m2, ...] of length N.
    """
    m = np.full(N, m1, dtype=float)
    m[1::2] = m2
    return m


def build_original_dynamical_matrix(N, k, m1, m2):
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


def build_transformed_matrix_from_H(H, k, m1, m2):
    """
    Thesis finite-chain transformed matrix:
        H_tilde = (m1*m2/k) * H - (m1 + m2) * I

    This is the general-k version of thesis Eq. 4.6.
    """
    N = H.shape[0]
    return (m1 * m2 / k) * H - (m1 + m2) * np.eye(N)


def build_ssh_like_reduced_matrix(N, m1, m2):
    """
    Reduced SSH-like matrix acting on q_n = y_n - y_{n+1}, size (N-1)x(N-1).

    Its diagonal is zero, and its off-diagonals alternate:
        m1, m2, m1, m2, ...

    This matches thesis Eq. 4.8 for a chain starting with mass m1 at site 1.
    """
    M = np.zeros((N - 1, N - 1), dtype=float)

    for i in range(N - 2):
        hop = m1 if (i % 2 == 0) else m2
        M[i, i + 1] = hop
        M[i + 1, i] = hop

    return M


def difference_operator(N):
    """
    D maps y -> q where q_n = y_n - y_{n+1}.
    Shape: (N-1, N)
    """
    D = np.zeros((N - 1, N), dtype=float)
    for i in range(N - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    return D


def sort_eigensystem(A):
    """
    Returns eigenvalues/eigenvectors sorted ascending by eigenvalue.
    Assumes A is real symmetric or nearly so.
    """
    vals, vecs = np.linalg.eig(A)
    vals = np.real_if_close(vals, tol=1000)
    vecs = np.real_if_close(vecs, tol=1000)

    idx = np.argsort(np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])

    return vals, vecs


def main():
    # ------------------------------------------------------------
    # Your constants
    # ------------------------------------------------------------
    N = 11
    k = 0.0056


    # These look like inertias in kg*m^2, not translational masses.
    # Kept exactly as provided.
    m1 = 12.5 / (1000.0 ** 2)
    m2 = 20.4 / (1000.0 ** 2)

    # ------------------------------------------------------------
    # Original model
    # ------------------------------------------------------------
    H, masses = build_original_dynamical_matrix(N, k, m1, m2)
    lam, V = sort_eigensystem(H)

    # Clean tiny negatives from roundoff
    lam[np.abs(lam) < 1e-14] = 0.0
    if np.any(lam < -1e-10):
        print("Warning: significantly negative eigenvalues found in H.")

    omega = np.sqrt(np.clip(lam, 0.0, None))
    freq = omega / (2.0 * np.pi)

    # ------------------------------------------------------------
    # Thesis transformed model
    # ------------------------------------------------------------
    H_tilde = build_transformed_matrix_from_H(H, k, m1, m2)
    E_full, W_full = sort_eigensystem(H_tilde)

    # Reduced SSH-like model on q_n = y_n - y_{n+1}
    SSH_like = build_ssh_like_reduced_matrix(N, m1, m2)
    E_red, U_red = sort_eigensystem(SSH_like)

    # Verify thesis equivalence numerically:
    # The transformed full matrix should have the reduced SSH-like spectrum
    # plus one extra eigenvalue near 0 coming from the rigid-translation mode,
    # because q = D y removes the uniform displacement.
    print("Original frequencies [Hz]:")
    for i, (lam_i, f_i) in enumerate(zip(lam, freq), start=1):
        print(f"mode {i:2d}: lambda = {lam_i:.12g},  f = {f_i:.12g} Hz")

    print("\nTransformed full-chain eigenvalues E_full:")
    print(E_full)

    print("\nReduced SSH-like eigenvalues E_red:")
    print(E_red)

    # Try to match spectra up to the extra mode
    # Remove the eigenvalue closest to zero from the full transformed spectrum
    zero_idx = np.argmin(np.abs(E_full))
    E_full_nozero = np.delete(E_full, zero_idx)

    print("\nMax |E_full_nozero - E_red|:",
          np.max(np.abs(np.sort(E_full_nozero) - np.sort(E_red))))

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Original frequencies
    ax = axes[0, 0]

    ax.scatter(np.arange(1, N + 1), freq, s=40, c='k', marker='o')

    ax.set_xlabel("Mode number")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Original mass-spring chain")

    # Original mode shapes
    ax = axes[0, 1]
    for i in range(N):
        v = V[:, i].copy()
        vmax = np.max(np.abs(v))
        if vmax > 0:
            v = v / vmax
        ax.plot(np.arange(1, N + 1), v + (i + 1), '.-k', linewidth=1)
    ax.set_xlabel("Site")
    ax.set_ylabel("Normalized mode + offset")
    ax.set_title("Original eigenvectors")

    # Transformed spectra comparison
    ax = axes[1, 0]
    ax.plot(np.arange(1, N + 1), E_full, 'o', label='full transformed')
    ax.plot(np.arange(1, N), E_red, 'x', label='reduced SSH-like')
    ax.set_xlabel("Sorted eigenvalue index")
    ax.set_ylabel("E")
    ax.set_title("Thesis transformed spectrum")
    ax.legend()

    # Central SSH-like mode in q-space
    ax = axes[1, 1]
    mid = (N - 1) // 2
    u = U_red[:, mid].copy()
    umax = np.max(np.abs(u))
    if umax > 0:
        u = u / umax
    ax.plot(np.arange(1, N), u, '.-k')
    ax.set_xlabel("Bond index n  (q_n = y_n - y_{n+1})")
    ax.set_ylabel("Amplitude")
    ax.set_title("Representative SSH-like mode in q-space")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------
    print("\nInterpretation:")
    if m1 < m2:
        print("m1 < m2: this is the thesis' topological ordering if the chain starts with m1.")
    elif m1 > m2:
        print("m1 > m2: this is the thesis' non-topological ordering if the chain starts with m1.")
    else:
        print("m1 == m2: gap closes; this is the critical case.")


if __name__ == "__main__":
    main()
