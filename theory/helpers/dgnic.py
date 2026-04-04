from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DISORDER_SEED = 7
DEFAULT_G = 9.81
DEFAULT_L = 59 * 0.0254
DEFAULT_N = 10
DEFAULT_M1 = 28e-3
DEFAULT_M2 = 60e-3
DEFAULT_K = 194.0


@dataclass(frozen=True)
class ChainResult:
    H: np.ndarray
    masses: np.ndarray
    springs: np.ndarray
    lam: np.ndarray
    V: np.ndarray
    freq: np.ndarray
    H_tilde: np.ndarray
    E_full: np.ndarray
    W_full: np.ndarray
    E_red: np.ndarray
    U_red: np.ndarray
    m1: float
    m2: float
    k: float
    g: float
    L: float
    N: int
    disorder: float
    seed: int
    switched: bool


def build_mass_pattern(N: int, m1: float, m2: float) -> np.ndarray:
    masses = np.full(N, m1, dtype=float)
    masses[1::2] = m2
    return masses


def build_spring_pattern(N: int, k: float) -> np.ndarray:
    return np.full(N - 1, k, dtype=float)


def apply_uniform_disorder(values: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    delta = rng.uniform(-strength, strength, size=np.shape(values))
    return np.asarray(values, dtype=float) * (1.0 + delta)


def build_original_dynamical_matrix(
    N: int,
    k: float,
    m1: float,
    m2: float,
    *,
    g: float = DEFAULT_G,
    L: float = DEFAULT_L,
) -> tuple[np.ndarray, np.ndarray]:
    masses = build_mass_pattern(N, m1, m2)
    H = np.zeros((N, N), dtype=float)
    omega_p_sq = g / L

    for i in range(N):
        if i == 0 or i == N - 1:
            diag_spring = k / masses[i]
        else:
            diag_spring = 2.0 * k / masses[i]

        H[i, i] = diag_spring + omega_p_sq

        if i > 0:
            H[i, i - 1] = -k / masses[i]
        if i < N - 1:
            H[i, i + 1] = -k / masses[i]

    return H, masses


def build_disordered_dynamical_matrix(
    N: int,
    k: float,
    m1: float,
    m2: float,
    *,
    rng: np.random.Generator,
    disorder_strength: float,
    mass_disorder_ratio: float = 1.0,
    g: float = DEFAULT_G,
    L: float = DEFAULT_L,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_masses = build_mass_pattern(N, m1, m2)
    base_springs = build_spring_pattern(N, k)

    masses = apply_uniform_disorder(base_masses, disorder_strength * mass_disorder_ratio, rng)
    springs = apply_uniform_disorder(base_springs, disorder_strength, rng)

    H = np.zeros((N, N), dtype=float)
    omega_p_sq = g / L

    for i in range(N):
        k_left = springs[i - 1] if i > 0 else 0.0
        k_right = springs[i] if i < N - 1 else 0.0
        H[i, i] = (k_left + k_right) / masses[i] + omega_p_sq

        if i > 0:
            H[i, i - 1] = -k_left / masses[i]
        if i < N - 1:
            H[i, i + 1] = -k_right / masses[i]

    return H, masses, springs


def build_transformed_matrix_from_H(H: np.ndarray, k: float, m1: float, m2: float) -> np.ndarray:
    return (m1 * m2 / k) * H - (m1 + m2) * np.eye(H.shape[0])


def build_ssh_like_reduced_matrix(N: int, m1: float, m2: float) -> np.ndarray:
    M = np.zeros((N - 1, N - 1), dtype=float)
    for i in range(N - 2):
        hop = m1 if (i % 2 == 0) else m2
        M[i, i + 1] = hop
        M[i + 1, i] = hop
    return M


def difference_operator(N: int) -> np.ndarray:
    D = np.zeros((N - 1, N), dtype=float)
    for i in range(N - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    return D


def sort_eigensystem(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eig(A)
    vals = np.real_if_close(vals, tol=1000)
    vecs = np.real_if_close(vecs, tol=1000)

    idx = np.argsort(np.real(vals))
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    return vals, vecs


def solve_chain(
    *,
    N: int = DEFAULT_N,
    m1: float = DEFAULT_M1,
    m2: float = DEFAULT_M2,
    k: float = DEFAULT_K,
    g: float = DEFAULT_G,
    L: float = DEFAULT_L,
    switch: bool = False,
    disorder: float = 0.0,
    mass_disorder_ratio: float = 1.0,
    seed: int = DISORDER_SEED,
) -> ChainResult:
    if N < 2:
        raise ValueError("N must be at least 2")
    if disorder < 0:
        raise ValueError("disorder must be non-negative")
    if mass_disorder_ratio < 0:
        raise ValueError("mass_disorder_ratio must be non-negative")
    if min(m1, m2, k, g, L) <= 0:
        raise ValueError("m1, m2, k, g, and L must be positive")

    if switch:
        m1, m2 = m2, m1

    rng = np.random.default_rng(seed)
    if disorder > 0:
        H, masses, springs = build_disordered_dynamical_matrix(
            N,
            k,
            m1,
            m2,
            rng=rng,
            disorder_strength=disorder,
            mass_disorder_ratio=mass_disorder_ratio,
            g=g,
            L=L,
        )
    else:
        H, masses = build_original_dynamical_matrix(N, k, m1, m2, g=g, L=L)
        springs = build_spring_pattern(N, k)

    lam, V = sort_eigensystem(H)
    lam[np.abs(lam) < 1e-14] = 0.0
    omega = np.sqrt(np.clip(lam, 0.0, None))
    freq = omega / (2.0 * np.pi)

    H_tilde = build_transformed_matrix_from_H(H, k, m1, m2)
    E_full, W_full = sort_eigensystem(H_tilde)

    ssh_like = build_ssh_like_reduced_matrix(N, m1, m2)
    E_red, U_red = sort_eigensystem(ssh_like)

    return ChainResult(
        H=H,
        masses=masses,
        springs=springs,
        lam=lam,
        V=V,
        freq=freq,
        H_tilde=H_tilde,
        E_full=E_full,
        W_full=W_full,
        E_red=E_red,
        U_red=U_red,
        m1=m1,
        m2=m2,
        k=k,
        g=g,
        L=L,
        N=N,
        disorder=disorder,
        seed=seed,
        switched=switch,
    )


def predicted_frequencies(**kwargs: float) -> np.ndarray:
    return solve_chain(**kwargs).freq


def select_frequencies(
    freq: np.ndarray,
    *,
    skip_lowest: int = 0,
    count: int | None = None,
) -> np.ndarray:
    selected = np.asarray(freq, dtype=float)
    if skip_lowest < 0:
        raise ValueError("skip_lowest must be non-negative")
    if skip_lowest >= selected.size:
        raise ValueError("skip_lowest removes all modes")
    selected = selected[skip_lowest:]
    if count is not None:
        if count < 1:
            raise ValueError("count must be at least 1")
        if count > selected.size:
            raise ValueError("count exceeds available modes after skipping")
        selected = selected[:count]
    return selected


def load_peaks_csv(path: str | Path) -> np.ndarray:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Peaks file not found: {csv_path}")

    peaks: list[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if value > 0:
                    peaks.append(value)

    if not peaks:
        raise ValueError(f"No valid positive peaks found in {csv_path}")

    peaks_arr = np.asarray(peaks, dtype=float)
    if np.any(np.diff(peaks_arr) <= 0):
        raise ValueError("Peak CSV must be strictly increasing")
    return peaks_arr
