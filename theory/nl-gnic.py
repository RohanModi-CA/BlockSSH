import argparse
import numpy as np
import matplotlib.pyplot as plt


PANEL_CHOICES = [
    "freqs",
    "eigenvectors",
    "qspace",
    "thesis_transformed",
    "lineplot",
    "timeseries",
    "fft_spectrum",
    "fft_image",
]
DISORDER_SEED = 7
DISORDER_STRENGTH = 0.10
DEFAULT_L_INCHES = 59.0
DEFAULT_MAX_DISP_INCHES = 5.0
DEFAULT_BLOCK_SPACING_INCHES = 2.0
DEFAULT_KICK_INTERVAL = 45.0
DEFAULT_TMAX = 800.0
DEFAULT_DT_INTERNAL = 1.0 / 600.0
DEFAULT_SAMPLE_RATE = 60.0
DEFAULT_DECAY_TIME = 30.0
DEFAULT_DECAY_FRACTION = 1e-3
DEFAULT_WELCH_SEGMENT = 100.0
DEFAULT_WELCH_OVERLAP = 0.5
DEFAULT_IMAGE_COLS = 200
DEFAULT_NONLINEAR_STRENGTH = 1.0
DEFAULT_DRIVE_SITE = "first"
DEFAULT_DRIVE_TIME_RANDOMNESS = 0.8
DEFAULT_FORCE_PEAK_N = 10.0
DEFAULT_CONTACT_MS = 1.0


def inches_to_meters(x):
    return 0.0254 * x


def build_mass_pattern(N, m1, m2):
    m = np.full(N, m1, dtype=float)
    m[1::2] = m2
    return m


def apply_uniform_disorder(values, strength, rng):
    delta = rng.uniform(-strength, strength, size=np.shape(values))
    return np.asarray(values, dtype=float) * (1.0 + delta)


def build_spring_pattern(N, k):
    return np.full(N - 1, k, dtype=float)


def build_original_dynamical_matrix(N, k, m1, m2, g=9.81, L=DEFAULT_L_INCHES * 0.0254):
    m = build_mass_pattern(N, m1, m2)
    H = np.zeros((N, N), dtype=float)
    omega_p_sq = g / L

    for i in range(N):
        if i == 0 or i == N - 1:
            diag_spring = k / m[i]
        else:
            diag_spring = 2.0 * k / m[i]

        H[i, i] = diag_spring + omega_p_sq

        if i > 0:
            H[i, i - 1] = -k / m[i]
        if i < N - 1:
            H[i, i + 1] = -k / m[i]

    return H, m


def build_disordered_dynamical_matrix(
    N, k, m1, m2, rng, disorder_strength, g=9.81, L=DEFAULT_L_INCHES * 0.0254
):
    base_masses = build_mass_pattern(N, m1, m2)
    base_springs = build_spring_pattern(N, k)

    masses = apply_uniform_disorder(base_masses, disorder_strength, rng)
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


def build_transformed_matrix_from_H(H, k, m1, m2):
    N = H.shape[0]
    return (m1 * m2 / k) * H - (m1 + m2) * np.eye(N)


def build_ssh_like_reduced_matrix(N, m1, m2):
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


def make_axes(n_panels):
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
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


def build_spectrum_image_db(freq, psd_db, fmax, n_cols, n_rows=800):
    img = np.full((n_rows, n_cols), np.min(psd_db) if len(psd_db) else -18.0, dtype=float)
    if fmax <= 0 or len(freq) == 0:
        return img

    row_coords = np.linspace(0.0, fmax, n_rows)
    column = np.interp(row_coords, freq, psd_db, left=psd_db[0], right=psd_db[-1])
    img[:, :] = column[:, None]
    return img


def damping_gamma_from_decay(decay_time, decay_fraction):
    """
    For x'' + gamma x' + ... = 0, the underdamped amplitude envelope is exp(-gamma t / 2).
    """
    return -2.0 * np.log(decay_fraction) / decay_time


def gaussian_impulse_from_peak_force(force_peak_N, contact_ms):
    """
    Convert a Gaussian contact with the given peak force and FWHM duration
    into its impulse integral.
    """
    sigma = (contact_ms * 1e-3) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return force_peak_N * sigma * np.sqrt(2.0 * np.pi)


def build_nonlinear_contact_arrays(
    N,
    rng,
    contact_model,
    contact_k3,
    contact_gap,
    contact_kh,
    contact_disorder,
):
    """
    Build bond-wise nonlinear coefficients. These are separate from the linear springs.
    """
    n_bonds = N - 1
    k3 = np.full(n_bonds, contact_k3, dtype=float)
    gap = np.full(n_bonds, contact_gap, dtype=float)
    kh = np.full(n_bonds, contact_kh, dtype=float)

    if n_bonds > 0 and contact_model != "off" and contact_disorder > 0:
        strength = contact_disorder / 100.0

        if np.any(k3 != 0.0):
            k3 = apply_uniform_disorder(k3, strength, rng)
        if np.any(kh != 0.0):
            kh = apply_uniform_disorder(kh, strength, rng)
        if np.any(gap != 0.0):
            gap = np.maximum(0.0, apply_uniform_disorder(gap, strength, rng))

    return k3, gap, kh


def compute_onsite_cubic_accel(pos, L, onsite_nonlinear_strength, disable_onsite_cubic):
    """
    Optional legacy Duffing-like onsite term kept only for backward compatibility.
    """
    if disable_onsite_cubic or onsite_nonlinear_strength == 0.0:
        return np.zeros_like(pos)

    cubic_coeff = onsite_nonlinear_strength * 9.81 / (6.0 * L ** 3)
    return cubic_coeff * pos ** 3


def compute_nonlinear_contact_accel(
    pos,
    masses,
    contact_model,
    contact_k3,
    contact_gap,
    contact_kh,
    contact_exponent,
):
    """
    Bond/contact nonlinearity depends on relative displacement delta_i = x_{i+1} - x_i.
    Unlike an onsite Duffing term, this acts through equal-and-opposite bond forces.

    Sign convention:
    - delta_i > 0 means site i+1 is displaced to the right of site i.
    - A positive bond force acts to increase site i and decrease site i+1,
      which is restoring because it reduces delta_i.

    Hertz-gap engagement rule:
    - Use a symmetric magnitude-based contact law.
    - Contact engages only when |delta_i| exceeds gap_i.
    - The restoring force direction follows sign(delta_i).
    """
    if contact_model == "off" or len(pos) < 2:
        return np.zeros_like(pos)

    delta = pos[1:] - pos[:-1]

    if contact_model == "cubic":
        bond_force = contact_k3 * delta ** 3
    elif contact_model == "hertz_gap":
        compression = np.maximum(0.0, np.abs(delta) - contact_gap)
        bond_force = contact_kh * compression ** contact_exponent * np.sign(delta)
    else:
        raise ValueError(f"Unknown contact model: {contact_model}")

    accel = np.zeros_like(pos)
    accel[:-1] += bond_force / masses[:-1]
    accel[1:] -= bond_force / masses[1:]
    return accel


def rk4_step(y, v, dt, acceleration_fn):
    k1y = v
    k1v = acceleration_fn(y, v)

    y2 = y + 0.5 * dt * k1y
    v2 = v + 0.5 * dt * k1v
    k2y = v2
    k2v = acceleration_fn(y2, v2)

    y3 = y + 0.5 * dt * k2y
    v3 = v + 0.5 * dt * k2v
    k3y = v3
    k3v = acceleration_fn(y3, v3)

    y4 = y + dt * k3y
    v4 = v + dt * k3v
    k4y = v4
    k4v = acceleration_fn(y4, v4)

    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return y_next, v_next


def pick_drive_site(site_spec, N, rng):
    if site_spec == "first":
        return 0
    if site_spec == "last":
        return N - 1
    if site_spec == "random":
        return int(rng.integers(0, N))
    site = int(site_spec)
    if not (0 <= site < N):
        raise ValueError(f"Drive site index must be between 0 and {N - 1}")
    return site


def build_kick_schedule(
    n_steps,
    dt,
    interval,
    N,
    masses,
    force_peak_N,
    contact_ms,
    rng,
    drive_site,
    time_randomness,
):
    kicks = np.zeros((n_steps, N), dtype=float)
    if interval <= 0 or force_peak_N <= 0 or contact_ms <= 0:
        return kicks

    min_factor = max(0.0, 1.0 - time_randomness)
    max_factor = 1.0 + time_randomness
    base_impulse = gaussian_impulse_from_peak_force(force_peak_N, contact_ms)

    event_time = 0.0
    while event_time < n_steps * dt:
        step = int(round(event_time / dt))
        if step >= n_steps:
            break

        site = pick_drive_site(drive_site, N, rng)
        impulse = base_impulse * rng.uniform(0.5, 1.0)
        amplitude = (impulse / masses[site]) * rng.choice([-1.0, 1.0])
        kicks[step, site] += amplitude

        jitter_factor = rng.uniform(min_factor, max_factor)
        event_time += interval * jitter_factor

    return kicks


def simulate_nonlinear_chain(
    H_linear,
    masses,
    L,
    rng,
    tmax,
    dt_internal,
    sample_rate,
    kick_interval,
    force_peak_N,
    contact_ms,
    damping_gamma,
    disable_onsite_cubic,
    onsite_nonlinear_strength,
    contact_model,
    contact_k3,
    contact_gap,
    contact_kh,
    contact_exponent,
    drive_site,
    drive_time_randomness,
):
    n_steps = int(np.floor(tmax / dt_internal)) + 1
    if n_steps < 2:
        raise ValueError("Need at least two time steps for simulation")
    if sample_rate <= 0:
        raise ValueError("--sample_rate must be positive")

    N = H_linear.shape[0]
    y = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)
    kicks = build_kick_schedule(
        n_steps,
        dt_internal,
        kick_interval,
        N,
        masses,
        force_peak_N,
        contact_ms,
        rng,
        drive_site,
        drive_time_randomness,
    )

    sample_interval = 1.0 / sample_rate
    sample_step = max(1, int(round(sample_interval / dt_internal)))
    n_samples = (n_steps - 1) // sample_step + 1
    times = np.zeros(n_samples, dtype=float)
    traj = np.zeros((n_samples, N), dtype=float)

    def acceleration(pos, vel):
        linear_accel = -(H_linear @ pos)
        onsite_accel = compute_onsite_cubic_accel(
            pos, L, onsite_nonlinear_strength, disable_onsite_cubic
        )
        contact_accel = compute_nonlinear_contact_accel(
            pos,
            masses,
            contact_model,
            contact_k3,
            contact_gap,
            contact_kh,
            contact_exponent,
        )
        return linear_accel - damping_gamma * vel + onsite_accel + contact_accel

    sample_idx = 0
    v += kicks[0]
    times[sample_idx] = 0.0
    traj[sample_idx] = y
    sample_idx += 1

    for step in range(1, n_steps):
        y, v = rk4_step(y, v, dt_internal, acceleration)
        v += kicks[step]

        if step % sample_step == 0:
            times[sample_idx] = step * dt_internal
            traj[sample_idx] = y
            sample_idx += 1

    onsite_cubic_coeff = 0.0
    if not disable_onsite_cubic and onsite_nonlinear_strength != 0.0:
        onsite_cubic_coeff = onsite_nonlinear_strength * 9.81 / (6.0 * L ** 3)

    return times[:sample_idx], traj[:sample_idx], kicks, onsite_cubic_coeff


def compute_welch_psd(times, traj, segment_seconds, overlap_fraction):
    if len(times) < 2:
        raise ValueError("Need at least two stored samples for Welch PSD")
    if segment_seconds <= 0:
        raise ValueError("--welch_segment_seconds must be positive")
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("--welch_overlap must satisfy 0 <= overlap < 1")

    dt_sample = times[1] - times[0]
    fs = 1.0 / dt_sample
    centered_traj = traj - np.mean(traj, axis=0, keepdims=True)

    segment_len = max(8, int(round(segment_seconds * fs)))
    if segment_len > len(centered_traj):
        segment_len = len(centered_traj)
    overlap_len = int(round(overlap_fraction * segment_len))
    step = max(1, segment_len - overlap_len)

    window = np.hanning(segment_len)
    window_norm = np.sum(window ** 2)
    segments = []
    centers = []

    for start in range(0, len(centered_traj) - segment_len + 1, step):
        chunk = centered_traj[start:start + segment_len, :]
        chunk = chunk - np.mean(chunk, axis=0, keepdims=True)
        chunk = chunk * window[:, None]
        fft_vals = np.fft.rfft(chunk, axis=0)
        psd_per_site = (np.abs(fft_vals) ** 2) / (fs * window_norm)
        psd = np.mean(psd_per_site, axis=1)
        segments.append(psd)
        centers.append(times[start + segment_len // 2])

    if not segments:
        raise ValueError("Welch segmentation produced no segments")

    welch_matrix = np.stack(segments, axis=1)
    freqs = np.fft.rfftfreq(segment_len, d=dt_sample)
    avg_psd = np.mean(welch_matrix, axis=1)
    return freqs, avg_psd, welch_matrix, np.asarray(centers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disordered linear chain visualizer with optional onsite and bond/contact nonlinearities."
    )
    parser.add_argument("--m1", type=float, default=None)
    parser.add_argument("--m2", type=float, default=None)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--switch", action="store_true")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DISORDER_SEED)
    parser.add_argument(
        "--disorder",
        type=float,
        default=100.0 * DISORDER_STRENGTH,
        help="Percent disorder applied independently to all masses and springs.",
    )
    parser.add_argument(
        "--L_inches",
        type=float,
        default=DEFAULT_L_INCHES,
        help="Pendulum length in inches.",
    )
    parser.add_argument(
        "--max_disp_inches",
        type=float,
        default=DEFAULT_MAX_DISP_INCHES,
        help="Heuristic displacement scale used to set the random impulse strength.",
    )
    parser.add_argument(
        "--block_spacing_inches",
        type=float,
        default=DEFAULT_BLOCK_SPACING_INCHES,
        help="Nominal block-to-block spacing used for reporting bond stretch scale.",
    )
    parser.add_argument(
        "--kick_interval",
        type=float,
        default=DEFAULT_KICK_INTERVAL,
        help="Seconds between seeded random impulse kicks.",
    )
    parser.add_argument(
        "--force_peak_N",
        type=float,
        default=DEFAULT_FORCE_PEAK_N,
        help="Peak force in newtons used to define the default Gaussian-contact impulse scale.",
    )
    parser.add_argument(
        "--contact_ms",
        type=float,
        default=DEFAULT_CONTACT_MS,
        help="Gaussian contact duration in milliseconds, interpreted as FWHM.",
    )
    parser.add_argument(
        "--drive_site",
        default=DEFAULT_DRIVE_SITE,
        help='Drive site: "first", "last", "random", or a zero-based site index.',
    )
    parser.add_argument(
        "--drive_time_randomness",
        type=float,
        default=DEFAULT_DRIVE_TIME_RANDOMNESS,
        help="Relative randomness of drive timing. 0 means periodic, 0.8 means interval factors in [0.2, 1.8].",
    )
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument(
        "--dt_internal",
        type=float,
        default=DEFAULT_DT_INTERNAL,
        help="Internal integration timestep in seconds.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help="Stored output sample rate in Hz for Welch and plots.",
    )
    parser.add_argument(
        "--decay_time",
        type=float,
        default=DEFAULT_DECAY_TIME,
        help="Target amplitude decay time in seconds.",
    )
    parser.add_argument(
        "--decay_fraction",
        type=float,
        default=DEFAULT_DECAY_FRACTION,
        help="Target remaining amplitude fraction after --decay_time.",
    )
    parser.add_argument(
        "--welch_segment_seconds",
        type=float,
        default=DEFAULT_WELCH_SEGMENT,
        help="Segment length in seconds for the Welch PSD.",
    )
    parser.add_argument(
        "--welch_overlap",
        type=float,
        default=DEFAULT_WELCH_OVERLAP,
        help="Welch overlap fraction in [0, 1).",
    )
    parser.add_argument(
        "--image_cols",
        type=int,
        default=DEFAULT_IMAGE_COLS,
        help="Arbitrary horizontal width of the FFT image panel.",
    )
    parser.add_argument(
        "--disable_onsite_cubic",
        action="store_true",
        help="Disable the legacy onsite cubic term even if an onsite strength is provided.",
    )
    parser.add_argument(
        "--onsite_nonlinear_strength",
        type=float,
        default=0.0,
        help="Scales the legacy onsite cubic pendulum term relative to its original value.",
    )
    parser.add_argument(
        "--nonlinear_strength",
        type=float,
        default=None,
        help="Legacy alias for --onsite_nonlinear_strength.",
    )
    parser.add_argument(
        "--contact_model",
        choices=["off", "cubic", "hertz_gap"],
        default="off",
    )
    parser.add_argument("--contact_k3", type=float, default=0.0)
    parser.add_argument("--contact_gap", type=float, default=0.0)
    parser.add_argument("--contact_kh", type=float, default=0.0)
    parser.add_argument("--contact_exponent", type=float, default=1.5)
    parser.add_argument(
        "--contact_disorder",
        type=float,
        default=0.0,
        help="Percent disorder applied independently to nonlinear bond coefficients.",
    )
    parser.add_argument(
        "--fft_max_freq",
        type=float,
        default=None,
        help="Maximum plotted frequency for FFT-based panels.",
    )
    parser.add_argument(
        "--lineplot_max_freq",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=PANEL_CHOICES,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.disorder < 0:
        raise ValueError("--disorder must be non-negative")
    if args.L_inches <= 0:
        raise ValueError("--L_inches must be positive")
    if args.max_disp_inches < 0:
        raise ValueError("--max_disp_inches must be non-negative")
    if args.block_spacing_inches <= 0:
        raise ValueError("--block_spacing_inches must be positive")
    if args.dt_internal <= 0:
        raise ValueError("--dt_internal must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample_rate must be positive")
    if args.tmax <= 0:
        raise ValueError("--tmax must be positive")
    if args.decay_time <= 0:
        raise ValueError("--decay_time must be positive")
    if not (0.0 < args.decay_fraction < 1.0):
        raise ValueError("--decay_fraction must satisfy 0 < value < 1")
    if args.image_cols < 1:
        raise ValueError("--image_cols must be at least 1")
    if args.drive_time_randomness < 0:
        raise ValueError("--drive_time_randomness must be non-negative")
    if args.force_peak_N < 0:
        raise ValueError("--force_peak_N must be non-negative")
    if args.contact_ms <= 0:
        raise ValueError("--contact_ms must be positive")
    if args.onsite_nonlinear_strength < 0:
        raise ValueError("--onsite_nonlinear_strength must be non-negative")
    if args.contact_k3 < 0:
        raise ValueError("--contact_k3 must be non-negative")
    if args.contact_gap < 0:
        raise ValueError("--contact_gap must be non-negative")
    if args.contact_kh < 0:
        raise ValueError("--contact_kh must be non-negative")
    if args.contact_exponent <= 0:
        raise ValueError("--contact_exponent must be positive")
    if args.contact_disorder < 0:
        raise ValueError("--contact_disorder must be non-negative")

    disorder_strength = args.disorder / 100.0
    L = inches_to_meters(args.L_inches)
    max_disp = inches_to_meters(args.max_disp_inches)
    block_spacing = inches_to_meters(args.block_spacing_inches)
    damping_gamma = damping_gamma_from_decay(args.decay_time, args.decay_fraction)

    rng = np.random.default_rng(args.seed)

    k = 194.0
    m1 = 37e-3
    m2 = 74e-3
    N = args.N

    if args.m1 is not None:
        m1 = args.m1
    if args.m2 is not None:
        m2 = args.m2
    if args.k is not None:
        k = args.k

    if args.switch:
        m1, m2 = m2, m1

    selected_panels = PANEL_CHOICES if args.only is None else args.only
    onsite_nonlinear_strength = args.onsite_nonlinear_strength
    if args.nonlinear_strength is not None and onsite_nonlinear_strength == 0.0:
        onsite_nonlinear_strength = args.nonlinear_strength

    H, masses, springs = build_disordered_dynamical_matrix(
        N, k, m1, m2, rng, disorder_strength, L=L
    )
    contact_k3, contact_gap, contact_kh = build_nonlinear_contact_arrays(
        N,
        rng,
        args.contact_model,
        args.contact_k3,
        args.contact_gap,
        args.contact_kh,
        args.contact_disorder,
    )
    lam, V = sort_eigensystem(H)
    lam[np.abs(lam) < 1e-14] = 0.0
    omega = np.sqrt(np.clip(lam, 0.0, None))
    freq = omega / (2.0 * np.pi)

    H_tilde = build_transformed_matrix_from_H(H, k, m1, m2)
    E_full, W_full = sort_eigensystem(H_tilde)
    del W_full

    SSH_like = build_ssh_like_reduced_matrix(N, m1, m2)
    E_red, U_red = sort_eigensystem(SSH_like)
    del U_red

    times, traj, kicks, onsite_cubic_coeff = simulate_nonlinear_chain(
        H,
        masses,
        L,
        rng,
        tmax=args.tmax,
        dt_internal=args.dt_internal,
        sample_rate=args.sample_rate,
        kick_interval=args.kick_interval,
        force_peak_N=args.force_peak_N,
        contact_ms=args.contact_ms,
        damping_gamma=damping_gamma,
        disable_onsite_cubic=args.disable_onsite_cubic,
        onsite_nonlinear_strength=onsite_nonlinear_strength,
        contact_model=args.contact_model,
        contact_k3=contact_k3,
        contact_gap=contact_gap,
        contact_kh=contact_kh,
        contact_exponent=args.contact_exponent,
        drive_site=args.drive_site,
        drive_time_randomness=args.drive_time_randomness,
    )
    fft_freqs, fft_spectrum, welch_matrix, welch_centers = compute_welch_psd(
        times,
        traj,
        segment_seconds=args.welch_segment_seconds,
        overlap_fraction=args.welch_overlap,
    )
    del welch_matrix, welch_centers

    fft_max_freq = args.fft_max_freq
    if fft_max_freq is None:
        positive = fft_freqs[fft_freqs > 0]
        fft_max_freq = 1.05 * positive[-1] if len(positive) else 1.0

    lineplot_max_freq = args.lineplot_max_freq
    if lineplot_max_freq is None:
        max_freq = float(np.max(freq)) if len(freq) else 0.0
        lineplot_max_freq = 1.05 * max_freq if max_freq > 0 else 1.0

    max_abs_disp = float(np.max(np.abs(traj))) if traj.size else 0.0
    bond_stretch = np.diff(traj, axis=1) if N > 1 else np.zeros((len(times), 0), dtype=float)
    max_abs_bond_stretch = float(np.max(np.abs(bond_stretch))) if bond_stretch.size else 0.0
    kick_events = int(np.count_nonzero(np.linalg.norm(kicks, axis=1)))

    print(f"Disorder seed: {args.seed}")
    print(f"Disorder strength: +/-{args.disorder:.1f}%")
    print(f"Pendulum length: {args.L_inches:.3f} in ({L:.6f} m)")
    print(f"Block spacing heuristic: {args.block_spacing_inches:.3f} in ({block_spacing:.6f} m)")
    print(f"Max displacement heuristic: {args.max_disp_inches:.3f} in ({max_disp:.6f} m)")
    print(f"Kick interval: {args.kick_interval:.4f} s")
    print(f"Drive site mode: {args.drive_site}")
    print(f"Drive time randomness: {args.drive_time_randomness:.3f}")
    print(f"Force peak: {args.force_peak_N:.6f} N")
    print(f"Contact duration: {args.contact_ms:.6f} ms (Gaussian FWHM)")
    print(
        f"Impulse scale: {gaussian_impulse_from_peak_force(args.force_peak_N, args.contact_ms):.12g} N s"
    )
    onsite_enabled = (not args.disable_onsite_cubic) and onsite_nonlinear_strength != 0.0
    print(f"Contact model: {args.contact_model}")
    print(f"Onsite cubic enabled: {onsite_enabled}")
    print(f"Onsite nonlinear strength: {onsite_nonlinear_strength:.6f} x")
    print(f"Contact k3: {args.contact_k3:.12g}")
    print(f"Contact gap: {args.contact_gap:.12g}")
    print(f"Contact kh: {args.contact_kh:.12g}")
    print(f"Contact exponent: {args.contact_exponent:.12g}")
    print(f"Contact disorder: +/-{args.contact_disorder:.1f}%")
    print(
        f"Damping gamma: {damping_gamma:.12g} 1/s "
        f"(amplitude -> {args.decay_fraction:.6g} after {args.decay_time:.3f} s)"
    )
    print(f"Onsite cubic coefficient g/(6L^3) scale: {onsite_cubic_coeff:.12g}")
    print(
        f"Welch segment: {args.welch_segment_seconds:.3f} s, "
        f"overlap: {args.welch_overlap:.3f}"
    )
    print("Disordered masses:")
    print(masses)
    print("Disordered springs:")
    print(springs)
    print("Nonlinear contact k3 array:")
    print(contact_k3)
    print("Nonlinear contact gap array:")
    print(contact_gap)
    print("Nonlinear contact kh array:")
    print(contact_kh)

    print("Disordered linearized frequencies [Hz]:")
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

    print("\nNonlinear drive summary:")
    print(f"stored samples: {len(times)}")
    print(f"kick events: {kick_events}")
    print(f"max |y_i(t)|: {max_abs_disp:.6f} m = {max_abs_disp / 0.0254:.3f} in")
    print(
        f"max |y_i(t) - y_(i+1)(t)|: {max_abs_bond_stretch:.6f} m = "
        f"{max_abs_bond_stretch / 0.0254:.3f} in"
    )
    print(
        f"max bond stretch / spacing heuristic: "
        f"{(max_abs_bond_stretch / block_spacing) if block_spacing > 0 else np.nan:.6f}"
    )

    fig, axes = make_axes(len(selected_panels))

    freq_mask = fft_freqs <= fft_max_freq
    fft_freqs_plot = fft_freqs[freq_mask]
    fft_spectrum_plot = fft_spectrum[freq_mask]
    fft_spectrum_plot_db = 10.0 * np.log10(fft_spectrum_plot + 1e-18)
    fft_image_db = build_spectrum_image_db(
        fft_freqs_plot,
        fft_spectrum_plot_db,
        fft_max_freq,
        args.image_cols,
    )
    linear_prediction_image = build_lineplot_image(freq, fft_max_freq, args.image_cols)

    for ax, panel in zip(axes, selected_panels):
        if panel == "freqs":
            ax.scatter(np.arange(1, N + 1), freq, s=40, c="k")
            ax.set_xlabel("Mode number")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_title("Disordered linearized frequencies")

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
                ax.plot(np.arange(1, N + 1), v + offset, ".-", color=color, linewidth=1)

            ax.set_xlabel("Site")
            ax.set_ylabel("Normalized mode + offset")
            ax.set_title("Disordered linear eigenvectors")

        elif panel == "thesis_transformed":
            ax.plot(np.arange(1, N + 1), E_full, "o", label="full transformed")
            ax.plot(np.arange(1, N), E_red, "x", label="reduced SSH-like")
            ax.set_xlabel("Sorted eigenvalue index")
            ax.set_ylabel("E")
            ax.set_title("Thesis transformed spectrum")
            ax.legend()

        elif panel == "qspace":
            spacing = 1.5
            colors = ["steelblue", "gray"]
            D = difference_operator(N)
            target_lam = k * (m1 + m2) / (m1 * m2)

            for i in range(N):
                y_vec = V[:, i].copy()
                q_vec = D @ y_vec

                qmax = np.max(np.abs(q_vec))
                if qmax > 1e-12:
                    q_vec = q_vec / qmax
                else:
                    q_vec = np.zeros_like(q_vec)

                offset = (i + 1) * spacing
                is_edge = np.abs(lam[i] - target_lam) < (0.05 * target_lam)

                if is_edge:
                    ax.plot(
                        np.arange(1, N),
                        q_vec + offset,
                        "o-",
                        color="red",
                        linewidth=2.5,
                        zorder=10,
                    )
                else:
                    ax.plot(
                        np.arange(1, N),
                        q_vec + offset,
                        ".-",
                        color=colors[i % 2],
                        linewidth=1,
                    )

            ax.set_xlabel("Bond index $n$")
            ax.set_ylabel("Normalized Bond Stretch + offset")
            ax.set_title("Linear modes in q-space")

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
                vmax=1,
            )
            ax.set_xlabel("Arbitrary index")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_title("Linear eigenfrequency image plot")

        elif panel == "timeseries":
            scale_in = traj / 0.0254
            for i in range(N):
                ax.plot(times, scale_in[:, i], linewidth=1)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Site displacement y_i [in]")
            ax.set_title("Displacement of every mass vs time")

        elif panel == "fft_spectrum":
            ax.plot(fft_freqs_plot, fft_spectrum_plot_db, color="black", linewidth=1.5)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Welch PSD [dB]")
            ax.set_title("Aggregate Welch spectrum")

        elif panel == "fft_image":
            subgrid = ax.get_subplotspec().subgridspec(1, 2, wspace=0.05)
            left_ax = fig.add_subplot(subgrid[0, 0])
            right_ax = fig.add_subplot(subgrid[0, 1], sharex=left_ax, sharey=left_ax)
            ax.remove()

            extent = [0.5, args.image_cols + 0.5, 0.0, fft_max_freq]

            left_ax.imshow(
                fft_image_db,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="inferno",
                interpolation="nearest",
            )
            left_ax.set_xlabel("Arbitrary index")
            left_ax.set_ylabel("Frequency [Hz]")
            left_ax.set_title("Welch image")

            right_ax.imshow(
                linear_prediction_image,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            right_ax.set_xlabel("Arbitrary index")
            right_ax.set_title("Linear prediction")
            right_ax.tick_params(labelleft=False)

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
