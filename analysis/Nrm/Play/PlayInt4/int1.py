#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
PLAY_DIR = SCRIPT_DIR.parent


def add_repo_root_to_path() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing the 'analysis' package.")


REPO_ROOT = add_repo_root_to_path()
if str(PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(PLAY_DIR))

from analysis.go.Play.fft_flattening import compute_flattened_component_spectra
from play5 import _bicoherence_from_indices
from play7 import build_windows, score_windows, score_windows_surrogate, empirical_pvalue

from PlayInt2.int1 import detect_observed_peaks






from dataclasses import dataclass
from analysis.tools.signal import preprocess_signal, hann_window_periodic, next_power_of_two
from analysis.Nrm.Tools.post_hit_regions import extract_post_hit_regions, EnabledRegionConfig

@dataclass(frozen=True)
class SegmentRecord:
    bond_id: int
    start_s: float
    stop_s: float
    mid_s: float
    spectrum: np.ndarray

def collect_segments(
    dataset: str,
    component: str,
    bond_ids: list[int],
    segment_len_s: float,
    overlap: float,
    *,
    bond_spacing_mode: str = "default",
) -> tuple[np.ndarray, list[SegmentRecord], np.ndarray]:
    records: list[SegmentRecord] = []
    nperseg: int | None = None
    freqs: np.ndarray | None = None

    for bond_id in bond_ids:
        result = extract_post_hit_regions(
            dataset=dataset,
            component=component,
            bond_id=bond_id,
            config=EnabledRegionConfig(bond_spacing_mode=str(bond_spacing_mode)),
        )
        processed, err = preprocess_signal(result.frame_times_s, result.signal, longest=False, handlenan=False)
        if processed is None:
            raise ValueError(f"Failed preprocessing bond {bond_id}: {err}")

        local_nperseg = max(8, int(round(segment_len_s * processed.Fs)))
        if local_nperseg > processed.y.size:
            continue
        step = max(1, local_nperseg - int(round(overlap * local_nperseg)))
        local_nfft = max(local_nperseg, next_power_of_two(local_nperseg))
        window = hann_window_periodic(local_nperseg)
        window_norm = float(np.sum(window))

        if nperseg is None:
            nperseg = local_nperseg
            freqs = np.fft.rfftfreq(local_nfft, d=processed.dt)
        elif local_nperseg != nperseg:
            raise ValueError("Segment length resolved differently across bonds")

        for start in range(0, processed.y.size - local_nperseg + 1, step):
            stop = start + local_nperseg
            segment = processed.y[start:stop] * window
            spectrum = np.fft.rfft(segment, n=local_nfft) / window_norm
            if spectrum.size > 2:
                spectrum = spectrum.copy()
                spectrum[1:-1] *= 2.0
            records.append(
                SegmentRecord(
                    bond_id=bond_id,
                    start_s=float(processed.t[start]),
                    stop_s=float(processed.t[stop - 1]),
                    mid_s=float(np.mean(processed.t[start:stop])),
                    spectrum=spectrum,
                )
            )

    if not records or freqs is None:
        raise ValueError("No valid segments could be collected from the specified bonds.")
    
    mean_amplitude = np.mean(np.abs([r.spectrum for r in records]), axis=0)
    return freqs, records, mean_amplitude


@dataclass(frozen=True)
class ResolvedPeak:
    label: str
    role: str
    certainty: str
    anchor_hz: float
    resolved_hz: float
    amplitude: float
    prominence: float


@dataclass(frozen=True)
class RelationResult:
    stage: str
    relation: str
    source_a: str
    source_b: str
    source_a_hz: float
    source_b_hz: float
    child_hz: float
    matched_peak_hz: float
    matched_peak_amp: float
    matched_peak_prom: float
    match_error_hz: float
    max_bicoherence: float
    best_phase_lock: float
    surrogate_p: float
    best_start_s: float
    best_stop_s: float
    passes: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrated staged coherence/bicoherence scan using prior fundamental classifications.",
    )
    parser.add_argument("dataset", help="Dataset base name.")
    parser.add_argument("--show", action="store_true", help="Show figures.")
    parser.add_argument("--component", default="x")
    parser.add_argument("--segment-len-s", type=float, default=100.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--analysis-window-s", type=float, default=100.0)
    parser.add_argument("--analysis-step-s", type=float, default=25.0)
    parser.add_argument("--min-window-segments", type=int, default=6)
    parser.add_argument("--bond-spacing-mode", choices=("default", "purecomoving"), default="purecomoving")
    parser.add_argument("--snap-bins", type=int, default=3)
    parser.add_argument("--peak-prominence", type=float, default=0.03)
    parser.add_argument("--merge-hz", type=float, default=0.08)
    parser.add_argument("--peak-match-tol-hz", type=float, default=0.15)
    parser.add_argument("--n-surrogates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-min-bicoherence", type=float, default=0.20)
    parser.add_argument("--support-max-p", type=float, default=0.15)
    return parser


def configure_matplotlib(show: bool) -> None:
    if show:
        try:
            matplotlib.use("QtAgg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")


def flattened_welch_spectrum(*, dataset: str, component: str, bond_spacing_mode: str) -> tuple[np.ndarray, np.ndarray]:
    results = compute_flattened_component_spectra(
        dataset=dataset,
        bond_spacing_mode=bond_spacing_mode,
        components=(component,),
        use_welch=True,
    )
    result = results[component]
    return np.asarray(result.freq_hz, dtype=float), np.asarray(result.flattened, dtype=float)


def resolve_prior(anchor_hz: float, observed_peaks, *, label: str, role: str, certainty: str) -> ResolvedPeak:
    nearest = min(observed_peaks, key=lambda peak: abs(peak.freq_hz - anchor_hz))
    return ResolvedPeak(
        label=label,
        role=role,
        certainty=certainty,
        anchor_hz=float(anchor_hz),
        resolved_hz=float(nearest.freq_hz),
        amplitude=float(nearest.amplitude),
        prominence=float(nearest.prominence),
    )


def _snap_index(index: int, amplitude: np.ndarray, radius: int) -> int:
    if radius <= 0:
        return int(index)
    lo = max(0, int(index) - int(radius))
    hi = min(amplitude.size, int(index) + int(radius) + 1)
    return int(lo + np.argmax(amplitude[lo:hi]))


def _resolve_source_indices(
    freqs: np.ndarray,
    mean_amplitude: np.ndarray,
    peak_a_hz: float,
    peak_b_hz: float,
    snap_bins: int,
) -> tuple[int, int]:
    df = float(freqs[1] - freqs[0])
    ia = _snap_index(int(round(peak_a_hz / df)), mean_amplitude, snap_bins)
    ib = _snap_index(int(round(peak_b_hz / df)), mean_amplitude, snap_bins)
    return int(ia), int(ib)


def _best_phase_lock(i1: int, i2: int, i3: int, X_subset: np.ndarray) -> float:
    _, phase_lock = _bicoherence_from_indices(i1, i2, i3, X_subset)
    return float(phase_lock)


def evaluate_relation(
    *,
    stage: str,
    relation: str,
    peak_a: ResolvedPeak,
    peak_b: ResolvedPeak,
    freqs: np.ndarray,
    mean_amplitude: np.ndarray,
    X: np.ndarray,
    windows,
    observed_peaks,
    snap_bins: int,
    peak_match_tol_hz: float,
    n_surrogates: int,
    rng: np.random.Generator,
) -> RelationResult | None:
    ia, ib = _resolve_source_indices(freqs, mean_amplitude, peak_a.resolved_hz, peak_b.resolved_hz, snap_bins)

    if relation == "2x":
        i1 = ia
        i2 = ia
        i3 = ia + ia
        source_a = peak_a
        source_b = peak_a
    elif relation == "sum":
        i1 = min(ia, ib)
        i2 = max(ia, ib)
        i3 = i1 + i2
        source_a = peak_a if ia <= ib else peak_b
        source_b = peak_b if ia <= ib else peak_a
    elif relation == "diff":
        ih = max(ia, ib)
        il = min(ia, ib)
        i1 = ih - il
        i2 = il
        i3 = ih
        source_a = peak_a if ia >= ib else peak_b
        source_b = peak_b if ia >= ib else peak_a
    else:
        raise ValueError(f"Unsupported relation {relation}")

    if i1 <= 0 or i2 <= 0 or i3 >= freqs.size:
        return None

    child_hz = float(freqs[i1]) if relation == "diff" else float(freqs[i3])
    nearest_peak = min(observed_peaks, key=lambda peak: abs(peak.freq_hz - child_hz))
    match_error_hz = abs(float(nearest_peak.freq_hz) - child_hz)
    if match_error_hz > peak_match_tol_hz:
        return None

    if any(abs(float(nearest_peak.freq_hz) - src.resolved_hz) < 0.05 for src in (peak_a, peak_b)):
        return None

    scores = score_windows(X, windows, i1=i1, i2=i2, i3=i3)
    if scores.size == 0:
        return None
    best_idx = int(np.argmax(scores))
    best_window = windows[best_idx]
    best_subset = X[best_window.segment_indices]
    best_phase_lock = _best_phase_lock(i1, i2, i3, best_subset)

    surrogate_max = []
    for _ in range(n_surrogates):
        surrogate_scores = score_windows_surrogate(X, windows, i1=i1, i2=i2, i3=i3, rng=rng)
        surrogate_max.append(float(np.max(surrogate_scores)))
    surrogate_p = empirical_pvalue(float(scores[best_idx]), np.asarray(surrogate_max, dtype=float))
    passes = int(scores[best_idx] >= 0.20 and surrogate_p <= 0.15)

    return RelationResult(
        stage=stage,
        relation=relation,
        source_a=source_a.label,
        source_b=source_b.label,
        source_a_hz=float(freqs[i2] if relation == "diff" else freqs[i1]),
        source_b_hz=float(freqs[i3] if relation == "diff" else freqs[i2]),
        child_hz=child_hz,
        matched_peak_hz=float(nearest_peak.freq_hz),
        matched_peak_amp=float(nearest_peak.amplitude),
        matched_peak_prom=float(nearest_peak.prominence),
        match_error_hz=float(match_error_hz),
        max_bicoherence=float(scores[best_idx]),
        best_phase_lock=float(best_phase_lock),
        surrogate_p=float(surrogate_p),
        best_start_s=float(best_window.start_s),
        best_stop_s=float(best_window.stop_s),
        passes=passes,
    )


def stage_candidates(peaks: list[ResolvedPeak], *, include_diff: bool = True) -> list[tuple[str, ResolvedPeak, ResolvedPeak]]:
    out: list[tuple[str, ResolvedPeak, ResolvedPeak]] = []
    for idx, peak_a in enumerate(peaks):
        out.append(("2x", peak_a, peak_a))
        for peak_b in peaks[idx + 1 :]:
            out.append(("sum", peak_a, peak_b))
            if include_diff:
                out.append(("diff", peak_a, peak_b))
    return out


def write_csv(path: Path, rows: list[RelationResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "relation",
                "source_a",
                "source_b",
                "source_a_hz",
                "source_b_hz",
                "child_hz",
                "matched_peak_hz",
                "matched_peak_amp",
                "matched_peak_prom",
                "match_error_hz",
                "max_bicoherence",
                "best_phase_lock",
                "surrogate_p",
                "best_start_s",
                "best_stop_s",
                "passes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.stage,
                    row.relation,
                    row.source_a,
                    row.source_b,
                    f"{row.source_a_hz:.6f}",
                    f"{row.source_b_hz:.6f}",
                    f"{row.child_hz:.6f}",
                    f"{row.matched_peak_hz:.6f}",
                    f"{row.matched_peak_amp:.6f}",
                    f"{row.matched_peak_prom:.6f}",
                    f"{row.match_error_hz:.6f}",
                    f"{row.max_bicoherence:.6f}",
                    f"{row.best_phase_lock:.6f}",
                    f"{row.surrogate_p:.6f}",
                    f"{row.best_start_s:.3f}",
                    f"{row.best_stop_s:.3f}",
                    row.passes,
                ]
            )
    return path


def plot_results(
    *,
    freqs_flat: np.ndarray,
    amp_flat: np.ndarray,
    priors: list[ResolvedPeak],
    stage1: list[RelationResult],
    stage2: list[RelationResult],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)

    ax = axes[0]
    mask = (freqs_flat >= 0.0) & (freqs_flat <= 28.0)
    ax.plot(freqs_flat[mask], amp_flat[mask], color="black", lw=1.0)
    for peak in priors:
        color = "#1b9e77" if peak.role == "fundamental" else "#d95f02"
        ax.axvline(peak.resolved_hz, color=color, alpha=0.55, lw=1.0)
        ax.text(peak.resolved_hz, peak.amplitude, f"{peak.label}\n{peak.resolved_hz:.2f}", fontsize=7, color=color, ha="left", va="bottom", rotation=60)
    for row in stage1:
        if row.passes:
            ax.axvline(row.matched_peak_hz, color="#7570b3", alpha=0.35, lw=0.8)
    for row in stage2:
        if row.passes:
            ax.axvline(row.matched_peak_hz, color="#e7298a", alpha=0.35, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Flattened amplitude")
    ax.set_title("Flattened Spectrum With Prior Fundamentals and Phase-Supported Children")
    ax.grid(alpha=0.25)

    def _plot_stage(ax, rows: list[RelationResult], title: str, color: str) -> None:
        rows = sorted(rows, key=lambda row: row.max_bicoherence, reverse=True)[:12]
        if not rows:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No matched relations", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        labels = [f"{row.source_a}{'+' if row.relation != 'diff' else '-'}{row.source_b} -> {row.matched_peak_hz:.2f}" for row in rows]
        y = np.arange(len(rows))
        values = np.array([row.max_bicoherence for row in rows], dtype=float)
        alpha = np.clip(1.0 - np.array([row.surrogate_p for row in rows], dtype=float), 0.2, 1.0)
        ax.barh(y, values, color=[color] * len(rows), alpha=0.85)
        for patch, a in zip(ax.patches, alpha):
            patch.set_alpha(float(a))
        for yi, row in zip(y, rows):
            ax.text(row.max_bicoherence + 0.01, yi, f"p={row.surrogate_p:.2f}, PL={row.best_phase_lock:.2f}", fontsize=7, va="center")
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, max(0.35, float(np.max(values) * 1.25)))
        ax.set_xlabel("Max localized bicoherence")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    _plot_stage(axes[1], stage1, "Stage 1: Direct Relations From Prior Fundamentals", "#7570b3")
    _plot_stage(axes[2], stage2, "Stage 2: Cascades Using Stage-1 Supported Children", "#e7298a")

    path = OUTPUT_DIR / "int1_phase_pipeline.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(
    *,
    priors: list[ResolvedPeak],
    stage1: list[RelationResult],
    stage2: list[RelationResult],
    figure_path: Path,
    csv_path: Path,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("Resolved prior peaks:")
    for peak in priors:
        lines.append(
            f"  {peak.label}: {peak.resolved_hz:.3f} Hz | role={peak.role} | certainty={peak.certainty} | amp={peak.amplitude:.4f}"
        )
    lines.append("")
    lines.append("Stage 1 strongest direct relations:")
    for row in sorted(stage1, key=lambda row: row.max_bicoherence, reverse=True)[:10]:
        lines.append(
            f"  {row.source_a} {row.relation} {row.source_b} -> {row.matched_peak_hz:.3f} | "
            f"b2={row.max_bicoherence:.3f} | phase_lock={row.best_phase_lock:.3f} | p={row.surrogate_p:.3f} | "
            f"window={row.best_start_s:.1f}-{row.best_stop_s:.1f}s"
        )
    lines.append("")
    lines.append("Stage 2 strongest cascade relations:")
    for row in sorted(stage2, key=lambda row: row.max_bicoherence, reverse=True)[:12]:
        lines.append(
            f"  {row.source_a} {row.relation} {row.source_b} -> {row.matched_peak_hz:.3f} | "
            f"b2={row.max_bicoherence:.3f} | phase_lock={row.best_phase_lock:.3f} | p={row.surrogate_p:.3f} | "
            f"window={row.best_start_s:.1f}-{row.best_stop_s:.1f}s"
        )
    lines.append("")
    lines.append("Saved files:")
    lines.append(str(figure_path))
    lines.append(str(csv_path))
    path = OUTPUT_DIR / "int1_phase_pipeline_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib(args.show)
    rng = np.random.default_rng(args.seed)

    freqs_flat, amp_flat = flattened_welch_spectrum(
        dataset=args.dataset,
        component=args.component,
        bond_spacing_mode=str(args.bond_spacing_mode)
    )
    observed_peaks = detect_observed_peaks(
        freqs_flat,
        amp_flat,
        peak_prominence=float(args.peak_prominence),
        merge_hz=float(args.merge_hz),
        min_freq_hz=0.2,
        max_freq_hz=28.0,
    )

    priors = [
        resolve_prior(0.414, observed_peaks, label="f0.41", role="fundamental", certainty="100%"),
        resolve_prior(3.357, observed_peaks, label="f3.35", role="fundamental", certainty="100%"),
        resolve_prior(12.0, observed_peaks, label="f12.0", role="fundamental", certainty="100%"),
        resolve_prior(16.6, observed_peaks, label="f16.6", role="fundamental", certainty="100%"),
        resolve_prior(6.45, observed_peaks, label="f6.45", role="fundamental", certainty="100%"),
        resolve_prior(8.96, observed_peaks, label="f8.96", role="fundamental", certainty="100%"),
        resolve_prior(11.3, observed_peaks, label="u11.3", role="suspect_child", certainty="not fundamental?"),
        resolve_prior(15.9, observed_peaks, label="u15.9", role="suspect_child", certainty="not fundamental?"),
        resolve_prior(18.0, observed_peaks, label="u18.0", role="child", certainty="not fundamental?"),
        resolve_prior(18.3, observed_peaks, label="u18.3", role="child", certainty="100% not fundamental"),
    ]

    from analysis.tools.bonds import load_bond_signal_dataset
    bond_dataset = load_bond_signal_dataset(
        dataset=f"{args.dataset}_{args.component}",
        bond_spacing_mode=str(args.bond_spacing_mode),
        component=args.component,
    )
    bond_ids = list(range(bond_dataset.signal_matrix.shape[1]))

    freqs_seg, records, mean_amplitude = collect_segments(
        args.dataset,
        args.component,
        bond_ids,
        float(args.segment_len_s),
        float(args.overlap),
        bond_spacing_mode=str(args.bond_spacing_mode),
    )
    X = np.vstack([record.spectrum for record in records])
    mids = np.asarray([record.mid_s for record in records], dtype=float)
    windows = build_windows(
        mids,
        analysis_window_s=float(args.analysis_window_s),
        analysis_step_s=float(args.analysis_step_s),
        min_segments=int(args.min_window_segments),
    )

    fundamentals = [peak for peak in priors if peak.role == "fundamental"]
    stage1_results: list[RelationResult] = []
    seen_children: set[tuple[str, str, str, float]] = set()
    for relation, peak_a, peak_b in stage_candidates(fundamentals):
        row = evaluate_relation(
            stage="stage1",
            relation=relation,
            peak_a=peak_a,
            peak_b=peak_b,
            freqs=freqs_seg,
            mean_amplitude=mean_amplitude,
            X=X,
            windows=windows,
            observed_peaks=observed_peaks,
            snap_bins=int(args.snap_bins),
            peak_match_tol_hz=float(args.peak_match_tol_hz),
            n_surrogates=int(args.n_surrogates),
            rng=rng,
        )
        if row is None:
            continue
        key = (row.relation, row.source_a, row.source_b, round(row.matched_peak_hz, 3))
        if key in seen_children:
            continue
        seen_children.add(key)
        stage1_results.append(row)

    supported_children = []
    for row in stage1_results:
        if row.max_bicoherence >= float(args.support_min_bicoherence) and row.surrogate_p <= float(args.support_max_p):
            supported_children.append(
                ResolvedPeak(
                    label=f"c{row.matched_peak_hz:.2f}",
                    role="supported_child",
                    certainty="phase-supported",
                    anchor_hz=row.matched_peak_hz,
                    resolved_hz=row.matched_peak_hz,
                    amplitude=row.matched_peak_amp,
                    prominence=row.matched_peak_prom,
                )
            )

    # Keep only distinct supported children and prefer stronger stage-1 support.
    best_child_by_hz: dict[float, ResolvedPeak] = {}
    for child in supported_children:
        best_child_by_hz.setdefault(round(child.resolved_hz, 3), child)
    supported_children = list(best_child_by_hz.values())

    target_like = [peak for peak in priors if peak.role != "fundamental"]
    target_hz = np.asarray([peak.resolved_hz for peak in target_like], dtype=float)
    stage2_sources = fundamentals + supported_children
    stage2_results: list[RelationResult] = []
    seen_stage2: set[tuple[str, str, str, float]] = set()
    for relation, peak_a, peak_b in stage_candidates(stage2_sources):
        if peak_a.role == "fundamental" and peak_b.role == "fundamental":
            continue
        row = evaluate_relation(
            stage="stage2",
            relation=relation,
            peak_a=peak_a,
            peak_b=peak_b,
            freqs=freqs_seg,
            mean_amplitude=mean_amplitude,
            X=X,
            windows=windows,
            observed_peaks=observed_peaks,
            snap_bins=int(args.snap_bins),
            peak_match_tol_hz=float(args.peak_match_tol_hz),
            n_surrogates=int(args.n_surrogates),
            rng=rng,
        )
        if row is None:
            continue
        if np.min(np.abs(target_hz - row.matched_peak_hz)) > 0.5:
            continue
        key = (row.relation, row.source_a, row.source_b, round(row.matched_peak_hz, 3))
        if key in seen_stage2:
            continue
        seen_stage2.add(key)
        stage2_results.append(row)

    stage1_results.sort(key=lambda row: row.max_bicoherence, reverse=True)
    stage2_results.sort(key=lambda row: row.max_bicoherence, reverse=True)

    csv_path = write_csv(OUTPUT_DIR / "int1_phase_pipeline.csv", stage1_results + stage2_results)
    figure_path = plot_results(
        freqs_flat=freqs_flat,
        amp_flat=amp_flat,
        priors=priors,
        stage1=stage1_results,
        stage2=stage2_results,
    )
    summary_path = write_summary(
        priors=priors,
        stage1=stage1_results,
        stage2=stage2_results,
        figure_path=figure_path,
        csv_path=csv_path,
    )

    print("Resolved priors:")
    for peak in priors:
        print(f"  {peak.label}: {peak.resolved_hz:.3f} Hz | {peak.role} | {peak.certainty}")
    print("\nStage 1 strongest:")
    for row in stage1_results[:8]:
        print(
            f"  {row.source_a} {row.relation} {row.source_b} -> {row.matched_peak_hz:.3f} | "
            f"b2={row.max_bicoherence:.3f} | PL={row.best_phase_lock:.3f} | p={row.surrogate_p:.3f}"
        )
    print("\nStage 2 strongest:")
    for row in stage2_results[:8]:
        print(
            f"  {row.source_a} {row.relation} {row.source_b} -> {row.matched_peak_hz:.3f} | "
            f"b2={row.max_bicoherence:.3f} | PL={row.best_phase_lock:.3f} | p={row.surrogate_p:.3f}"
        )
    print(f"\nSaved figure to {figure_path}")
    print(f"Saved summary to {summary_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
