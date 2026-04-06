#!/usr/bin/env python3
"""Synthesize a hit-smushed dataset by phase-aligning inter-hit segments in purecomoving space.

Reads source x/y track2 data, derives purecomoving signal matrices (longitudinal
for x, transverse for y), identifies inter-hit regions, and concatenates them
with sub-sample time shifts chosen so that the globally dominant frequency
(2-28 Hz) has a common reference phase at the start of every segment.

The smushed purecomoving signals are stored in the ``purecomovingSignal`` cache
field of the track2 msgpack so downstream analysis tools use them directly
without recomputing from xPositions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import msgpack
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.tools.bonds import _derive_purecomoving_signal_matrices
from analysis.tools.io import DEFAULT_TRACK_DATA_ROOT, load_track2_dataset
from analysis.tools.signal import compute_one_sided_fft_complex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create a synthetic hit-smushed track2 dataset with phase-aligned inter-hit segments (purecomoving space).",
    )
    p.add_argument("dataset", help="Source dataset stem, e.g. 11triv")
    p.add_argument("--exclude-after", type=float, default=5.0, help="Seconds after each hit to exclude.")
    p.add_argument("--exclude-before", type=float, default=1.0, help="Seconds before next hit to exclude.")
    p.add_argument("--analysis-window-s", type=float, default=2.0, help="Fixed-length FFT analysis window at start of each region.")
    p.add_argument("--norm-min-hz", type=float, default=2.0, help="Lower bound for dominant-frequency search.")
    p.add_argument("--norm-max-hz", type=float, default=28.0, help="Upper bound for dominant-frequency search.")
    p.add_argument("--track-data-root", default=None, help="Override track data root.")
    p.add_argument("--no-write", action="store_true", help="Dry-run: compute shifts but do not write files.")
    return p


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_hits_csv(path: Path) -> list[float]:
    times: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            times.append(float(row["time_s"]))
    return times


def _inter_hit_regions(hit_times: np.ndarray, t_end: float, ea: float, eb: float) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    for i in range(len(hit_times)):
        start = hit_times[i] + ea
        end = hit_times[i + 1] - eb if i < len(hit_times) - 1 else t_end - 1.0
        if end > start + 0.5:
            regions.append((start, end))
    return regions


def _wrap_to_pi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _circular_std(phi: np.ndarray) -> float:
    phi = np.asarray(phi, dtype=float)
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        return 0.0
    R = np.abs(np.mean(np.exp(1j * phi)))
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-15))))


def _msgpackable(obj):
    """Recursively convert numpy types to plain Python for msgpack."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _msgpackable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_msgpackable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def synthesize(
    dataset: str,
    *,
    exclude_after: float,
    exclude_before: float,
    analysis_window_s: float,
    norm_min_hz: float,
    norm_max_hz: float,
    track_data_root: str | None,
    dry_run: bool,
) -> None:
    root = Path(track_data_root) if track_data_root else DEFAULT_TRACK_DATA_ROOT

    # 1. Load x and y track2
    t2x = load_track2_dataset(dataset=f"{dataset}_x", track_data_root=track_data_root)
    t2y = load_track2_dataset(dataset=f"{dataset}_y", track_data_root=track_data_root)
    t2a = load_track2_dataset(dataset=f"{dataset}_a", track_data_root=track_data_root)

    block_colors = t2x.block_colors
    n_frames, n_blocks = t2x.x_positions.shape
    n_bonds = n_blocks - 1
    print(f"Source: {n_frames} frames, {n_blocks} blocks, {n_bonds} bonds")

    # 2. Derive purecomoving signals
    long_sig, trans_sig = _derive_purecomoving_signal_matrices(t2x, t2y)
    # long_sig = longitudinal (x), trans_sig = transverse (y)
    frame_times = t2x.frame_times_s.copy()

    # Clean frame_times: remove duplicates and ensure monotonicity for np.interp
    # Some datasets (e.g. 10MassNew) have duplicate timestamps and wraparound jumps.
    keep = np.zeros(len(frame_times), dtype=bool)
    keep[0] = True
    last_t = frame_times[0]
    for i in range(1, len(frame_times)):
        if frame_times[i] > last_t:
            keep[i] = True
            last_t = frame_times[i]
    n_dropped = int(np.sum(~keep))
    if n_dropped:
        print(f"Dropped {n_dropped} duplicate/non-monotonic frames from time vector")
    frame_times = frame_times[keep]
    long_sig = long_sig[keep]
    trans_sig = trans_sig[keep]
    # Also clean xPositions for all components
    t2x_xpos = t2x.x_positions[keep]
    t2y_xpos = t2y.x_positions[keep]
    t2a_xpos = t2a.x_positions[keep]
    t2x_ftimes = t2x.frame_times_s[keep]
    t2y_ftimes = t2y.frame_times_s[keep]
    t2a_ftimes = t2a.frame_times_s[keep]
    n_frames = len(frame_times)
    print(f"Clean frames: {n_frames}")

    # 3. Load hits
    repo_root = Path(__file__).resolve().parents[2]
    hits_csv = repo_root / "analysis" / "NL" / "out" / f"{dataset}_comparison_purecomoving" / f"{dataset}__x__prototype_hits.csv"
    if not hits_csv.exists():
        raise FileNotFoundError(f"Hits CSV not found: {hits_csv}")
    hit_times = np.asarray(_load_hits_csv(hits_csv), dtype=float)
    print(f"Loaded {len(hit_times)} hits")

    # 4. Define inter-hit regions
    regions = _inter_hit_regions(hit_times, float(frame_times[-1]), exclude_after, exclude_before)
    if not regions:
        raise ValueError("No inter-hit regions with these exclusion parameters.")
    print(f"Found {len(regions)} inter-hit regions")

    # 5. Average longitudinal signal across bonds for phase analysis
    agg_long = np.nanmean(long_sig, axis=1)
    dt = float(np.median(np.diff(frame_times)))
    fs = 1.0 / dt

    # Replace NaNs with 0 for FFT (purecomoving has NaN in frame 0)
    y_clean = np.where(np.isfinite(agg_long), agg_long, 0.0)

    # 6. Compute complex FFT for each region's analysis window
    fft_results: list[tuple[int, int, np.ndarray, np.ndarray, float, float]] = []
    # (start_frame, end_frame, freqs, spectrum, r_start_time, r_end_time)

    for r_start, r_end in regions:
        # Convert times to frame indices
        start_idx = int(np.searchsorted(frame_times, r_start, side="left"))
        end_idx = int(np.searchsorted(frame_times, min(r_start + analysis_window_s, r_end), side="right"))
        if end_idx - start_idx < 16:
            continue

        y_seg = y_clean[start_idx:end_idx]
        result = compute_one_sided_fft_complex(y_seg, dt)

        # Check enough energy in norm range
        f_mask = (result.freq >= norm_min_hz) & (result.freq <= norm_max_hz)
        if not np.any(f_mask) or np.max(np.abs(result.spectrum[f_mask])) < 1e-15:
            continue

        fft_results.append((start_idx, end_idx, result.freq, result.spectrum, r_start, r_end))

    if not fft_results:
        raise ValueError("No regions produced a valid FFT in the norm range.")

    # 7. Find globally dominant frequency
    freq_mask = (fft_results[0][2] >= norm_min_hz) & (fft_results[0][2] <= norm_max_hz)
    avg_amp = np.mean([np.abs(sp[freq_mask]) for (_, _, _, sp, _, _) in fft_results], axis=0)
    dom_idx_in_mask = int(np.argmax(avg_amp))
    all_freqs = fft_results[0][2]
    dominant_freq = float(all_freqs[freq_mask][dom_idx_in_mask])
    dominant_global_idx = int(np.where(freq_mask)[0][dom_idx_in_mask])
    print(f"Global dominant frequency: {dominant_freq:.4f} Hz")

    # 8. Second-dominant frequency
    amp_copy = avg_amp.copy()
    freq_res = float(all_freqs[1] - all_freqs[0]) if len(all_freqs) > 1 else 0.1
    half_bins = max(3, int(round(0.5 / freq_res)))
    lo = max(0, dom_idx_in_mask - half_bins)
    hi = min(len(amp_copy), dom_idx_in_mask + half_bins + 1)
    amp_copy[lo:hi] = 0.0
    sec_idx_in_mask = int(np.argmax(amp_copy))
    second_freq = float(all_freqs[freq_mask][sec_idx_in_mask])
    second_global_idx = int(np.where(freq_mask)[0][sec_idx_in_mask])
    print(f"Second dominant frequency: {second_freq:.4f} Hz")

    # 9. Extract phases, compute time shifts
    ref_phase = np.median([np.angle(sp[dominant_global_idx]) for (_, _, _, sp, _, _) in fft_results])

    shifts: list[float] = []
    phases_dom: list[float] = []
    phases_2nd: list[float] = []

    for start_idx, end_idx, freqs, sp, r_start, r_end in fft_results:
        phi_dom = float(np.angle(sp[dominant_global_idx]))
        phi_2nd = float(np.angle(sp[second_global_idx]))
        delta_phi = float(_wrap_to_pi(np.array([ref_phase - phi_dom]))[0])
        delta_t = delta_phi / (2.0 * np.pi * dominant_freq)
        shifts.append(delta_t)
        phases_dom.append(phi_dom)
        phases_2nd.append(phi_2nd)

    # 10. Second-peak phase consistency check
    shifted_phases_2nd = []
    for phi_2, dt_i in zip(phases_2nd, shifts):
        phi_2s = phi_2 + 2.0 * np.pi * second_freq * dt_i
        shifted_phases_2nd.append(float(_wrap_to_pi(np.array([phi_2s]))[0]))

    circ_std_2nd = _circular_std(np.array(shifted_phases_2nd))
    if circ_std_2nd > 0.8:
        print(f"WARNING: Second peak ({second_freq:.4f} Hz) has inconsistent phase after alignment "
              f"(circular std = {circ_std_2nd:.3f}). Multiple modes cannot be simultaneously phase-aligned.")
    else:
        print(f"Second peak ({second_freq:.4f} Hz) phase reasonably consistent "
              f"(circular std = {circ_std_2nd:.3f}).")

    # 11. Clamp shifts to available budget
    max_earlier = exclude_after - 0.5
    clamped_shifts: list[float] = []
    clamped_count = 0
    for i, dt_i in enumerate(shifts):
        r_start = fft_results[i][4]
        r_end = fft_results[i][5]
        r_len = r_end - r_start
        max_later = max(0.0, r_len - analysis_window_s - 0.5)
        clamped_dt = float(np.clip(dt_i, -max_later, max_earlier))
        if abs(clamped_dt - dt_i) > 1e-9:
            clamped_count += 1
        clamped_shifts.append(clamped_dt)
    if clamped_count:
        print(f"WARNING: {clamped_count} shift(s) clamped to exclusion budget.")

    # 12. Extract and concatenate smushed segments (all bonds, both x and y purecomoving)
    #     Interpolate per-bond for sub-sample precision.
    synth_long_parts: list[np.ndarray] = []
    synth_trans_parts: list[np.ndarray] = []

    for i, (start_idx, end_idx, _, _, r_start, r_end) in enumerate(fft_results):
        dt_i = clamped_shifts[i]
        actual_start = r_start + dt_i
        actual_end = r_end
        if actual_end <= actual_start:
            continue

        n_out = int(round((actual_end - actual_start) / dt)) + 1
        target_t = np.linspace(actual_start, actual_end, n_out)

        # Interpolate each bond column
        long_seg = np.zeros((n_out, n_bonds), dtype=float)
        trans_seg = np.zeros((n_out, n_bonds), dtype=float)
        for b in range(n_bonds):
            long_seg[:, b] = np.interp(target_t, frame_times, long_sig[:, b])
            trans_seg[:, b] = np.interp(target_t, frame_times, trans_sig[:, b])

        synth_long_parts.append(long_seg)
        synth_trans_parts.append(trans_seg)

    if not synth_long_parts:
        raise ValueError("No segments survived concatenation.")

    synth_long = np.concatenate(synth_long_parts, axis=0)
    synth_trans = np.concatenate(synth_trans_parts, axis=0)
    n_synth = synth_long.shape[0]
    print(f"Synthetic: {n_synth} frames, {n_bonds} bonds")

    # 13. Reconstruct time vector
    synth_times = np.arange(n_synth, dtype=float) * dt + float(frame_times[0])
    synth_frames = np.arange(n_synth, dtype=int)

    # 14. For xPositions we need something valid.  Smush the source xPositions
    #     with the same time shifts so the loader can compute purecomoving if
    #     the cache field is ever ignored.
    synth_xpos_parts: list[np.ndarray] = []
    synth_ypos_parts: list[np.ndarray] = []
    synth_apos_parts: list[np.ndarray] = []

    for i in range(len(fft_results)):
        dt_i = clamped_shifts[i]
        r_start = fft_results[i][4]
        r_end = fft_results[i][5]
        actual_start = r_start + dt_i
        actual_end = r_end
        if actual_end <= actual_start:
            continue

        n_out = int(round((actual_end - actual_start) / dt)) + 1
        target_t = np.linspace(actual_start, actual_end, n_out)

        xpos_seg = np.zeros((n_out, n_blocks), dtype=float)
        ypos_seg = np.zeros((n_out, n_blocks), dtype=float)
        apos_seg = np.zeros((n_out, n_blocks), dtype=float)
        for b in range(n_blocks):
            xpos_seg[:, b] = np.interp(target_t, t2x_ftimes, t2x_xpos[:, b])
            ypos_seg[:, b] = np.interp(target_t, t2y_ftimes, t2y_xpos[:, b])
            apos_seg[:, b] = np.interp(target_t, t2a_ftimes, t2a_xpos[:, b])

        synth_xpos_parts.append(xpos_seg)
        synth_ypos_parts.append(ypos_seg)
        synth_apos_parts.append(apos_seg)

    synth_xpos = np.concatenate(synth_xpos_parts, axis=0)
    synth_ypos = np.concatenate(synth_ypos_parts, axis=0)
    synth_apos = np.concatenate(synth_apos_parts, axis=0)

    # 15. Build synthetic dataset name
    synth_name = f"{dataset}HITSYNTH_EA{int(exclude_after)}_EB{int(exclude_before)}"
    out_dir = root / synth_name

    print(f"\nSynthetic dataset: {synth_name}")
    print(f"Output directory: {out_dir}")
    print(f"Per-region shifts (s): {['{:.6f}'.format(s) for s in clamped_shifts]}")

    if dry_run:
        print("Dry run — skipping file write.")
        return

    # 16. Write msgpack files
    for comp, xpos, purecomoving in [
        ("x", synth_xpos, synth_long),
        ("y", synth_ypos, synth_trans),
        ("a", synth_apos, None),
    ]:
        comp_dir = out_dir / "components" / comp
        comp_dir.mkdir(parents=True, exist_ok=True)
        msgpack_path = comp_dir / "track2_permanence.msgpack"

        payload: dict = {
            "blockColors": block_colors,
            "xPositions": xpos.tolist(),
            "frameTimes_s": synth_times.tolist(),
            "frameNumbers": synth_frames.tolist(),
            "originalVideoPath": "",
            "trackingResultsPath": "",
        }
        if purecomoving is not None:
            payload["purecomovingSignal"] = purecomoving.tolist()

        with msgpack_path.open("wb") as fh:
            fh.write(msgpack.packb(_msgpackable(payload), use_bin_type=True))
        print(f"  Wrote {msgpack_path}")

    # manifest.json
    manifest = {
        "dataset": synth_name,
        "tracking_mode": "synthetic_hit_smush",
        "version": 1,
        "source_video": "",
        "params_file": "",
        "track1_file": "",
        "labels_file": "labels.json",
        "components": {
            "x": "components/x/track2_permanence.msgpack",
            "y": "components/y/track2_permanence.msgpack",
            "a": "components/a/track2_permanence.msgpack",
        },
        "rotation_deg": 0,
        "crop_rect": {"x0": 0, "x1": None, "y0": 0, "y1": None},
        "notes": f"Synthetic: phase-aligned inter-hit segments from {dataset} in purecomoving space (EA={exclude_after}s, EB={exclude_before}s)",
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # labels.json
    site_labels = {str(i + 1): str(i + 1) for i in range(n_blocks)}
    labels = {
        "dataset": synth_name,
        "version": 1,
        "bond_labels": {},
        "site_labels": site_labels,
        "disabled": {"sites": [], "components": {}},
        "notes": f"Synthetic labels for {synth_name}",
        "created_by": "synthesize_hit_smushed.py",
    }
    with (out_dir / "labels.json").open("w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)

    print(f"\nWrote synthetic dataset: {synth_name}")


def main() -> int:
    args = build_parser().parse_args()
    synthesize(
        args.dataset,
        exclude_after=args.exclude_after,
        exclude_before=args.exclude_before,
        analysis_window_s=args.analysis_window_s,
        norm_min_hz=args.norm_min_hz,
        norm_max_hz=args.norm_max_hz,
        track_data_root=args.track_data_root,
        dry_run=args.no_write,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
