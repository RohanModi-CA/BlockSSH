#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def add_repo_root_to_path() -> Path:
    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (parent / "analysis").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root")


add_repo_root_to_path()

from analysis.go.Play.fft_flattening import compute_flattened_component_spectra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten a dataset, take the top N peaks, and visualize their sum/harmonic candidates."
    )
    parser.add_argument(
        "--dataset",
        default="IMG_0681_rot270",
        help="Dataset base name (default IMG_0681_rot270).",
    )
    parser.add_argument(
        "--bond-spacing-mode",
        choices=("purecomoving", "default"),
        default="purecomoving",
        help="Which bond spacing mode to flatten (default comoving).",
    )
    parser.add_argument(
        "--component",
        choices=("x", "y", "a"),
        default="x",
        help="Component to plot (default x).",
    )
    parser.add_argument("--top-n", type=int, default=7, help="Show the top-N peaks (default 7).")
    parser.add_argument("--peak-prominence", type=float, default=0.01, help="Minimum prominence for peaks.")
    parser.add_argument(
        "--min-peak-separation-hz",
        type=float,
        default=0.1,
        help="Minimum frequency separation (Hz) between peaks (default 0.1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/play2b_peaks.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure.")
    return parser


def flattened_spectrum(dataset: str, component: str, bond_spacing: str) -> tuple[np.ndarray, np.ndarray]:
    results = compute_flattened_component_spectra(
        dataset=dataset,
        bond_spacing_mode=bond_spacing,
        components=(component,),
        use_welch=True,
    )
    res = results[component]
    return np.asarray(res.freq_hz, dtype=float), np.asarray(res.flattened, dtype=float)


def top_peaks(freqs: np.ndarray, amplitude: np.ndarray, *, n: int, prominence: float, min_separation_hz: float = 0.1) -> np.ndarray:
    # Calculate the minimum distance in samples based on frequency resolution
    if freqs.size > 1:
        df = np.mean(np.diff(freqs))
        distance_samples = max(1, int(np.ceil(min_separation_hz / df)))
    else:
        distance_samples = 1 # No meaningful distance if only one frequency point
    indices, props = find_peaks(amplitude, prominence=prominence, distance=distance_samples)
    if indices.size == 0:
        return np.array([], dtype=float)
    sorted_by_amp = indices[np.argsort(amplitude[indices])[::-1]]
    selected = sorted_by_amp[:n]
    return freqs[selected]


def build_candidate_lines(peaks: Iterable[float]) -> list[tuple[float, list[str], int]]:
    # Use a dictionary to group labels by frequency
    # key: frequency (float), value: list of unique descriptive label strings
    grouped_by_freq: dict[float, set[str]] = {}
    peaks = list(peaks) # top_peaks returns sorted frequencies

    for idx, base in enumerate(peaks):
        # Harmonic
        harmonic = 2.0 * base
        if harmonic not in grouped_by_freq:
            grouped_by_freq[harmonic] = set()
        grouped_by_freq[harmonic].add(f"2x{base:.2f}")

        # Sums and Differences
        for partner in peaks[idx:]:
            # Sum: base + partner (already base <= partner from iteration)
            sum_freq = base + partner
            if sum_freq not in grouped_by_freq:
                grouped_by_freq[sum_freq] = set()
            grouped_by_freq[sum_freq].add(f"{base:.2f}+{partner:.2f}")

            # Difference: |base - partner|. Since base <= partner, this is partner - base.
            if base < partner: # Ensures unique pair and positive difference
                diff_freq = partner - base
                if diff_freq not in grouped_by_freq:
                    grouped_by_freq[diff_freq] = set()
                grouped_by_freq[diff_freq].add(f"|{partner:.2f}-{base:.2f}|") # Canonical label

    # Sort the unique frequencies and assign numerical labels
    sorted_unique_freqs = sorted(grouped_by_freq.keys())
    numbered_lines: list[tuple[float, list[str], int]] = []
    for i, freq in enumerate(sorted_unique_freqs):
        # Sort labels for consistent key display
        combined_labels = sorted(list(grouped_by_freq[freq]))
        numbered_lines.append((freq, combined_labels, i + 1))

    return numbered_lines


def clamp_label_y(ax, y_value: float) -> float:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        return y_value
    return np.clip(y_value, ymin + 0.05 * span, ymax - 0.05 * span)


def label_slot_y(ax: plt.Axes, slot: int, top_fraction: float) -> float:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        return ymax
    return clamp_label_y(ax, ymax - top_fraction * span - slot * 0.04 * span)


def plot_peaks(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    peaks: np.ndarray,
    lines: list[tuple[float, str, int]], # Modified to accept (freq, original_label, num_label)
    output: Path,
    show: bool,
) -> None:
    fig, ax_spectrum = plt.subplots(figsize=(11, 6))
    mask = (freqs >= 0.1) & (freqs <= 30)
    ax_spectrum.plot(freqs[mask], amplitude[mask], color="black", lw=1.2)
    ax_spectrum.set_yscale("log")
    ax_spectrum.set_xlim(freqs[mask][0], freqs[mask][-1])
    ax_spectrum.set_xlabel("Frequency (Hz)")
    ax_spectrum.set_ylabel("Flattened amplitude")
    ax_spectrum.set_title("Top peaks + candidate sum/harmonic/difference lines")

    # Store labels for the legend
    legend_labels: list[str] = []
    for idx, (freq_line, original_label, num_label) in enumerate(lines):
        legend_labels.append(f"{num_label}: {original_label}")

        if freq_line <= 0 or freq_line > freqs[-1]:
            continue
        ax_spectrum.axvline(freq_line, color="#d95f02", ls=":", lw=0.8, alpha=0.7)
        
        # Use a more spread-out staggering pattern
        slot = idx % 5 # Increased number of slots for staggering
        ypos = label_slot_y(ax_spectrum, slot, top_fraction=0.08) # Increased top_fraction and vertical step
        ax_spectrum.text(
            freq_line,
            ypos,
            str(num_label), # Plot numerical label
            rotation=90,
            ha="center",
            va="top",
            fontsize=6,
            color="#d95f02",
            clip_on=True,
        )

    for idx, freq_pe in enumerate(peaks):
        ax_spectrum.axvline(freq_pe, color="#1b9e77", ls="--", lw=1.1, alpha=0.8)
        base_amp = amplitude[np.argmin(np.abs(freqs - freq_pe))]
        slot = idx % 3
        ypos = clamp_label_y(ax_spectrum, base_amp * (1.05 + 0.03 * slot))
        ax_spectrum.text(
            freq_pe,
            ypos,
            f"{freq_pe:.3f}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=True,
        )

def plot_peaks(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    peaks: np.ndarray,
    lines: list[tuple[float, list[str], int]], # Modified to accept (freq, list[original_labels], num_label)
    output: Path,
    show: bool,
) -> None:
    # Key parameters
    num_key_cols = 4
    line_height_factor = 0.25 # Height for each legend row in inches
    key_padding_inches = 0.5

    # Calculate required height for the key based on how many labels are associated with each frequency
    # We'll use a simpler estimate: roughly one row per unique frequency, plus some for longer combined labels
    key_rows_estimate = (len(lines) + num_key_cols - 1) // num_key_cols
    key_area_height_inches = key_rows_estimate * line_height_factor + key_padding_inches

    fig, (ax_spectrum, ax_key) = plt.subplots(
        2, 1, figsize=(11, 6 + key_area_height_inches), # Adjust figure height dynamically
        gridspec_kw={'height_ratios': [6, key_area_height_inches]}, # Allocate space
        constrained_layout=False # Manual layout for key positioning
    )

    # Hide the key subplot frame and ticks
    ax_key.axis('off')
    ax_key.set_xlim(0, 1) # Set limits for text placement
    ax_key.set_ylim(0, 1) # Set limits for text placement

    mask = (freqs >= 0.1) & (freqs <= 30)
    ax_spectrum.plot(freqs[mask], amplitude[mask], color="black", lw=1.2)
    ax_spectrum.set_yscale("log")
    ax_spectrum.set_xlim(freqs[mask][0], freqs[mask][-1])
    ax_spectrum.set_xlabel("Frequency (Hz)")
    ax_spectrum.set_ylabel("Flattened amplitude")
    ax_spectrum.set_title("Top peaks + candidate sum/harmonic/difference lines")

    # Store labels for the key subplot
    key_labels_for_display: list[str] = []
    for idx, (freq_line, combined_labels, num_label) in enumerate(lines):
        key_labels_for_display.append(f"{num_label}: {', '.join(combined_labels)}")

        if freq_line <= 0 or freq_line > freqs[-1]:
            continue
        ax_spectrum.axvline(freq_line, color="#d95f02", ls=":", lw=0.8, alpha=0.7)
        
        # Significantly increased staggering for numerical labels on the plot
        slot = idx % 25 # Increased slots for vertical staggering
        ypos = label_slot_y(ax_spectrum, slot, top_fraction=0.03) # Smaller top_fraction, increased step per slot
        ax_spectrum.text(
            freq_line,
            ypos,
            str(num_label), # Plot numerical label
            rotation=90,
            ha="center",
            va="top",
            fontsize=6,
            color="#d95f02",
            clip_on=True,
        )

    for idx, freq_pe in enumerate(peaks):
        ax_spectrum.axvline(freq_pe, color="#1b9e77", ls="--", lw=1.1, alpha=0.8)
        base_amp = amplitude[np.argmin(np.abs(freqs - freq_pe))]
        slot = idx % 3
        ypos = clamp_label_y(ax_spectrum, base_amp * (1.05 + 0.03 * slot))
        ax_spectrum.text(
            freq_pe,
            ypos,
            f"{freq_pe:.3f}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=True,
        )

    # Place key entries in ax_key
    col_width = 1.0 / num_key_cols
    vertical_offset_per_row = 0.08 # Adjust this for spacing between rows in the key
    start_y_pos = 1.0 # Start from the top of the ax_key subplot

    # Track current row and column to lay out text
    current_row = 0
    current_col_item_count = [0] * num_key_cols # Count items per column to determine max height

    for i, label_text in enumerate(key_labels_for_display):
        col = i % num_key_cols
        # Simple row increment for now, will refine if text lines wrap
        row = i // num_key_cols # This assumes roughly equal height per item

        x_pos = col * col_width
        y_pos = start_y_pos - (row * vertical_offset_per_row)
        
        ax_key.text(x_pos, y_pos, label_text, fontsize=6, color='#d95f02', va='top', transform=ax_key.transAxes)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    freqs, amplitude = flattened_spectrum(
        dataset=args.dataset,
        component=args.component,
        bond_spacing=args.bond_spacing_mode,
    )
    peaks = top_peaks(freqs, amplitude, n=args.top_n, prominence=args.peak_prominence, min_separation_hz=args.min_peak_separation_hz)
    combos = build_candidate_lines(peaks)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_peaks(freqs, amplitude, peaks, combos, output, args.show)
    print(f"Saved hybrid peak figure to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
