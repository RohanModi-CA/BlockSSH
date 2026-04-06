#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.GoHit.tools.cli import add_hit_mode_args, describe_hit_region_settings
from analysis.GoHit.tools.hits import (
    HitRegion,
    build_interhit_regions,
    build_posthit_regions,
    load_catalog_if_available,
    summarize_catalog,
)
from analysis.GoHit.tools.region_spectra import compute_region_averaged_fft
from analysis.plotting.common import ensure_parent_dir, render_figure
from analysis.plotting.frequency import (
    plot_average_component_comparison,
    plot_average_spectrum,
    plot_component_pair_frequency_grid,
    plot_pair_frequency_grid,
)
from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.io import load_track2_dataset
from analysis.tools.models import FFTResult, PairFrequencyAnalysisResult
from analysis.tools.signal import compute_complex_spectrogram, compute_one_sided_fft, normalize_processed_signal_rms, preprocess_signal
from analysis.tools.spectrasave import (
    build_default_spectrasave_name,
    resolve_spectrasave_path,
    save_spectrum_msgpack,
)
from analysis.viz import see_fft_xya as legacy_fft


def build_parser():
    parser = legacy_fft.build_parser()
    add_hit_mode_args(parser)
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to exclude after each hit for interhit regions. Default: 1.0",
    )
    parser.add_argument(
        "--exclude-before",
        type=float,
        default=1.0,
        help="Seconds to exclude before the next hit for interhit regions. Default: 1.0",
    )
    parser.add_argument(
        "--hit-window",
        type=float,
        default=5.0,
        help="Window length in seconds for posthit regions. Default: 5.0",
    )
    return parser


def _regions_for_catalog(catalog, *, args, t_stop_s: float) -> list[HitRegion]:
    if args.posthit:
        return build_posthit_regions(
            catalog.hit_times_s,
            t_stop_s=t_stop_s,
            window_s=float(getattr(args, "hit_window", 5.0)),
        )
    return build_interhit_regions(
        catalog.hit_times_s,
        t_stop_s=t_stop_s,
        exclude_after_s=float(getattr(args, "delay", 1.0)),
        exclude_before_s=float(getattr(args, "exclude_before", 1.0)),
    )


def _analyze_component_results(args, *, base_dataset: str, component: str):
    dataset_name = f"{base_dataset}_{component}"
    track2 = load_track2_dataset(dataset=dataset_name, track_data_root=args.track_data_root)
    bond_dataset = load_bond_signal_dataset(
        dataset=dataset_name,
        track_data_root=args.track_data_root,
        bond_spacing_mode=args.bond_spacing_mode,
        component=component,
    )
    catalog = load_catalog_if_available(base_dataset)
    if catalog is None:
        raise ValueError(
            f"No GoHit confirmed hit catalog exists for dataset '{base_dataset}'. "
            f"Run python3 analysis/GoHit/HitReview.py {base_dataset} first."
        )

    t_stop_s = float(np.nanmax(bond_dataset.frame_times_s))
    regions = _regions_for_catalog(catalog, args=args, t_stop_s=t_stop_s)
    if len(regions) == 0:
        raise ValueError(f"No usable {'posthit' if args.posthit else 'interhit'} regions were available")

    results: list[PairFrequencyAnalysisResult] = []
    disabled = set(args.disable or [])
    for pair_idx in range(int(bond_dataset.signal_matrix.shape[1])):
        if pair_idx in disabled:
            continue
        label = bond_dataset.pair_labels[pair_idx] if pair_idx < len(bond_dataset.pair_labels) else "?"
        processed, error_msg = preprocess_signal(
            bond_dataset.frame_times_s,
            bond_dataset.signal_matrix[:, pair_idx],
            longest=args.longest,
            handlenan=args.handlenan,
        )
        if processed is None:
            results.append(
                PairFrequencyAnalysisResult(
                    pair_index=pair_idx,
                    label=label,
                    processed=None,
                    fft_result=None,
                    spectrogram_result=None,
                    error_message=error_msg,
                )
            )
            continue

        if args.timeseriesnorm:
            processed_norm, _, norm_error = normalize_processed_signal_rms(processed)
            if processed_norm is None:
                results.append(
                    PairFrequencyAnalysisResult(
                        pair_index=pair_idx,
                        label=label,
                        processed=None,
                        fft_result=None,
                        spectrogram_result=None,
                        error_message=f"time-series RMS normalization failed: {norm_error}",
                    )
                )
                continue
            processed = processed_norm

        region_average = compute_region_averaged_fft(
            processed,
            regions,
            grid_mode="coarsest" if args.coarsest else "finest",
            interp_kind=args.interp_kind,
        )
        fft_result = region_average.fft_result
        spectrogram_result = compute_complex_spectrogram(processed.y, processed.Fs, args.sliding_len_s)
        spectrogram_error = None if spectrogram_result is not None else "window too short"
        error_message = None if fft_result is not None else (
            f"no valid {'posthit' if args.posthit else 'interhit'} FFT regions"
        )
        results.append(
            PairFrequencyAnalysisResult(
                pair_index=pair_idx,
                label=label,
                processed=processed,
                fft_result=fft_result,
                spectrogram_result=spectrogram_result,
                spectrogram_error_message=spectrogram_error,
                error_message=error_message,
            )
        )

    print(f"Track2 ({component}): {track2.track2_path}")
    dt = np.diff(track2.frame_times_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size > 0:
        approx_fs = 1.0 / float(np.median(dt))
        approx_nyquist = 0.5 * approx_fs
        print(f"Approx sampling rate ({component}): {approx_fs:.4f} Hz | Approx Nyquist: {approx_nyquist:.4f} Hz")
    print(summarize_catalog(catalog))
    for line in describe_hit_region_settings(
        posthit=bool(args.posthit),
        delay=float(args.delay),
        exclude_before=float(args.exclude_before),
        hit_window=float(args.hit_window),
    ):
        print(line)
    print(f"Usable regions: {len(regions)}")

    return results, track2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.track2 is not None:
        print("Error: GoHit FFT currently requires a dataset stem rather than --track2", file=sys.stderr)
        return 1
    if args.dataset is None:
        print("Error: GoHit FFT requires a dataset argument", file=sys.stderr)
        return 1
    explicit_welch = "--welch" in sys.argv[1:]
    if explicit_welch:
        print("Error: GoHit FFT uses interhit/posthit averaged FFTs, not Welch spectra", file=sys.stderr)
        return 1
    args.welch = False

    args.disable = legacy_fft._convert_pair_indices_to_zero_based(args.disable)
    args.only_pairs = legacy_fft._convert_pair_indices_to_zero_based(args.only_pairs)

    freq_window_error = legacy_fft.validate_frequency_window_args(args)
    if freq_window_error is not None:
        print(freq_window_error, file=sys.stderr)
        return 1
    tickspace_error = legacy_fft.validate_tickspace_arg(args)
    if tickspace_error is not None:
        print(tickspace_error, file=sys.stderr)
        return 1
    flatten_error = legacy_fft._validate_flatten_args(args)
    if flatten_error is not None:
        print(flatten_error, file=sys.stderr)
        return 1

    base_dataset = legacy_fft._strip_component_suffix(args.dataset)
    disabled_components = set(args.disable_component)
    only_components = set(args.only_component)
    component_results: dict[str, list[PairFrequencyAnalysisResult]] = {}
    component_track2: dict[str, object] = {}

    try:
        for component in legacy_fft.COMPONENT_SUFFIXES:
            if component in disabled_components:
                continue
            if only_components and component not in only_components:
                continue
            dataset_name = f"{base_dataset}_{component}"
            try:
                results, track2 = _analyze_component_results(args, base_dataset=base_dataset, component=component)
            except FileNotFoundError:
                continue
            if args.only_pairs is not None:
                results = legacy_fft._filter_to_only_pairs(results, set(args.only_pairs))
            component_results[component] = results
            component_track2[component] = track2

        if len(component_results) == 0:
            print("Error: No component sibling datasets were found for GoHit FFT", file=sys.stderr)
            return 1

        if args.average:
            averaged_raw_by_component = {}
            for component, results in component_results.items():
                try:
                    averaged_raw_by_component[component] = legacy_fft._average_result_from_pair_results(
                        results,
                        dataset_name=(
                            component_track2[component].dataset_name
                            or Path(component_track2[component].track2_path).resolve().parent.name
                        ),
                        use_welch=False,
                        args=args,
                    )
                except Exception as e:
                    print(f"Warning: Skipping {component} averaging: {e}", file=sys.stderr)

            if not averaged_raw_by_component:
                print("Error: No valid bond spectra were available to average", file=sys.stderr)
                return 1

            averaged_by_component, flattening_by_component = legacy_fft._maybe_apply_flattening(
                args,
                averaged_raw_by_component,
            )
            dataset_name = (
                component_track2[next(iter(component_track2))].dataset_name
                or Path(component_track2[next(iter(component_track2))].track2_path).resolve().parent.name
            )
            legacy_fft._maybe_emit_flatten_plot(
                args,
                dataset_name=dataset_name,
                results_by_component=averaged_raw_by_component,
                flattening_by_component=flattening_by_component,
            )

            if args.spectrasave is not None:
                available_components = list(averaged_by_component)
                if len(available_components) == 1:
                    component = available_components[0]
                else:
                    component = legacy_fft._prompt_save_component_choice(available_components, spectrum_kind="fft")

                result = averaged_by_component[component]
                default_name = build_default_spectrasave_name(dataset_name, "average-bonds", component, "fft")
                export_path = resolve_spectrasave_path(
                    args.spectrasave,
                    default_name=default_name,
                )
                if export_path is not None:
                    save_spectrum_msgpack(
                        export_path,
                        freq=result.freq_grid,
                        amplitude=result.avg_amp,
                        label=f"{dataset_name} average bonds {component} FFT",
                        metadata={
                            "sourceKind": "single",
                            "spectrumKind": "fft",
                            "dataset": dataset_name,
                            "component": component,
                            "averageBonds": True,
                            "contributors": len(result.contributors),
                        },
                    )
                    print(f"Spectrum saved to: {export_path}")

            if len(averaged_by_component) == 1:
                only_component = next(iter(averaged_by_component))
                fig = plot_average_spectrum(
                    averaged_by_component[only_component],
                    full_image=args.full_image,
                    plot_scale=args.sliding_plot_scale if args.full_image else ("log" if args.fft_log else "linear"),
                    cmap_index=args.cm,
                    title=args.title or f"{dataset_name} average bonds",
                    tickspace_hz=args.tickspace_hz,
                )
            else:
                fig = plot_average_component_comparison(
                    averaged_by_component,
                    full_image=args.full_image,
                    plot_scale=args.sliding_plot_scale if args.full_image else ("log" if args.fft_log else "linear"),
                    cmap_index=args.cm,
                    title=args.title or f"{dataset_name} average bonds",
                    tickspace_hz=args.tickspace_hz,
                )
        else:
            component_results = legacy_fft._maybe_apply_flattening_to_component_results(args, component_results)
            if len(component_results) == 1:
                only_component = next(iter(component_results))
                fig = plot_pair_frequency_grid(
                    component_results[only_component],
                    fft_log=args.fft_log,
                    sliding_plot_scale=args.sliding_plot_scale,
                    only=args.only,
                    full_image=args.full_image,
                    full_couple=args.full_couple,
                    fft_min_hz=args.freq_min_hz,
                    fft_max_hz=args.freq_max_hz,
                    sliding_min_hz=args.freq_min_hz,
                    sliding_max_hz=args.freq_max_hz,
                    time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                    cmap_index=args.cm,
                    title=args.title,
                    tickspace_hz=args.tickspace_hz,
                )
            else:
                fig = plot_component_pair_frequency_grid(
                    component_results,
                    fft_log=args.fft_log,
                    welch_log=False,
                    sliding_plot_scale=args.sliding_plot_scale,
                    only=args.only,
                    full_image=args.full_image,
                    full_couple=args.full_couple,
                    use_welch=False,
                    fft_min_hz=args.freq_min_hz,
                    fft_max_hz=args.freq_max_hz,
                    welch_min_hz=args.freq_min_hz,
                    welch_max_hz=args.freq_max_hz,
                    sliding_min_hz=args.freq_min_hz,
                    sliding_max_hz=args.freq_max_hz,
                    time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
                    cmap_index=args.cm,
                    title=args.title,
                    tickspace_hz=args.tickspace_hz,
                )

        backend = plt.get_backend().lower()
        if args.save is not None and "agg" in backend:
            save_path = ensure_parent_dir(args.save)
            fig.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")
            plt.close(fig)
        else:
            render_figure(fig, save=args.save)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
