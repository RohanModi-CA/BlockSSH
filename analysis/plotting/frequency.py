from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tools.models import (
    AverageSpectrumResult,
    PairFrequencyAnalysisResult,
    PairWelchFrequencyAnalysisResult,
    SiteAmplitudeAnalysisResult,
)
from .common import (
    apply_major_tick_spacing,
    centers_to_edges,
    colormap_name,
    resolve_clipped_window,
    robust_nonnegative_norm,
)
from .indexed import overlay_indexed_points

COMPONENT_COLORS = {
    "x": "orange",
    "y": "tab:blue",
    "a": "lightgrey",
}


def _set_panel_message(
    *,
    ax_fft=None,
    ax_spec=None,
    only: str | None = None,
    full_image: bool = False,
    title: str,
) -> None:
    if only != "sliding" and ax_fft is not None:
        ax_fft.set_title(title)
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")

    if only != "fft" and ax_spec is not None:
        ax_spec.set_title(title)
        ax_spec.set_xlabel("Arbitrary X" if full_image else "Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")


def _plot_fft_curve_panel(
    ax,
    *,
    freq: np.ndarray,
    amp: np.ndarray,
    log_scale: bool,
    x_min: float,
    x_max: float,
    title: str,
    x_tickspace_hz: float | None = None,
) -> None:
    if log_scale:
        ax.semilogy(freq, amp, linewidth=1)
        positive_vals = amp[amp > 0]
        if positive_vals.size > 0:
            ymin = np.percentile(positive_vals, 0.1) * 0.7
            ymax = np.max(positive_vals) * 1.3
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                ax.set_ylim(ymin, ymax)
    else:
        ax.plot(freq, amp, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(x_min, x_max)
    apply_major_tick_spacing(ax, x_tickspace_hz, axis="x")
    ax.grid(True, alpha=0.3)


def _plot_spectrum_curve_panel(
    ax,
    *,
    freq: np.ndarray,
    amp: np.ndarray,
    log_scale: bool,
    x_min: float,
    x_max: float,
    title: str,
    y_label: str,
    x_tickspace_hz: float | None = None,
) -> None:
    _plot_fft_curve_panel(
        ax,
        freq=freq,
        amp=amp,
        log_scale=log_scale,
        x_min=x_min,
        x_max=x_max,
        title=title,
        x_tickspace_hz=x_tickspace_hz,
    )
    ax.set_ylabel(y_label)


def _plot_frequency_image(
    fig,
    ax,
    *,
    freq: np.ndarray,
    amp: np.ndarray,
    plot_scale: str,
    cmap_index: int,
    y_min: float,
    y_max: float,
    x_label: str = "Arbitrary X",
    x_max: float = 1.0,
    title: str,
    linear_color_label: str = "Amplitude",
    log_color_label: str = "Amplitude (dB)",
    reference_amp_for_norm: np.ndarray | None = None,
    overlay: dict | None = None,
    show_colorbar: bool = True,
    annotate_range: bool = False,
    y_tickspace_hz: float | None = None,
) -> None:
    mask = (freq >= y_min) & (freq <= y_max)
    if not np.any(mask):
        ax.set_title(title + " - no frequencies in range")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Frequency (Hz)")
        return

    freq_img = freq[mask]
    amp_img = amp[mask]

    x_cols = 64
    image_2d = np.tile(amp_img[:, None], (1, x_cols))
    eps = np.finfo(float).eps

    if plot_scale == "log":
        image_plot = 20.0 * np.log10(image_2d + eps)
        pcm_norm = None
        color_label = log_color_label
    else:
        image_plot = image_2d
        color_label = linear_color_label
        norm_source = reference_amp_for_norm if reference_amp_for_norm is not None else image_plot
        pcm_norm = robust_nonnegative_norm(norm_source)

    x_edges = np.linspace(0.0, x_max, x_cols + 1)
    fallback_step = float(np.median(np.diff(freq_img))) if freq_img.size > 1 else max(1e-6, y_max - y_min)
    y_edges = centers_to_edges(freq_img, fallback_step=fallback_step)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        image_plot,
        shading="flat",
        cmap=colormap_name(cmap_index),
        norm=pcm_norm,
    )
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=color_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)
    apply_major_tick_spacing(ax, y_tickspace_hz, axis="y")
    ax.set_title(title)

    if annotate_range:
        finite_vals = image_plot[np.isfinite(image_plot)]
        if finite_vals.size > 0:
            stat_label = "dB" if plot_scale == "log" else ""
            stat_suffix = f" {stat_label}" if stat_label else ""
            ax.text(
                0.015,
                0.985,
                f"max {np.max(finite_vals):.1f}{stat_suffix}\nmin {np.min(finite_vals):.1f}{stat_suffix}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.45,
                },
            )

    if overlay is None:
        if x_max == 1.0:
            ax.set_xticks([])
    else:
        overlay_indexed_points(
            ax,
            overlay["x_values"],
            overlay["y_values"],
            mode=overlay.get("mode", "scatter"),
            color=overlay.get("color", "red"),
            marker_size=float(overlay.get("marker_size", 30.0)),
        )
        if overlay.get("integer_ticks", True):
            x_ticks_stop = int(np.ceil(x_max))
            ax.set_xticks(np.arange(0, x_ticks_stop, 1))


def _plot_spectrogram_panel(
    fig,
    ax,
    *,
    f_spec: np.ndarray,
    t_spec: np.ndarray,
    s_complex: np.ndarray,
    t_start: float,
    plot_scale: str,
    cmap_index: int,
    y_min: float,
    y_max: float,
    title: str,
    time_interval: tuple[float, float] | None = None,
    show_colorbar: bool = True,
    annotate_range: bool = False,
    y_tickspace_hz: float | None = None,
) -> None:
    mag = np.abs(s_complex)
    eps = np.finfo(float).eps

    if plot_scale == "log":
        s_plot = 20.0 * np.log10(mag + eps)
        pcm_norm = None
        color_label = "dB"
    else:
        s_plot = mag ** 2
        pcm_norm = robust_nonnegative_norm(s_plot)
        color_label = "Power"

    t_global = t_spec + t_start
    t_step = (t_global[1] - t_global[0]) if t_global.size > 1 else 1.0
    f_step = (f_spec[1] - f_spec[0]) if f_spec.size > 1 else max(1e-6, y_max - y_min)

    t_edges = centers_to_edges(t_global, fallback_step=t_step)
    f_edges = centers_to_edges(f_spec, fallback_step=f_step)

    pcm = ax.pcolormesh(
        t_edges,
        f_edges,
        s_plot,
        shading="flat",
        cmap=colormap_name(cmap_index),
        norm=pcm_norm,
    )
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=color_label)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(y_min, y_max)
    apply_major_tick_spacing(ax, y_tickspace_hz, axis="y")
    if time_interval is not None:
        ax.set_xlim(*time_interval)

    if annotate_range:
        finite_vals = s_plot[np.isfinite(s_plot)]
        if finite_vals.size > 0:
            stat_label = "dB" if plot_scale == "log" else ""
            stat_suffix = f" {stat_label}" if stat_label else ""
            ax.text(
                0.015,
                0.985,
                f"max {np.max(finite_vals):.1f}{stat_suffix}\nmin {np.min(finite_vals):.1f}{stat_suffix}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.45,
                },
            )


def _apply_compact_image_axis_style(
    ax,
    *,
    show_right_ylabel: bool,
) -> None:
    if show_right_ylabel:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel("Frequency (Hz)", labelpad=10)
        ax.tick_params(axis="y", labelright=True, labelleft=False)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False, labelright=False)

    ax.tick_params(axis="both", which="both", length=2, pad=1)


def _link_fft_frequency_to_image_frequency(
    fft_axes: list,
    image_axes: list,
) -> None:
    if len(fft_axes) == 0 or len(image_axes) == 0:
        return

    sync_state = {"active": False}

    def _set_all_fft_xlim(xlim):
        xlim_tuple = tuple(xlim)
        for ax in fft_axes:
            if tuple(ax.get_xlim()) != xlim_tuple:
                ax.set_xlim(xlim_tuple)

    def _set_all_image_ylim(ylim):
        ylim_tuple = tuple(ylim)
        for ax in image_axes:
            if tuple(ax.get_ylim()) != ylim_tuple:
                ax.set_ylim(ylim_tuple)

    def _sync_from_fft(changed_ax):
        if sync_state["active"]:
            return
        sync_state["active"] = True
        try:
            xlim = changed_ax.get_xlim()
            _set_all_fft_xlim(xlim)
            _set_all_image_ylim(xlim)
        finally:
            sync_state["active"] = False

    def _sync_from_image(changed_ax):
        if sync_state["active"]:
            return
        sync_state["active"] = True
        try:
            ylim = changed_ax.get_ylim()
            _set_all_image_ylim(ylim)
            _set_all_fft_xlim(ylim)
        finally:
            sync_state["active"] = False

    for ax in fft_axes:
        ax.callbacks.connect("xlim_changed", _sync_from_fft)

    for ax in image_axes:
        ax.callbacks.connect("ylim_changed", _sync_from_image)


def plot_pair_frequency_grid(
    results: list[PairFrequencyAnalysisResult],
    *,
    fft_log: bool = False,
    sliding_plot_scale: str = "log",
    only: str | None = None,
    full_image: bool = False,
    full_couple: bool = False,
    fft_min_hz: float | None = None,
    fft_max_hz: float | None = None,
    sliding_min_hz: float | None = None,
    sliding_max_hz: float | None = None,
    time_interval: tuple[float, float] | None = None,
    cmap_index: int = 6,
    title: str | None = None,
    tickspace_hz: float | None = None,
):
    if len(results) == 0:
        raise ValueError("No pair results to plot")

    ncols = 1 if only in ("fft", "sliding") else 2
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=ncols,
        figsize=(14, 3.5 * len(results)),
        constrained_layout=True,
    )

    if ncols == 1:
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_2d(axes)

    for row_idx, result in enumerate(results):
        if only == "fft":
            ax_fft = axes[row_idx]
            ax_spec = None
        elif only == "sliding":
            ax_fft = None
            ax_spec = axes[row_idx]
        else:
            ax_fft = axes[row_idx][0]
            ax_spec = axes[row_idx][1]

        pair_title_prefix = f"Pair {result.pair_index} ({str(result.label).upper()})"

        if result.error_message is not None or result.processed is None or result.fft_result is None:
            _set_panel_message(
                ax_fft=ax_fft,
                ax_spec=ax_spec,
                only=only,
                full_image=full_image,
                title=f"{pair_title_prefix} - {result.error_message or 'invalid data'}",
            )
            continue

        if only != "sliding":
            try:
                fft_xlim_min, fft_xlim_max = resolve_clipped_window(
                    max(0.0, float(result.fft_result.freq[0])),
                    min(float(result.processed.nyquist), float(result.fft_result.freq[-1])),
                    fft_min_hz,
                    fft_max_hz,
                )
            except ValueError:
                _set_panel_message(
                    ax_fft=ax_fft,
                    ax_spec=ax_spec,
                    only=only,
                    full_image=full_image,
                    title=f"{pair_title_prefix} - invalid frequency range",
                )
                continue
            _plot_fft_curve_panel(
                ax_fft,
                freq=result.fft_result.freq,
                amp=result.fft_result.amplitude,
                log_scale=fft_log,
                x_min=fft_xlim_min,
                x_max=fft_xlim_max,
                title=f"{pair_title_prefix} FFT | {result.processed.proc_msg}",
                x_tickspace_hz=tickspace_hz,
            )

        if only == "fft":
            continue

        try:
            y_min, y_max = resolve_clipped_window(
                max(0.01, float(result.fft_result.freq[0])),
                min(float(result.processed.nyquist), float(result.fft_result.freq[-1])),
                sliding_min_hz,
                sliding_max_hz,
            )
        except ValueError:
            _set_panel_message(
                ax_fft=ax_fft,
                ax_spec=ax_spec,
                only=only,
                full_image=full_image,
                title=f"{pair_title_prefix} - invalid frequency range",
            )
            continue

        if full_image:
            _plot_frequency_image(
                fig,
                ax_spec,
                freq=result.fft_result.freq,
                amp=result.fft_result.amplitude,
                plot_scale=sliding_plot_scale,
                cmap_index=cmap_index,
                y_min=y_min,
                y_max=y_max,
                title=f"{pair_title_prefix} Full FFT Image",
                linear_color_label="Amplitude",
                log_color_label="Amplitude (dB)",
                y_tickspace_hz=tickspace_hz,
            )
            continue

        if result.spectrogram_result is None:
            _set_panel_message(
                ax_fft=ax_fft,
                ax_spec=ax_spec,
                only=only,
                full_image=full_image,
                title=f"{pair_title_prefix} - {result.spectrogram_error_message or 'window too short'}",
            )
            continue

        _plot_spectrogram_panel(
            fig,
            ax_spec,
            f_spec=result.spectrogram_result.f,
            t_spec=result.spectrogram_result.t,
            s_complex=result.spectrogram_result.S_complex,
            t_start=float(result.processed.t[0]),
            plot_scale=sliding_plot_scale,
            cmap_index=cmap_index,
            y_min=y_min,
            y_max=y_max,
            title=f"{pair_title_prefix} Sliding FFT",
            time_interval=time_interval,
            y_tickspace_hz=tickspace_hz,
        )

    if full_couple and full_image and only != "sliding":
        if ncols == 1:
            fft_axes = list(np.atleast_1d(axes))
            image_axes: list = []
        else:
            fft_axes = [axes[row_idx][0] for row_idx in range(len(results))]
            image_axes = [axes[row_idx][1] for row_idx in range(len(results))]
        _link_fft_frequency_to_image_frequency(fft_axes, image_axes)

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def plot_component_pair_frequency_grid(
    component_results: dict[str, list[PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult]],
    *,
    fft_log: bool = False,
    welch_log: bool = False,
    sliding_plot_scale: str = "log",
    only: str | None = None,
    full_image: bool = False,
    full_couple: bool = False,
    use_welch: bool = False,
    fft_min_hz: float | None = None,
    fft_max_hz: float | None = None,
    welch_min_hz: float | None = None,
    welch_max_hz: float | None = None,
    sliding_min_hz: float | None = None,
    sliding_max_hz: float | None = None,
    time_interval: tuple[float, float] | None = None,
    cmap_index: int = 6,
    title: str | None = None,
    tickspace_hz: float | None = None,
):
    if len(component_results) == 0:
        raise ValueError("No component results to plot")

    components = [component for component in ("x", "y", "a") if component in component_results]
    if len(components) == 0:
        raise ValueError("No component results to plot")

    ref_results = component_results[components[0]]
    if len(ref_results) == 0:
        raise ValueError("No pair results to plot")

    ref_keys = [(result.pair_index, result.label) for result in ref_results]
    for component in components[1:]:
        keys = [(result.pair_index, result.label) for result in component_results[component]]
        if keys != ref_keys:
            raise ValueError("Component datasets do not share the same bond layout after filtering")

    ncols = len(components) if only == "sliding" else 1 if only == "fft" else 1 + len(components)
    width_ratios = None
    if only is None:
        image_panel_width = 0.42 if full_image else 0.5
        width_ratios = [1.0 + len(components) * (1.0 - image_panel_width)] + [image_panel_width] * len(components)
    use_compact_image_layout = only != "fft"
    fig, axes = plt.subplots(
        nrows=len(ref_results),
        ncols=ncols,
        figsize=((4.7 if use_compact_image_layout else 5.2) * ncols, 3.5 * len(ref_results)),
        constrained_layout=not use_compact_image_layout,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios is not None else None,
    )

    if len(ref_results) == 1 and ncols == 1:
        axes_grid = np.array([[axes]], dtype=object)
    elif len(ref_results) == 1:
        axes_grid = np.array([np.atleast_1d(axes)], dtype=object)
    elif ncols == 1:
        axes_grid = np.asarray(axes, dtype=object)[:, None]
    else:
        axes_grid = np.asarray(axes, dtype=object)

    if use_compact_image_layout:
        fig.subplots_adjust(
            left=0.055,
            right=0.97,
            bottom=0.07,
            top=0.94 if title else 0.98,
            wspace=0.02,
            hspace=0.22,
        )

        image_col_offset = 0 if only == "sliding" else 1
        lead_ax = axes_grid[0, image_col_offset]
        for row_idx in range(len(ref_results)):
            for comp_idx in range(len(components)):
                ax = axes_grid[row_idx, image_col_offset + comp_idx]
                if ax is lead_ax:
                    continue
                ax.sharey(lead_ax)
                ax.sharex(lead_ax)
        if only != "sliding":
            lead_fft_ax = axes_grid[0, 0]
            for row_idx in range(1, len(ref_results)):
                axes_grid[row_idx, 0].sharex(lead_fft_ax)

    for row_idx, ref_result in enumerate(ref_results):
        pair_title_prefix = f"Pair {ref_result.pair_index} ({str(ref_result.label).upper()})"
        row_component_results = {
            component: component_results[component][row_idx]
            for component in components
        }

        col_offset = 0
        if only != "sliding":
            ax_fft = axes_grid[row_idx, 0]
            col_offset = 1

            spectrum_panel_name = "Welch" if use_welch else "FFT"
            valid_fft_results = [
                (component, result)
                for component, result in row_component_results.items()
                if result.error_message is None
                and result.processed is not None
                and (
                    getattr(result, "welch_result", None) is not None
                    if use_welch
                    else getattr(result, "fft_result", None) is not None
                )
            ]

            if len(valid_fft_results) == 0:
                error_bits = [
                    f"{component}: {result.error_message or 'invalid data'}"
                    for component, result in row_component_results.items()
                ]
                _set_panel_message(
                    ax_fft=ax_fft,
                    only="fft",
                    title=f"{pair_title_prefix} {spectrum_panel_name} - {' | '.join(error_bits)}",
                )
            else:
                supported_min = max(
                    max(0.0, float(getattr(result, "welch_result" if use_welch else "fft_result").freq[0]))
                    for _, result in valid_fft_results
                )
                supported_max = min(
                    min(
                        float(result.processed.nyquist),
                        float(getattr(result, "welch_result" if use_welch else "fft_result").freq[-1]),
                    )
                    for _, result in valid_fft_results
                )
                try:
                    spectrum_xlim_min, spectrum_xlim_max = resolve_clipped_window(
                        supported_min,
                        supported_max,
                        welch_min_hz if use_welch else fft_min_hz,
                        welch_max_hz if use_welch else fft_max_hz,
                    )
                except ValueError:
                    _set_panel_message(
                        ax_fft=ax_fft,
                        only="fft",
                        title=f"{pair_title_prefix} {spectrum_panel_name} - invalid frequency range",
                    )
                    continue

                any_positive = False
                positive_vals_all: list[np.ndarray] = []
                for component, result in valid_fft_results:
                    spectrum_result = getattr(result, "welch_result", None) if use_welch else result.fft_result
                    freq = spectrum_result.freq
                    amp = spectrum_result.amplitude
                    mask = (freq >= spectrum_xlim_min) & (freq <= spectrum_xlim_max)
                    freq_plot = freq[mask]
                    amp_plot = amp[mask]
                    if freq_plot.size == 0:
                        continue
                    if welch_log if use_welch else fft_log:
                        ax_fft.semilogy(
                            freq_plot,
                            amp_plot,
                            linewidth=1.2,
                            color=COMPONENT_COLORS[component],
                            label=component,
                        )
                        positive_vals = amp_plot[amp_plot > 0]
                        if positive_vals.size > 0:
                            any_positive = True
                            positive_vals_all.append(positive_vals)
                    else:
                        ax_fft.plot(
                            freq_plot,
                            amp_plot,
                            linewidth=1.2,
                            color=COMPONENT_COLORS[component],
                            label=component,
                        )

                ax_fft.set_title(f"{pair_title_prefix} {'Welch' if use_welch else 'FFT'}")
                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_ylabel("Welch Amplitude" if use_welch else "Amplitude")
                ax_fft.set_xlim(spectrum_xlim_min, spectrum_xlim_max)
                apply_major_tick_spacing(ax_fft, tickspace_hz, axis="x")
                ax_fft.grid(True, alpha=0.3)
                ax_fft.legend()

                if (welch_log if use_welch else fft_log) and any_positive:
                    positive_concat = np.concatenate(positive_vals_all)
                    ymin = np.percentile(positive_concat, 0.1) * 0.7
                    ymax = np.max(positive_concat) * 1.3
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                        ax_fft.set_ylim(ymin, ymax)

        if only == "fft":
            continue

        for comp_idx, component in enumerate(components):
            result = row_component_results[component]
            ax_spec = axes_grid[row_idx, col_offset + comp_idx]
            panel_title = (
                f"{pair_title_prefix} {component.upper()} "
                f"{'Full Welch Image' if full_image and use_welch else 'Full FFT Image' if full_image else 'Sliding FFT'}"
            )

            spectrum_result = getattr(result, "welch_result", None) if use_welch else result.fft_result

            if result.error_message is not None or result.processed is None or (full_image and spectrum_result is None):
                _set_panel_message(
                    ax_spec=ax_spec,
                    only="sliding",
                    full_image=full_image,
                    title=f"{panel_title} - {result.error_message or 'invalid data'}",
                )
                continue

            try:
                y_min, y_max = resolve_clipped_window(
                    max(0.01, float(spectrum_result.freq[0])),
                    min(float(result.processed.nyquist), float(spectrum_result.freq[-1])),
                    sliding_min_hz,
                    sliding_max_hz,
                )
            except ValueError:
                _set_panel_message(
                    ax_spec=ax_spec,
                    only="sliding",
                    full_image=full_image,
                    title=f"{panel_title} - invalid frequency range",
                )
                continue

            if full_image:
                show_right_ylabel = comp_idx == len(components) - 1
                _plot_frequency_image(
                    fig,
                    ax_spec,
                    freq=spectrum_result.freq,
                    amp=spectrum_result.amplitude,
                    plot_scale=sliding_plot_scale,
                    cmap_index=cmap_index,
                    y_min=y_min,
                    y_max=y_max,
                    title=f"{pair_title_prefix} {component.upper()} {'Full Welch Image' if use_welch else 'Full FFT Image'}",
                    linear_color_label="Welch Amplitude" if use_welch else "Amplitude",
                    log_color_label="Welch Amplitude (dB)" if use_welch else "Amplitude (dB)",
                    show_colorbar=not use_compact_image_layout,
                    annotate_range=use_compact_image_layout,
                    y_tickspace_hz=tickspace_hz,
                )
                if use_compact_image_layout:
                    _apply_compact_image_axis_style(
                        ax_spec,
                        show_right_ylabel=show_right_ylabel,
                    )
                continue

            if result.spectrogram_result is None:
                _set_panel_message(
                    ax_spec=ax_spec,
                    only="sliding",
                    full_image=full_image,
                    title=f"{panel_title} - {result.spectrogram_error_message or 'window too short'}",
                )
                continue

            _plot_spectrogram_panel(
                fig,
                ax_spec,
                f_spec=result.spectrogram_result.f,
                t_spec=result.spectrogram_result.t,
                s_complex=result.spectrogram_result.S_complex,
                t_start=float(result.processed.t[0]),
                plot_scale=sliding_plot_scale,
                cmap_index=cmap_index,
                y_min=y_min,
                y_max=y_max,
                title=panel_title,
                time_interval=time_interval,
                show_colorbar=not use_compact_image_layout,
                annotate_range=use_compact_image_layout,
                y_tickspace_hz=tickspace_hz,
            )
            if use_compact_image_layout:
                _apply_compact_image_axis_style(
                    ax_spec,
                    show_right_ylabel=comp_idx == len(components) - 1,
                )

    if title:
        fig.suptitle(title, fontsize=14)

    if use_compact_image_layout and full_couple and only != "sliding":
        fft_axes = [axes_grid[row_idx, 0] for row_idx in range(len(ref_results))]
        image_col_offset = 1
        image_axes = [
            axes_grid[row_idx, image_col_offset + comp_idx]
            for row_idx in range(len(ref_results))
            for comp_idx in range(len(components))
        ]
        _link_fft_frequency_to_image_frequency(fft_axes, image_axes)

    return fig


def plot_component_pair_frequency_grid_single_row_groups(
    component_results: dict[str, list[PairFrequencyAnalysisResult | PairWelchFrequencyAnalysisResult]],
    *,
    fft_log: bool = False,
    welch_log: bool = False,
    sliding_plot_scale: str = "log",
    only: str | None = None,
    full_image: bool = False,
    full_couple: bool = False,
    use_welch: bool = False,
    fft_min_hz: float | None = None,
    fft_max_hz: float | None = None,
    welch_min_hz: float | None = None,
    welch_max_hz: float | None = None,
    sliding_min_hz: float | None = None,
    sliding_max_hz: float | None = None,
    time_interval: tuple[float, float] | None = None,
    cmap_index: int = 6,
    title: str | None = None,
    tickspace_hz: float | None = None,
):
    if len(component_results) == 0:
        raise ValueError("No component results to plot")

    components = [component for component in ("x", "y", "a") if component in component_results]
    if len(components) == 0:
        raise ValueError("No component results to plot")

    ref_results = component_results[components[0]]
    if len(ref_results) == 0:
        raise ValueError("No pair results to plot")

    ref_keys = [(result.pair_index, result.label) for result in ref_results]
    for component in components[1:]:
        keys = [(result.pair_index, result.label) for result in component_results[component]]
        if keys != ref_keys:
            raise ValueError("Component datasets do not share the same bond layout after filtering")

    group_width = len(components) if only == "sliding" else 1 if only == "fft" else 1 + len(components)
    ncols = group_width * len(ref_results)
    width_ratios = None
    if only is None:
        image_panel_width = 0.42 if full_image else 0.5
        per_group_width_ratios = [1.0 + len(components) * (1.0 - image_panel_width)] + [image_panel_width] * len(components)
        width_ratios = per_group_width_ratios * len(ref_results)

    use_compact_image_layout = only != "fft"
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=((4.7 if use_compact_image_layout else 5.2) * ncols, 3.5),
        constrained_layout=not use_compact_image_layout,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios is not None else None,
    )
    axes_row = np.atleast_1d(axes)

    def _axis_at(pair_idx: int, col_idx: int):
        return axes_row[pair_idx * group_width + col_idx]

    if use_compact_image_layout:
        fig.subplots_adjust(
            left=0.055,
            right=0.97,
            bottom=0.07,
            top=0.94 if title else 0.98,
            wspace=0.02,
            hspace=0.22,
        )

        image_col_offset = 0 if only == "sliding" else 1
        lead_ax = _axis_at(0, image_col_offset)
        for pair_idx in range(len(ref_results)):
            for comp_idx in range(len(components)):
                ax = _axis_at(pair_idx, image_col_offset + comp_idx)
                if ax is lead_ax:
                    continue
                ax.sharey(lead_ax)
                ax.sharex(lead_ax)
        if only != "sliding":
            lead_fft_ax = _axis_at(0, 0)
            for pair_idx in range(1, len(ref_results)):
                _axis_at(pair_idx, 0).sharex(lead_fft_ax)

    for pair_idx, ref_result in enumerate(ref_results):
        pair_title_prefix = f"Pair {ref_result.pair_index} ({str(ref_result.label).upper()})"
        row_component_results = {
            component: component_results[component][pair_idx]
            for component in components
        }

        col_offset = 0
        if only != "sliding":
            ax_fft = _axis_at(pair_idx, 0)
            col_offset = 1

            spectrum_panel_name = "Welch" if use_welch else "FFT"
            valid_fft_results = [
                (component, result)
                for component, result in row_component_results.items()
                if result.error_message is None
                and result.processed is not None
                and (
                    getattr(result, "welch_result", None) is not None
                    if use_welch
                    else getattr(result, "fft_result", None) is not None
                )
            ]

            if len(valid_fft_results) == 0:
                error_bits = [
                    f"{component}: {result.error_message or 'invalid data'}"
                    for component, result in row_component_results.items()
                ]
                _set_panel_message(
                    ax_fft=ax_fft,
                    only="fft",
                    title=f"{pair_title_prefix} {spectrum_panel_name} - {' | '.join(error_bits)}",
                )
            else:
                supported_min = max(
                    max(0.0, float(getattr(result, "welch_result" if use_welch else "fft_result").freq[0]))
                    for _, result in valid_fft_results
                )
                supported_max = min(
                    min(
                        float(result.processed.nyquist),
                        float(getattr(result, "welch_result" if use_welch else "fft_result").freq[-1]),
                    )
                    for _, result in valid_fft_results
                )
                try:
                    spectrum_xlim_min, spectrum_xlim_max = resolve_clipped_window(
                        supported_min,
                        supported_max,
                        welch_min_hz if use_welch else fft_min_hz,
                        welch_max_hz if use_welch else fft_max_hz,
                    )
                except ValueError:
                    _set_panel_message(
                        ax_fft=ax_fft,
                        only="fft",
                        title=f"{pair_title_prefix} {spectrum_panel_name} - invalid frequency range",
                    )
                    continue

                any_positive = False
                positive_vals_all: list[np.ndarray] = []
                for component, result in valid_fft_results:
                    spectrum_result = getattr(result, "welch_result", None) if use_welch else result.fft_result
                    freq = spectrum_result.freq
                    amp = spectrum_result.amplitude
                    mask = (freq >= spectrum_xlim_min) & (freq <= spectrum_xlim_max)
                    freq_plot = freq[mask]
                    amp_plot = amp[mask]
                    if freq_plot.size == 0:
                        continue
                    if welch_log if use_welch else fft_log:
                        ax_fft.semilogy(
                            freq_plot,
                            amp_plot,
                            linewidth=1.2,
                            color=COMPONENT_COLORS[component],
                            label=component,
                        )
                        positive_vals = amp_plot[amp_plot > 0]
                        if positive_vals.size > 0:
                            any_positive = True
                            positive_vals_all.append(positive_vals)
                    else:
                        ax_fft.plot(
                            freq_plot,
                            amp_plot,
                            linewidth=1.2,
                            color=COMPONENT_COLORS[component],
                            label=component,
                        )

                ax_fft.set_title(f"{pair_title_prefix} {'Welch' if use_welch else 'FFT'}")
                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_ylabel("Welch Amplitude" if use_welch else "Amplitude")
                ax_fft.set_xlim(spectrum_xlim_min, spectrum_xlim_max)
                apply_major_tick_spacing(ax_fft, tickspace_hz, axis="x")
                ax_fft.grid(True, alpha=0.3)
                ax_fft.legend()

                if (welch_log if use_welch else fft_log) and any_positive:
                    positive_concat = np.concatenate(positive_vals_all)
                    ymin = np.percentile(positive_concat, 0.1) * 0.7
                    ymax = np.max(positive_concat) * 1.3
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                        ax_fft.set_ylim(ymin, ymax)

        if only == "fft":
            continue

        for comp_idx, component in enumerate(components):
            result = row_component_results[component]
            ax_spec = _axis_at(pair_idx, col_offset + comp_idx)
            panel_title = (
                f"{pair_title_prefix} {component.upper()} "
                f"{'Full Welch Image' if full_image and use_welch else 'Full FFT Image' if full_image else 'Sliding FFT'}"
            )

            spectrum_result = getattr(result, "welch_result", None) if use_welch else result.fft_result

            if result.error_message is not None or result.processed is None or (full_image and spectrum_result is None):
                _set_panel_message(
                    ax_spec=ax_spec,
                    only="sliding",
                    full_image=full_image,
                    title=f"{panel_title} - {result.error_message or 'invalid data'}",
                )
                continue

            try:
                y_min, y_max = resolve_clipped_window(
                    max(0.01, float(spectrum_result.freq[0])),
                    min(float(result.processed.nyquist), float(spectrum_result.freq[-1])),
                    sliding_min_hz,
                    sliding_max_hz,
                )
            except ValueError:
                _set_panel_message(
                    ax_spec=ax_spec,
                    only="sliding",
                    full_image=full_image,
                    title=f"{panel_title} - invalid frequency range",
                )
                continue

            if full_image:
                show_right_ylabel = comp_idx == len(components) - 1
                _plot_frequency_image(
                    fig,
                    ax_spec,
                    freq=spectrum_result.freq,
                    amp=spectrum_result.amplitude,
                    plot_scale=sliding_plot_scale,
                    cmap_index=cmap_index,
                    y_min=y_min,
                    y_max=y_max,
                    title=f"{pair_title_prefix} {component.upper()} {'Full Welch Image' if use_welch else 'Full FFT Image'}",
                    linear_color_label="Welch Amplitude" if use_welch else "Amplitude",
                    log_color_label="Welch Amplitude (dB)" if use_welch else "Amplitude (dB)",
                    show_colorbar=not use_compact_image_layout,
                    annotate_range=use_compact_image_layout,
                    y_tickspace_hz=tickspace_hz,
                )
                if use_compact_image_layout:
                    _apply_compact_image_axis_style(
                        ax_spec,
                        show_right_ylabel=show_right_ylabel,
                    )
                continue

            if result.spectrogram_result is None:
                _set_panel_message(
                    ax_spec=ax_spec,
                    only="sliding",
                    full_image=full_image,
                    title=f"{panel_title} - {result.spectrogram_error_message or 'window too short'}",
                )
                continue

            _plot_spectrogram_panel(
                fig,
                ax_spec,
                f_spec=result.spectrogram_result.f,
                t_spec=result.spectrogram_result.t,
                s_complex=result.spectrogram_result.S_complex,
                t_start=float(result.processed.t[0]),
                plot_scale=sliding_plot_scale,
                cmap_index=cmap_index,
                y_min=y_min,
                y_max=y_max,
                title=panel_title,
                time_interval=time_interval,
                show_colorbar=not use_compact_image_layout,
                annotate_range=use_compact_image_layout,
                y_tickspace_hz=tickspace_hz,
            )
            if use_compact_image_layout:
                _apply_compact_image_axis_style(
                    ax_spec,
                    show_right_ylabel=comp_idx == len(components) - 1,
                )

    if title:
        fig.suptitle(title, fontsize=14)

    if use_compact_image_layout and full_couple and only != "sliding":
        fft_axes = [_axis_at(pair_idx, 0) for pair_idx in range(len(ref_results))]
        image_col_offset = 1
        image_axes = [
            _axis_at(pair_idx, image_col_offset + comp_idx)
            for pair_idx in range(len(ref_results))
            for comp_idx in range(len(components))
        ]
        _link_fft_frequency_to_image_frequency(fft_axes, image_axes)

    return fig


def plot_pair_welch_frequency_grid(
    results: list[PairWelchFrequencyAnalysisResult],
    *,
    welch_log: bool = False,
    sliding_plot_scale: str = "log",
    only: str | None = None,
    full_image: bool = False,
    full_couple: bool = False,
    welch_min_hz: float | None = None,
    welch_max_hz: float | None = None,
    sliding_min_hz: float | None = None,
    sliding_max_hz: float | None = None,
    time_interval: tuple[float, float] | None = None,
    cmap_index: int = 6,
    title: str | None = None,
    tickspace_hz: float | None = None,
):
    if len(results) == 0:
        raise ValueError("No pair results to plot")

    ncols = 1 if only in ("welch", "sliding") else 2
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=ncols,
        figsize=(14, 3.5 * len(results)),
        constrained_layout=True,
    )

    if ncols == 1:
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_2d(axes)

    for row_idx, result in enumerate(results):
        if only == "welch":
            ax_welch = axes[row_idx]
            ax_spec = None
        elif only == "sliding":
            ax_welch = None
            ax_spec = axes[row_idx]
        else:
            ax_welch = axes[row_idx][0]
            ax_spec = axes[row_idx][1]

        pair_title_prefix = f"Pair {result.pair_index} ({str(result.label).upper()})"

        if result.error_message is not None or result.processed is None or result.welch_result is None:
            _set_panel_message(
                ax_fft=ax_welch,
                ax_spec=ax_spec,
                only="fft" if only == "welch" else only,
                full_image=full_image,
                title=f"{pair_title_prefix} - {result.error_message or 'invalid data'}",
            )
            continue

        if only != "sliding":
            try:
                welch_xlim_min, welch_xlim_max = resolve_clipped_window(
                    max(0.0, float(result.welch_result.freq[0])),
                    min(float(result.processed.nyquist), float(result.welch_result.freq[-1])),
                    welch_min_hz,
                    welch_max_hz,
                )
            except ValueError:
                _set_panel_message(
                    ax_fft=ax_welch,
                    ax_spec=ax_spec,
                    only="fft" if only == "welch" else only,
                    full_image=full_image,
                    title=f"{pair_title_prefix} - invalid frequency range",
                )
                continue
            _plot_spectrum_curve_panel(
                ax_welch,
                freq=result.welch_result.freq,
                amp=result.welch_result.amplitude,
                log_scale=welch_log,
                x_min=welch_xlim_min,
                x_max=welch_xlim_max,
                title=(
                    f"{pair_title_prefix} Welch | {result.processed.proc_msg}"
                    f" | nperseg={result.welch_result.nperseg}, overlap={result.welch_result.noverlap}"
                ),
                y_label="Welch Amplitude",
                x_tickspace_hz=tickspace_hz,
            )

        if only == "welch":
            continue

        try:
            y_min, y_max = resolve_clipped_window(
                max(0.01, float(result.welch_result.freq[0])),
                min(float(result.processed.nyquist), float(result.welch_result.freq[-1])),
                sliding_min_hz,
                sliding_max_hz,
            )
        except ValueError:
            _set_panel_message(
                ax_fft=ax_welch,
                ax_spec=ax_spec,
                only="fft" if only == "welch" else only,
                full_image=full_image,
                title=f"{pair_title_prefix} - invalid frequency range",
            )
            continue

        if full_image:
            _plot_frequency_image(
                fig,
                ax_spec,
                freq=result.welch_result.freq,
                amp=result.welch_result.amplitude,
                plot_scale=sliding_plot_scale,
                cmap_index=cmap_index,
                y_min=y_min,
                y_max=y_max,
                title=f"{pair_title_prefix} Full Welch Image",
                linear_color_label="Welch Amplitude",
                log_color_label="Welch Amplitude (dB)",
                y_tickspace_hz=tickspace_hz,
            )
            continue

        if result.spectrogram_result is None:
            _set_panel_message(
                ax_fft=ax_welch,
                ax_spec=ax_spec,
                only="fft" if only == "welch" else only,
                full_image=full_image,
                title=f"{pair_title_prefix} - {result.spectrogram_error_message or 'window too short'}",
            )
            continue

        _plot_spectrogram_panel(
            fig,
            ax_spec,
            f_spec=result.spectrogram_result.f,
            t_spec=result.spectrogram_result.t,
            s_complex=result.spectrogram_result.S_complex,
            t_start=float(result.processed.t[0]),
            plot_scale=sliding_plot_scale,
            cmap_index=cmap_index,
            y_min=y_min,
            y_max=y_max,
            title=f"{pair_title_prefix} Sliding FFT",
            time_interval=time_interval,
            y_tickspace_hz=tickspace_hz,
        )

    if full_couple and full_image and only != "sliding":
        if ncols == 1:
            welch_axes = list(np.atleast_1d(axes))
            image_axes: list = []
        else:
            welch_axes = [axes[row_idx][0] for row_idx in range(len(results))]
            image_axes = [axes[row_idx][1] for row_idx in range(len(results))]
        _link_fft_frequency_to_image_frequency(welch_axes, image_axes)

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def plot_average_spectrum(
    result: AverageSpectrumResult,
    *,
    full_image: bool = False,
    plot_scale: str = "linear",
    cmap_index: int = 6,
    title: str | None = None,
    reference_amp_for_norm: np.ndarray | None = None,
    overlay: dict | None = None,
    tickspace_hz: float | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    if full_image:
        x_label = overlay.get("x_label", "Arbitrary X") if overlay is not None else "Arbitrary X"
        x_max = float(overlay.get("x_max", 1.0)) if overlay is not None else 1.0

        _plot_frequency_image(
            fig,
            ax,
            freq=result.freq_grid,
            amp=result.avg_amp,
            plot_scale=plot_scale,
            cmap_index=cmap_index,
            y_min=float(result.freq_grid[0]),
            y_max=float(result.freq_grid[-1]),
            x_label=x_label,
            x_max=x_max,
            title=title or "Average FFT",
            linear_color_label="Normalized Amplitude",
            log_color_label="Amplitude (dB)",
            reference_amp_for_norm=reference_amp_for_norm,
            overlay=overlay,
            y_tickspace_hz=tickspace_hz,
        )
    else:
        if plot_scale == "log":
            ax.semilogy(result.freq_grid, result.avg_amp, linewidth=1.5)
        else:
            ax.plot(result.freq_grid, result.avg_amp, linewidth=1.5)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlim(result.freq_grid[0], result.freq_grid[-1])
        apply_major_tick_spacing(ax, tickspace_hz, axis="x")
        ax.grid(True, alpha=0.3)
        ax.set_title(title or "Average FFT")

    fig.tight_layout()
    return fig


def plot_average_component_comparison(
    results_by_component: dict[str, AverageSpectrumResult],
    *,
    full_image: bool = False,
    plot_scale: str = "linear",
    cmap_index: int = 6,
    title: str | None = None,
    reference_amp_for_norm_by_component: dict[str, np.ndarray] | None = None,
    tickspace_hz: float | None = None,
):
    components = [component for component in ("x", "y", "a") if component in results_by_component]
    if len(components) == 0:
        raise ValueError("No component averages to plot")

    if not full_image:
        fig, ax = plt.subplots(figsize=(12, 5))
        for component in components:
            result = results_by_component[component]
            if plot_scale == "log":
                ax.semilogy(
                    result.freq_grid,
                    result.avg_amp,
                    linewidth=1.5,
                    color=COMPONENT_COLORS[component],
                    label=component,
                )
            else:
                ax.plot(
                    result.freq_grid,
                    result.avg_amp,
                    linewidth=1.5,
                    color=COMPONENT_COLORS[component],
                    label=component,
                )

        lead_result = results_by_component[components[0]]
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlim(lead_result.freq_grid[0], lead_result.freq_grid[-1])
        apply_major_tick_spacing(ax, tickspace_hz, axis="x")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title or "Average Component Comparison")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(
        1,
        len(components),
        figsize=(4.1 * len(components), 5.0),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        bottom=0.09,
        top=0.92 if title else 0.97,
        wspace=0.02,
    )

    lead_ax = axes[0]
    for ax in axes[1:]:
        ax.sharex(lead_ax)
        ax.sharey(lead_ax)

    for idx, component in enumerate(components):
        result = results_by_component[component]
        reference_amp = None
        if reference_amp_for_norm_by_component is not None:
            reference_amp = reference_amp_for_norm_by_component.get(component)

        _plot_frequency_image(
            fig,
            axes[idx],
            freq=result.freq_grid,
            amp=result.avg_amp,
            plot_scale=plot_scale,
            cmap_index=cmap_index,
            y_min=float(result.freq_grid[0]),
            y_max=float(result.freq_grid[-1]),
            title=component.upper(),
            linear_color_label="Normalized Amplitude",
            log_color_label="Amplitude (dB)",
            reference_amp_for_norm=reference_amp,
            show_colorbar=False,
            annotate_range=True,
            y_tickspace_hz=tickspace_hz,
        )
        _apply_compact_image_axis_style(
            axes[idx],
            show_right_ylabel=idx == len(components) - 1,
        )

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def plot_site_amplitude_previews(
    result: SiteAmplitudeAnalysisResult,
    *,
    title: str | None = None,
):
    if len(result.bonds) == 0:
        raise ValueError("No bond spectra available to preview")

    nrows = len(result.bonds)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(11, 2.8 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]

    for ax, bond in zip(axes, result.bonds):
        ax.plot(bond.roi_freq, bond.roi_normalized_amplitude, linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)

        for peak in bond.peak_integrals:
            ax.axvspan(peak.low_hz, peak.high_hz, alpha=0.12)
            ax.axvline(peak.peak_hz, linestyle="--", linewidth=0.9, alpha=0.4, color="black")

        ax.set_ylabel("Norm. Amp.")
        ax.set_title(
            f"Bond {bond.display_bond_index} | contributors={len(bond.contributors)} | "
            f"area={bond.normalization_integral:.6g}"
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(result.roi_low, result.roi_high)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    return fig
