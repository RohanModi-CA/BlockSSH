from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tools.models import LocalizationPeakDiagnostic, LocalizationProfile


def overlay_indexed_points(
    ax,
    x_values,
    y_values,
    *,
    mode: str = "scatter",
    color: str = "red",
    marker_size: float = 30.0,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if mode == "scatter":
        ax.scatter(
            x_values,
            y_values,
            s=marker_size,
            c=color,
            marker="o",
            zorder=10,
        )
    elif mode == "line":
        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            color=color,
            linestyle="-",
            zorder=10,
        )
    else:
        raise ValueError("mode must be 'scatter' or 'line'")


def plot_localization_profiles(
    profiles: list[LocalizationProfile],
    *,
    xlabel: str,
    ylabel: str = "Norm. Amplitude",
    title: str | None = None,
    line_color: str | None = None,
    diagnostics_by_entity: dict[str, list[LocalizationPeakDiagnostic]] | None = None,
    one_fig: bool = False,
    show_zero: bool = True,
):
    if len(profiles) == 0:
        raise ValueError("No localization profiles to plot")
    if diagnostics_by_entity is not None:
        for label, diagnostics in diagnostics_by_entity.items():
            if len(diagnostics) != len(profiles):
                raise ValueError(
                    f"diagnostics for selector '{label}' must match the number of profiles"
                )

    nrows = len(profiles)
    has_diagnostics = diagnostics_by_entity is not None and len(diagnostics_by_entity) > 0

    if one_fig and has_diagnostics:
        fig = plt.figure(figsize=(12, 3 * nrows), constrained_layout=True)
        gs = fig.add_gridspec(nrows, 2, width_ratios=[3.6, 1.4])
        left_ax = fig.add_subplot(gs[:, 0])
        axes = [[left_ax, fig.add_subplot(gs[row_idx, 1])] for row_idx in range(nrows)]
    elif one_fig:
        fig = plt.figure(figsize=(12, 4.5), constrained_layout=True)
        left_ax = fig.add_subplot(111)
        axes = [[left_ax]]
    else:
        ncols = 2 if has_diagnostics else 1
        width_ratios = [3.6, 1.4] if has_diagnostics else None
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(12, 3 * nrows),
            sharex=(not has_diagnostics),
            constrained_layout=True,
            gridspec_kw={"width_ratios": width_ratios} if width_ratios is not None else None,
        )
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [list(np.atleast_1d(axes))]
        elif ncols == 1:
            axes = [[ax] for ax in np.atleast_1d(axes)]

    all_ids = sorted(
        {
            int(entity_id)
            for profile in profiles
            for entity_id in profile.entity_ids.tolist()
        }
    )
    xticks = np.arange(min(all_ids), max(all_ids) + 1, 1) if all_ids else np.array([])
    diagnostics_order = []
    if has_diagnostics:
        diagnostics_order = sorted(
            diagnostics_by_entity.keys(),
            key=lambda label: (-1 if label == "All" else int(label)),
        )
    diag_axes = []
    diag_lines = []
    diag_spans = []
    diag_target_lines = []
    diag_selected_lines = []
    diag_texts = []

    line_kwargs = dict(
        marker="o",
        linestyle="-",
        linewidth=2.25,
        markersize=5,
        color=line_color if line_color is not None else "black",
    )

    if one_fig:
        ax = axes[0][0]
        spacing = 1.4
        for row_idx, profile in enumerate(profiles):
            x_vals = profile.entity_ids
            y_vals = profile.mean_amplitudes

            offset = spacing * float(nrows - 1 - row_idx)
            if x_vals.size == 0:
                ax.text(
                    0.02,
                    offset,
                    f"Peak {profile.peak_index}: {profile.frequency} Hz | No Data",
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                )
                continue

            finite = y_vals[np.isfinite(y_vals)]
            if finite.size > 0:
                ymin = float(np.min(finite))
                ymax = float(np.max(finite))
                span = ymax - ymin
                scaled = (y_vals - ymin) / span if span > 1e-12 else np.zeros_like(y_vals)
            else:
                scaled = np.zeros_like(y_vals)

            display_y = scaled + offset
            ax.plot(x_vals, display_y, **line_kwargs)
            if show_zero:
                ax.axhline(offset, color="0.55", linewidth=1.0, alpha=0.9, linestyle=":")
            ax.text(
                0.02,
                offset + 0.5,
                f"Peak {profile.peak_index}: {profile.frequency} Hz",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.85"},
            )

        ax.set_ylabel("Arb. Offset")
        ax.grid(True, alpha=0.3)
        if xticks.size > 0:
            ax.set_xticks(xticks)
            ax.set_xlim(xticks[0], xticks[-1])
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
    else:
        for row_idx, profile in enumerate(profiles):
            ax = axes[row_idx][0]
            x_vals = profile.entity_ids
            y_vals = profile.mean_amplitudes
            y_errs = profile.std_amplitudes

            if x_vals.size == 0:
                ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"Peak {profile.peak_index}: {profile.frequency} Hz")
                if xticks.size > 0:
                    ax.set_xticks(xticks)
                    ax.tick_params(axis="x", labelbottom=True)
            else:
                ax.plot(x_vals, y_vals, **line_kwargs)
                if show_zero:
                    ax.axhline(0.0, color="0.55", linewidth=1.0, alpha=0.9, linestyle=":")

                if np.any(y_errs > 0):
                    ax.fill_between(
                        x_vals,
                        y_vals - y_errs,
                        y_vals + y_errs,
                        alpha=0.2,
                        color=line_kwargs["color"],
                    )

                ax.set_title(f"Peak {profile.peak_index}: {profile.frequency} Hz")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

                if xticks.size > 0:
                    ax.set_xticks(xticks)
                    ax.set_xlim(xticks[0], xticks[-1])
                    ax.tick_params(axis="x", labelbottom=True)

    if has_diagnostics:
        for row_idx in range(nrows):
            diag_ax = axes[row_idx][1]
            diag = diagnostics_by_entity[diagnostics_order[0]][row_idx]
            (line,) = diag_ax.plot(diag.display_freq, diag.display_amplitude, color="tab:blue", linewidth=1.3)
            span = diag_ax.axvspan(diag.window_low, diag.window_high, color="tab:orange", alpha=0.15)
            target_line = diag_ax.axvline(diag.target_frequency, color="tab:orange", linestyle="--", linewidth=1.0)
            selected_x = diag.selected_frequency if diag.found and np.isfinite(diag.selected_frequency) else diag.target_frequency
            selected_line = diag_ax.axvline(selected_x, color="tab:red", linewidth=1.1)
            selected_line.set_visible(bool(diag.found and np.isfinite(diag.selected_frequency)))
            diag_ax.set_xlim(diag.display_freq[0], diag.display_freq[-1])
            diag_ax.grid(True, alpha=0.25)
            diag_ax.set_title("Peak Window", fontsize=10)
            if row_idx == nrows - 1:
                diag_ax.set_xlabel("Hz")
            if row_idx == 0:
                diag_ax.set_ylabel("Avg Amp")
            label = "sel" if diag.found else "not found"
            text = diag_ax.text(
                0.02,
                0.96,
                f"t={diag.target_frequency:.3f}\n{label}"
                + (
                    f"\ns={diag.selected_frequency:.3f}"
                    if diag.found and np.isfinite(diag.selected_frequency)
                    else ""
                ),
                transform=diag_ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )
            diag_axes.append(diag_ax)
            diag_lines.append(line)
            diag_spans.append(span)
            diag_target_lines.append(target_line)
            diag_selected_lines.append(selected_line)
            diag_texts.append(text)

    if not one_fig:
        for row in axes:
            row[0].set_xlabel(xlabel)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    if has_diagnostics and len(diagnostics_order) > 1:
        fig.subplots_adjust(bottom=0.08)
        label_text = fig.text(
            0.5,
            0.02,
            f"Diagnostic Bond: {diagnostics_order[0]} | click left plot to select",
            ha="center",
            va="center",
        )

        def _update_diagnostics(current_label: str) -> None:
            label_text.set_text(f"Diagnostic Bond: {current_label}")
            current_diags = diagnostics_by_entity[current_label]
            for diag_ax, line, span, target_line, selected_line, text, diag in zip(
                diag_axes,
                diag_lines,
                diag_spans,
                diag_target_lines,
                diag_selected_lines,
                diag_texts,
                current_diags,
            ):
                line.set_data(diag.display_freq, diag.display_amplitude)
                span.set_xy(
                    np.array(
                        [
                            [diag.window_low, 0.0],
                            [diag.window_low, 1.0],
                            [diag.window_high, 1.0],
                            [diag.window_high, 0.0],
                            [diag.window_low, 0.0],
                        ]
                    )
                )
                span.set_transform(diag_ax.get_xaxis_transform())
                target_line.set_xdata([diag.target_frequency, diag.target_frequency])
                if diag.found and np.isfinite(diag.selected_frequency):
                    selected_line.set_xdata([diag.selected_frequency, diag.selected_frequency])
                    selected_line.set_visible(True)
                else:
                    selected_line.set_xdata([diag.target_frequency, diag.target_frequency])
                    selected_line.set_visible(False)
                diag_ax.set_xlim(diag.display_freq[0], diag.display_freq[-1])
                finite_amp = diag.display_amplitude[np.isfinite(diag.display_amplitude)]
                if finite_amp.size > 0:
                    ymax = float(np.max(finite_amp))
                    diag_ax.set_ylim(0.0, ymax * 1.1 if ymax > 0 else 1.0)
                text.set_text(
                    f"t={diag.target_frequency:.3f}\n"
                    + ("sel" if diag.found else "not found")
                    + (
                        f"\ns={diag.selected_frequency:.3f}"
                        if diag.found and np.isfinite(diag.selected_frequency)
                        else ""
                    )
                )
            fig.canvas.draw_idle()

        left_axes = [row[0] for row in axes]

        def _handle_click(event) -> None:
            if event.inaxes not in left_axes or event.xdata is None:
                return

            bond_id = int(np.round(float(event.xdata)))
            bond_label = str(bond_id)
            if bond_label not in diagnostics_by_entity:
                return
            _update_diagnostics(bond_label)

        fig.canvas.mpl_connect("button_press_event", _handle_click)
        fig._localization_diag_label = label_text

    return fig
