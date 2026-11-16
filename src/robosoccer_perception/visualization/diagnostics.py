"""Run diagnostics: error and filter-health time series, and particle snapshots.

Each quantity gets its own panel on a shared time axis (no dual axes). Needs matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from robosoccer_perception.evaluation.metrics import MetricThresholds
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.visualization import field_plot, style


def plot_errors(result, thresholds: MetricThresholds | None = None, title: str = ""):
    """Position error, yaw error, effective sample size and confidence over time."""
    th = thresholds or MetricThresholds()
    figure = style.new_figure(9.0, 8.0)
    axes = figure.subplots(4, 1, sharex=True)
    time = result.time
    panels = (
        (result.position_error, "Position error [m]", style.ESTIMATE, th.localized_position,
         f"localized < {th.localized_position:g} m"),
        (result.yaw_error_deg, "Yaw error [deg]", style.ESTIMATE, th.localized_yaw_deg,
         f"localized < {th.localized_yaw_deg:g} deg"),
        (result.ess_fraction, "Effective sample size / N", style.PARTICLE, None, ""),
        (result.confidence, "Filter confidence (its own estimate, no ground truth)",
         style.PARTICLE, None, ""),
    )
    for ax, (values, name, color, limit, limit_label) in zip(axes, panels):
        style.style_axes(ax)
        ax.plot(time, values, color=color, linewidth=1.6)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylim(bottom=0.0)
        if limit is not None:
            ax.axhline(limit, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)))
            ax.annotate(limit_label, (time[-1], limit), xytext=(-4, 4),
                        textcoords="offset points", ha="right", fontsize=8,
                        color=style.TEXT_SECONDARY)
        else:
            ax.set_ylim(0.0, 1.05)
        for step in result.kidnap_steps:
            ax.axvline(time[step], color=style.REFERENCE, linewidth=1.0)
    for step in result.kidnap_steps:
        axes[0].annotate("kidnapped", (time[step], axes[0].get_ylim()[1]), xytext=(4, -10),
                         textcoords="offset points", fontsize=8, color=style.TEXT_SECONDARY)
    axes[-1].set_xlabel("time [s]")
    if title:
        figure.suptitle(title, fontsize=11, color=style.TEXT, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    return figure


def plot_snapshots(field: FieldGeometry, result, title: str = ""):
    """The particle cloud at the stored snapshot steps (start, mid-run, end), side by side."""
    steps = sorted(result.snapshots)
    half_x, half_y = field.half_extent
    width = 5.2 * len(steps)
    figure = style.new_figure(width, 5.2 * half_y / half_x + 1.5)
    axes = np.atleast_1d(figure.subplots(1, len(steps)))
    for ax, step in zip(axes, steps):
        snapshot = result.snapshots[step]
        field_plot.plot_localization(
            field, result.truth[step], result.estimate[step], snapshot.particles,
            snapshot.weights, snapshot.frame, ax=ax,
            title=f"t = {result.time[step]:.1f} s   error {result.position_error[step]:.2f} m, "
                  f"{result.yaw_error_deg[step]:.1f} deg")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    for ax in axes[1:]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    figure.legend(seen.values(), seen.keys(), loc="lower center", ncol=len(seen),
                  frameon=False, fontsize=9)
    if title:
        figure.suptitle(title, fontsize=11, color=style.TEXT, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95 if title else 1))
    return figure


def save_scenario_figures(scenario, result, directory,
                          thresholds: MetricThresholds | None = None) -> dict[str, Path]:
    """Write the particle, error and trajectory figures of one run as PNG files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    field = FieldGeometry.load(scenario.field)
    label = f"{scenario.id} (seed {result.seed}): {scenario.title}"
    figures = {
        "errors": plot_errors(result, thresholds, label),
        "trajectory": field_plot.plot_trajectory(field, result.truth, result.estimate, label),
    }
    if result.snapshots:
        figures["particles"] = plot_snapshots(field, result, label)
    paths = {}
    for name, figure in figures.items():
        paths[name] = directory / f"{scenario.id}_{name}.png"
        figure.savefig(paths[name])
    return paths
