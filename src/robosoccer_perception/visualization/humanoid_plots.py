"""Figures of the Phase 4 benchmark (Matplotlib, no GUI backend).

Colour follows the ARM, in every figure: the same arm has the same colour and marker whether
it is plotted against scenarios, calibration errors or timing offsets. The orientation oracle
is a reference, not a competitor, and is drawn in neutral grey. Every figure carries a legend,
and markers differ in shape so that colour is never the only cue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.visualization import style

#: arm -> (colour, marker, label). The colours are the first categorical slots of the
#: repository's palette (visualization/style.py), assigned in a fixed order.
ARM_STYLE = {
    "fixed": ("#2a78d6", "o", "A  fixed camera"),
    "imu": ("#eb6834", "s", "C  IMU"),
    "imu_uncertainty": ("#1baf7a", "D", "D  IMU + uncertainty"),
    "imu_selfcal": ("#eda100", "^", "D + mount self-calibration"),
    "imu_latest": ("#e87ba4", "v", "C, newest IMU sample (no interpolation)"),
}
ORACLE_STYLE = (style.REFERENCE, "x", "B  orientation oracle (uses ground truth)")
ORACLE = "oracle_orientation"


def _arm_style(name: str) -> tuple[str, str, str]:
    return ORACLE_STYLE if name == ORACLE else ARM_STYLE.get(name, (style.PARTICLE, "o", name))


def _save(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=style.SURFACE, bbox_inches="tight")
    return path


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(frameon=False, fontsize=8.5, labelcolor=style.TEXT_SECONDARY, **kwargs)
    for text in legend.get_texts():
        text.set_color(style.TEXT_SECONDARY)


# ------------------------------------------------------------------ all scenarios, all arms
def save_overview(report: dict[str, Any], path: Path) -> Path:
    """One row per scenario, one marker per arm: position RMSE and camera pitch RMSE."""
    scenarios = list(report["scenarios"])
    arms = list(next(iter(report["scenarios"].values()))["arms"])
    panels = (("position_rmse_all", "Position RMSE over the whole run [m]", True),
              ("camera_pitch_rmse_deg", "Camera pitch error, RMS [deg]", False),
              ("position_consistency", "Consistency: RMS error / RMS reported std", True))
    figure = style.new_figure(15.0, 0.42 * len(scenarios) + 2.2)
    axes = figure.subplots(1, len(panels), sharey=True)
    rows = np.arange(len(scenarios))[::-1]
    for ax, (key, title, log) in zip(axes, panels):
        style.style_axes(ax)
        for row in rows:
            ax.axhline(row, color=style.GRID, linewidth=0.6, zorder=0)
        for index, arm in enumerate(arms):
            colour, marker, label = _arm_style(arm)
            values = [report["scenarios"][s]["arms"][arm]["aggregate"].get(key)
                      for s in scenarios]
            values = np.array([np.nan if v is None else v for v in values], dtype=float)
            jitter = (index - (len(arms) - 1) / 2.0) * 0.11
            outline = {} if marker == "x" else {"edgecolors": style.SURFACE, "linewidths": 0.8}
            ax.scatter(values, rows + jitter, s=38, marker=marker, color=colour, label=label,
                       zorder=3, **outline)
        if log:
            ax.set_xscale("log")
        if key == "position_consistency":
            ax.axvline(1.0, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)))
            ax.text(1.0, len(scenarios) - 0.45, " consistent", color=style.TEXT_SECONDARY,
                    fontsize=8, va="bottom")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_ylim(-0.6, len(scenarios) - 0.2)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([f"{s}" for s in scenarios], fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="lower center", ncol=min(len(arms), 3),
                           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    for text in legend.get_texts():
        text.set_color(style.TEXT_SECONDARY)
    figure.suptitle("Phase 4 benchmark: every scenario, every arm (synthetic)", fontsize=12,
                    color=style.TEXT, x=0.01, ha="left")
    figure.subplots_adjust(wspace=0.08, bottom=0.14, top=0.92)
    return _save(figure, path)


# ------------------------------------------------------------------ one run in time
def save_run_figure(run, path: Path, title: str = "") -> Path:
    """Camera pitch and roll (true, assumed, estimated) and the position error of one run."""
    figure = style.new_figure(11.0, 8.2)
    axes = figure.subplots(3, 1, sharex=True)
    time = run.time
    for ax, key, truth, label in (
            (axes[0], "camera_pitch_error_deg", run.true_camera_pitch_deg, "Camera pitch [deg]"),
            (axes[1], "camera_roll_error_deg", run.true_camera_roll_deg, "Camera roll [deg]")):
        style.style_axes(ax)
        ax.plot(time, truth, color=style.TEXT, linewidth=2.0, label="true")
        for name in ("fixed", "imu"):
            if name not in run.arms:
                continue
            colour, _, text = _arm_style(name)
            ax.plot(time, truth + getattr(run.arms[name], key), color=colour, linewidth=1.5,
                    label=f"{text} (what the robot uses)")
        ax.set_ylabel(label, fontsize=9.5)
        _legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=3)    # above the data
    ax = axes[2]
    style.style_axes(ax)
    for name, arm in run.arms.items():
        colour, _, text = _arm_style(name)
        ax.plot(time, arm.result.position_error, color=colour, linewidth=1.6, label=text,
                linestyle=(0, (4, 2)) if name == ORACLE else "-")
    ax.set_ylabel("Position error [m]", fontsize=9.5)
    ax.set_xlabel("time [s]", fontsize=9.5)
    ax.set_ylim(bottom=0.0)
    _legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=3)
    figure.suptitle(title or f"{run.scenario_id}, seed {run.seed}", fontsize=11.5,
                    color=style.TEXT, x=0.01, ha="left")
    figure.subplots_adjust(hspace=0.30, top=0.90)
    return _save(figure, path)


def save_height_figure(report: dict[str, Any], path: Path) -> Path | None:
    """Position RMSE against the virtual camera height, for the scenarios that differ only in
    the mount."""
    points = []
    for scenario_id, entry in report["scenarios"].items():
        info = entry["scenario"]
        if info["motion"] == "walking" and info["robot"] == "humanoid_prototype_1" \
                and info["imu"] == "nominal" and not info["imu_overrides"] \
                and not any(info["mount_error"].values()):
            points.append((info["camera_height_standing_m"], info["mount"], scenario_id, entry))
    points.sort(key=lambda item: item[0])
    seen, unique = set(), []
    for point in points:
        if point[1] not in seen:
            seen.add(point[1])
            unique.append(point)
    if len(unique) < 2:
        return None
    figure = style.new_figure(10.5, 4.2)
    axes = figure.subplots(1, 2)
    heights = [p[0] for p in unique]
    for ax, key, label in ((axes[0], "position_rmse_all", "Position RMSE [m]"),
                           (axes[1], "yaw_rmse_all", "Yaw RMSE [deg]")):
        style.style_axes(ax)
        for arm in unique[0][3]["arms"]:
            colour, marker, text = _arm_style(arm)
            values = [p[3]["arms"][arm]["aggregate"][key] for p in unique]
            ax.plot(heights, values, color=colour, marker=marker, markersize=6.5, linewidth=2.0,
                    label=text, linestyle=(0, (4, 2)) if arm == ORACLE else "-")
        ax.set_xticks(heights)
        ax.set_xticklabels([f"{h:.2f} m\n{p[1]}" for h, p in zip(heights, unique)], fontsize=8.5)
        ax.set_ylabel(label, fontsize=9.5)
        ax.set_ylim(bottom=0.0)
    axes[0].set_xlabel("camera height above the ground (virtual mount preset)", fontsize=9.5)
    _legend(axes[1], loc="upper right")
    figure.suptitle("Virtual head height: walking a rectangle (synthetic; every mount is an "
                    "assumption)", fontsize=11.5, color=style.TEXT, x=0.01, ha="left")
    figure.subplots_adjust(wspace=0.22, top=0.88)
    return _save(figure, path)


def save_benchmark_figures(report: dict[str, Any], runs: dict[str, Any], directory: Path
                           ) -> list[Path]:
    directory = Path(directory)
    written = [save_overview(report, directory / "overview.png")]
    height = save_height_figure(report, directory / "camera_height.png")
    if height is not None:
        written.append(height)
    for scenario_id, run in runs.items():
        title = f"{scenario_id}: {report['scenarios'][scenario_id]['scenario']['title']} " \
                f"(seed {run.seed})"
        written.append(save_run_figure(run, directory / f"run_{scenario_id}.png", title))
    return written


# ------------------------------------------------------------------ sweeps
def save_sweep_figure(report: dict[str, Any], path: Path) -> Path:
    """Position RMSE against a deliberate calibration error, one panel per parameter."""
    parameters = report["parameters"]
    figure = style.new_figure(5.0 * len(parameters), 4.4)
    axes = np.atleast_1d(figure.subplots(1, len(parameters), sharey=True))
    oracle = report["oracle"]["position_rmse_all"]
    for ax, (parameter, entry) in zip(axes, parameters.items()):
        style.style_axes(ax)
        scale = 100.0 if entry["unit"] == "m" else 1.0
        unit = "cm" if entry["unit"] == "m" else entry["unit"]
        for base, points in entry["bases"].items():
            colour, marker, label = _arm_style(base)
            ax.plot([p["value"] * scale for p in points],
                    [p["aggregate"]["position_rmse_all"] for p in points], color=colour,
                    marker=marker, markersize=6.5, linewidth=2.0, label=label)
        ax.axhline(oracle, color=ORACLE_STYLE[0], linewidth=1.2, linestyle=(0, (4, 3)),
                   label=ORACLE_STYLE[2])
        ax.set_title(entry["title"], fontsize=10.5, loc="left")
        ax.set_xlabel(f"error: estimate minus truth [{unit}]", fontsize=9.5)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel("Position RMSE [m]", fontsize=9.5)
    _legend(axes[-1], loc="upper center")
    figure.suptitle(f"Sensitivity to a wrong camera calibration ({report['scenario']}, "
                    "synthetic)", fontsize=11.5, color=style.TEXT, x=0.01, ha="left")
    figure.subplots_adjust(wspace=0.06, top=0.86)
    return _save(figure, path)


def save_timing_figure(report: dict[str, Any], path: Path) -> Path:
    """Position RMSE and camera pitch error against a camera / IMU timestamp mismatch."""
    figure = style.new_figure(11.0, 4.2)
    axes = figure.subplots(1, 2)
    offsets = report["offsets_ms"]
    for ax, key, label in ((axes[0], "position_rmse_all", "Position RMSE [m]"),
                           (axes[1], "camera_pitch_rmse_deg", "Camera pitch error, RMS [deg]")):
        style.style_axes(ax)
        for arm in ("imu", "imu_uncertainty"):
            colour, marker, text = _arm_style(arm)
            ax.plot(offsets, [point[key] for point in report[arm]], color=colour, marker=marker,
                    markersize=6.5, linewidth=2.0, label=text)
        for name, source in (("fixed", report["fixed"]), ("imu_latest", report["imu_latest"]),
                             (ORACLE, report["oracle"])):
            colour, _, text = _arm_style(name)
            ax.axhline(source[key], color=colour, linewidth=1.3, linestyle=(0, (4, 3)),
                       label=text)
        ax.set_xlabel("timestamp mismatch between image and IMU [ms]", fontsize=9.5)
        ax.set_ylabel(label, fontsize=9.5)
        ax.set_ylim(bottom=0.0)
    _legend(axes[1], loc="upper center")
    figure.suptitle(f"Sensitivity to camera / IMU timing ({report['scenario']}, synthetic)",
                    fontsize=11.5, color=style.TEXT, x=0.01, ha="left")
    figure.subplots_adjust(wspace=0.22, top=0.88)
    return _save(figure, path)
