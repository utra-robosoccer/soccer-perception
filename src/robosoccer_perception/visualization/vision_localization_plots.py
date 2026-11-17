"""Figures for Phase 3: what was extracted from a frame, and how the arms compare.

Colour follows the entity, as in `visualization/style.py`. For the benchmark arms:

    vision          orange   (the estimate under test)
    oracle          blue     (the idealised reference)
    perfect_vision  aqua     (labels through the same geometry; diagnostic)
    odometry        grey     (dead reckoning; a baseline, drawn recessive)

These are the first three slots of the project's validated palette plus a neutral. Every
series is also named in a legend and drawn with its own dash pattern, so colour is never the
only cue. Needs matplotlib; the image overlays need OpenCV.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.perception.types import FeatureSet
from robosoccer_perception.visualization import field_plot, style

ARM_STYLE = {
    "vision": (style.ESTIMATE, "-", "vision (YOLO + U-Net)"),
    "oracle": (style.TRUTH, (0, (5, 2)), "oracle observations"),
    "perfect_vision": (style.OBSERVATION, (0, (1, 1.5)), "perfect vision (labels)"),
    "odometry": (style.REFERENCE, (0, (6, 2, 1, 2)), "odometry only"),
}
JUNCTION_LABELS = {LandmarkType.L_INTERSECTION: "L", LandmarkType.T_INTERSECTION: "T",
                   LandmarkType.X_INTERSECTION: "X"}
# RGB colours for drawing on camera images (chosen to stand off turf green and line white).
_POINT, _JUNCTION, _POST, _POST_BEARING, _CIRCLE = (
    (255, 122, 26), (0, 229, 255), (255, 214, 10), (255, 90, 90), (190, 120, 255))


def arm_style(name: str) -> tuple[Any, Any, str]:
    return ARM_STYLE.get(name, (style.TEXT_SECONDARY, "-", name))


# ------------------------------------------------------------------ on the camera image
def feature_overlay(image_rgb: np.ndarray, features: FeatureSet, posts: bool = True,
                    lines: bool = True) -> np.ndarray:
    """The camera image with the extracted features drawn on it.

    Dots: line points given to the localizer. Cyan squares with a letter: classed junctions.
    Yellow marker: a goal-post foot with a trusted range; red marker: bearing only.
    """
    import cv2

    out = image_rgb.copy()
    scale = max(1, round(out.shape[1] / 640))
    if lines:
        for u, v in features.line_points.pixels:
            cv2.circle(out, (int(round(u)), int(round(v))), 3 * scale, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (int(round(u)), int(round(v))), 2 * scale, _POINT, -1, cv2.LINE_AA)
    if posts:
        for post in features.goal_posts:
            colour = _POST if post.range_valid else _POST_BEARING
            x0, y0, x1, y1 = (int(round(v)) for v in post.bbox_xyxy)
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), colour, scale, cv2.LINE_AA)
            centre = (int(round(post.pixel[0])), int(round(post.pixel[1])))
            cv2.drawMarker(out, centre, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 14 * scale, 3 * scale)
            cv2.drawMarker(out, centre, colour, cv2.MARKER_TILTED_CROSS, 12 * scale, scale)
            label = f"{np.linalg.norm(post.position):.1f} m" if post.range_valid else "bearing"
            cv2.putText(out, label, (centre[0] + 8 * scale, centre[1] - 6 * scale),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 0, 0), 3 * scale, cv2.LINE_AA)
            cv2.putText(out, label, (centre[0] + 8 * scale, centre[1] - 6 * scale),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, colour, scale, cv2.LINE_AA)
    return out


def draw_ground_features(image_rgb: np.ndarray, features: FeatureSet, calibration, pose
                         ) -> np.ndarray:
    """Junctions and the circle centre drawn where the robot's own calibration puts them."""
    import cv2

    from robosoccer_perception.geometry.ground_projection import ground_to_pixels

    out = image_rgb.copy()
    scale = max(1, round(out.shape[1] / 640))
    items = [(j.position, JUNCTION_LABELS[j.type], _JUNCTION) for j in features.typed_intersections]
    items += [(c.center, "circle", _CIRCLE) for c in features.valid_circles]
    for position, label, colour in items:
        pixel, seen = ground_to_pixels(np.asarray(position)[None], calibration, pose)
        if not seen[0]:
            continue
        centre = (int(round(pixel[0, 0])), int(round(pixel[0, 1])))
        cv2.rectangle(out, (centre[0] - 6 * scale, centre[1] - 6 * scale),
                      (centre[0] + 6 * scale, centre[1] + 6 * scale), (0, 0, 0), 3 * scale)
        cv2.rectangle(out, (centre[0] - 6 * scale, centre[1] - 6 * scale),
                      (centre[0] + 6 * scale, centre[1] + 6 * scale), colour, scale)
        cv2.putText(out, label, (centre[0] + 9 * scale, centre[1] + 5 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 3 * scale, cv2.LINE_AA)
        cv2.putText(out, label, (centre[0] + 9 * scale, centre[1] + 5 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, colour, scale, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------ on the ground
def plot_ground_features(features: FeatureSet, field: FieldGeometry | None = None,
                         truth_pose=None, title: str = ""):
    """Top view in the ROBOT frame: what was extracted, where, and how uncertain.

    With ``field`` and ``truth_pose`` the true line paint is drawn underneath (transformed
    with the TRUE pose: an evaluation aid, not something the robot has).
    """
    figure = style.new_figure(7.6, 6.4)
    ax = figure.subplots()
    style.style_axes(ax)
    if field is not None and truth_pose is not None:
        truth_pose = np.asarray(truth_pose, dtype=float)
        paint = tf.se2_inverse_transform_points(truth_pose, field.sample_line_points(0.03, True))
        ax.scatter(-paint[:, 1], paint[:, 0], s=2, color=style.FIELD_LINE, linewidths=0,
                   label="true paint (evaluation only)", zorder=1)
    points = features.line_points
    if len(points):
        ax.scatter(-points.points[:, 1], points.points[:, 0], s=14, color=style.OBSERVATION,
                   linewidths=0, zorder=3, label=f"line points ({len(points)}), bars: 1 sigma")
        # One-sigma bars: across the line where its direction is known (that is what the
        # reported value means), along the viewing direction elsewhere.
        radial = points.points / np.maximum(np.linalg.norm(points.points, axis=1, keepdims=True), 1e-9)
        known = np.isfinite(points.normals).all(axis=1)
        directions = np.where(known[:, None], points.normals, radial)
        for point, direction, sigma in zip(points.points, directions, points.std):
            a, b = point - sigma * direction, point + sigma * direction
            ax.plot([-a[1], -b[1]], [a[0], b[0]], color=style.OBSERVATION, linewidth=0.7,
                    alpha=0.5, zorder=2)
    for segment in features.segments:
        ax.plot([-segment.start[1], -segment.end[1]], [segment.start[0], segment.end[0]],
                color=style.TEXT_SECONDARY, linewidth=1.0, zorder=2)
    for junction in features.typed_intersections:
        ax.plot(-junction.position[1], junction.position[0], marker="s", markersize=9,
                markerfacecolor="none", markeredgecolor=style.TRUTH, markeredgewidth=1.8,
                linestyle="none", zorder=5)
        ax.annotate(JUNCTION_LABELS[junction.type], (-junction.position[1], junction.position[0]),
                    xytext=(7, 4), textcoords="offset points", fontsize=9, color=style.TEXT)
    for circle in features.valid_circles:
        angles = np.linspace(0.0, 2.0 * math.pi, 120)
        ax.plot(-(circle.center[1] + circle.radius * np.sin(angles)),
                circle.center[0] + circle.radius * np.cos(angles), color=style.TRUTH,
                linewidth=1.2, zorder=4)
        ax.plot(-circle.center[1], circle.center[0], marker="+", markersize=10,
                color=style.TRUTH, linestyle="none", zorder=5)
        ax.annotate("circle centre", (-circle.center[1], circle.center[0]), xytext=(7, 4),
                    textcoords="offset points", fontsize=9, color=style.TEXT)
    for post in features.goal_posts:
        x, y = -post.position[1], post.position[0]
        if post.range_valid:
            ax.plot(x, y, marker="D", markersize=9, markerfacecolor=style.ESTIMATE,
                    markeredgecolor=style.SURFACE, markeredgewidth=1.2, linestyle="none", zorder=6)
            ax.annotate("goal post", (x, y), xytext=(7, 4), textcoords="offset points",
                        fontsize=9, color=style.TEXT)
        else:
            bearing = math.atan2(post.position[1], post.position[0])
            reach = 9.0
            ax.plot([0.0, -reach * math.sin(bearing)], [0.0, reach * math.cos(bearing)],
                    color=style.ESTIMATE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
            ax.annotate("goal post, bearing only", (x, y), xytext=(7, 4),
                        textcoords="offset points", fontsize=9, color=style.TEXT)
    ax.plot(0.0, 0.0, marker="^", markersize=11, color=style.TEXT, linestyle="none", zorder=7,
            label="robot (looking up the page)")
    ax.set_aspect("equal")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xlabel("to the robot's right [m]")
    ax.set_ylabel("ahead of the robot [m]")
    if title:
        ax.set_title(title, fontsize=10.5, loc="left")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=style.SURFACE,
              edgecolor=style.GRID)
    figure.tight_layout()
    return figure


# ------------------------------------------------------------------ arm comparison
def plot_arm_errors(results: dict[str, Any], thresholds, title: str = ""):
    """Position and yaw error over time of every arm of one run: two panels, one time axis."""
    figure = style.new_figure(9.0, 6.0)
    axes = figure.subplots(2, 1, sharex=True)
    panels = (("position_error", "Position error [m]", thresholds.localized_position,
               f"localized < {thresholds.localized_position:g} m"),
              ("yaw_error_deg", "Yaw error [deg]", thresholds.localized_yaw_deg,
               f"localized < {thresholds.localized_yaw_deg:g} deg"))
    time = None
    for ax, (key, name, limit, limit_label) in zip(axes, panels):
        style.style_axes(ax)
        top = 0.0
        for arm, result in results.items():
            colour, dash, label = arm_style(arm)
            values = getattr(result, key)
            time = result.time
            ax.plot(time, values, color=colour, linestyle=dash, linewidth=2.0,
                    label=label, zorder=3 if arm == "vision" else 2)
            top = max(top, float(np.percentile(values, 99)))
        ax.axhline(limit, color=style.GRID, linewidth=1.0, zorder=1)
        ax.annotate(limit_label, (time[-1], limit), xytext=(-4, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=style.TEXT_SECONDARY)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylim(0.0, max(1.3 * top, 1.6 * limit))
        result = next(iter(results.values()))
        for step in result.kidnap_steps:
            ax.axvline(time[step], color=style.REFERENCE, linewidth=1.0)
    for step in next(iter(results.values())).kidnap_steps:
        axes[0].annotate("kidnapped", (time[step], axes[0].get_ylim()[1]), xytext=(4, -10),
                         textcoords="offset points", fontsize=8, color=style.TEXT_SECONDARY)
    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
                           fontsize=9, handlelength=3.2)
    for text in legend.get_texts():
        text.set_color(style.TEXT)
    if title:
        figure.suptitle(title, fontsize=11, color=style.TEXT, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.06, 1, 0.96 if title else 1))
    return figure


def plot_arm_summary(report: dict[str, Any], metric: str = "position_rmse_all",
                     label: str = "Position RMSE over the whole run [m]", limit: float | None = 0.20):
    """One row per scenario, one dot per arm: the benchmark at a glance (log axis)."""
    scenarios = list(report["scenarios"])
    arms = [arm for arm in ARM_STYLE if any(arm in report["scenarios"][s]["arms"] for s in scenarios)]
    figure = style.new_figure(9.2, 0.42 * len(scenarios) + 1.9)
    ax = figure.subplots()
    style.style_axes(ax)
    markers = {"vision": "o", "oracle": "s", "perfect_vision": "D", "odometry": "X"}
    for row, scenario in enumerate(scenarios):
        ax.axhline(row, color=style.GRID, linewidth=0.8, zorder=1)
        for arm in arms:
            entry = report["scenarios"][scenario]["arms"].get(arm)
            value = None if entry is None else entry["aggregate"].get(metric)
            if value is None:
                continue
            colour, _, _ = arm_style(arm)
            ax.plot(max(value, 1e-3), row, marker=markers[arm], markersize=9, color=colour,
                    markeredgecolor=style.SURFACE, markeredgewidth=1.4, linestyle="none",
                    zorder=4 if arm == "vision" else 3)
    for arm in arms:
        colour, _, name = arm_style(arm)
        ax.plot([], [], marker=markers[arm], markersize=8, color=colour, linestyle="none",
                label=name)
    if limit is not None:
        ax.axvline(limit, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"target {limit:g}", (limit, -0.75), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=style.TEXT_SECONDARY, va="center")
    ax.set_xscale("log")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8.5)
    ax.set_ylim(len(scenarios) - 0.4, -1.1)
    ax.grid(True, axis="x", color=style.GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_xlabel(label + "  (log scale)")
    legend = figure.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=len(arms),
                           frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(style.TEXT)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


# ------------------------------------------------------------------ calibration sensitivity
def plot_sensitivity(sweep: dict[str, Any], metrics=(("position_rmse_all", "Position RMSE [m]"),
                                                     ("yaw_rmse_all", "Yaw RMSE [deg]"))):
    """Localization error against each calibration error: one column per parameter, one row
    per metric. The zero-error point of each curve is the nominal vision arm."""
    parameters = list(sweep["parameters"])
    figure = style.new_figure(3.1 * len(parameters) + 0.6, 2.7 * len(metrics) + 0.9)
    axes = np.atleast_2d(figure.subplots(len(metrics), len(parameters), sharey="row",
                                         squeeze=False))
    for column, parameter in enumerate(parameters):
        entry = sweep["parameters"][parameter]
        values = np.array([point["value"] for point in entry["points"]], dtype=float)
        for row, (key, name) in enumerate(metrics):
            ax = axes[row, column]
            style.style_axes(ax)
            data = np.array([np.nan if point["aggregate"][key] is None else point["aggregate"][key]
                             for point in entry["points"]], dtype=float)
            ax.plot(values, data, color=style.ESTIMATE, linewidth=2.0, marker="o", markersize=7,
                    markeredgecolor=style.SURFACE, markeredgewidth=1.2,
                    label="configured belief about the calibration")
            wider = entry.get("wider_belief")
            if wider:
                ax.plot([point["value"] for point in wider["points"]],
                        [np.nan if point["aggregate"][key] is None else point["aggregate"][key]
                         for point in wider["points"]], color=style.TRUTH, linewidth=2.0,
                        linestyle=(0, (5, 2)), marker="s", markersize=6,
                        markeredgecolor=style.SURFACE, markeredgewidth=1.2, label=wider["label"])
            diverged = np.array([point["aggregate"]["divergence_rate"] > 0.0
                                 for point in entry["points"]])
            if diverged.any():
                ax.plot(values[diverged], data[diverged], linestyle="none", marker="o",
                        markersize=13, markerfacecolor="none", markeredgecolor=style.TEXT,
                        markeredgewidth=1.4, label="some runs ended not localized")
            ax.set_ylim(bottom=0.0)
            if row == 0:
                ax.set_title(entry["label"], fontsize=10, loc="left")
            if row == len(metrics) - 1:
                ax.set_xlabel(entry["axis"])
            if column == 0:
                ax.set_ylabel(name)
    unique: dict[str, Any] = {}
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            unique.setdefault(label, handle)
    handles = list(unique.values())
    if handles:
        legend = figure.legend(handles, list(unique), loc="lower center", frameon=False,
                               fontsize=9, ncol=len(handles))
        for text in legend.get_texts():
            text.set_color(style.TEXT)
    figure.suptitle(f"Sensitivity of vision localization to camera-calibration error "
                    f"({sweep['scenario']}, {len(sweep['seeds'])} seeds)", fontsize=11,
                    color=style.TEXT, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.05 if handles else 0, 1, 0.94))
    return figure


def plot_particles(field: FieldGeometry, record, title: str = ""):
    """The particle cloud of one kept frame on the field, with the observations drawn from
    the estimated pose."""
    return field_plot.plot_localization(field, record.truth_pose, record.estimate_pose,
                                        record.particles, record.weights, record.observations,
                                        title=title)
