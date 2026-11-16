"""Top-down field plots: lines, goals, particles, poses and observations.

Everything is drawn from a `FieldGeometry`, in field coordinates (metres), so the plot shows
exactly the map the localizer uses. Needs matplotlib (the ``viz`` extra).
"""
from __future__ import annotations

import math

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.observations.types import ObservationFrame
from robosoccer_perception.visualization import style

LANDMARK_LABELS = {
    LandmarkType.L_INTERSECTION: "L", LandmarkType.T_INTERSECTION: "T",
    LandmarkType.X_INTERSECTION: "X", LandmarkType.GOAL_POST: "post",
    LandmarkType.PENALTY_MARK: "mark", LandmarkType.CENTER_MARK: "centre",
}


def draw_field(ax, field: FieldGeometry) -> None:
    """Draw the carpet, the painted lines at their true width, and the goals."""
    from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

    half_x, half_y = field.half_extent
    ax.add_patch(Rectangle((-half_x, -half_y), 2 * half_x, 2 * half_y,
                           facecolor=style.FIELD_TURF, edgecolor="none", zorder=0))
    for segment in field.segments:
        start, end = np.asarray(segment.start), np.asarray(segment.end)
        along = (end - start) / np.linalg.norm(end - start)
        across = 0.5 * segment.width * np.array([-along[1], along[0]])
        ax.add_patch(Polygon([start + across, end + across, end - across, start - across],
                             closed=True, facecolor=style.FIELD_LINE, edgecolor="none", zorder=1))
    for arc in field.arcs:
        ax.add_patch(Wedge(arc.center, arc.radius + arc.width / 2,
                           math.degrees(arc.start_angle), math.degrees(arc.end_angle),
                           width=arc.width, facecolor=style.FIELD_LINE, edgecolor="none",
                           zorder=1))
    for goal in field.goals:
        (x, y_pos), (_, y_neg) = goal.posts
        back = x + goal.side * goal.dims.depth
        ax.plot([x, back, back, x], [y_pos, y_pos, y_neg, y_neg], color=style.TEXT_SECONDARY,
                linewidth=1.0, zorder=1)
        for post in goal.posts:
            ax.add_patch(Circle(post, max(goal.dims.post_diameter / 2, 0.06),
                                facecolor=style.TEXT_SECONDARY, edgecolor="none", zorder=2))
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_aspect("equal")
    ax.set_xlabel("field x [m]")
    ax.set_ylabel("field y [m]")
    style.style_axes(ax, grid=False)


def draw_pose(ax, pose, color: str, label: str | None = None, size: float = 0.35,
              hollow: bool = False) -> None:
    """A pose as a dot with a heading stroke of length ``size`` metres.

    ``hollow`` draws a larger open ring instead, so two coinciding poses both stay visible.
    """
    x, y, yaw = (float(v) for v in pose)
    reach = size * (1.5 if hollow else 1.0)
    ax.plot([x, x + reach * math.cos(yaw)], [y, y + reach * math.sin(yaw)], color=color,
            linewidth=2.2, solid_capstyle="round", zorder=6)
    if hollow:
        ax.plot([x], [y], marker="o", markersize=14, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=2.4, linestyle="none", zorder=6,
                label=label)
    else:
        ax.plot([x], [y], marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor=style.SURFACE, markeredgewidth=1.2, linestyle="none", zorder=7,
                label=label)


def draw_particles(ax, particles: np.ndarray, weights: np.ndarray | None = None,
                   max_drawn: int = 1500, label: str | None = "particles") -> None:
    """Particles as small dots with short heading ticks. More weight = more opaque."""
    from matplotlib.collections import LineCollection

    count = len(particles)
    if count > max_drawn:
        keep = np.linspace(0, count - 1, max_drawn).astype(int)
        particles = particles[keep]
        weights = None if weights is None else weights[keep]
    tick = 0.12
    ends = particles[:, :2] + tick * np.column_stack([np.cos(particles[:, 2]),
                                                      np.sin(particles[:, 2])])
    ax.add_collection(LineCollection(np.stack([particles[:, :2], ends], axis=1),
                                     colors=style.PARTICLE, linewidths=0.5, alpha=0.35, zorder=3))
    from matplotlib.colors import to_rgba

    colors = np.tile(to_rgba(style.PARTICLE), (len(particles), 1))
    colors[:, 3] = 0.45
    if weights is not None and weights.max() > 0:
        colors[:, 3] = 0.15 + 0.6 * np.clip(weights / weights.max(), 0.0, 1.0)
    ax.scatter(particles[:, 0], particles[:, 1], s=7, color=colors, linewidths=0, zorder=4)
    if label:       # a fully opaque stand-in for the legend
        ax.plot([], [], marker="o", markersize=4, linestyle="none", color=style.PARTICLE,
                label=label)


def draw_observations(ax, frame: ObservationFrame, reference_pose, label_prefix: str = "observed"
                      ) -> None:
    """Draw one frame's observations, placed in the field using ``reference_pose``.

    Observations are relative to the robot, so they need a pose to be drawn on the field. With
    the ESTIMATED pose (the usual choice) they show what the robot believes it is seeing: they
    line up with the map when the estimate is right and visibly miss it when it is wrong.
    """
    pose = np.asarray(reference_pose, dtype=float)
    if len(frame.line_points):
        points = tf.se2_transform_points(pose, frame.line_points.points)
        ax.scatter(points[:, 0], points[:, 1], s=10, color=style.OBSERVATION, linewidths=0,
                   zorder=5, label=f"{label_prefix} line points")
    if frame.landmarks:
        positions = tf.se2_transform_points(
            pose, np.array([o.position for o in frame.landmarks]))
        ax.scatter(positions[:, 0], positions[:, 1], s=70, marker="D",
                   facecolor=style.OBSERVATION, edgecolor=style.SURFACE, linewidths=1.2,
                   zorder=5, label=f"{label_prefix} landmarks")
        for observation, (x, y) in zip(frame.landmarks, positions):
            ax.annotate(LANDMARK_LABELS[observation.type], (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color=style.TEXT, zorder=8)


def plot_localization(field: FieldGeometry, truth_pose=None, estimate_pose=None,
                      particles: np.ndarray | None = None, weights: np.ndarray | None = None,
                      frame: ObservationFrame | None = None, title: str = "", ax=None):
    """One field view with everything on it. Returns the figure (or None if ``ax`` was given)."""
    figure = None
    if ax is None:
        half_x, half_y = field.half_extent
        figure = style.new_figure(9.0, 9.0 * half_y / half_x + 1.2)
        ax = figure.subplots()
    draw_field(ax, field)
    if particles is not None:
        draw_particles(ax, particles, weights)
    if frame is not None and estimate_pose is not None:
        draw_observations(ax, frame, estimate_pose)
    if truth_pose is not None:
        draw_pose(ax, truth_pose, style.TRUTH, "ground truth", hollow=True)
    if estimate_pose is not None:
        draw_pose(ax, estimate_pose, style.ESTIMATE, "estimate")
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    if figure is not None:
        _legend_below(figure, ax)
    return figure


def plot_trajectory(field: FieldGeometry, truth: np.ndarray, estimate: np.ndarray,
                    title: str = ""):
    """True and estimated paths over a whole run."""
    half_x, half_y = field.half_extent
    figure = style.new_figure(9.0, 9.0 * half_y / half_x + 1.2)
    ax = figure.subplots()
    draw_field(ax, field)
    ax.plot(truth[:, 0], truth[:, 1], color=style.TRUTH, linewidth=2.0, zorder=5,
            label="ground truth path")
    ax.plot(estimate[:, 0], estimate[:, 1], color=style.ESTIMATE, linewidth=1.4, zorder=6,
            label="estimated path")
    draw_pose(ax, truth[0], style.TRUTH, None, hollow=True)
    ax.annotate("start", truth[0, :2], xytext=(10, -14), textcoords="offset points", fontsize=9,
                color=style.TEXT)
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    _legend_below(figure, ax)
    return figure


def _legend_below(figure, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = figure.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5),
                               frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.4)
        for text in legend.get_texts():
            text.set_color(style.TEXT)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
