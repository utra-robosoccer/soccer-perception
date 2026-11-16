"""Projected landmarks drawn over a rendered camera image.

A visual check of the whole frame chain: if ``T_field_camera`` and the intrinsics are right,
every marker sits exactly on the feature it names. Needs matplotlib.
"""
from __future__ import annotations

import numpy as np

from robosoccer_perception.geometry.camera import RigState, StereoCameraModel
from robosoccer_perception.geometry.landmarks import LandmarkMap
from robosoccer_perception.visualization import style
from robosoccer_perception.visualization.field_plot import LANDMARK_LABELS


def plot_camera_overlay(image: np.ndarray, camera: StereoCameraModel, landmarks: LandmarkMap,
                        base_pose, rig: RigState | None = None, side: str = "left",
                        max_range: float = 8.0, title: str = ""):
    """The image with a labelled marker at the projection of every landmark in view."""
    height, width = image.shape[:2]
    figure = style.new_figure(10.0, 10.0 * height / width + 0.5)
    ax = figure.subplots()
    ax.imshow(image)
    positions = np.array([[*lm.position, 0.0] for lm in landmarks.landmarks])
    pixels, depth, visible = camera.project_field_points(base_pose, positions, side, rig)
    visible &= depth < max_range
    ax.scatter(pixels[visible, 0], pixels[visible, 1], s=80, marker="o", facecolor="none",
               edgecolor=style.ESTIMATE, linewidths=1.8, label="projected map landmarks")
    for landmark, (u, v), shown in zip(landmarks.landmarks, pixels, visible):
        if shown:
            ax.annotate(LANDMARK_LABELS[landmark.type], (u, v), xytext=(7, -7),
                        textcoords="offset points", fontsize=9, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55,
                                  edgecolor="none"))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, loc="left", color=style.TEXT)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=style.SURFACE)
    figure.tight_layout()
    return figure
