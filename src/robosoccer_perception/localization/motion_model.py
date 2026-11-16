"""Odometry input and the particle motion (prediction) model. No ROS, no simulator.

The localizer is told how the robot believes it moved since the last update (`OdometryDelta`),
never where it is. `predict` moves every particle by that motion plus sampled noise, so the
particle cloud spreads by exactly as much as the odometry cannot be trusted.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf


@dataclasses.dataclass(frozen=True)
class OdometryDelta:
    """Relative motion of the base over one interval, as reported by odometry.

    ``(dx, dy, dyaw)`` is the pose of the base at the end of the interval expressed in the base
    frame at its start: dx forward, dy to the left, dyaw counter-clockwise. In ROS terms it is
    the change of the ``odom -> base_footprint`` transform between two stamps.
    """
    dx: float
    dy: float
    dyaw: float
    timestamp: float = 0.0                      # end of the interval, seconds
    dt: float = 0.0                             # length of the interval, seconds
    covariance: np.ndarray | None = None        # 3x3 claimed by the source, order (x, y, yaw)

    def as_array(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dyaw])


@dataclasses.dataclass(frozen=True)
class MotionModelConfig:
    """How much the localizer distrusts odometry. Standard deviations are per update."""
    translation_noise: float = 0.15             # m of error per m walked
    rotation_noise: float = 0.15                # rad of error per rad turned
    translation_from_rotation: float = 0.02     # m of error per rad turned
    rotation_from_translation_deg: float = 4.0  # deg of heading error per m walked
    min_translation_std: float = 0.004          # m; keeps the cloud alive when standing still
    min_rotation_std_deg: float = 0.2
    use_reported_covariance: bool = False       # also add the covariance the odometry claims
    # Used when an odometry message is missing: the robot may have moved, direction unknown.
    missing_translation_std: float = 0.04
    missing_rotation_std_deg: float = 4.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MotionModelConfig":
        return cls(**(data or {}))


def motion_noise_std(delta: OdometryDelta | None, cfg: MotionModelConfig) -> tuple[float, float]:
    """(translation std in m, rotation std in rad) to add to the particles for one update."""
    if delta is None:
        return cfg.missing_translation_std, math.radians(cfg.missing_rotation_std_deg)
    distance, turn = math.hypot(delta.dx, delta.dy), abs(delta.dyaw)
    translation = math.hypot(cfg.translation_noise * distance + cfg.translation_from_rotation * turn,
                             cfg.min_translation_std)
    rotation = math.hypot(cfg.rotation_noise * turn
                          + math.radians(cfg.rotation_from_translation_deg) * distance,
                          math.radians(cfg.min_rotation_std_deg))
    if cfg.use_reported_covariance and delta.covariance is not None:
        translation = math.hypot(translation, math.sqrt(max(delta.covariance[0, 0],
                                                            delta.covariance[1, 1])))
        rotation = math.hypot(rotation, math.sqrt(delta.covariance[2, 2]))
    return translation, rotation


def predict(particles: np.ndarray, delta: OdometryDelta | None, cfg: MotionModelConfig,
            rng: np.random.Generator) -> np.ndarray:
    """Move particles (N, 3) by the odometry step, each with its own noise sample.

    With ``delta=None`` (a missed odometry message) the particles only diffuse. Returns a new
    array; yaw stays wrapped to [-pi, pi).
    """
    translation_std, rotation_std = motion_noise_std(delta, cfg)
    step = np.zeros(3) if delta is None else delta.as_array()
    noise = rng.normal(size=particles.shape) * [translation_std, translation_std, rotation_std]
    return tf.se2_compose(particles, step + noise)
