"""Synthetic odometry: the true relative motion, corrupted the way real odometry is.

The simulator knows the true poses; the localizer must not. This module is the only bridge for
motion: it turns two consecutive true poses into a noisy, drifting `OdometryDelta`. A kidnapping
(the robot being picked up and moved) is invisible to odometry, so it reports no motion then.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.localization.motion_model import OdometryDelta


@dataclasses.dataclass(frozen=True)
class OdometryNoiseConfig:
    # Random error, proportional to how far the robot moved in the interval.
    translation_noise: float = 0.05             # m (1 sigma) per m walked
    rotation_noise: float = 0.05                # rad per rad turned
    translation_from_rotation: float = 0.005    # m per rad turned
    rotation_from_translation_deg: float = 1.5  # deg per m walked
    # Systematic error (drift), drawn once per run and then constant.
    scale_bias_std: float = 0.03                # relative error of every reported distance
    yaw_bias_deg_per_m_std: float = 1.0         # heading drift per metre walked
    yaw_drift_deg_per_s: float = 0.0            # constant heading drift (gyro-like)
    missed_update_prob: float = 0.0             # probability that an update is lost

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OdometryNoiseConfig":
        return cls(**(data or {}))

    def replace(self, **changes) -> "OdometryNoiseConfig":
        return dataclasses.replace(self, **changes)

    def scaled(self, factor: float) -> "OdometryNoiseConfig":
        """Every error source multiplied by ``factor`` (missed updates unchanged)."""
        return dataclasses.replace(
            self, **{f.name: getattr(self, f.name) * factor for f in dataclasses.fields(self)
                     if f.name != "missed_update_prob"})


class SyntheticOdometry:
    def __init__(self, cfg: OdometryNoiseConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.scale_bias = float(rng.normal(0.0, cfg.scale_bias_std)) if cfg.scale_bias_std else 0.0
        self.yaw_bias_per_m = math.radians(
            float(rng.normal(0.0, cfg.yaw_bias_deg_per_m_std)) if cfg.yaw_bias_deg_per_m_std else 0.0)

    def measure(self, previous_true_pose, current_true_pose, timestamp: float, dt: float,
                kidnapped: bool = False) -> OdometryDelta | None:
        """Odometry for the interval between two true poses, or None if the update is lost."""
        cfg, rng = self.cfg, self.rng
        true_step = np.zeros(3) if kidnapped else \
            tf.se2_relative(previous_true_pose, current_true_pose)
        distance, turn = math.hypot(true_step[0], true_step[1]), abs(true_step[2])
        translation_std = cfg.translation_noise * distance + cfg.translation_from_rotation * turn
        rotation_std = (cfg.rotation_noise * turn
                        + math.radians(cfg.rotation_from_translation_deg) * distance)
        noise = rng.normal(size=3) * [translation_std, translation_std, rotation_std]
        lost = rng.random() < cfg.missed_update_prob
        if lost:
            return None
        dx = true_step[0] * (1.0 + self.scale_bias) + noise[0]
        dy = true_step[1] * (1.0 + self.scale_bias) + noise[1]
        dyaw = (true_step[2] + self.yaw_bias_per_m * distance
                + math.radians(cfg.yaw_drift_deg_per_s) * dt + noise[2])
        covariance = np.diag([translation_std ** 2, translation_std ** 2, rotation_std ** 2])
        return OdometryDelta(float(dx), float(dy), float(tf.wrap_angle(dyaw)), float(timestamp),
                             float(dt), covariance)
