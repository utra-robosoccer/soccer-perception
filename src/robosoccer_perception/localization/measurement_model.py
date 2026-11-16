"""Measurement (sensor) model: how well does a pose hypothesis explain what was observed?

For every particle the model returns a log-likelihood built from two kinds of evidence:

* **Field-line points.** Each observed point is moved into the field frame with the particle's
  pose and looked up in the line distance map. A point that lands on a line scores 1, one that
  lands far from every line scores close to a small floor.
* **Landmarks.** Each observed landmark is compared, along and across the viewing direction,
  with every map landmark of the same class as seen from the particle. Identities are unknown, so either the
  best-matching candidate counts (``nearest``) or all of them together (``mixture``).

Both use ``floor + (1 - floor) * exp(-0.5 * error^2)``. The floor is what makes the filter
robust: a false detection or an outlier can lower a particle's weight only by a bounded factor,
instead of vetoing an otherwise good hypothesis.

Nothing here knows the true pose.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry.distance_map import get_distance_map
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import LandmarkMap, build_landmark_map
from robosoccer_perception.observations.types import ObservationFrame


@dataclasses.dataclass(frozen=True)
class MeasurementModelConfig:
    distance_map_resolution: float = 0.025
    use_line_points: bool = True
    max_line_points: int = 40
    line_std_floor: float = 0.05
    line_outlier_floor: float = 0.10
    line_point_weight: float = 0.3
    use_landmarks: bool = True
    correspondence: str = "nearest"
    landmark_std_scale: float = 1.5
    landmark_range_std_floor: float = 0.05
    landmark_bearing_std_floor_deg: float = 1.5
    landmark_outlier_floor: float = 0.05
    landmark_weight: float = 1.0
    off_field_log_penalty: float = -20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MeasurementModelConfig":
        cfg = cls(**(data or {}))
        if cfg.correspondence not in ("nearest", "mixture"):
            raise ValueError("correspondence must be 'nearest' or 'mixture'")
        return cfg


@dataclasses.dataclass(frozen=True)
class Likelihood:
    log_likelihood: np.ndarray      # (N,) one value per particle
    evidence_weight: float          # total weight of the terms that went in (0 = no evidence)


class MeasurementModel:
    def __init__(self, field: FieldGeometry, landmarks: LandmarkMap | None = None,
                 cfg: MeasurementModelConfig | None = None):
        self.cfg = cfg or MeasurementModelConfig()
        self.field = field
        self.landmarks = landmarks or build_landmark_map(field)
        self.distance_map = get_distance_map(field, self.cfg.distance_map_resolution)

    def log_likelihood(self, particles: np.ndarray, frame: ObservationFrame,
                       rng: np.random.Generator) -> Likelihood:
        """Log-likelihood of one observation frame for every particle (N, 3)."""
        cfg = self.cfg
        total = np.zeros(len(particles))
        evidence = 0.0
        cos, sin = np.cos(particles[:, 2:3]), np.sin(particles[:, 2:3])       # (N, 1)

        points = frame.line_points
        if cfg.use_line_points and len(points) > 0:
            xy, std = points.points, points.std
            if len(xy) > cfg.max_line_points:
                keep = rng.choice(len(xy), size=cfg.max_line_points, replace=False)
                xy, std = xy[keep], std[keep]
            total += cfg.line_point_weight * self._line_log_likelihood(particles, cos, sin, xy, std)
            evidence += cfg.line_point_weight * len(xy)

        if cfg.use_landmarks:
            predicted: dict = {}        # class -> where its landmarks would appear, per particle
            for observation in frame.landmarks:
                kind = observation.type
                if kind not in predicted:
                    predicted[kind] = self._predict_landmarks(particles, cos, sin, kind)
                if predicted[kind] is None:
                    continue
                total += cfg.landmark_weight * self._landmark_log_likelihood(
                    observation, *predicted[kind])
                evidence += cfg.landmark_weight

        if evidence > 0.0:
            total = np.where(self.field.contains(particles[:, :2]), total,
                             total + cfg.off_field_log_penalty)
        return Likelihood(total, evidence)

    def _line_log_likelihood(self, particles, cos, sin, points_base, std) -> np.ndarray:
        """Sum over the points of the log-likelihood that each lies on a field line."""
        cfg = self.cfg
        bx, by = points_base[:, 0], points_base[:, 1]                         # (M,)
        x = particles[:, 0:1] + cos * bx - sin * by                           # (N, M) field frame
        y = particles[:, 1:2] + sin * bx + cos * by
        distance = self.distance_map.lookup_xy(x, y)
        sigma = np.hypot(std, cfg.line_std_floor)                             # (M,)
        fit = np.exp(-0.5 * (distance / sigma) ** 2)
        return np.log(cfg.line_outlier_floor + (1.0 - cfg.line_outlier_floor) * fit).sum(axis=1)

    def _predict_landmarks(self, particles, cos, sin, kind):
        """Base-frame coordinates (N, K) of every map landmark of a class, for each particle."""
        candidates = self.landmarks.positions(kind)
        if len(candidates) == 0:
            return None
        dx = candidates[:, 0] - particles[:, 0:1]                             # (N, K)
        dy = candidates[:, 1] - particles[:, 1:2]
        return cos * dx + sin * dy, -sin * dx + cos * dy

    def _landmark_log_likelihood(self, observation, predicted_x, predicted_y) -> np.ndarray:
        """Log-likelihood of one landmark observation against all candidates of its class.

        The error between the observed and each predicted position is split into a component
        along the viewing direction (scaled by the range std) and one across it (scaled by
        range * bearing std). For small errors this equals comparing range and bearing, and it
        needs no trigonometry per particle.
        """
        cfg = self.cfg
        distance = max(observation.range, 1e-6)
        ux, uy = observation.position[0] / distance, observation.position[1] / distance
        sigma_along = cfg.landmark_std_scale * math.hypot(
            observation.range_std, cfg.landmark_range_std_floor)
        sigma_across = cfg.landmark_std_scale * distance * math.hypot(
            observation.bearing_std, math.radians(cfg.landmark_bearing_std_floor_deg))
        ex = observation.position[0] - predicted_x                            # (N, K)
        ey = observation.position[1] - predicted_y
        along = (ex * ux + ey * uy) / sigma_along
        across = (ey * ux - ex * uy) / sigma_across
        fit = np.exp(-0.5 * (along * along + across * across))
        if cfg.correspondence == "nearest":
            fit = fit.max(axis=1)
        else:
            fit = np.minimum(fit.sum(axis=1), 1.0)
        return np.log(cfg.landmark_outlier_floor + (1.0 - cfg.landmark_outlier_floor) * fit)
