"""Oracle perception: noisy, anonymous detections generated from known simulator geometry.

The oracle stands in for the neural detectors and geometric extraction that do not exist yet.
Given the TRUE robot pose it works out what the camera could see, then degrades it the way real
perception would: limited field of view and range, occlusion, missed detections, measurement
noise that grows with range, spurious detections, gross outliers and class mix-ups.

What it returns is expressed in the robot's base frame and carries no identities and no global
pose (`ObservationFrame`). Identities are returned separately in `OracleDebug` for evaluation.

Pure NumPy: visibility uses the same `StereoCameraModel` as the renderer, so no rendering is
needed to generate observations.
"""
from __future__ import annotations

import math

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import RigState, StereoCameraModel
from robosoccer_perception.geometry.field import FieldGeometry, sample_arc
from robosoccer_perception.geometry.landmarks import LandmarkMap, LandmarkType
from robosoccer_perception.observations.noise import OracleConfig, apply_pitch_error
from robosoccer_perception.observations.types import (
    LandmarkObservation, LinePointsObservation, ObservationFrame, OracleDebug)

ALL_TYPES = tuple(LandmarkType)
JUNCTION_TYPES = (LandmarkType.L_INTERSECTION, LandmarkType.T_INTERSECTION,
                  LandmarkType.X_INTERSECTION)


class OracleObservationGenerator:
    """Turns a true pose into the observations a camera-based perception stack might produce."""

    def __init__(self, field: FieldGeometry, landmarks: LandmarkMap, camera: StereoCameraModel,
                 cfg: OracleConfig | None = None):
        self.field = field
        self.landmarks = landmarks
        self.camera = camera
        self.cfg = cfg or OracleConfig()
        self._line_candidates = field.sample_line_points(self.cfg.line_point_spacing)
        self._circle_points = sample_arc(field.arc("center_circle"), 0.05)
        # Vertical cylinders (x, y, radius, height) that block the view: the goal posts.
        self._post_occluders = np.array(
            [[*post, goal.dims.post_diameter / 2.0, goal.dims.height]
             for goal in field.goals for post in goal.posts])

    # ------------------------------------------------------------------ visibility
    def _occluders(self, ball_position, ball_radius: float, extra) -> np.ndarray:
        cylinders = [self._post_occluders]
        if ball_position is not None:
            cylinders.append([[ball_position[0], ball_position[1], ball_radius,
                               2.0 * ball_radius]])
        if extra is not None and len(extra):
            cylinders.append(np.asarray(extra, dtype=float).reshape(-1, 4))
        return np.vstack(cylinders)

    def visible(self, base_pose, rig: RigState, points_field: np.ndarray,
                occluders: np.ndarray | None = None) -> np.ndarray:
        """Which ground points (N, 2) the left camera can detect from this pose."""
        cfg = self.cfg
        points3 = np.column_stack([points_field, np.zeros(len(points_field))])
        pixels, depth, in_view = self.camera.project_field_points(base_pose, points3, "left", rig)
        in_view &= self.camera.intrinsics.in_image(pixels, cfg.image_margin_px)
        ground_range = np.linalg.norm(points_field - np.asarray(base_pose[:2], float), axis=1)
        in_view &= (ground_range >= cfg.min_range) & (ground_range <= cfg.max_range)
        if cfg.occlusion and occluders is not None and in_view.any():
            camera_position = self.camera.T_field_camera(base_pose, "left", rig)[:3, 3]
            hidden = _occluded(camera_position, points3[in_view], occluders)
            in_view[np.flatnonzero(in_view)[hidden]] = False
        return in_view

    # ------------------------------------------------------------------ generation
    def generate(self, base_pose, rng: np.random.Generator, rig: RigState | None = None,
                 timestamp: float = 0.0, ball_position=None, ball_radius: float = 0.07,
                 extra_occluders=None) -> tuple[ObservationFrame, OracleDebug]:
        """Observations from the true ``base_pose`` ``[x, y, yaw]``.

        ``extra_occluders`` is an optional (K, 4) array of vertical cylinders
        ``(x, y, radius, height)``, for example other robots.
        """
        cfg = self.cfg
        rig = rig or self.camera.config.mount
        base_pose = np.asarray(base_pose, dtype=float)
        if rng.random() < cfg.frame_dropout_prob:
            return ObservationFrame(timestamp), OracleDebug((), 0, 0, frame_dropped=True)

        occluders = self._occluders(ball_position, ball_radius, extra_occluders)
        pitch_error = rng.normal(0.0, math.radians(cfg.pitch_jitter_std_deg)) \
            if cfg.pitch_jitter_std_deg > 0 else 0.0

        def measure(points_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Noisy base-frame positions and their range std for true field positions."""
            relative = tf.se2_inverse_transform_points(base_pose, points_field)
            distance = np.hypot(relative[:, 0], relative[:, 1])
            bearing = np.arctan2(relative[:, 1], relative[:, 0])
            distance = apply_pitch_error(distance, rig.height, pitch_error)
            distance = distance + rng.normal(0.0, cfg.range_std(distance))
            bearing = bearing + rng.normal(0.0, cfg.bearing_std, size=len(bearing))
            distance = np.maximum(distance, 0.05)
            return _polar_to_xy(distance, bearing), cfg.range_std(distance)

        # ---- landmarks
        observations: list[LandmarkObservation] = []
        ids: list[str | None] = []
        for kind in LandmarkType:
            positions = self.landmarks.positions(kind)
            if kind == LandmarkType.CENTER_MARK:
                # The circle's centre is inferred from the visible part of the circle.
                on_circle = self.visible(base_pose, rig, self._circle_points, occluders)
                seen = np.array([on_circle.mean() >= cfg.min_circle_visible_fraction])
            else:
                seen = self.visible(base_pose, rig, positions, occluders)
            seen &= rng.random(len(positions)) >= cfg.landmark_dropout_prob
            if not seen.any():
                continue
            measured, range_std = measure(positions[seen])
            for landmark, position, std in zip(
                    np.asarray(self.landmarks.of_type(kind), dtype=object)[seen],
                    measured, range_std):
                reported = kind
                if rng.random() < cfg.landmark_outlier_prob:
                    position = _polar_to_xy(
                        np.hypot(*position) * rng.uniform(0.5, 1.6),
                        np.arctan2(position[1], position[0]) + rng.normal(0.0, math.radians(15)))
                if kind in JUNCTION_TYPES and rng.random() < cfg.class_confusion_prob:
                    others = [t for t in JUNCTION_TYPES if t != kind]
                    reported = others[rng.integers(len(others))]
                observations.append(LandmarkObservation(
                    reported, np.asarray(position, float), float(std),
                    cfg.bearing_std))
                ids.append(landmark.id)

        half_fov = 0.5 * self.camera.intrinsics.horizontal_fov

        def spurious(count: int) -> np.ndarray:
            """Random base-frame positions inside the camera's horizontal view sector."""
            distance = rng.uniform(max(cfg.min_range, 0.5), cfg.max_range, size=count)
            bearing = rig.pan + rng.uniform(-half_fov, half_fov, size=count)
            return _polar_to_xy(distance, bearing)

        for position in spurious(rng.poisson(cfg.false_landmark_rate)):
            kind = ALL_TYPES[rng.integers(len(ALL_TYPES))]
            observations.append(LandmarkObservation(
                kind, position, float(cfg.range_std(np.hypot(*position))),
                cfg.bearing_std))
            ids.append(None)

        order = rng.permutation(len(observations))        # the order must not carry information
        observations = [observations[i] for i in order]
        ids = [ids[i] for i in order]

        # ---- field-line points
        seen = self.visible(base_pose, rig, self._line_candidates, occluders)
        seen &= rng.random(len(seen)) >= cfg.line_point_dropout_prob
        chosen = np.flatnonzero(seen)
        if len(chosen) > cfg.max_line_points:
            chosen = rng.choice(chosen, size=cfg.max_line_points, replace=False)
        points, std = measure(self._line_candidates[chosen]) if len(chosen) else \
            (np.zeros((0, 2)), np.zeros(0))
        false_count = int(rng.binomial(len(chosen), cfg.false_line_point_fraction)) \
            if len(chosen) else 0
        if false_count:
            false_points = spurious(false_count)
            points = np.vstack([points, false_points])
            std = np.concatenate([std, cfg.range_std(np.hypot(false_points[:, 0],
                                                              false_points[:, 1]))])
            order = rng.permutation(len(points))
            points, std = points[order], std[order]

        frame = ObservationFrame(timestamp, tuple(observations),
                                 LinePointsObservation(points, std))
        return frame, OracleDebug(tuple(ids), len(chosen), false_count)


def _polar_to_xy(distance, bearing) -> np.ndarray:
    return np.stack([distance * np.cos(bearing), distance * np.sin(bearing)], axis=-1)


def _occluded(camera_position: np.ndarray, points: np.ndarray, cylinders: np.ndarray) -> np.ndarray:
    """True for points (N, 3) whose line of sight from the camera passes through a cylinder.

    Each cylinder is vertical: ``(x, y, radius, height)``, standing on the ground. A point
    inside a cylinder's footprint (a goal post's own base) is not occluded by that cylinder.
    """
    hidden = np.zeros(len(points), dtype=bool)
    origin = camera_position[:2]
    direction = points[:, :2] - origin                       # ray: origin + t * direction
    a = np.einsum("ij,ij->i", direction, direction)
    for cx, cy, radius, height in cylinders:
        offset = origin - [cx, cy]
        b = 2.0 * direction @ offset
        c = offset @ offset - radius ** 2
        discriminant = b * b - 4.0 * a * c
        with np.errstate(invalid="ignore", divide="ignore"):
            root = np.sqrt(np.maximum(discriminant, 0.0))
            t_in = np.clip((-b - root) / (2.0 * a), 0.0, 1.0)
            t_out = np.clip((-b + root) / (2.0 * a), 0.0, 1.0)
        crosses = (discriminant > 0.0) & (t_out > t_in)
        # Height of the ray while it is inside the cylinder's footprint (lowest at t_out).
        ray_height = camera_position[2] + t_out * (points[:, 2] - camera_position[2])
        target_outside = np.linalg.norm(points[:, :2] - [cx, cy], axis=1) > radius + 1e-9
        hidden |= crosses & (ray_height <= height) & target_outside
    return hidden
