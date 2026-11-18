"""Watching the camera mount calibration with the one thing that is known exactly: the field.

An IMU tells the robot how its BODY is tilted. It says nothing about how the camera sits on
the body, so a mount that is pitched two degrees off its calibration misplaces every ground
point of every frame, and no inertial measurement notices. The painted lines do: a pitch error
moves a point at range ``r`` by about ``(r^2 + h^2) / h`` times the error, so near lines and
far lines stop agreeing on where the robot is. The localizer can absorb a shift, but not a
stretch that grows with range.

What this module does, per frame, with image-derived data and the static map only:

1. takes the line points the extractor reported (their pixels) and the localizer's pose;
2. finds, for each point, the nearest painted line of the map and the signed distance to it
   along that line's normal (a point-to-line residual);
3. linearises those residuals in five small corrections: the pose (x, y, yaw) and the
   camera's pitch and roll about its own axes;
4. eliminates the pose (it is a nuisance: the localizer's pose is itself bent by the camera
   error), which leaves what this frame says about pitch and roll and how firmly it says it;
5. accumulates that over many frames, with a prior that the mount is within a couple of
   degrees of its calibration and with old frames slowly forgotten.

Out of it come: a FIELD-LINE CONSISTENCY RESIDUAL (how well the lines fit, in metres), an
estimate of the persistent pitch / roll bias with its standard deviation, a status, and a
BOUNDED CORRECTION that is applied only when the estimate is well determined
(`MonitorConfig`). The correction never exceeds ``max_correction_deg`` and moves slowly.

How to read the numbers. A LARGE bias is UNDERESTIMATED at first: far points, which say the
most, are then too far from their lines to be matched, and the localizer's pose has already
bent to hide part of the error. With the correction switched off the reported bias is
therefore a lower bound in size (about 0.8 deg for a true 2 deg in the tests). With the
correction on, each step makes the next estimate better and the loop converges on the full
bias within some tens of seconds; this is why the memory is short and the correction gradual.
The reported standard deviation is the formal one of the line fit. It does not include that
underestimation and should be read as "how repeatable", not "how accurate".

What it cannot do, and does not claim:

* tell a pitch error from a camera-HEIGHT or focal-length error. All three stretch ranges;
  only pitch and roll are estimated, so a height error is partly absorbed as pitch;
* work without lines at several ranges in view, or before the localizer has converged (it
  then simply collects nothing);
* use ground truth. Nothing here has access to the simulator.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry, sample_arc
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ProjectionLimits, project_to_ground,
    require_estimate)

_LIMITS = ProjectionLimits(min_depression_deg=0.5, max_range=60.0, min_range=0.0)
_ANGLE_STEP = math.radians(0.1)
STATUSES = ("collecting", "ok", "bias_suspected", "correcting")


@dataclasses.dataclass(frozen=True)
class MonitorConfig:
    enabled_correction: bool = True     # False: estimate and warn, never correct
    gate: float = 0.25                  # metres: points further than this from every line are ignored
    gate_pitch_deg: float = 1.5         # ... widened for far points by what this pitch error would do
    huber: float = 2.0                  # residuals beyond this many sigma are down-weighted
    point_std_floor: float = 0.03       # metres added in quadrature to each point's pixel std
    min_points: int = 10                # a frame with fewer usable points is skipped
    max_position_std: float = 0.25      # metres: the localizer must be at least this sure
    pose_prior_xy: float = 0.30         # metres: how far the pose may move to fit the lines
    pose_prior_yaw_deg: float = 10.0
    prior_std_deg: float = 2.0          # belief: the mount is within about this of calibration
    forgetting: float = 0.985           # per frame; 0.985 keeps about 7 s at 10 Hz
    frame_weight: float = 0.2           # consecutive frames repeat each other's errors
    min_frames: int = 40                # frames collected before anything is said
    max_std_deg: float = 0.35           # a component must be this well determined to be corrected
    warn_deg: float = 0.8               # a bias above this (and above 3 sigma) raises the warning
    max_correction_deg: float = 3.0     # the correction is never larger than this
    correction_rate: float = 0.10       # share of the remaining gap closed per frame

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MonitorConfig":
        data = dict(data or {})
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown MonitorConfig keys: {sorted(unknown)}")
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class MonitorState:
    """What the monitor knows after a frame. Angles in degrees, (roll, pitch)."""
    status: str                         # one of STATUSES
    bias_deg: tuple[float, float]       # estimated persistent mount bias
    bias_std_deg: tuple[float, float]
    correction_deg: tuple[float, float]     # what is being applied
    frames: int                         # frames that contributed
    used: bool                          # this frame contributed
    points: int                         # line points used in this frame
    residual_rms: float                 # metres: recent field-line consistency residual

    @property
    def warning(self) -> bool:
        return self.status in ("bias_suspected", "correcting")


class CalibrationMonitor:
    """Persistent camera pitch / roll bias from field-line residuals. See the module docstring."""

    def __init__(self, field: FieldGeometry, cfg: MonitorConfig | None = None):
        if not isinstance(field, FieldGeometry):
            raise TypeError("field must be a FieldGeometry (the static map)")
        self.cfg = cfg or MonitorConfig()
        starts = [np.asarray(s.start, float) for s in field.segments]
        ends = [np.asarray(s.end, float) for s in field.segments]
        for arc in field.arcs:
            points = sample_arc(arc, 0.05)
            if arc.is_full_circle:
                points = np.vstack([points, points[:1]])
            starts += list(points[:-1])
            ends += list(points[1:])
        self._start = np.array(starts)
        self._delta = np.array(ends) - self._start
        self._length2 = np.maximum((self._delta ** 2).sum(axis=1), 1e-12)
        self.reset()

    def reset(self) -> None:
        self._information = np.zeros((2, 2))
        self._vector = np.zeros(2)
        self._correction = np.zeros(2)          # roll, pitch, radians
        self._residual_ms = math.nan
        self.frames = 0
        self.state = MonitorState("collecting", (0.0, 0.0), (math.inf, math.inf), (0.0, 0.0),
                                  0, False, 0, math.nan)

    @property
    def correction(self) -> tuple[float, float]:
        """The bounded (roll, pitch) correction to apply to the mount, radians."""
        return float(self._correction[0]), float(self._correction[1])

    # ------------------------------------------------------------------ geometry
    def _nearest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For field-frame points (K, 2): the nearest point on any painted line's centreline
        (K, 2) and the distance to it (K,)."""
        offset = points[:, None, :] - self._start[None, :, :]                 # (K, M, 2)
        along = np.clip((offset * self._delta).sum(axis=2) / self._length2, 0.0, 1.0)
        closest = self._start[None] + along[..., None] * self._delta[None]
        distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
        best = distance.argmin(axis=1)
        rows = np.arange(len(points))
        return closest[rows, best], distance[rows, best]

    # ------------------------------------------------------------------ one frame
    def update(self, features, estimate, calibration: EstimatedCameraCalibration,
               pose: EstimatedCameraPose) -> MonitorState:
        """Take in one frame: the extracted features, the localizer's estimate and the camera
        calibration and pose the features were projected with."""
        require_estimate(calibration, pose)
        cfg = self.cfg
        line_points = features.line_points
        used, count = False, 0
        if (len(line_points) >= cfg.min_points and estimate.converged
                and not estimate.symmetric_ambiguity
                and estimate.position_std <= cfg.max_position_std):
            used, count = self._accumulate(line_points, np.asarray(estimate.pose, float),
                                           calibration, pose)
        if not used:
            # Nothing learned from this frame; what was learned earlier still fades.
            self._information *= cfg.forgetting
            self._vector *= cfg.forgetting
        self.state = self._evaluate(used, count)
        return self.state

    def _accumulate(self, line_points, robot_pose: np.ndarray, calibration, pose
                    ) -> tuple[bool, int]:
        cfg = self.cfg
        pixels = np.asarray(line_points.pixels, dtype=float)
        ground = np.asarray(line_points.points, dtype=float)
        in_field = tf.se2_transform_points(robot_pose, ground)
        closest, distance = self._nearest(in_field)
        # A pitch error displaces a point at range r by about (r^2 + h^2) / h per radian, so
        # a far point may sit further from its line than a near one and still belong to it.
        height = max(pose.height, 0.1)
        reach = (np.sum(ground ** 2, axis=1) + height ** 2) / height
        keep = distance <= cfg.gate + reach * math.radians(cfg.gate_pitch_deg)
        if int(keep.sum()) < cfg.min_points:
            return False, int(keep.sum())
        pixels, ground, in_field = pixels[keep], ground[keep], in_field[keep]
        closest, distance = closest[keep], distance[keep]
        # The line's normal at each point. Where the point sits on the centreline the
        # direction is undefined and the point carries no signed residual: drop it.
        defined = distance > 1e-6
        normal = np.where(defined[:, None], (in_field - closest)
                          / np.maximum(distance, 1e-6)[:, None], 0.0)
        residual = distance                                     # signed along ``normal``
        sigma = np.hypot(np.asarray(line_points.pixel_std, float)[keep], cfg.point_std_floor)

        # ---- Jacobian of each field point in (x, y, yaw, camera pitch, camera roll)
        cos, sin = math.cos(robot_pose[2]), math.sin(robot_pose[2])
        rotation = np.array([[cos, -sin], [sin, cos]])
        d_yaw = (ground @ np.array([[-sin, -cos], [cos, -sin]]).T)          # d(R p)/d yaw
        columns = [np.tile([1.0, 0.0], (len(ground), 1)), np.tile([0.0, 1.0], (len(ground), 1)),
                   d_yaw]
        for name in ("d_pitch", "d_roll"):
            plus = project_to_ground(pixels, calibration, pose.perturbed(**{name: _ANGLE_STEP}),
                                     _LIMITS).points
            minus = project_to_ground(pixels, calibration,
                                      pose.perturbed(**{name: -_ANGLE_STEP}), _LIMITS).points
            columns.append(((plus - minus) / (2.0 * _ANGLE_STEP)) @ rotation.T)
        jacobian = np.stack([(column * normal).sum(axis=1) for column in columns], axis=1)
        finite = np.isfinite(jacobian).all(axis=1) & defined
        if int(finite.sum()) < cfg.min_points:
            return False, int(finite.sum())
        jacobian, residual, sigma = jacobian[finite], residual[finite], sigma[finite]

        scaled = residual / sigma
        weight = np.where(np.abs(scaled) <= cfg.huber, 1.0, cfg.huber / np.abs(scaled)) \
            / (sigma * sigma)
        hessian = jacobian.T @ (jacobian * weight[:, None])
        gradient = jacobian.T @ (weight * residual)
        hessian[:3, :3] += np.diag([cfg.pose_prior_xy ** -2, cfg.pose_prior_xy ** -2,
                                    math.radians(cfg.pose_prior_yaw_deg) ** -2])
        # ---- eliminate the pose: what is left is this frame's say about (pitch, roll)
        solve = np.linalg.solve(hessian[:3, :3], np.column_stack([hessian[:3, 3:],
                                                                  gradient[:3]]))
        information = hessian[3:, 3:] - hessian[3:, :3] @ solve[:, :2]
        reduced = gradient[3:] - hessian[3:, :3] @ solve[:, 2]
        # ``information @ (remaining bias) = -reduced``; the total bias is the correction that
        # is already applied plus what remains. Order here is (pitch, roll): store (roll, pitch).
        swap = np.array([[0.0, 1.0], [1.0, 0.0]])
        information = swap @ information @ swap
        reduced = swap @ reduced
        self._information = cfg.forgetting * self._information + cfg.frame_weight * information
        self._vector = cfg.forgetting * self._vector + cfg.frame_weight * (
            information @ self._correction - reduced)
        self.frames += 1
        mean_square = float(np.mean(residual ** 2))
        self._residual_ms = mean_square if math.isnan(self._residual_ms) \
            else 0.9 * self._residual_ms + 0.1 * mean_square
        return True, int(finite.sum())

    def _evaluate(self, used: bool, count: int) -> MonitorState:
        cfg = self.cfg
        prior = np.eye(2) / math.radians(cfg.prior_std_deg) ** 2
        covariance = np.linalg.inv(self._information + prior)
        bias = covariance @ self._vector
        std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        limit = math.radians(cfg.max_correction_deg)
        determined = std <= math.radians(cfg.max_std_deg)
        status = "collecting"
        if self.frames >= cfg.min_frames:
            status = "ok"
            significant = (np.abs(bias) > math.radians(cfg.warn_deg)) & (np.abs(bias) > 3.0 * std)
            if significant.any():
                status = "bias_suspected"
            if cfg.enabled_correction:
                target = np.where(determined, np.clip(bias, -limit, limit), self._correction)
                self._correction = self._correction + cfg.correction_rate * (
                    target - self._correction)
                if np.any(np.abs(self._correction) > math.radians(0.2)):
                    status = "correcting"
        rms = math.sqrt(self._residual_ms) if math.isfinite(self._residual_ms) else math.nan
        return MonitorState(status, tuple(float(v) for v in np.degrees(bias)),
                            tuple(float(v) for v in np.degrees(std)),
                            tuple(float(v) for v in np.degrees(self._correction)),
                            self.frames, used, count, rms)
