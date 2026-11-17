"""Image pixel -> point on the ground in the robot's base frame, with validity and uncertainty.

Pure NumPy, no simulator. See docs/geometric_perception.md for the derivation.

What the robot knows, and what it does not
------------------------------------------
Everything here works from the robot's own BELIEF about its camera:

* `EstimatedCameraCalibration`: intrinsics and the fixed part of the mount, plus how far each
  of them is trusted (1-sigma values);
* `EstimatedCameraPose`: where the camera's optical frame is relative to the base in this
  frame, from head kinematics and (optionally) the IMU.

Neither is the simulator's true camera pose. The true pose lives in
`vision_benchmark.camera_truth.SimulationCameraTruth`, a different type that this module never
sees; the two cannot be passed in each other's place (`require_estimate`).

Geometry
--------
A pixel ``(u, v)`` (pixel-centre convention of `geometry.camera`) has the viewing ray
``d_c = [(u - cx) / fx, (v - cy) / fy, 1]`` in the optical frame. In the base frame the ray is
``o + s * R d_c`` with ``R, o`` the rotation and translation of ``T_base_camera``. It meets
the ground ``z = 0`` at ``s = -o_z / (R d_c)_z``, which exists only for rays that point down.

Uncertainty
-----------
The ground point is a function of the pixel and of the camera parameters (height, pitch, roll,
yaw, focal length). Its covariance is propagated with numerical Jacobians and kept in two
parts, because they behave differently across a frame:

* the PIXEL part is independent from one pixel to the next;
* the CAMERA part is common to every pixel of the frame: a pitch error moves all points
  together. Averaging many points does not reduce it.
"""
from __future__ import annotations

import dataclasses
import enum
import math

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import CameraIntrinsics, StereoCameraConfig


# --------------------------------------------------------------------------------------------
# What the robot believes about its camera
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CalibrationUncertainty:
    """1-sigma trust in the camera model. These are beliefs, not measurements of a real rig."""
    pixel_std: float = 1.0          # px: how well a feature's pixel is known
    height_std: float = 0.02        # m
    pitch_std_deg: float = 0.5
    roll_std_deg: float = 0.5
    yaw_std_deg: float = 0.5
    focal_rel_std: float = 0.01     # relative error of fx and fy

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if getattr(self, field.name) < 0.0:
                raise ValueError(f"{field.name} must be >= 0")

    @classmethod
    def from_dict(cls, data: dict | None) -> "CalibrationUncertainty":
        return cls(**(data or {}))

    @property
    def angle_stds(self) -> tuple[float, float, float]:
        """(pitch, roll, yaw) in radians."""
        return (math.radians(self.pitch_std_deg), math.radians(self.roll_std_deg),
                math.radians(self.yaw_std_deg))


@dataclasses.dataclass(frozen=True)
class EstimatedCameraCalibration:
    """The robot's calibration of one camera: intrinsics, the fixed mount offsets, and trust.

    ``forward_offset`` / ``lateral_offset`` locate the rig centre in the base frame;
    ``optical_offset_y`` is the optical centre relative to the rig centre along the rig's +y
    (the left camera of a stereo pair sits at ``+baseline / 2``). ``nominal_height`` and
    ``nominal_pitch`` are the mount's design values, used when no per-frame value is given.
    """
    intrinsics: CameraIntrinsics
    nominal_height: float
    nominal_pitch: float                # radians, positive looks down
    forward_offset: float = 0.0
    lateral_offset: float = 0.0
    optical_offset_y: float = 0.0
    uncertainty: CalibrationUncertainty = dataclasses.field(default_factory=CalibrationUncertainty)

    @classmethod
    def from_camera_config(cls, cfg: StereoCameraConfig, side: str = "left",
                           uncertainty: CalibrationUncertainty | None = None
                           ) -> "EstimatedCameraCalibration":
        """The nominal calibration written in a ``configs/cameras/*.yaml`` file."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        offset = 0.5 * cfg.baseline * (1.0 if side == "left" else -1.0)
        return cls(cfg.intrinsics, cfg.mount.height, cfg.mount.pitch, cfg.mount.forward_offset,
                   cfg.mount.lateral_offset, offset, uncertainty or CalibrationUncertainty())

    def replace(self, **changes) -> "EstimatedCameraCalibration":
        return dataclasses.replace(self, **changes)

    def with_intrinsics_error(self, focal_scale: float = 1.0, cx_shift: float = 0.0,
                              cy_shift: float = 0.0) -> "EstimatedCameraCalibration":
        """A deliberately wrong calibration, for sensitivity studies."""
        k = self.intrinsics
        return self.replace(intrinsics=CameraIntrinsics(
            k.width, k.height, k.fx * focal_scale, k.fy * focal_scale, k.cx + cx_shift,
            k.cy + cy_shift))

    def pose(self, pan: float = 0.0, pitch: float | None = None, roll: float = 0.0,
             height: float | None = None) -> "EstimatedCameraPose":
        """The camera pose the robot believes it has for a head pan / pitch / roll (radians)."""
        return EstimatedCameraPose.from_mount(
            self, self.nominal_height if height is None else height,
            self.nominal_pitch if pitch is None else pitch, roll, pan)


@dataclasses.dataclass(frozen=True)
class EstimatedCameraPose:
    """The robot's estimate of ``T_base_camera`` (optical frame in the base frame) for one frame.

    It comes from the mount calibration and head joint angles, optionally corrected by the
    IMU's roll and pitch (`with_base_tilt`). It is an estimate: nothing guarantees that it
    equals the true camera pose, and the benchmark deliberately makes it differ.
    """
    T_base_camera: np.ndarray

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_base_camera, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("T_base_camera must be a 4x4 homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("T_base_camera does not contain a rotation matrix")
        object.__setattr__(self, "T_base_camera", transform)

    @classmethod
    def from_mount(cls, calibration: EstimatedCameraCalibration, height: float, pitch: float,
                   roll: float = 0.0, pan: float = 0.0) -> "EstimatedCameraPose":
        """Same chain as `StereoCameraModel.T_base_camera`:
        ``Trans(forward, lateral, height) Rz(pan) Ry(pitch) Rx(roll) Trans(0, off_y, 0) R_BODY_OPTICAL``.
        """
        rotation = tf.rot_z(pan) @ tf.rot_y(pitch) @ tf.rot_x(roll)
        T_base_rig = tf.make_transform(
            rotation, [calibration.forward_offset, calibration.lateral_offset, height])
        T_rig_optical = tf.make_transform(tf.R_BODY_OPTICAL,
                                          [0.0, calibration.optical_offset_y, 0.0])
        return cls(T_base_rig @ T_rig_optical)

    # ------------------------------------------------------------------ derived quantities
    @property
    def height(self) -> float:
        """Height of the optical centre above the ground, metres."""
        return float(self.T_base_camera[2, 3])

    @property
    def depression(self) -> float:
        """Angle of the optical axis below the horizontal, radians (the effective pitch)."""
        axis = self.T_base_camera[:3, 2]
        return float(math.atan2(-axis[2], math.hypot(axis[0], axis[1])))

    @property
    def heading(self) -> float:
        """Direction of the optical axis on the ground, radians from base +x."""
        axis = self.T_base_camera[:3, 2]
        return float(math.atan2(axis[1], axis[0]))

    # ------------------------------------------------------------------ changes
    def perturbed(self, d_height: float = 0.0, d_pitch: float = 0.0, d_roll: float = 0.0,
                  d_yaw: float = 0.0) -> "EstimatedCameraPose":
        """This pose with small changes: height along base +z, pitch about the camera's own
        lateral axis (positive looks further down), roll about its viewing direction, yaw about
        base +z. Used for uncertainty propagation and for calibration-error experiments."""
        rotation, translation = self.T_base_camera[:3, :3], self.T_base_camera[:3, 3]
        R_base_rig = rotation @ tf.R_BODY_OPTICAL.T
        yaw = tf.rot_z(d_yaw)
        new_rotation = yaw @ R_base_rig @ tf.rot_y(d_pitch) @ tf.rot_x(d_roll) @ tf.R_BODY_OPTICAL
        new_translation = yaw @ translation + np.array([0.0, 0.0, d_height])
        return EstimatedCameraPose(tf.make_transform(new_rotation, new_translation))

    def _jacobian_transforms(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``T_base_camera`` at +- one finite-difference step of each pose parameter. Computed
        once per pose: every feature of a frame is propagated through the same set."""
        cached = self.__dict__.get("_jacobian_cache")
        if cached is None:
            cached = {name: (self.perturbed(**{name: step}).T_base_camera,
                             self.perturbed(**{name: -step}).T_base_camera)
                      for name, step in (("d_height", _HEIGHT_STEP), ("d_pitch", _ANGLE_STEP),
                                         ("d_roll", _ANGLE_STEP), ("d_yaw", _ANGLE_STEP))}
            object.__setattr__(self, "_jacobian_cache", cached)
        return cached

    def with_base_tilt(self, roll: float, pitch: float) -> "EstimatedCameraPose":
        """Account for a tilted torso: the IMU says the base is rolled / pitched (radians)
        relative to the level footprint frame that the ground plane is defined in."""
        tilt = tf.make_transform(tf.rot_y(pitch) @ tf.rot_x(roll))
        return EstimatedCameraPose(tilt @ self.T_base_camera)


def require_estimate(calibration, pose) -> None:
    """Refuse anything that is not the robot's own estimate (for example simulator truth)."""
    if type(calibration) is not EstimatedCameraCalibration:
        raise TypeError("camera_calibration must be an EstimatedCameraCalibration, got "
                        f"{type(calibration).__name__}")
    if type(pose) is not EstimatedCameraPose:
        raise TypeError("estimated_camera_pose must be an EstimatedCameraPose, got "
                        f"{type(pose).__name__}")


# --------------------------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------------------------
class ProjectionStatus(enum.IntFlag):
    """Why a pixel has no usable ground point. 0 means valid."""
    VALID = 0
    ABOVE_HORIZON = 1       # the ray does not point down: it never reaches the ground
    NEAR_HORIZON = 2        # it does, but at too shallow an angle to be trusted
    TOO_FAR = 4
    TOO_CLOSE = 8
    UNSTABLE = 16           # non-finite input or result
    OUTSIDE_IMAGE = 32      # the pixel is not inside the image


@dataclasses.dataclass(frozen=True)
class ProjectionLimits:
    min_depression_deg: float = 4.0     # rays flatter than this are rejected
    max_range: float = 7.0              # metres from the robot (base origin)
    min_range: float = 0.20
    image_margin_px: float = 0.0        # pixels this close to the border are rejected

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProjectionLimits":
        return cls(**(data or {}))


@dataclasses.dataclass(frozen=True)
class GroundProjection:
    points: np.ndarray          # (N, 2) metres in the base frame; NaN where the ray misses
    valid: np.ndarray           # (N,) bool
    status: np.ndarray          # (N,) uint8, OR of `ProjectionStatus` flags
    range: np.ndarray           # (N,) metres from the base origin
    depression: np.ndarray      # (N,) radians below the horizontal (negative = looking up)

    def __len__(self) -> int:
        return len(self.points)

    def count(self, flag: ProjectionStatus) -> int:
        return int(np.count_nonzero(self.status & int(flag)))


def pixel_rays(pixels_uv, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Unit viewing rays (..., 3) in the optical frame of pixel-centre coordinates (..., 2)."""
    rays = intrinsics.ray(np.asarray(pixels_uv, dtype=float))
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _intersect(pixels: np.ndarray, intrinsics: CameraIntrinsics, T_base_camera: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Raw ground points (N, 2) and ray depression (N,); NaN where the ray does not descend."""
    rays = intrinsics.ray(pixels) @ T_base_camera[:3, :3].T
    origin = T_base_camera[:3, 3]
    horizontal = np.hypot(rays[:, 0], rays[:, 1])
    depression = np.arctan2(-rays[:, 2], horizontal)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(rays[:, 2] < -1e-12, -origin[2] / rays[:, 2], np.nan)
        points = origin[:2] + scale[:, None] * rays[:, :2]
    return points, depression


def project_to_ground(pixels_uv, calibration: EstimatedCameraCalibration,
                      pose: EstimatedCameraPose, limits: ProjectionLimits | None = None
                      ) -> GroundProjection:
    """Intersect the viewing rays of pixels (N, 2) with the ground plane of the base frame."""
    require_estimate(calibration, pose)
    limits = limits or ProjectionLimits()
    pixels = np.asarray(pixels_uv, dtype=float).reshape(-1, 2)
    k = calibration.intrinsics
    status = np.zeros(len(pixels), dtype=np.uint8)
    finite = np.isfinite(pixels).all(axis=1)
    safe = np.where(finite[:, None], pixels, 0.0)
    points, depression = _intersect(safe, k, pose.T_base_camera)
    def flag(where: np.ndarray, reason: ProjectionStatus) -> None:
        status[where] |= np.uint8(int(reason))

    flag(~finite, ProjectionStatus.UNSTABLE)
    flag(~k.in_image(safe, limits.image_margin_px), ProjectionStatus.OUTSIDE_IMAGE)
    if pose.height <= 0.0:
        flag(np.ones(len(pixels), dtype=bool), ProjectionStatus.UNSTABLE)
    above = depression <= 0.0
    flag(above, ProjectionStatus.ABOVE_HORIZON)
    flag(~above & (depression < math.radians(limits.min_depression_deg)),
         ProjectionStatus.NEAR_HORIZON)
    with np.errstate(invalid="ignore"):
        distance = np.hypot(points[:, 0], points[:, 1])
        flag(distance > limits.max_range, ProjectionStatus.TOO_FAR)
        flag(distance < limits.min_range, ProjectionStatus.TOO_CLOSE)
    flag(~above & ~np.isfinite(distance), ProjectionStatus.UNSTABLE)
    valid = status == 0
    return GroundProjection(points, valid, status, distance, depression)


def ground_to_pixels(points_base, calibration: EstimatedCameraCalibration,
                     pose: EstimatedCameraPose) -> tuple[np.ndarray, np.ndarray]:
    """Where ground points (N, 2) of the base frame appear in the image, by the robot's own
    calibration. Returns ``(pixels (N, 2), in_image (N,))``; pixels behind the camera are NaN."""
    require_estimate(calibration, pose)
    points = np.asarray(points_base, dtype=float).reshape(-1, 2)
    points3 = np.column_stack([points, np.zeros(len(points))])
    in_camera = tf.transform_points(tf.invert_transform(pose.T_base_camera), points3)
    pixels, depth = calibration.intrinsics.project(in_camera)
    return pixels, (depth > 0.0) & calibration.intrinsics.in_image(pixels)


# --------------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ProjectionCovariance:
    """Covariance of ground points in the base frame, split by cause.

    ``pixel`` (N, 2, 2) is independent between points; ``camera`` (N, 2, 2) is the effect of
    the uncertain camera pose and focal length, common to all points of one frame.
    """
    pixel: np.ndarray
    camera: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.pixel + self.camera

    def std_along(self, directions, part: str = "total") -> np.ndarray:
        """1-sigma error (N,) of each point along a unit direction (N, 2) or (2,)."""
        directions = np.broadcast_to(np.asarray(directions, dtype=float), (len(self.pixel), 2))
        covariance = getattr(self, part)
        return np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", directions, covariance, directions),
                                  0.0))

    def std_rms(self, part: str = "total") -> np.ndarray:
        """Direction-free 1-sigma (N,): sqrt of the mean of the two principal variances."""
        covariance = getattr(self, part)
        return np.sqrt(np.maximum(0.5 * (covariance[:, 0, 0] + covariance[:, 1, 1]), 0.0))

    def polar(self, points_base, part: str = "total") -> tuple[np.ndarray, np.ndarray]:
        """(range std in metres, bearing std in radians) of points (N, 2) seen from the base."""
        points = np.asarray(points_base, dtype=float).reshape(-1, 2)
        distance = np.maximum(np.hypot(points[:, 0], points[:, 1]), 1e-6)
        radial = points / distance[:, None]
        tangential = np.stack([-radial[:, 1], radial[:, 0]], axis=1)
        return self.std_along(radial, part), self.std_along(tangential, part) / distance


#: Finite-difference steps for the numerical Jacobians.
_PIXEL_STEP, _ANGLE_STEP, _HEIGHT_STEP, _FOCAL_STEP = 0.5, 1e-3, 1e-3, 1e-3
#: Variance given to a point whose neighbourhood cannot be projected (metres^2).
_UNDEFINED_VARIANCE = 1e4


def projection_covariance(pixels_uv, calibration: EstimatedCameraCalibration,
                          pose: EstimatedCameraPose,
                          pixel_std: float | np.ndarray | tuple | None = None
                          ) -> ProjectionCovariance:
    """First-order covariance of the ground points of pixels (N, 2).

    ``pixel_std`` overrides the calibration's value: a scalar, one value per pixel, or
    ``(std_u, std_v)`` for different horizontal and vertical uncertainty.
    Points whose perturbed rays leave the ground get a huge variance instead of NaN.
    """
    require_estimate(calibration, pose)
    pixels = np.asarray(pixels_uv, dtype=float).reshape(-1, 2)
    k, trust = calibration.intrinsics, calibration.uncertainty

    transform = pose.T_base_camera

    def at(pix=pixels, intrinsics=k, camera=transform) -> np.ndarray:
        return _intersect(pix, intrinsics, camera)[0]

    def derivative(plus: np.ndarray, minus: np.ndarray, step: float) -> np.ndarray:
        return (plus - minus) / (2.0 * step)

    du = derivative(at(pixels + [_PIXEL_STEP, 0.0]), at(pixels - [_PIXEL_STEP, 0.0]), _PIXEL_STEP)
    dv = derivative(at(pixels + [0.0, _PIXEL_STEP]), at(pixels - [0.0, _PIXEL_STEP]), _PIXEL_STEP)
    if pixel_std is None:
        std_u = std_v = trust.pixel_std
    elif isinstance(pixel_std, tuple):
        std_u, std_v = pixel_std
    else:
        std_u = std_v = np.asarray(pixel_std, dtype=float)
    scaled_u = du * np.reshape(std_u, (-1, 1)) if np.ndim(std_u) else du * std_u
    scaled_v = dv * np.reshape(std_v, (-1, 1)) if np.ndim(std_v) else dv * std_v
    pixel_part = _outer(scaled_u) + _outer(scaled_v)

    pitch_std, roll_std, yaw_std = trust.angle_stds
    perturbed = pose._jacobian_transforms()
    camera_part = np.zeros_like(pixel_part)
    for std, name, step in ((trust.height_std, "d_height", _HEIGHT_STEP),
                            (pitch_std, "d_pitch", _ANGLE_STEP),
                            (roll_std, "d_roll", _ANGLE_STEP),
                            (yaw_std, "d_yaw", _ANGLE_STEP)):
        if std > 0.0:
            plus, minus = perturbed[name]
            camera_part += _outer(derivative(at(camera=plus), at(camera=minus), step) * std)
    if trust.focal_rel_std > 0.0:
        def scaled(factor: float) -> CameraIntrinsics:
            return CameraIntrinsics(k.width, k.height, k.fx * factor, k.fy * factor, k.cx, k.cy)
        jac = derivative(at(intrinsics=scaled(1.0 + _FOCAL_STEP)),
                         at(intrinsics=scaled(1.0 - _FOCAL_STEP)), _FOCAL_STEP)
        camera_part += _outer(jac * trust.focal_rel_std)

    undefined = ~(np.isfinite(pixel_part).all(axis=(1, 2))
                  & np.isfinite(camera_part).all(axis=(1, 2)))
    if undefined.any():
        pixel_part[undefined] = np.eye(2) * _UNDEFINED_VARIANCE
        camera_part[undefined] = 0.0
    return ProjectionCovariance(pixel_part, camera_part)


def _outer(vectors: np.ndarray) -> np.ndarray:
    return vectors[:, :, None] * vectors[:, None, :]


def range_sensitivity_to_pitch(ground_range, camera_height: float) -> np.ndarray:
    """Analytic ``d range / d pitch`` (metres per radian) for a point straight ahead:
    ``(r^2 + h^2) / h``. A reference for the tests and the documentation."""
    ground_range = np.asarray(ground_range, dtype=float)
    return (ground_range ** 2 + camera_height ** 2) / camera_height
