"""The robot's estimate of its camera pose for each image, from the mount calibration and the
IMU. Pure Python / NumPy; no simulator.

    IMU samples ──► TiltEstimator ──► OrientationHistory ──┐
                                                           ├─► tilt at the image timestamp
    image timestamp ───────────────────────────────────────┘          │
    CameraMount (the robot's calibration of the mount) ───────────────┤
    BodyGeometry (standing height of the body frame) ─────────────────┤
                                                                      ▼
          T_footprint_camera = T_footprint_body(height, roll, pitch) @ T_body_rig @ T_rig_optical
                                                                      │
                                                                      ▼
                                       `EstimatedCameraPose` + per-frame `CalibrationUncertainty`

The result goes into `VisionLocalizationPipeline.step`, whose interface is unchanged from
Phase 3: an image, a timestamp, an odometry increment, a calibration and an estimated camera
pose.

Modes (`FusionConfig.mode`)

* ``fixed``  the body is assumed upright and level in every frame. This is the Phase 3
             behaviour on a moving body, and the baseline of the Phase 4 benchmark.
* ``imu``    roll and pitch come from the orientation filter, interpolated to the image
             timestamp (``sync: interpolate``) or, for comparison, simply the newest estimate
             (``sync: latest``).

``use_uncertainty`` additionally turns the filter's and the synchronisation's uncertainty into
the calibration uncertainty of that frame, so geometric extraction widens the standard
deviations of what it reports when the camera orientation is poorly known.

LIMIT. This estimator knows the BODY's tilt. It turns that into the CAMERA's orientation with
the mount transform it was given. If that transform is wrong, every frame is wrong by the same
amount and no IMU reading can reveal it; `estimation.calibration_monitor` looks for that kind
of error in the images instead. A correction found there is applied here through
`set_mount_correction`, bounded and labelled.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.estimation.imu_types import ImuSample
from robosoccer_perception.estimation.orientation import TiltEstimator, TiltFilterConfig
from robosoccer_perception.estimation.sync import (
    NONE, OrientationHistory, SyncConfig, SyncedTilt)
from robosoccer_perception.geometry import mount as mount_geometry
from robosoccer_perception.geometry import quaternion as quat
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.ground_projection import (
    CalibrationUncertainty, EstimatedCameraCalibration, EstimatedCameraPose)
from robosoccer_perception.geometry.mount import BodyGeometry, CameraMount

MODES = ("fixed", "imu")
SYNC_MODES = ("interpolate", "latest")


@dataclasses.dataclass(frozen=True)
class FusionConfig:
    """How the camera pose is estimated. One ``fusion`` block of a benchmark arm."""
    mode: str = "imu"
    sync: str = "interpolate"
    use_uncertainty: bool = False
    height_std_moving: float = 0.010        # metres: the body bobs while it walks
    moving_rate: float = 0.15               # rad/s of body rate above which it "moves"
    max_tilt_std_deg: float = 6.0           # a filter that reports more than this is not used
    filter: TiltFilterConfig = dataclasses.field(default_factory=TiltFilterConfig)
    synchronisation: SyncConfig = dataclasses.field(default_factory=SyncConfig)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {SYNC_MODES}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FusionConfig":
        data = dict(data or {})
        data["filter"] = TiltFilterConfig.from_dict(data.get("filter"))
        data["synchronisation"] = SyncConfig(**(data.get("synchronisation") or {}))
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown FusionConfig keys: {sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "FusionConfig":
        return dataclasses.replace(self, **changes)


@dataclasses.dataclass(frozen=True)
class CameraPoseEstimate:
    """What the robot believes about its camera for one image."""
    pose: EstimatedCameraPose
    calibration: EstimatedCameraCalibration     # intrinsics and this frame's uncertainty
    body_roll: float                # radians: the body tilt that was used (0 in ``fixed`` mode)
    body_pitch: float
    tilt_std: float                 # radians; NaN when no IMU estimate was used
    source: str                     # "nominal" | "imu" | "imu_latest" | "nominal_fallback"
    sync: SyncedTilt | None = None

    @property
    def camera_height(self) -> float:
        return self.pose.height

    @property
    def camera_roll_pitch(self) -> tuple[float, float]:
        """Roll and pitch of the camera (rig axes) in the footprint frame, radians."""
        return camera_roll_pitch(self.pose.T_base_camera)


def camera_roll_pitch(T_footprint_optical: np.ndarray) -> tuple[float, float]:
    """``(roll, pitch)`` of the rig axes of a camera whose optical frame is given: pitch is the
    angle the camera looks down by, roll the rotation about its viewing direction."""
    R_footprint_rig = np.asarray(T_footprint_optical)[:3, :3] @ tf.R_BODY_OPTICAL.T
    roll, pitch, _ = quat.rpy_from_matrix(R_footprint_rig)
    return roll, pitch


class CameraPoseEstimator:
    """Per-image camera pose from the mount calibration, the body geometry and the IMU."""

    def __init__(self, calibration: EstimatedCameraCalibration, mount: CameraMount,
                 body: BodyGeometry, baseline: float, cfg: FusionConfig | None = None,
                 side: str = "left"):
        if type(calibration) is not EstimatedCameraCalibration:
            raise TypeError("calibration must be an EstimatedCameraCalibration")
        self.cfg = cfg or FusionConfig()
        self.calibration, self.mount, self.body = calibration, mount, body
        self._T_body_optical = mount.T_body_rig(body) @ mount_geometry.T_rig_optical(baseline, side)
        self.filter = TiltEstimator(self.cfg.filter, body.T_body_imu[:3, :3])
        self.history = OrientationHistory(self.cfg.synchronisation)
        self._correction = (0.0, 0.0)           # roll, pitch added to the mount, radians
        self.imu_samples = 0

    # ------------------------------------------------------------------ inputs
    def add_imu(self, sample: ImuSample) -> None:
        """Hand over one IMU sample (in the order of arrival)."""
        self.imu_samples += 1
        estimate = self.filter.update(sample)
        if estimate is not None:
            self.history.add(estimate)

    def set_mount_correction(self, roll: float, pitch: float) -> None:
        """A small correction of the mount's roll and pitch (radians, about the camera's own
        axes), found by a calibration monitor. Zero by default."""
        self._correction = (float(roll), float(pitch))

    @property
    def mount_correction(self) -> tuple[float, float]:
        return self._correction

    # ------------------------------------------------------------------ output
    def nominal_pose(self) -> EstimatedCameraPose:
        """The camera pose of an upright, level body: what Phase 3 assumed in every frame."""
        return self._pose(0.0, 0.0)

    def _pose(self, roll: float, pitch: float) -> EstimatedCameraPose:
        transform = mount_geometry.T_footprint_body(self.body.standing_height, roll, pitch) \
            @ self._T_body_optical
        pose = EstimatedCameraPose(transform)
        if self._correction != (0.0, 0.0):
            pose = pose.perturbed(d_roll=self._correction[0], d_pitch=self._correction[1])
        return pose

    def estimate(self, image_timestamp: float) -> CameraPoseEstimate:
        """The camera pose the robot believes it had at ``image_timestamp``."""
        cfg = self.cfg
        if cfg.mode == "fixed":
            return CameraPoseEstimate(self.nominal_pose(), self.calibration, 0.0, 0.0,
                                      math.nan, "nominal")
        tilt = self.history.at(image_timestamp) if cfg.sync == "interpolate" \
            else self.history.latest(image_timestamp)
        usable = tilt.method != NONE and math.isfinite(tilt.tilt_std) \
            and tilt.tilt_std <= math.radians(cfg.max_tilt_std_deg)
        if not usable:
            # No orientation worth using (start-up, or a long outage): fall back to the level
            # body, and say how little is known when uncertainty is being reported.
            calibration = self.calibration
            if cfg.use_uncertainty:
                calibration = self._with_tilt_std(math.radians(cfg.max_tilt_std_deg), True)
            return CameraPoseEstimate(self.nominal_pose(), calibration, 0.0, 0.0, math.nan,
                                      "nominal_fallback", tilt)
        calibration = self.calibration
        if cfg.use_uncertainty:
            calibration = self._with_tilt_std(tilt.tilt_std, tilt.rate > cfg.moving_rate)
        source = "imu" if cfg.sync == "interpolate" else "imu_latest"
        return CameraPoseEstimate(self._pose(tilt.roll, tilt.pitch), calibration, tilt.roll,
                                  tilt.pitch, tilt.tilt_std, source, tilt)

    def _with_tilt_std(self, tilt_std: float, moving: bool) -> EstimatedCameraCalibration:
        """The calibration with this frame's orientation uncertainty: the mount's own
        (calibration) uncertainty and the body-tilt uncertainty are independent, so they add
        in quadrature."""
        base = self.calibration.uncertainty
        extra = math.degrees(tilt_std)
        height_std = math.hypot(base.height_std, self.cfg.height_std_moving) if moving \
            else base.height_std
        return self.calibration.replace(uncertainty=CalibrationUncertainty(
            pixel_std=base.pixel_std, height_std=height_std,
            pitch_std_deg=math.hypot(base.pitch_std_deg, extra),
            roll_std_deg=math.hypot(base.roll_std_deg, extra),
            yaw_std_deg=base.yaw_std_deg, focal_rel_std=base.focal_rel_std))
