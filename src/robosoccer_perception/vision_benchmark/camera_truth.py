"""The TRUE camera of the simulator, kept apart from what the robot believes.

Two worlds meet in the Phase 3 benchmark and must not be confused:

* the robot's belief: `geometry.ground_projection.EstimatedCameraCalibration` and
  `EstimatedCameraPose`, built from the nominal mount and the commanded head angles;
* the simulator's truth: `SimulationCameraTruth` below, which is what the image is rendered
  with. It differs from the belief by a hidden `MountError` (the camera is not mounted where
  the drawing says) and by `HeadJitter` (the head shakes a little in every frame).

`SimulationCameraTruth` is a different type with different field names. The localization
pipeline rejects it (`require_estimate`), and nothing in this module produces an estimate from
it. It is used for exactly two things: placing the camera in the MuJoCo scene, and scoring
extracted features against ground truth.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from robosoccer_perception.geometry.camera import (
    CameraIntrinsics, RigState, StereoCameraConfig, StereoCameraModel)


@dataclasses.dataclass(frozen=True)
class MountError:
    """Where the camera really is, minus where the robot believes it is. Hidden from the robot."""
    height: float = 0.0             # metres
    pitch_deg: float = 0.0          # positive: really looks further down than believed
    roll_deg: float = 0.0
    pan_deg: float = 0.0

    def describe(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HeadJitter:
    """Frame-to-frame shake of the real head that joint encoders do not report (1 sigma)."""
    pitch_std_deg: float = 0.25
    roll_std_deg: float = 0.25
    pan_std_deg: float = 0.0
    height_std: float = 0.0

    @classmethod
    def none(cls) -> "HeadJitter":
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class SimulationCameraTruth:
    """The true state of the simulated camera in one frame. For rendering and scoring only."""
    true_rig: RigState
    true_intrinsics: CameraIntrinsics
    baseline: float

    def true_T_base_camera(self, side: str = "left") -> np.ndarray:
        """True ``T_base_<side>_optical``. Never give this to the localizer."""
        config = StereoCameraConfig("truth", True, self.baseline, 1.0, self.true_intrinsics,
                                    self.true_rig, 0.05, 60.0)
        return StereoCameraModel(config).T_base_camera(side, self.true_rig)


class TrueCameraModel:
    """Generates the true camera state from what the robot COMMANDED.

    ``commanded`` head angles are things the robot knows (it set them); the mount error and
    the jitter are added here, on the simulator's side, and are never reported back.
    """

    def __init__(self, camera_cfg: StereoCameraConfig, mount_error: MountError | None = None,
                 jitter: HeadJitter | None = None, rng: np.random.Generator | None = None):
        self.camera_cfg = camera_cfg
        self.mount_error = mount_error or MountError()
        self.jitter = jitter or HeadJitter()
        self.rng = rng or np.random.default_rng(0)

    def sample(self, commanded_pan: float = 0.0, commanded_pitch: float | None = None
               ) -> SimulationCameraTruth:
        mount, error, jitter = self.camera_cfg.mount, self.mount_error, self.jitter
        noise = self.rng.normal(size=4)
        pitch = mount.pitch if commanded_pitch is None else commanded_pitch
        rig = mount.replace(
            height=mount.height + error.height + noise[0] * jitter.height_std,
            pitch=pitch + math.radians(error.pitch_deg + noise[1] * jitter.pitch_std_deg),
            roll=mount.roll + math.radians(error.roll_deg + noise[2] * jitter.roll_std_deg),
            pan=commanded_pan + math.radians(error.pan_deg + noise[3] * jitter.pan_std_deg))
        return SimulationCameraTruth(rig, self.camera_cfg.intrinsics, self.camera_cfg.baseline)
