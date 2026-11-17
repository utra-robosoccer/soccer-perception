"""The complete offline camera-to-pose pipeline. See docs/vision_localization.md.

    RGB image ─► VisionPipeline (YOLO + U-Net) ─► VisionResult
              ─► GeometricFeatureExtractor      ─► FeatureSet        (robot frame, metres)
              ─► build_observation_frame        ─► ObservationFrame  (same type as the oracle's)
    odometry  ─► ParticleFilter.predict / update / estimate ─► PoseEstimate

What crosses into `VisionLocalizationPipeline.step`, and nothing else:

* the RGB image;
* the timestamp;
* an `OdometryDelta` (or None when the message was lost);
* the robot's `EstimatedCameraCalibration` and its `EstimatedCameraPose` for this frame,
  optionally with the IMU's roll and pitch.

The static field description is given once, at construction. A pose in the field frame enters
only through `init_gaussian`, `init_uniform` and `resolve_symmetry`, exactly as in Phase 1.

This module does not import MuJoCo, the simulator package or the dataset reader, and `step`
refuses arguments of any other type (tests/test_ground_truth_isolation.py). Ground truth is
handled by a separate evaluator (`vision_benchmark/`).
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, require_estimate)
from robosoccer_perception.geometry.landmarks import LandmarkMap, build_landmark_map
from robosoccer_perception.inference.types import VisionResult
from robosoccer_perception.localization.estimation import PoseEstimate
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.localization.particle_filter import (
    ParticleFilter, ParticleFilterConfig, UpdateInfo)
from robosoccer_perception.observations.types import ObservationFrame
from robosoccer_perception.perception.config import (
    FeatureExtractionConfig, MeasurementNoiseConfig)
from robosoccer_perception.perception.extractor import GeometricFeatureExtractor
from robosoccer_perception.perception.observation_builder import (
    ObservationSelection, build_observation_frame)
from robosoccer_perception.perception.types import FeatureSet


@dataclasses.dataclass(frozen=True)
class VisionLocalizationInput:
    """Everything one localization step is allowed to know."""
    image: np.ndarray                               # (H, W, 3) uint8 RGB
    timestamp: float                                # seconds
    odometry_delta: OdometryDelta | None            # motion since the previous step
    camera_calibration: EstimatedCameraCalibration
    estimated_camera_pose: EstimatedCameraPose
    imu_roll: float | None = None                   # radians: torso tilt from the IMU, if any
    imu_pitch: float | None = None

    def validate(self) -> None:
        image = self.image
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 \
                or image.dtype != np.uint8:
            raise TypeError("image must be an (H, W, 3) uint8 RGB array")
        _check_common(self.timestamp, self.odometry_delta, self.camera_calibration,
                      self.estimated_camera_pose)
        for value in (self.imu_roll, self.imu_pitch):
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError("imu_roll and imu_pitch must be numbers (radians) or None")


def _check_common(timestamp, odometry_delta, calibration, pose) -> None:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float, np.floating)):
        raise TypeError("timestamp must be a number of seconds")
    if odometry_delta is not None and type(odometry_delta) is not OdometryDelta:
        raise TypeError("odometry_delta must be an OdometryDelta or None, got "
                        f"{type(odometry_delta).__name__}")
    require_estimate(calibration, pose)


@dataclasses.dataclass(frozen=True)
class VisionLocalizationOutput:
    estimate: PoseEstimate              # pose, covariance, convergence and ambiguity flags
    update: UpdateInfo                  # what the measurement update did
    observations: ObservationFrame      # what the localizer was given
    features: FeatureSet                # everything extracted, with diagnostics
    vision: VisionResult                # the raw model predictions
    timing_ms: dict[str, float]         # detector, segmenter, geometry, filter, total

    @property
    def pose(self) -> np.ndarray:
        """``[x, y, yaw]`` in the field frame."""
        return self.estimate.pose


class VisionLocalizationPipeline:
    """Localization from camera images: the vision models, geometric extraction and the
    Phase 1 particle filter behind one ``step`` call.

    ``vision`` is anything with ``run(image_rgb) -> VisionResult``, normally a
    `inference.pipeline.VisionPipeline`.
    """

    def __init__(self, vision, field: FieldGeometry, *,
                 extraction: FeatureExtractionConfig | None = None,
                 noise: MeasurementNoiseConfig | None = None,
                 selection: ObservationSelection | None = None,
                 filter_config: ParticleFilterConfig | None = None,
                 landmarks: LandmarkMap | None = None, seed: int | None = 0):
        if not isinstance(field, FieldGeometry):
            raise TypeError("field must be a FieldGeometry (the static map)")
        self.vision = vision
        self.field = field
        self.extractor = GeometricFeatureExtractor(field, extraction, noise)
        self.selection = selection or self.extractor.cfg.observations
        self.localizer = ParticleFilter(field, landmarks or build_landmark_map(field),
                                        filter_config, seed=seed)
        self.steps = 0

    @classmethod
    def from_checkpoints(cls, detector_weights, segmenter_checkpoint, field: str = "s_field",
                         device: str = "auto", conf: float = 0.25, imgsz: int = 640,
                         half: bool = False, **kwargs: Any) -> "VisionLocalizationPipeline":
        """Load the trained Phase 2 models and the configured extraction and filter settings."""
        from robosoccer_perception.inference.pipeline import VisionPipeline

        vision = VisionPipeline.from_checkpoints(detector_weights, segmenter_checkpoint,
                                                 device=device, conf=conf, imgsz=imgsz, half=half)
        kwargs.setdefault("extraction", FeatureExtractionConfig.load())
        kwargs.setdefault("noise", MeasurementNoiseConfig.load())
        return cls(vision, FieldGeometry.load(field), **kwargs)

    # ------------------------------------------------------------------ priors (as in Phase 1)
    def init_gaussian(self, pose, position_std: float | None = None,
                      yaw_std: float | None = None) -> None:
        """Start from an approximately known pose ``[x, y, yaw]``."""
        self.localizer.init_gaussian(pose, position_std, yaw_std)
        self.steps = 0

    def init_uniform(self, x_range=None, y_range=None, yaw_range=None) -> None:
        """Start with no idea of the pose, or with a partial prior such as "our own half"."""
        self.localizer.init_uniform(x_range, y_range, yaw_range)
        self.steps = 0

    def resolve_symmetry(self, reference_xy) -> None:
        """Apply OUTSIDE information about which half the robot is in (game state, a referee
        command). Images of a symmetric field can never supply it."""
        self.localizer.resolve_symmetry(reference_xy)

    # ------------------------------------------------------------------ one cycle
    def step(self, image: np.ndarray, timestamp: float, odometry_delta: OdometryDelta | None,
             camera_calibration: EstimatedCameraCalibration,
             estimated_camera_pose: EstimatedCameraPose, *, imu_roll: float | None = None,
             imu_pitch: float | None = None) -> VisionLocalizationOutput:
        """Process one camera frame and return the pose estimate with diagnostics.

        ``odometry_delta`` is the motion since the previous step; None means the message was
        lost (on the very first step it means "no motion yet").
        """
        return self.step_input(VisionLocalizationInput(
            image, timestamp, odometry_delta, camera_calibration, estimated_camera_pose,
            imu_roll, imu_pitch))

    def step_input(self, data: VisionLocalizationInput) -> VisionLocalizationOutput:
        if type(data) is not VisionLocalizationInput:
            raise TypeError("expected a VisionLocalizationInput")
        data.validate()
        started = time.perf_counter()
        vision = self.vision.run(data.image)
        pose = data.estimated_camera_pose
        if data.imu_roll is not None or data.imu_pitch is not None:
            pose = pose.with_base_tilt(data.imu_roll or 0.0, data.imu_pitch or 0.0)
        return self._localize(vision, data.timestamp, data.odometry_delta,
                              data.camera_calibration, pose, started)

    def step_with_predictions(self, vision: VisionResult, timestamp: float,
                              odometry_delta: OdometryDelta | None,
                              camera_calibration: EstimatedCameraCalibration,
                              estimated_camera_pose: EstimatedCameraPose
                              ) -> VisionLocalizationOutput:
        """The same cycle from model predictions that already exist: saved results, or one
        inference shared by several localizers in a sensitivity sweep.

        Whoever calls this is responsible for where ``vision`` came from. The benchmark's
        "perfect vision" diagnostic passes rendered labels here, under that name; the claimed
        vision pipeline always goes through `step`.
        """
        if type(vision) is not VisionResult:
            raise TypeError("vision must be a VisionResult")
        _check_common(timestamp, odometry_delta, camera_calibration, estimated_camera_pose)
        return self._localize(vision, timestamp, odometry_delta, camera_calibration,
                              estimated_camera_pose, time.perf_counter())

    def _localize(self, vision, timestamp, odometry_delta, calibration, pose, started
                  ) -> VisionLocalizationOutput:
        extract_started = time.perf_counter()
        features = self.extractor.extract(vision, calibration, pose)
        frame = build_observation_frame(features, timestamp, self.selection)
        filter_started = time.perf_counter()
        if self.steps > 0 or odometry_delta is not None:
            self.localizer.predict(odometry_delta)
        update = self.localizer.update(frame)
        estimate = self.localizer.estimate()
        finished = time.perf_counter()
        self.steps += 1
        timing = {"detector": float(vision.timing_ms.get("detector", 0.0)),
                  "segmenter": float(vision.timing_ms.get("segmenter", 0.0)),
                  "geometry": (filter_started - extract_started) * 1000.0,
                  "filter": (finished - filter_started) * 1000.0,
                  "total": (finished - started) * 1000.0}
        return VisionLocalizationOutput(estimate, update, frame, features, vision, timing)
