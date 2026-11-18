"""One benchmark scenario, run one frame at a time for the viewer.

`LiveSession` does what `vision_benchmark.runner.run_vision_scenario` does for its `vision`
arm, with the same random streams, the same initial guess and the same filter seed, so a
session reproduces the benchmark's vision arm exactly (tests/test_viewer.py checks it bit for
bit). The difference is the shape: the benchmark is a loop that returns statistics, a session
hands back one `ViewerFrame` per `step()` and can be paused, stepped and reset.

GROUND TRUTH. Each step has three parts that do not share variables:

    _simulate   the simulator's side. Knows the true pose and the true camera. Returns a
                `_Sensors` packet: the image, the timestamp, the odometry increment and the
                head angles the robot commanded. The true pose stays in a second return value.
    _localize   the robot's side. Receives the `_Sensors` packet and nothing else, and calls
                `VisionLocalizationPipeline.step` (which rejects every other type).
    _observe    the viewer's side. Puts the pipeline's output and, for display only, the true
                pose into a `ViewerFrame`.

``with_truth=False`` leaves the true pose out of the frames altogether; the estimates are
identical either way (also tested).

The MuJoCo renderer belongs to the thread that created it: construct, step and close a
session in the same thread.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np

from robosoccer_perception.datasets import randomization
from robosoccer_perception.evaluation import metrics as metrics_module
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ground_to_pixels)
from robosoccer_perception.geometry.landmarks import build_landmark_map
from robosoccer_perception.localization.estimation import mirror_pose
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.pipeline.vision_localization import (
    VisionLocalizationOutput, VisionLocalizationPipeline)
from robosoccer_perception.simulation.odometry import OdometryNoiseConfig, SyntheticOdometry
from robosoccer_perception.simulation.scene_objects import DEFAULT_ROBOT_APPEARANCE
from robosoccer_perception.viewer.frame import TruthView, ViewerFrame, landmark_code
from robosoccer_perception.vision_benchmark import runner
from robosoccer_perception.vision_benchmark.camera_truth import TrueCameraModel
from robosoccer_perception.vision_benchmark.corruption import CorruptedVision
from robosoccer_perception.vision_benchmark.scenarios import VisionScenario
from robosoccer_perception.vision_benchmark.sim_camera import SimulatedCamera

CIRCLE_OUTLINE_POINTS = 64
PIXEL_LIMIT = 1.0e4                     # projections further out than this are not drawable


@dataclasses.dataclass(frozen=True)
class _Sensors:
    """Everything that crosses from the simulator to the robot in one step."""
    image: np.ndarray
    timestamp: float
    odometry: OdometryDelta | None
    commanded_pan: float
    commanded_pitch: float | None


class LiveSession:
    """A scenario and seed, advanced by `step()`."""

    def __init__(self, scenario: VisionScenario, seed: int,
                 resources: runner.BenchmarkResources, *, max_particles: int = 600,
                 with_truth: bool = True, max_steps: int | None = None):
        self.scenario, self.seed, self.resources = scenario, int(seed), resources
        self.max_particles, self.with_truth = int(max_particles), bool(with_truth)
        self.field = FieldGeometry.load(scenario.field)
        self.landmarks = build_landmark_map(self.field)
        self.camera_cfg = resources.camera_config()
        self.trajectory = runner.build_trajectory(scenario, self.seed)
        self.total_steps = len(self.trajectory) if max_steps is None \
            else min(len(self.trajectory), int(max_steps))
        self.dt = float(self.trajectory.dt)

        # The scene's look is drawn once per (scenario, seed), as in the benchmark.
        scene_rng = runner._streams(scenario.id, self.seed)[7]
        scene_overrides: dict[str, Any] = {}
        photometric = None
        if scenario.randomization:
            preset = randomization.load_preset(scenario.randomization)
            scene_overrides = randomization.sample_scene_overrides(
                scene_rng, preset, self.field, len(scenario.robots))
            photometric = preset.get("image")
        elif scenario.robots:
            scene_overrides = {"robots": [dict(DEFAULT_ROBOT_APPEARANCE)
                                          for _ in scenario.robots]}
        self._sim = SimulatedCamera(self.field, self.camera_cfg, scene_overrides, False,
                                    photometric)
        self._closed = False
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Back to step 0 with fresh random streams: the run that follows is the same run."""
        scenario, resources = self.scenario, self.resources
        (_trajectory, _oracle, odometry_rng, init_rng, filter_rng, camera_rng, corruption_rng,
         _scene, self._photo_rng) = runner._streams(scenario.id, self.seed)

        # ---- the simulator's side
        self._true_camera = TrueCameraModel(self.camera_cfg, scenario.mount_error,
                                            scenario.jitter, camera_rng)
        self._odometry = SyntheticOdometry(
            OdometryNoiseConfig().scaled(scenario.odometry_scale).replace(
                **scenario.odometry_overrides), odometry_rng)

        # ---- the robot's side
        self._calibration = EstimatedCameraCalibration.from_camera_config(
            self.camera_cfg, uncertainty=resources.noise.calibration)
        vision = resources.vision
        if scenario.corruption is not None:
            vision = CorruptedVision(vision, scenario.corruption, corruption_rng)
        self._pipeline = VisionLocalizationPipeline(
            vision, self.field, extraction=resources.extraction, noise=resources.noise,
            filter_config=resources.particle_filter_config(image_fed=True),
            landmarks=self.landmarks, seed=int(filter_rng.integers(2 ** 31)))

        # ---- initialisation: the only place a field-frame pose is given to the filter,
        # by the same rule as the benchmark (a noisy guess, a half, or nothing).
        start = self.trajectory.poses[0]
        position_std = scenario.init_position_std
        yaw_std = math.radians(scenario.init_yaw_std_deg)
        guess_error = init_rng.normal(size=3) * [position_std, position_std, yaw_std]
        half_x, _ = self.field.half_extent
        if scenario.init == "gaussian":
            self._pipeline.init_gaussian(start + guess_error, position_std, yaw_std)
        elif scenario.init == "mirrored":
            self._pipeline.init_gaussian(mirror_pose(start) + guess_error, position_std,
                                         yaw_std)
        elif scenario.init == "own_half":
            self._pipeline.init_uniform(
                x_range=(-half_x, 0.0) if start[0] < 0 else (0.0, half_x))
        else:
            self._pipeline.init_uniform()

        self._step = 0
        self._last_call: float | None = None
        self._fps = 0.0

    def close(self) -> None:
        if not self._closed:
            self._sim.close()
            self._closed = True

    def __enter__(self) -> "LiveSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def next_step(self) -> int:
        return self._step

    @property
    def finished(self) -> bool:
        return self._step >= self.total_steps

    def meta(self) -> dict[str, Any]:
        """What a recording needs to be replayed without the simulator."""
        vision = self.resources.vision
        describe = getattr(vision, "describe", None)
        return {
            "scenario": self.scenario.id, "title": self.scenario.title,
            "description": self.scenario.description, "expectation": self.scenario.expectation,
            "seed": self.seed, "field": self.scenario.field, "dt": self.dt,
            "steps": self.total_steps, "resolution": list(self.resources.resolution),
            "kidnap_steps": [int(s) for s in self.trajectory.kidnap_steps],
            "models": describe() if callable(describe) else None,
            "device": str(getattr(vision, "device", "cpu")),
            "has_truth": self.with_truth}

    # ------------------------------------------------------------------ one step
    def step(self) -> ViewerFrame | None:
        """Advance by one frame. None once the trajectory is over."""
        if self._closed:
            raise RuntimeError("the session is closed")
        if self.finished:
            return None
        step = self._step
        tic = time.perf_counter()
        sensors, true_pose, kidnapped = self._simulate(step)
        render_ms = (time.perf_counter() - tic) * 1000.0
        believed_camera, output = self._localize(sensors)
        frame = self._observe(step, sensors, believed_camera, output, render_ms, true_pose,
                              kidnapped)
        self._step += 1
        return frame

    def _simulate(self, step: int) -> tuple[_Sensors, np.ndarray, bool]:
        """THE SIMULATOR. Returns what the robot senses, and separately the true pose."""
        trajectory, scenario = self.trajectory, self.scenario
        timestamp = step * trajectory.dt
        current = trajectory.poses[step]
        pan = 0.0 if trajectory.pan is None else float(trajectory.pan[step])
        pitch = None if trajectory.pitch is None else float(trajectory.pitch[step])
        camera_truth = self._true_camera.sample(pan, pitch)
        image = self._sim.capture(current, camera_truth, scenario.ball_xy, scenario.robots,
                                  self._photo_rng)
        kidnapped = step in trajectory.kidnap_steps
        delta = None
        if step > 0:
            delta = self._odometry.measure(trajectory.poses[step - 1], current, timestamp,
                                           trajectory.dt, kidnapped=kidnapped)
        return _Sensors(image, timestamp, delta, pan, pitch), current.copy(), kidnapped

    def _localize(self, sensors: _Sensors
                  ) -> tuple[EstimatedCameraPose, VisionLocalizationOutput]:
        """THE ROBOT. Sees the sensor packet, its own calibration and nothing else."""
        believed_camera = self._calibration.pose(pan=sensors.commanded_pan,
                                                 pitch=sensors.commanded_pitch)
        output = self._pipeline.step(sensors.image, sensors.timestamp, sensors.odometry,
                                     self._calibration, believed_camera)
        return believed_camera, output

    def _observe(self, step: int, sensors: _Sensors, camera: EstimatedCameraPose,
                 output: VisionLocalizationOutput, render_ms: float, true_pose: np.ndarray,
                 kidnapped: bool) -> ViewerFrame:
        """THE VIEWER. Reads the pipeline's output; nothing here flows back into it."""
        self._fps, self._last_call = update_rate(self._fps, self._last_call)
        truth = None
        if self.with_truth:
            truth = truth_view(output.estimate.pose, true_pose, kidnapped)
        return make_frame(step, sensors.timestamp, sensors.image, output, self._calibration,
                          camera, self._pipeline.localizer, self.field, self.max_particles,
                          render_ms, self._fps, step > 0 and sensors.odometry is None, truth)


# --------------------------------------------------------------------------------------------
# Shared by every kind of session: turning one step's results into a `ViewerFrame`
# --------------------------------------------------------------------------------------------
def update_rate(fps: float, last_call: float | None) -> tuple[float, float]:
    """Smoothed steps per wall-clock second, and the time of this call."""
    now = time.perf_counter()
    if last_call is not None and 0.0 < now - last_call < 2.0:
        instant = 1.0 / (now - last_call)
        fps = instant if fps == 0.0 else 0.8 * fps + 0.2 * instant
    return fps, now


def truth_view(estimate_pose, true_pose, kidnapped: bool = False, **extra) -> TruthView:
    """DISPLAY ONLY: the true pose and the errors of the estimate against it."""
    position_error, yaw_error = metrics_module.pose_errors(estimate_pose, true_pose)
    mirror_position, mirror_yaw = metrics_module.mirror_errors(estimate_pose, true_pose)
    return TruthView(np.asarray(true_pose, float).copy(), float(position_error),
                     float(yaw_error), float(mirror_position), float(mirror_yaw),
                     bool(kidnapped), **extra)


def make_frame(step: int, timestamp: float, image: np.ndarray,
               output: VisionLocalizationOutput, calibration: EstimatedCameraCalibration,
               camera: EstimatedCameraPose, localizer, field: FieldGeometry, max_particles: int,
               render_ms: float, fps: float, odometry_lost: bool, truth: TruthView | None,
               fusion=None, robot_view: np.ndarray | None = None) -> ViewerFrame:
    """One `ViewerFrame` from a pipeline output. ``camera`` is the camera pose the features
    were projected with (the robot's estimate); it is used to draw them back into the image."""
    features, estimate, update = output.features, output.estimate, output.update

    def to_pixels(points) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(points):
            return np.zeros((0, 2))
        pixels, _ = ground_to_pixels(points, calibration, camera)
        pixels = np.array(pixels, dtype=float)
        pixels[~np.isfinite(pixels).all(axis=1) | (np.abs(pixels) > PIXEL_LIMIT).any(axis=1)] \
            = np.nan
        return pixels

    segments = np.array([[s.start, s.end] for s in features.segments]).reshape(-1, 2, 2)
    junctions = np.array([j.position for j in features.intersections]).reshape(-1, 2)
    centers = np.array([c.center for c in features.circles]).reshape(-1, 2)
    radius = float(features.circles[0].radius) if features.circles \
        else float(field.center_circle_radius)
    angles = np.linspace(0.0, 2.0 * math.pi, CIRCLE_OUTLINE_POINTS)
    ring = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    outlines = np.array([to_pixels(center + ring) for center in centers]).reshape(
        -1, CIRCLE_OUTLINE_POINTS, 2)
    posts = np.array([p.position for p in features.goal_posts]).reshape(-1, 2)
    observed = output.observations

    count = localizer.num_particles
    chosen = np.unique(np.linspace(0, count - 1, min(count, max_particles)).astype(int))
    weights = np.asarray(localizer.weights, dtype=float)[chosen]
    weights = weights / max(float(weights.sum()), 1e-300)

    return ViewerFrame(
        step=step, timestamp=float(timestamp),
        image=image, mask=np.asarray(output.vision.segmentation.mask, np.uint8),
        detections=tuple(output.vision.detections),
        line_pixels=np.asarray(features.line_points.pixels, float).reshape(-1, 2),
        line_points=np.asarray(features.line_points.points, float).reshape(-1, 2),
        line_std=np.asarray(features.line_points.std, float).reshape(-1),
        segment_pixels=to_pixels(segments.reshape(-1, 2)).reshape(-1, 2, 2),
        segments=segments,
        junction_pixels=to_pixels(junctions), junctions=junctions,
        junction_codes=np.array([landmark_code(j.type) for j in features.intersections],
                                dtype=int),
        circle_centers=centers, circle_center_pixels=to_pixels(centers),
        circle_valid=np.array([c.valid for c in features.circles], dtype=bool),
        circle_radius=radius, circle_outlines=outlines,
        post_pixels=np.array([p.pixel for p in features.goal_posts]).reshape(-1, 2),
        posts=posts,
        post_range_valid=np.array([p.range_valid for p in features.goal_posts], dtype=bool),
        landmarks=np.array([lm.position for lm in observed.landmarks]).reshape(-1, 2),
        landmark_codes=np.array([landmark_code(lm.type) for lm in observed.landmarks],
                                dtype=int),
        observed_line_points=len(observed.line_points),
        pose=np.asarray(estimate.pose, float).copy(),
        covariance=np.asarray(estimate.covariance, float).copy(),
        position_std=float(estimate.position_std),
        yaw_std_deg=math.degrees(float(estimate.yaw_std)),
        confidence=float(estimate.confidence), mode_weight=float(estimate.mode_weight),
        converged=bool(estimate.converged), ambiguous=bool(estimate.symmetric_ambiguity),
        modes=np.array([[*m.pose, m.weight] for m in estimate.modes]).reshape(-1, 4),
        ess_fraction=float(update.effective_sample_size) / count,
        resampled=bool(update.resampled), injected=int(update.injected),
        particles=np.asarray(localizer.particles, np.float32)[chosen].copy(),
        particle_weights=weights.astype(np.float32), num_particles=int(count),
        odometry_lost=bool(odometry_lost),
        timing_ms={**{k: float(v) for k, v in output.timing_ms.items()},
                   "render": float(render_ms)},
        sim_fps=float(fps), truth=truth, fusion=fusion, robot_view=robot_view)
