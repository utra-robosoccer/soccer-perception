"""Running one Phase 3 scenario: the simulator on one side, localizers on the other.

    true trajectory ──► SyntheticOdometry ──► OdometryDelta ──────────────────────────────┐
          │                                                                                ▼
          ├─► TrueCameraModel ─► MuJoCo render ─► RGB image ─► VisionLocalizationPipeline.step
          │        (hidden)                                        ▲                │
          │                       commanded head angles ─► EstimatedCameraPose      │
          │                                                                         ▼
          └───────────────────► evaluation (pose errors, feature scores) ◄──── estimate, features

One pass over the trajectory renders each frame once and feeds several ARMS. Every arm is an
independent localizer with the same trajectory, the same odometry stream, the same initial
guess and the same filter seed; arms differ only in where their observations come from:

    vision          the trained networks on the rendered image (the Phase 3 pipeline)
    oracle          Phase 1's oracle observations from simulator geometry   (reference)
    odometry        no observations at all: dead reckoning                  (reference)
    perfect_vision  rendered LABELS through the same geometric extraction   (diagnostic)

A vision arm may also be given a deliberately wrong camera estimate (`EstimateError`), which
is how the calibration sweeps run on a single inference per frame.

Ground truth is used here for three things only: driving the simulator, generating the
reference arms' inputs, and scoring. The vision arm receives the image, the odometry delta
and the robot's own camera estimate; tests/test_ground_truth_isolation.py records and replays
exactly that.
"""
from __future__ import annotations

import dataclasses
import math
import time
import zlib
from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.datasets import randomization
from robosoccer_perception.evaluation import metrics as metrics_module
from robosoccer_perception.evaluation.benchmark import RunResult, Snapshot
from robosoccer_perception.geometry.camera import StereoCameraConfig, StereoCameraModel
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import (
    CalibrationUncertainty, EstimatedCameraCalibration, EstimatedCameraPose)
from robosoccer_perception.geometry.landmarks import build_landmark_map
from robosoccer_perception.inference import profiling
from robosoccer_perception.inference.types import VisionResult
from robosoccer_perception.localization.estimation import mirror_pose
from robosoccer_perception.localization.particle_filter import (
    ParticleFilter, ParticleFilterConfig, UpdateInfo)
from robosoccer_perception.observations.noise import OracleConfig
from robosoccer_perception.observations.oracle import OracleObservationGenerator
from robosoccer_perception.observations.types import ObservationFrame
from robosoccer_perception.perception.config import (
    FeatureExtractionConfig, MeasurementNoiseConfig)
from robosoccer_perception.perception.observation_builder import ObservationSelection
from robosoccer_perception.perception.types import FeatureSet
from robosoccer_perception.pipeline.vision_localization import VisionLocalizationPipeline
from robosoccer_perception.simulation.odometry import OdometryNoiseConfig, SyntheticOdometry
from robosoccer_perception.simulation.scene_objects import DEFAULT_ROBOT_APPEARANCE
from robosoccer_perception.vision_benchmark.camera_truth import TrueCameraModel
from robosoccer_perception.vision_benchmark.corruption import CorruptedVision
from robosoccer_perception.vision_benchmark.feature_metrics import (
    FeatureScorer, FeatureStatistics, FrameTruth)
from robosoccer_perception.vision_benchmark.scenarios import VisionScenario, vary_trajectory
from robosoccer_perception.vision_benchmark.sim_camera import SimulatedCamera

IMAGE_SOURCES = ("vision", "perfect_vision")
ROBOT_OCCLUDER_RADIUS = 0.15        # metres: a standing placeholder robot as a cylinder
TIMING_KEYS = ("detector", "segmenter", "geometry", "filter", "total")


@dataclasses.dataclass(frozen=True)
class EstimateError:
    """What the robot gets wrong about its camera in one arm: ESTIMATE MINUS TRUTH.

    Applied to the robot's estimate, never to the simulator. A scenario's `MountError` is the
    other way round (truth minus what the robot believes), so a mount pitch error of +2 deg
    and an estimate error of -2 deg describe the same mismatch.
    """
    height: float = 0.0             # metres
    pitch_deg: float = 0.0          # positive: the robot believes it looks further down
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    focal_scale: float = 1.0
    cx_shift: float = 0.0           # pixels
    cy_shift: float = 0.0

    @property
    def is_zero(self) -> bool:
        return self == EstimateError()

    def apply(self, calibration: EstimatedCameraCalibration, pose: EstimatedCameraPose
              ) -> tuple[EstimatedCameraCalibration, EstimatedCameraPose]:
        if self.is_zero:
            return calibration, pose
        return (calibration.with_intrinsics_error(self.focal_scale, self.cx_shift, self.cy_shift),
                pose.perturbed(self.height, math.radians(self.pitch_deg),
                               math.radians(self.roll_deg), math.radians(self.yaw_deg)))


@dataclasses.dataclass(frozen=True)
class Arm:
    """One localizer in a scenario run."""
    name: str
    source: str                                     # vision | oracle | odometry | perfect_vision
    estimate_error: EstimateError = dataclasses.field(default_factory=EstimateError)
    selection: ObservationSelection | None = None   # which features reach the filter
    description: str = ""
    # What this arm BELIEVES about its calibration error (1 sigma); None = the configured one.
    believed_uncertainty: CalibrationUncertainty | None = None


DEFAULT_ARMS = (
    Arm("vision", "vision", description="trained YOLO11n + U-Net predictions from the image"),
    Arm("oracle", "oracle", description="Phase 1 oracle observations (simulator geometry)"),
    Arm("odometry", "odometry", description="dead reckoning, no observations"),
    Arm("perfect_vision", "perfect_vision",
        description="rendered labels through the same geometric extraction (diagnostic)"),
)


@dataclasses.dataclass
class BenchmarkResources:
    """Everything that is loaded once and shared by all runs."""
    vision: Any                                     # has run(image_rgb) -> VisionResult
    camera: str = "zed_mini_sim"
    resolution: tuple[int, int] = (640, 360)
    extraction: FeatureExtractionConfig = dataclasses.field(
        default_factory=FeatureExtractionConfig)
    noise: MeasurementNoiseConfig = dataclasses.field(default_factory=MeasurementNoiseConfig)
    filter_config: str = "particle_filter"
    filter_overrides: dict[str, Any] | None = None          # every arm
    vision_filter_overrides: dict[str, Any] | None = None   # image-fed arms only, on top

    def camera_config(self) -> StereoCameraConfig:
        return StereoCameraConfig.load(self.camera).with_resolution(*self.resolution)

    def particle_filter_config(self, image_fed: bool) -> ParticleFilterConfig:
        overrides = config.deep_update(self.filter_overrides or {},
                                       self.vision_filter_overrides if image_fed else None)
        return ParticleFilterConfig.load(self.filter_config, overrides)


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """One frame of the primary vision arm, kept for figures."""
    step: int
    image: np.ndarray
    vision: VisionResult
    features: FeatureSet
    observations: ObservationFrame
    truth_pose: np.ndarray
    estimate_pose: np.ndarray
    particles: np.ndarray
    weights: np.ndarray
    calibration: EstimatedCameraCalibration
    camera_pose: EstimatedCameraPose


@dataclasses.dataclass
class ArmRun:
    arm: Arm
    result: RunResult
    timing_ms: dict[str, np.ndarray] | None = None      # the primary vision arm only
    features: FeatureStatistics | None = None           # image-fed arms only


@dataclasses.dataclass
class ScenarioRun:
    scenario_id: str
    seed: int
    arms: dict[str, ArmRun]
    frames: dict[int, FrameRecord]
    gpu_memory_mb: dict[str, float] | None
    render_ms_mean: float


def _streams(scenario_id: str, seed: int, count: int = 9) -> list[np.random.Generator]:
    root = np.random.SeedSequence([int(seed), zlib.crc32(scenario_id.encode())])
    return [np.random.default_rng(child) for child in root.spawn(count)]


def build_trajectory(scenario: VisionScenario, seed: int):
    """The true trajectory `run_vision_scenario` uses for this scenario and seed."""
    field = FieldGeometry.load(scenario.field)
    rng = _streams(scenario.id, seed)[0]
    trajectory = vary_trajectory(scenario.build_trajectory(field, rng), rng,
                                 scenario.start_shift, scenario.start_turn_deg)
    if scenario.commanded_pitch_deg is not None:
        trajectory = trajectory.with_pitch(math.radians(scenario.commanded_pitch_deg))
    return trajectory


class _Series:
    """Per-step records of one arm, turned into a Phase 1 `RunResult` at the end."""
    NAMES = ("position_std", "yaw_std_deg", "mode_weight", "confidence", "ess_fraction",
             "agreement", "injected", "num_landmarks", "num_line_points", "update_time_s")
    FLAGS = ("filter_converged", "symmetric_ambiguity", "resampled")

    def __init__(self, steps: int):
        self.estimate = np.zeros((steps, 3))
        self.values = {name: np.zeros(steps) for name in self.NAMES}
        self.flags = {name: np.zeros(steps, dtype=bool) for name in self.FLAGS}
        self.snapshots: dict[int, Snapshot] = {}

    def record(self, step: int, estimate, info: UpdateInfo, frame: ObservationFrame | None,
               num_particles: int, seconds: float) -> None:
        self.estimate[step] = estimate.pose
        values, flags = self.values, self.flags
        values["position_std"][step] = estimate.position_std
        values["yaw_std_deg"][step] = math.degrees(estimate.yaw_std)
        values["mode_weight"][step] = estimate.mode_weight
        values["confidence"][step] = estimate.confidence
        values["ess_fraction"][step] = info.effective_sample_size / num_particles
        values["agreement"][step] = info.agreement
        values["injected"][step] = info.injected
        values["num_landmarks"][step] = 0 if frame is None else len(frame.landmarks)
        values["num_line_points"][step] = 0 if frame is None else len(frame.line_points)
        values["update_time_s"][step] = seconds
        flags["filter_converged"][step] = estimate.converged
        flags["symmetric_ambiguity"][step] = estimate.symmetric_ambiguity
        flags["resampled"][step] = info.resampled


def run_vision_scenario(scenario: VisionScenario, seed: int, resources: BenchmarkResources,
                        arms: tuple[Arm, ...] | None = None, *, keep_frames=(),
                        snapshot_steps=(), score_features: bool = True) -> ScenarioRun:
    """Run one scenario with one seed for every arm and return their time series.

    ``keep_frames`` lists steps whose image, predictions and features are kept (``"auto"``:
    three steps spread over the run). ``snapshot_steps`` does the same for particle sets.
    """
    arms = tuple(arms if arms is not None else
                 [arm for arm in DEFAULT_ARMS if arm.source in scenario.sources])
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    (trajectory_rng, oracle_rng, odometry_rng, init_rng, filter_rng, camera_rng, corruption_rng,
     scene_rng, photo_rng) = _streams(scenario.id, seed)
    del trajectory_rng                                  # consumed by build_trajectory
    field = FieldGeometry.load(scenario.field)
    landmarks = build_landmark_map(field)
    camera_cfg = resources.camera_config()
    trajectory = build_trajectory(scenario, seed)
    steps = len(trajectory)

    # ---- the simulator's side: true camera, scene, reference observation sources
    true_camera = TrueCameraModel(camera_cfg, scenario.mount_error, scenario.jitter, camera_rng)
    scene_overrides: dict[str, Any] = {}
    photometric = None
    if scenario.randomization:
        preset = randomization.load_preset(scenario.randomization)
        scene_overrides = randomization.sample_scene_overrides(scene_rng, preset, field,
                                                               len(scenario.robots))
        photometric = preset.get("image")
    elif scenario.robots:
        scene_overrides = {"robots": [dict(DEFAULT_ROBOT_APPEARANCE) for _ in scenario.robots]}
    needs_labels = any(arm.source == "perfect_vision" for arm in arms)
    oracle = OracleObservationGenerator(
        field, landmarks, StereoCameraModel(camera_cfg),
        OracleConfig.load(scenario.oracle_profile, scenario.oracle_overrides))
    odometry = SyntheticOdometry(
        OdometryNoiseConfig().scaled(scenario.odometry_scale).replace(
            **scenario.odometry_overrides), odometry_rng)
    robot_heights = [float(look.get("height", 0.8))
                     for look in scene_overrides.get("robots", [])]
    robot_cylinders = np.array(
        [[*robot.xy, ROBOT_OCCLUDER_RADIUS, height]
         for robot, height in zip(scenario.robots, robot_heights)]).reshape(-1, 4)
    scorer = FeatureScorer(field, landmarks, camera_cfg) if score_features else None

    # ---- the robot's side: its calibration, and one localizer per arm
    calibration = EstimatedCameraCalibration.from_camera_config(
        camera_cfg, uncertainty=resources.noise.calibration)
    filter_seed = int(filter_rng.integers(2 ** 31))
    primary = next((arm.name for arm in arms if arm.source == "vision"), None)
    vision = resources.vision
    if scenario.corruption is not None:
        vision = CorruptedVision(vision, scenario.corruption, corruption_rng)
    localizers: dict[str, Any] = {}
    for arm in arms:
        image_fed = arm.source in IMAGE_SOURCES
        pf_config = resources.particle_filter_config(image_fed)
        if image_fed:
            localizers[arm.name] = VisionLocalizationPipeline(
                vision if arm.source == "vision" else None, field,
                extraction=resources.extraction, noise=resources.noise, selection=arm.selection,
                filter_config=pf_config, landmarks=landmarks, seed=filter_seed)
        else:
            localizers[arm.name] = ParticleFilter(field, landmarks, pf_config, seed=filter_seed)

    # ---- initialisation: the only place a global pose is given to any filter
    start = trajectory.poses[0]
    position_std = scenario.init_position_std
    yaw_std = math.radians(scenario.init_yaw_std_deg)
    guess_error = init_rng.normal(size=3) * [position_std, position_std, yaw_std]
    half_x, _ = field.half_extent
    for localizer in localizers.values():
        if scenario.init == "gaussian":
            localizer.init_gaussian(start + guess_error, position_std, yaw_std)
        elif scenario.init == "mirrored":
            localizer.init_gaussian(mirror_pose(start) + guess_error, position_std, yaw_std)
        elif scenario.init == "own_half":
            localizer.init_uniform(x_range=(-half_x, 0.0) if start[0] < 0 else (0.0, half_x))
        else:
            localizer.init_uniform()

    series = {arm.name: _Series(steps) for arm in arms}
    statistics = {arm.name: FeatureStatistics() for arm in arms
                  if arm.source in IMAGE_SOURCES and scorer is not None}
    timing = {key: np.zeros(steps) for key in TIMING_KEYS}
    frames: dict[int, FrameRecord] = {}
    if keep_frames == "auto":
        keep_frames = (steps // 6, steps // 2, (5 * steps) // 6)
    if snapshot_steps == "auto":
        middle = trajectory.kidnap_steps[0] + 5 if trajectory.kidnap_steps else steps // 3
        snapshot_steps = (0, min(middle, steps - 1), steps - 1)
    keep_frames = {int(s) % steps for s in keep_frames}
    snapshot_steps = {int(s) % steps for s in snapshot_steps}
    truth = np.zeros((steps, 3))
    render_seconds = 0.0
    device = getattr(resources.vision, "device", "cpu")
    profiling.reset_gpu_peak(device)
    wall_start = time.perf_counter()

    with SimulatedCamera(field, camera_cfg, scene_overrides, needs_labels, photometric) as sim:
        previous = start
        for step in range(steps):
            timestamp = step * trajectory.dt
            current = trajectory.poses[step]
            truth[step] = current
            # ---- what the robot commands, and therefore knows
            pan = 0.0 if trajectory.pan is None else float(trajectory.pan[step])
            pitch = None if trajectory.pitch is None else float(trajectory.pitch[step])
            believed_pose = calibration.pose(pan=pan, pitch=pitch)
            # ---- the simulator (ground truth stays on this side)
            camera_truth = true_camera.sample(pan, pitch)
            tic = time.perf_counter()
            image = sim.capture(current, camera_truth, scenario.ball_xy, scenario.robots,
                                photo_rng)
            render_seconds += time.perf_counter() - tic
            delta = None
            if step > 0:
                delta = odometry.measure(previous, current, timestamp, trajectory.dt,
                                         kidnapped=step in trajectory.kidnap_steps)
            previous = current
            occluders = robot_cylinders
            if scenario.ball_xy is not None:
                occluders = np.vstack([occluders, [*scenario.ball_xy, sim.ball_radius,
                                                   2.0 * sim.ball_radius]])
            frame_truth = FrameTruth(current, camera_truth, occluders)
            shared: VisionResult | None = None
            labels: VisionResult | None = None

            # The primary vision arm goes first: its inference is reused by sweep arms.
            for arm in sorted(arms, key=lambda a: a.name != primary):
                localizer, record = localizers[arm.name], series[arm.name]
                if arm.source in IMAGE_SOURCES:
                    arm_calibration, arm_pose = arm.estimate_error.apply(calibration,
                                                                         believed_pose)
                    if arm.believed_uncertainty is not None:
                        arm_calibration = arm_calibration.replace(
                            uncertainty=arm.believed_uncertainty)
                    if arm.name == primary:
                        # THE PHASE 3 PIPELINE: image, odometry and the robot's own estimate.
                        output = localizer.step(image, timestamp, delta, arm_calibration,
                                                arm_pose)
                        shared = output.vision
                        for key in TIMING_KEYS:
                            timing[key][step] = output.timing_ms[key]
                    elif arm.source == "vision":
                        output = localizer.step_with_predictions(shared, timestamp, delta,
                                                                 arm_calibration, arm_pose)
                    else:
                        labels = labels if labels is not None else sim.labels_as_vision()
                        output = localizer.step_with_predictions(labels, timestamp, delta,
                                                                 arm_calibration, arm_pose)
                    estimate, info, frame = output.estimate, output.update, output.observations
                    seconds = output.timing_ms["filter"] / 1000.0
                    particles = localizer.localizer
                    if arm.name in statistics:
                        statistics[arm.name].add(scorer.score(output.features, frame_truth))
                    if arm.name == primary and step in keep_frames:
                        frames[step] = FrameRecord(
                            step, image, output.vision, output.features, frame, current.copy(),
                            estimate.pose.copy(), particles.particles.copy(),
                            particles.weights.copy(), arm_calibration, arm_pose)
                else:
                    frame = None
                    if arm.source == "oracle":
                        frame, _ = oracle.generate(
                            current, oracle_rng, camera_truth.true_rig, timestamp,
                            ball_position=scenario.ball_xy, ball_radius=sim.ball_radius,
                            extra_occluders=robot_cylinders)
                    tic = time.perf_counter()
                    if step > 0:
                        localizer.predict(delta)
                    info = localizer.update(frame) if frame is not None else UpdateInfo(
                        localizer.effective_sample_size, False, 0, 1.0, used=False)
                    estimate = localizer.estimate()
                    seconds = time.perf_counter() - tic
                    particles = localizer
                record.record(step, estimate, info, frame, particles.num_particles, seconds)
                if step in snapshot_steps:
                    record.snapshots[step] = Snapshot(
                        step, particles.particles.copy(), particles.weights.copy(),
                        frame if frame is not None else ObservationFrame(timestamp))

    wall = time.perf_counter() - wall_start
    runs: dict[str, ArmRun] = {}
    for arm in arms:
        record = series[arm.name]
        position_error, yaw_error = metrics_module.pose_errors(record.estimate, truth)
        mirror_position, mirror_yaw = metrics_module.mirror_errors(record.estimate, truth)
        result = RunResult(
            scenario_id=scenario.id, field_name=field.name, seed=int(seed), dt=trajectory.dt,
            kidnap_steps=trajectory.kidnap_steps, time=trajectory.times, truth=truth,
            estimate=record.estimate, position_error=position_error, yaw_error_deg=yaw_error,
            mirror_position_error=mirror_position, mirror_yaw_error_deg=mirror_yaw,
            wall_time_s=wall, snapshots=record.snapshots, **record.values, **record.flags)
        runs[arm.name] = ArmRun(arm, result, timing if arm.name == primary else None,
                                statistics.get(arm.name))
    return ScenarioRun(scenario.id, int(seed), runs, frames, profiling.gpu_memory_mb(device),
                       1000.0 * render_seconds / steps)
