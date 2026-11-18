"""Running one Phase 4 scenario: a moving body on one side, camera-pose estimators on the other.

    path ─► BodyMotion (6-DOF, synthetic) ─┬─► true camera pose ─► MuJoCo render ─► RGB image ─┐
                 (hidden)                  ├─► simulated IMU ─► ImuSample stream ──────────────┤
                                           └─► footprint poses ─► SyntheticOdometry ───────────┤
                                                                                               ▼
       per ARM:  CameraPoseEstimator (mount belief + IMU) ─► EstimatedCameraPose ─► Phase 3 pipeline
                                                                                               │
       evaluation (pose error, tilt error, projection error, consistency) ◄────────────────────┘

`HumanoidSimulation` is the simulator's side. Per frame it hands out a `SensorPacket` (image,
timestamp, odometry increment, the IMU samples that arrived since the previous frame) and,
separately, a `StepTruth` for scoring and display.

`ArmEstimator` is the robot's side for one ARM. It receives sensor packets and nothing else.
All arms of a run see the same packets; the networks run once per frame and the other arms
reuse their predictions, so the arms differ ONLY in the camera pose they project with:

    fixed               an upright, level body in every frame (the Phase 3 assumption)
    oracle_orientation  the camera's TRUE orientation. An upper bound that uses simulator
                        ground truth, labelled as such, and built here in the evaluator.
    imu                 roll / pitch from the simulated IMU, interpolated to the image time
    imu_uncertainty     the same, with per-frame uncertainty and the shared-error model
    imu_latest          the same as ``imu`` but pairing each image with the NEWEST IMU estimate
    imu_selfcal         ``imu_uncertainty`` plus the bounded mount-bias correction

The viewer's live session (`viewer.humanoid_session`) drives the same two classes.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np

from robosoccer_perception.estimation.calibration_monitor import (
    CalibrationMonitor, MonitorConfig)
from robosoccer_perception.estimation.camera_pose import (
    CameraPoseEstimate, CameraPoseEstimator, FusionConfig, camera_roll_pitch)
from robosoccer_perception.estimation.imu_types import ImuSample
from robosoccer_perception.evaluation import metrics as metrics_module
from robosoccer_perception.evaluation.benchmark import RunResult
from robosoccer_perception.geometry import mount as mount_geometry
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ProjectionLimits, ground_to_pixels,
    project_to_ground)
from robosoccer_perception.geometry.landmarks import build_landmark_map
from robosoccer_perception.geometry.mount import CameraMount
from robosoccer_perception.geometry.robot_description import RobotDescription
from robosoccer_perception.humanoid_benchmark.scenarios import HumanoidScenario
from robosoccer_perception.inference import profiling
from robosoccer_perception.inference.types import VisionResult
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.pipeline.vision_localization import (
    VisionLocalizationOutput, VisionLocalizationPipeline)
from robosoccer_perception.simulation.body_motion import BodyMotion, MotionProfile
from robosoccer_perception.simulation.humanoid import BodyPose
from robosoccer_perception.simulation.imu import ImuConfig, ImuStream, simulate_imu
from robosoccer_perception.simulation.odometry import OdometryNoiseConfig, SyntheticOdometry
from robosoccer_perception.vision_benchmark import runner as vision_runner
from robosoccer_perception.vision_benchmark.camera_truth import SimulationCameraTruth
from robosoccer_perception.vision_benchmark.runner import BenchmarkResources, EstimateError
from robosoccer_perception.vision_benchmark.scenarios import vary_trajectory
from robosoccer_perception.vision_benchmark.sim_camera import SimulatedCamera

TIMING_KEYS = ("orientation", "detector", "segmenter", "geometry", "filter", "total")
#: Ground points (footprint frame) at which the camera-projection error is measured.
_REFERENCE_POINTS = np.array([[r * math.cos(b), r * math.sin(b)]
                              for r in (1.0, 2.0, 3.0, 4.0)
                              for b in np.radians([-25.0, 0.0, 25.0])])
_WIDE_LIMITS = ProjectionLimits(min_depression_deg=0.5, max_range=60.0, min_range=0.0)


# --------------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FusionArm:
    """One way of estimating the camera pose, run on the shared sensor data."""
    name: str
    mode: str = "imu"                       # fixed | imu | oracle
    sync: str = "interpolate"               # interpolate | latest
    use_uncertainty: bool = False           # per-frame uncertainty + shared-error model
    self_calibration: bool = False          # bounded mount-bias correction from the images
    timestamp_bias_s: float = 0.0           # the image timestamp the arm believes, minus truth
    estimate_error: EstimateError = dataclasses.field(default_factory=EstimateError)
    description: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "imu", "oracle"):
            raise ValueError("mode must be fixed, imu or oracle")

    @property
    def is_oracle(self) -> bool:
        return self.mode == "oracle"

    def fusion(self, base: FusionConfig) -> FusionConfig:
        mode = "fixed" if self.mode in ("fixed", "oracle") else "imu"
        return base.replace(mode=mode, sync=self.sync, use_uncertainty=self.use_uncertainty)


FIXED = FusionArm("fixed", "fixed", description="A  fixed camera: an upright, level body in "
                                                "every frame (Phase 3)")
ORACLE = FusionArm("oracle_orientation", "oracle",
                   description="B  ORACLE: the true camera orientation (upper bound, uses "
                               "simulator truth)")
IMU = FusionArm("imu", "imu", description="C  IMU tilt, interpolated to the image timestamp")
IMU_UNCERTAINTY = FusionArm("imu_uncertainty", "imu", use_uncertainty=True,
                            description="D  IMU tilt with per-frame uncertainty and the "
                                        "shared-error model")
IMU_LATEST = FusionArm("imu_latest", "imu", sync="latest",
                       description="C with naive pairing: the newest IMU estimate")
IMU_SELFCAL = FusionArm("imu_selfcal", "imu", use_uncertainty=True, self_calibration=True,
                        description="D plus the bounded mount-bias correction from field lines")
DEFAULT_ARMS = (FIXED, ORACLE, IMU, IMU_UNCERTAINTY, IMU_LATEST, IMU_SELFCAL)
ARMS = {arm.name: arm for arm in DEFAULT_ARMS}


# --------------------------------------------------------------------------------------------
# What crosses from the simulator to the robot, and what does not
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SensorPacket:
    """Everything the robot receives for one camera frame."""
    image: np.ndarray                       # (H, W, 3) uint8 RGB
    timestamp: float                        # the image's own timestamp, seconds
    odometry: OdometryDelta | None
    imu: tuple[ImuSample, ...]              # samples that arrived since the previous frame


@dataclasses.dataclass(frozen=True)
class StepTruth:
    """The simulator's truth of one frame. For scoring and display; never for an estimator."""
    step: int
    footprint: np.ndarray                   # (3,) true [x, y, yaw]
    body_roll: float                        # radians
    body_pitch: float
    body_height: float                      # metres
    T_footprint_optical: np.ndarray         # 4x4 true left optical frame
    camera: SimulationCameraTruth
    gait_phase: float
    activity: float

    @property
    def camera_height(self) -> float:
        return float(self.T_footprint_optical[2, 3])

    @property
    def camera_roll_pitch(self) -> tuple[float, float]:
        return camera_roll_pitch(self.T_footprint_optical)


@dataclasses.dataclass(frozen=True)
class RobotKnowledge:
    """What the robot is given once, before the run: the map, its camera's intrinsics, its
    body's geometry and the mount it BELIEVES it has."""
    field: FieldGeometry
    landmarks: Any
    calibration: EstimatedCameraCalibration
    mount: CameraMount
    robot: RobotDescription
    baseline: float


class HumanoidSimulation:
    """The simulator's side of one scenario and seed."""

    def __init__(self, scenario: HumanoidScenario, seed: int, resources: BenchmarkResources,
                 max_steps: int | None = None):
        self.scenario, self.seed, self.resources = scenario, int(seed), resources
        streams = vision_runner._streams(scenario.id, self.seed, 9)
        trajectory_rng, motion_rng, imu_rng = streams[0], streams[5], streams[6]
        self.field = FieldGeometry.load(scenario.field)
        self.landmarks = build_landmark_map(self.field)
        self.camera_cfg = resources.camera_config()
        self.robot = RobotDescription.load(scenario.robot)
        body = self.robot.body
        self.believed_mount = CameraMount.load(scenario.mount)
        error = scenario.mount_error
        self.true_mount = self.believed_mount.shifted(
            up=error.height, pitch=math.radians(error.pitch_deg),
            roll=math.radians(error.roll_deg), yaw=math.radians(error.pan_deg))

        path = vary_trajectory(scenario.build_trajectory(self.field, trajectory_rng),
                               trajectory_rng, scenario.start_shift, scenario.start_turn_deg)
        self.profile = MotionProfile.load(scenario.motion)
        self.motion = BodyMotion(path, self.profile, body.standing_height, motion_rng)
        self.body_samples = self.motion.sample(path.times)
        self.trajectory = self.motion.footprint_trajectory()
        self.dt = float(path.dt)
        self.total_steps = len(path) if max_steps is None else min(len(path), int(max_steps))
        self.imu_config = ImuConfig.load(scenario.imu, scenario.imu_overrides)
        self.imu: ImuStream = simulate_imu(self.motion, self.imu_config, imu_rng,
                                           body.T_body_imu)
        self._sim = SimulatedCamera(self.field, self.camera_cfg, {}, False, None,
                                    robot=self.robot, mount=self.true_mount,
                                    show_self=scenario.show_self)
        self._closed = False
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Fresh random streams for everything that is drawn frame by frame."""
        streams = vision_runner._streams(self.scenario.id, self.seed, 9)
        self._odometry = SyntheticOdometry(
            OdometryNoiseConfig().scaled(self.scenario.odometry_scale), streams[1])
        self.init_rng, self.filter_rng, self._flex_rng = streams[2], streams[3], streams[4]
        self._last_delivery = -math.inf

    def close(self) -> None:
        if not self._closed:
            self._sim.close()
            self._closed = True

    def __enter__(self) -> "HumanoidSimulation":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ what the robot is told
    def knowledge(self) -> RobotKnowledge:
        """The robot's prior knowledge: built from the BELIEVED mount, never the true one."""
        body, mount, cfg = self.robot.body, self.believed_mount, self.camera_cfg
        position = mount.position_in_body(body)
        calibration = EstimatedCameraCalibration(
            cfg.intrinsics, nominal_height=mount.standing_height(body),
            nominal_pitch=mount.rpy[1], forward_offset=float(position[0]),
            lateral_offset=float(position[1]), optical_offset_y=0.5 * cfg.baseline,
            uncertainty=self.resources.noise.calibration)
        return RobotKnowledge(self.field, self.landmarks, calibration, mount, self.robot,
                              cfg.baseline)

    def start_guess(self) -> tuple[np.ndarray, float, float]:
        """A noisy guess of the first pose, as a referee or a previous run would give it."""
        scenario = self.scenario
        position_std = scenario.init_position_std
        yaw_std = math.radians(scenario.init_yaw_std_deg)
        guess = self.trajectory.poses[0] \
            + self.init_rng.normal(size=3) * [position_std, position_std, yaw_std]
        return guess, position_std, yaw_std

    # ------------------------------------------------------------------ one frame
    def truth_at(self, step: int, flex: np.ndarray | None = None) -> StepTruth:
        sample, body = self.body_samples, self.robot.body
        roll, pitch = float(sample.roll[step]), float(sample.pitch[step])
        height = float(sample.height[step])
        mount = self.true_mount
        if flex is not None:
            mount = mount.shifted(roll=float(flex[0]), pitch=float(flex[1]))
        T_footprint_rig = mount_geometry.T_footprint_rig(mount, body, height, roll, pitch)
        rig = mount_geometry.rig_state_from_transform(T_footprint_rig)
        camera = SimulationCameraTruth(rig, self.camera_cfg.intrinsics, self.camera_cfg.baseline)
        T_optical = T_footprint_rig @ mount_geometry.T_rig_optical(self.camera_cfg.baseline)
        return StepTruth(step, sample.footprint[step].copy(), roll, pitch, height, T_optical,
                         camera, float(sample.gait_phase[step]), float(sample.activity[step]))

    def sense(self, step: int, chase: bool = False
              ) -> tuple[SensorPacket, StepTruth, np.ndarray | None]:
        """Render frame ``step`` and collect what the robot receives with it. Returns the
        sensor packet, the truth (kept apart) and, on request, a third-person picture."""
        if self._closed:
            raise RuntimeError("the simulation is closed")
        scenario = self.scenario
        timestamp = step * self.dt
        flex_std = np.radians([scenario.flex.roll_std_deg, scenario.flex.pitch_std_deg])
        truth = self.truth_at(step, self._flex_rng.normal(size=2) * flex_std)
        pose = BodyPose(mount_geometry.T_footprint_body(truth.body_height, truth.body_roll,
                                                        truth.body_pitch),
                        truth.gait_phase, truth.activity)
        image = self._sim.capture(truth.footprint, truth.camera, scenario.ball_xy, (), None,
                                  body=pose)
        picture = self._sim.chase_view() if chase else None
        delta = None
        if step > 0:
            delta = self._odometry.measure(self.trajectory.poses[step - 1], truth.footprint,
                                           timestamp, self.dt)
        # The image reaches the pipeline ``camera_latency_s`` after its exposure; every IMU
        # sample that has arrived by then (and was not delivered with an earlier frame) is
        # handed over with it.
        delivery = timestamp + scenario.camera_latency_s
        samples = tuple(self.imu.arrived_between(self._last_delivery, delivery))
        self._last_delivery = delivery
        return SensorPacket(image, timestamp, delta, samples), truth, picture


# --------------------------------------------------------------------------------------------
# The robot's side
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ArmOutput:
    localization: VisionLocalizationOutput
    camera: CameraPoseEstimate                  # what was estimated before any sweep error
    pose: EstimatedCameraPose                   # the camera pose that was projected with
    calibration: EstimatedCameraCalibration
    orientation_ms: float
    monitor: Any = None                         # MonitorState, when self-calibration runs


class ArmEstimator:
    """One arm: IMU filter, camera-pose estimator and the Phase 3 localization pipeline."""

    def __init__(self, arm: FusionArm, knowledge: RobotKnowledge, resources: BenchmarkResources,
                 vision, filter_seed: int, fusion: FusionConfig | None = None,
                 monitor: MonitorConfig | None = None):
        self.arm, self.knowledge = arm, knowledge
        noise = resources.noise
        if arm.use_uncertainty:
            noise = dataclasses.replace(noise, shared_camera_error=True)
        self.pipeline = VisionLocalizationPipeline(
            vision, knowledge.field, extraction=resources.extraction, noise=noise,
            filter_config=resources.particle_filter_config(image_fed=True),
            landmarks=knowledge.landmarks, seed=filter_seed)
        self.camera = CameraPoseEstimator(
            knowledge.calibration, knowledge.mount, knowledge.robot.body, knowledge.baseline,
            arm.fusion(fusion or FusionConfig()))
        self.monitor = CalibrationMonitor(knowledge.field, monitor) if arm.self_calibration \
            else None

    def initialise(self, guess, position_std: float, yaw_std: float) -> None:
        self.pipeline.init_gaussian(guess, position_std, yaw_std)

    def step(self, packet: SensorPacket, predictions: VisionResult | None = None,
             oracle_pose: EstimatedCameraPose | None = None) -> ArmOutput:
        """Process one frame. ``predictions`` reuses another arm's network output for the same
        image. ``oracle_pose`` is given to the ORACLE arm only, by the evaluator."""
        if type(packet) is not SensorPacket:
            raise TypeError("an arm is driven by SensorPacket objects")
        if (oracle_pose is not None) != self.arm.is_oracle:
            raise ValueError("an oracle camera pose is for the oracle arm, and only for it")
        tic = time.perf_counter()
        for sample in packet.imu:
            self.camera.add_imu(sample)
        estimate = self.camera.estimate(packet.timestamp + self.arm.timestamp_bias_s)
        orientation_ms = (time.perf_counter() - tic) * 1000.0
        pose = oracle_pose if oracle_pose is not None else estimate.pose
        calibration, pose = self.arm.estimate_error.apply(estimate.calibration, pose)
        if predictions is None:
            output = self.pipeline.step(packet.image, packet.timestamp, packet.odometry,
                                        calibration, pose)
        else:
            output = self.pipeline.step_with_predictions(predictions, packet.timestamp,
                                                         packet.odometry, calibration, pose)
        state = None
        if self.monitor is not None:
            state = self.monitor.update(output.features, output.estimate, calibration, pose)
            self.camera.set_mount_correction(*self.monitor.correction)
        return ArmOutput(output, estimate, pose, calibration, orientation_ms, state)


def oracle_camera_pose(truth: StepTruth, nominal: EstimatedCameraPose) -> EstimatedCameraPose:
    """EVALUATOR ONLY: the true camera ORIENTATION at the nominal camera position. The upper
    bound for anything that estimates orientation, including a perfect IMU on a perfectly
    calibrated mount."""
    return EstimatedCameraPose(tf.make_transform(truth.T_footprint_optical[:3, :3],
                                                 nominal.T_base_camera[:3, 3]))


def projection_error(truth: StepTruth, calibration: EstimatedCameraCalibration,
                     pose: EstimatedCameraPose) -> float:
    """Mean distance in metres by which the estimated camera pose misplaces reference ground
    points 1 to 4 m ahead: each point is seen where the TRUE camera puts it in the image and
    projected back with the ESTIMATED pose. NaN when none of them is in view."""
    true_pose = EstimatedCameraPose(truth.T_footprint_optical)
    pixels, visible = ground_to_pixels(_REFERENCE_POINTS, calibration, true_pose)
    if not visible.any():
        return math.nan
    ground = project_to_ground(pixels[visible], calibration, pose, _WIDE_LIMITS)
    error = np.linalg.norm(ground.points - _REFERENCE_POINTS[visible], axis=1)
    error = error[ground.valid & np.isfinite(error)]
    return float(error.mean()) if len(error) else math.nan


# --------------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FusionArmRun:
    arm: FusionArm
    result: RunResult
    camera_roll_error_deg: np.ndarray       # (T,) estimated minus true camera roll
    camera_pitch_error_deg: np.ndarray
    camera_height_error: np.ndarray         # (T,) metres
    body_roll_deg: np.ndarray               # (T,) the body tilt the arm used
    body_pitch_deg: np.ndarray
    tilt_std_deg: np.ndarray                # (T,) NaN where the arm reports none
    projection_error: np.ndarray            # (T,) metres
    sources: dict[str, int]                 # how the camera pose was obtained, per frame
    sync: dict[str, Any] | None
    correction_deg: np.ndarray | None = None    # (T, 2) roll, pitch mount correction applied
    timing_ms: dict[str, np.ndarray] | None = None      # the primary arm only


@dataclasses.dataclass
class FusionRun:
    scenario_id: str
    seed: int
    arms: dict[str, FusionArmRun]
    time: np.ndarray
    truth: np.ndarray                       # (T, 3) footprint poses
    true_body_roll_deg: np.ndarray
    true_body_pitch_deg: np.ndarray
    true_camera_roll_deg: np.ndarray
    true_camera_pitch_deg: np.ndarray
    true_camera_height: np.ndarray
    imu: dict[str, Any]                     # rate, share of samples lost, ...
    render_ms_mean: float
    gpu_memory_mb: dict[str, float] | None


def run_humanoid_scenario(scenario: HumanoidScenario, seed: int, resources: BenchmarkResources,
                          arms: tuple[FusionArm, ...] | None = None, *,
                          fusion: FusionConfig | None = None,
                          monitor: MonitorConfig | None = None,
                          max_steps: int | None = None) -> FusionRun:
    """Run one scenario with one seed for every arm and return their time series."""
    arms = tuple(arms if arms is not None else DEFAULT_ARMS)
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    device = getattr(resources.vision, "device", "cpu")
    profiling.reset_gpu_peak(device)
    wall_start = time.perf_counter()
    with HumanoidSimulation(scenario, seed, resources, max_steps) as sim:
        steps = sim.total_steps
        knowledge = sim.knowledge()
        filter_seed = int(sim.filter_rng.integers(2 ** 31))
        primary = next(arm.name for arm in arms)
        estimators = {arm.name: ArmEstimator(arm, knowledge, resources, resources.vision,
                                             filter_seed, fusion, monitor) for arm in arms}
        guess, position_std, yaw_std = sim.start_guess()
        for estimator in estimators.values():
            estimator.initialise(guess, position_std, yaw_std)

        series = {arm.name: vision_runner._Series(steps) for arm in arms}
        extra = {arm.name: {key: np.full(steps, np.nan) for key in (
            "roll_error", "pitch_error", "height_error", "body_roll", "body_pitch", "tilt_std",
            "projection")} for arm in arms}
        corrections = {arm.name: np.zeros((steps, 2)) for arm in arms if arm.self_calibration}
        sources: dict[str, dict[str, int]] = {arm.name: {} for arm in arms}
        timing = {key: np.zeros(steps) for key in TIMING_KEYS}
        truth_pose = np.zeros((steps, 3))
        true_values = {key: np.zeros(steps) for key in (
            "body_roll", "body_pitch", "camera_roll", "camera_pitch", "camera_height")}
        render_seconds = 0.0

        for step in range(steps):
            tic = time.perf_counter()
            packet, truth, _ = sim.sense(step)
            render_seconds += time.perf_counter() - tic
            truth_pose[step] = truth.footprint
            true_roll, true_pitch = truth.camera_roll_pitch
            for key, value in (("body_roll", truth.body_roll), ("body_pitch", truth.body_pitch),
                               ("camera_roll", true_roll), ("camera_pitch", true_pitch)):
                true_values[key][step] = math.degrees(value)
            true_values["camera_height"][step] = truth.camera_height
            shared: VisionResult | None = None
            for arm in arms:
                estimator = estimators[arm.name]
                oracle = oracle_camera_pose(truth, estimator.camera.nominal_pose()) \
                    if arm.is_oracle else None
                out = estimator.step(packet, shared, oracle)
                if shared is None:
                    shared = out.localization.vision
                    timing["orientation"][step] = out.orientation_ms
                    for key in TIMING_KEYS[1:]:
                        timing[key][step] = out.localization.timing_ms[key]
                    timing["total"][step] += out.orientation_ms
                localization = out.localization
                series[arm.name].record(
                    step, localization.estimate, localization.update, localization.observations,
                    estimator.pipeline.localizer.num_particles,
                    localization.timing_ms["filter"] / 1000.0)
                values = extra[arm.name]
                roll, pitch = camera_roll_pitch(out.pose.T_base_camera)
                values["roll_error"][step] = math.degrees(roll - true_roll)
                values["pitch_error"][step] = math.degrees(pitch - true_pitch)
                values["height_error"][step] = out.pose.height - truth.camera_height
                values["body_roll"][step] = math.degrees(out.camera.body_roll)
                values["body_pitch"][step] = math.degrees(out.camera.body_pitch)
                values["tilt_std"][step] = math.degrees(out.camera.tilt_std)
                values["projection"][step] = projection_error(truth, out.calibration, out.pose)
                source = "oracle" if arm.is_oracle else out.camera.source
                sources[arm.name][source] = sources[arm.name].get(source, 0) + 1
                if arm.name in corrections:
                    corrections[arm.name][step] = np.degrees(estimator.camera.mount_correction)

        imu_summary = {"rate_hz": sim.imu_config.rate_hz, "samples": len(sim.imu),
                       "lost_fraction": sim.imu.lost_fraction}
        trajectory = sim.trajectory
        sync = {name: (None if arms_by_name.is_oracle or arms_by_name.mode == "fixed"
                       else estimators[name].camera.history.statistics.summary())
                for name, arms_by_name in ((arm.name, arm) for arm in arms)}

    wall = time.perf_counter() - wall_start
    times = trajectory.times[:steps]
    runs: dict[str, FusionArmRun] = {}
    for arm in arms:
        record, values = series[arm.name], extra[arm.name]
        position_error, yaw_error = metrics_module.pose_errors(record.estimate, truth_pose)
        mirror_position, mirror_yaw = metrics_module.mirror_errors(record.estimate, truth_pose)
        result = RunResult(
            scenario_id=scenario.id, field_name=sim.field.name, seed=int(seed), dt=sim.dt,
            kidnap_steps=(), time=times, truth=truth_pose, estimate=record.estimate,
            position_error=position_error, yaw_error_deg=yaw_error,
            mirror_position_error=mirror_position, mirror_yaw_error_deg=mirror_yaw,
            wall_time_s=wall, snapshots={}, **record.values, **record.flags)
        runs[arm.name] = FusionArmRun(
            arm, result, values["roll_error"], values["pitch_error"], values["height_error"],
            values["body_roll"], values["body_pitch"], values["tilt_std"], values["projection"],
            sources[arm.name], sync[arm.name], corrections.get(arm.name),
            timing if arm.name == primary else None)
    return FusionRun(scenario.id, int(seed), runs, times, truth_pose,
                     true_values["body_roll"], true_values["body_pitch"],
                     true_values["camera_roll"], true_values["camera_pitch"],
                     true_values["camera_height"], imu_summary,
                     1000.0 * render_seconds / steps, profiling.gpu_memory_mb(device))
