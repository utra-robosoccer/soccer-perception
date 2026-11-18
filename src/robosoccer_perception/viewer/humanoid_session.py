"""A Phase 4 scenario, run one frame at a time for the viewer: the real humanoid body (or the
placeholder), a virtual camera mount, synthetic body motion, a simulated IMU and one way of
estimating the camera pose.

`HumanoidLiveSession` has the same shape as `session.LiveSession` (``step``, ``reset``,
``close``, ``meta``), so the worker thread, the window, recordings and video export handle both
alike. It is a thin layer over the benchmark's own two classes:

    HumanoidSimulation   THE SIMULATOR. Hands out a `SensorPacket` and, separately, the truth.
    ArmEstimator         THE ROBOT. Receives the sensor packet and nothing else.

so what the viewer shows for an arm is what the benchmark measures for that arm. The truth of
a step goes into the frame's `TruthView` for display and is never given to the estimator; the
robot's own belief goes into `FusionView`.

`SessionOptions` are the viewer's controls: which body, which mount preset, which motion
profile and which camera-pose estimator. ``None`` keeps the scenario's own choice.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

from robosoccer_perception.estimation.calibration_monitor import MonitorConfig
from robosoccer_perception.estimation.camera_pose import FusionConfig, camera_roll_pitch
from robosoccer_perception.geometry.mount import available_mounts
from robosoccer_perception.geometry.robot_description import available_robots
from robosoccer_perception.humanoid_benchmark import runner
from robosoccer_perception.humanoid_benchmark.scenarios import HumanoidScenario
from robosoccer_perception.simulation.body_motion import available_profiles
from robosoccer_perception.viewer.frame import FusionView, ViewerFrame
from robosoccer_perception.viewer.session import make_frame, truth_view, update_rate

#: camera-pose estimators offered by the viewer: label -> benchmark arm (never the oracle)
FUSION_CHOICES = {"fixed": runner.FIXED, "imu": runner.IMU,
                  "imu_uncertainty": runner.IMU_UNCERTAINTY, "imu_selfcal": runner.IMU_SELFCAL,
                  "imu_latest": runner.IMU_LATEST}
FUSION_LABELS = {"fixed": "Fixed camera (Phase 3)", "imu": "IMU-assisted",
                 "imu_uncertainty": "IMU + uncertainty", "imu_selfcal": "IMU + mount self-cal.",
                 "imu_latest": "IMU, newest sample (naive)"}
DEFAULT_FUSION = "imu_uncertainty"


@dataclasses.dataclass(frozen=True)
class SessionOptions:
    """The viewer's Phase 4 controls. ``None`` means "as the scenario defines it"."""
    robot: str | None = None            # configs/robots/
    mount: str | None = None            # configs/mounts/
    motion: str | None = None           # configs/motion/
    fusion: str = DEFAULT_FUSION        # one of FUSION_CHOICES
    robot_view: bool = True             # render the third-person picture of the robot

    def __post_init__(self) -> None:
        if self.fusion not in FUSION_CHOICES:
            raise ValueError(f"fusion must be one of {sorted(FUSION_CHOICES)}")
        for value, known, kind in ((self.robot, available_robots(), "robot"),
                                   (self.mount, available_mounts(), "mount"),
                                   (self.motion, available_profiles(), "motion profile")):
            if value is not None and value not in known:
                raise ValueError(f"unknown {kind} '{value}' (available: {known})")

    def apply(self, scenario: HumanoidScenario) -> HumanoidScenario:
        changes = {key: value for key, value in (("robot", self.robot), ("mount", self.mount),
                                                 ("motion", self.motion)) if value is not None}
        return scenario.replace(**changes) if changes else scenario


class HumanoidLiveSession:
    """A Phase 4 scenario and seed, advanced by `step()`."""

    def __init__(self, scenario: HumanoidScenario, seed: int, resources, *,
                 options: SessionOptions | None = None, max_particles: int = 600,
                 with_truth: bool = True, max_steps: int | None = None,
                 fusion: FusionConfig | None = None, monitor: MonitorConfig | None = None):
        self.options = options or SessionOptions()
        self.base_scenario = scenario
        self.scenario = self.options.apply(scenario)
        self.seed, self.resources = int(seed), resources
        self.max_particles, self.with_truth = int(max_particles), bool(with_truth)
        self.arm = FUSION_CHOICES[self.options.fusion]
        self._fusion, self._monitor = fusion, monitor
        self._sim = runner.HumanoidSimulation(self.scenario, self.seed, resources, max_steps)
        self.field, self.dt = self._sim.field, self._sim.dt
        self.total_steps = self._sim.total_steps
        self._closed = False
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Back to step 0 with fresh random streams: the run that follows is the same run."""
        self._sim.reset()
        knowledge = self._sim.knowledge()
        self._knowledge = knowledge
        filter_seed = int(self._sim.filter_rng.integers(2 ** 31))
        self._estimator = runner.ArmEstimator(self.arm, knowledge, self.resources,
                                              self.resources.vision, filter_seed, self._fusion,
                                              self._monitor)
        self._estimator.initialise(*self._sim.start_guess())
        self._step = 0
        self._last_call: float | None = None
        self._fps = 0.0

    def close(self) -> None:
        if not self._closed:
            self._sim.close()
            self._closed = True

    def __enter__(self) -> "HumanoidLiveSession":
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
        scenario, sim = self.scenario, self._sim
        vision = self.resources.vision
        describe = getattr(vision, "describe", None)
        body, mount = sim.robot.body, sim.believed_mount
        return {
            "scenario": scenario.id, "title": scenario.title,
            "description": scenario.description, "expectation": scenario.expectation,
            "seed": self.seed, "field": scenario.field, "dt": self.dt,
            "steps": self.total_steps, "resolution": list(self.resources.resolution),
            "kidnap_steps": [], "models": describe() if callable(describe) else None,
            "device": str(getattr(vision, "device", "cpu")), "has_truth": self.with_truth,
            # ---- Phase 4
            "phase4": True, "robot": scenario.robot, "mount": scenario.mount,
            "motion": scenario.motion, "imu": scenario.imu, "fusion": self.options.fusion,
            "fusion_label": FUSION_LABELS[self.options.fusion],
            "mount_description": mount.describe(body),
            "camera_height_standing": mount.standing_height(body),
            "mount_experimental": bool(mount.experimental),
            "body_frame": body.body_frame}

    # ------------------------------------------------------------------ one step
    def step(self) -> ViewerFrame | None:
        """Advance by one frame. None once the trajectory is over."""
        if self._closed:
            raise RuntimeError("the session is closed")
        if self.finished:
            return None
        step = self._step
        tic = time.perf_counter()
        # THE SIMULATOR: sensor data for the robot; truth and the chase picture for display.
        packet, truth, picture = self._sim.sense(step, chase=self.options.robot_view)
        render_ms = (time.perf_counter() - tic) * 1000.0
        # THE ROBOT: the sensor packet and nothing else.
        out = self._estimator.step(packet)
        # THE VIEWER: reads both; nothing flows back.
        localization = out.localization
        timing = dict(localization.timing_ms)
        timing["total"] = timing["total"] + out.orientation_ms
        localization = dataclasses.replace(localization, timing_ms=timing)
        roll, pitch = camera_roll_pitch(out.pose.T_base_camera)
        estimate, state = out.camera, out.monitor
        fusion = FusionView(
            source=estimate.source, body_roll_deg=math.degrees(estimate.body_roll),
            body_pitch_deg=math.degrees(estimate.body_pitch),
            tilt_std_deg=math.degrees(estimate.tilt_std), camera_roll_deg=math.degrees(roll),
            camera_pitch_deg=math.degrees(pitch), camera_height=out.pose.height,
            sync_method="" if estimate.sync is None else estimate.sync.method,
            sync_offset_ms=math.nan if estimate.sync is None
            else 1000.0 * estimate.sync.offset, imu_samples=len(packet.imu),
            monitor_status="" if state is None else state.status,
            monitor_bias_deg=(math.nan, math.nan) if state is None else state.bias_deg,
            monitor_correction_deg=(0.0, 0.0) if state is None else state.correction_deg,
            line_residual=math.nan if state is None else state.residual_rms)
        self._fps, self._last_call = update_rate(self._fps, self._last_call)
        shown = None
        if self.with_truth:
            true_roll, true_pitch = truth.camera_roll_pitch
            shown = truth_view(
                localization.estimate.pose, truth.footprint,
                body_roll_deg=math.degrees(truth.body_roll),
                body_pitch_deg=math.degrees(truth.body_pitch),
                camera_roll_deg=math.degrees(true_roll),
                camera_pitch_deg=math.degrees(true_pitch), camera_height=truth.camera_height)
        frame = make_frame(step, packet.timestamp, packet.image, localization, out.calibration,
                           out.pose, self._estimator.pipeline.localizer, self.field,
                           self.max_particles, render_ms, self._fps,
                           step > 0 and packet.odometry is None, shown, fusion, picture)
        self._step += 1
        return frame
