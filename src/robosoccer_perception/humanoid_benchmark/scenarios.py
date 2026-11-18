"""The Phase 4 scenarios: the robot's body moves, and the camera moves with it.

A scenario fixes the path over the field, the body (the real humanoid or the placeholder), the
virtual camera mount, the synthetic body motion, the simulated IMU and whatever the robot gets
wrong about its mount. Every arm of the benchmark (`humanoid_benchmark.runner`) then localizes
from the SAME rendered images, IMU samples and odometry, and differs only in how it estimates
the camera pose.

All body motion here is SYNTHETIC (`simulation.body_motion`) and every mount is an EXPERIMENTAL
assumption (`configs/mounts/`): the robot has no head and cannot walk yet.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import numpy as np

from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.simulation import trajectories as tj
from robosoccer_perception.simulation.trajectories import Trajectory
from robosoccer_perception.vision_benchmark.camera_truth import HeadJitter, MountError

TrajectoryBuilder = Callable[[FieldGeometry, np.random.Generator], Trajectory]
STAND_FIRST = 2.0           # seconds the robot stands before it starts to walk


@dataclasses.dataclass(frozen=True)
class HumanoidScenario:
    id: str
    title: str
    description: str
    build_trajectory: TrajectoryBuilder
    # ---- the body and its camera
    robot: str = "humanoid_prototype_1"     # configs/robots/
    mount: str = "head_mid"                 # configs/mounts/: what the robot BELIEVES
    motion: str = "walking"                 # configs/motion/
    # ---- the IMU and its timing
    imu: str = "nominal"                    # configs/imu/
    imu_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    camera_latency_s: float = 0.030         # exposure to the image reaching the pipeline
    # ---- hidden from the robot
    mount_error: MountError = dataclasses.field(default_factory=MountError)   # truth - belief
    flex: HeadJitter = dataclasses.field(
        default_factory=lambda: HeadJitter(0.10, 0.10, 0.0, 0.0))  # mount flex per frame
    # ---- the rest, as in Phase 3
    field: str = "s_field"
    init_position_std: float = 0.30
    init_yaw_std_deg: float = 10.0
    odometry_scale: float = 1.0
    ball_xy: tuple[float, float] | None = (1.0, 0.5)
    start_shift: float = 0.20
    start_turn_deg: float = 5.0
    show_self: bool = True                  # the camera sees the robot's own body if in view
    expectation: str = "should localize"

    def replace(self, **changes) -> "HumanoidScenario":
        return dataclasses.replace(self, **changes)

    def describe(self) -> dict[str, Any]:
        return {"robot": self.robot, "mount": self.mount, "motion": self.motion,
                "imu": self.imu, "imu_overrides": dict(self.imu_overrides),
                "camera_latency_s": self.camera_latency_s,
                "mount_error": self.mount_error.describe(),
                "flex": dataclasses.asdict(self.flex)}


# ------------------------------------------------------------------ paths
def _standing(field: FieldGeometry, rng) -> Trajectory:
    pose = [-0.22 * field.dims.length, -0.17 * field.dims.width, math.radians(20.0)]
    return tj.stationary(pose, duration=20.0)


def _straight(field: FieldGeometry, rng) -> Trajectory:
    path = tj.PathBuilder([-0.36 * field.dims.length, -0.22 * field.dims.width, 0.0])
    path.hold(STAND_FIRST).forward(0.72 * field.dims.length, speed=0.3).hold(1.0)
    return path.build("straight_line")


def _rectangle(field: FieldGeometry, rng) -> Trajectory:
    size_x, size_y = 0.4 * field.dims.length, 0.4 * field.dims.width
    path = tj.PathBuilder([-size_x / 2, -size_y / 2, 0.0])
    path.hold(STAND_FIRST)
    for side in (size_x, size_y, size_x, size_y):
        path.forward(side, 0.3).turn(math.pi / 2)
    return path.build("rectangle")


def _turning(field: FieldGeometry, rng) -> Trajectory:
    path = tj.PathBuilder([0.11 * field.dims.length, 0.08 * field.dims.width, 0.0])
    path.hold(STAND_FIRST).turn(4.0 * math.pi, math.radians(30.0))
    return path.build("turning")


def _build() -> list[HumanoidScenario]:
    walk = dict(build_trajectory=_rectangle, motion="walking")
    return [
        HumanoidScenario(
            "h01_standing", "Stationary standing",
            "The robot stands for 20 s with slow postural sway and a little wobble.",
            _standing, motion="standing"),
        HumanoidScenario(
            "h02_sway_pitch", "Forward / backward sway",
            "Standing, the body pitches +-4 deg about the ankles twice in four seconds.",
            _standing, motion="sway_pitch",
            expectation="a fixed camera model sees every line slide to and fro"),
        HumanoidScenario(
            "h03_sway_roll", "Lateral sway",
            "Standing, the body rolls +-4 deg about the ankles.", _standing, motion="sway_roll"),
        HumanoidScenario(
            "h04_walk_straight", "Walking forward",
            "Two seconds standing, then a straight walk along most of the field at 0.3 m/s "
            "with the walking profile.", _straight, motion="walking"),
        HumanoidScenario(
            "h05_walk_rectangle", "Walking a rectangle",
            "Standing, then one lap of a 3.6 x 2.4 m rectangle with turns in place.", **walk),
        HumanoidScenario(
            "h06_turning", "Turning in place",
            "Standing, then two full turns on the spot at 30 deg/s while stepping.",
            _turning, motion="walking"),
        HumanoidScenario(
            "h07_disturbances", "Abrupt disturbances while standing",
            "Standing; every few seconds the body is tilted 4 to 8 deg in 0.12 s and recovers.",
            _standing, motion="disturbed"),
        HumanoidScenario(
            "h08_walk_disturbed", "Walking with disturbances",
            "The rectangle lap with the same abrupt tilts on top of the gait.",
            _rectangle, motion="walking_disturbed"),
        HumanoidScenario(
            "h09_walk_rough", "An unsteady gait",
            "The rectangle lap with about 1.7 times the body motion of the walking profile.",
            _rectangle, motion="walking_rough"),
        HumanoidScenario(
            "h10_mount_pitch_error", "Camera mount pitch error",
            "The rectangle lap; the camera is really pitched 2 deg further down than the mount "
            "calibration says.", mount_error=MountError(pitch_deg=2.0), **walk,
            expectation="the IMU cannot see this: every arm but the orientation oracle is "
                        "biased the same way"),
        HumanoidScenario(
            "h11_mount_roll_height_error", "Camera mount roll and height error",
            "The rectangle lap; the camera is really rolled 1.5 deg and 3 cm higher than the "
            "mount calibration says.", mount_error=MountError(roll_deg=1.5, height=0.03), **walk,
            expectation="as h10: a calibration error, not an orientation-estimation error"),
        HumanoidScenario(
            "h12_head_low", "Lower virtual head",
            "The rectangle lap with the camera 6 cm above the torso (about 0.74 m).",
            mount="head_low", **walk,
            expectation="below the camera heights the networks were trained at (0.78 to 1.02 m)"),
        HumanoidScenario(
            "h13_head_high", "Higher virtual head",
            "The rectangle lap with the camera 18 cm above the torso (about 0.86 m).",
            mount="head_high", **walk),
        HumanoidScenario(
            "h14_head_tall", "Tallest virtual head",
            "The rectangle lap with the camera 22 cm above the torso (about 0.90 m, the height "
            "Phases 1 to 3 assumed).", mount="head_tall", **walk),
        HumanoidScenario(
            "h15_imu_dropouts", "Missing IMU measurements",
            "The rectangle lap; 30 % of the IMU samples are lost and there is an outage of 0.1 "
            "to 0.4 s about every four seconds.", imu="dropouts", **walk),
        HumanoidScenario(
            "h16_timing_offset", "Camera / IMU clock offset",
            "The rectangle lap; the IMU's timestamps are 40 ms late with respect to the "
            "camera's, and the robot does not know.", imu_overrides=dict(timestamp_offset_s=0.040),
            **walk, expectation="the orientation is taken from the wrong instant"),
        HumanoidScenario(
            "h17_imu_late", "IMU arrives after the image",
            "The rectangle lap; IMU samples take 80 ms to arrive, so an image is processed "
            "before the IMU samples of its own instant are there.",
            imu_overrides=dict(latency_s=0.080), **walk,
            expectation="orientation has to be extrapolated; uncertainty should say so"),
        HumanoidScenario(
            "h18_placeholder_walk", "Placeholder rig, walking",
            "The rectangle lap with the Phase 1 to 3 capsule platform and its 0.90 m rig "
            "instead of the humanoid.", robot="placeholder", mount="legacy_rig", **walk,
            expectation="the fallback platform under the same motion"),
    ]


SCENARIOS: dict[str, HumanoidScenario] = {s.id: s for s in _build()}


def get_scenario(scenario_id: str) -> HumanoidScenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario '{scenario_id}' (available: {sorted(SCENARIOS)})")
    return SCENARIOS[scenario_id]


def select_scenarios(patterns: list[str] | None) -> list[HumanoidScenario]:
    """Scenarios whose id starts with any of the given prefixes; all of them for None."""
    if not patterns:
        return list(SCENARIOS.values())
    chosen = [s for s in SCENARIOS.values() if any(s.id.startswith(p) for p in patterns)]
    if not chosen:
        raise KeyError(f"no scenario matches {patterns} (available: {sorted(SCENARIOS)})")
    return chosen
