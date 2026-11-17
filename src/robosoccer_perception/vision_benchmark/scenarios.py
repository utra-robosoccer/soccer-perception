"""The Phase 3 benchmark scenarios. Every scenario is deterministic given a seed.

A scenario fixes the true trajectory, the true camera (mount error, head shake), what stands
on the field, how the scene looks, how bad the odometry is and how the filter is started. The
same scenario is then run by several ARMS that differ only in where their observations come
from (`vision_benchmark/runner.py`), so that oracle, vision and odometry-only localization are
compared on identical trajectories, odometry streams and initial uncertainty.

Unlike Phase 1, the true trajectory is shifted and turned a little for every seed
(`vary_trajectory`). In a deterministic renderer the image at a given pose is always the same;
without this, every seed would repeat the same pixels and the same perception errors, and the
seeds would only differ in odometry and filter noise.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.simulation import trajectories as tj
from robosoccer_perception.simulation.scene_objects import RobotState
from robosoccer_perception.simulation.trajectories import Trajectory
from robosoccer_perception.vision_benchmark.camera_truth import HeadJitter, MountError
from robosoccer_perception.vision_benchmark.corruption import VisionCorruption

TrajectoryBuilder = Callable[[FieldGeometry, np.random.Generator], Trajectory]
INIT_MODES = ("gaussian", "global", "own_half", "mirrored")
DEFAULT_BALL = (1.0, 0.5)
ALL_SOURCES = ("vision", "oracle", "odometry", "perfect_vision")


@dataclasses.dataclass(frozen=True)
class VisionScenario:
    id: str
    title: str
    description: str
    build_trajectory: TrajectoryBuilder
    field: str = "s_field"
    init: str = "gaussian"
    init_position_std: float = 0.30
    init_yaw_std_deg: float = 10.0
    odometry_scale: float = 1.0
    odometry_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    # ---- the true camera (hidden from the robot)
    mount_error: MountError = dataclasses.field(default_factory=MountError)
    jitter: HeadJitter = dataclasses.field(default_factory=HeadJitter)
    # ---- what the robot commands and therefore knows
    commanded_pitch_deg: float | None = None    # None = the mount's nominal pitch
    # ---- the world
    ball_xy: tuple[float, float] | None = DEFAULT_BALL
    robots: tuple[RobotState, ...] = ()
    randomization: str | None = None            # appearance preset, drawn anew for every seed
    # ---- degradations
    corruption: VisionCorruption | None = None  # applied to the vision arm's predictions
    oracle_profile: str = "nominal"
    oracle_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    # ---- bookkeeping
    start_shift: float = 0.20                   # metres: per-seed shift of the whole path
    start_turn_deg: float = 5.0                 # degrees: per-seed turn about its start
    sources: tuple[str, ...] = ALL_SOURCES
    expectation: str = "should localize"

    def __post_init__(self) -> None:
        if self.init not in INIT_MODES:
            raise ValueError(f"init must be one of {INIT_MODES}")
        unknown = set(self.sources) - set(ALL_SOURCES)
        if unknown:
            raise ValueError(f"unknown observation sources {sorted(unknown)}")


def vary_trajectory(trajectory: Trajectory, rng: np.random.Generator, shift: float,
                    turn_deg: float) -> Trajectory:
    """The same path, moved rigidly: its start shifted by up to ``shift`` metres in x and y and
    the whole path turned about the start by up to ``turn_deg``."""
    start = trajectory.poses[0]
    relative = tf.se2_relative(start, trajectory.poses)
    offset = np.array([rng.uniform(-shift, shift), rng.uniform(-shift, shift),
                       math.radians(rng.uniform(-turn_deg, turn_deg))])
    moved = start + offset
    moved[2] = tf.wrap_angle(moved[2])
    return dataclasses.replace(trajectory, poses=tf.se2_compose(moved, relative))


# ------------------------------------------------------------------ trajectories
def _scan(trajectory: Trajectory) -> Trajectory:
    """Look left and right while moving: +-50 degrees every 4 s (a commanded head motion)."""
    return trajectory.with_head_scan(math.radians(50.0), 4.0)


def _stationary(field: FieldGeometry, rng) -> Trajectory:
    pose = [-0.22 * field.dims.length, -0.17 * field.dims.width, math.radians(20.0)]
    return tj.stationary(pose, duration=20.0)


def _rotation(field: FieldGeometry, rng) -> Trajectory:
    pose = [0.11 * field.dims.length, 0.08 * field.dims.width, 0.0]
    return tj.rotate_in_place(pose, turns=2.0, rate=math.radians(30.0))


def _straight(field: FieldGeometry, rng) -> Trajectory:
    start = [-0.36 * field.dims.length, -0.22 * field.dims.width, 0.0]
    return tj.straight_line(start, 0.72 * field.dims.length, speed=0.3)


def _rectangle(fraction: float = 0.5, speed: float = 0.3) -> TrajectoryBuilder:
    def build(field: FieldGeometry, rng) -> Trajectory:
        size_x, size_y = fraction * field.dims.length, fraction * field.dims.width
        return tj.rectangle([-size_x / 2, -size_y / 2, 0.0], size_x, size_y, 1, speed)
    return build


def _curve(field: FieldGeometry, rng) -> Trajectory:
    radius = 0.30 * field.dims.width
    return tj.circle([0.0, -radius, 0.0], radius, turns=1.0, speed=0.3)


def _goal_approach(field: FieldGeometry, rng) -> Trajectory:
    """Walk toward the +x goal, sidestep across its mouth close to the posts, and turn: the
    posts are cut by the image border and hidden behind other robots for much of the run."""
    a, b = field.half_length, field.half_width
    path = tj.PathBuilder([0.30 * a, -0.25 * b, math.radians(10.0)])
    path.go_to(0.62 * a, -0.10 * b).turn_to(0.0)
    path.sidestep(0.45 * b).turn(math.radians(-35.0)).turn(math.radians(70.0))
    path.turn_to(math.pi).forward(0.25 * a).turn_to(0.0).hold(2.0)
    return path.build("goal_approach").with_head_scan(math.radians(35.0), 5.0)


def _goal_robots(field: FieldGeometry) -> tuple[RobotState, ...]:
    """A goalkeeper and two players standing in front of the +x goal posts."""
    line_x = field.goals[0].line_x
    post_y = field.goals[0].posts[0][1]
    return (RobotState((line_x - 0.25, 0.10), math.pi),
            RobotState((line_x - 0.45, post_y - 0.05), math.radians(170.0)),
            RobotState((line_x - 0.60, -post_y + 0.15), math.radians(-160.0)))


def _symmetry(field: FieldGeometry, rng) -> Trajectory:
    """A fixed start in the -x half, a look around, then a walk that stays in that half."""
    a, b = field.half_length, field.half_width
    path = tj.PathBuilder([-0.45 * a, -0.35 * b, math.radians(30.0)])
    path.turn(2.0 * math.pi, math.radians(30.0))
    path.go_to(-0.25 * a, 0.4 * b).go_to(-0.65 * a, 0.1 * b)
    return _scan(path.build("symmetry"))


def _kidnapped(far: bool) -> TrajectoryBuilder:
    def build(field: FieldGeometry, rng) -> Trajectory:
        a, b = field.half_length, field.half_width
        path = tj.PathBuilder([-0.6 * a, -0.5 * b, 0.0])
        path.forward(0.5 * a).turn(math.pi / 2).forward(0.6 * b)
        if far:
            path.teleport([0.55 * a, 0.5 * b, math.radians(-150)])
            path.go_to(0.2 * a, -0.35 * b).go_to(0.7 * a, -0.1 * b)
        else:
            path.teleport([-0.7 * a, 0.5 * b, math.radians(-60)])
            path.go_to(-0.3 * a, -0.35 * b).go_to(-0.75 * a, -0.1 * b)
        path.hold(5.0)
        return _scan(path.build("kidnapped"))
    return build


_HARSH_ORACLE = dict(landmark_dropout_prob=0.5, line_point_dropout_prob=0.5,
                     frame_dropout_prob=0.10, false_landmark_rate=1.5,
                     false_line_point_fraction=0.30, landmark_outlier_prob=0.10,
                     class_confusion_prob=0.10)


def _build_scenarios() -> list[VisionScenario]:
    field = FieldGeometry.load("s_field")
    return [
        VisionScenario(
            "v01_stationary", "Stationary robot",
            "The robot stands still for 20 s; the filter starts from an approximate pose.",
            _stationary),
        VisionScenario(
            "v02_rotation", "Rotation in place",
            "Two full turns on the spot at 30 deg/s.", _rotation),
        VisionScenario(
            "v03_straight_line", "Straight-line trajectory",
            "A straight walk along most of the field's length at 0.3 m/s.", _straight),
        VisionScenario(
            "v04_rectangle", "Rectangular trajectory",
            "One lap of a rectangle with turns in place at the corners.", _rectangle()),
        VisionScenario(
            "v05_curve", "Curved trajectory",
            "One lap of a circle around the centre mark, always turning left.", _curve),
        VisionScenario(
            "v06_partial_visibility", "Partial field visibility",
            "The camera is commanded to look 50 deg down, so it sees only the 0.2 to 2.4 m of "
            "turf in front of the robot: often a single line, sometimes nothing.", _rectangle(),
            commanded_pitch_deg=50.0,
            expectation="tracks, with larger error where nothing is in view"),
        VisionScenario(
            "v07_noisy_odometry", "Noisy, drifting odometry with lost updates",
            "Odometry errors four times nominal and 15 % of the odometry messages lost.",
            _rectangle(), odometry_scale=4.0, odometry_overrides=dict(missed_update_prob=0.15),
            expectation="tracks; vision has to correct the dead-reckoning drift"),
        VisionScenario(
            "v08_missing_false_features", "Missing and false-positive visual features",
            "The network predictions are damaged before use: 10 % of frames lost, half of the "
            "detections missed, 0.5 invented goal posts per frame, about one rectangle erased "
            "from the line mask and one false stroke painted into it per frame. The oracle arm "
            "uses Phase 1's harsh dropout / false-positive profile.", _rectangle(),
            corruption=VisionCorruption(), oracle_overrides=_HARSH_ORACLE,
            expectation="tracks with degraded accuracy"),
        VisionScenario(
            "v09_camera_pitch_error", "Camera pitch error",
            "The camera is really pitched 2 deg further down than the robot believes.",
            _rectangle(), mount_error=MountError(pitch_deg=2.0),
            expectation="tracks with a range bias: every projected point lands too far away"),
        VisionScenario(
            "v10_camera_height_error", "Camera height error",
            "The camera is really 5 cm higher than the robot believes (0.95 m, not 0.90 m).",
            _rectangle(), mount_error=MountError(height=0.05),
            expectation="tracks with a scale error of about 5 % in every range"),
        VisionScenario(
            "v11_goal_post_occlusion", "Goal-post truncation and occlusion",
            "The robot works in front of the +x goal while three other robots stand at the "
            "posts. Posts are cut by the image border or hidden at the foot for much of the run.",
            _goal_approach, robots=_goal_robots(field),
            expectation="tracks; posts fall back to bearing-only when the foot is not trusted"),
        VisionScenario(
            "v12a_symmetry_no_prior", "180-degree field symmetry, no prior",
            "Global initialisation with no side information.", _symmetry, init="global",
            expectation="ambiguous by construction: about half of the runs settle on the "
                        "mirrored pose, and nothing in the images can prevent it"),
        VisionScenario(
            "v12b_symmetry_own_half_prior", "180-degree field symmetry, own-half prior",
            "The same run, but the filter is told which half of the field it starts in. That "
            "is EXTERNAL game-state information, not something the camera provides.",
            _symmetry, init="own_half",
            expectation="the prior breaks the symmetry; should localize correctly"),
        VisionScenario(
            "v13a_kidnapped_same_side", "Kidnapped robot, moved within its side",
            "After walking for a while the robot is picked up and put down a few metres away "
            "on the same side of the field centre. Odometry reports nothing.",
            _kidnapped(far=False),
            expectation="detects the mismatch and recovers the true pose"),
        VisionScenario(
            "v13b_kidnapped_across_field", "Kidnapped robot, moved across the field",
            "As above, but the robot is put down on the other side of the field centre.",
            _kidnapped(far=True),
            expectation="recovers a pose that explains the images, but the same-side prior "
                        "picks the MIRRORED pose: a wrong-half failure that only outside "
                        "information can fix"),
        VisionScenario(
            "v14_strong_randomization", "Strong domain-randomization appearance",
            "The rectangle lap in a scene whose turf, lighting, shadows, backdrop and ball are "
            "drawn from the `strong` preset (a new look for every seed), with its photometric "
            "image effects (exposure, blur, noise, JPEG) applied to every frame.", _rectangle(),
            randomization="strong",
            expectation="tracks with degraded accuracy; the models were trained on `mild`"),
    ]


SCENARIOS: dict[str, VisionScenario] = {s.id: s for s in _build_scenarios()}


def get_scenario(scenario_id: str) -> VisionScenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario '{scenario_id}' (available: {sorted(SCENARIOS)})")
    return SCENARIOS[scenario_id]


def select_scenarios(patterns: list[str] | None) -> list[VisionScenario]:
    """Scenarios whose id starts with any of the given prefixes; all of them for None."""
    if not patterns:
        return list(SCENARIOS.values())
    chosen = [s for s in SCENARIOS.values() if any(s.id.startswith(p) for p in patterns)]
    if not chosen:
        raise KeyError(f"no scenario matches {patterns} (available: {sorted(SCENARIOS)})")
    return chosen
