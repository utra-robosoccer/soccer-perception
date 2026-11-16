"""The benchmark scenarios. Every scenario is deterministic given a seed.

A scenario fixes the field, the true trajectory, how the filter is initialised and how bad the
sensing and the odometry are. Trajectories are sized relative to the field, so the same scenario
definition works on any field configuration.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import numpy as np

from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.simulation import trajectories as tj
from robosoccer_perception.simulation.trajectories import Trajectory

TrajectoryBuilder = Callable[[FieldGeometry, np.random.Generator], Trajectory]

#: How the filter is started.
#:   gaussian  around a guess that is itself off by a draw from the initial uncertainty
#:   global    uniform over the whole carpet, any heading (no prior at all)
#:   own_half  uniform over the half of the field the robot is really in (a side prior)
#:   mirrored  gaussian around the half-turn mirror image of the true pose (a wrong prior)
INIT_MODES = ("gaussian", "global", "own_half", "mirrored")


@dataclasses.dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    description: str
    build_trajectory: TrajectoryBuilder
    field: str = "s_field"
    init: str = "gaussian"
    init_position_std: float = 0.30
    init_yaw_std_deg: float = 10.0
    oracle_profile: str = "nominal"
    oracle_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    odometry_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    odometry_scale: float = 1.0                 # multiplies every odometry error source
    filter_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    expectation: str = "should localize"        # stated up front, printed next to the result

    def __post_init__(self):
        if self.init not in INIT_MODES:
            raise ValueError(f"init must be one of {INIT_MODES}")


def _scan(trajectory: Trajectory) -> Trajectory:
    """Look left and right while moving, +-50 degrees every 4 s."""
    return trajectory.with_head_scan(math.radians(50.0), 4.0)


def _loop(fraction: float = 0.55, speed: float = 0.3, laps: int = 1) -> TrajectoryBuilder:
    """A rectangular lap covering ``fraction`` of the field's length and width."""
    def build(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
        size_x, size_y = fraction * field.dims.length, fraction * field.dims.width
        return tj.rectangle([-size_x / 2, -size_y / 2, 0.0], size_x, size_y, laps, speed)
    return build


def _stationary(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
    pose = [-0.22 * field.dims.length, -0.17 * field.dims.width, math.radians(20.0)]
    return tj.stationary(pose, duration=30.0)


def _rotation(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
    pose = [0.11 * field.dims.length, 0.08 * field.dims.width, 0.0]
    return tj.rotate_in_place(pose, turns=2.0, rate=math.radians(30.0))


def _straight(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
    start = [-0.39 * field.dims.length, -0.25 * field.dims.width, 0.0]
    return tj.straight_line(start, 0.78 * field.dims.length, speed=0.3)


def _circle(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
    radius = 0.33 * field.dims.width
    return tj.circle([0.0, -radius, 0.0], radius, turns=1.0, speed=0.3)


def _random_start(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
    """Start anywhere, look around (one full turn), then visit random points."""
    bounds = (field.half_length - 0.5, field.half_width - 0.5)
    path = tj.PathBuilder(tj.random_pose(bounds, rng))
    path.turn(2.0 * math.pi, math.radians(30.0))
    for _ in range(4):
        x, y = rng.uniform([-bounds[0], -bounds[1]], [bounds[0], bounds[1]])
        path.go_to(float(x), float(y), speed=0.3)
    return _scan(path.build("random_start"))


def _kidnapped(far: bool) -> TrajectoryBuilder:
    def build(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
        a, b = field.half_length, field.half_width
        path = tj.PathBuilder([-0.6 * a, -0.5 * b, 0.0])
        path.forward(0.5 * a).turn(math.pi / 2).forward(0.6 * b)
        # Picked up and put down elsewhere: on the same side of the centre, or across the field.
        if far:
            path.teleport([0.55 * a, 0.5 * b, math.radians(-150)])
            path.go_to(0.2 * a, -0.35 * b).go_to(0.7 * a, -0.1 * b)
        else:
            path.teleport([-0.7 * a, 0.5 * b, math.radians(-60)])
            path.go_to(-0.3 * a, -0.35 * b).go_to(-0.75 * a, -0.1 * b)
        path.hold(5.0)
        return _scan(path.build("kidnapped"))
    return build


def _symmetry(field: FieldGeometry, rng: np.random.Generator) -> Trajectory:
    """A fixed start in the -x half, a look around, then a walk that stays in that half."""
    a, b = field.half_length, field.half_width
    path = tj.PathBuilder([-0.45 * a, -0.35 * b, math.radians(30.0)])
    path.turn(2.0 * math.pi, math.radians(30.0))
    path.go_to(-0.25 * a, 0.4 * b).go_to(-0.65 * a, 0.1 * b)
    return _scan(path.build("symmetry"))


_HARSH_PERCEPTION = dict(landmark_dropout_prob=0.5, line_point_dropout_prob=0.5,
                         frame_dropout_prob=0.10, false_landmark_rate=1.5,
                         false_line_point_fraction=0.30, landmark_outlier_prob=0.10,
                         class_confusion_prob=0.10)


def _build_scenarios() -> list[Scenario]:
    return [
        Scenario("s01_stationary", "Stationary at a known pose",
                 "The robot stands still; the filter starts from an approximate pose.",
                 _stationary),
        Scenario("s02_rotation", "Rotation in place",
                 "Two full turns on the spot at 30 deg/s, starting from an approximate pose.",
                 _rotation),
        Scenario("s03_straight_line", "Straight-line motion",
                 "A straight walk along most of the field's length at 0.3 m/s.", _straight),
        Scenario("s04_rectangle", "Rectangular trajectory",
                 "One lap of a rectangle with turns in place at the corners.", _loop()),
        Scenario("s05_circle", "Circular trajectory",
                 "One lap of a circle around the centre mark, always turning left.", _circle),
        Scenario("s06_random_start_global", "Random start, global initialisation",
                 "A random true start pose; the filter starts uniform over the whole carpet "
                 "with no side information. The robot looks around, then walks.",
                 _random_start, init="global",
                 expectation="localizes up to the field's half-turn symmetry; about half of "
                             "the runs are expected to settle on the mirrored pose"),
        Scenario("s07_partial_visibility", "Partial field visibility",
                 "Perception only reaches 2.5 m, so the robot often sees a single line or "
                 "nothing at all.", _loop(),
                 oracle_overrides=dict(max_range=2.5),
                 expectation="tracks, with larger error where nothing is in view"),
        Scenario("s08_noisy_odometry", "Noisy, drifting odometry with lost updates",
                 "Odometry errors four times nominal (so about 12 % scale drift) and 15 % of "
                 "the odometry messages lost.", _loop(),
                 odometry_scale=4.0, odometry_overrides=dict(missed_update_prob=0.15),
                 expectation="tracks; vision has to correct the dead-reckoning drift"),
        Scenario("s09_dropout_false_positives", "Landmark dropout and false positives",
                 "Half of all detections missed, 1.5 spurious landmarks per frame, 30 % "
                 "spurious line points, 10 % gross outliers, 10 % class mix-ups, 10 % of "
                 "frames lost.", _loop(), oracle_overrides=_HARSH_PERCEPTION,
                 expectation="tracks with degraded accuracy"),
        Scenario("s10a_kidnapped_same_side", "Kidnapped robot, moved within its side",
                 "After walking for a while the robot is picked up and put down a few metres "
                 "away, on the same side of the field centre. Odometry reports nothing.",
                 _kidnapped(far=False),
                 expectation="detects the mismatch and recovers the true pose"),
        Scenario("s10b_kidnapped_across_field", "Kidnapped robot, moved across the field",
                 "As above, but the robot is put down on the other side of the field centre.",
                 _kidnapped(far=True),
                 expectation="recovers a pose that explains the observations, but the "
                             "same-side prior picks the MIRRORED pose: a wrong-half failure "
                             "that only outside information can fix"),
        Scenario("s11a_symmetry_no_prior", "180-degree ambiguity, no prior",
                 "Global initialisation with no side information from a fixed start pose.",
                 _symmetry, init="global",
                 expectation="ambiguous by construction: about half of the runs settle on "
                             "the mirrored pose"),
        Scenario("s11b_symmetry_own_half_prior", "180-degree ambiguity, own-half prior",
                 "The same run, but the filter is told which half of the field it starts in.",
                 _symmetry, init="own_half",
                 expectation="the prior breaks the symmetry; should localize correctly"),
        Scenario("s11c_symmetry_wrong_prior", "180-degree ambiguity, wrong prior",
                 "The same run, but the filter is initialised at the mirrored pose.",
                 _symmetry, init="mirrored",
                 expectation="stays confidently on the mirrored pose; the filter cannot know, "
                             "the benchmark must flag it as a wrong-half failure"),
        Scenario("s12_field_s", "Field switching: S field",
                 "One rectangular lap at 0.5 m/s on the S field.", _loop(0.5, 0.5),
                 field="s_field"),
        Scenario("s12_field_m", "Field switching: M field",
                 "The same scenario definition on the M field, selected by configuration only.",
                 _loop(0.5, 0.5), field="m_field"),
        Scenario("s12_field_l", "Field switching: L field",
                 "The same scenario definition on the L field, selected by configuration only.",
                 _loop(0.5, 0.5), field="l_field",
                 expectation="tracks; on a 22 m field much of the walk has few lines within "
                             "the 8 m perception range"),
    ]


SCENARIOS: dict[str, Scenario] = {scenario.id: scenario for scenario in _build_scenarios()}


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario '{scenario_id}' (available: {sorted(SCENARIOS)})")
    return SCENARIOS[scenario_id]


def select_scenarios(patterns: list[str] | None) -> list[Scenario]:
    """Scenarios whose id starts with any of the given prefixes; all of them for None."""
    if not patterns:
        return list(SCENARIOS.values())
    chosen = [s for s in SCENARIOS.values() if any(s.id.startswith(p) for p in patterns)]
    if not chosen:
        raise KeyError(f"no scenario matches {patterns} (available: {sorted(SCENARIOS)})")
    return chosen
