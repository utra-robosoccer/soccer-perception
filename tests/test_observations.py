"""Oracle observations, their noise model, and synthetic odometry."""
import dataclasses
import math

import numpy as np
import pytest

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import RigState
from robosoccer_perception.geometry.distance_map import get_distance_map
from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.localization.motion_model import (
    MotionModelConfig, OdometryDelta, predict)
from robosoccer_perception.observations.noise import OracleConfig, apply_pitch_error
from robosoccer_perception.observations.oracle import OracleObservationGenerator
from robosoccer_perception.observations.types import (
    LandmarkObservation, LinePointsObservation, ObservationFrame)
from robosoccer_perception.simulation.odometry import OdometryNoiseConfig, SyntheticOdometry

PERFECT = dict(range_std_base=0.0, range_std_quadratic=0.0, bearing_std_deg=0.0,
               pitch_jitter_std_deg=0.0, landmark_dropout_prob=0.0, line_point_dropout_prob=0.0,
               frame_dropout_prob=0.0, false_landmark_rate=0.0, false_line_point_fraction=0.0,
               landmark_outlier_prob=0.0, class_confusion_prob=0.0)
RIG = RigState(height=0.9, pitch=math.radians(25), forward_offset=0.05)
FACING_GOAL = np.array([1.5, 0.0, 0.0])        # looking at the +x goal from 3 m


def make_oracle(s_field, s_landmarks, camera_model, **overrides):
    cfg = OracleConfig.load("nominal", {**PERFECT, **overrides})
    return OracleObservationGenerator(s_field, s_landmarks, camera_model, cfg)


def test_configs_load():
    nominal, clean = OracleConfig.load("nominal"), OracleConfig.load("clean")
    assert nominal.landmark_dropout_prob > 0 and clean.landmark_dropout_prob == 0
    assert nominal.range_std(4.0) > nominal.range_std(1.0) > 0
    with pytest.raises(ValueError):
        OracleConfig.from_dict({"landmark_dropout_prob": 1.5})
    with pytest.raises(ValueError):
        OracleConfig.from_dict({"not_a_key": 1})


def test_observation_types_carry_no_identity_or_global_pose():
    """The localizer's inputs must not be able to smuggle ground truth."""
    assert {f.name for f in dataclasses.fields(LandmarkObservation)} == {
        "type", "position", "range_std", "bearing_std", "confidence"}
    assert {f.name for f in dataclasses.fields(LinePointsObservation)} == {"points", "std"}
    assert {f.name for f in dataclasses.fields(ObservationFrame)} == {
        "timestamp", "landmarks", "line_points", "frame_id"}
    assert {f.name for f in dataclasses.fields(OdometryDelta)} == {
        "dx", "dy", "dyaw", "timestamp", "dt", "covariance"}


def test_noise_free_observations_are_exact_and_in_the_base_frame(s_field, s_landmarks,
                                                                 camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model)
    frame, debug = oracle.generate(FACING_GOAL, rng, RIG, timestamp=1.5)
    assert frame.timestamp == 1.5 and frame.frame_id == "base_footprint"
    assert len(frame.landmarks) == len(debug.landmark_ids) >= 6
    for observation, landmark_id in zip(frame.landmarks, debug.landmark_ids):
        truth = s_landmarks.by_id(landmark_id)
        assert observation.type == truth.type
        in_field = tf.se2_transform_points(FACING_GOAL, observation.position[None])[0]
        np.testing.assert_allclose(in_field, truth.position, atol=1e-9)
        assert observation.position[0] > 0, "everything seen is in front of the robot"
        assert observation.range == pytest.approx(np.hypot(*observation.position))
    seen = {observation.type for observation in frame.landmarks}
    assert LandmarkType.GOAL_POST in seen and LandmarkType.PENALTY_MARK in seen
    posts = [o for o in frame.landmarks if o.type == LandmarkType.GOAL_POST]
    assert len(posts) == 2 and posts[0].bearing * posts[1].bearing < 0, "one post on each side"


def test_field_of_view_limits_what_is_seen(s_field, s_landmarks, camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model)
    k = camera_model.intrinsics
    for pose in ([0, 0, 0.3], [-3, 1, 2.0], [2, -2, -1.0], FACING_GOAL):
        frame, _ = oracle.generate(pose, rng, RIG)
        # Everything reported re-projects inside the left image (minus the detection margin).
        seen = [o.position for o in frame.landmarks if o.type != LandmarkType.CENTER_MARK]
        on_ground = tf.se2_transform_points(np.asarray(pose, float),
                                            np.vstack([*seen, frame.line_points.points]))
        pixels, depth, visible = camera_model.project_field_points(
            pose, np.column_stack([on_ground, np.zeros(len(on_ground))]), "left", RIG)
        assert len(on_ground) > 10 and visible.all() and np.all(depth > 0)
        assert pixels[:, 0].min() >= 3.5 and pixels[:, 0].max() <= k.width - 4.5
        assert pixels[:, 1].min() >= 3.5 and pixels[:, 1].max() <= k.height - 4.5
    away, _ = oracle.generate([1.5, 0.0, math.pi], rng, RIG)       # back to the +x goal
    ids_away = {o.type for o in away.landmarks}
    toward, debug = oracle.generate(FACING_GOAL, rng, RIG)
    assert all("goal_pos_x" in i or "pos_x" in i for i in debug.landmark_ids)
    assert LandmarkType.CENTER_MARK in ids_away


def test_head_pan_changes_the_view(s_field, s_landmarks, camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model)
    ahead, _ = oracle.generate([0, -2.5, 0.0], rng, RIG)
    left, _ = oracle.generate([0, -2.5, 0.0], rng, RIG.replace(pan=math.radians(90)))
    assert LandmarkType.CENTER_MARK not in {o.type for o in ahead.landmarks}
    assert LandmarkType.CENTER_MARK in {o.type for o in left.landmarks}


def test_range_limits(s_field, s_landmarks, camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model, max_range=2.0, min_range=1.0)
    frame, _ = oracle.generate([-4.0, 0.0, 0.0], rng, RIG)
    ranges = [o.range for o in frame.landmarks if o.type != LandmarkType.CENTER_MARK]
    line_ranges = np.hypot(*frame.line_points.points.T)
    assert ranges and all(1.0 - 1e-9 <= r <= 2.0 + 1e-9 for r in ranges)
    assert line_ranges.min() >= 1.0 - 1e-9 and line_ranges.max() <= 2.0 + 1e-9


def test_symmetric_poses_give_the_same_observations(s_field, s_landmarks, camera_model):
    """Nothing in an observation tells the two halves of the field apart."""
    oracle = make_oracle(s_field, s_landmarks, camera_model)
    for pose in ([1.5, 0.4, 0.2], [-2.0, 1.0, -2.0], [3.0, -2.0, 2.5]):
        mirrored = [-pose[0], -pose[1], pose[2] + math.pi]
        a, _ = oracle.generate(pose, np.random.default_rng(3), RIG)
        b, _ = oracle.generate(mirrored, np.random.default_rng(3), RIG)
        key = lambda frame: sorted((o.type.value, *np.round(o.position, 6)) for o in frame.landmarks)  # noqa: E731
        assert key(a) == key(b) and len(a.landmarks) > 0
        points_a = a.line_points.points[np.lexsort(np.round(a.line_points.points, 6).T)]
        points_b = b.line_points.points[np.lexsort(np.round(b.line_points.points, 6).T)]
        assert len(points_a) == len(points_b)


def test_line_points(s_field, s_landmarks, camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model, max_line_points=40)
    frame, debug = oracle.generate([-1.0, -0.5, 0.4], rng, RIG)
    points = frame.line_points
    assert len(points) == 40 == debug.num_true_line_points and points.std.shape == (40,)
    in_field = tf.se2_transform_points(np.array([-1.0, -0.5, 0.4]), points.points)
    assert np.all(s_field.distance_to_paint(in_field) < 1e-9), "noise-free points lie on paint"
    assert get_distance_map(s_field).lookup(in_field).max() < 0.01

    noisy = make_oracle(s_field, s_landmarks, camera_model, range_std_base=0.02,
                        range_std_quadratic=0.01, bearing_std_deg=1.0,
                        false_line_point_fraction=0.5)
    frame, debug = noisy.generate([-1.0, -0.5, 0.4], rng, RIG)
    assert debug.num_false_line_points > 5
    assert len(frame.line_points) == debug.num_true_line_points + debug.num_false_line_points
    in_field = tf.se2_transform_points(np.array([-1.0, -0.5, 0.4]), frame.line_points.points)
    distance = s_field.distance_to_paint(in_field)
    assert 0.0 < np.median(distance) < 0.15 and distance.max() > 0.3


def test_measurement_noise_grows_with_range(s_field, s_landmarks, camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model, range_std_base=0.02,
                         range_std_quadratic=0.01, bearing_std_deg=1.0)
    errors = {"near": [], "far": []}
    bearing_errors = []
    for _ in range(400):
        frame, debug = oracle.generate(FACING_GOAL, rng, RIG)
        for observation, landmark_id in zip(frame.landmarks, debug.landmark_ids):
            truth = tf.se2_inverse_transform_points(
                FACING_GOAL, np.asarray(s_landmarks.by_id(landmark_id).position)[None])[0]
            true_range = np.hypot(*truth)
            bucket = "near" if true_range < 2.0 else "far" if true_range > 2.8 else None
            if bucket:
                errors[bucket].append((observation.range - true_range,
                                       oracle.cfg.range_std(true_range)))
            bearing_errors.append(observation.bearing - math.atan2(truth[1], truth[0]))
            assert observation.range_std == pytest.approx(
                float(oracle.cfg.range_std(observation.range)))
    near, far = np.array(errors["near"]), np.array(errors["far"])
    assert far[:, 0].std() > 1.5 * near[:, 0].std()
    for bucket in (near, far):                     # normalised errors have unit variance
        assert (bucket[:, 0] / bucket[:, 1]).std() == pytest.approx(1.0, abs=0.12)
    assert np.std(bearing_errors) == pytest.approx(math.radians(1.0), rel=0.1)


def test_pitch_error_model():
    ranges = np.array([1.0, 3.0, 6.0])
    np.testing.assert_allclose(apply_pitch_error(ranges, 0.9, 0.0), ranges)
    shifted = apply_pitch_error(ranges, 0.9, math.radians(0.5))
    assert np.all(shifted < ranges), "assuming the camera looks further down pulls points in"
    assert np.all(np.diff(ranges - shifted) > 0), "far points move more"


def test_dropout_false_positives_and_outliers(s_field, s_landmarks, camera_model, rng):
    baseline, _ = make_oracle(s_field, s_landmarks, camera_model).generate(FACING_GOAL, rng, RIG)
    full = len(baseline.landmarks)

    none = make_oracle(s_field, s_landmarks, camera_model, landmark_dropout_prob=1.0,
                       line_point_dropout_prob=1.0)
    frame, _ = none.generate(FACING_GOAL, rng, RIG)
    assert frame.is_empty

    half = make_oracle(s_field, s_landmarks, camera_model, landmark_dropout_prob=0.5)
    counts = [len(half.generate(FACING_GOAL, rng, RIG)[0].landmarks) for _ in range(300)]
    assert np.mean(counts) == pytest.approx(0.5 * full, rel=0.1)

    lost = make_oracle(s_field, s_landmarks, camera_model, frame_dropout_prob=0.3)
    dropped = [lost.generate(FACING_GOAL, rng, RIG)[1].frame_dropped for _ in range(500)]
    assert np.mean(dropped) == pytest.approx(0.3, abs=0.06)

    spurious = make_oracle(s_field, s_landmarks, camera_model, false_landmark_rate=2.0)
    false_counts = [sum(i is None for i in spurious.generate(FACING_GOAL, rng, RIG)[1].landmark_ids)
                    for _ in range(300)]
    assert np.mean(false_counts) == pytest.approx(2.0, rel=0.15)

    gross = make_oracle(s_field, s_landmarks, camera_model, landmark_outlier_prob=1.0)
    frame, debug = gross.generate(FACING_GOAL, rng, RIG)
    errors = [np.linalg.norm(tf.se2_transform_points(FACING_GOAL, o.position[None])[0]
                             - s_landmarks.by_id(i).position)
              for o, i in zip(frame.landmarks, debug.landmark_ids)]
    assert np.median(errors) > 0.2

    confused = make_oracle(s_field, s_landmarks, camera_model, class_confusion_prob=1.0)
    frame, debug = confused.generate(FACING_GOAL, rng, RIG)
    for observation, landmark_id in zip(frame.landmarks, debug.landmark_ids):
        truth = s_landmarks.by_id(landmark_id).type
        if truth in (LandmarkType.L_INTERSECTION, LandmarkType.T_INTERSECTION):
            assert observation.type != truth


def test_occlusion(s_field, s_landmarks, camera_model, rng):
    oracle = make_oracle(s_field, s_landmarks, camera_model)
    clear, _ = oracle.generate(FACING_GOAL, rng, RIG)
    # A robot-sized cylinder 1 m straight ahead hides what is behind it, not what is beside it.
    blocker = [[2.5, 0.0, 0.25, 1.5]]
    blocked, debug = oracle.generate(FACING_GOAL, rng, RIG, extra_occluders=blocker)
    assert LandmarkType.PENALTY_MARK in {o.type for o in clear.landmarks}
    assert LandmarkType.PENALTY_MARK not in {o.type for o in blocked.landmarks}
    assert LandmarkType.GOAL_POST in {o.type for o in blocked.landmarks}
    assert len(blocked.line_points) < len(clear.line_points) or len(clear.line_points) == 60
    in_field = tf.se2_transform_points(FACING_GOAL, blocked.line_points.points)
    behind = (in_field[:, 0] > 2.8) & (np.abs(in_field[:, 1]) < 0.2)
    assert not behind.any()
    # A low ball does not hide the far goal posts from a camera 0.9 m up.
    ball, _ = oracle.generate(FACING_GOAL, rng, RIG, ball_position=[2.2, 0.95 * 0.7])
    assert sum(o.type == LandmarkType.GOAL_POST for o in ball.landmarks) == 2
    # With occlusion switched off the blocker changes nothing.
    off = make_oracle(s_field, s_landmarks, camera_model, occlusion=False)
    frame, _ = off.generate(FACING_GOAL, rng, RIG, extra_occluders=blocker)
    assert len(frame.landmarks) == len(clear.landmarks)


def test_oracle_is_reproducible(s_field, s_landmarks, camera_model):
    oracle = OracleObservationGenerator(s_field, s_landmarks, camera_model,
                                        OracleConfig.load("nominal"))
    a, _ = oracle.generate(FACING_GOAL, np.random.default_rng(11), RIG)
    b, _ = oracle.generate(FACING_GOAL, np.random.default_rng(11), RIG)
    c, _ = oracle.generate(FACING_GOAL, np.random.default_rng(12), RIG)
    np.testing.assert_array_equal(a.line_points.points, b.line_points.points)
    assert [o.type for o in a.landmarks] == [o.type for o in b.landmarks]
    assert not np.array_equal(a.line_points.points, c.line_points.points)


# ------------------------------------------------------------------ odometry
EXACT_ODOMETRY = OdometryNoiseConfig().scaled(0.0)


def test_exact_odometry_is_the_true_relative_motion(rng):
    odometry = SyntheticOdometry(EXACT_ODOMETRY, rng)
    start, end = np.array([1.0, 2.0, math.pi / 2]), np.array([1.0, 3.0, math.pi])
    delta = odometry.measure(start, end, timestamp=0.1, dt=0.1)
    # Facing +y and moving +y is 1 m forward; the turn is +90 deg.
    np.testing.assert_allclose(delta.as_array(), [1.0, 0.0, math.pi / 2], atol=1e-12)
    np.testing.assert_allclose(tf.se2_compose(start, delta.as_array()), [1.0, 3.0, -math.pi],
                               atol=1e-9)
    assert delta.covariance.shape == (3, 3) and delta.dt == 0.1


def test_odometry_does_not_see_a_kidnapping(rng):
    odometry = SyntheticOdometry(EXACT_ODOMETRY, rng)
    delta = odometry.measure([0, 0, 0], [3, 2, 1], timestamp=0.1, dt=0.1, kidnapped=True)
    np.testing.assert_allclose(delta.as_array(), 0.0)


def test_noisy_odometry_drifts_and_is_reproducible():
    cfg = OdometryNoiseConfig()
    poses = np.column_stack([np.linspace(0, 6, 201), np.zeros(201), np.zeros(201)])

    def integrate(seed):
        odometry = SyntheticOdometry(cfg, np.random.default_rng(seed))
        estimate = poses[0].copy()
        for i in range(1, len(poses)):
            estimate = tf.se2_compose(estimate, odometry.measure(poses[i - 1], poses[i],
                                                                 i * 0.1, 0.1).as_array())
        return estimate, odometry

    first, odometry = integrate(5)
    again, _ = integrate(5)
    np.testing.assert_array_equal(first, again)
    finals = np.array([integrate(seed)[0] for seed in range(30)])
    drift = np.linalg.norm(finals[:, :2] - poses[-1, :2], axis=1)
    assert 0.02 < drift.mean() < 1.0, "dead reckoning alone drifts"
    assert odometry.scale_bias != 0.0
    assert np.std(finals[:, 0]) > 0.05, "the scale bias differs from run to run"


def test_missed_odometry_updates(rng):
    odometry = SyntheticOdometry(OdometryNoiseConfig(missed_update_prob=0.4), rng)
    results = [odometry.measure([0, 0, 0], [0.03, 0, 0], 0.1, 0.1) for _ in range(500)]
    assert np.mean([r is None for r in results]) == pytest.approx(0.4, abs=0.06)


# ------------------------------------------------------------------ motion model
def test_predict_without_noise_is_exact_composition(rng):
    quiet = MotionModelConfig(translation_noise=0, rotation_noise=0, translation_from_rotation=0,
                              rotation_from_translation_deg=0, min_translation_std=0,
                              min_rotation_std_deg=0)
    particles = rng.uniform([-4, -3, -math.pi], [4, 3, math.pi], size=(50, 3))
    delta = OdometryDelta(0.3, -0.1, 0.5)
    moved = predict(particles, delta, quiet, rng)
    np.testing.assert_allclose(moved, tf.se2_compose(particles, delta.as_array()), atol=1e-12)
    assert moved is not particles


def test_predict_spreads_particles_and_wraps_yaw(rng):
    cfg = MotionModelConfig()
    particles = np.tile([0.0, 0.0, math.pi - 0.01], (5000, 1))
    moved = predict(particles, OdometryDelta(1.0, 0.0, 0.5), cfg, rng)
    assert np.all((moved[:, 2] >= -math.pi) & (moved[:, 2] < math.pi))
    assert moved[:, 0].std() > 0.05 and moved[:, 1].std() > 0.05
    assert abs(tf.angle_difference(tf.circular_mean(moved[:, 2]), math.pi - 0.01 + 0.5)) < 0.02
    still = predict(particles, OdometryDelta(0.0, 0.0, 0.0), cfg, rng)
    assert 0 < still[:, 0].std() < 0.01, "a standing robot's cloud barely diffuses"
    blind = predict(particles, None, cfg, rng)
    assert blind[:, 0].std() > 3 * still[:, 0].std(), "missing odometry: assume it may have moved"
