"""The Phase 3 benchmark machinery: scenarios, the damage model, feature scoring against
ground truth, calibration-error arms, and the runner and report writer end to end.

The end-to-end tests render with MuJoCo but use a colour-threshold stand-in for the networks,
so they need no trained checkpoints (tests/phase3_helpers.py)."""
import dataclasses
import json
import math

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mujoco")

import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.geometry import ground_projection as gp  # noqa: E402
from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.camera import StereoCameraConfig  # noqa: E402
from robosoccer_perception.geometry.landmarks import LandmarkType  # noqa: E402
from robosoccer_perception.inference.types import Detection  # noqa: E402
from robosoccer_perception.perception.extractor import GeometricFeatureExtractor  # noqa: E402
from robosoccer_perception.perception.types import (  # noqa: E402
    CircleFeature, FeatureSet, GoalPostFeature, IntersectionFeature, LinePointFeatures,
    MeasurementUncertainty)
from robosoccer_perception.simulation import trajectories as tj  # noqa: E402
from robosoccer_perception.vision_benchmark import benchmark, corruption, runner, scenarios  # noqa: E402
from robosoccer_perception.vision_benchmark.camera_truth import (  # noqa: E402
    HeadJitter, MountError, TrueCameraModel)
from robosoccer_perception.vision_benchmark.feature_metrics import (  # noqa: E402
    FeatureScorer, FeatureStatistics, FrameTruth)

REQUIRED = {       # the brief's scenario list -> scenario ids
    "stationary": "v01_stationary", "rotation in place": "v02_rotation",
    "straight line": "v03_straight_line", "rectangle": "v04_rectangle", "curve": "v05_curve",
    "partial visibility": "v06_partial_visibility", "noisy odometry": "v07_noisy_odometry",
    "missing / false features": "v08_missing_false_features",
    "camera pitch error": "v09_camera_pitch_error",
    "camera height error": "v10_camera_height_error",
    "goal-post truncation / occlusion": "v11_goal_post_occlusion",
    "field symmetry": "v12a_symmetry_no_prior", "kidnapped": "v13a_kidnapped_same_side",
    "strong domain randomization": "v14_strong_randomization",
}


# ------------------------------------------------------------------ scenarios
def test_every_required_scenario_exists_and_says_what_it_changes():
    for name, scenario_id in REQUIRED.items():
        assert scenario_id in scenarios.SCENARIOS, name
    s = scenarios.SCENARIOS
    assert s["v06_partial_visibility"].commanded_pitch_deg == 50.0
    assert s["v07_noisy_odometry"].odometry_scale == 4.0
    assert s["v08_missing_false_features"].corruption is not None
    assert s["v09_camera_pitch_error"].mount_error == MountError(pitch_deg=2.0)
    assert s["v10_camera_height_error"].mount_error == MountError(height=0.05)
    assert len(s["v11_goal_post_occlusion"].robots) == 3
    assert s["v12a_symmetry_no_prior"].init == "global"
    assert s["v12b_symmetry_own_half_prior"].init == "own_half"
    assert s["v14_strong_randomization"].randomization == "strong"
    clean = [x for x in s.values() if x.mount_error == MountError() and x.corruption is None]
    assert len(clean) >= 10
    assert all(x.field == "s_field" and set(x.sources) == set(scenarios.ALL_SOURCES)
               for x in s.values())
    assert [x.id for x in scenarios.select_scenarios(["v12", "v13"])] == [
        "v12a_symmetry_no_prior", "v12b_symmetry_own_half_prior", "v13a_kidnapped_same_side",
        "v13b_kidnapped_across_field"]
    with pytest.raises(KeyError):
        scenarios.get_scenario("v99")
    with pytest.raises(ValueError):
        dataclasses.replace(s["v01_stationary"], init="psychic")


@pytest.mark.parametrize("scenario_id", sorted(scenarios.SCENARIOS))
def test_trajectories_are_reproducible_differ_by_seed_and_stay_on_the_carpet(s_field, scenario_id):
    scenario = scenarios.get_scenario(scenario_id)
    first, again, other = (runner.build_trajectory(scenario, seed) for seed in (0, 0, 1))
    np.testing.assert_array_equal(first.poses, again.poses)
    assert not np.array_equal(first.poses, other.poses), \
        "every seed sees the field from slightly different places"
    assert np.abs(first.poses[0, :2] - other.poses[0, :2]).max() <= 2 * scenario.start_shift
    for trajectory in (first, other):
        assert s_field.contains(trajectory.poses[:, :2], margin=0.2).all()
    kidnaps = first.kidnap_steps
    assert bool(kidnaps) == scenario_id.startswith("v13")
    if scenario.commanded_pitch_deg is not None:
        np.testing.assert_allclose(first.pitch, math.radians(scenario.commanded_pitch_deg))


def test_varying_a_trajectory_moves_it_rigidly(rng):
    trajectory = tj.PathBuilder([1.0, 0.5, 0.3]).forward(2.0).turn(1.0).forward(1.0).build("t")
    moved = scenarios.vary_trajectory(trajectory, rng, 0.2, 5.0)
    np.testing.assert_allclose(tf.se2_relative(moved.poses[0], moved.poses),
                               tf.se2_relative(trajectory.poses[0], trajectory.poses), atol=1e-9)
    assert 0.0 < np.linalg.norm(moved.poses[0, :2] - trajectory.poses[0, :2]) < 0.3
    assert abs(math.degrees(moved.poses[0, 2] - trajectory.poses[0, 2])) <= 5.0


# ------------------------------------------------------------------ damage to predictions
def _predictions():
    mask = np.full((90, 160), schema.FIELD, np.uint8)
    mask[40:44] = schema.FIELD_LINE
    detections = [Detection(schema.GOAL_POST, 0.9, (10.0, 5.0, 16.0, 60.0)),
                  Detection(schema.BALL, 0.8, (80.0, 50.0, 90.0, 60.0))] * 3
    return helpers.mask_to_vision(mask, detections)


def test_damage_is_reproducible_and_does_what_it_says():
    clean = _predictions()
    cfg = corruption.VisionCorruption(frame_drop_prob=0.0)
    a = corruption.corrupt(clean, cfg, np.random.default_rng(4))
    b = corruption.corrupt(clean, cfg, np.random.default_rng(4))
    assert a.detections == b.detections
    np.testing.assert_array_equal(a.segmentation.mask, b.segmentation.mask)
    assert clean.segmentation.mask[40:44].all() and len(clean.detections) == 6, "input untouched"
    lines_lost = lines_added = posts_added = dropped = 0
    for seed in range(40):
        damaged = corruption.corrupt(clean, cfg, np.random.default_rng(seed))
        mask = damaged.segmentation.mask
        np.testing.assert_allclose(damaged.segmentation.probabilities.sum(axis=0), 1.0, atol=1e-5)
        np.testing.assert_array_equal(mask, damaged.segmentation.probabilities.argmax(axis=0))
        lines_lost += int((mask[40:44] != schema.FIELD_LINE).any())
        outside = np.ones_like(mask, bool)
        outside[40:44] = False
        lines_added += int((mask[outside] == schema.FIELD_LINE).any())
        real = [d for d in damaged.detections if d in clean.detections]
        dropped += 6 - len(real)
        posts_added += len(damaged.detections) - len(real)
    assert lines_lost > 10 and lines_added > 10
    assert 80 < dropped < 160, "about half of 240 detections are missed"
    assert 8 < posts_added < 40, "about 0.5 invented posts per frame"
    gone = corruption.corrupt(clean, corruption.VisionCorruption(frame_drop_prob=1.0),
                              np.random.default_rng(0))
    assert gone.detections == () and gone.segmentation is None
    assert (gone.width, gone.height) == (clean.width, clean.height)


def test_damaged_vision_sees_only_the_image():
    vision = helpers.ColourVision()
    wrapped = corruption.CorruptedVision(vision, corruption.VisionCorruption(),
                                         np.random.default_rng(0))
    image = helpers.mask_to_image(_predictions().segmentation.mask)
    result = wrapped.run(image)
    assert vision.calls == 1 and (result.width, result.height) == (160, 90)


# ------------------------------------------------------------------ scoring against truth
@pytest.fixture(scope="module")
def scoring(s_field, s_landmarks):
    cfg = StereoCameraConfig.load("zed_mini_sim").with_resolution(640, 360)
    truth = TrueCameraModel(cfg, MountError(), HeadJitter.none()).sample()
    return FeatureScorer(s_field, s_landmarks, cfg), truth


def _uncertainty(range_std=0.1, bearing_std=0.02):
    return MeasurementUncertainty(range_std, bearing_std, np.eye(2) * 1e-3, np.eye(2) * 1e-3)


def _base(pose, points):
    return tf.se2_inverse_transform_points(np.asarray(pose, float), np.asarray(points, float))


def test_scoring_counts_true_wrong_class_and_false_junctions(scoring, s_field, s_landmarks):
    scorer, camera = scoring
    pose = np.array([2.0, -0.5, -0.3])
    tees = s_landmarks.positions(LandmarkType.T_INTERSECTION)
    ells = s_landmarks.positions(LandmarkType.L_INTERSECTION)
    seen_t = tees[np.argmin(np.linalg.norm(tees - [4.475, -1.475], axis=1))]
    seen_l = ells[np.argmin(np.linalg.norm(ells - [3.525, -1.475], axis=1))]

    def junction(kind, field_xy, shift=(0.0, 0.0)):
        position = _base(pose, [np.asarray(field_xy) + shift])[0]
        return IntersectionFeature(kind, position, _uncertainty(), 0.9)

    features = FeatureSet(intersections=(
        junction(LandmarkType.T_INTERSECTION, seen_t, (0.05, 0.0)),     # right, 5 cm off
        junction(LandmarkType.T_INTERSECTION, seen_l),                  # an L called a T
        junction(LandmarkType.L_INTERSECTION, [3.0, 0.7]),              # nothing is there
        junction(None, seen_l),                                         # unclassed
    ))
    score = scorer.score(features, FrameTruth(pose, camera))
    outcomes = [j["outcome"] for j in score["junctions"]]
    assert outcomes == ["true", "wrong_class", "false", "unclassed"]
    assert score["junctions"][0]["error"] == pytest.approx(0.05, abs=1e-6)
    assert score["junction_recalled"]["T"] == 1 and score["junction_recalled"]["L"] == 0
    assert score["junction_visible"]["T"] >= 1 and score["junction_visible"]["L"] >= 1
    assert score["junction_decidable"]["T"] <= score["junction_visible"]["T"]

    statistics = FeatureStatistics()
    statistics.add(score)
    summary = statistics.summary()["intersections"]
    assert summary["T"]["reported"] == 2 and summary["T"]["precision"] == 0.5
    assert summary["T"]["wrong_class"] == 1 and summary["L"]["false"] == 1
    assert summary["all"]["reported"] == 3 and summary["unclassed"] == 1
    assert summary["T"]["position_error_m"]["mean"] == pytest.approx(0.05, abs=1e-4)
    assert 0.0 < summary["T"]["recall"] <= 1.0


def test_scoring_measures_line_points_posts_and_the_circle(scoring, s_field, s_landmarks):
    scorer, camera = scoring
    pose = np.array([-2.0, 0.0, 0.0])
    on_paint = _base(pose, [[0.0, 0.5], [0.0, -0.5], [-0.3, 2.975]])
    off_paint = _base(pose, [[-1.0, 0.0]])                      # 0.275 m inside the circle
    points = np.vstack([on_paint, off_paint])
    lines = LinePointFeatures(np.zeros((4, 2)), points, np.full(4, 0.1), np.ones(4),
                              np.full(4, 0.05), np.full(4, 0.05), np.full((4, 2), np.nan))
    posts_field = s_landmarks.positions(LandmarkType.GOAL_POST)
    target = posts_field[np.argmax(posts_field[:, 0] + posts_field[:, 1])]
    exact = _base(pose, [target])[0]
    bearing = math.atan2(exact[1], exact[0]) + math.radians(1.0)
    features = FeatureSet(
        line_points=lines,
        goal_posts=(
            GoalPostFeature(np.zeros(2), exact + [0.2, 0.0], _uncertainty(0.3), 0.9, True),
            GoalPostFeature(np.zeros(2), 3.0 * np.array([math.cos(bearing), math.sin(bearing)]),
                            _uncertainty(2.0), 0.8, False, ("cut_bottom",)),
            GoalPostFeature(np.zeros(2), np.array([1.0, -3.0]), _uncertainty(), 0.5, True)),
        circles=(CircleFeature(_base(pose, [[0.04, 0.0]])[0], 0.725, _uncertainty(), 0.9, 300.0,
                               60, True),
                 CircleFeature(np.array([1.0, 2.5]), 0.725, _uncertainty(), 0.5, 160.0, 20, True),
                 CircleFeature(np.array([1.0, 1.0]), 0.725, _uncertainty(), 0.2, 90.0, 9, False)))
    score = scorer.score(features, FrameTruth(pose, camera))
    np.testing.assert_allclose(score["line_error"], [0.0, 0.0, 0.0, 0.25], atol=1e-6)
    ranged, bearing_only, stray = score["posts"]
    assert ranged["matched"] and ranged["position_error"] == pytest.approx(0.2, abs=1e-6)
    assert bearing_only["matched"] and bearing_only["bearing_error_deg"] == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(bearing_only["position_error"]) and not stray["matched"]
    assert score["circle"]["expected"] and score["circle"]["reported"] == 2, \
        "a circle candidate that failed its checks is not a report"
    assert score["circle"]["correct"] == 1 and score["circle"]["false"] == 1
    assert score["circle"]["errors"] == pytest.approx([0.04], abs=1e-6)

    statistics = FeatureStatistics()
    statistics.add(score)
    statistics.add(scorer.score(FeatureSet(), FrameTruth(pose, camera)))
    summary = statistics.summary()
    assert summary["frames"] == 2
    assert summary["line_points"]["error_m"]["count"] == 4
    assert summary["line_points"]["within_5cm"] == 0.75
    assert summary["line_points"]["within_2_sigma"] == 0.75
    assert summary["goal_posts"]["with_range"] == 2 and summary["goal_posts"]["bearing_only"] == 1
    assert summary["goal_posts"]["unmatched"] == 1 and summary["goal_posts"]["flags"] == {"cut_bottom": 1}
    assert summary["center_circle"]["frames_expected"] == 2
    assert summary["center_circle"]["found_when_expected"] == 1
    assert summary["center_circle"]["precision"] == 0.5
    assert summary["observations_per_frame"]["line_points"] == 2.0
    assert summary["observations_per_frame"]["frames_without_any_observation"] == 0.5
    assert FeatureStatistics().summary() == {"frames": 0}


def test_perfect_features_score_perfectly(scoring, s_field):
    """Extraction from an exact view, scored with the same camera: no error anywhere."""
    scorer, camera = scoring
    calibration = helpers.small_calibration(640, 360)
    extractor = GeometricFeatureExtractor(s_field)
    statistics = FeatureStatistics()
    for pose in ([-2.6, -1.4, 0.6], [2.0, -0.5, -0.3], [-2.0, 0.0, 0.0]):
        _, vision = helpers.analytic_view(s_field, pose, calibration, calibration.pose())
        features = extractor.extract(vision, calibration, calibration.pose())
        statistics.add(scorer.score(features, FrameTruth(np.array(pose), camera)))
    summary = statistics.summary()
    assert summary["line_points"]["error_m"]["p95"] < 0.02
    junctions = summary["intersections"]["all"]
    assert junctions["reported"] >= 6 and junctions["precision"] == 1.0
    assert junctions["position_error_m"]["mean"] < 0.03
    assert summary["center_circle"]["false"] == 0 and summary["center_circle"]["correct"] >= 2


# ------------------------------------------------------------------ calibration-error arms
def test_estimate_error_is_the_mirror_image_of_a_mount_error():
    """Estimate minus truth. A camera really 2 deg lower than believed (mount error +2) and a
    robot believing 2 deg less than the truth (estimate error -2) are the same mismatch."""
    cfg = StereoCameraConfig.load("zed_mini_sim")
    calibration = gp.EstimatedCameraCalibration.from_camera_config(cfg)
    nominal = calibration.pose()
    truth_a = TrueCameraModel(cfg, MountError(pitch_deg=2.0, height=0.05),
                              HeadJitter.none()).sample().true_T_base_camera()
    estimate_a = nominal.T_base_camera
    _, estimate_b = runner.EstimateError(pitch_deg=-2.0, height=-0.05).apply(calibration, nominal)
    truth_b = nominal.T_base_camera
    relative_a = tf.invert_transform(truth_a) @ estimate_a
    relative_b = tf.invert_transform(truth_b) @ estimate_b.T_base_camera
    np.testing.assert_allclose(relative_a[:3, :3], relative_b[:3, :3], atol=1e-12)
    # The 5 cm height offset is seen from frames that differ by 2 deg: equal to first order.
    np.testing.assert_allclose(relative_a[:3, 3], relative_b[:3, 3], atol=0.05 * math.sin(math.radians(2.0)) + 1e-9)

    same_calibration, same_pose = runner.EstimateError().apply(calibration, nominal)
    assert same_calibration is calibration and same_pose is nominal
    wrong, _ = runner.EstimateError(focal_scale=1.05, cx_shift=4.0).apply(calibration, nominal)
    assert wrong.intrinsics.fx == pytest.approx(1.05 * calibration.intrinsics.fx)
    assert wrong.intrinsics.cx == pytest.approx(calibration.intrinsics.cx + 4.0)


def test_sweep_arms_cover_every_value_once_and_share_the_nominal_arm():
    sweep = {"pitch_deg": (-1.0, 0.0, 1.0), "focal_scale": (0.95, 1.0)}
    arms, layout, belief = benchmark.sweep_arms(
        sweep, {"parameter": "pitch_deg", "pitch_std_deg": 2.0}, gp.CalibrationUncertainty())
    names = [arm.name for arm in arms]
    assert len(set(names)) == len(names) == 1 + 2 + 1 + 3
    assert all(arm.source == "vision" for arm in arms)
    assert [name for _, name in layout["pitch_deg"]] == ["pitch_deg=-1", "vision", "pitch_deg=+1"]
    assert layout["focal_scale"][1] == (1.0, "vision")
    by_name = {arm.name: arm for arm in arms}
    assert by_name["pitch_deg=+1"].estimate_error == runner.EstimateError(pitch_deg=1.0)
    assert by_name["focal=0.95"].estimate_error == runner.EstimateError(focal_scale=0.95)
    wide = [by_name[name] for _, name in belief["pitch_deg"]]
    assert [arm.estimate_error.pitch_deg for arm in wide] == [-1.0, 0.0, 1.0]
    assert all(arm.believed_uncertainty.pitch_std_deg == 2.0 for arm in wide)
    assert by_name["pitch_deg=+1"].believed_uncertainty is None


# ------------------------------------------------------------------ the runner, end to end
def _walk(field, rng):
    return tj.PathBuilder([-2.4, -1.2, 0.2]).forward(1.2, 0.3).turn(math.radians(50.0)) \
        .forward(0.6, 0.3).build("walk").with_head_scan(math.radians(30.0), 3.0)


SHORT = scenarios.VisionScenario("t_walk", "A short walk", "Test scenario.", _walk)


@pytest.fixture(scope="module")
def resources():
    return runner.BenchmarkResources(helpers.ColourVision(), resolution=(320, 180),
                                     filter_overrides={"num_particles": 300})


@pytest.fixture(scope="module")
def short_run(gl_context, resources):
    return runner.run_vision_scenario(SHORT, 0, resources, keep_frames="auto",
                                      snapshot_steps="auto")


@pytest.mark.render
@pytest.mark.slow
def test_a_run_gives_every_arm_the_same_world_and_scores_them_separately(short_run):
    run = short_run
    assert list(run.arms) == ["vision", "oracle", "odometry", "perfect_vision"]
    steps = len(run.arms["vision"].result.time)
    truth = run.arms["vision"].result.truth
    for name, arm in run.arms.items():
        result = arm.result
        np.testing.assert_array_equal(result.truth, truth)        # identical trajectory
        assert result.estimate.shape == (steps, 3) and result.scenario_id == "t_walk"
        assert np.isfinite(result.position_error).all()
    # Identical start: dead reckoning never looks, so its first estimate IS the initial guess
    # that every arm was given; it is off by a draw from the 0.3 m prior, not exact.
    guess_error = run.arms["odometry"].result.position_error[0]
    assert 0.02 < guess_error < 1.0
    final = {name: arm.result.position_error[-10:].mean() for name, arm in run.arms.items()}
    assert final["perfect_vision"] < 0.10 and final["oracle"] < 0.10
    assert final["vision"] < 0.25, "even a colour threshold localizes on a clean render"
    assert final["odometry"] > final["perfect_vision"]
    assert run.arms["odometry"].result.num_line_points.max() == 0
    assert run.arms["vision"].result.num_line_points.mean() > 10
    # Timing and feature scores belong to the image-fed arms only.
    assert run.arms["vision"].timing_ms is not None and run.arms["oracle"].timing_ms is None
    assert set(run.arms["vision"].timing_ms) == set(runner.TIMING_KEYS)
    assert run.arms["vision"].timing_ms["total"].shape == (steps,)
    assert run.arms["oracle"].features is None
    perfect = run.arms["perfect_vision"].features.summary()
    assert perfect["frames"] == steps and perfect["line_points"]["error_m"]["mean"] < 0.05
    assert perfect["intersections"]["all"]["precision"] in (None, 1.0)
    assert len(run.frames) == 3 and len(run.arms["vision"].result.snapshots) == 3
    record = next(iter(run.frames.values()))
    assert record.image.shape == (180, 320, 3)
    assert type(record.camera_pose) is gp.EstimatedCameraPose


@pytest.mark.render
@pytest.mark.slow
def test_runs_are_reproducible_and_seeds_differ(gl_context, resources, short_run):
    arms = (runner.Arm("vision", "vision"), runner.Arm("oracle", "oracle"))
    again = runner.run_vision_scenario(SHORT, 0, resources, arms)
    other = runner.run_vision_scenario(SHORT, 1, resources, arms)
    for name in ("vision", "oracle"):
        np.testing.assert_array_equal(again.arms[name].result.estimate,
                                      short_run.arms[name].result.estimate)
        assert not np.array_equal(other.arms[name].result.truth, again.arms[name].result.truth)
    with pytest.raises(ValueError, match="unique"):
        runner.run_vision_scenario(SHORT, 0, resources, (arms[0], arms[0]))


@pytest.mark.render
@pytest.mark.slow
def test_a_hidden_mount_error_hurts_the_image_fed_arms_and_not_the_oracle(gl_context, resources):
    off = dataclasses.replace(SHORT, id="t_pitch", mount_error=MountError(pitch_deg=1.5),
                              jitter=HeadJitter.none())
    exact = dataclasses.replace(SHORT, id="t_pitch", jitter=HeadJitter.none())
    wrong = runner.run_vision_scenario(off, 0, resources, (
        runner.Arm("perfect_vision", "perfect_vision"), runner.Arm("oracle", "oracle")))
    # The same mismatch expressed as an estimate error on an exact camera (the sweep's way).
    right = runner.run_vision_scenario(exact, 0, resources, (
        runner.Arm("perfect_vision", "perfect_vision"),
        runner.Arm("believes_less_pitch", "perfect_vision", runner.EstimateError(pitch_deg=-1.5))))

    def tail(run, name):
        return run.arms[name].result.position_error[-15:].mean()

    assert tail(right, "perfect_vision") < 0.05
    assert tail(wrong, "perfect_vision") > 4.0 * tail(right, "perfect_vision")
    assert tail(wrong, "oracle") < 0.05, "the oracle does not project pixels: it cannot be hurt"
    assert tail(right, "believes_less_pitch") == pytest.approx(tail(wrong, "perfect_vision"),
                                                               rel=0.35)


@pytest.mark.render
@pytest.mark.slow
def test_the_benchmark_and_the_sweep_write_machine_and_human_readable_results(gl_context,
                                                                             resources, tmp_path):
    pytest.importorskip("matplotlib")
    occluded = dataclasses.replace(
        SHORT, id="t_damaged", corruption=corruption.VisionCorruption(),
        robots=scenarios.SCENARIOS["v11_goal_post_occlusion"].robots)
    report = benchmark.run_vision_benchmark([SHORT, occluded], [0, 1], resources,
                                            tmp_path / "benchmark", log=lambda line: None)
    assert set(report["scenarios"]) == {"t_walk", "t_damaged"}
    entry = report["scenarios"]["t_walk"]
    assert set(entry["arms"]) == {"vision", "oracle", "odometry", "perfect_vision"}
    for arm in entry["arms"].values():
        assert arm["aggregate"]["runs"] == 2 and len(arm["runs"]) == 2
        assert {"position_rmse_all", "position_mae_all", "position_p95_all", "yaw_rmse_all",
                "yaw_mae_all", "yaw_p95_all", "convergence_rate", "divergence_rate",
                "wrong_half_rate", "recovery_rate"} <= set(arm["aggregate"])
    assert set(entry["features"]) == {"vision", "perfect_vision"}
    assert entry["features"]["vision"]["frames"] == 2 * entry["steps"]
    assert set(entry["timing_ms"]) == set(runner.TIMING_KEYS)
    assert report["scenarios"]["t_damaged"]["corruption"]["frame_drop_prob"] == 0.10
    assert "synthetic_only" in report["meta"] and report["meta"]["seeds"] == [0, 1]
    assert report["overall"]["timing_ms"]["total"]["count"] > 0

    output = tmp_path / "benchmark"
    for name in ("results.json", "summary.md", "summary.csv", "runs.csv", "features.csv",
                 "plots/t_walk_arms.png", "plots/t_walk_vision_trajectory.png",
                 "plots/t_walk_vision_particles.png", "plots/overview_position_rmse.png",
                 "timeseries/t_walk_vision_seed1.csv", "timeseries/t_damaged_oracle_seed0.csv"):
        assert (output / name).is_file(), name
    frames = sorted((output / "frames" / "t_walk").iterdir())
    assert len(frames) == 15, "3 kept frames x (rgb, predictions, features, ground, particles)"
    saved = json.loads((output / "results.json").read_text())
    assert saved["scenarios"]["t_walk"]["arms"]["oracle"]["aggregate"]["runs"] == 2
    summary = (output / "summary.md").read_text()
    assert "| t_walk | vision |" in summary and "| t_walk | odometry |" in summary
    assert "simulator" in summary and "Geometric perception" in summary
    assert len((output / "summary.csv").read_text().splitlines()) == 1 + 2 * 4

    sweep = benchmark.run_calibration_sweep(
        SHORT, [0], resources, tmp_path / "sweep",
        sweep={"pitch_deg": (-2.0, 0.0, 2.0), "height_m": (0.0, 0.08)},
        belief={"parameter": "pitch_deg", "pitch_std_deg": 2.0}, log=lambda line: None)
    points = sweep["parameters"]["pitch_deg"]["points"]
    assert [p["raw_value"] for p in points] == [-2.0, 0.0, 2.0]
    errors = [p["aggregate"]["position_rmse_all"] for p in points]
    assert errors[0] > errors[1] and errors[2] > errors[1], "worse on both sides of zero"
    assert len(sweep["parameters"]["pitch_deg"]["wider_belief"]["points"]) == 3
    assert "estimate minus the truth" in sweep["convention"]
    for name in ("sweep.json", "sweep.csv", "sensitivity.png"):
        assert (tmp_path / "sweep" / name).is_file(), name
    assert len((tmp_path / "sweep" / "sweep.csv").read_text().splitlines()) == 1 + 3 + 2 + 3
