"""The complete pipeline, image in and pose out, over many frames: tracking, field symmetry,
the effect of a wrong camera estimate, and the semantics of `step`.

No MuJoCo and no trained networks: frames are painted analytically and a colour-threshold
stand-in plays the networks (tests/phase3_helpers.py). The tests with the real models are in
tests/test_vision_localization_models.py."""
import math

import numpy as np
import pytest

pytest.importorskip("cv2")

import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.localization.estimation import PoseEstimate, mirror_pose  # noqa: E402
from robosoccer_perception.localization.motion_model import OdometryDelta  # noqa: E402
from robosoccer_perception.localization.particle_filter import (  # noqa: E402
    ParticleFilterConfig, UpdateInfo)
from robosoccer_perception.observations.types import ObservationFrame  # noqa: E402
from robosoccer_perception.perception.types import FeatureSet  # noqa: E402
from robosoccer_perception.pipeline.vision_localization import (  # noqa: E402
    VisionLocalizationInput, VisionLocalizationOutput, VisionLocalizationPipeline)
from robosoccer_perception.simulation import trajectories as tj  # noqa: E402
from robosoccer_perception.simulation.odometry import (  # noqa: E402
    OdometryNoiseConfig, SyntheticOdometry)

FAST = ParticleFilterConfig.load("particle_filter", {"num_particles": 400})


@pytest.fixture(scope="module")
def calibration():
    return helpers.small_calibration()


def _pipeline(field, seed=0, config=FAST):
    return VisionLocalizationPipeline(helpers.ColourVision(), field, filter_config=config,
                                      seed=seed)


def _drive(pipeline, field, trajectory, calibration, *, true_camera=None, believed=None,
           odometry_seed=1, start=0):
    """Feed a trajectory to the pipeline as IMAGES. The true poses are used here to paint the
    frames and to make odometry; the pipeline gets neither."""
    true_camera = true_camera or calibration.pose()
    believed = believed or calibration.pose()
    odometry = SyntheticOdometry(OdometryNoiseConfig(), np.random.default_rng(odometry_seed))
    outputs = []
    for step in range(start, len(trajectory)):
        pose = trajectory.poses[step]
        image, _ = helpers.analytic_view(field, pose, calibration, true_camera)
        delta = None if step == start else odometry.measure(
            trajectory.poses[step - 1], pose, step * trajectory.dt, trajectory.dt)
        outputs.append(pipeline.step(image, step * trajectory.dt, delta, calibration, believed))
    return outputs


def _errors(outputs, trajectory, start=0):
    estimate = np.array([o.pose for o in outputs])
    truth = trajectory.poses[start:start + len(outputs)]
    return (np.linalg.norm(estimate[:, :2] - truth[:, :2], axis=1),
            np.degrees(np.abs(tf.angle_difference(estimate[:, 2], truth[:, 2]))))


# ------------------------------------------------------------------ tracking
def test_the_pipeline_tracks_a_walk_from_images_and_odometry_alone(s_field, calibration):
    trajectory = tj.PathBuilder([-2.6, -1.4, 0.3]).forward(1.5, 0.3).turn(math.radians(60)) \
        .forward(0.6, 0.3).build("walk")
    pipeline = _pipeline(s_field)
    pipeline.init_gaussian(trajectory.poses[0] + [0.25, -0.2, math.radians(8.0)], 0.3,
                           math.radians(10.0))
    outputs = _drive(pipeline, s_field, trajectory, calibration)
    position, yaw = _errors(outputs, trajectory)
    assert position[0] < 0.45, "it starts from the wrong guess it was given"
    assert position[-20:].max() < 0.10 and yaw[-20:].max() < 3.0
    assert pipeline.steps == len(trajectory) and pipeline.vision.calls == len(trajectory)

    out = outputs[-1]
    assert type(out) is VisionLocalizationOutput
    assert type(out.estimate) is PoseEstimate and type(out.update) is UpdateInfo
    assert type(out.observations) is ObservationFrame and type(out.features) is FeatureSet
    assert out.estimate.covariance.shape == (3, 3) and out.estimate.position_std < 0.2
    assert out.estimate.converged and not out.estimate.symmetric_ambiguity
    assert set(out.timing_ms) == {"detector", "segmenter", "geometry", "filter", "total"}
    assert out.timing_ms["total"] >= out.timing_ms["geometry"] + out.timing_ms["filter"] > 0
    assert len(out.observations.line_points) == len(out.features.line_points) > 5
    np.testing.assert_array_equal(out.pose, out.estimate.pose)


def test_vision_corrects_what_dead_reckoning_cannot(s_field, calibration):
    """Same walk, same odometry, same wrong initial guess: with and without looking."""
    trajectory = tj.PathBuilder([-2.0, -1.0, 0.0]).forward(1.8, 0.3).build("walk")
    guess = trajectory.poses[0] + [0.3, 0.3, math.radians(9.0)]
    looking = _pipeline(s_field)
    looking.init_gaussian(guess, 0.3, math.radians(10.0))
    position, _ = _errors(_drive(looking, s_field, trajectory, calibration), trajectory)

    blind = _pipeline(s_field)
    blind.init_gaussian(guess, 0.3, math.radians(10.0))
    odometry = SyntheticOdometry(OdometryNoiseConfig(), np.random.default_rng(1))
    for step in range(1, len(trajectory)):
        blind.localizer.predict(odometry.measure(trajectory.poses[step - 1],
                                                 trajectory.poses[step], 0.1 * step, 0.1))
    blind_error = np.linalg.norm(blind.localizer.estimate().pose[:2] - trajectory.poses[-1, :2])
    assert position[-1] < 0.08 < 0.25 < blind_error


def test_runs_are_reproducible_and_depend_on_the_filter_seed(s_field, calibration):
    trajectory = tj.PathBuilder([1.0, 0.5, 2.0]).forward(0.9, 0.3).build("walk")

    def run(seed):
        pipeline = _pipeline(s_field, seed)
        pipeline.init_gaussian(trajectory.poses[0], 0.3, math.radians(10.0))
        return np.array([o.pose for o in _drive(pipeline, s_field, trajectory, calibration)])

    a, b, c = run(0), run(0), run(5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ------------------------------------------------------------------ a wrong camera estimate
def test_a_wrong_camera_estimate_biases_the_pose_and_the_right_one_does_not(s_field, calibration):
    """The pipeline cannot discover the true camera: what it is told is what it uses."""
    trajectory = tj.PathBuilder([-1.2, 0.3, 0.0]).hold(3.0).build("stand")
    truth = calibration.pose()

    def final_error(believed):
        pipeline = _pipeline(s_field)
        pipeline.init_gaussian(trajectory.poses[0], 0.2, math.radians(5.0))
        outputs = _drive(pipeline, s_field, trajectory, calibration, true_camera=truth,
                         believed=believed)
        return _errors(outputs, trajectory)[0][-10:].mean()

    exact = final_error(truth)
    pitched = final_error(truth.perturbed(d_pitch=math.radians(-2.0)))
    raised = final_error(truth.perturbed(d_height=0.09))
    assert exact < 0.05
    assert pitched > 0.12 and pitched > 4.0 * exact
    assert raised > 2.0 * exact


# ------------------------------------------------------------------ field symmetry
def test_a_robot_at_the_mirrored_pose_sees_the_same_image(s_field, calibration):
    for pose in ([-2.6, -1.4, 0.6], [1.0, 2.0, -2.0], [3.3, 0.4, 0.2]):
        here = helpers.semantic_mask(s_field, pose, calibration, calibration.pose())
        there = helpers.semantic_mask(s_field, mirror_pose(np.array(pose)), calibration,
                                      calibration.pose())
        assert np.mean(here != there) < 1e-3, "identical up to rounding at paint edges"


def test_without_side_information_both_mirror_hypotheses_survive(s_field, calibration):
    """From a global start the images can only say "here, or the same place seen from the
    other half". The pipeline must report that, not pick a side and call it converged."""
    trajectory = tj.PathBuilder([-2.2, -1.0, 0.5]).turn(2.0 * math.pi, math.radians(60.0)) \
        .forward(0.6, 0.3).build("look_around")
    pipeline = _pipeline(s_field, config=ParticleFilterConfig.load(
        "particle_filter", {"num_particles": 1500}))
    pipeline.init_uniform()
    outputs = _drive(pipeline, s_field, trajectory, calibration)
    truth = trajectory.poses[-1]
    last = outputs[-1].estimate
    to_truth = np.linalg.norm(last.pose[:2] - truth[:2])
    to_mirror = np.linalg.norm(last.pose[:2] - mirror_pose(truth)[:2])
    assert min(to_truth, to_mirror) < 0.15, "it found the pose, up to the symmetry"
    late = [o.estimate for o in outputs[-15:]]
    assert sum(e.symmetric_ambiguity for e in late) >= 12, "and it knows that it is ambiguous"
    assert not any(e.converged for e in late), "an ambiguous estimate is never 'converged'"
    assert len(last.modes) >= 2
    twin = last.modes[1]
    assert np.linalg.norm(twin.pose[:2] - mirror_pose(last.pose)[:2]) < 0.3
    assert 0.25 < last.mode_weight < 0.75, "the two hypotheses share the belief"

    # Outside information (which half we are in) settles it, either way.
    pipeline.resolve_symmetry(truth[:2])
    settled = pipeline.localizer.estimate()
    assert np.linalg.norm(settled.pose[:2] - truth[:2]) < 0.15 and not settled.symmetric_ambiguity
    pipeline.resolve_symmetry(mirror_pose(truth)[:2])
    flipped = pipeline.localizer.estimate()
    assert np.linalg.norm(flipped.pose[:2] - mirror_pose(truth)[:2]) < 0.15


def test_an_own_half_prior_breaks_the_symmetry_and_a_wrong_prior_is_believed(s_field,
                                                                              calibration):
    trajectory = tj.PathBuilder([-2.2, -1.0, 0.5]).turn(2.0 * math.pi, math.radians(60.0)) \
        .forward(0.6, 0.3).build("look_around")
    config = ParticleFilterConfig.load("particle_filter", {"num_particles": 1500})
    half_x = s_field.half_extent[0]

    told = _pipeline(s_field, config=config)
    told.init_uniform(x_range=(-half_x, 0.0))              # external game-state information
    position, yaw = _errors(_drive(told, s_field, trajectory, calibration), trajectory)
    assert position[-1] < 0.15 and yaw[-1] < 4.0

    misled = _pipeline(s_field, config=config)
    misled.init_gaussian(mirror_pose(trajectory.poses[0]), 0.3, math.radians(10.0))
    outputs = _drive(misled, s_field, trajectory, calibration)
    estimate = outputs[-1].estimate
    assert np.linalg.norm(estimate.pose[:2] - mirror_pose(trajectory.poses[-1])[:2]) < 0.15
    assert estimate.converged, "confidently wrong: only the evaluator can tell"


# ------------------------------------------------------------------ the step interface
def test_step_accepts_only_its_declared_inputs(s_field, calibration):
    pipeline = _pipeline(s_field)
    pipeline.init_gaussian([0.0, 0.0, 0.0])
    image, _ = helpers.analytic_view(s_field, [0.0, 0.0, 0.0], calibration, calibration.pose())
    pose = calibration.pose()
    good = OdometryDelta(0.01, 0.0, 0.0)
    pipeline.step(image, 0.0, None, calibration, pose)
    pipeline.step(image, 0.1, good, calibration, pose, imu_roll=0.0, imu_pitch=0.01)
    for bad_image in (image.astype(float), image[..., 0], image.tolist(), None):
        with pytest.raises(TypeError):
            pipeline.step(bad_image, 0.2, good, calibration, pose)
    for bad_odometry in (np.array([0.01, 0.0, 0.0]), (0.01, 0.0, 0.0), [1.0, 2.0, 0.3]):
        with pytest.raises(TypeError):
            pipeline.step(image, 0.2, bad_odometry, calibration, pose)
    for bad_pose in (pose.T_base_camera, np.array([1.0, 2.0, 0.3]), None):
        with pytest.raises(TypeError):
            pipeline.step(image, 0.2, good, calibration, bad_pose)
    for bad_calibration in (calibration.intrinsics, None):
        with pytest.raises(TypeError):
            pipeline.step(image, 0.2, good, bad_calibration, pose)
    with pytest.raises(TypeError):
        pipeline.step(image, "now", good, calibration, pose)
    with pytest.raises(TypeError):
        pipeline.step(image, 0.2, good, calibration, pose, imu_pitch="level")
    with pytest.raises(TypeError):
        pipeline.step_input((image, 0.2, good, calibration, pose))
    with pytest.raises(TypeError):
        pipeline.step_with_predictions(image, 0.2, good, calibration, pose)
    with pytest.raises(TypeError):
        VisionLocalizationPipeline(helpers.ColourVision(), "s_field")
    assert pipeline.steps == 2, "rejected calls did not advance the filter"


def test_step_input_and_step_with_predictions_run_the_same_cycle(s_field, calibration):
    trajectory = tj.PathBuilder([-2.0, -1.0, 0.0]).forward(0.6, 0.3).build("walk")
    by_image = _pipeline(s_field)
    by_input = _pipeline(s_field)
    by_predictions = _pipeline(s_field)
    for pipeline in (by_image, by_input, by_predictions):
        pipeline.init_gaussian(trajectory.poses[0], 0.3, math.radians(10.0))
    vision = helpers.ColourVision()
    odometry = SyntheticOdometry(OdometryNoiseConfig(), np.random.default_rng(1))
    pose = calibration.pose()
    for step in range(len(trajectory)):
        image, _ = helpers.analytic_view(s_field, trajectory.poses[step], calibration, pose)
        delta = None if step == 0 else odometry.measure(
            trajectory.poses[step - 1], trajectory.poses[step], 0.1 * step, 0.1)
        a = by_image.step(image, 0.1 * step, delta, calibration, pose)
        b = by_input.step_input(VisionLocalizationInput(image, 0.1 * step, delta, calibration, pose))
        c = by_predictions.step_with_predictions(vision.run(image), 0.1 * step, delta,
                                                 calibration, pose)
        np.testing.assert_array_equal(a.pose, b.pose)
        np.testing.assert_array_equal(a.pose, c.pose)


def test_lost_odometry_diffuses_but_the_first_frame_does_not(s_field, calibration):
    image, _ = helpers.analytic_view(s_field, [0.5, 0.2, 0.0], calibration, calibration.pose())
    blank = np.zeros_like(image)                    # nothing to see: only the motion model acts
    pose = calibration.pose()
    pipeline = _pipeline(s_field)
    pipeline.init_gaussian([0.5, 0.2, 0.0], 0.05, math.radians(1.0))
    before = pipeline.localizer.particles.copy()
    first = pipeline.step(blank, 0.0, None, calibration, pose)
    np.testing.assert_array_equal(pipeline.localizer.particles, before)
    assert not first.update.used and first.observations.is_empty
    spread = np.std(pipeline.localizer.particles[:, 0])
    for step in range(1, 6):
        pipeline.step(blank, 0.1 * step, None, calibration, pose)      # five lost messages
    assert np.std(pipeline.localizer.particles[:, 0]) > 1.5 * spread
    pipeline.init_gaussian([0.5, 0.2, 0.0], 0.05, math.radians(1.0))
    assert pipeline.steps == 0, "re-initialising starts a new run"


def test_imu_tilt_changes_where_features_are_put(s_field, calibration):
    image, _ = helpers.analytic_view(s_field, [-2.0, 0.0, 0.0], calibration, calibration.pose())
    pose = calibration.pose()
    level = _pipeline(s_field)
    tilted = _pipeline(s_field)
    for pipeline in (level, tilted):
        pipeline.init_gaussian([-2.0, 0.0, 0.0], 0.1, math.radians(2.0))
    a = level.step(image, 0.0, None, calibration, pose)
    b = tilted.step(image, 0.0, None, calibration, pose, imu_pitch=math.radians(3.0))
    far_a = np.linalg.norm(a.features.line_points.points, axis=1).max()
    far_b = np.linalg.norm(b.features.line_points.points, axis=1).max()
    assert far_b < far_a - 0.3, "a torso pitched forward makes the same pixels land nearer"
