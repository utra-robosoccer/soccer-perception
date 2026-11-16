"""The particle filter, its measurement model and the pose estimator."""
import math

import numpy as np
import pytest

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import RigState
from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.localization.estimation import (
    EstimatorConfig, estimate_pose, mirror_pose)
from robosoccer_perception.localization.measurement_model import MeasurementModel
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.localization.particle_filter import (
    ParticleFilter, ParticleFilterConfig)
from robosoccer_perception.observations.noise import OracleConfig
from robosoccer_perception.observations.oracle import OracleObservationGenerator
from robosoccer_perception.observations.types import (
    LandmarkObservation, LinePointsObservation, ObservationFrame)

RIG = RigState(height=0.9, pitch=math.radians(25), forward_offset=0.05)
TRUE_POSE = np.array([1.5, 0.3, 0.2])


@pytest.fixture(scope="module")
def clean_oracle(s_field, s_landmarks, camera_model):
    return OracleObservationGenerator(s_field, s_landmarks, camera_model,
                                      OracleConfig.load("clean"))


def observe(oracle, pose, seed=0, rig=RIG):
    return oracle.generate(pose, np.random.default_rng(seed), rig)[0]


def cloud(centre, count, position_std, yaw_std, rng):
    return np.asarray(centre) + rng.normal(size=(count, 3)) * [position_std, position_std, yaw_std]


# ------------------------------------------------------------------ configuration
def test_config_loads_and_validates():
    cfg = ParticleFilterConfig.load()
    assert cfg.num_particles == 1000 and 0 < cfg.resample_ess_fraction < 1
    assert cfg.measurement.correspondence in ("nearest", "mixture")
    assert ParticleFilterConfig.load(overrides={"num_particles": 250}).num_particles == 250
    nested = ParticleFilterConfig.load(overrides={"recovery": {"symmetry_prior": "none"}})
    assert nested.recovery.symmetry_prior == "none"
    assert nested.recovery.alpha_slow == cfg.recovery.alpha_slow, "other keys keep their values"
    with pytest.raises(ValueError):
        ParticleFilterConfig.load(overrides={"recovery": {"symmetry_prior": "left"}})
    with pytest.raises(TypeError):
        ParticleFilterConfig.load(overrides={"no_such_option": 1})


def test_yaml_and_dataclass_defaults_agree():
    assert ParticleFilterConfig.load() == ParticleFilterConfig()


# ------------------------------------------------------------------ initialisation
def test_gaussian_initialisation(s_field):
    pf = ParticleFilter(s_field, seed=1)
    pf.init_gaussian([1.0, -2.0, 3.0], position_std=0.2, yaw_std=0.1)
    assert pf.particles.shape == (1000, 3)
    np.testing.assert_allclose(pf.particles[:, :2].mean(axis=0), [1.0, -2.0], atol=0.03)
    np.testing.assert_allclose(pf.particles[:, :2].std(axis=0), 0.2, atol=0.02)
    assert abs(tf.angle_difference(tf.circular_mean(pf.particles[:, 2]), 3.0)) < 0.02
    assert np.all((pf.particles[:, 2] >= -math.pi) & (pf.particles[:, 2] < math.pi))
    np.testing.assert_allclose(pf.weights, 1e-3)
    assert pf.effective_sample_size == pytest.approx(1000)


def test_uniform_initialisation_covers_the_carpet(s_field):
    pf = ParticleFilter(s_field, seed=1)
    pf.init_uniform()
    half_x, half_y = s_field.half_extent
    assert np.all(s_field.contains(pf.particles[:, :2]))
    assert pf.particles[:, 0].min() < -0.9 * half_x and pf.particles[:, 0].max() > 0.9 * half_x
    assert pf.particles[:, 1].min() < -0.9 * half_y and pf.particles[:, 1].max() > 0.9 * half_y
    histogram, _ = np.histogram(pf.particles[:, 2], bins=8, range=(-math.pi, math.pi))
    assert histogram.min() > 80, "all headings are represented"
    pf.init_uniform(x_range=(-half_x, 0.0))
    assert pf.particles[:, 0].max() <= 0.0, "a side prior restricts the particles"


def test_reinitialisation_and_particle_count(s_field):
    pf = ParticleFilter(s_field, cfg=ParticleFilterConfig(num_particles=300), seed=1)
    assert pf.num_particles == 300
    pf.init_gaussian([0, 0, 0])
    pf.init_uniform()                       # reinitialising discards the old belief
    assert pf.particles[:, :2].std(axis=0).min() > 1.0
    pf.init_gaussian([2, 1, 0])
    assert np.linalg.norm(pf.particles[:, :2].mean(axis=0) - [2, 1]) < 0.1


# ------------------------------------------------------------------ measurement model
def test_likelihood_prefers_the_true_pose(s_field, clean_oracle):
    model = MeasurementModel(s_field)
    frame = observe(clean_oracle, TRUE_POSE)
    rng = np.random.default_rng(0)
    candidates = np.array([TRUE_POSE,
                           TRUE_POSE + [0.5, 0.0, 0.0],
                           TRUE_POSE + [0.0, 0.5, 0.0],
                           TRUE_POSE + [0.0, 0.0, math.radians(20)],
                           [-3.0, 2.0, 2.0]])
    result = model.log_likelihood(candidates, frame, rng)
    assert result.evidence_weight > 0
    assert np.argmax(result.log_likelihood) == 0
    assert np.all(result.log_likelihood[0] - result.log_likelihood[1:] > 3.0)


def test_likelihood_is_identical_at_the_mirrored_pose(s_field, clean_oracle):
    """The field's half-turn symmetry is in the map: the model cannot tell the halves apart."""
    model = MeasurementModel(s_field)
    frame = observe(clean_oracle, TRUE_POSE)
    poses = np.array([TRUE_POSE, mirror_pose(TRUE_POSE)])
    a = model.log_likelihood(poses, frame, np.random.default_rng(0)).log_likelihood
    assert a[0] == pytest.approx(a[1], abs=1e-6)


@pytest.mark.parametrize("use_lines,use_landmarks", [(True, False), (False, True)])
def test_each_cue_alone_is_informative(s_field, clean_oracle, use_lines, use_landmarks):
    cfg = ParticleFilterConfig.load(overrides={"measurement": {
        "use_line_points": use_lines, "use_landmarks": use_landmarks}}).measurement
    model = MeasurementModel(s_field, cfg=cfg)
    frame = observe(clean_oracle, TRUE_POSE)
    poses = np.array([TRUE_POSE, TRUE_POSE + [0.4, 0.3, 0.2]])
    scores = model.log_likelihood(poses, frame, np.random.default_rng(0)).log_likelihood
    assert scores[0] > scores[1] + 2.0


def test_ambiguous_correspondence_is_resolved_by_the_model(s_field, s_landmarks):
    """One observed T junction is explained equally well from every pose that sees *a* T there."""
    model = MeasurementModel(s_field, s_landmarks)
    observation = LandmarkObservation(LandmarkType.T_INTERSECTION, np.array([2.0, 0.0]), 0.05,
                                      math.radians(1.0))
    frame = ObservationFrame(0.0, (observation,))
    t_junctions = s_landmarks.positions(LandmarkType.T_INTERSECTION)
    poses = []
    for landmark in t_junctions:            # stand 2 m from each T junction, facing it
        heading = 0.7
        poses.append([landmark[0] - 2.0 * math.cos(heading), landmark[1] - 2.0 * math.sin(heading),
                      heading])
    poses.append([0.3, 0.2, 2.0])           # a pose with no T junction 2 m ahead
    scores = model.log_likelihood(np.array(poses), frame, np.random.default_rng(0)).log_likelihood
    on_carpet = s_field.contains(np.array(poses)[:-1, :2])
    assert on_carpet.sum() >= 5
    np.testing.assert_allclose(scores[:-1][on_carpet], 0.0, atol=1e-6)   # all equally good
    assert scores[-1] < -2.0


def test_outliers_have_bounded_influence(s_field, clean_oracle):
    model = MeasurementModel(s_field)
    frame = observe(clean_oracle, TRUE_POSE)
    bogus = LandmarkObservation(LandmarkType.GOAL_POST, np.array([1.0, 3.0]), 0.03,
                                math.radians(1.0))
    spoiled = ObservationFrame(0.0, (*frame.landmarks, bogus), frame.line_points)
    poses = np.array([TRUE_POSE, TRUE_POSE + [0.5, 0.0, 0.0]])
    clean = model.log_likelihood(poses, frame, np.random.default_rng(0)).log_likelihood
    dirty = model.log_likelihood(poses, spoiled, np.random.default_rng(0)).log_likelihood
    penalty = clean[0] - dirty[0]
    assert 0 < penalty <= -math.log(model.cfg.landmark_outlier_floor) + 1e-9
    assert dirty[0] > dirty[1], "the true pose still wins"


def test_empty_frame_carries_no_evidence(s_field):
    model = MeasurementModel(s_field)
    result = model.log_likelihood(np.zeros((5, 3)), ObservationFrame(0.0), np.random.default_rng(0))
    assert result.evidence_weight == 0 and np.all(result.log_likelihood == 0)


def test_mixture_correspondence_also_works(s_field, clean_oracle):
    cfg = ParticleFilterConfig.load(overrides={"measurement": {"correspondence": "mixture"}})
    model = MeasurementModel(s_field, cfg=cfg.measurement)
    frame = observe(clean_oracle, TRUE_POSE)
    poses = np.array([TRUE_POSE, TRUE_POSE + [0.5, 0.2, 0.3]])
    scores = model.log_likelihood(poses, frame, np.random.default_rng(0)).log_likelihood
    assert scores[0] > scores[1] + 3.0


# ------------------------------------------------------------------ resampling
def test_systematic_resampling_follows_the_weights(s_field):
    pf = ParticleFilter(s_field, cfg=ParticleFilterConfig(num_particles=1000), seed=3)
    pf.weights = np.zeros(1000)
    pf.weights[[10, 20]] = [0.75, 0.25]
    picks = pf._systematic_resample(1000)
    assert set(picks) == {10, 20}
    assert (picks == 10).sum() == 750 and (picks == 20).sum() == 250   # low variance: exact


def test_effective_sample_size():
    from robosoccer_perception.geometry.field import FieldGeometry
    pf = ParticleFilter(FieldGeometry.load("s_field"), cfg=ParticleFilterConfig(num_particles=100))
    assert pf.effective_sample_size == pytest.approx(100)
    pf.weights = np.zeros(100)
    pf.weights[0] = 1.0
    assert pf.effective_sample_size == pytest.approx(1.0)
    pf.weights = np.r_[np.full(50, 0.02), np.zeros(50)]
    assert pf.effective_sample_size == pytest.approx(50)


def test_resampling_only_when_needed(s_field, clean_oracle):
    pf = ParticleFilter(s_field, seed=4)
    pf.init_gaussian(TRUE_POSE, position_std=0.5, yaw_std=0.3)
    first = pf.update(observe(clean_oracle, TRUE_POSE))
    assert first.used and first.resampled, "a broad cloud meeting sharp evidence must resample"
    assert first.effective_sample_size < 0.5 * pf.num_particles
    resampled = []
    for step in range(40):                  # standing still, already converged
        pf.predict(OdometryDelta(0.0, 0.0, 0.0))
        resampled.append(pf.update(observe(clean_oracle, TRUE_POSE, seed=step + 1)).resampled)
    assert 0 < np.mean(resampled) < 0.9, "resampling is triggered by the ESS, not every update"
    idle = pf.update(ObservationFrame(0.0))
    assert not idle.used and not idle.resampled


# ------------------------------------------------------------------ pose estimation
def test_estimate_of_a_single_cluster(rng):
    particles = cloud([1.0, -1.0, math.pi - 0.05], 2000, 0.05, 0.05, rng)
    particles[:, 2] = tf.wrap_angle(particles[:, 2])        # straddles the +-pi wrap
    estimate = estimate_pose(particles, np.full(2000, 1 / 2000))
    np.testing.assert_allclose(estimate.pose[:2], [1.0, -1.0], atol=0.01)
    assert abs(tf.angle_difference(estimate.yaw, math.pi - 0.05)) < 0.01
    assert estimate.mode_weight > 0.99 and estimate.converged and not estimate.symmetric_ambiguity
    assert estimate.position_std == pytest.approx(0.05 * math.sqrt(2), rel=0.1)
    assert estimate.yaw_std == pytest.approx(0.05, rel=0.1)
    assert estimate.confidence > 0.9 and estimate.covariance.shape == (3, 3)


def test_two_symmetric_modes_are_not_averaged(rng):
    """The naive mean of two mirrored clusters is the field centre; the estimator picks a mode."""
    pose = np.array([2.5, 1.0, 0.4])
    particles = np.vstack([cloud(pose, 600, 0.05, 0.03, rng),
                           cloud(mirror_pose(pose), 400, 0.05, 0.03, rng)])
    particles[:, 2] = tf.wrap_angle(particles[:, 2])
    weights = np.full(1000, 1e-3)
    assert np.linalg.norm(weights @ particles[:, :2]) < 0.6, "the mean lands near the centre"
    estimate = estimate_pose(particles, weights)
    np.testing.assert_allclose(estimate.pose[:2], pose[:2], atol=0.02)
    assert estimate.mode_weight == pytest.approx(0.6, abs=0.01)
    assert estimate.symmetric_ambiguity and not estimate.converged
    assert len(estimate.modes) == 2
    np.testing.assert_allclose(estimate.modes[1].pose[:2], -pose[:2], atol=0.02)
    assert estimate.modes[1].weight == pytest.approx(0.4, abs=0.01)
    assert estimate.confidence < 0.65


def test_estimate_respects_weights(rng):
    particles = np.vstack([cloud([-3.0, 0.0, 0.0], 900, 0.05, 0.03, rng),
                           cloud([3.0, 1.0, 1.0], 100, 0.05, 0.03, rng)])
    weights = np.r_[np.full(900, 0.1 / 900), np.full(100, 0.9 / 100)]
    estimate = estimate_pose(particles, weights)
    np.testing.assert_allclose(estimate.pose[:2], [3.0, 1.0], atol=0.03)
    assert estimate.mode_weight == pytest.approx(0.9, abs=0.01)


def test_uniform_cloud_is_reported_as_not_converged(s_field):
    pf = ParticleFilter(s_field, seed=2)
    pf.init_uniform()
    estimate = pf.estimate()
    assert not estimate.converged and estimate.mode_weight < 0.1 and estimate.confidence < 0.1


def test_estimator_thresholds_are_configurable(rng):
    particles = cloud([0.0, 0.0, 0.0], 1000, 0.12, 0.05, rng)
    weights = np.full(1000, 1e-3)
    assert estimate_pose(particles, weights).converged
    strict = EstimatorConfig(converged_max_position_std=0.05)
    assert not estimate_pose(particles, weights, strict).converged


# ------------------------------------------------------------------ the whole filter
def run(pf, oracle, poses, rig=RIG, seed=100, pan=None):
    """Drive the filter along true poses with exact odometry; return the final estimate."""
    previous = None
    for step, pose in enumerate(poses):
        if previous is not None:
            pf.predict(OdometryDelta(*tf.se2_relative(previous, pose)))
        view = rig if pan is None else rig.replace(pan=pan[step])
        pf.update(observe(oracle, pose, seed + step, view))
        estimate = pf.estimate()
        previous = pose
    return estimate


def errors(estimate, pose):
    return (float(np.linalg.norm(estimate.pose[:2] - pose[:2])),
            math.degrees(abs(tf.angle_difference(estimate.yaw, pose[2]))))


def test_tracking_from_an_approximate_pose(s_field, clean_oracle):
    pf = ParticleFilter(s_field, seed=5)
    pf.init_gaussian(TRUE_POSE + [0.25, -0.2, 0.15], position_std=0.3, yaw_std=0.2)
    poses = [TRUE_POSE + [0.03 * i, 0.0, 0.0] for i in range(40)]
    estimate = run(pf, clean_oracle, poses)
    position_error, yaw_error = errors(estimate, poses[-1])
    assert position_error < 0.08 and yaw_error < 2.0
    assert estimate.converged and estimate.confidence > 0.8


@pytest.mark.slow
def test_global_localisation_converges_up_to_symmetry(s_field, clean_oracle):
    pf = ParticleFilter(s_field, seed=6)
    pf.init_uniform()
    assert not pf.estimate().converged
    start = np.array([-2.0, 1.0, 0.5])
    poses = [start + [0.0, 0.0, math.radians(3.0) * i] for i in range(120)]     # one full turn
    estimate = run(pf, clean_oracle, poses)
    truth = poses[-1]
    direct = errors(estimate, truth)
    mirrored = errors(estimate, mirror_pose(truth))
    assert min(direct[0], mirrored[0]) < 0.15 and min(direct[1], mirrored[1]) < 4.0
    # With no side information the filter must not pretend to know which half it is in:
    # it holds the pose and its mirror image, flags the ambiguity and withholds "converged".
    assert estimate.symmetric_ambiguity and not estimate.converged
    assert estimate.modes[0].weight + estimate.modes[1].weight > 0.8
    assert 0.25 < estimate.mode_weight < 0.75
    np.testing.assert_allclose(estimate.modes[1].pose[:2], -estimate.modes[0].pose[:2], atol=0.2)

    # Outside information ("we are in the -x half") resolves it, in whichever direction is true.
    pf.resolve_symmetry([-2.75, 0.0])
    resolved = pf.estimate()
    position_error, yaw_error = errors(resolved, truth)
    assert position_error < 0.15 and yaw_error < 4.0
    assert not resolved.symmetric_ambiguity and resolved.mode_weight > 0.8
    estimate = run(pf, clean_oracle, [truth] * 30, seed=900)       # and it stays resolved
    assert errors(estimate, truth)[0] < 0.15 and estimate.converged


@pytest.mark.slow
def test_without_symmetry_maintenance_the_filter_picks_a_side_at_random(s_field, clean_oracle):
    """The behaviour `maintain_symmetry` exists to prevent: a confident coin flip."""
    cfg = ParticleFilterConfig.load(overrides={"recovery": {"maintain_symmetry": False}})
    start = np.array([-2.0, 1.0, 0.5])
    poses = [start + [0.0, 0.0, math.radians(3.0) * i] for i in range(120)]
    sides = []
    for seed in range(8):
        pf = ParticleFilter(s_field, cfg=cfg, seed=seed)
        pf.init_uniform()
        estimate = run(pf, clean_oracle, poses)
        direct, mirrored = errors(estimate, poses[-1]), errors(estimate, mirror_pose(poses[-1]))
        assert min(direct[0], mirrored[0]) < 0.15
        sides.append(direct[0] < mirrored[0])
    assert 0 < sum(sides) < 8, "some seeds end on the true pose, others on its mirror image"


@pytest.mark.slow
def test_side_prior_resolves_the_symmetry(s_field, clean_oracle):
    start = np.array([-2.0, 1.0, 0.5])
    poses = [start + [0.0, 0.0, math.radians(3.0) * i] for i in range(120)]
    for seed in range(4):
        pf = ParticleFilter(s_field, seed=seed)
        pf.init_uniform(x_range=(-s_field.half_extent[0], 0.0))
        position_error, yaw_error = errors(run(pf, clean_oracle, poses), poses[-1])
        assert position_error < 0.15 and yaw_error < 4.0, seed


@pytest.mark.slow
def test_kidnapped_robot_recovers_without_ground_truth(s_field, clean_oracle):
    pf = ParticleFilter(s_field, seed=7)
    before = np.array([-2.5, -1.0, 0.3])
    pf.init_gaussian(before, position_std=0.2, yaw_std=0.1)
    estimate = run(pf, clean_oracle, [before] * 30)
    assert errors(estimate, before)[0] < 0.1 and estimate.converged
    # Moved 3 m without any odometry. The filter is only ever given observations.
    after = np.array([-1.0, 1.6, -1.2])
    injected = 0
    for step in range(60):
        pf.predict(OdometryDelta(0.0, 0.0, 0.0))
        injected += pf.update(observe(clean_oracle, after, 500 + step)).injected
        estimate = pf.estimate()
    position_error, yaw_error = errors(estimate, after)
    assert injected > 0, "recovery works by injecting fresh hypotheses"
    assert position_error < 0.15 and yaw_error < 4.0


def test_missing_observations_and_odometry_are_tolerated(s_field, clean_oracle):
    pf = ParticleFilter(s_field, seed=8)
    pf.init_gaussian(TRUE_POSE, position_std=0.1, yaw_std=0.05)
    pose = TRUE_POSE.copy()
    for step in range(60):
        nxt = pose + [0.02, 0.0, 0.0]
        delta = None if step % 5 == 0 else OdometryDelta(*tf.se2_relative(pose, nxt))
        frame = observe(clean_oracle, nxt, step) if step % 3 == 0 else None
        estimate, info = pf.step(delta, frame)
        assert info.used == (frame is not None)
        pose = nxt
    position_error, yaw_error = errors(estimate, pose)
    assert position_error < 0.15 and yaw_error < 4.0
    assert np.isfinite(pf.particles).all() and pf.weights.sum() == pytest.approx(1.0)


def test_filter_is_deterministic_for_a_seed(s_field, clean_oracle):
    def final(seed):
        pf = ParticleFilter(s_field, seed=seed)
        pf.init_gaussian(TRUE_POSE)
        return run(pf, clean_oracle, [TRUE_POSE] * 10).pose
    np.testing.assert_array_equal(final(1), final(1))
    assert not np.array_equal(final(1), final(2))


def test_injected_hypotheses_respect_the_symmetry_prior(s_field, clean_oracle):
    frame = observe(clean_oracle, TRUE_POSE)
    pf = ParticleFilter(s_field, seed=9)
    pf.init_gaussian(TRUE_POSE)                         # anchors the belief on TRUE_POSE's side
    proposals = pf._propose(400, frame)
    to_anchor = np.linalg.norm(proposals[:, :2] - TRUE_POSE[:2], axis=1)
    to_mirror = np.linalg.norm(proposals[:, :2] + TRUE_POSE[:2], axis=1)
    assert np.all(to_anchor <= to_mirror + 1e-9)
    assert np.all(s_field.contains(proposals[:, :2]))
    # Sensor-reset proposals explain an observed landmark: some land close to the true pose.
    assert (to_anchor < 0.3).sum() >= 3

    free = ParticleFilter(s_field, cfg=ParticleFilterConfig.load(
        overrides={"recovery": {"symmetry_prior": "none"}}), seed=9)
    free.init_gaussian(TRUE_POSE)
    proposals = free._propose(400, frame)
    to_anchor = np.linalg.norm(proposals[:, :2] - TRUE_POSE[:2], axis=1)
    to_mirror = np.linalg.norm(proposals[:, :2] + TRUE_POSE[:2], axis=1)
    assert 0.25 < np.mean(to_anchor > to_mirror) < 0.75, "without the prior both sides are proposed"
