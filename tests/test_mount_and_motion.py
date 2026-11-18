"""Phase 4 geometry and motion, without a simulator: rotations, the configurable camera mount,
virtual head heights, and the synthetic body motion that moves the camera."""
import dataclasses
import math

import numpy as np
import pytest

from robosoccer_perception import config
from robosoccer_perception.geometry import mount as mg
from robosoccer_perception.geometry import quaternion as quat
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import StereoCameraConfig, StereoCameraModel
from robosoccer_perception.geometry.mount import BodyGeometry, CameraMount
from robosoccer_perception.geometry.robot_description import (
    RobotDescription, available_robots, robot_is_installed)
from robosoccer_perception.simulation import trajectories as tj
from robosoccer_perception.simulation.body_motion import (
    BodyMotion, MotionProfile, UniformSpline, available_profiles)

BODY = BodyGeometry("test_body", "pelvis", standing_height=0.40, torso_top=(0.0, 0.0, 0.27))


# ------------------------------------------------------------------ rotations
def test_quaternion_products_follow_the_rotation_matrices(rng):
    for _ in range(20):
        a, b = (quat.from_rotation_vector(rng.normal(size=3)) for _ in range(2))
        assert np.allclose(quat.to_matrix(quat.multiply(a, b)),
                           quat.to_matrix(a) @ quat.to_matrix(b), atol=1e-12)
        vector = rng.normal(size=3)
        assert np.allclose(quat.to_rotation_vector(quat.from_rotation_vector(vector)), vector) \
            or np.linalg.norm(vector) > math.pi


def test_positive_pitch_looks_down_and_positive_roll_lifts_the_left_side():
    nose = quat.matrix_from_rpy(0.0, math.radians(10.0)) @ [1.0, 0.0, 0.0]
    assert nose[2] < 0.0                                    # +x tips toward the ground
    left = quat.matrix_from_rpy(math.radians(10.0), 0.0) @ [0.0, 1.0, 0.0]
    assert left[2] > 0.0


def test_tilt_is_recovered_whatever_the_yaw(rng):
    for _ in range(30):
        roll, pitch = rng.uniform(-0.5, 0.5, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        rotation = quat.matrix_from_rpy(roll, pitch, yaw)
        assert np.allclose(quat.rpy_from_matrix(rotation), (roll, pitch, yaw), atol=1e-12)
        assert np.allclose(quat.tilt_from_quat(quat.from_matrix(rotation)), (roll, pitch),
                           atol=1e-9)
        assert np.allclose(quat.up_from_tilt(roll, pitch), rotation[2, :], atol=1e-12)


def test_slerp_moves_at_a_constant_rate_and_extrapolates():
    q0 = quat.from_rotation_vector([0.0, 0.2, 0.0])
    q1 = quat.multiply(q0, quat.from_rotation_vector([0.0, 0.4, 0.0]))
    for fraction in (0.0, 0.25, 0.5, 1.0, 1.5):
        expected = quat.multiply(q0, quat.from_rotation_vector([0.0, 0.4 * fraction, 0.0]))
        assert quat.angle_between(quat.slerp(q0, q1, fraction), expected) < 1e-9
    assert quat.angle_between(quat.slerp(q0, -q1, 0.5), quat.slerp(q0, q1, 0.5)) < 1e-9


# ------------------------------------------------------------------ the mount
def test_mount_height_comes_from_the_body_geometry_plus_the_offset():
    mount = CameraMount("m", (0.03, 0.0, 0.12), (0.0, math.radians(25.0), 0.0))
    assert mount.reference == "torso_top"
    assert mount.standing_height(BODY) == pytest.approx(0.40 + 0.27 + 0.12)
    taller = dataclasses.replace(BODY, torso_top=(0.0, 0.0, 0.32))
    assert mount.standing_height(taller) == pytest.approx(0.40 + 0.32 + 0.12)   # follows the body
    assert mount.shifted(up=0.05).standing_height(BODY) == pytest.approx(0.84)


@pytest.mark.parametrize("reference, expected", [("body", 0.40 + 0.5), ("torso_top", 0.67 + 0.5),
                                                 ("ground", 0.5)])
def test_mount_references(reference, expected):
    mount = CameraMount("m", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), reference=reference)
    assert mount.standing_height(BODY) == pytest.approx(expected)


def test_a_mount_without_a_height_is_refused():
    data = {"name": "bad", "xyz_m": [0.0, 0.0, None], "rpy_deg": [0.0, 25.0, 0.0]}
    with pytest.raises(ValueError, match="never assumed"):
        CameraMount.from_dict(data)
    with pytest.raises(ValueError):
        CameraMount("bad", (0.0, 0.0, math.nan), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        CameraMount("bad", (0.0, 0.0, 0.1), (0.0, 0.0, 0.0), reference="somewhere")


def test_every_shipped_mount_is_marked_experimental_and_loads():
    names = mg.available_mounts()
    assert {"head_low", "head_mid", "head_high", "head_tall", "legacy_rig"} <= set(names)
    for name in names:
        mount = CameraMount.load(name)
        assert mount.experimental, name                  # nothing claims to be the real head
        assert mount.parent_frame == "pelvis"
        assert CameraMount.from_dict(mount.to_dict()) == mount
        assert "EXPERIMENTAL" in mount.describe(BODY)
    heights = [CameraMount.load(n).standing_height(BODY)
               for n in ("head_low", "head_mid", "head_high", "head_tall")]
    assert heights == sorted(heights) and heights[-1] - heights[0] > 0.1


def test_mount_chain_matches_the_phase_3_camera_model_for_a_level_body():
    """With an upright body, the mount chain and `StereoCameraModel` give the same camera."""
    cfg = StereoCameraConfig.load("zed_mini_sim")
    model = StereoCameraModel(cfg)
    mount = CameraMount("legacy", (cfg.mount.forward_offset, 0.0, cfg.mount.height),
                        (0.0, cfg.mount.pitch, 0.0), reference="ground")
    for side in ("left", "right"):
        chain = mg.T_footprint_rig(mount, BODY, BODY.standing_height) \
            @ mg.T_rig_optical(cfg.baseline, side)
        assert np.allclose(chain, model.T_base_camera(side), atol=1e-12)


def test_rig_state_reproduces_any_rig_pose_of_a_tilted_body(rng):
    cfg = StereoCameraConfig.load("zed_mini_sim")
    model = StereoCameraModel(cfg)
    mount = CameraMount.load("head_mid").shifted(roll=0.03, yaw=-0.05, left=0.01)
    for _ in range(20):
        roll, pitch = rng.uniform(-0.2, 0.2, 2)
        height = rng.uniform(0.35, 0.45)
        transform = mg.T_footprint_rig(mount, BODY, height, roll, pitch)
        rig = mg.rig_state_from_transform(transform)
        assert np.allclose(model.T_base_rig(rig), transform, atol=1e-12)
        assert np.allclose(mg.transform_from_rig_state(rig), transform, atol=1e-12)


def test_body_pitch_and_mount_pitch_are_different_things():
    """A body pitched forward moves the camera forward and down; a mount pitched by the same
    angle only turns it. The two must not be interchangeable."""
    mount = CameraMount.load("head_mid")
    angle = math.radians(5.0)
    body_pitched = mg.T_footprint_rig(mount, BODY, BODY.standing_height, pitch=angle)
    mount_pitched = mg.T_footprint_rig(mount.shifted(pitch=angle), BODY, BODY.standing_height)
    assert np.allclose(body_pitched[:3, :3], mount_pitched[:3, :3], atol=1e-12)
    assert body_pitched[0, 3] > mount_pitched[0, 3] + 0.02      # the lever arm of the torso
    assert body_pitched[2, 3] < mount_pitched[2, 3]


# ------------------------------------------------------------------ robot descriptions
def test_robot_descriptions():
    assert {"placeholder", "humanoid_prototype_1"} <= set(available_robots())
    placeholder = RobotDescription.load("placeholder")
    assert placeholder.is_placeholder and placeholder.body.placeholder
    if not robot_is_installed("humanoid_prototype_1"):
        pytest.skip("humanoid model not imported")
    robot = RobotDescription.load("humanoid_prototype_1")
    assert not robot.is_placeholder and robot.body.body_frame == "pelvis"
    assert 0.38 < robot.body.standing_height < 0.43
    assert 0.65 < robot.body.torso_top_height < 0.72           # the torso's top: there is no head
    assert np.allclose(robot.body.T_body_imu, np.eye(4))       # placeholder IMU at the pelvis
    # None of the experimental heads is the old hard-coded 0.90 m by accident of typing:
    # each height is the body's plus the preset's offset.
    for name in ("head_low", "head_mid", "head_high", "head_tall"):
        mount = CameraMount.load(name)
        assert mount.standing_height(robot.body) == pytest.approx(
            robot.body.torso_top_height + mount.offset[2])


def test_asset_directory_can_be_redirected(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ASSET_DIR_ENV, str(tmp_path))
    assert config.assets_dir() == tmp_path
    assert not robot_is_installed("humanoid_prototype_1")
    with pytest.raises(FileNotFoundError, match="import_humanoid_model"):
        RobotDescription.load("humanoid_prototype_1")


# ------------------------------------------------------------------ body motion
def _walk(profile: str, seed: int = 0, trajectory=None) -> BodyMotion:
    trajectory = trajectory or tj.straight_line([-2.0, 0.5, 0.3], 3.0)
    return BodyMotion(trajectory, MotionProfile.load(profile), 0.40, np.random.default_rng(seed))


def test_every_motion_profile_loads_and_says_what_it_is():
    names = available_profiles()
    assert {"rigid", "standing", "sway_pitch", "sway_roll", "walking", "disturbed"} <= set(names)
    for name in names:
        profile = MotionProfile.load(name)
        assert profile.name == name and profile.description
        assert profile.is_still == (name == "rigid")
    with pytest.raises(ValueError, match="unknown MotionProfile keys"):
        MotionProfile.from_dict({"name": "x", "wobble": 1.0})


def test_spline_passes_through_its_samples_and_continues_straight():
    values = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 2.0], [3.0, 2.5]])
    spline = UniformSpline(values, 0.1, extrapolate=True)
    assert np.allclose(spline(0.1 * np.arange(4)), values)
    before, after = spline([-0.1])[0], spline([0.4])[0]
    assert np.allclose(before, values[0] - (values[1] - values[0]))
    assert np.allclose(after, values[3] + (values[3] - values[2]))
    assert np.allclose(UniformSpline(values, 0.1)([-1.0, 9.0]), values[[0, -1]])    # held


def test_a_rigid_body_is_level_and_follows_the_path_exactly():
    trajectory = tj.rectangle([-1.0, -1.0, 0.0], 2.0, 1.5)
    motion = BodyMotion(trajectory, MotionProfile.load("rigid"), 0.40)
    sample = motion.sample(trajectory.times)
    assert np.allclose(sample.roll, 0.0) and np.allclose(sample.pitch, 0.0)
    assert np.allclose(sample.height, 0.40)
    assert np.allclose(sample.footprint[:, :2], trajectory.poses[:, :2], atol=1e-12)
    assert np.allclose(tf.angle_difference(sample.footprint[:, 2], trajectory.poses[:, 2]), 0.0,
                       atol=1e-12)
    assert np.allclose(motion.footprint_trajectory().poses, sample.footprint)


@pytest.mark.parametrize("profile, axis", [("sway_pitch", "pitch"), ("sway_roll", "roll")])
def test_sway_is_periodic_tilt_about_the_ankles(profile, axis):
    motion = _walk(profile, trajectory=tj.stationary([0.5, 0.2, 0.0], 20.0))
    times = np.arange(0.0, 20.0, 0.01)
    sample = motion.sample(times)
    angle = np.degrees(getattr(sample, axis))
    other = np.degrees(sample.roll if axis == "pitch" else sample.pitch)
    assert 3.0 < np.abs(angle).max() < 5.5 and np.abs(other).max() < 1.5
    # Inverted pendulum: pitching forward carries the body forward, rolling left-side-up
    # carries it to the right, and the body sinks a little either way.
    shift = sample.position[:, 0] - 0.5 if axis == "pitch" else -(sample.position[:, 1] - 0.2)
    assert np.corrcoef(shift, np.sin(np.radians(angle)))[0, 1] > 0.98
    assert np.abs(shift).max() == pytest.approx(0.40 * math.sin(math.radians(4.0)), rel=0.25)
    assert sample.height.max() <= 0.40 + 0.003
    spectrum = np.abs(np.fft.rfft(angle - angle.mean()))
    peak = np.fft.rfftfreq(len(times), 0.01)[spectrum.argmax()]
    assert peak == pytest.approx(getattr(motion.profile, f"sway_{axis}_hz"), abs=0.06)


def test_walking_moves_every_degree_of_freedom_and_only_while_walking():
    path = tj.PathBuilder([-2.0, 0.0, 0.0]).hold(4.0).forward(3.0).hold(4.0).build("walk")
    motion = BodyMotion(path, MotionProfile.load("walking"), 0.40, np.random.default_rng(1))
    times = np.arange(0.0, path.duration, 0.01)
    sample = motion.sample(times)
    walking = (times > 6.0) & (times < 12.0)
    standing = times < 2.5
    assert sample.activity[walking].min() > 0.95 and sample.activity[standing].max() < 0.05
    for values, low in ((np.degrees(sample.roll), 1.5), (np.degrees(sample.pitch), 1.0),
                        (1000.0 * sample.height, 5.0)):
        assert np.ptp(values[walking]) > 2.0 * low
        assert np.ptp(values[standing]) < np.ptp(values[walking]) / 2.0
    # lateral shift and yaw wobble are part of the true footprint, as for a real pelvis
    lateral = sample.footprint[walking, 1]
    assert 0.02 < np.ptp(lateral) < 0.08
    assert 2.0 < np.degrees(np.ptp(sample.footprint[walking, 2])) < 6.0
    # two steps per stride: the bob is at the step frequency, the roll at half of it
    def peak(values):
        spectrum = np.abs(np.fft.rfft(values - values.mean()))
        return np.fft.rfftfreq(len(values), 0.01)[spectrum.argmax()]
    assert peak(sample.height[walking]) == pytest.approx(motion.profile.step_frequency_hz, abs=0.2)
    assert peak(sample.roll[walking]) == pytest.approx(motion.profile.step_frequency_hz / 2,
                                                       abs=0.2)


def test_disturbances_are_brief_abrupt_tilts():
    motion = _walk("disturbed", seed=3, trajectory=tj.stationary([0.0, 0.0, 0.0], 30.0))
    assert len(motion.disturbances) >= 3
    times = np.arange(0.0, 30.0, 0.002)
    sample = motion.sample(times)
    tilt = np.degrees(np.hypot(sample.roll, sample.pitch))
    assert 4.0 <= tilt.max() <= 9.5
    for event in motion.disturbances:
        assert event.start >= motion.profile.disturbance_start_s
        peak = np.degrees(math.hypot(event.roll, event.pitch))
        assert 4.0 <= peak <= 8.0
        window = (times > event.start) & (times < event.start + motion.profile.disturbance_rise_s)
        assert np.ptp(tilt[window]) > 0.6 * peak                 # most of it within the rise
    rate = np.abs(np.gradient(sample.pitch, times)) + np.abs(np.gradient(sample.roll, times))
    assert np.degrees(rate.max()) > 30.0                         # abrupt ...
    assert np.degrees(np.abs(np.diff(sample.pitch)).max()) < 0.5   # ... but continuous


def test_motion_is_deterministic_per_seed_and_differs_between_seeds():
    times = np.linspace(0.0, 8.0, 200)
    first, again, other = (_walk("walking_disturbed", seed).sample(times) for seed in (4, 4, 5))
    assert np.array_equal(first.position, again.position)
    assert np.array_equal(first.rotation, again.rotation)
    assert not np.allclose(first.roll, other.roll)


def test_body_pose_decomposes_into_footprint_times_tilt():
    """``T_field_body = T_field_footprint @ T_footprint_body(height, roll, pitch)``: the
    footprint carries x, y and yaw, the tilt carries nothing but roll, pitch and height."""
    sample = _walk("walking_rough", seed=2).sample(np.linspace(0.0, 9.0, 60))
    for index in range(len(sample)):
        expected = tf.T_field_base_from_pose(sample.footprint[index]) @ mg.T_footprint_body(
            sample.height[index], sample.roll[index], sample.pitch[index])
        assert np.allclose(sample.T_field_body(index), expected, atol=1e-12)


def test_a_teleporting_path_is_refused():
    path = tj.PathBuilder([0.0, 0.0, 0.0]).forward(1.0).teleport([2.0, 1.0, 0.0]).build("k")
    with pytest.raises(ValueError, match="continuous"):
        BodyMotion(path, MotionProfile.load("walking"), 0.40)
