"""The simulated IMU, the tilt filter, camera / IMU synchronisation and the per-image camera
pose. No MuJoCo, no rendering: body motion in, sensor samples through, estimates out."""
import dataclasses
import math

import numpy as np
import pytest

from robosoccer_perception.estimation.camera_pose import (
    CameraPoseEstimator, FusionConfig, camera_roll_pitch)
from robosoccer_perception.estimation.imu_types import GRAVITY, ImuSample
from robosoccer_perception.estimation.orientation import (
    OrientationEstimate, TiltEstimator, TiltFilterConfig)
from robosoccer_perception.estimation.sync import (
    EXTRAPOLATED, HELD, INTERPOLATED, LATEST, NONE, STALE, OrientationHistory, SyncConfig)
from robosoccer_perception.geometry import mount as mg
from robosoccer_perception.geometry import quaternion as quat
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose)
from robosoccer_perception.geometry.mount import BodyGeometry, CameraMount
from robosoccer_perception.simulation import trajectories as tj
from robosoccer_perception.simulation.body_motion import BodyMotion, MotionProfile
from robosoccer_perception.simulation.imu import (
    ImuConfig, ideal_measurements, simulate_imu)

HEIGHT = 0.40
BODY = BodyGeometry("test_body", "pelvis", standing_height=HEIGHT, torso_top=(0.0, 0.0, 0.27))
STILL = tj.stationary([0.3, -0.2, 0.4], 30.0)


def motion_for(profile, trajectory=STILL, seed=0, **changes) -> BodyMotion:
    profile = MotionProfile.load(profile) if isinstance(profile, str) else profile
    return BodyMotion(trajectory, profile.replace(**changes), HEIGHT, np.random.default_rng(seed))


def run_filter(stream, cfg=None, R_body_sensor=None):
    """Feed a whole stream; return the filter and (errors in deg (N, 2), reported std in deg)."""
    estimator = TiltEstimator(cfg, R_body_sensor)
    errors, stds = [], []
    for index in range(len(stream)):
        estimate = estimator.update(stream.sample(index))
        if estimate is not None:
            errors.append([estimate.roll - stream.truth_roll[index],
                           estimate.pitch - stream.truth_pitch[index]])
            stds.append(estimate.tilt_std)
    return estimator, np.degrees(np.array(errors)), np.degrees(np.array(stds))


# ------------------------------------------------------------------ ideal measurements
def test_a_level_sensor_at_rest_reads_plus_g_and_no_rotation():
    gyro, force = ideal_measurements(motion_for("rigid"), [1.0, 7.5])
    assert np.allclose(gyro, 0.0, atol=1e-9)
    assert np.allclose(force, [0.0, 0.0, GRAVITY], atol=1e-6)


def test_ideal_measurements_follow_the_motion_analytically():
    """Pure pitch sway about the ankles, A sin(w t): the gyro reads the pitch rate about +y,
    and the specific force is gravity seen from the tilted body plus the pendulum's own
    acceleration."""
    profile = MotionProfile("pure", sway_pitch_deg=5.0, sway_pitch_hz=0.5)
    motion = motion_for(profile)
    times = np.linspace(2.0, 6.0, 81)
    gyro, force = ideal_measurements(motion, times)
    sample = motion.sample(times)
    rate = np.gradient(sample.pitch, times)
    assert np.allclose(gyro[2:-2, 1], rate[2:-2], atol=2e-3)       # np.gradient is rough at the ends
    assert np.allclose(gyro[:, [0, 2]], 0.0, atol=1e-6)
    # at rest in a tilted pose the x axis would read -g sin(pitch); the pendulum adds
    # HEIGHT * pitch'' (tangential), which is small at 0.5 Hz but not zero
    alpha = np.gradient(rate, times)
    assert np.allclose(force[4:-4, 0], (-GRAVITY * np.sin(sample.pitch) + HEIGHT * alpha)[4:-4],
                       atol=0.02)
    assert np.allclose(np.linalg.norm(force, axis=1), GRAVITY, atol=0.15)


def test_a_lever_arm_adds_centripetal_and_tangential_acceleration():
    profile = MotionProfile("pure", sway_roll_deg=6.0, sway_roll_hz=0.8)
    motion = motion_for(profile)
    times = np.linspace(1.0, 5.0, 200)
    lever = tf.make_transform(None, [0.0, 0.0, 0.25])
    _, at_origin = ideal_measurements(motion, times)
    gyro, at_lever = ideal_measurements(motion, times, lever)
    assert np.abs(at_lever - at_origin).max() > 0.3             # m/s^2: clearly different
    # Further from the ankles the same swing means a larger acceleration: the total
    # pendulum arm is HEIGHT + 0.25, so the lateral term grows by that ratio.
    sample = motion.sample(times)
    lateral = lambda f: f[:, 1] - GRAVITY * np.sin(sample.roll)     # noqa: E731
    ratio = np.polyfit(lateral(at_origin), lateral(at_lever), 1)[0]
    assert ratio == pytest.approx((HEIGHT + 0.25) / HEIGHT, rel=0.05)
    assert np.allclose(gyro, ideal_measurements(motion, times)[0])   # rotation is rigid-body


def test_a_rotated_sensor_reports_in_its_own_axes():
    motion = motion_for("sway_pitch")
    times = np.linspace(1.0, 4.0, 50)
    turned = tf.make_transform(tf.rot_z(math.pi / 2.0))             # sensor x = body y
    gyro_body, force_body = ideal_measurements(motion, times)
    gyro, force = ideal_measurements(motion, times, turned)
    assert np.allclose(gyro @ turned[:3, :3].T, gyro_body, atol=1e-9)
    assert np.allclose(force @ turned[:3, :3].T, force_body, atol=1e-9)


# ------------------------------------------------------------------ imperfections
def test_the_stream_never_contains_an_orientation():
    assert {f.name for f in dataclasses.fields(ImuSample)} == {"timestamp", "gyro", "accel"}
    with pytest.raises(ValueError):
        ImuSample(0.0, np.zeros(2), np.zeros(3))


def test_noise_and_bias_have_the_configured_size():
    cfg = ImuConfig(rate_jitter=0.0, gyro_bias_walk=0.0, accel_bias_walk=0.0, latency_s=0.0,
                    latency_jitter_s=0.0)
    biases = []
    for seed in range(40):
        stream = simulate_imu(motion_for("rigid"), cfg, np.random.default_rng(seed), end=5.0)
        gyro_error, accel_error = stream.gyro, stream.accel - [0.0, 0.0, GRAVITY]
        biases.append(np.concatenate([gyro_error.mean(0), accel_error.mean(0)]))
        if seed == 0:
            assert (gyro_error - gyro_error.mean(0)).std() == pytest.approx(cfg.gyro_noise_std,
                                                                            rel=0.1)
            assert (accel_error - accel_error.mean(0)).std() == pytest.approx(
                cfg.accel_noise_std, rel=0.1)
    biases = np.array(biases)
    assert biases[:, :3].std() == pytest.approx(cfg.gyro_bias_std, rel=0.3)
    assert biases[:, 3:].std() == pytest.approx(cfg.accel_bias_std, rel=0.3)
    assert abs(biases.mean()) < 0.01


def test_bias_drifts_as_a_random_walk():
    cfg = ImuConfig(gyro_noise_std=0.0, gyro_bias_std=0.0, gyro_bias_walk=0.01, rate_jitter=0.0)
    spreads = []
    for duration in (5.0, 20.0):
        ends = [simulate_imu(motion_for("rigid"), cfg, np.random.default_rng(seed),
                             end=duration).gyro[-1] for seed in range(40)]
        spreads.append(np.std(ends))
    assert spreads[0] == pytest.approx(0.01 * math.sqrt(5.0), rel=0.3)
    assert spreads[1] / spreads[0] == pytest.approx(2.0, rel=0.3)        # grows as sqrt(t)


def test_rate_latency_dropouts_and_clock_offset():
    motion = motion_for("standing")
    nominal = simulate_imu(motion, ImuConfig(), np.random.default_rng(0), end=10.0)
    intervals = np.diff(nominal.true_time)
    assert 1.0 / intervals.mean() == pytest.approx(200.0, rel=0.01)
    assert intervals.std() / intervals.mean() == pytest.approx(0.02, rel=0.25)     # variable rate
    delay = nominal.arrival - nominal.true_time
    assert delay.mean() == pytest.approx(0.004, abs=0.0005) and (np.diff(nominal.arrival) >= 0).all()
    assert np.array_equal(nominal.timestamp, nominal.true_time)

    late = simulate_imu(motion, ImuConfig(latency_s=0.08, timestamp_offset_s=0.04),
                        np.random.default_rng(0), end=10.0)
    assert (late.arrival - late.true_time).mean() == pytest.approx(0.08, abs=0.001)
    assert np.allclose(late.timestamp - late.true_time, 0.04)       # the clock is off, silently

    lossy = simulate_imu(motion, ImuConfig.load("dropouts"), np.random.default_rng(0), end=30.0)
    assert 0.3 < lossy.lost_fraction < 0.5
    assert np.diff(lossy.true_time).max() > 0.09                    # an outage, not single losses
    assert len(lossy) + round(lossy.lost_fraction * lossy.generated) == lossy.generated


def test_stream_delivery_is_by_arrival_time_and_deterministic():
    motion = motion_for("walking", trajectory=tj.straight_line([-2.0, 0.0, 0.0], 2.0))
    a = simulate_imu(motion, ImuConfig.load("nominal"), np.random.default_rng(7))
    b = simulate_imu(motion, ImuConfig.load("nominal"), np.random.default_rng(7))
    assert np.array_equal(a.gyro, b.gyro) and np.array_equal(a.arrival, b.arrival)
    first = list(a.arrived_between(-math.inf, 0.5))
    second = list(a.arrived_between(0.5, 1.0))
    assert len(first) + len(second) == int((a.arrival <= 1.0).sum())
    assert first[-1].timestamp < second[0].timestamp
    assert all(type(sample) is ImuSample for sample in first[:3])
    with pytest.raises(ValueError, match="unknown ImuConfig keys"):
        ImuConfig.from_dict({"gps": True})


# ------------------------------------------------------------------ the tilt filter
def test_filter_converges_on_a_static_tilt_and_learns_the_gyro_bias():
    up = quat.up_from_tilt(math.radians(4.0), math.radians(-7.0))
    bias = np.array([0.012, -0.008, 0.0])
    estimator = TiltEstimator()
    for index in range(6000):
        estimate = estimator.update(ImuSample(0.005 * index, bias, GRAVITY * up))
    assert math.degrees(estimate.roll) == pytest.approx(4.0, abs=0.05)
    assert math.degrees(estimate.pitch) == pytest.approx(-7.0, abs=0.05)
    assert np.allclose(estimator.gyro_bias[:2], bias[:2], atol=5e-4)
    assert estimator.gyro_bias[2] == 0.0                        # yaw-rate bias: not observable
    assert estimate.accel_weight > 0.95


def test_filter_needs_the_gyro_between_accelerometer_corrections():
    """A fast tilt is followed at once through the gyro, long before the slow accelerometer
    feedback could have got there."""
    motion = motion_for(MotionProfile("fast", sway_pitch_deg=5.0, sway_pitch_hz=1.5))
    stream = simulate_imu(motion, ImuConfig.load("ideal"), np.random.default_rng(0), end=20.0)
    _, errors, _ = run_filter(stream)
    assert np.abs(errors[1600:]).max() < 0.8                    # the motion itself is +-5 deg
    _, blind, _ = run_filter(dataclasses.replace(stream, gyro=np.zeros_like(stream.gyro)))
    assert np.abs(blind[1600:]).max() > 3.0                     # accelerometer alone lags badly


def test_dynamic_acceleration_is_not_mistaken_for_tilt():
    """Walking in a straight line with a level torso: the accelerometer swings by several
    degrees of apparent tilt at every step, and the estimate must not follow it."""
    profile = MotionProfile("steps", gait_lateral_m=0.03, gait_vertical_m=0.01,
                            gait_surge_m=0.01, gait_always=True)
    motion = motion_for(profile, trajectory=tj.straight_line([-2.0, 0.0, 0.0], 4.0))
    stream = simulate_imu(motion, ImuConfig.load("ideal"), np.random.default_rng(0))
    apparent = np.degrees(np.arctan2(np.hypot(stream.accel[:, 0], stream.accel[:, 1]),
                                     stream.accel[:, 2]))
    assert apparent.max() > 6.0                                 # what a naive reading would say
    _, errors, _ = run_filter(stream)
    assert np.sqrt((errors[600:] ** 2).mean()) < 0.8
    assert np.abs(errors[600:]).max() < 0.3 * apparent.max()


@pytest.mark.parametrize("profile, limit", [("standing", 0.8), ("sway_pitch", 0.9),
                                            ("sway_roll", 0.9), ("walking", 1.3),
                                            ("disturbed", 0.9)])
def test_tilt_accuracy_with_the_nominal_imu(profile, limit):
    """RMS tilt error over several seeds, after the first three seconds. The floor is the
    accelerometer bias (5 mg is 0.3 deg), which no filter of this kind can see."""
    path = tj.PathBuilder([-2.0, -1.0, 0.0]).hold(2.0).forward(3.0).turn(1.5).forward(2.0) \
        .build("path") if "walk" in profile else STILL
    squared, ratios = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        motion = BodyMotion(path, MotionProfile.load(profile), HEIGHT, rng)
        stream = simulate_imu(motion, ImuConfig.load("nominal"), rng)
        _, errors, stds = run_filter(stream)
        squared.append((errors[600:] ** 2).mean())
        ratios.append(np.sqrt(((errors[600:] / stds[600:, None]) ** 2).mean()))
    assert math.sqrt(np.mean(squared)) < limit
    # the reported standard deviation is of the right size: neither blind nor useless
    assert 0.3 < np.mean(ratios) < 2.0


def test_estimated_tilt_beats_assuming_a_level_body_whenever_the_body_moves():
    for profile in ("sway_pitch", "walking", "walking_rough"):
        path = tj.straight_line([-3.0, 0.0, 0.0], 6.0) if "walk" in profile else STILL
        rng = np.random.default_rng(11)
        motion = BodyMotion(path, MotionProfile.load(profile), HEIGHT, rng)
        stream = simulate_imu(motion, ImuConfig.load("nominal"), rng)
        _, errors, _ = run_filter(stream)
        level = np.degrees(np.hypot(stream.truth_roll, stream.truth_pitch))[-len(errors):]
        assert np.sqrt((errors[600:] ** 2).sum(1).mean()) < 0.6 * np.sqrt((level[600:] ** 2).mean())


def test_uncertainty_grows_when_samples_are_missing_and_when_the_body_is_shaken():
    rng = np.random.default_rng(2)
    motion = BodyMotion(tj.straight_line([-3.0, 0.0, 0.0], 6.0), MotionProfile.load("walking"),
                        HEIGHT, rng)
    stream = simulate_imu(motion, ImuConfig.load("nominal"), rng)
    keep = (stream.true_time < 10.0) | (stream.true_time > 10.6)        # a 0.6 s outage
    gapped = dataclasses.replace(stream, **{name: getattr(stream, name)[keep] for name in (
        "timestamp", "arrival", "gyro", "accel", "true_time", "truth_roll", "truth_pitch")})
    estimator = TiltEstimator()
    before = after = None
    for index in range(len(gapped)):
        estimate = estimator.update(gapped.sample(index))
        if estimate is None:
            continue
        if estimate.timestamp < 10.0:
            before = estimate
        elif after is None:
            after = estimate
    assert estimator.dropouts == 1 and estimator.longest_gap > 0.55
    assert after.tilt_std > 2.0 * before.tilt_std and after.gap > 0.55
    # and it settles again
    assert estimator.latest.tilt_std < 1.5 * before.tilt_std

    calm = run_filter(simulate_imu(motion_for("standing"), ImuConfig.load("nominal"),
                                   np.random.default_rng(3)))[2]
    assert before.tilt_std > 1.3 * math.radians(calm[2000:].mean())


def test_filter_refuses_what_it_cannot_use():
    estimator = TiltEstimator()
    with pytest.raises(TypeError):
        estimator.update((0.0, np.zeros(3), np.zeros(3)))
    still = [0.0, 0.0, GRAVITY]
    for index in range(20):
        estimator.update(ImuSample(0.005 * index, np.zeros(3), still))
    newest = estimator.latest
    assert estimator.update(ImuSample(0.05, np.zeros(3), still)) is None      # out of order
    assert estimator.update(ImuSample(0.2, [np.nan, 0, 0], still)) is None
    assert estimator.update(ImuSample(0.2, np.zeros(3), np.zeros(3))) is None
    assert estimator.rejected == 3 and estimator.latest is newest
    with pytest.raises(ValueError):
        TiltFilterConfig(kp=0.0)


def test_a_sensor_mounted_at_an_angle_needs_its_mounting_rotation():
    motion = motion_for("sway_pitch")
    turned = tf.make_transform(tf.rot_z(math.pi / 2.0))
    stream = simulate_imu(motion, ImuConfig.load("ideal"), np.random.default_rng(0), turned)
    _, right, _ = run_filter(stream, R_body_sensor=turned[:3, :3])
    _, wrong, _ = run_filter(stream)
    assert np.abs(right[1500:]).max() < 0.6
    assert np.abs(wrong[1500:]).max() > 3.0         # pitch is reported as roll


# ------------------------------------------------------------------ synchronisation
def _history(rate=math.radians(40.0), period=0.005, count=200, cfg=None) -> OrientationHistory:
    """A body pitching at a constant rate, sampled every ``period``."""
    history = OrientationHistory(cfg)
    for index in range(count):
        time = period * index
        orientation = quat.from_rotation_vector([0.0, rate * time, 0.0])
        history.add(OrientationEstimate(time, orientation, 0.0, rate * time, math.radians(0.3),
                                        np.array([0.0, rate, 0.0]), 1.0, period))
    return history


def test_interpolation_gives_the_orientation_at_the_image_timestamp():
    rate = math.radians(40.0)
    history = _history(rate)
    for time in (0.3012, 0.5, 0.74999, 0.0025):
        tilt = history.at(time)
        assert tilt.method == INTERPOLATED
        assert tilt.pitch == pytest.approx(rate * time, abs=1e-9)
        assert tilt.gap == pytest.approx(0.005) and abs(tilt.offset) <= 0.0025 + 1e-9


def test_the_newest_sample_is_the_wrong_orientation_for_an_older_image():
    rate = math.radians(40.0)
    history = _history(rate)
    image_time = 0.995 - 0.030                       # the image is 30 ms old when processed
    naive, proper = history.latest(image_time), history.at(image_time)
    assert naive.method == LATEST and naive.newest_age == pytest.approx(0.030)
    assert math.degrees(naive.pitch - rate * image_time) == pytest.approx(40.0 * 0.030, rel=1e-6)
    assert abs(proper.pitch - rate * image_time) < 1e-9


def test_extrapolation_holding_and_stale_requests_say_so_and_cost_uncertainty():
    rate = math.radians(40.0)
    history = _history(rate)
    newest = 0.995
    inside = history.at(0.5)
    ahead = history.at(newest + 0.03)
    assert ahead.method == EXTRAPOLATED and ahead.pitch == pytest.approx(rate * (newest + 0.03))
    assert ahead.tilt_std > inside.tilt_std and ahead.newest_age == pytest.approx(-0.03)
    far = history.at(newest + 0.4)
    assert far.method == HELD and far.pitch == pytest.approx(rate * newest)
    assert far.tilt_std > ahead.tilt_std
    old = history.at(-1.0)
    assert old.method == STALE and old.tilt_std > inside.tilt_std
    empty = OrientationHistory().at(0.0)
    assert empty.method == NONE and not empty.available and math.isinf(empty.tilt_std)


def test_history_is_bounded_and_counts_what_happened():
    history = _history(count=1000, cfg=SyncConfig(history_s=1.0))
    assert len(history) == pytest.approx(200, abs=2)
    for time in (4.5, 4.9, 4.995 + 0.02, 4.995 + 1.0):
        history.at(time)
    history.latest(4.9)
    history.at(4.7, record=False)
    summary = history.statistics.summary()
    assert summary["frames"] == 5
    assert summary["share_interpolated"] == pytest.approx(0.4)
    assert summary["share_extrapolated"] == summary["share_held"] == pytest.approx(0.2)
    assert summary["max_bracket_gap_ms"] == pytest.approx(5.0)


def test_a_sparse_bracket_is_flagged_and_priced():
    history = OrientationHistory()
    rate = np.array([0.0, 0.5, 0.0])
    for time in (0.0, 0.005, 0.2, 0.205):                       # a 195 ms hole
        history.add(OrientationEstimate(time, quat.IDENTITY, 0.0, 0.0, 0.005, rate, 1.0, 0.005))
    tight, wide = history.at(0.002), history.at(0.1)
    assert history.statistics.sparse == 1
    assert wide.gap == pytest.approx(0.195) and wide.tilt_std > 3.0 * tight.tilt_std
    with pytest.raises(TypeError):
        history.add("not an estimate")


# ------------------------------------------------------------------ the camera pose
def _calibration() -> EstimatedCameraCalibration:
    cfg = StereoCameraConfig.load("zed_mini_sim").with_resolution(320, 180)
    return EstimatedCameraCalibration.from_camera_config(cfg)


def _estimator(mount=None, **fusion) -> CameraPoseEstimator:
    return CameraPoseEstimator(_calibration(), mount or CameraMount.load("head_mid"), BODY, 0.063,
                               FusionConfig(**fusion))


def test_fixed_mode_is_the_level_body_of_phase_3():
    estimator = _estimator(mode="fixed")
    estimate = estimator.estimate(1.0)
    mount = CameraMount.load("head_mid")
    assert estimate.source == "nominal" and math.isnan(estimate.tilt_std)
    expected = mg.T_footprint_rig(mount, BODY, HEIGHT) @ mg.T_rig_optical(0.063, "left")
    assert np.allclose(estimate.pose.T_base_camera, expected)
    assert estimate.camera_height == pytest.approx(mount.standing_height(BODY))
    roll, pitch = estimate.camera_roll_pitch
    assert roll == pytest.approx(0.0, abs=1e-12) and pitch == pytest.approx(math.radians(25.0))
    assert estimate.calibration is estimator.calibration


def _feed(estimator, stream, until):
    for sample in stream.arrived_between(-math.inf, until):
        estimator.add_imu(sample)


def test_imu_mode_reproduces_the_true_camera_pose_of_a_swaying_body():
    motion = motion_for("sway_pitch")
    stream = simulate_imu(motion, ImuConfig.load("ideal"), np.random.default_rng(0))
    mount = CameraMount.load("head_mid")
    estimator, fixed = _estimator(mount), _estimator(mount, mode="fixed")
    errors, fixed_errors = [], []
    for step in range(60, 250):
        time = 0.1 * step
        _feed(estimator, stream, time + 0.03)
        truth = motion.sample(time)
        true_camera = mg.T_footprint_rig(mount, BODY, truth.height[0], truth.roll[0],
                                         truth.pitch[0]) @ mg.T_rig_optical(0.063)
        estimate = estimator.estimate(time)
        assert estimate.source == "imu" and estimate.sync.method == INTERPOLATED
        true_pitch = camera_roll_pitch(true_camera)[1]
        errors.append(estimate.camera_roll_pitch[1] - true_pitch)
        fixed_errors.append(fixed.estimate(time).camera_roll_pitch[1] - true_pitch)
    # What is left is the pendulum's own acceleration leaking through the accelerometer
    # feedback: about a tenth of the +-4 deg the fixed camera model gets wrong.
    assert np.degrees(np.abs(errors)).max() < 0.6
    assert np.degrees(np.abs(fixed_errors)).max() > 3.5


def test_before_any_imu_sample_the_estimator_falls_back_and_admits_it():
    plain, aware = _estimator(), _estimator(use_uncertainty=True)
    for estimator in (plain, aware):
        estimate = estimator.estimate(0.0)
        assert estimate.source == "nominal_fallback" and estimate.sync.method == NONE
        assert np.allclose(estimate.pose.T_base_camera, estimator.nominal_pose().T_base_camera)
    assert plain.estimate(0.0).calibration.uncertainty == plain.calibration.uncertainty
    assert aware.estimate(0.0).calibration.uncertainty.pitch_std_deg > 5.0


def test_uncertainty_mode_widens_the_calibration_by_the_tilt_uncertainty():
    rng = np.random.default_rng(1)
    motion = BodyMotion(tj.straight_line([-3.0, 0.0, 0.0], 6.0), MotionProfile.load("walking"),
                        HEIGHT, rng)
    stream = simulate_imu(motion, ImuConfig.load("nominal"), rng)
    plain, aware = _estimator(), _estimator(use_uncertainty=True)
    for estimator in (plain, aware):
        _feed(estimator, stream, 10.0)
    base = plain.calibration.uncertainty
    a, b = plain.estimate(9.9), aware.estimate(9.9)
    assert np.array_equal(a.pose.T_base_camera, b.pose.T_base_camera)    # same pose, other trust
    assert a.calibration.uncertainty == base
    widened = b.calibration.uncertainty
    assert widened.pitch_std_deg == pytest.approx(
        math.hypot(base.pitch_std_deg, math.degrees(b.tilt_std)))
    assert widened.roll_std_deg == pytest.approx(widened.pitch_std_deg)
    assert widened.height_std > base.height_std                  # the body bobs while it walks
    assert widened.yaw_std_deg == base.yaw_std_deg and widened.pixel_std == base.pixel_std


def test_an_imu_cannot_see_a_wrong_mount_calibration():
    """The body's tilt is estimated perfectly, the mount is believed to be where it is not:
    the camera pitch is wrong by exactly the mount error, in every frame. Nothing in the
    IMU's data can change that."""
    motion = motion_for("sway_pitch")
    stream = simulate_imu(motion, ImuConfig.load("ideal"), np.random.default_rng(0))
    believed = CameraMount.load("head_mid")
    true_mount = believed.shifted(pitch=math.radians(2.0))
    estimator = _estimator(believed)
    errors = []
    for step in range(60, 200):
        time = 0.1 * step
        _feed(estimator, stream, time + 0.03)
        truth = motion.sample(time)
        true_camera = mg.T_footprint_rig(true_mount, BODY, truth.height[0], truth.roll[0],
                                         truth.pitch[0]) @ mg.T_rig_optical(0.063)
        errors.append(math.degrees(estimator.estimate(time).camera_roll_pitch[1]
                                   - camera_roll_pitch(true_camera)[1]))
    assert np.mean(errors) == pytest.approx(-2.0, abs=0.1) and np.std(errors) < 0.4
    # a correction from OUTSIDE the IMU (the calibration monitor) does remove it
    estimator.set_mount_correction(0.0, math.radians(2.0))
    corrected = estimator.estimate(19.9)
    truth = motion.sample(19.9)
    true_camera = mg.T_footprint_rig(true_mount, BODY, truth.height[0], truth.roll[0],
                                     truth.pitch[0]) @ mg.T_rig_optical(0.063)
    assert corrected.camera_roll_pitch[1] == pytest.approx(camera_roll_pitch(true_camera)[1],
                                                           abs=math.radians(0.6))


def test_latest_sync_and_configuration_errors():
    motion = motion_for("sway_pitch")
    stream = simulate_imu(motion, ImuConfig.load("ideal"), np.random.default_rng(0))
    naive = _estimator(sync="latest")
    _feed(naive, stream, 10.03)
    estimate = naive.estimate(10.0)
    assert estimate.source == "imu_latest" and estimate.sync.method == LATEST
    assert estimate.sync.newest_age == pytest.approx(0.03, abs=0.006)
    assert type(estimate.pose) is EstimatedCameraPose
    for bad in (dict(mode="gps"), dict(sync="nearest")):
        with pytest.raises(ValueError):
            FusionConfig(**bad)
    with pytest.raises(ValueError, match="unknown FusionConfig keys"):
        FusionConfig.from_dict({"oracle": True})
    with pytest.raises(TypeError):
        CameraPoseEstimator("calibration", CameraMount.load("head_mid"), BODY, 0.063)
