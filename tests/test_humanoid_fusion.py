"""Localization on a moving body: the Phase 4 runner and benchmark, uncertainty-aware
projection, the mount-calibration monitor, and ground-truth isolation of the new estimators.

A colour-threshold stand-in replaces the networks, so the runs here use the real simulator, the
real IMU chain and the real localizer and still take seconds.
"""
import dataclasses
import json
import math

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mujoco")

import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.estimation.calibration_monitor import (  # noqa: E402
    CalibrationMonitor, MonitorConfig)
from robosoccer_perception.estimation.camera_pose import FusionConfig  # noqa: E402
from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.field import FieldGeometry  # noqa: E402
from robosoccer_perception.geometry.ground_projection import (  # noqa: E402
    CalibrationUncertainty, EstimatedCameraPose)
from robosoccer_perception.geometry.robot_description import robot_is_installed  # noqa: E402
from robosoccer_perception.humanoid_benchmark import benchmark, runner, scenarios  # noqa: E402
from robosoccer_perception.humanoid_benchmark import setup as fusion_setup  # noqa: E402
from robosoccer_perception.localization.motion_model import OdometryDelta  # noqa: E402
from robosoccer_perception.perception.config import MeasurementNoiseConfig  # noqa: E402
from robosoccer_perception.perception.extractor import GeometricFeatureExtractor  # noqa: E402
from robosoccer_perception.perception.uncertainty import shared_error_inflation  # noqa: E402
from robosoccer_perception.pipeline.vision_localization import (  # noqa: E402
    VisionLocalizationPipeline)
from robosoccer_perception.simulation import trajectories as tj  # noqa: E402
from robosoccer_perception.vision_benchmark.runner import (  # noqa: E402
    BenchmarkResources, EstimateError)

needs_humanoid = pytest.mark.skipif(not robot_is_installed("humanoid_prototype_1"),
                                    reason="humanoid model not imported")
RENDER = [pytest.mark.render, pytest.mark.slow, needs_humanoid]


def _resources() -> BenchmarkResources:
    return BenchmarkResources(helpers.ColourVision(), resolution=(320, 180),
                              filter_overrides={"num_particles": 300})


def _rmse(values, skip: int = 20) -> float:
    values = np.asarray(values, dtype=float)[skip:]
    return float(np.sqrt(np.nanmean(values ** 2)))


# ------------------------------------------------------------------ scenarios and arms
def test_scenarios_cover_the_required_conditions():
    all_scenarios = scenarios.select_scenarios(None)
    motions = {s.motion for s in all_scenarios}
    assert {"standing", "sway_pitch", "sway_roll", "walking", "disturbed",
            "walking_disturbed"} <= motions
    assert {s.mount for s in all_scenarios} >= {"head_low", "head_mid", "head_high", "head_tall",
                                                "legacy_rig"}
    assert {s.robot for s in all_scenarios} == {"humanoid_prototype_1", "placeholder"}
    assert any(s.mount_error.pitch_deg for s in all_scenarios)
    assert any(s.imu == "dropouts" for s in all_scenarios)
    assert any(s.imu_overrides.get("timestamp_offset_s") for s in all_scenarios)
    assert any("turn" in s.id for s in all_scenarios)
    assert len({s.id for s in all_scenarios}) == len(all_scenarios)
    assert scenarios.select_scenarios(["h05"])[0].id == "h05_walk_rectangle"
    with pytest.raises(KeyError):
        scenarios.get_scenario("h99")


def test_arms_are_what_the_benchmark_says_they_are():
    names = [arm.name for arm in runner.DEFAULT_ARMS]
    assert names[:4] == ["fixed", "oracle_orientation", "imu", "imu_uncertainty"]
    assert [arm.name for arm in runner.DEFAULT_ARMS if arm.is_oracle] == ["oracle_orientation"]
    assert "ORACLE" in runner.ORACLE.description and "truth" in runner.ORACLE.description
    base = FusionConfig()
    assert runner.FIXED.fusion(base).mode == "fixed"
    assert runner.IMU.fusion(base).use_uncertainty is False
    assert runner.IMU_UNCERTAINTY.fusion(base).use_uncertainty is True
    assert runner.IMU_LATEST.fusion(base).sync == "latest"
    with pytest.raises(ValueError):
        runner.FusionArm("x", mode="gps")
    arms, index = benchmark.sweep_arms({"pitch_deg": (-2.0, 0.0, 2.0)}, (runner.IMU,))
    assert [arm.name for arm in arms] == ["imu@0", "imu@pitch_deg=-2", "imu@pitch_deg=2"]
    assert arms[1].estimate_error == EstimateError(pitch_deg=-2.0)
    assert index["imu@pitch_deg=2"] == ("imu", "pitch_deg", 2.0)


def test_settings_load_on_top_of_the_phase_3_settings():
    settings = fusion_setup.load_settings()
    assert settings["detector"].endswith("best.pt") and settings["resolution"] == [640, 360]
    assert settings["calibration_sweep"]["parameters"]["pitch_deg"] == [
        -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert settings["calibration_sweep"]["parameters"]["roll_deg"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert settings["calibration_sweep"]["parameters"]["height_m"] == [
        -0.05, -0.02, 0.0, 0.02, 0.05]
    assert isinstance(fusion_setup.fusion_config(settings), FusionConfig)
    assert isinstance(fusion_setup.monitor_config(settings), MonitorConfig)


# ------------------------------------------------------------------ uncertainty-aware projection
def test_shared_error_inflation_counts_the_terms_that_share_the_camera_error():
    off = MeasurementNoiseConfig()
    on = dataclasses.replace(off, shared_camera_error=True)
    assert shared_error_inflation(60, off) == 1.0                 # Phase 3 behaviour
    assert shared_error_inflation(2, on) == 1.0                   # never below one
    assert shared_error_inflation(20, on) == pytest.approx(0.3 * 20)
    assert shared_error_inflation(60, on) == pytest.approx(0.3 * 40)   # the filter uses 40
    capped = dataclasses.replace(on, shared_error_max_inflation=5.0)
    assert shared_error_inflation(60, capped) == 5.0


def test_shared_error_model_only_widens_and_leaves_phase_3_untouched(s_field):
    calibration = helpers.small_calibration()
    pose = calibration.pose()
    _, vision = helpers.analytic_view(s_field, [-1.5, 0.4, 0.3], calibration, pose)
    plain = GeometricFeatureExtractor(s_field).extract(vision, calibration, pose)
    again = GeometricFeatureExtractor(s_field, noise=MeasurementNoiseConfig()).extract(
        vision, calibration, pose)
    aware = GeometricFeatureExtractor(s_field, noise=dataclasses.replace(
        MeasurementNoiseConfig(), shared_camera_error=True)).extract(vision, calibration, pose)
    assert np.array_equal(plain.line_points.std, again.line_points.std)
    assert "shared_error_inflation" not in plain.diagnostics["lines"]
    factor = aware.diagnostics["lines"]["shared_error_inflation"]
    assert factor > 3.0
    # the same points, the same pixel and camera parts; only the reported value is wider,
    # most of all where the camera part dominates (far points)
    common = min(len(plain.line_points), len(aware.line_points))
    assert common >= 10 and len(aware.line_points) <= len(plain.line_points)
    lookup = {tuple(p): i for i, p in enumerate(np.round(plain.line_points.pixels, 3))}
    ratios, distances = [], []
    for j, pixel in enumerate(np.round(aware.line_points.pixels, 3)):
        i = lookup[tuple(pixel)]
        assert aware.line_points.camera_std[j] == pytest.approx(plain.line_points.camera_std[i])
        ratios.append(aware.line_points.std[j] / plain.line_points.std[i])
        distances.append(np.linalg.norm(aware.line_points.points[j]))
    ratios, distances = np.array(ratios), np.array(distances)
    assert (ratios >= 1.0 - 1e-9).all() and ratios.max() < math.sqrt(factor) + 1e-9
    assert ratios[distances > np.median(distances)].mean() > \
        ratios[distances <= np.median(distances)].mean()


def test_a_frame_never_claims_more_about_the_camera_than_the_robot_knows(s_field):
    """The point of the shared-error model: summed over the points a localizer uses, the
    information about a common pitch error stays below what the pitch estimate itself has."""
    trust = CalibrationUncertainty(pixel_std=1.0, height_std=0.0, pitch_std_deg=1.0,
                                   roll_std_deg=0.0, yaw_std_deg=0.0, focal_rel_std=0.0)
    calibration = helpers.small_calibration().replace(uncertainty=trust)
    pose = calibration.pose()
    _, vision = helpers.analytic_view(s_field, [-1.5, 0.4, 0.3], calibration, pose)
    information = {}
    for shared in (False, True):
        noise = dataclasses.replace(MeasurementNoiseConfig(), shared_camera_error=shared,
                                    line_std_max=10.0)
        points = GeometricFeatureExtractor(s_field, noise=noise).extract(
            vision, calibration, pose).line_points
        used = min(len(points), noise.shared_error_max_points)
        order = np.argsort(points.std)[:used]
        # d(point) / d(pitch) along the line normal, in metres per radian, is camera_std / std(pitch)
        sensitivity = points.camera_std[order] / math.radians(1.0)
        information[shared] = noise.shared_error_point_weight * float(
            np.sum(sensitivity ** 2 / points.std[order] ** 2))
    prior = 1.0 / math.radians(1.0) ** 2
    assert information[False] > 1.5 * prior          # independent treatment: overconfident
    assert information[True] <= prior * 1.001        # shared treatment: bounded by the prior


# ------------------------------------------------------------------ the calibration monitor
def _monitored_run(field, pitch_error_deg: float, steps: int = 240, monitor_cfg=None,
                   correct: bool = True):
    """A robot walking a rectangle whose camera is really pitched ``pitch_error_deg`` further
    down than it believes. Analytic images, exact odometry; the monitor sees image-derived
    features and the localizer's estimate only."""
    calibration = helpers.small_calibration()
    believed = calibration.pose()
    true_camera = believed.perturbed(d_pitch=math.radians(pitch_error_deg))
    path = tj.rectangle([-1.6, -1.1, 0.0], 3.2, 2.2, speed=0.5, turn_rate=math.radians(60.0))
    pipeline = VisionLocalizationPipeline(helpers.ColourVision(), field, seed=1)
    pipeline.init_gaussian(path.poses[0] + [0.05, -0.04, 0.03], 0.2, math.radians(6.0))
    monitor = CalibrationMonitor(field, monitor_cfg)
    states, errors = [], []
    for step in range(min(steps, len(path))):
        pose = path.poses[step]
        image, _ = helpers.analytic_view(field, pose, calibration, true_camera)
        delta = None
        if step:
            dx, dy, dyaw = tf.se2_relative(path.poses[step - 1], pose)
            delta = OdometryDelta(float(dx), float(dy), float(dyaw), 0.1 * step, 0.1,
                                  np.diag([1e-4, 1e-4, 1e-4]))
        camera = believed
        if correct:
            roll, pitch = monitor.correction
            camera = believed.perturbed(d_roll=roll, d_pitch=pitch)
        output = pipeline.step(image, 0.1 * step, delta, calibration, camera)
        states.append(monitor.update(output.features, output.estimate, calibration, camera))
        errors.append(np.linalg.norm(output.pose[:2] - pose[:2]))
    return states, np.array(errors)


@pytest.mark.slow
def test_monitor_finds_a_persistent_camera_pitch_bias_from_field_lines(s_field):
    states, errors = _monitored_run(s_field, 2.0)
    final = states[-1]
    assert final.status == "correcting" and final.warning
    assert final.bias_deg[1] == pytest.approx(2.0, abs=0.6)       # (roll, pitch)
    assert abs(final.bias_deg[0]) < 0.8
    assert final.correction_deg[1] == pytest.approx(2.0, abs=0.7)
    assert final.bias_std_deg[1] < 0.35 and final.frames >= 100
    assert all(s.correction_deg[1] <= final.correction_deg[1] + 0.3 for s in states)   # no overshoot
    # the correction pays: late in the run the pose is much better than without it
    _, uncorrected = _monitored_run(s_field, 2.0, correct=False)
    assert errors[-60:].mean() < 0.5 * uncorrected[-60:].mean()
    assert states[0].status == "collecting" and not states[0].warning


@pytest.mark.slow
def test_monitor_stays_quiet_when_the_calibration_is_right(s_field):
    states, _ = _monitored_run(s_field, 0.0)
    final = states[-1]
    assert final.status == "ok" and not final.warning
    assert np.abs(final.bias_deg).max() < 0.5 and np.abs(final.correction_deg).max() < 0.3
    assert final.residual_rms < 0.05                              # the lines fit


@pytest.mark.slow
def test_monitor_correction_is_bounded_and_can_be_switched_to_warning_only(s_field):
    cfg = MonitorConfig(max_correction_deg=1.0)
    bounded, _ = _monitored_run(s_field, 3.0, monitor_cfg=cfg)
    assert max(abs(s.correction_deg[1]) for s in bounded) <= 1.0 + 1e-9
    warned, _ = _monitored_run(s_field, 2.0, monitor_cfg=MonitorConfig(enabled_correction=False))
    assert warned[-1].status == "bias_suspected" and warned[-1].correction_deg == (0.0, 0.0)
    # Without the correction a large bias is only partly seen: a lower bound, with the right sign.
    assert 0.5 < warned[-1].bias_deg[1] < 2.0


def test_monitor_learns_nothing_from_a_filter_that_has_not_converged(s_field):
    calibration = helpers.small_calibration()
    pose = calibration.pose()
    image, _ = helpers.analytic_view(s_field, [-1.0, 0.3, 0.2], calibration, pose)
    pipeline = VisionLocalizationPipeline(helpers.ColourVision(), s_field, seed=0)
    pipeline.init_uniform()                                      # no idea where it is
    monitor = CalibrationMonitor(s_field)
    output = pipeline.step(image, 0.0, None, calibration, pose)
    assert not output.estimate.converged
    state = monitor.update(output.features, output.estimate, calibration, pose)
    assert not state.used and state.frames == 0 and state.status == "collecting"
    assert monitor.correction == (0.0, 0.0)
    with pytest.raises(TypeError):
        monitor.update(output.features, output.estimate, calibration, pose.T_base_camera)
    with pytest.raises(TypeError):
        CalibrationMonitor("s_field")
    with pytest.raises(ValueError, match="unknown MonitorConfig keys"):
        MonitorConfig.from_dict({"ground_truth": True})


# ------------------------------------------------------------------ runs with the simulator
@pytest.fixture(scope="module")
def sway_run(gl_context):
    scenario = scenarios.get_scenario("h02_sway_pitch")
    return runner.run_humanoid_scenario(scenario, 0, _resources(), max_steps=110)


class TestRunner:
    pytestmark = RENDER

    def test_localization_under_camera_motion_needs_the_imu(self, sway_run):
        arms = sway_run.arms
        assert set(arms) == {arm.name for arm in runner.DEFAULT_ARMS}
        # the body really pitches, and a fixed camera model is wrong by that much
        assert 2.0 < sway_run.true_body_pitch_deg.std() < 4.0
        assert _rmse(arms["fixed"].camera_pitch_error_deg) > 2.0
        assert _rmse(arms["imu"].camera_pitch_error_deg) < 0.8
        assert _rmse(arms["oracle_orientation"].camera_pitch_error_deg) < 1e-6
        # ... which shows in where the ground points land, and in the pose
        assert np.nanmean(arms["imu"].projection_error[20:]) \
            < 0.4 * np.nanmean(arms["fixed"].projection_error[20:])
        fixed, imu, oracle = (_rmse(arms[name].result.position_error)
                              for name in ("fixed", "imu", "oracle_orientation"))
        assert imu < 0.6 * fixed and oracle <= imu + 0.02 and imu < 0.12
        assert arms["fixed"].sources == {"nominal": 110}
        assert arms["oracle_orientation"].sources == {"oracle": 110}
        assert arms["imu"].sources["imu"] >= 108

    def test_estimated_and_true_tilt_are_reported_with_an_honest_uncertainty(self, sway_run):
        arm = sway_run.arms["imu_uncertainty"]
        error = arm.body_pitch_deg - sway_run.true_body_pitch_deg
        assert _rmse(error, 30) < 0.8
        assert 0.2 < np.nanmean(arm.tilt_std_deg[30:]) < 1.5
        assert _rmse(error[30:] / arm.tilt_std_deg[30:], 0) < 2.5
        assert np.isnan(sway_run.arms["fixed"].tilt_std_deg).all()
        # uncertainty-aware: the filter reports a wider, more honest spread
        plain, aware = sway_run.arms["imu"].result, arm.result
        assert aware.position_std[30:].mean() > plain.position_std[30:].mean()
        assert sway_run.arms["imu"].sync["share_interpolated"] > 0.97
        assert sway_run.imu["samples"] > 2000 and sway_run.imu["lost_fraction"] == 0.0

    def test_runs_are_deterministic_and_arms_do_not_influence_each_other(self, sway_run):
        scenario = scenarios.get_scenario("h02_sway_pitch")
        alone = runner.run_humanoid_scenario(scenario, 0, _resources(),
                                             (runner.IMU, runner.FIXED), max_steps=60)
        for name in ("imu", "fixed"):
            assert np.array_equal(alone.arms[name].result.estimate,
                                  sway_run.arms[name].result.estimate[:60])
        assert np.array_equal(alone.truth, sway_run.truth[:60])
        other = runner.run_humanoid_scenario(scenario, 1, _resources(), (runner.IMU,),
                                             max_steps=30)
        assert not np.allclose(other.truth, sway_run.truth[:30])

    def test_a_mount_error_is_a_bias_the_imu_arm_keeps(self, gl_context):
        # an ideal IMU, so that what is left is the mount error and nothing else
        scenario = scenarios.get_scenario("h10_mount_pitch_error").replace(motion="standing",
                                                                         imu="ideal")
        run = runner.run_humanoid_scenario(scenario, 0, _resources(),
                                           (runner.ORACLE, runner.IMU), max_steps=60)
        bias = np.nanmean(run.arms["imu"].camera_pitch_error_deg[20:])
        assert bias == pytest.approx(-2.0, abs=0.6)       # estimate minus truth: believes less
        assert _rmse(run.arms["oracle_orientation"].camera_pitch_error_deg) < 1e-6
        assert np.nanmean(run.arms["imu"].projection_error[20:]) > 0.25

    def test_the_placeholder_platform_runs_through_the_same_pipeline(self, gl_context):
        scenario = scenarios.get_scenario("h18_placeholder_walk")
        run = runner.run_humanoid_scenario(scenario, 0, _resources(),
                                           (runner.IMU, runner.FIXED), max_steps=50)
        assert run.true_camera_height.mean() == pytest.approx(0.90, abs=0.02)
        assert _rmse(run.arms["imu"].result.position_error) < 0.25

    def test_benchmark_writes_its_report(self, gl_context, tmp_path):
        chosen = [scenarios.get_scenario("h01_standing")]
        report = benchmark.run_fusion_benchmark(
            chosen, [0], _resources(), tmp_path, arms=(runner.FIXED, runner.ORACLE, runner.IMU),
            max_steps=40, plots=True)
        entry = report["scenarios"]["h01_standing"]
        assert set(entry["arms"]) == {"fixed", "oracle_orientation", "imu"}
        aggregate = entry["arms"]["imu"]["aggregate"]
        for key in ("position_rmse_all", "yaw_rmse_all", "camera_pitch_rmse_deg",
                    "camera_roll_rmse_deg", "projection_error_mean_m", "position_consistency",
                    "position_within_2std", "divergence_rate", "overconfident_fraction",
                    "tilt_consistency", "body_pitch_rmse_deg"):
            assert key in aggregate, key
        assert entry["timing"]["fps"] > 0 and "orientation" in entry["timing"]
        assert report["meta"]["arms"]["oracle_orientation"]["uses_ground_truth"] is True
        assert report["meta"]["arms"]["imu"]["uses_ground_truth"] is False
        assert "no head" in report["meta"]["synthetic_only"]
        assert entry["scenario"]["camera_height_standing_m"] == pytest.approx(0.799, abs=0.01)
        for name in ("results.json", "summary.md", "summary.csv", "plots/overview.png",
                     "plots/run_h01_standing.png"):
            assert (tmp_path / name).is_file(), name
        summary = (tmp_path / "summary.md").read_text()
        assert "Synthetic only" in summary and "uses simulator ground truth" in summary
        json.loads((tmp_path / "results.json").read_text())


# ------------------------------------------------------------------ ground-truth isolation
class TestIsolation:
    pytestmark = RENDER

    def test_what_crosses_to_the_robot_carries_no_pose_and_no_orientation(self):
        assert [f.name for f in dataclasses.fields(runner.SensorPacket)] == [
            "image", "timestamp", "odometry", "imu"]
        assert {f.name for f in dataclasses.fields(runner.RobotKnowledge)} == {
            "field", "landmarks", "calibration", "mount", "robot", "baseline"}

    def test_the_robot_is_told_its_believed_mount_never_the_true_one(self, gl_context):
        scenario = scenarios.get_scenario("h10_mount_pitch_error")
        with runner.HumanoidSimulation(scenario, 0, _resources(), max_steps=3) as sim:
            knowledge = sim.knowledge()
            assert knowledge.mount == sim.believed_mount != sim.true_mount
            assert math.degrees(sim.true_mount.rpy[1] - knowledge.mount.rpy[1]) == \
                pytest.approx(2.0)
            assert knowledge.calibration.nominal_pitch == pytest.approx(math.radians(25.0))
            packet, truth, _ = sim.sense(0)
            assert type(packet) is runner.SensorPacket
            true_pitch = truth.camera_roll_pitch[1]
            assert math.degrees(true_pitch) == pytest.approx(27.0, abs=1.0)

    def test_an_estimator_refuses_truth_and_the_oracle_is_only_for_the_oracle_arm(self, gl_context):
        scenario = scenarios.get_scenario("h01_standing")
        resources = _resources()
        with runner.HumanoidSimulation(scenario, 0, resources, max_steps=3) as sim:
            knowledge = sim.knowledge()
            packet, truth, _ = sim.sense(0)
            imu_arm = runner.ArmEstimator(runner.IMU, knowledge, resources, resources.vision, 1)
            oracle_arm = runner.ArmEstimator(runner.ORACLE, knowledge, resources,
                                             resources.vision, 1)
            for arm in (imu_arm, oracle_arm):
                arm.initialise(truth.footprint, 0.3, 0.2)
            oracle = runner.oracle_camera_pose(truth, imu_arm.camera.nominal_pose())
            with pytest.raises(ValueError, match="oracle"):
                imu_arm.step(packet, oracle_pose=oracle)
            with pytest.raises(ValueError, match="oracle"):
                oracle_arm.step(packet)
            with pytest.raises(TypeError):
                imu_arm.step(truth)                              # truth is not a sensor packet
            assert type(oracle) is EstimatedCameraPose
            assert np.allclose(oracle.T_base_camera[:3, :3], truth.T_footprint_optical[:3, :3])
            assert np.allclose(oracle.T_base_camera[:3, 3],
                               imu_arm.camera.nominal_pose().T_base_camera[:3, 3])

    def test_recorded_sensor_packets_alone_reproduce_the_estimates(self, gl_context):
        """Record what the simulator hands over, close the simulator, replay into a fresh
        estimator: the estimates are identical, so nothing else can have reached them."""
        scenario = scenarios.get_scenario("h04_walk_straight")
        resources = _resources()
        arm = runner.IMU_SELFCAL
        with runner.HumanoidSimulation(scenario, 2, resources, max_steps=45) as sim:
            knowledge, seed = sim.knowledge(), 77
            guess = sim.start_guess()
            live = runner.ArmEstimator(arm, knowledge, resources, resources.vision, seed)
            live.initialise(*guess)
            packets, estimates = [], []
            for step in range(sim.total_steps):
                packet, _truth, _ = sim.sense(step)
                packets.append(packet)
                estimates.append(live.step(packet).localization.pose.copy())
        replay = runner.ArmEstimator(arm, knowledge, resources, helpers.ColourVision(), seed)
        replay.initialise(*guess)
        replayed = [replay.step(packet).localization.pose for packet in packets]
        assert np.array_equal(np.array(replayed), np.array(estimates))
        assert sum(len(packet.imu) for packet in packets) > 600      # the IMU was really used
