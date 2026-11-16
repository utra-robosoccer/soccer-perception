"""Scenarios, metrics, the benchmark runner, and the ground-truth firewall around the filter."""
import csv
import json
import math

import numpy as np
import pytest

from robosoccer_perception.evaluation import benchmark, metrics
from robosoccer_perception.evaluation.benchmark import RunResult, build_trajectory, run_scenario
from robosoccer_perception.evaluation.metrics import MetricThresholds, compute_metrics
from robosoccer_perception.evaluation.scenarios import SCENARIOS, get_scenario, select_scenarios
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.localization.estimation import mirror_pose
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.localization.particle_filter import ParticleFilter
from robosoccer_perception.observations.types import ObservationFrame

FAST = {"num_particles": 400}


def synthetic_result(position_error, yaw_error_deg, *, mirror_position_error=None,
                     mirror_yaw_error_deg=None, kidnap_steps=(), filter_converged=None,
                     dt=0.1) -> RunResult:
    """A RunResult with chosen error traces, for testing the metric definitions."""
    position_error = np.asarray(position_error, dtype=float)
    count = len(position_error)
    yaw_error_deg = np.broadcast_to(np.asarray(yaw_error_deg, dtype=float), count).copy()
    far = np.full(count, 5.0)
    zeros = np.zeros(count)
    return RunResult(
        scenario_id="synthetic", field_name="s_field", seed=0, dt=dt,
        kidnap_steps=tuple(kidnap_steps), time=dt * np.arange(count),
        truth=np.zeros((count, 3)), estimate=np.zeros((count, 3)),
        position_error=position_error, yaw_error_deg=yaw_error_deg,
        mirror_position_error=far if mirror_position_error is None
        else np.asarray(mirror_position_error, dtype=float),
        mirror_yaw_error_deg=np.full(count, 180.0) if mirror_yaw_error_deg is None
        else np.asarray(mirror_yaw_error_deg, dtype=float),
        position_std=zeros, yaw_std_deg=zeros, mode_weight=np.ones(count),
        confidence=np.ones(count),
        filter_converged=np.ones(count, bool) if filter_converged is None
        else np.asarray(filter_converged, bool),
        symmetric_ambiguity=np.zeros(count, bool), ess_fraction=np.ones(count),
        agreement=np.ones(count), resampled=np.zeros(count, bool), injected=zeros,
        num_landmarks=zeros, num_line_points=zeros, update_time_s=np.full(count, 1e-3))


# ------------------------------------------------------------------ scenarios
def test_all_required_scenarios_exist():
    required = ["s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10", "s11",
                "s12_field_s", "s12_field_m", "s12_field_l"]
    for prefix in required:
        assert select_scenarios([prefix]), prefix
    assert {get_scenario(f"s12_field_{k}").field for k in "sml"} == {
        "s_field", "m_field", "l_field"}
    with pytest.raises(KeyError):
        get_scenario("s99")


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_scenario_trajectories_are_deterministic_and_stay_on_the_carpet(scenario_id):
    scenario = get_scenario(scenario_id)
    field = FieldGeometry.load(scenario.field)
    first, again = build_trajectory(scenario, 3), build_trajectory(scenario, 3)
    np.testing.assert_array_equal(first.poses, again.poses)
    assert len(first) > 100 and first.dt == pytest.approx(0.1)
    assert np.all(field.contains(first.poses[:, :2], margin=0.3)), "stays on the field"
    steps = np.linalg.norm(np.diff(first.poses[:, :2], axis=0), axis=1)
    jumps = np.flatnonzero(steps > 0.1)
    assert set(jumps + 1) == set(first.kidnap_steps), "only declared kidnappings jump"


def test_random_scenarios_vary_with_the_seed():
    scenario = get_scenario("s06_random_start_global")
    a, b = build_trajectory(scenario, 0), build_trajectory(scenario, 1)
    assert not np.allclose(a.poses[0], b.poses[0])


# ------------------------------------------------------------------ metric definitions
def test_pose_errors_use_circular_yaw_difference():
    estimate = np.array([[1.0, 2.0, math.radians(179.0)]])
    truth = np.array([[1.3, 2.4, math.radians(-179.0)]])
    position, yaw = metrics.pose_errors(estimate, truth)
    assert position[0] == pytest.approx(0.5) and yaw[0] == pytest.approx(2.0)
    mirrored_position, mirrored_yaw = metrics.mirror_errors(mirror_pose(truth), truth)
    assert mirrored_position[0] == pytest.approx(0.0, abs=1e-12)
    assert mirrored_yaw[0] == pytest.approx(0.0, abs=1e-9)


def test_convergence_time_and_tracking_errors():
    # 3 s of large error, then 0.1 m / 2 deg, with one brief spike that breaks nothing.
    position = np.r_[np.full(30, 2.0), np.full(70, 0.1)]
    yaw = np.r_[np.full(30, 40.0), np.full(70, 2.0)]
    m = compute_metrics(synthetic_result(position, yaw))
    assert m["converged"] and m["convergence_time_s"] == pytest.approx(3.0)
    assert m["position_rmse"] == pytest.approx(0.1) and m["position_mae"] == pytest.approx(0.1)
    assert m["yaw_rmse"] == pytest.approx(2.0) and m["yaw_p95"] == pytest.approx(2.0)
    assert m["position_rmse_all"] == pytest.approx(math.sqrt((30 * 4.0 + 70 * 0.01) / 100))
    assert m["tracking_fraction"] == pytest.approx(0.7)
    assert m["diverged_fraction"] == 0.0 and not m["final_diverged"] and m["final_localized"]


def test_short_lucky_moments_are_not_convergence():
    position = np.full(100, 2.0)
    position[10:15] = 0.1                       # 0.5 s localized, less than the 2 s hold
    m = compute_metrics(synthetic_result(position, 2.0))
    assert not m["converged"] and m["convergence_time_s"] is None
    assert m["position_rmse"] is None and m["final_diverged"]


def test_divergence_after_convergence_is_counted_not_hidden():
    position = np.r_[np.full(50, 0.05), np.full(50, 3.0)]
    m = compute_metrics(synthetic_result(position, 1.0))
    assert m["converged"] and m["convergence_time_s"] == 0.0
    assert m["diverged_fraction"] == pytest.approx(0.5)
    assert m["position_rmse"] > 2.0, "errors after losing track stay in the statistics"
    assert m["final_diverged"]


def test_yaw_alone_can_fail_localization():
    m = compute_metrics(synthetic_result(np.full(100, 0.05), 45.0))
    assert not m["converged"] and m["final_diverged"]


def test_wrong_half_detection():
    count = 100
    result = synthetic_result(np.full(count, 6.0), 180.0,
                              mirror_position_error=np.full(count, 0.05),
                              mirror_yaw_error_deg=np.full(count, 1.0))
    m = compute_metrics(result)
    assert not m["converged"], "being on the mirrored pose is not being localized"
    assert m["final_wrong_half"] and m["wrong_half_fraction"] == 1.0
    assert m["convergence_time_symmetric_s"] == 0.0
    assert m["overconfident_fraction"] == 1.0, "the filter claimed convergence throughout"
    # Near the field centre both the pose and its mirror are close: that is not a wrong half.
    centred = synthetic_result(np.full(count, 0.05), 1.0,
                               mirror_position_error=np.full(count, 0.1),
                               mirror_yaw_error_deg=np.full(count, 1.0))
    assert compute_metrics(centred)["wrong_half_fraction"] == 0.0


def test_kidnap_recovery_time():
    position = np.r_[np.full(50, 0.05), np.full(15, 3.0), np.full(35, 0.05)]
    m = compute_metrics(synthetic_result(position, 1.0, kidnap_steps=(50,)))
    assert m["kidnap_times_s"] == [5.0]
    assert m["recovery_times_s"] == [pytest.approx(1.5)]
    assert m["position_rmse"] == pytest.approx(0.05), "the recovery window is reported as a time"
    never = compute_metrics(synthetic_result(np.r_[np.full(50, 0.05), np.full(50, 3.0)], 1.0,
                                             kidnap_steps=(50,)))
    assert never["recovery_times_s"] == [None] and never["final_diverged"]


def test_aggregation_pools_errors_and_counts_failures():
    good = synthetic_result(np.full(100, 0.1), 1.0)
    bad = synthetic_result(np.full(100, 4.0), 90.0)
    per_run = [compute_metrics(good), compute_metrics(bad)]
    aggregate = metrics.aggregate_metrics([good, bad], per_run)
    assert aggregate["runs"] == 2 and aggregate["convergence_rate"] == 0.5
    assert aggregate["divergence_rate"] == 0.5
    assert aggregate["position_rmse"] == pytest.approx(0.1)        # tracking window only
    assert aggregate["position_rmse_all"] == pytest.approx(math.sqrt((0.01 + 16.0) / 2))
    assert aggregate["update_latency_ms_mean"] == pytest.approx(1.0)


def test_thresholds_are_configurable():
    result = synthetic_result(np.full(100, 0.2), 1.0)
    assert compute_metrics(result)["converged"]
    assert not compute_metrics(result, MetricThresholds(localized_position=0.1))["converged"]


# ------------------------------------------------------------------ runner
@pytest.mark.slow
def test_runs_are_reproducible():
    scenario = get_scenario("s01_stationary")
    a = run_scenario(scenario, 3, filter_overrides=FAST)
    b = run_scenario(scenario, 3, filter_overrides=FAST)
    c = run_scenario(scenario, 4, filter_overrides=FAST)
    np.testing.assert_array_equal(a.estimate, b.estimate)
    np.testing.assert_array_equal(a.truth, b.truth)
    assert not np.array_equal(a.estimate, c.estimate)


@pytest.mark.slow
def test_only_odometry_and_observations_reach_the_filter(monkeypatch):
    """Record everything the benchmark hands to the filter, then replay it without the benchmark.

    The replay has no simulator, no trajectory and no ground truth. If it reproduces the
    estimates exactly, nothing else can have influenced the filter.
    """
    recorded = {"init": [], "predict": [], "update": []}
    originals = {name: getattr(ParticleFilter, name)
                 for name in ("init_gaussian", "init_uniform", "predict", "update")}

    def spy(name, bucket):
        def wrapper(self, *args, **kwargs):
            recorded[bucket].append((name, args, kwargs))
            return originals[name](self, *args, **kwargs)
        return wrapper

    constructed = []
    original_init = ParticleFilter.__init__

    def record_construction(self, *args, **kwargs):
        constructed.append((args, kwargs))
        original_init(self, *args, **kwargs)
        recorded["init"].clear()            # drop the constructor's own default initialisation

    monkeypatch.setattr(ParticleFilter, "__init__", record_construction)
    monkeypatch.setattr(ParticleFilter, "init_gaussian", spy("init_gaussian", "init"))
    monkeypatch.setattr(ParticleFilter, "init_uniform", spy("init_uniform", "init"))
    monkeypatch.setattr(ParticleFilter, "predict", spy("predict", "predict"))
    monkeypatch.setattr(ParticleFilter, "update", spy("update", "update"))
    result = run_scenario(get_scenario("s04_rectangle"), 0, filter_overrides=FAST)
    monkeypatch.undo()

    assert len(recorded["init"]) == 1, "a global pose enters exactly once, at initialisation"
    assert len(recorded["update"]) == len(result.time)
    assert len(recorded["predict"]) == len(result.time) - 1
    for _, args, kwargs in recorded["predict"]:
        assert not kwargs and len(args) == 1
        assert args[0] is None or type(args[0]) is OdometryDelta
    for _, args, kwargs in recorded["update"]:
        assert not kwargs and len(args) == 1 and type(args[0]) is ObservationFrame
    # Odometry is relative and small; it cannot encode where on the field the robot is.
    steps = np.array([a[0].as_array() for _, a, _ in recorded["predict"] if a[0] is not None])
    assert np.abs(steps[:, :2]).max() < 0.1 and np.abs(steps[:, 2]).max() < 0.2

    (args, kwargs), = constructed
    replay = ParticleFilter(*args, **kwargs)
    name, init_args, init_kwargs = recorded["init"][0]
    getattr(replay, name)(*init_args, **init_kwargs)
    estimates = []
    for step, (_, update_args, _) in enumerate(recorded["update"]):
        if step > 0:
            replay.predict(*recorded["predict"][step - 1][1])
        replay.update(*update_args)
        estimates.append(replay.estimate().pose)
    np.testing.assert_array_equal(np.array(estimates), result.estimate)


@pytest.mark.slow
def test_a_tracking_scenario_converges_accurately():
    result = run_scenario(get_scenario("s04_rectangle"), 0)
    m = compute_metrics(result)
    assert m["converged"] and m["convergence_time_s"] < 2.0
    assert m["position_rmse"] < 0.15 and m["yaw_rmse"] < 3.0       # the engineering targets
    assert not m["final_diverged"] and m["wrong_half_fraction"] == 0.0
    assert m["update_latency_ms_mean"] < 50.0


@pytest.mark.slow
def test_kidnapped_scenario_recovers():
    result = run_scenario(get_scenario("s10a_kidnapped_same_side"), 0, snapshot_steps="auto")
    m = compute_metrics(result)
    kidnap = result.kidnap_steps[0]
    assert result.position_error[kidnap] > 1.0, "the kidnapping really displaced the robot"
    assert m["recovery_times_s"][0] is not None and m["recovery_times_s"][0] < 10.0
    assert m["final_localized"] and result.injected[kidnap:].sum() > 0
    assert set(result.snapshots) == {0, kidnap + 5, len(result.time) - 1}


@pytest.mark.slow
def test_the_benchmark_detects_a_confidently_wrong_filter():
    result = run_scenario(get_scenario("s11c_symmetry_wrong_prior"), 0)
    m = compute_metrics(result)
    assert not m["converged"] and m["final_wrong_half"] and m["final_diverged"]
    assert m["overconfident_fraction"] > 0.8, "the filter itself believes it is fine"
    assert m["convergence_time_symmetric_s"] is not None


@pytest.mark.slow
def test_global_initialisation_represents_both_symmetric_hypotheses():
    result = run_scenario(get_scenario("s11a_symmetry_no_prior"), 0, snapshot_steps=(15,))
    m = compute_metrics(result)
    assert m["convergence_time_symmetric_s"] is not None
    assert result.symmetric_ambiguity.any(), "the filter flags the mirrored hypothesis itself"
    snapshot = result.snapshots[15]
    assert snapshot.particles[:, 0].min() < -1.0 and snapshot.particles[:, 0].max() > 1.0


@pytest.mark.slow
@pytest.mark.parametrize("scenario_id", ["s12_field_m", "s12_field_l"])
def test_larger_fields_run_from_configuration_alone(scenario_id):
    result = run_scenario(get_scenario(scenario_id), 0)
    m = compute_metrics(result)
    assert result.field_name == get_scenario(scenario_id).field
    assert m["converged"] and m["final_localized"]


@pytest.mark.slow
def test_mujoco_ground_truth_gives_the_same_run():
    pytest.importorskip("mujoco")
    from robosoccer_perception.simulation.mujoco_scene import MujocoScene

    scenario = get_scenario("s02_rotation")
    kinematic = run_scenario(scenario, 1, filter_overrides=FAST)
    with MujocoScene(FieldGeometry.load(scenario.field), scenario_id=scenario.id) as scene:
        simulated = run_scenario(scenario, 1, filter_overrides=FAST, scene=scene)
    np.testing.assert_allclose(simulated.truth, kinematic.truth, atol=1e-9)
    np.testing.assert_allclose(simulated.position_error, kinematic.position_error, atol=1e-6)


@pytest.mark.slow
def test_benchmark_writes_machine_and_human_readable_results(tmp_path):
    pytest.importorskip("matplotlib")
    scenarios = select_scenarios(["s01", "s10a"])
    report = benchmark.run_benchmark(scenarios, [0, 1], tmp_path, filter_overrides=FAST,
                                     log=lambda line: None)
    stored = json.loads((tmp_path / "results.json").read_text())
    assert set(stored["scenarios"]) == {"s01_stationary", "s10a_kidnapped_same_side"}
    assert stored["meta"]["seeds"] == [0, 1]
    assert stored["meta"]["filter_config"]["num_particles"] == 400
    entry = stored["scenarios"]["s01_stationary"]
    assert len(entry["runs"]) == 2 and entry["aggregate"]["runs"] == 2
    for key in ("position_rmse", "position_mae", "position_p95", "yaw_rmse", "yaw_mae", "yaw_p95",
                "convergence_time_s_mean", "divergence_rate", "wrong_half_rate",
                "update_latency_ms_mean", "update_latency_ms_p95", "wall_time_s_total"):
        assert key in entry["aggregate"], key
    assert stored["scenarios"]["s10a_kidnapped_same_side"]["aggregate"]["kidnaps"] == 2
    assert report["scenarios"].keys() == stored["scenarios"].keys()

    with open(tmp_path / "summary.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["scenario"] for row in rows] == ["s01_stationary", "s10a_kidnapped_same_side"]
    assert float(rows[0]["position_rmse"]) < 0.15
    with open(tmp_path / "runs.csv") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    with open(tmp_path / "timeseries" / "s01_stationary_seed0.csv") as handle:
        series = list(csv.DictReader(handle))
    assert len(series) == 301 and {"truth_x", "est_x", "position_error", "ess_fraction",
                                   "filter_converged", "update_time_ms"} <= set(series[0])
    summary = (tmp_path / "summary.md").read_text()
    assert "s10a_kidnapped_same_side" in summary and "Pos RMSE" in summary
    for name in ("particles", "errors", "trajectory"):
        figure = tmp_path / "plots" / f"s01_stationary_{name}.png"
        assert figure.is_file() and figure.stat().st_size > 10_000
