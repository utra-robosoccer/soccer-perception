"""Running scenarios and writing benchmark results.

The data flow of one run is the whole point of this module::

    true trajectory ──► SyntheticOdometry ──► OdometryDelta ────┐
          │                                                      ▼
          └────────► OracleObservationGenerator ─► ObservationFrame ─► ParticleFilter ─► estimate
          │                                                                                │
          └──────────────────────────► evaluation (errors, metrics, plots) ◄──────────────┘

The true pose reaches the filter only as noisy relative odometry and as anonymous observations
in the robot frame. `run_scenario` never passes ground truth to the filter after initialisation
(tests/test_benchmarks.py checks the types that cross that boundary).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import platform
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.evaluation import metrics as metrics_module
from robosoccer_perception.evaluation.metrics import MetricThresholds
from robosoccer_perception.evaluation.scenarios import Scenario
from robosoccer_perception.geometry.camera import StereoCameraConfig, StereoCameraModel
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import build_landmark_map
from robosoccer_perception.localization.estimation import mirror_pose
from robosoccer_perception.localization.particle_filter import (
    ParticleFilter, ParticleFilterConfig)
from robosoccer_perception.observations.noise import OracleConfig
from robosoccer_perception.observations.oracle import OracleObservationGenerator
from robosoccer_perception.observations.types import ObservationFrame
from robosoccer_perception.simulation.odometry import OdometryNoiseConfig, SyntheticOdometry

DEFAULT_BALL = (1.0, 0.5)


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """The filter's state at one step, kept for plotting."""
    step: int
    particles: np.ndarray
    weights: np.ndarray
    frame: ObservationFrame


@dataclasses.dataclass
class RunResult:
    """Time series of one scenario run. Arrays have one entry per step."""
    scenario_id: str
    field_name: str
    seed: int
    dt: float
    kidnap_steps: tuple[int, ...]
    time: np.ndarray
    truth: np.ndarray                   # (T, 3) ground-truth poses, for evaluation only
    estimate: np.ndarray                # (T, 3)
    position_error: np.ndarray          # metres
    yaw_error_deg: np.ndarray           # degrees, absolute, circular
    mirror_position_error: np.ndarray   # same, against the half-turn mirror of the truth
    mirror_yaw_error_deg: np.ndarray
    position_std: np.ndarray            # filter's own uncertainty
    yaw_std_deg: np.ndarray
    mode_weight: np.ndarray
    confidence: np.ndarray
    filter_converged: np.ndarray        # bool: the filter's own convergence flag
    symmetric_ambiguity: np.ndarray     # bool
    ess_fraction: np.ndarray            # effective sample size / N
    agreement: np.ndarray
    resampled: np.ndarray               # bool
    injected: np.ndarray                # particles injected at that update
    num_landmarks: np.ndarray
    num_line_points: np.ndarray
    update_time_s: np.ndarray           # predict + update + estimate wall time
    wall_time_s: float = 0.0
    snapshots: dict[int, Snapshot] = dataclasses.field(default_factory=dict)

    COLUMNS = ("time", "truth_x", "truth_y", "truth_yaw", "est_x", "est_y", "est_yaw",
               "position_error", "yaw_error_deg", "mirror_position_error",
               "mirror_yaw_error_deg", "position_std", "yaw_std_deg", "mode_weight",
               "confidence", "filter_converged", "symmetric_ambiguity", "ess_fraction",
               "agreement", "resampled", "injected", "num_landmarks", "num_line_points",
               "update_time_ms")

    def rows(self):
        for i in range(len(self.time)):
            yield (self.time[i], *self.truth[i], *self.estimate[i], self.position_error[i],
                   self.yaw_error_deg[i], self.mirror_position_error[i],
                   self.mirror_yaw_error_deg[i], self.position_std[i], self.yaw_std_deg[i],
                   self.mode_weight[i], self.confidence[i], int(self.filter_converged[i]),
                   int(self.symmetric_ambiguity[i]), self.ess_fraction[i], self.agreement[i],
                   int(self.resampled[i]), int(self.injected[i]), int(self.num_landmarks[i]),
                   int(self.num_line_points[i]), 1e3 * self.update_time_s[i])

    def write_csv(self, path) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            for row in self.rows():
                writer.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])


def _streams(scenario_id: str, seed: int) -> list[np.random.Generator]:
    """Independent, reproducible random streams for one (scenario, seed)."""
    root = np.random.SeedSequence([int(seed), zlib.crc32(scenario_id.encode())])
    return [np.random.default_rng(child) for child in root.spawn(5)]


def build_trajectory(scenario: Scenario, seed: int = 0):
    """The true trajectory that `run_scenario` uses for this scenario and seed."""
    return scenario.build_trajectory(FieldGeometry.load(scenario.field),
                                     _streams(scenario.id, seed)[0])


def run_scenario(scenario: Scenario, seed: int = 0, *,
                 filter_config: str = "particle_filter",
                 filter_overrides: dict[str, Any] | None = None,
                 camera: str = "zed_mini_sim", snapshot_steps=(), scene=None) -> RunResult:
    """Run one scenario with one seed and return its time series.

    ``snapshot_steps`` lists steps at which to keep the particle set (``"auto"`` picks the
    start, a mid-run step and the end). ``scene`` may be a `MujocoScene` for the scenario's field. The platform is then driven
    through MuJoCo and the ground truth is read back from it on every step; otherwise the
    ground truth is the commanded trajectory itself (they are identical; see the tests).
    """
    trajectory_rng, oracle_rng, odometry_rng, init_rng, filter_rng = _streams(scenario.id, seed)
    field = FieldGeometry.load(scenario.field)
    landmarks = build_landmark_map(field)
    camera_cfg = StereoCameraConfig.load(camera)
    camera_model = StereoCameraModel(camera_cfg)
    trajectory = scenario.build_trajectory(field, trajectory_rng)
    oracle = OracleObservationGenerator(
        field, landmarks, camera_model,
        OracleConfig.load(scenario.oracle_profile, scenario.oracle_overrides))
    odometry_cfg = OdometryNoiseConfig().scaled(scenario.odometry_scale).replace(
        **scenario.odometry_overrides)
    odometry = SyntheticOdometry(odometry_cfg, odometry_rng)
    overrides = dict(scenario.filter_overrides)
    for key, value in (filter_overrides or {}).items():
        overrides[key] = {**overrides.get(key, {}), **value} if isinstance(value, dict) else value
    pf_config = ParticleFilterConfig.load(filter_config, overrides)
    localizer = ParticleFilter(field, landmarks, pf_config,
                               seed=int(filter_rng.integers(2 ** 31)))

    def true_pose(step: int) -> np.ndarray:
        if scene is None:
            return trajectory.poses[step]
        scene.follow(trajectory, step)
        return scene.ground_truth().base_pose

    def rig_at(step: int):
        rig = camera_cfg.mount
        if trajectory.pan is not None:
            rig = rig.replace(pan=float(trajectory.pan[step]))
        if trajectory.pitch is not None:
            rig = rig.replace(pitch=float(trajectory.pitch[step]))
        return rig

    # ---- initialisation: the only place a global pose is given to the filter
    start = true_pose(0)
    position_std = scenario.init_position_std
    yaw_std = math.radians(scenario.init_yaw_std_deg)
    guess_error = init_rng.normal(size=3) * [position_std, position_std, yaw_std]
    half_x, _ = field.half_extent
    if scenario.init == "gaussian":
        localizer.init_gaussian(start + guess_error, position_std, yaw_std)
    elif scenario.init == "mirrored":
        localizer.init_gaussian(mirror_pose(start) + guess_error, position_std, yaw_std)
    elif scenario.init == "own_half":
        localizer.init_uniform(x_range=(-half_x, 0.0) if start[0] < 0 else (0.0, half_x))
    else:
        localizer.init_uniform()

    steps = len(trajectory)
    truth = np.zeros((steps, 3))
    estimate = np.zeros((steps, 3))
    series = {name: np.zeros(steps) for name in (
        "position_std", "yaw_std_deg", "mode_weight", "confidence", "ess_fraction", "agreement",
        "injected", "num_landmarks", "num_line_points", "update_time_s")}
    flags = {name: np.zeros(steps, dtype=bool) for name in (
        "filter_converged", "symmetric_ambiguity", "resampled")}
    snapshots: dict[int, Snapshot] = {}
    if snapshot_steps == "auto":
        # Start, a mid-run step worth seeing (just after a kidnap if there is one), and the end.
        middle = trajectory.kidnap_steps[0] + 5 if trajectory.kidnap_steps else steps // 3
        snapshot_steps = (0, min(middle, steps - 1), steps - 1)
    snapshot_steps = {s % steps for s in snapshot_steps}
    wall_start = time.perf_counter()

    previous = start
    for step in range(steps):
        timestamp = step * trajectory.dt
        current = true_pose(step)
        truth[step] = current

        # ---- what the robot would measure (uses ground truth; the filter does not see it)
        delta = None
        if step > 0:
            delta = odometry.measure(previous, current, timestamp, trajectory.dt,
                                     kidnapped=step in trajectory.kidnap_steps)
        frame, _debug = oracle.generate(current, oracle_rng, rig_at(step), timestamp,
                                        ball_position=DEFAULT_BALL)
        previous = current

        # ---- the localizer: only `delta` and `frame` cross this line
        tic = time.perf_counter()
        if step > 0:
            localizer.predict(delta)
        info = localizer.update(frame)
        pose_estimate = localizer.estimate()
        series["update_time_s"][step] = time.perf_counter() - tic

        estimate[step] = pose_estimate.pose
        series["position_std"][step] = pose_estimate.position_std
        series["yaw_std_deg"][step] = math.degrees(pose_estimate.yaw_std)
        series["mode_weight"][step] = pose_estimate.mode_weight
        series["confidence"][step] = pose_estimate.confidence
        series["ess_fraction"][step] = info.effective_sample_size / localizer.num_particles
        series["agreement"][step] = info.agreement
        series["injected"][step] = info.injected
        series["num_landmarks"][step] = len(frame.landmarks)
        series["num_line_points"][step] = len(frame.line_points)
        flags["filter_converged"][step] = pose_estimate.converged
        flags["symmetric_ambiguity"][step] = pose_estimate.symmetric_ambiguity
        flags["resampled"][step] = info.resampled
        if step in snapshot_steps:
            snapshots[step] = Snapshot(step, localizer.particles.copy(),
                                       localizer.weights.copy(), frame)

    position_error, yaw_error = metrics_module.pose_errors(estimate, truth)
    mirror_position_error, mirror_yaw_error = metrics_module.mirror_errors(estimate, truth)
    return RunResult(
        scenario_id=scenario.id, field_name=field.name, seed=int(seed), dt=trajectory.dt,
        kidnap_steps=trajectory.kidnap_steps, time=trajectory.times, truth=truth,
        estimate=estimate, position_error=position_error, yaw_error_deg=yaw_error,
        mirror_position_error=mirror_position_error, mirror_yaw_error_deg=mirror_yaw_error,
        wall_time_s=time.perf_counter() - wall_start, snapshots=snapshots, **series, **flags)


# ------------------------------------------------------------------ whole benchmark
SUMMARY_COLUMNS = (
    "scenario", "field", "runs", "convergence_rate", "convergence_time_s_mean",
    "convergence_rate_symmetric", "convergence_time_symmetric_s_mean",
    "position_rmse", "position_mae", "position_p95", "yaw_rmse", "yaw_mae", "yaw_p95",
    "position_rmse_all", "yaw_rmse_all", "tracking_fraction", "diverged_fraction",
    "divergence_rate", "wrong_half_rate", "recovery_rate", "recovery_time_s_mean",
    "recovery_rate_symmetric", "recovery_time_symmetric_s_mean", "overconfident_fraction",
    "mean_ess_fraction", "resample_rate", "update_latency_ms_mean", "update_latency_ms_p95",
    "update_latency_ms_max")


def run_benchmark(scenarios: list[Scenario], seeds: list[int], output_dir=None, *,
                  thresholds: MetricThresholds | None = None,
                  filter_overrides: dict[str, Any] | None = None, plots: bool = True,
                  log=print) -> dict[str, Any]:
    """Run every scenario with every seed; write JSON, CSV, a summary and plots.

    Returns the same dictionary that is written to ``results.json``.
    """
    thresholds = thresholds or MetricThresholds()
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        (output / "timeseries").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "meta": {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seeds": list(seeds),
            "thresholds": dataclasses.asdict(thresholds),
            "filter_config": dataclasses.asdict(
                ParticleFilterConfig.load("particle_filter", filter_overrides)),
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.processor() or platform.machine(),
        },
        "scenarios": {},
    }
    for scenario in scenarios:
        results, per_run = [], []
        for index, seed in enumerate(seeds):
            # Keep particle snapshots of the first seed for the particle-cloud figure.
            result = run_scenario(scenario, seed, filter_overrides=filter_overrides,
                                  snapshot_steps="auto" if index == 0 else ())
            results.append(result)
            per_run.append({"seed": seed, **metrics_module.compute_metrics(result, thresholds)})
            if output is not None:
                result.write_csv(output / "timeseries" / f"{scenario.id}_seed{seed}.csv")
        aggregate = metrics_module.aggregate_metrics(results, per_run, thresholds)
        report["scenarios"][scenario.id] = {
            "title": scenario.title, "description": scenario.description,
            "expectation": scenario.expectation, "field": scenario.field,
            "init": scenario.init, "aggregate": aggregate, "runs": per_run}
        log(format_row(scenario.id, aggregate))
        if output is not None and plots:
            from robosoccer_perception.visualization import diagnostics
            diagnostics.save_scenario_figures(scenario, results[0], output / "plots", thresholds)

    if output is not None:
        (output / "results.json").write_text(json.dumps(report, indent=2))
        with open(output / "summary.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_COLUMNS)
            for scenario_id, entry in report["scenarios"].items():
                agg = entry["aggregate"]
                writer.writerow([scenario_id, entry["field"], *(
                    _cell(agg.get(column)) for column in SUMMARY_COLUMNS[2:])])
        with open(output / "runs.csv", "w", newline="") as handle:
            keys = [k for k in next(iter(report["scenarios"].values()))["runs"][0]
                    if not k.endswith("times_s")]
            writer = csv.writer(handle)
            writer.writerow(["scenario", *keys])
            for scenario_id, entry in report["scenarios"].items():
                for run in entry["runs"]:
                    writer.writerow([scenario_id, *(_cell(run[k]) for k in keys)])
        (output / "summary.md").write_text(format_summary(report))
    return report


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


HEADER = (f"{'scenario':<30}{'conv':>6}{'sym':>6}{'t_conv':>8}{'posRMSE':>9}{'posP95':>8}"
          f"{'yawRMSE':>9}{'yawP95':>8}{'diverg':>8}{'wrongHalf':>10}{'recov':>8}{'ms':>7}")


def _recovery_text(agg: dict[str, Any]) -> str:
    """Kidnap recovery: mean time to the TRUE pose, or how many reached the mirrored pose."""
    if agg["recovery_rate"] is None:
        return "-"
    if agg["recovery_time_s_mean"] is not None:
        return f"{agg['recovery_time_s_mean']:.1f} s ({agg['recovery_rate'] * 100:.0f} %)"
    if agg["recovery_time_symmetric_s_mean"] is not None:
        return (f"mirror only, {agg['recovery_time_symmetric_s_mean']:.1f} s "
                f"({agg['recovery_rate_symmetric'] * 100:.0f} %)")
    return "none"


def format_row(scenario_id: str, agg: dict[str, Any]) -> str:
    recovery = _recovery_text(agg).split(" (")[0].replace("mirror only, ", "~")
    return (f"{scenario_id:<30}{agg['convergence_rate'] * 100:>5.0f}%"
            f"{agg['convergence_rate_symmetric'] * 100:>5.0f}%"
            f"{_fmt(agg['convergence_time_s_mean'], 1, 's'):>8}"
            f"{_fmt(agg['position_rmse'], 3):>9}{_fmt(agg['position_p95'], 3):>8}"
            f"{_fmt(agg['yaw_rmse'], 2):>9}{_fmt(agg['yaw_p95'], 2):>8}"
            f"{agg['divergence_rate'] * 100:>7.0f}%{agg['wrong_half_rate'] * 100:>9.0f}%"
            f"{recovery:>8}{agg['update_latency_ms_mean']:>7.2f}")


def format_summary(report: dict[str, Any]) -> str:
    """Human-readable Markdown summary of a benchmark report."""
    meta = report["meta"]
    th = meta["thresholds"]
    lines = [
        "# Localization benchmark", "",
        f"Created {meta['created']}. Seeds: {meta['seeds']}. "
        f"Particles: {meta['filter_config']['num_particles']}.", "",
        "How to read the table:", "",
        f"- **Localized** means position error < {th['localized_position']} m and yaw error < "
        f"{th['localized_yaw_deg']} deg. **Conv.** is the share of runs that became localized "
        f"and stayed so for {th['hold_time']} s; **Conv. (sym.)** counts a run that did so "
        "with respect to either the true pose or its half-turn mirror image.",
        "- **RMSE** and **P95** are pooled over all seeds, over the steps after convergence "
        "(`summary.csv` also has `_all` columns covering every step).",
        "- **Diverged** and **Wrong half** are the share of runs that END not localized / "
        "on the mirrored pose.",
        "- **Recovery** is the mean time from a kidnapping until the run is localized again.",
        "",
        "| Scenario | Field | Conv. | Conv. (sym.) | t conv [s] | Pos RMSE [m] | Pos P95 [m] "
        "| Yaw RMSE [deg] | Yaw P95 [deg] | Diverged | Wrong half | Recovery | Update [ms] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scenario_id, entry in report["scenarios"].items():
        agg = entry["aggregate"]
        lines.append(
            f"| {scenario_id} | {entry['field']} | {agg['convergence_rate'] * 100:.0f} % "
            f"| {agg['convergence_rate_symmetric'] * 100:.0f} % "
            f"| {_fmt(agg['convergence_time_s_mean'], 1)} | {_fmt(agg['position_rmse'], 3)} "
            f"| {_fmt(agg['position_p95'], 3)} | {_fmt(agg['yaw_rmse'], 2)} "
            f"| {_fmt(agg['yaw_p95'], 2)} | {agg['divergence_rate'] * 100:.0f} % "
            f"| {agg['wrong_half_rate'] * 100:.0f} % | {_recovery_text(agg)} "
            f"| {agg['update_latency_ms_mean']:.2f} |")
    lines += ["", "## Scenarios", ""]
    for scenario_id, entry in report["scenarios"].items():
        agg = entry["aggregate"]
        lines += [f"**{scenario_id}: {entry['title']}.** {entry['description']} "
                  f"Initialisation: `{entry['init']}`. Expectation: {entry['expectation']}. "
                  f"The filter reported convergence while not localized in "
                  f"{(agg['overconfident_fraction'] or 0) * 100:.1f} % of steps.", ""]
    return "\n".join(lines) + "\n"
