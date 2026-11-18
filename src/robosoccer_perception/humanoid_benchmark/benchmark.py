"""The Phase 4 benchmark: scenarios x seeds x arms, the calibration sweep and the timing sweep.

Statistics per scenario and arm, pooled over the seeds:

* the Phase 1 localization metrics (`evaluation.metrics`): position and yaw RMSE over the
  whole run (``*_all``) and over the tracking window, divergence, wrong-half and
  overconfident ("converged" while not localized) shares;
* camera orientation: RMSE of the estimated camera roll and pitch against the true ones, the
  mean pitch error (a bias means a calibration problem, not noise), the body-tilt RMSE;
* camera projection error: how far ground points 1 to 4 m ahead are misplaced;
* consistency: the localizer's ``RMS error / RMS reported std`` and the share of frames with
  the error inside two reported standard deviations; for IMU arms the tilt's
  ``RMS error / RMS reported std``. 1.0 is consistent, above 1 is overconfident;
* latency per stage and the frame rate of the primary arm.

The first ``SETTLE_S`` seconds of every run are left out of the orientation and consistency
statistics (the filters start from a rough guess); the localization metrics cover the whole
run. Everything is SYNTHETIC: rendered images, synthetic body motion, a simulated IMU.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robosoccer_perception.estimation.calibration_monitor import MonitorConfig
from robosoccer_perception.estimation.camera_pose import FusionConfig
from robosoccer_perception.evaluation.metrics import (
    MetricThresholds, aggregate_metrics, compute_metrics)
from robosoccer_perception.geometry.mount import CameraMount
from robosoccer_perception.geometry.robot_description import RobotDescription
from robosoccer_perception.humanoid_benchmark.runner import (
    DEFAULT_ARMS, FIXED, IMU, IMU_SELFCAL, IMU_UNCERTAINTY, ORACLE, FusionArm, FusionRun,
    run_humanoid_scenario)
from robosoccer_perception.humanoid_benchmark.scenarios import HumanoidScenario
from robosoccer_perception.inference import profiling
from robosoccer_perception.vision_benchmark.runner import BenchmarkResources, EstimateError

SETTLE_S = 3.0          # seconds left out of the orientation and consistency statistics
WARMUP_STEPS = 5        # first frames of a process are slower (lazy initialisation)
HEADER = (f"{'scenario':<28}{'arm':<20}{'pos RMSE':>9}{'yaw RMSE':>9}{'cam pitch':>10}"
          f"{'cam roll':>9}{'proj err':>9}{'consist.':>9}{'not loc.':>9}")
SYNTHETIC_NOTE = (
    "Every image is rendered by this repository's simulator, the body motion is synthetic "
    "(not measured on the robot), the IMU is simulated and every camera mount is an "
    "experimental assumption: the robot has no head and cannot walk yet. Nothing here is "
    "evidence about real hardware.")


def _rms(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else None


def _mean(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def arm_statistics(runs: list[FusionRun], name: str, thresholds: MetricThresholds
                   ) -> dict[str, Any]:
    """Aggregate of one arm over the seeds of one scenario."""
    results = [run.arms[name].result for run in runs]
    per_run = [compute_metrics(result, thresholds) for result in results]
    aggregate = aggregate_metrics(results, per_run, thresholds)
    pooled: dict[str, list[np.ndarray]] = {}

    def add(key: str, values) -> None:
        pooled.setdefault(key, []).append(np.asarray(values, dtype=float))

    for run in runs:
        arm = run.arms[name]
        keep = run.time >= SETTLE_S
        result = arm.result
        add("pitch", arm.camera_pitch_error_deg[keep])
        add("roll", arm.camera_roll_error_deg[keep])
        add("height", arm.camera_height_error[keep])
        add("projection", arm.projection_error[keep])
        body_roll = arm.body_roll_deg[keep] - run.true_body_roll_deg[keep]
        body_pitch = arm.body_pitch_deg[keep] - run.true_body_pitch_deg[keep]
        add("body_roll", body_roll)
        add("body_pitch", body_pitch)
        add("tilt_std", arm.tilt_std_deg[keep])
        add("tilt_ratio", np.concatenate([body_roll, body_pitch])
            / np.tile(arm.tilt_std_deg[keep], 2))
        add("position_error", result.position_error[keep])
        add("position_std", result.position_std[keep])
        add("within", result.position_error[keep] <= 2.0 * result.position_std[keep])
    cat = {key: np.concatenate(values) for key, values in pooled.items()}
    tilt_std = _rms(cat["tilt_std"])
    position_std = _rms(cat["position_std"])
    aggregate.update({
        "camera_pitch_rmse_deg": _rms(cat["pitch"]),
        "camera_roll_rmse_deg": _rms(cat["roll"]),
        "camera_pitch_bias_deg": _mean(cat["pitch"]),
        "camera_roll_bias_deg": _mean(cat["roll"]),
        "camera_height_rmse_m": _rms(cat["height"]),
        "body_roll_rmse_deg": _rms(cat["body_roll"]),
        "body_pitch_rmse_deg": _rms(cat["body_pitch"]),
        "tilt_std_rms_deg": tilt_std,
        "tilt_consistency": _rms(cat["tilt_ratio"]) if tilt_std is not None else None,
        "projection_error_mean_m": _mean(cat["projection"]),
        "projection_error_p95_m": float(np.nanpercentile(cat["projection"], 95))
        if np.isfinite(cat["projection"]).any() else None,
        "position_std_rms_m": position_std,
        "position_consistency": (_rms(cat["position_error"]) / position_std
                                 if position_std else None),
        "position_within_2std": float(cat["within"].mean()),
    })
    sources: dict[str, int] = {}
    for run in runs:
        for source, count in run.arms[name].sources.items():
            sources[source] = sources.get(source, 0) + count
    corrections = [run.arms[name].correction_deg[-1] for run in runs
                   if run.arms[name].correction_deg is not None]
    return {"aggregate": aggregate, "runs": per_run, "sources": sources,
            "sync": [run.arms[name].sync for run in runs],
            "final_correction_deg": [c.tolist() for c in corrections] or None}


def timing_statistics(runs: list[FusionRun]) -> dict[str, Any] | None:
    """Latency per stage of the primary arm (the one that ran the networks)."""
    timed = [arm.timing_ms for run in runs for arm in run.arms.values()
             if arm.timing_ms is not None]
    if not timed:
        return None
    out: dict[str, Any] = {}
    for key in timed[0]:
        values = np.concatenate([t[key][WARMUP_STEPS:] for t in timed])
        out[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95))}
    out["fps"] = 1000.0 / out["total"]["mean"]
    out["render_ms_mean"] = float(np.mean([run.render_ms_mean for run in runs]))
    return out


def _meta(resources: BenchmarkResources, seeds, arms, fusion: FusionConfig,
          monitor: MonitorConfig) -> dict[str, Any]:
    vision = resources.vision
    device = getattr(vision, "device", "cpu")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seeds": list(seeds),
        "resolution": list(resources.resolution), "camera": resources.camera,
        "models": vision.describe() if hasattr(vision, "describe") else str(type(vision)),
        "hardware": profiling.hardware_summary(device),
        "python": platform.python_version(), "numpy": np.__version__,
        "measurement_noise": dataclasses.asdict(resources.noise),
        "fusion": dataclasses.asdict(fusion), "monitor": dataclasses.asdict(monitor),
        "settle_s": SETTLE_S,
        "arms": {arm.name: {"description": arm.description, "mode": arm.mode, "sync": arm.sync,
                            "use_uncertainty": arm.use_uncertainty,
                            "self_calibration": arm.self_calibration,
                            "uses_ground_truth": arm.is_oracle} for arm in arms},
        "synthetic_only": SYNTHETIC_NOTE}


def describe_scenario(scenario: HumanoidScenario) -> dict[str, Any]:
    robot = RobotDescription.load(scenario.robot)
    mount = CameraMount.load(scenario.mount)
    return {"title": scenario.title, "description": scenario.description,
            "expectation": scenario.expectation, **scenario.describe(),
            "camera_height_standing_m": mount.standing_height(robot.body),
            "mount_transform": mount.describe(robot.body)}


def format_row(scenario_id: str, arm: str, agg: dict[str, Any]) -> str:
    def cell(value, digits=2, width=9) -> str:
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    return (f"{scenario_id:<28}{arm:<20}{cell(agg['position_rmse_all'], 3)}"
            f"{cell(agg['yaw_rmse_all'])}{cell(agg['camera_pitch_rmse_deg'], 2, 10)}"
            f"{cell(agg['camera_roll_rmse_deg'])}{cell(agg['projection_error_mean_m'], 3)}"
            f"{cell(agg['position_consistency'])}{100.0 * agg['divergence_rate']:>8.0f}%")


def run_fusion_benchmark(scenarios: list[HumanoidScenario], seeds: list[int],
                         resources: BenchmarkResources, output_dir=None, *,
                         arms: tuple[FusionArm, ...] | None = None,
                         fusion: FusionConfig | None = None,
                         monitor: MonitorConfig | None = None, plots: bool = True,
                         max_steps: int | None = None,
                         log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run every scenario for every seed and arm; write ``results.json``, ``summary.md``,
    ``summary.csv`` and plots to ``output_dir`` when it is given."""
    arms = tuple(arms if arms is not None else DEFAULT_ARMS)
    fusion, monitor = fusion or FusionConfig(), monitor or MonitorConfig()
    thresholds = MetricThresholds()
    report: dict[str, Any] = {"meta": _meta(resources, seeds, arms, fusion, monitor),
                              "scenarios": {}}
    kept: dict[str, FusionRun] = {}
    for scenario in scenarios:
        runs = [run_humanoid_scenario(scenario, seed, resources, arms, fusion=fusion,
                                      monitor=monitor, max_steps=max_steps) for seed in seeds]
        entry = {"scenario": describe_scenario(scenario),
                 "truth": {"body_roll_rms_deg": _rms(np.concatenate(
                               [r.true_body_roll_deg - r.true_body_roll_deg.mean() for r in runs])),
                           "body_pitch_rms_deg": _rms(np.concatenate(
                               [r.true_body_pitch_deg - r.true_body_pitch_deg.mean()
                                for r in runs])),
                           "body_pitch_mean_deg": _mean(np.concatenate(
                               [r.true_body_pitch_deg for r in runs])),
                           "camera_height_mean_m": _mean(np.concatenate(
                               [r.true_camera_height for r in runs]))},
                 "imu": runs[0].imu, "timing": timing_statistics(runs),
                 "gpu_memory_mb": runs[0].gpu_memory_mb,
                 "arms": {arm.name: arm_statistics(runs, arm.name, thresholds) for arm in arms}}
        report["scenarios"][scenario.id] = entry
        kept[scenario.id] = runs[0]
        if log is not None:
            for arm in arms:
                log(format_row(scenario.id, arm.name, entry["arms"][arm.name]["aggregate"]))
    if output_dir is not None:
        output = Path(output_dir)
        write_report(report, output)
        if plots:
            from robosoccer_perception.visualization import humanoid_plots

            humanoid_plots.save_benchmark_figures(report, kept, output / "plots")
    return report


# ------------------------------------------------------------------ report files
SUMMARY_COLUMNS = (
    "runs", "position_rmse_all", "position_p95_all", "yaw_rmse_all", "yaw_p95_all",
    "position_rmse", "yaw_rmse", "tracking_fraction", "divergence_rate", "diverged_fraction",
    "wrong_half_rate", "overconfident_fraction", "camera_pitch_rmse_deg",
    "camera_roll_rmse_deg", "camera_pitch_bias_deg", "camera_roll_bias_deg",
    "camera_height_rmse_m", "body_pitch_rmse_deg", "body_roll_rmse_deg", "tilt_std_rms_deg",
    "tilt_consistency", "projection_error_mean_m", "projection_error_p95_m",
    "position_std_rms_m", "position_consistency", "position_within_2std")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.5g}"
    return str(value)


def _md(value, digits: int = 2, scale: float = 1.0, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * scale:.{digits}f}{suffix}"


def format_summary(report: dict[str, Any]) -> str:
    """The benchmark as Markdown: one table per scenario."""
    meta = report["meta"]
    lines = ["# Phase 4 benchmark: localization on a moving body", "",
             f"Created {meta['created']}; seeds {meta['seeds']}; images "
             f"{meta['resolution'][0]} x {meta['resolution'][1]}.", "",
             f"**Synthetic only.** {meta['synthetic_only']}", "", "## Arms", ""]
    for name, arm in meta["arms"].items():
        flag = "  **(uses simulator ground truth)**" if arm["uses_ground_truth"] else ""
        lines.append(f"- `{name}`: {arm['description']}{flag}")
    lines += ["", "Position and yaw RMSE cover the whole run. Camera errors, projection error "
              f"and consistency leave out the first {meta['settle_s']:.0f} s. Consistency is "
              "RMS position error over RMS reported standard deviation: 1 is consistent, above "
              "1 is overconfident.", ""]
    for scenario_id, entry in report["scenarios"].items():
        info, truth = entry["scenario"], entry["truth"]
        lines += [f"## {scenario_id}: {info['title']}", "", info["description"], "",
                  f"Robot `{info['robot']}`, mount `{info['mount']}` (camera "
                  f"{info['camera_height_standing_m']:.3f} m above the ground when standing), "
                  f"motion `{info['motion']}`, IMU `{info['imu']}`"
                  + (f" with {info['imu_overrides']}" if info["imu_overrides"] else "") + ". "
                  f"True body tilt: roll {_md(truth['body_roll_rms_deg'])} deg RMS, pitch "
                  f"{_md(truth['body_pitch_rms_deg'])} deg RMS about a mean of "
                  f"{_md(truth['body_pitch_mean_deg'])} deg. Expectation: {info['expectation']}.",
                  "",
                  "| arm | pos RMSE [m] | yaw RMSE [deg] | cam pitch RMSE [deg] | cam roll RMSE "
                  "[deg] | pitch bias [deg] | projection error [m] | consistency | within 2 std "
                  "| ended not localized | converged but wrong |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, data in entry["arms"].items():
            a = data["aggregate"]
            lines.append(
                f"| `{name}` | {_md(a['position_rmse_all'], 3)} | {_md(a['yaw_rmse_all'])} | "
                f"{_md(a['camera_pitch_rmse_deg'])} | {_md(a['camera_roll_rmse_deg'])} | "
                f"{_md(a['camera_pitch_bias_deg'])} | {_md(a['projection_error_mean_m'], 3)} | "
                f"{_md(a['position_consistency'])} | {_md(a['position_within_2std'], 0, 100, ' %')}"
                f" | {_md(a['divergence_rate'], 0, 100, ' %')} | "
                f"{_md(a['overconfident_fraction'], 1, 100, ' %')} |")
        timing = entry["timing"]
        if timing:
            lines += ["", f"Latency of the primary arm: orientation "
                      f"{timing['orientation']['mean']:.2f} ms, detector "
                      f"{timing['detector']['mean']:.1f}, segmenter "
                      f"{timing['segmenter']['mean']:.1f}, geometry "
                      f"{timing['geometry']['mean']:.1f}, filter {timing['filter']['mean']:.1f}; "
                      f"total {timing['total']['mean']:.1f} ms ({timing['fps']:.1f} FPS); "
                      f"rendering {timing['render_ms_mean']:.1f} ms is the simulator's."]
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, default=_json_default))
    (output / "summary.md").write_text(format_summary(report))
    with open(output / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "arm", *SUMMARY_COLUMNS])
        for scenario_id, entry in report["scenarios"].items():
            for arm, data in entry["arms"].items():
                writer.writerow([scenario_id, arm, *(_cell(data["aggregate"].get(column))
                                                     for column in SUMMARY_COLUMNS)])


# ------------------------------------------------------------------ sweeps
#: parameter -> (EstimateError field, unit shown, title)
SWEEP_PARAMETERS = {"pitch_deg": ("pitch_deg", "deg", "Camera pitch error"),
                    "roll_deg": ("roll_deg", "deg", "Camera roll error"),
                    "height_m": ("height", "m", "Camera height error")}
DEFAULT_SWEEP = {"pitch_deg": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
                 "roll_deg": (-2.0, -1.0, 0.0, 1.0, 2.0),
                 "height_m": (-0.05, -0.02, 0.0, 0.02, 0.05)}
SWEEP_BASES = (FIXED, IMU, IMU_UNCERTAINTY, IMU_SELFCAL)
DEFAULT_TIMING_MS = (-80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0)


def sweep_arms(sweep: dict[str, tuple[float, ...]], bases: tuple[FusionArm, ...] = SWEEP_BASES
               ) -> tuple[list[FusionArm], dict[str, tuple[str, str, float]]]:
    """Arms for a calibration sweep: every base arm with every deliberate error (ESTIMATE
    MINUS TRUTH). Returns the arms and ``name -> (base, parameter, value)``. The zero-error
    arm of each base is shared by the three parameters."""
    arms: list[FusionArm] = []
    index: dict[str, tuple[str, str, float]] = {}
    for base in bases:
        arms.append(dataclasses.replace(base, name=f"{base.name}@0"))
        for parameter, values in sweep.items():
            field = SWEEP_PARAMETERS[parameter][0]
            for value in values:
                if value == 0.0:
                    index[f"{base.name}@{parameter}=0"] = (base.name, parameter, 0.0)
                    continue
                name = f"{base.name}@{parameter}={value:g}"
                arms.append(dataclasses.replace(
                    base, name=name, estimate_error=EstimateError(**{field: value})))
                index[name] = (base.name, parameter, float(value))
    return arms, index


def run_calibration_sweep(scenario: HumanoidScenario, seeds: list[int],
                          resources: BenchmarkResources, output_dir=None, *,
                          sweep: dict[str, tuple[float, ...]] | None = None,
                          fusion: FusionConfig | None = None,
                          monitor: MonitorConfig | None = None, plots: bool = True,
                          max_steps: int | None = None,
                          log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Localization accuracy against a deliberately wrong camera calibration, for the fixed,
    the IMU-assisted, the uncertainty-aware and the self-calibrating arm. Every arm of a seed
    works from the same images, IMU samples and odometry."""
    sweep = {key: tuple(values) for key, values in (sweep or DEFAULT_SWEEP).items()}
    arms, index = sweep_arms(sweep)
    arms = [ORACLE, *arms]                       # the oracle goes first: it runs the networks
    thresholds = MetricThresholds()
    runs = []
    for seed in seeds:
        runs.append(run_humanoid_scenario(scenario, seed, resources, tuple(arms), fusion=fusion,
                                          monitor=monitor, max_steps=max_steps))
        if log is not None:
            log(f"calibration sweep: seed {seed} done ({len(arms)} arms)")
    stats = {arm.name: arm_statistics(runs, arm.name, thresholds) for arm in arms}
    report: dict[str, Any] = {
        "scenario": scenario.id, "seeds": list(seeds), "synthetic_only": SYNTHETIC_NOTE,
        "convention": "error = the robot's estimate minus the truth",
        "oracle": stats[ORACLE.name]["aggregate"], "parameters": {}}
    for parameter, values in sweep.items():
        _, unit, title = SWEEP_PARAMETERS[parameter]
        entry: dict[str, Any] = {"title": title, "unit": unit, "bases": {}}
        for base in SWEEP_BASES:
            points = []
            for value in values:
                name = f"{base.name}@0" if value == 0.0 else f"{base.name}@{parameter}={value:g}"
                points.append({"value": float(value), "aggregate": stats[name]["aggregate"],
                               "final_correction_deg": stats[name]["final_correction_deg"]})
            entry["bases"][base.name] = points
        report["parameters"][parameter] = entry
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "calibration_sweep.json").write_text(
            json.dumps(report, indent=2, default=_json_default))
        with open(output / "calibration_sweep.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parameter", "unit", "arm", "error", "position_rmse_all",
                             "yaw_rmse_all", "camera_pitch_bias_deg", "projection_error_mean_m",
                             "position_consistency", "divergence_rate",
                             "final_correction_roll_pitch_deg"])
            for parameter, entry in report["parameters"].items():
                for base, points in entry["bases"].items():
                    for point in points:
                        a = point["aggregate"]
                        writer.writerow([parameter, entry["unit"], base, point["value"], *(
                            _cell(a[k]) for k in (
                                "position_rmse_all", "yaw_rmse_all", "camera_pitch_bias_deg",
                                "projection_error_mean_m", "position_consistency",
                                "divergence_rate")),
                            json.dumps(point["final_correction_deg"])])
        if plots:
            from robosoccer_perception.visualization import humanoid_plots

            humanoid_plots.save_sweep_figure(report, output / "calibration_sensitivity.png")
    return report


def run_timing_sweep(scenario: HumanoidScenario, seeds: list[int],
                     resources: BenchmarkResources, output_dir=None, *,
                     offsets_ms: tuple[float, ...] = DEFAULT_TIMING_MS,
                     fusion: FusionConfig | None = None, plots: bool = True,
                     max_steps: int | None = None,
                     log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Localization accuracy against a camera / IMU timestamp mismatch: the arm pairs every
    image with the orientation of ``offset`` milliseconds later (positive) or earlier."""
    arms = [ORACLE, FIXED]
    for offset in offsets_ms:
        for base in (IMU, IMU_UNCERTAINTY):
            arms.append(dataclasses.replace(base, name=f"{base.name}@{offset:g}ms",
                                            timestamp_bias_s=offset / 1000.0))
    arms.append(dataclasses.replace(IMU, name="imu_latest", sync="latest"))
    thresholds = MetricThresholds()
    runs = []
    for seed in seeds:
        runs.append(run_humanoid_scenario(scenario, seed, resources, tuple(arms), fusion=fusion,
                                          max_steps=max_steps))
        if log is not None:
            log(f"timing sweep: seed {seed} done ({len(arms)} arms)")
    stats = {arm.name: arm_statistics(runs, arm.name, thresholds)["aggregate"] for arm in arms}
    report = {"scenario": scenario.id, "seeds": list(seeds), "synthetic_only": SYNTHETIC_NOTE,
              "camera_latency_ms": 1000.0 * scenario.camera_latency_s,
              "convention": "offset = the image timestamp the arm uses minus the true one",
              "offsets_ms": list(offsets_ms), "oracle": stats[ORACLE.name],
              "fixed": stats[FIXED.name], "imu_latest": stats["imu_latest"],
              "imu": [stats[f"imu@{o:g}ms"] for o in offsets_ms],
              "imu_uncertainty": [stats[f"imu_uncertainty@{o:g}ms"] for o in offsets_ms]}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "timing_sweep.json").write_text(
            json.dumps(report, indent=2, default=_json_default))
        if plots:
            from robosoccer_perception.visualization import humanoid_plots

            humanoid_plots.save_timing_figure(report, output / "timing_sensitivity.png")
    return report
