"""The Phase 3 benchmark: every scenario, every seed, every arm; and the calibration sweep.

`run_vision_benchmark` writes, under its output directory:

    results.json          everything: configuration, per-scenario per-arm aggregates and runs,
                          feature-extraction scores, latency, GPU memory
    summary.md            the oracle / vision / odometry / perfect-vision comparison table
    summary.csv           one row per scenario and arm
    runs.csv              one row per scenario, arm and seed
    features.csv          geometric-perception scores per scenario and image-fed arm
    timeseries/           per-step truth, estimate, errors and filter state (CSV)
    plots/                error and trajectory figures of the first seed, and the overview
    frames/               kept camera frames of the first seed with overlays

`run_calibration_sweep` runs one scenario with a set of deliberately wrong camera estimates
and writes ``sweep.json``, ``sweep.csv`` and ``sensitivity.png``.

Localization metrics are Phase 1's (`evaluation/metrics.py`), unchanged, so that numbers are
comparable with docs/validation.md.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.evaluation import metrics as metrics_module
from robosoccer_perception.evaluation.metrics import MetricThresholds
from robosoccer_perception.inference import profiling
from robosoccer_perception.vision_benchmark.feature_metrics import FeatureStatistics
from robosoccer_perception.vision_benchmark.runner import (
    DEFAULT_ARMS, TIMING_KEYS, Arm, BenchmarkResources, EstimateError, ScenarioRun,
    run_vision_scenario)
from robosoccer_perception.vision_benchmark.scenarios import VisionScenario

WARMUP_STEPS = 5        # first frames of a process are slower (lazy initialisation)
SUMMARY_COLUMNS = (
    "runs", "convergence_rate", "convergence_time_s_mean", "convergence_rate_symmetric",
    "position_rmse", "position_mae", "position_p95", "yaw_rmse", "yaw_mae", "yaw_p95",
    "position_rmse_all", "position_mae_all", "position_p95_all", "yaw_rmse_all", "yaw_mae_all",
    "yaw_p95_all", "tracking_fraction", "diverged_fraction", "divergence_rate",
    "wrong_half_rate", "wrong_half_fraction", "ambiguity_flag_fraction", "recovery_rate",
    "recovery_time_s_mean", "recovery_rate_symmetric", "recovery_time_symmetric_s_mean",
    "overconfident_fraction", "update_latency_ms_mean")


def _meta(resources: BenchmarkResources, seeds, thresholds, arms) -> dict[str, Any]:
    vision = resources.vision
    device = getattr(vision, "device", "cpu")
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seeds": list(seeds),
        "thresholds": dataclasses.asdict(thresholds),
        "resolution": list(resources.resolution), "camera": resources.camera,
        "models": vision.describe() if hasattr(vision, "describe") else str(type(vision)),
        "hardware": profiling.hardware_summary(device),
        "python": platform.python_version(), "numpy": np.__version__,
        "feature_extraction": dataclasses.asdict(resources.extraction),
        "measurement_noise": dataclasses.asdict(resources.noise),
        "particle_filter": dataclasses.asdict(resources.particle_filter_config(False)),
        "particle_filter_image_fed": dataclasses.asdict(resources.particle_filter_config(True)),
        "arms": {arm.name: {"source": arm.source, "description": arm.description,
                            "estimate_error": dataclasses.asdict(arm.estimate_error)}
                 for arm in arms},
        "synthetic_only": "Every image is rendered by this repository's simulator and the "
                          "models were trained on images from the same simulator. Nothing "
                          "here is evidence about real camera footage.",
    }


def _aggregate_arm(runs: list[ScenarioRun], name: str, thresholds: MetricThresholds
                   ) -> dict[str, Any]:
    results = [run.arms[name].result for run in runs]
    per_run = [{"seed": run.seed, **metrics_module.compute_metrics(run.arms[name].result,
                                                                  thresholds)} for run in runs]
    return {"aggregate": metrics_module.aggregate_metrics(results, per_run, thresholds),
            "runs": per_run}


def _timing(runs: list[ScenarioRun]) -> dict[str, Any] | None:
    samples: dict[str, list[np.ndarray]] = {key: [] for key in TIMING_KEYS}
    for run in runs:
        for arm in run.arms.values():
            if arm.timing_ms is not None:
                for key in TIMING_KEYS:
                    samples[key].append(arm.timing_ms[key][WARMUP_STEPS:])
    if not samples["total"]:
        return None
    return {key: profiling.latency_stats(np.concatenate(values)) for key, values in samples.items()}


def _features(runs: list[ScenarioRun]) -> dict[str, Any]:
    pooled: dict[str, FeatureStatistics] = {}
    for run in runs:
        for name, arm in run.arms.items():
            if arm.features is not None:
                pooled.setdefault(name, FeatureStatistics()).extend(arm.features)
    return {name: statistics.summary() for name, statistics in pooled.items()}


def run_vision_benchmark(scenarios: list[VisionScenario], seeds: list[int],
                         resources: BenchmarkResources, output_dir=None, *,
                         arms: tuple[Arm, ...] | None = None,
                         thresholds: MetricThresholds | None = None, plots: bool = True,
                         log=print) -> dict[str, Any]:
    """Run every scenario with every seed and every arm. Returns what goes into results.json."""
    thresholds = thresholds or MetricThresholds()
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        (output / "timeseries").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"meta": _meta(resources, seeds, thresholds, arms or DEFAULT_ARMS),
                              "scenarios": {}}
    all_timing: list[ScenarioRun] = []
    gpu_peak: dict[str, float] = {}
    for scenario in scenarios:
        chosen = arms if arms is not None else tuple(
            arm for arm in DEFAULT_ARMS if arm.source in scenario.sources)
        runs: list[ScenarioRun] = []
        started = time.perf_counter()
        for index, seed in enumerate(seeds):
            first = index == 0
            run = run_vision_scenario(scenario, seed, resources, chosen,
                                      keep_frames="auto" if first and plots else (),
                                      snapshot_steps="auto" if first and plots else ())
            runs.append(run)
            for name, value in (run.gpu_memory_mb or {}).items():
                gpu_peak[name] = max(gpu_peak.get(name, 0.0), value)
            if output is not None:
                for name, arm in run.arms.items():
                    arm.result.write_csv(
                        output / "timeseries" / f"{scenario.id}_{name}_seed{seed}.csv")
            if first and output is not None and plots:
                _save_figures(scenario, run, output, thresholds)
            # Drop what is only needed for figures before the next seed (images are large).
            run.frames.clear()
            for arm in run.arms.values():
                arm.result.snapshots.clear()
        entry = {
            "title": scenario.title, "description": scenario.description,
            "expectation": scenario.expectation, "field": scenario.field, "init": scenario.init,
            "mount_error": scenario.mount_error.describe(),
            "head_jitter": dataclasses.asdict(scenario.jitter),
            "corruption": None if scenario.corruption is None else scenario.corruption.describe(),
            "randomization": scenario.randomization, "robots": len(scenario.robots),
            "steps": int(len(runs[0].arms[chosen[0].name].result.time)),
            "arms": {arm.name: _aggregate_arm(runs, arm.name, thresholds) for arm in chosen},
            "features": _features(runs), "timing_ms": _timing(runs),
            "render_ms_mean": float(np.mean([run.render_ms_mean for run in runs])),
            "wall_time_s": time.perf_counter() - started,
        }
        report["scenarios"][scenario.id] = entry
        all_timing.extend(runs)
        for arm in chosen:
            log(format_row(scenario.id, arm.name, entry["arms"][arm.name]["aggregate"]))
        # Keep only what the overall timing needs.
        for run in runs:
            for arm in run.arms.values():
                arm.features = None
    report["overall"] = {"timing_ms": _timing(all_timing), "gpu_memory_mb": gpu_peak or None,
                         "process_rss_mb": profiling.process_memory_mb()}
    if output is not None:
        write_report(report, output)
        if plots:
            from robosoccer_perception.visualization import vision_localization_plots as vlp
            vlp.plot_arm_summary(report).savefig(output / "plots" / "overview_position_rmse.png")
            vlp.plot_arm_summary(report, "yaw_rmse_all", "Yaw RMSE over the whole run [deg]",
                                 5.0).savefig(output / "plots" / "overview_yaw_rmse.png")
    return report


def _save_figures(scenario: VisionScenario, run: ScenarioRun, output: Path,
                  thresholds: MetricThresholds) -> None:
    from robosoccer_perception.geometry.field import FieldGeometry
    from robosoccer_perception.visualization import diagnostics, field_plot
    from robosoccer_perception.visualization import vision_localization_plots as vlp
    from robosoccer_perception.inference.pipeline import render_overlay
    from robosoccer_perception.visualization.vision_overlay import save_image

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    field = FieldGeometry.load(scenario.field)
    label = f"{scenario.id} (seed {run.seed}): {scenario.title}"
    results = {name: arm.result for name, arm in run.arms.items()}
    vlp.plot_arm_errors(results, thresholds, label).savefig(plots / f"{scenario.id}_arms.png")
    if "vision" in run.arms:
        vision = run.arms["vision"].result
        field_plot.plot_trajectory(field, vision.truth, vision.estimate, label + " [vision]"
                                   ).savefig(plots / f"{scenario.id}_vision_trajectory.png")
        if vision.snapshots:
            diagnostics.plot_snapshots(field, vision, label + " [vision]").savefig(
                plots / f"{scenario.id}_vision_particles.png")
    frames = output / "frames" / scenario.id
    for step, record in sorted(run.frames.items()):
        frames.mkdir(parents=True, exist_ok=True)
        stem = frames / f"step{step:04d}"
        save_image(f"{stem}_rgb.jpg", record.image)
        save_image(f"{stem}_predictions.jpg", render_overlay(record.image, record.vision))
        overlay = vlp.feature_overlay(record.image, record.features)
        overlay = vlp.draw_ground_features(overlay, record.features, record.calibration,
                                           record.camera_pose)
        save_image(f"{stem}_features.jpg", overlay)
        vlp.plot_ground_features(
            record.features, field, record.truth_pose,
            f"{scenario.id}, step {step}: extracted features in the robot frame"
        ).savefig(f"{stem}_ground.png")
        vlp.plot_particles(field, record, f"{scenario.id}, step {step}: particle cloud").savefig(
            f"{stem}_particles.png")


# ------------------------------------------------------------------ writing
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


def _percent(value) -> str:
    return "-" if value is None else f"{value * 100:.0f} %"


HEADER = (f"{'scenario':<30}{'arm':<16}{'conv':>6}{'posRMSE':>9}{'posP95':>8}{'yawRMSE':>9}"
          f"{'RMSEall':>9}{'yawAll':>8}{'diverg':>8}{'wrongHalf':>10}{'recov':>8}")


def format_row(scenario_id: str, arm: str, agg: dict[str, Any]) -> str:
    recovery = "-" if agg["recovery_time_s_mean"] is None else f"{agg['recovery_time_s_mean']:.1f}s"
    return (f"{scenario_id:<30}{arm:<16}{agg['convergence_rate'] * 100:>5.0f}%"
            f"{_fmt(agg['position_rmse'], 3):>9}{_fmt(agg['position_p95'], 3):>8}"
            f"{_fmt(agg['yaw_rmse'], 2):>9}{_fmt(agg['position_rmse_all'], 3):>9}"
            f"{_fmt(agg['yaw_rmse_all'], 2):>8}{agg['divergence_rate'] * 100:>7.0f}%"
            f"{agg['wrong_half_rate'] * 100:>9.0f}%{recovery:>8}")


def _recovery_text(agg: dict[str, Any]) -> str:
    if agg["recovery_rate"] is None:
        return "-"
    if agg["recovery_time_s_mean"] is not None:
        return f"{agg['recovery_time_s_mean']:.1f} s ({agg['recovery_rate'] * 100:.0f} %)"
    if agg["recovery_time_symmetric_s_mean"] is not None:
        return (f"mirror only, {agg['recovery_time_symmetric_s_mean']:.1f} s "
                f"({agg['recovery_rate_symmetric'] * 100:.0f} %)")
    return "none"


def format_summary(report: dict[str, Any]) -> str:
    meta = report["meta"]
    th = meta["thresholds"]
    lines = [
        "# Vision-localization benchmark (Phase 3)", "",
        f"Created {meta['created']}. Seeds: {meta['seeds']}. Images: "
        f"{meta['resolution'][0]}x{meta['resolution'][1]}. Device: {meta['hardware'].get('gpu', meta['hardware']['device'])}.",
        "", f"**{meta['synthetic_only']}**", "",
        "Arms (identical trajectory, odometry stream, initial guess and filter seed):", "",
    ]
    lines += [f"- **{name}**: {arm['description']}" for name, arm in meta["arms"].items()]
    lines += [
        "", "How to read the table:", "",
        f"- **Localized** means position error < {th['localized_position']} m and yaw error < "
        f"{th['localized_yaw_deg']} deg. **Conv.** is the share of runs that became localized "
        f"and stayed so for {th['hold_time']} s.",
        "- **RMSE / MAE / P95 (tracking)** are pooled over all seeds, over the steps after "
        "convergence. **RMSE (all)** covers every step of every run, converged or not: it is "
        "the number to use for arms that never converge, such as dead reckoning.",
        "- **Diverged** and **Wrong half** are the share of runs that END not localized / on "
        "the mirrored pose. **Recovery** is the mean time from a kidnapping until localized.",
        "",
        "| Scenario | Arm | Conv. | t conv [s] | Pos RMSE [m] | Pos MAE [m] | Pos P95 [m] | "
        "Yaw RMSE [deg] | Yaw MAE [deg] | Yaw P95 [deg] | Pos RMSE (all) [m] | "
        "Yaw RMSE (all) [deg] | Diverged | Wrong half | Recovery |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scenario_id, entry in report["scenarios"].items():
        for arm, data in entry["arms"].items():
            agg = data["aggregate"]
            lines.append(
                f"| {scenario_id} | {arm} | {_percent(agg['convergence_rate'])} "
                f"| {_fmt(agg['convergence_time_s_mean'], 1)} | {_fmt(agg['position_rmse'], 3)} "
                f"| {_fmt(agg['position_mae'], 3)} | {_fmt(agg['position_p95'], 3)} "
                f"| {_fmt(agg['yaw_rmse'], 2)} | {_fmt(agg['yaw_mae'], 2)} "
                f"| {_fmt(agg['yaw_p95'], 2)} | {_fmt(agg['position_rmse_all'], 3)} "
                f"| {_fmt(agg['yaw_rmse_all'], 2)} | {_percent(agg['divergence_rate'])} "
                f"| {_percent(agg['wrong_half_rate'])} | {_recovery_text(agg)} |")
    lines += ["", "## Geometric perception (vision arm, scored against simulator ground truth)", "",
              "| Scenario | Line pts / frame | Line error mean / P95 [m] | Posts with range / "
              "bearing only | Post ground error mean [m] | Post bearing error mean [deg] | "
              "Junction precision | Junction recall (arms in view) | Junction error mean [m] | "
              "Circle found / expected | False circles | Valid projected | Frames with nothing |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for scenario_id, entry in report["scenarios"].items():
        f = entry["features"].get("vision")
        if not f or not f.get("frames"):
            continue
        junction, posts, circle = f["intersections"]["all"], f["goal_posts"], f["center_circle"]
        lines.append(
            f"| {scenario_id} | {f['observations_per_frame']['line_points']:.1f} "
            f"| {_fmt(f['line_points']['error_m']['mean'], 3)} / "
            f"{_fmt(f['line_points']['error_m']['p95'], 3)} "
            f"| {posts['with_range']} / {posts['bearing_only']} "
            f"| {_fmt(posts['ground_point_error_m']['mean'], 3)} "
            f"| {_fmt(posts['bearing_error_deg']['mean'], 2)} "
            f"| {_percent(junction['precision'])} ({junction['true']}/{junction['reported']}) "
            f"| {_percent(junction['recall_arms_in_view'])} "
            f"| {_fmt(junction['position_error_m']['mean'], 3)} "
            f"| {circle['found_when_expected']} / {circle['frames_expected']} | {circle['false']} "
            f"| {_percent(f['projection']['valid_share'])} "
            f"| {_percent(f['observations_per_frame']['frames_without_any_observation'])} |")
    timing = report.get("overall", {}).get("timing_ms")
    if timing:
        lines += ["", "## Latency of the vision arm (per frame, after warm-up, all scenarios)", "",
                  "| Stage | Mean [ms] | Median [ms] | P95 [ms] | Max [ms] |", "|---|---|---|---|---|"]
        names = {"detector": "YOLO11n detector", "segmenter": "U-Net segmenter",
                 "geometry": "Geometric extraction + ObservationFrame",
                 "filter": "Particle filter (predict, update, estimate)",
                 "total": "Total pipeline step"}
        for key in TIMING_KEYS:
            t = timing[key]
            lines.append(f"| {names[key]} | {t['mean_ms']:.1f} | {t['median_ms']:.1f} "
                         f"| {t['p95_ms']:.1f} | {t['max_ms']:.1f} |")
        memory = report["overall"].get("gpu_memory_mb")
        if memory:
            lines += ["", f"Peak GPU memory held by PyTorch: {memory['peak_allocated_mb']:.0f} MB "
                          f"allocated, {memory['peak_reserved_mb']:.0f} MB reserved (the CUDA "
                          "context is extra). Rendering is not part of the pipeline and is not "
                          "included."]
    lines += ["", "## Scenarios", ""]
    for scenario_id, entry in report["scenarios"].items():
        lines += [f"**{scenario_id}: {entry['title']}.** {entry['description']} "
                  f"Initialisation: `{entry['init']}`. Expectation: {entry['expectation']}.", ""]
    return "\n".join(lines) + "\n"


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
    with open(output / "runs.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        keys = None
        for scenario_id, entry in report["scenarios"].items():
            for arm, data in entry["arms"].items():
                for run in data["runs"]:
                    if keys is None:
                        keys = [k for k in run if not k.endswith("times_s")]
                        writer.writerow(["scenario", "arm", *keys])
                    writer.writerow([scenario_id, arm, *(_cell(run[k]) for k in keys)])
    with open(output / "features.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario", "arm", "frames", "line_points_per_frame", "line_error_mean_m",
            "line_error_p95_m", "line_within_2_sigma", "valid_projected_share",
            "rejected_share", "posts_with_range", "posts_bearing_only", "posts_unmatched",
            "post_ground_error_mean_m", "post_ground_error_p95_m", "post_bearing_error_mean_deg",
            "junctions_reported", "junction_precision", "junction_recall_in_view",
            "junction_recall_arms_in_view", "junction_error_mean_m", "junctions_unclassed",
            "circle_expected", "circle_found", "circle_false", "circle_error_mean_m",
            "frames_without_any_observation"])
        for scenario_id, entry in report["scenarios"].items():
            for arm, f in entry["features"].items():
                if not f.get("frames"):
                    continue
                junction, posts, circle = (f["intersections"]["all"], f["goal_posts"],
                                           f["center_circle"])
                writer.writerow([scenario_id, arm, f["frames"], *(_cell(v) for v in (
                    f["observations_per_frame"]["line_points"],
                    f["line_points"]["error_m"]["mean"], f["line_points"]["error_m"]["p95"],
                    f["line_points"]["within_2_sigma"], f["projection"]["valid_share"],
                    f["projection"]["rejected_share"], posts["with_range"], posts["bearing_only"],
                    posts["unmatched"], posts["ground_point_error_m"]["mean"],
                    posts["ground_point_error_m"]["p95"], posts["bearing_error_deg"]["mean"],
                    junction["reported"], junction["precision"], junction["recall"],
                    junction["recall_arms_in_view"], junction["position_error_m"]["mean"],
                    f["intersections"]["unclassed"], circle["frames_expected"],
                    circle["found_when_expected"], circle["false"],
                    circle["centre_error_m"]["mean"],
                    f["observations_per_frame"]["frames_without_any_observation"]))])


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


# ------------------------------------------------------------------ calibration sweep
#: parameter -> (EstimateError field, axis label, panel title)
SWEEP_PARAMETERS = {
    "pitch_deg": ("pitch_deg", "pitch error [deg]", "Camera pitch"),
    "height_m": ("height", "height error [m]", "Camera height"),
    "roll_deg": ("roll_deg", "roll error [deg]", "Camera roll"),
    "yaw_deg": ("yaw_deg", "yaw error [deg]", "Camera yaw (pan)"),
    "focal_scale": ("focal_scale", "focal length error [%]", "Focal length"),
}
DEFAULT_SWEEP = {
    "pitch_deg": (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0),
    "height_m": (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
    "roll_deg": (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
    "yaw_deg": (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0),
    "focal_scale": (0.95, 0.98, 1.0, 1.02, 1.05),
}


def sweep_arms(sweep: dict[str, tuple[float, ...]], belief: dict[str, Any] | None = None,
               uncertainty=None) -> tuple[list[Arm], dict[str, list], dict[str, list]]:
    """Arms for a calibration sweep: the nominal vision arm plus one arm per wrong estimate.

    ``belief`` optionally repeats one parameter's sweep with a different BELIEVED calibration
    uncertainty, e.g. ``{"parameter": "pitch_deg", "pitch_std_deg": 2.0}``: the same wrong
    estimates, but the robot admits that its pitch is poorly known. ``uncertainty`` is the
    configured `CalibrationUncertainty` that the belief overrides.
    Returns ``(arms, layout, belief_layout)``; a layout maps parameter -> [(value, arm name)].
    """
    arms = [Arm("vision", "vision", description="nominal camera estimate")]
    layout: dict[str, list] = {}
    for parameter, values in sweep.items():
        field = SWEEP_PARAMETERS[parameter][0]
        neutral = 1.0 if parameter == "focal_scale" else 0.0
        layout[parameter] = []
        for value in values:
            if abs(value - neutral) < 1e-12:
                layout[parameter].append((value, "vision"))
                continue
            name = f"{parameter}={value:+g}" if parameter != "focal_scale" else f"focal={value:g}"
            arms.append(Arm(name, "vision", EstimateError(**{field: value}),
                            description=f"camera estimate wrong by {parameter} {value:+g}"))
            layout[parameter].append((value, name))
    belief_layout: dict[str, list] = {}
    if belief:
        belief = dict(belief)
        parameter = belief.pop("parameter")
        believed = dataclasses.replace(uncertainty, **belief)
        field = SWEEP_PARAMETERS[parameter][0]
        belief_layout[parameter] = []
        for value in sweep[parameter]:
            name = f"{parameter}={value:+g} (wider belief)"
            arms.append(Arm(name, "vision", EstimateError(**{field: value}),
                            description=f"as {parameter} {value:+g}, believing {belief}",
                            believed_uncertainty=believed))
            belief_layout[parameter].append((value, name))
    return arms, layout, belief_layout


def run_calibration_sweep(scenario: VisionScenario, seeds: list[int],
                          resources: BenchmarkResources, output_dir=None, *,
                          sweep: dict[str, tuple[float, ...]] | None = None,
                          belief: dict[str, Any] | None = None,
                          thresholds: MetricThresholds | None = None, plots: bool = True,
                          log=print) -> dict[str, Any]:
    """Localization error as a function of camera-calibration error.

    The simulator's true camera is the scenario's; only the robot's ESTIMATE is changed, one
    parameter at a time (error = estimate minus truth). All arms share each frame's inference.
    """
    thresholds = thresholds or MetricThresholds()
    sweep = sweep or DEFAULT_SWEEP
    arms, layout, belief_layout = sweep_arms(sweep, belief, resources.noise.calibration)
    runs = []
    for seed in seeds:
        runs.append(run_vision_scenario(scenario, seed, resources, tuple(arms),
                                        score_features=False))
        log(f"sweep {scenario.id}: seed {seed} done ({len(arms)} arms)")
    aggregates = {arm.name: _aggregate_arm(runs, arm.name, thresholds) for arm in arms}
    report: dict[str, Any] = {
        "scenario": scenario.id, "seeds": list(seeds),
        "convention": "error = the robot's estimate minus the truth; the simulator's camera "
                      "is unchanged",
        "belief": belief, "meta": _meta(resources, seeds, thresholds, arms), "parameters": {}}
    for parameter, points in layout.items():
        _, axis, label = SWEEP_PARAMETERS[parameter]
        report["parameters"][parameter] = {
            "label": label, "axis": axis,
            "points": [{"value": (value - 1.0) * 100.0 if parameter == "focal_scale" else value,
                        "raw_value": value, "arm": name,
                        "aggregate": aggregates[name]["aggregate"]} for value, name in points]}
        if parameter in belief_layout:
            description = ", ".join(f"{key} = {value:g}" for key, value in belief.items()
                                    if key != "parameter")
            report["parameters"][parameter]["wider_belief"] = {
                "label": f"believing {description}",
                "points": [{"value": value, "raw_value": value, "arm": name,
                            "aggregate": aggregates[name]["aggregate"]}
                           for value, name in belief_layout[parameter]]}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "sweep.json").write_text(json.dumps(report, indent=2, default=_json_default))
        with open(output / "sweep.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parameter", "error", "arm", *SUMMARY_COLUMNS])
            for parameter, entry in report["parameters"].items():
                for point in [*entry["points"], *entry.get("wider_belief", {}).get("points", [])]:
                    writer.writerow([parameter, _cell(point["raw_value"]), point["arm"],
                                     *(_cell(point["aggregate"].get(c)) for c in SUMMARY_COLUMNS)])
        if plots:
            from robosoccer_perception.visualization import vision_localization_plots as vlp
            vlp.plot_sensitivity(report).savefig(output / "sensitivity.png")
    return report
