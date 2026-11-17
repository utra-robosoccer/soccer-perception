#!/usr/bin/env python3
"""Run ONE Phase 3 scenario with one seed and save what the pipeline saw and estimated.

Examples:
    python scripts/run_vision_localization.py --list
    python scripts/run_vision_localization.py v04_rectangle
    python scripts/run_vision_localization.py v13a_kidnapped_same_side --seed 3 --frames 6
    python scripts/run_vision_localization.py v04_rectangle --arms vision oracle

Writes to ``outputs/vision_localization/runs/<scenario>_seed<k>/``:
    arms.png                      position and yaw error of every arm over time
    vision_trajectory.png         true and estimated path (vision arm)
    vision_particles.png          particle cloud at the start, mid-run and the end
    <arm>.csv, metrics.json       per-step time series and metrics of every arm
    frames/step*_rgb.jpg          the rendered camera frame
    frames/step*_predictions.jpg  YOLO boxes and the U-Net mask
    frames/step*_features.jpg     extracted line points, junctions and goal-post feet
    frames/step*_ground.png       the same features on the ground, in the robot frame
    frames/step*_particles.png    the particle cloud on the field at that frame
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", default="v04_rectangle")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", help="subset of: vision oracle odometry perfect_vision")
    parser.add_argument("--frames", type=int, default=3, help="camera frames to save")
    parser.add_argument("--settings", default="vision_localization")
    parser.add_argument("--device")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="image size (default: the settings' 640x360; the camera's native "
                             "size is 1280 720)")
    parser.add_argument("--output", default="outputs/vision_localization/runs")
    args = parser.parse_args()

    from robosoccer_perception.vision_benchmark.scenarios import SCENARIOS, get_scenario

    if args.list:
        for scenario in SCENARIOS.values():
            print(f"{scenario.id:<32}{scenario.title}")
        return

    from robosoccer_perception.evaluation.metrics import MetricThresholds, compute_metrics
    from robosoccer_perception.vision_benchmark import benchmark, runner, setup

    scenario = get_scenario(args.scenario)
    overrides = {"device": args.device} if args.device else {}
    if args.resolution:
        overrides["resolution"] = list(args.resolution)
    settings = setup.load_settings(args.settings, overrides)
    resources = setup.load_resources(settings)
    arms = tuple(arm for arm in runner.DEFAULT_ARMS
                 if arm.source in scenario.sources and (not args.arms or arm.name in args.arms))
    steps = len(runner.build_trajectory(scenario, args.seed))
    keep = np.linspace(steps // 8, steps - 1 - steps // 8, max(args.frames, 1)).astype(int)
    start = time.perf_counter()
    run = runner.run_vision_scenario(scenario, args.seed, resources, arms,
                                     keep_frames=tuple(keep) if args.frames else (),
                                     snapshot_steps="auto")
    width, height = resources.resolution
    suffix = "" if (width, height) == (640, 360) else f"_{width}x{height}"
    output = Path(args.output) / f"{scenario.id}_seed{args.seed}{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    thresholds = MetricThresholds()
    metrics = {}
    print(f"{scenario.id}: {scenario.title} (seed {args.seed}, {steps} frames at "
          f"{width}x{height}, {time.perf_counter() - start:.0f} s)")
    print(f"{'arm':<16}{'converged':>10}{'pos RMSE':>10}{'yaw RMSE':>10}{'final err':>16}"
          f"{'wrong half':>12}")
    for name, arm in run.arms.items():
        arm.result.write_csv(output / f"{name}.csv")
        m = metrics[name] = compute_metrics(arm.result, thresholds)
        print(f"{name:<16}{str(m['converged']):>10}{m['position_rmse_all']:>9.3f}m"
              f"{m['yaw_rmse_all']:>8.2f}deg{m['final_position_error']:>8.3f}m "
              f"{m['final_yaw_error']:>5.2f}deg{str(m['final_wrong_half']):>10}")
        if arm.features is not None:
            metrics[name]["features"] = arm.features.summary()
        if arm.timing_ms is not None:
            metrics[name]["timing_ms"] = {key: float(values[benchmark.WARMUP_STEPS:].mean())
                                          for key, values in arm.timing_ms.items()}
            print("  per frame [ms]: " + ", ".join(
                f"{key} {value:.1f}" for key, value in metrics[name]["timing_ms"].items()))
    (output / "metrics.json").write_text(json.dumps(
        {"scenario": scenario.id, "seed": args.seed, "resolution": [width, height],
         "arms": metrics, "gpu_memory_mb": run.gpu_memory_mb}, indent=2,
        default=benchmark._json_default))
    # Reuse the benchmark's figure writer, then flatten its layout for a single run.
    benchmark._save_figures(scenario, run, output, thresholds)
    for path in (output / "plots").glob("*.png"):
        path.rename(output / path.name.replace(f"{scenario.id}_", ""))
    (output / "plots").rmdir()
    frames = output / "frames" / scenario.id
    if frames.is_dir():
        for path in frames.iterdir():
            path.rename(output / "frames" / path.name)
        frames.rmdir()
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
