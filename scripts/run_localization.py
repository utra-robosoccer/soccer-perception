#!/usr/bin/env python3
"""Run one benchmark scenario and save its figures, time series and metrics.

Examples:
    python scripts/run_localization.py                      # s04_rectangle, seed 0
    python scripts/run_localization.py s10a_kidnapped_same_side --seed 3
    python scripts/run_localization.py s04_rectangle --mujoco --frames 5
    python scripts/run_localization.py --list
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robosoccer_perception.evaluation.benchmark import build_trajectory, run_scenario
from robosoccer_perception.evaluation.metrics import compute_metrics
from robosoccer_perception.evaluation.scenarios import SCENARIOS, get_scenario
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.visualization import diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", default="s04_rectangle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--particles", type=int, help="override the number of particles")
    parser.add_argument("--mujoco", action="store_true",
                        help="drive the MuJoCo scene and read ground truth back from it")
    parser.add_argument("--frames", type=int, default=0,
                        help="with --mujoco: number of stereo frames to save along the run")
    parser.add_argument("--output", default="outputs/localization")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    args = parser.parse_args()

    if args.list:
        for scenario in SCENARIOS.values():
            print(f"{scenario.id:<32}{scenario.title}")
        return

    scenario = get_scenario(args.scenario)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    overrides = {"num_particles": args.particles} if args.particles else None

    scene = None
    if args.mujoco:
        from robosoccer_perception.simulation.mujoco_scene import MujocoScene
        scene = MujocoScene(FieldGeometry.load(scenario.field), scenario_id=scenario.id)
    try:
        result = run_scenario(scenario, args.seed, filter_overrides=overrides,
                              snapshot_steps="auto", scene=scene)
        if scene is not None and args.frames > 0:
            from robosoccer_perception.simulation.stereo_camera import (
                VirtualStereoCamera, save_stereo_frame)
            trajectory = build_trajectory(scenario, args.seed)
            camera = VirtualStereoCamera(scene)
            for step in np.linspace(0, len(trajectory) - 1, args.frames).astype(int):
                scene.follow(trajectory, int(step))
                save_stereo_frame(camera.capture(), output / "frames",
                                  stem=f"{scenario.id}_step{int(step):05d}")
    finally:
        if scene is not None:
            scene.close()

    metrics = compute_metrics(result)
    stem = f"{scenario.id}_seed{args.seed}"
    result.write_csv(output / f"{stem}.csv")
    (output / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    paths = diagnostics.save_scenario_figures(scenario, result, output)

    print(f"{scenario.id}: {scenario.title}")
    print(f"  expectation       {scenario.expectation}")
    for key in ("converged", "convergence_time_s", "position_rmse", "position_p95", "yaw_rmse",
                "yaw_p95", "final_position_error", "final_yaw_error", "final_wrong_half",
                "recovery_times_s", "update_latency_ms_mean"):
        value = metrics[key]
        print(f"  {key:<24}{value:.3f}" if isinstance(value, float) else f"  {key:<24}{value}")
    for path in (output / f"{stem}.csv", output / f"{stem}_metrics.json", *paths.values()):
        print(path)


if __name__ == "__main__":
    main()
