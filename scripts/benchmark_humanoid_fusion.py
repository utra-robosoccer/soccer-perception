#!/usr/bin/env python3
"""Phase 4 benchmark: localization from the humanoid's virtual camera while the body moves.

Runs every scenario (standing, sway, walking, turning, disturbances, mount errors, camera
heights, IMU dropouts, timing errors) for several seeds. Every arm works from the same
rendered images, IMU samples and odometry and differs only in its camera-pose estimate:

    fixed               A  an upright, level body in every frame (Phase 3)
    oracle_orientation  B  the true camera orientation: an upper bound that uses ground truth
    imu                 C  IMU roll / pitch, interpolated to the image timestamp
    imu_uncertainty     D  C with per-frame uncertainty and the shared-error model
    imu_latest             C with the newest IMU estimate instead of interpolation
    imu_selfcal            D with the bounded mount-bias correction

Then two sweeps on one scenario: a deliberately wrong calibration (pitch, roll, height) and a
camera / IMU timestamp mismatch. See docs/phase4_validation.md.

Examples:
    python scripts/benchmark_humanoid_fusion.py                      # everything (about 1 h)
    python scripts/benchmark_humanoid_fusion.py h05 h10 --seeds 1    # two scenarios, one seed
    python scripts/benchmark_humanoid_fusion.py --sweeps-only
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", help="scenario id prefixes (default: all)")
    parser.add_argument("--seeds", type=int, help="number of seeds (0 .. N-1)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--settings", default="humanoid_fusion",
                        help="configs/evaluation/<name>.yaml or a path")
    parser.add_argument("--output", default="outputs/humanoid_fusion")
    parser.add_argument("--device", help="override the device (cpu, cuda:0)")
    parser.add_argument("--max-steps", type=int, help="shorten every run (for a quick look)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--sweeps-only", action="store_true")
    args = parser.parse_args()

    from robosoccer_perception.humanoid_benchmark import benchmark, setup
    from robosoccer_perception.humanoid_benchmark.scenarios import get_scenario, select_scenarios
    from robosoccer_perception.vision_benchmark import setup as vision_setup

    settings = setup.load_settings(args.settings, {"device": args.device} if args.device else None)
    resources = vision_setup.load_resources(settings)
    fusion, monitor = setup.fusion_config(settings), setup.monitor_config(settings)
    output = Path(args.output)

    def log(line: str) -> None:
        print(line, flush=True)

    def seeds(count) -> list[int]:
        return list(range(args.first_seed, args.first_seed + int(count)))

    start = time.perf_counter()
    if not args.sweeps_only:
        print(benchmark.HEADER, flush=True)
        benchmark.run_fusion_benchmark(
            select_scenarios(args.scenarios), seeds(args.seeds or settings.get("seeds", 3)),
            resources, output / "benchmark", fusion=fusion, monitor=monitor,
            plots=not args.no_plots, max_steps=args.max_steps, log=log)
        print(f"\nbenchmark: {time.perf_counter() - start:.0f} s. Results in "
              f"{output / 'benchmark'}/ (summary.md, results.json, summary.csv, plots/)",
              flush=True)
    if not args.no_sweeps:
        sweep = settings.get("calibration_sweep", {})
        tic = time.perf_counter()
        report = benchmark.run_calibration_sweep(
            get_scenario(sweep.get("scenario", "h05_walk_rectangle")),
            seeds(args.seeds or sweep.get("seeds", 3)), resources, output / "sweeps",
            sweep=sweep.get("parameters"), fusion=fusion, monitor=monitor,
            plots=not args.no_plots, max_steps=args.max_steps, log=log)
        for parameter, entry in report["parameters"].items():
            print(f"\n{entry['title']} [{entry['unit']}]: position RMSE in metres")
            bases = list(entry["bases"])
            print("  " + f"{'error':>8}" + "".join(f"{b:>18}" for b in bases))
            for index, point in enumerate(entry["bases"][bases[0]]):
                print("  " + f"{point['value']:>8g}" + "".join(
                    f"{entry['bases'][b][index]['aggregate']['position_rmse_all']:>18.3f}"
                    for b in bases))
        print(f"\ncalibration sweep: {time.perf_counter() - tic:.0f} s", flush=True)
        timing = settings.get("timing_sweep", {})
        tic = time.perf_counter()
        report = benchmark.run_timing_sweep(
            get_scenario(timing.get("scenario", "h05_walk_rectangle")),
            seeds(args.seeds or timing.get("seeds", 3)), resources, output / "sweeps",
            offsets_ms=tuple(timing.get("offsets_ms", benchmark.DEFAULT_TIMING_MS)),
            fusion=fusion, plots=not args.no_plots, max_steps=args.max_steps, log=log)
        print("\nTimestamp mismatch [ms]: position RMSE [m], camera pitch RMSE [deg]")
        for offset, imu, aware in zip(report["offsets_ms"], report["imu"],
                                      report["imu_uncertainty"]):
            print(f"  {offset:>6g}  imu {imu['position_rmse_all']:.3f} m "
                  f"({imu['camera_pitch_rmse_deg']:.2f} deg)   imu_uncertainty "
                  f"{aware['position_rmse_all']:.3f} m")
        print(f"  newest-sample pairing: {report['imu_latest']['position_rmse_all']:.3f} m; "
              f"fixed camera: {report['fixed']['position_rmse_all']:.3f} m; oracle: "
              f"{report['oracle']['position_rmse_all']:.3f} m")
        print(f"\ntiming sweep: {time.perf_counter() - tic:.0f} s. Results in "
              f"{output / 'sweeps'}/", flush=True)


if __name__ == "__main__":
    main()
