#!/usr/bin/env python3
"""Run the Phase 3 benchmark: localization from rendered camera images, compared with the
oracle, with dead reckoning and with "perfect vision", on identical runs.

Examples:
    python scripts/benchmark_vision_localization.py                 # all scenarios, 5 seeds
    python scripts/benchmark_vision_localization.py --seeds 2 v01 v04
    python scripts/benchmark_vision_localization.py --sweep-only    # calibration sensitivity
    python scripts/benchmark_vision_localization.py --no-sweep --no-plots

Needs the trained detector and segmenter (README, "Vision"), a GPU or patience, and an OpenGL
context for MuJoCo. Results go to ``outputs/vision_localization/benchmark/`` (results.json,
summary.md, CSV files, timeseries/, plots/, frames/) and ``.../sweep/``.
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
    parser.add_argument("--settings", default="vision_localization",
                        help="configs/evaluation/<name>.yaml or a path")
    parser.add_argument("--output", default="outputs/vision_localization")
    parser.add_argument("--device", help="override the device (cpu, cuda:0)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-sweep", action="store_true", help="skip the calibration sweep")
    parser.add_argument("--sweep-only", action="store_true", help="run only the sweep")
    parser.add_argument("--sweep-seeds", type=int)
    args = parser.parse_args()

    from robosoccer_perception.vision_benchmark import benchmark, setup
    from robosoccer_perception.vision_benchmark.scenarios import get_scenario, select_scenarios

    settings = setup.load_settings(args.settings, {"device": args.device} if args.device else None)
    resources = setup.load_resources(settings)
    output = Path(args.output)
    start = time.perf_counter()
    if not args.sweep_only:
        seeds = list(range(args.seeds or int(settings.get("seeds", 5))))
        print(benchmark.HEADER, flush=True)
        benchmark.run_vision_benchmark(
            select_scenarios(args.scenarios), seeds, resources, output / "benchmark",
            plots=not args.no_plots, log=lambda line: print(line, flush=True))
        print(f"\nbenchmark: {time.perf_counter() - start:.0f} s. Results in "
              f"{output / 'benchmark'}/ (summary.md, results.json, plots/, frames/)", flush=True)
    if not args.no_sweep:
        sweep = settings.get("sweep", {})
        seeds = list(range(args.sweep_seeds or int(sweep.get("seeds", 3))))
        parameters = {key: tuple(values) for key, values in
                      (sweep.get("parameters") or benchmark.DEFAULT_SWEEP).items()}
        tic = time.perf_counter()
        report = benchmark.run_calibration_sweep(
            get_scenario(sweep.get("scenario", "v04_rectangle")), seeds, resources,
            output / "sweep", sweep=parameters, belief=sweep.get("wider_belief"),
            plots=not args.no_plots,
            log=lambda line: print(line, flush=True))
        for parameter, entry in report["parameters"].items():
            print(f"\n{entry['label']} ({entry['axis']}):")
            wider = entry.get("wider_belief")
            for point in [*entry["points"], *(wider["points"] if wider else [])]:
                if wider and point is wider["points"][0]:
                    print(f"  -- {wider['label']} --")
                agg = point["aggregate"]
                print(f"  {point['raw_value']:>7g}  pos RMSE {agg['position_rmse_all']:.3f} m  "
                      f"yaw RMSE {agg['yaw_rmse_all']:.2f} deg  "
                      f"ended not localized {agg['divergence_rate'] * 100:.0f} %")
        print(f"\nsweep: {time.perf_counter() - tic:.0f} s. Results in {output / 'sweep'}/ "
              "(sweep.json, sweep.csv, sensitivity.png)")


if __name__ == "__main__":
    main()
