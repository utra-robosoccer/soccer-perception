#!/usr/bin/env python3
"""Run the localization benchmark and write results.json, CSV files, a summary and plots.

Examples:
    python scripts/benchmark.py                           # every scenario, seeds 0-9
    python scripts/benchmark.py --seeds 3                 # seeds 0-2 (quicker)
    python scripts/benchmark.py s01 s02 s10               # scenarios whose id starts with these
    python scripts/benchmark.py --no-plots --output outputs/quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from robosoccer_perception.evaluation.benchmark import HEADER, run_benchmark
from robosoccer_perception.evaluation.scenarios import select_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", help="scenario id prefixes (default: all)")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0 .. N-1)")
    parser.add_argument("--particles", type=int, help="override the number of particles")
    parser.add_argument("--output", default="outputs/benchmark")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    overrides = {"num_particles": args.particles} if args.particles else None
    output = Path(args.output)
    print(HEADER)
    start = time.perf_counter()
    run_benchmark(select_scenarios(args.scenarios), list(range(args.seeds)), output,
                  filter_overrides=overrides, plots=not args.no_plots)
    print(f"\n{time.perf_counter() - start:.0f} s. Results in {output}/ "
          "(results.json, summary.md, summary.csv, runs.csv, timeseries/, plots/)")


if __name__ == "__main__":
    main()
