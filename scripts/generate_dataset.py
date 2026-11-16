#!/usr/bin/env python3
"""Generate a labelled synthetic dataset from the MuJoCo scene.

Examples:
    python scripts/generate_dataset.py smoke
    python scripts/generate_dataset.py baseline --output outputs/datasets/baseline
    python scripts/generate_dataset.py baseline --randomization strong --name baseline_strong
    python scripts/generate_dataset.py smoke --seed 3 --resolution 1280 720 --overwrite
    python scripts/generate_dataset.py baseline --splits test --randomization strong --name baseline_strong_test

The dataset goes to ``outputs/datasets/<name>/`` (not tracked by Git). Generation is
deterministic: the same configuration and seed give the same files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from robosoccer_perception.datasets.generator import DatasetConfig, generate_dataset
from robosoccer_perception.datasets.io import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="dataset configuration: a name in configs/datasets/ or a YAML path")
    parser.add_argument("--output", help="output directory (default: outputs/datasets/<name>)")
    parser.add_argument("--name", help="override the dataset name")
    parser.add_argument("--seed", type=int, help="override the generation seed")
    parser.add_argument("--randomization", help="override the randomization preset of every split "
                                                "(none, mild, strong or a YAML path)")
    parser.add_argument("--field", help="override the field configuration")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        help="generate only these splits (for example an extra test set)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing generated dataset in the output directory")
    parser.add_argument("--no-validate", action="store_true", help="skip the consistency check")
    args = parser.parse_args()

    overrides = {key: value for key, value in (
        ("name", args.name), ("seed", args.seed), ("field", args.field),
        ("resolution", args.resolution)) if value is not None}
    cfg = DatasetConfig.load(args.config, overrides)
    import dataclasses
    if args.splits:
        cfg = dataclasses.replace(cfg, splits={split: entry for split, entry in cfg.splits.items()
                                               if split in args.splits})
    if args.randomization:
        cfg = dataclasses.replace(cfg, splits={
            split: dataclasses.replace(entry, randomization=args.randomization)
            for split, entry in cfg.splits.items()})
    output = Path(args.output or f"outputs/datasets/{cfg.name}")
    print(f"generating '{cfg.name}': {cfg.num_frames} frames at {cfg.width}x{cfg.height} "
          f"-> {output}")
    started = time.time()
    generate_dataset(cfg, output, overwrite=args.overwrite, progress=print)
    print(f"done in {time.time() - started:.1f} s")
    if not args.no_validate:
        print(json.dumps(validate_dataset(output), indent=2))
    print(f"dataset:        {output}")
    print(f"Ultralytics:    {(output / 'data.yaml').resolve()}")
    print(f"previews:       python scripts/inspect_dataset.py {output}")


if __name__ == "__main__":
    main()
