#!/usr/bin/env python3
"""Train the field / field-line segmentation model on a generated dataset.

Examples:
    python scripts/train_segmentation.py                         # configs/segmentation/unet_small.yaml
    python scripts/train_segmentation.py unet_smoke              # 3-epoch smoke test
    python scripts/train_segmentation.py unet_small --dataset outputs/datasets/baseline --epochs 60
    python scripts/train_segmentation.py unet_small --device cpu --epochs 1 --batch 2

Writes ``<output>/<name>/``: ``best.pt``, ``last.pt``, ``history.csv``, ``history.json``,
``history.png`` and ``train_summary.json``.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="unet_small",
                        help="name in configs/segmentation/ or a YAML path (default: unet_small)")
    parser.add_argument("--dataset", help="dataset directory (overrides the configuration)")
    parser.add_argument("--output", help="output root (overrides the configuration)")
    parser.add_argument("--name", help="run name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--device", help="auto, cpu, cuda:0, ...")
    args = parser.parse_args()

    from robosoccer_perception.segmentation.trainer import (
        load_segmentation_config, train_segmentation)

    overrides = {key: getattr(args, key) for key in
                 ("dataset", "output", "name", "epochs", "batch", "workers", "patience", "seed",
                  "learning_rate", "device") if getattr(args, key) is not None}
    cfg = load_segmentation_config(args.config, overrides)
    summary = train_segmentation(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "hardware")}, indent=2))
    print(f"evaluate with:  python scripts/evaluate_segmentation.py {summary['best']} "
          f"--dataset {cfg['dataset']}")


if __name__ == "__main__":
    main()
