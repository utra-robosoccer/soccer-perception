#!/usr/bin/env python3
"""Train (fine-tune) the YOLO object detector on a generated dataset.

Examples:
    python scripts/train_detector.py --download yolo11n.pt      # once: fetch pretrained weights (5 MB)
    python scripts/train_detector.py                            # configs/detection/yolo11n.yaml
    python scripts/train_detector.py yolo11n_smoke              # 2-epoch smoke test
    python scripts/train_detector.py yolo11n --dataset outputs/datasets/baseline --epochs 60 --batch 8
    python scripts/train_detector.py yolo11n --device cpu --epochs 1

Writes ``<output>/<name>/``: ``weights/best.pt``, ``weights/last.pt``, ``results.csv`` (losses
and validation metrics per epoch), Ultralytics' plots, and ``train_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="yolo11n",
                        help="name in configs/detection/ or a YAML path (default: yolo11n)")
    parser.add_argument("--download", metavar="CHECKPOINT",
                        help="download a pretrained checkpoint into weights/ and exit")
    parser.add_argument("--dataset", help="dataset directory (overrides the configuration)")
    parser.add_argument("--output", help="output root (overrides the configuration)")
    parser.add_argument("--name", help="run name")
    parser.add_argument("--model", help="checkpoint or architecture, e.g. yolo26n.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, 0, ...")
    parser.add_argument("--resume", action="store_true", help="continue the run from last.pt")
    args = parser.parse_args()

    from robosoccer_perception.detection import yolo
    from robosoccer_perception.inference.profiling import hardware_summary

    if args.download:
        size, purpose = yolo.KNOWN_WEIGHTS.get(args.download, ("?", "unknown"))
        print(f"downloading {args.download}: {purpose}, {size}")
        print(yolo.download_weights(args.download))
        return

    overrides = {key: getattr(args, key) for key in
                 ("dataset", "output", "name", "model", "epochs", "batch", "imgsz", "workers",
                  "patience", "seed", "device") if getattr(args, key) is not None}
    cfg = yolo.load_detection_config(args.config, overrides)
    if args.resume:
        last = Path(cfg.get("output", "outputs/detection")) / cfg["name"] / "weights" / "last.pt"
        cfg.update(model=str(last), resume=True)
    started = time.time()
    summary = yolo.train_detector(cfg, cfg["dataset"])
    summary.update(config=cfg, hardware=hardware_summary(summary["device"]),
                   training_seconds=round(time.time() - started, 1))
    path = Path(summary["run_dir"]) / "train_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("run_dir", "best", "final_metrics",
                                              "training_seconds")}, indent=2))
    print(f"evaluate with:  python scripts/evaluate_detector.py {summary['best']} "
          f"--dataset {cfg['dataset']}")


if __name__ == "__main__":
    main()
