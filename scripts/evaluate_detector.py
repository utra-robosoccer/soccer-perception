#!/usr/bin/env python3
"""Evaluate a trained detector on one split of a generated dataset.

Examples:
    python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt
    python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt --split val
    python scripts/evaluate_detector.py best.pt --dataset outputs/datasets/baseline_strong --conf 0.4
    python scripts/evaluate_detector.py best.pt --device cpu --no-ultralytics-val

Writes ``<run>/eval_<dataset>_<split>/``: ``metrics.json`` (everything), ``summary.md``,
``stratified_recall.csv``, ``predictions.jsonl``, ``confusion_matrix.png``,
``recall_<class>.png`` and ``examples/{correct,false_positive,false_negative}/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", help="trained checkpoint (.pt)")
    parser.add_argument("--dataset", default="outputs/datasets/baseline")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--conf", type=float, default=0.25, help="operating confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU for a detection to count")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    parser.add_argument("--examples", type=int, default=8, help="example images per category")
    parser.add_argument("--limit", type=int, help="evaluate only the first N frames")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--no-ultralytics-val", action="store_true",
                        help="skip Ultralytics' own validation pass (the cross-check)")
    args = parser.parse_args()

    from robosoccer_perception.detection import yolo
    from robosoccer_perception.detection.evaluation import evaluate_detector, summary_markdown

    weights = Path(args.weights)
    dataset_name = Path(args.dataset).name
    output = Path(args.output) if args.output else (
        weights.parent.parent / f"eval_{dataset_name}_{args.split}")
    detector = yolo.YoloDetector(weights, device=args.device, conf=args.conf, imgsz=args.imgsz,
                                 half=args.half)
    result = evaluate_detector(detector, args.dataset, args.split, args.conf, args.iou, output,
                               max_examples=args.examples, limit=args.limit)
    if not args.no_ultralytics_val and args.limit is None:
        try:
            result["ultralytics_val"] = yolo.validate_detector(
                weights, args.dataset, args.split, imgsz=args.imgsz, device=args.device,
                output_dir=output / "ultralytics_val")
            (output / "metrics.json").write_text(json.dumps(result, indent=2))
        except Exception as error:  # noqa: BLE001 - the cross-check is optional
            # Ultralytics insists on a dataset with train and val splits; a test-only set fails.
            print(f"Ultralytics cross-check skipped: {type(error).__name__}: {error}")
    print(summary_markdown(result))
    if "ultralytics_val" in result:
        check = result["ultralytics_val"]
        print(f"Cross-check, Ultralytics' validator: mAP50 {check['map50']:.3f}, "
              f"mAP50-95 {check['map50_95']:.3f} (it counts unlabelled, heavily occluded "
              "objects as false positives; this report does not).")
    print(f"results: {output}")


if __name__ == "__main__":
    main()
