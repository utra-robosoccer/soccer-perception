#!/usr/bin/env python3
"""Evaluate a trained segmentation model on one split of a generated dataset.

Examples:
    python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt
    python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt --split val
    python scripts/evaluate_segmentation.py best.pt --dataset outputs/datasets/baseline_strong
    python scripts/evaluate_segmentation.py best.pt --device cpu

Writes ``<run>/eval_<dataset>_<split>/``: ``metrics.json``, ``summary.md``, ``per_frame.json``,
``confusion_matrix.png``, ``line_by_distance.png`` and ``examples/{typical,worst_line_iou}/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="trained checkpoint (best.pt)")
    parser.add_argument("--dataset", default="outputs/datasets/baseline")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    parser.add_argument("--tolerance", type=int, default=2,
                        help="pixels of slack for the tolerant line scores (default 2)")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--limit", type=int, help="evaluate only the first N frames")
    parser.add_argument("--output", help="output directory")
    args = parser.parse_args()

    from robosoccer_perception.segmentation.evaluation import evaluate_segmenter, summary_markdown
    from robosoccer_perception.segmentation.segmenter import Segmenter

    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else (
        checkpoint.parent / f"eval_{Path(args.dataset).name}_{args.split}")
    segmenter = Segmenter(checkpoint, device=args.device, half=args.half)
    result = evaluate_segmenter(segmenter, args.dataset, args.split, output,
                                max_examples=args.examples, limit=args.limit,
                                tolerance=args.tolerance)
    print(summary_markdown(result))
    print(f"results: {output}")


if __name__ == "__main__":
    main()
