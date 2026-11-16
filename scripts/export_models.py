#!/usr/bin/env python3
"""Export trained models for deployment experiments.

Examples:
    python scripts/export_models.py --detector outputs/detection/yolo11n/weights/best.pt
    python scripts/export_models.py --segmenter outputs/segmentation/unet_small/best.pt
    python scripts/export_models.py --detector best.pt --format torchscript --imgsz 640

The detector is exported by Ultralytics (``onnx`` or ``torchscript`` need nothing extra; other
formats may need packages that are not installed). The segmenter is exported to ONNX together
with a JSON file describing the pre-processing it expects. Exported files are written next to
the checkpoints and are not tracked by Git.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", help="trained detector checkpoint (.pt)")
    parser.add_argument("--segmenter", help="trained segmentation checkpoint (.pt)")
    parser.add_argument("--format", default="onnx", help="detector export format (default onnx)")
    parser.add_argument("--imgsz", type=int, default=640, help="detector input size")
    parser.add_argument("--half", action="store_true", help="FP16 detector export (GPU formats)")
    parser.add_argument("--size", type=int, nargs=2, default=(640, 368), metavar=("WIDTH", "HEIGHT"),
                        help="example segmenter input size, multiples of 16 (axes stay dynamic)")
    args = parser.parse_args()
    if not args.detector and not args.segmenter:
        parser.error("give --detector, --segmenter, or both")
    if args.detector:
        from robosoccer_perception.detection.yolo import export_detector
        print("detector:", export_detector(args.detector, args.format, args.imgsz, args.half))
    if args.segmenter:
        from robosoccer_perception.segmentation.export import export_segmenter_onnx
        print("segmenter:", export_segmenter_onnx(args.segmenter, width=args.size[0],
                                                  height=args.size[1]))


if __name__ == "__main__":
    main()
