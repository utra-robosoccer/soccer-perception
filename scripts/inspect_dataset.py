#!/usr/bin/env python3
"""Check a generated dataset and save label previews for visual inspection.

Examples:
    python scripts/inspect_dataset.py outputs/datasets/smoke
    python scripts/inspect_dataset.py outputs/datasets/baseline --split test --count 24
    python scripts/inspect_dataset.py outputs/datasets/baseline --mode ball_occluded

Each preview shows the camera image next to the same image with its labels drawn on it: the
semantic mask (field tinted, lines magenta, background dimmed) and the object boxes. Previews
go to ``<dataset>/previews/``, statistics to ``<dataset>/statistics.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robosoccer_perception.datasets.io import SoccerDataset, validate_dataset
from robosoccer_perception.visualization import vision_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset directory")
    parser.add_argument("--split", help="only this split")
    parser.add_argument("--mode", help="only frames of this layout mode")
    parser.add_argument("--count", type=int, default=12, help="number of previews (default 12)")
    parser.add_argument("--seed", type=int, default=0, help="which frames are picked")
    parser.add_argument("--output", help="preview directory (default: <dataset>/previews)")
    args = parser.parse_args()

    statistics = validate_dataset(args.dataset)
    print(json.dumps(statistics, indent=2))
    dataset = SoccerDataset(args.dataset)
    (dataset.root / "statistics.json").write_text(json.dumps(statistics, indent=2))
    records = [r for r in dataset.frames(args.split)
               if args.mode is None or r["frame"]["mode"] == args.mode]
    if not records:
        raise SystemExit("no frames match --split / --mode")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(records), size=min(args.count, len(records)), replace=False)
    output = Path(args.output) if args.output else dataset.root / "previews"
    panels = []
    for index in sorted(chosen):
        record = records[index]
        boxes = dataset.boxes(record)
        panel = vision_overlay.label_preview(
            dataset.image(record), dataset.mask(record), [box for _, box in boxes],
            [cls for cls, _ in boxes], f"{record['frame_id']}  {record['frame']['mode']}")
        vision_overlay.save_image(output / f"{record['frame_id']}.jpg", panel)
        panels.append(panel)
    sheet = output / "contact_sheet.jpg"
    vision_overlay.save_image(sheet, vision_overlay.tile(panels[:8], columns=1))
    print(f"{len(panels)} previews in {output}")
    print(f"contact sheet: {sheet}")


if __name__ == "__main__":
    main()
