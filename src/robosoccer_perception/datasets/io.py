"""Reading a generated dataset back, and checking that it is internally consistent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.datasets.generator import (
    HEADER_FILE, MANIFEST_FILE, SCENARIOS_FILE, YOLO_FILE)


class SoccerDataset:
    """A dataset directory written by `generator.generate_dataset`."""

    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / HEADER_FILE).is_file():
            raise FileNotFoundError(f"{self.root} is not a generated dataset (no {HEADER_FILE})")
        self.header: dict[str, Any] = json.loads((self.root / HEADER_FILE).read_text())
        if self.header["format_version"].split(".")[0] != schema.DATASET_FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"dataset format {self.header['format_version']} is not supported "
                             f"(this code reads {schema.DATASET_FORMAT_VERSION})")
        with open(self.root / MANIFEST_FILE, encoding="utf-8") as handle:
            self.records: list[dict[str, Any]] = [json.loads(line) for line in handle if line.strip()]

    @property
    def yolo_yaml(self) -> Path:
        return (self.root / YOLO_FILE).resolve()

    @property
    def splits(self) -> list[str]:
        return [split for split in schema.SPLITS if split in self.header["counts"]]

    def frames(self, split: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records if split is None or r["split"] == split]

    def scenarios(self) -> list[dict[str, Any]]:
        with open(self.root / SCENARIOS_FILE, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def image(self, record: dict[str, Any]) -> np.ndarray:
        """RGB uint8."""
        return load_image(self.root / record["image"])

    def mask(self, record: dict[str, Any]) -> np.ndarray:
        """(H, W) uint8 class ids."""
        mask = cv2.imread(str(self.root / record["mask"]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(self.root / record["mask"])
        return mask

    def boxes(self, record: dict[str, Any]) -> list[tuple[int, tuple[float, ...]]]:
        """``[(class_id, (x0, y0, x1, y1)), ...]`` read from the YOLO label file."""
        return schema.parse_yolo_labels((self.root / record["label"]).read_text(),
                                        record["width"], record["height"])

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)


def load_image(path) -> np.ndarray:
    """Read an image file as RGB uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_dataset(root) -> dict[str, Any]:
    """Check every frame of a dataset; raise `ValueError` listing what is wrong.

    Checks: files exist; masks have the image's size and only known class ids; YOLO labels are
    normalised, name known classes, and match the labelled objects of the manifest one to one;
    boxes lie inside the image; no scenario appears in two splits; no two scenarios share a
    seed. Returns summary statistics.
    """
    dataset = SoccerDataset(root)
    errors: list[str] = []
    scenario_split: dict[str, str] = {}
    boxes_per_class = dict.fromkeys(schema.DETECTION_CLASSES, 0)
    reasons: dict[str, int] = {}
    modes: dict[str, int] = {}
    semantic = np.zeros(len(schema.SEMANTIC_CLASSES), dtype=np.int64)
    for record in dataset:
        tag = record["frame_id"]
        if scenario_split.setdefault(record["scenario_id"], record["split"]) != record["split"]:
            errors.append(f"{tag}: scenario {record['scenario_id']} is in two splits")
        modes[record["frame"]["mode"]] = modes.get(record["frame"]["mode"], 0) + 1
        missing = [key for key in ("image", "label", "mask")
                   if not (dataset.root / record[key]).is_file()]
        if missing:
            errors.append(f"{tag}: missing files {missing}")
            continue
        image, mask = dataset.image(record), dataset.mask(record)
        if image.shape[:2] != (record["height"], record["width"]):
            errors.append(f"{tag}: image is {image.shape[:2]}, manifest says "
                          f"{(record['height'], record['width'])}")
        if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
            errors.append(f"{tag}: mask {mask.shape} {mask.dtype} does not match the image")
        if mask.max() >= len(schema.SEMANTIC_CLASSES):
            errors.append(f"{tag}: mask has unknown class ids {np.unique(mask).tolist()}")
        else:
            semantic += np.bincount(mask.ravel(), minlength=len(schema.SEMANTIC_CLASSES))
        raw = [line.split() for line in (dataset.root / record["label"]).read_text().splitlines()]
        labeled = [obj for obj in record["objects"] if obj["labeled"]]
        for obj in record["objects"]:
            reasons[obj["reason"]] = reasons.get(obj["reason"], 0) + 1
        if len(raw) != len(labeled):
            errors.append(f"{tag}: {len(raw)} label lines but {len(labeled)} labelled objects")
            continue
        for fields, obj in zip(raw, labeled):
            values = [float(v) for v in fields[1:]]
            if len(fields) != 5 or not 0 <= int(fields[0]) < len(schema.DETECTION_CLASSES):
                errors.append(f"{tag}: malformed label line {fields}")
                continue
            if int(fields[0]) != obj["class_id"]:
                errors.append(f"{tag}: label class {fields[0]} != manifest {obj['class_id']}")
            cx, cy, w, h = values
            if not (w > 0 and h > 0 and cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6
                    and cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6):
                errors.append(f"{tag}: box {values} is not inside the image")
            boxes_per_class[obj["class_name"]] += 1
    seeds = [tuple(s["seed_key"]) for s in dataset.scenarios()]
    if len(set(seeds)) != len(seeds):
        errors.append("two scenarios share a seed")
    if errors:
        raise ValueError(f"{len(errors)} problems in {root}:\n  " + "\n  ".join(errors[:30]))
    return {"frames": len(dataset), "scenarios": len(scenario_split),
            "frames_per_split": {s: len(dataset.frames(s)) for s in dataset.splits},
            "boxes_per_class": boxes_per_class, "object_reasons": reasons,
            "frames_per_mode": dict(sorted(modes.items())),
            "semantic_pixel_share": {name: float(semantic[i] / max(semantic.sum(), 1))
                                     for i, name in enumerate(schema.SEMANTIC_CLASSES)}}
