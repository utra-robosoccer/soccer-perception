"""Synthetic damage to vision predictions: missed and spurious features on demand.

`CorruptedVision` wraps a vision pipeline and degrades what it returns. It sees the image and
the model's predictions and nothing else, so what it produces is still image-derived; it
exists to test the localizer against failure modes that the clean simulator rarely shows:

* whole frames lost;
* detections missed, and goal-post boxes that are not there;
* parts of the line mask erased (missed lines), and straight strokes painted into it (false
  lines, such as a white cable or a seam in the carpet).

This is a stress test of the pipeline AFTER the networks. It is not a model of how the
networks fail on real images.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.inference.types import Detection, Segmentation, VisionResult


@dataclasses.dataclass(frozen=True)
class VisionCorruption:
    frame_drop_prob: float = 0.10       # the whole frame yields nothing
    detection_drop_prob: float = 0.5    # each detection is missed
    false_post_rate: float = 0.5        # mean number of invented goal-post boxes per frame
    line_erase_rate: float = 1.0        # mean number of rectangles erased from the line mask
    line_erase_size: tuple[float, float] = (0.15, 0.40)     # of the image width / height
    false_line_rate: float = 1.0        # mean number of straight strokes painted as "line"

    def describe(self) -> dict:
        return dataclasses.asdict(self)


class CorruptedVision:
    """A vision pipeline whose output is degraded. Same interface: ``run(image) -> VisionResult``."""

    def __init__(self, vision, cfg: VisionCorruption, rng: np.random.Generator):
        self.vision, self.cfg, self.rng = vision, cfg, rng

    def run(self, image_rgb: np.ndarray, source: str | None = None) -> VisionResult:
        return corrupt(self.vision.run(image_rgb), self.cfg, self.rng)


def corrupt(result: VisionResult, cfg: VisionCorruption, rng: np.random.Generator) -> VisionResult:
    width, height = result.width, result.height
    if rng.random() < cfg.frame_drop_prob:
        return dataclasses.replace(result, detections=(), segmentation=None)
    detections = [d for d in result.detections if rng.random() >= cfg.detection_drop_prob]
    for _ in range(rng.poisson(cfg.false_post_rate)):
        box_height = rng.uniform(0.15, 0.6) * height
        box_width = rng.uniform(0.02, 0.08) * width
        x0 = rng.uniform(0.0, width - box_width)
        y0 = rng.uniform(0.0, height - box_height)
        detections.append(Detection(schema.GOAL_POST, float(rng.uniform(0.4, 0.9)),
                                    (float(x0), float(y0), float(x0 + box_width),
                                     float(y0 + box_height))))
    segmentation = result.segmentation
    if segmentation is not None:
        probabilities = segmentation.probabilities.copy()
        line, turf = probabilities[schema.FIELD_LINE], probabilities[schema.FIELD]
        for _ in range(rng.poisson(cfg.line_erase_rate)):
            w = int(rng.uniform(*cfg.line_erase_size) * width)
            h = int(rng.uniform(*cfg.line_erase_size) * height)
            x0, y0 = int(rng.uniform(0, width - w)), int(rng.uniform(0, height - h))
            turf[y0:y0 + h, x0:x0 + w] += line[y0:y0 + h, x0:x0 + w]
            line[y0:y0 + h, x0:x0 + w] = 0.0
        strokes = np.zeros((height, width), np.uint8)
        for _ in range(rng.poisson(cfg.false_line_rate)):
            start = np.array([rng.uniform(0, width), rng.uniform(0.3 * height, height)])
            angle = rng.uniform(0.0, np.pi)
            end = start + rng.uniform(0.1, 0.4) * width * np.array([np.cos(angle), np.sin(angle)])
            cv2.line(strokes, tuple(int(v) for v in start), tuple(int(v) for v in end), 1,
                     int(rng.integers(2, 6)))
        painted = strokes > 0
        if painted.any():
            probabilities[:, painted] = 0.05
            line[painted] = 0.9
        segmentation = Segmentation(probabilities.argmax(axis=0).astype(np.uint8), probabilities)
    return dataclasses.replace(result, detections=tuple(detections), segmentation=segmentation)
