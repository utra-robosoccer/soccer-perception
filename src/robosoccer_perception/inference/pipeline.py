"""The offline vision pipeline: one RGB image in, one `VisionResult` out.

    image (H, W, 3 uint8 RGB)
        ├─ detector   -> ball / goal_post / robot boxes with confidences
        └─ segmenter  -> background / field / field_line mask with per-pixel probabilities

The two models are independent and each is optional, so either can be developed, replaced or
benchmarked alone. A detector is any object with ``predict(image_rgb) -> list[Detection]``; a
segmenter any object with ``predict(image_rgb) -> Segmentation``. No ROS, no ZED SDK, and no
simulator: the pipeline never sees ground truth.

What this does NOT do (Phase 3): convert boxes and masks into robot-frame observations, or feed
the particle filter. See docs/phase3_interfaces.md.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from robosoccer_perception.inference import profiling
from robosoccer_perception.inference.types import VisionResult


class VisionPipeline:
    def __init__(self, detector=None, segmenter=None):
        if detector is None and segmenter is None:
            raise ValueError("give the pipeline a detector, a segmenter, or both")
        self.detector, self.segmenter = detector, segmenter
        self.device = getattr(detector, "device", None) or getattr(segmenter, "device", "cpu")

    @classmethod
    def from_checkpoints(cls, detector_weights=None, segmenter_checkpoint=None,
                         device: str = "auto", conf: float = 0.25, imgsz: int = 640,
                         half: bool = False) -> "VisionPipeline":
        """Load the project's baseline models (YOLO detector, U-Net segmenter)."""
        detector = segmenter = None
        if detector_weights is not None:
            from robosoccer_perception.detection.yolo import YoloDetector
            detector = YoloDetector(detector_weights, device=device, conf=conf, imgsz=imgsz,
                                    half=half)
        if segmenter_checkpoint is not None:
            from robosoccer_perception.segmentation.segmenter import Segmenter
            segmenter = Segmenter(segmenter_checkpoint, device=device, half=half)
        return cls(detector, segmenter)

    def run(self, image_rgb: np.ndarray, source: str | None = None) -> VisionResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("expected an (H, W, 3) uint8 RGB image")
        timing: dict[str, float] = {}
        profiling.synchronize(self.device)
        started = time.perf_counter()
        detections: tuple = ()
        if self.detector is not None:
            detections = tuple(self.detector.predict(image_rgb))
            profiling.synchronize(self.device)
            timing["detector"] = (time.perf_counter() - started) * 1000.0
        segmentation = None
        if self.segmenter is not None:
            segment_started = time.perf_counter()
            segmentation = self.segmenter.predict(image_rgb)
            profiling.synchronize(self.device)
            timing["segmenter"] = (time.perf_counter() - segment_started) * 1000.0
        timing["total"] = (time.perf_counter() - started) * 1000.0
        return VisionResult(width=image_rgb.shape[1], height=image_rgb.shape[0],
                            detections=detections, segmentation=segmentation,
                            timing_ms=timing, source=source)

    def warmup(self, image_rgb: np.ndarray, runs: int = 3) -> None:
        """Run a few times so that lazy initialisation does not land in the first timing."""
        for _ in range(runs):
            self.run(image_rgb)

    def describe(self) -> dict[str, Any]:
        return {"detector": getattr(self.detector, "weights", None) and {
                    "weights": self.detector.weights, "conf": self.detector.conf,
                    "imgsz": self.detector.imgsz},
                "segmenter": getattr(self.segmenter, "checkpoint", None) and {
                    "checkpoint": self.segmenter.checkpoint,
                    "train_width": self.segmenter.train_width},
                "device": self.device}


def render_overlay(image_rgb: np.ndarray, result: VisionResult) -> np.ndarray:
    """The image with the predicted mask and boxes drawn on it."""
    from robosoccer_perception.visualization import vision_overlay as vo

    out = image_rgb if result.segmentation is None else vo.overlay_mask(image_rgb,
                                                                       result.segmentation.mask)
    return vo.draw_boxes(out, [d.bbox_xyxy for d in result.detections],
                         [d.class_id for d in result.detections],
                         [d.confidence for d in result.detections])
