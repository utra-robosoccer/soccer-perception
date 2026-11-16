"""What the vision models hand to the rest of the system. See docs/phase3_interfaces.md.

These types are plain dataclasses and NumPy arrays: no ROS, no torch, no model-specific
objects. A detector or segmenter of any kind can fill them, and Phase 3 (geometric feature
extraction, then localization) consumes them without knowing which model produced them.

Nothing here contains ground truth. A `VisionResult` is computed from an image alone.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.datasets import schema

VISION_RESULT_VERSION = "1.0"


@dataclasses.dataclass(frozen=True)
class Detection:
    """One detected object.

    ``bbox_xyxy`` is in pixels of the input image, in pixel-edge coordinates (the image spans
    ``[0, W] x [0, H]``); subtract 0.5 for the pixel-centre convention of `geometry.camera`.
    """
    class_id: int                       # index into datasets.schema.DETECTION_CLASSES
    confidence: float                   # detector score in [0, 1]
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def class_name(self) -> str:
        return schema.DETECTION_CLASSES[self.class_id]

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where a standing object (goal post, robot) meets the ground, if its foot is visible."""
        x0, _, x1, y1 = self.bbox_xyxy
        return (0.5 * (x0 + x1), y1)

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return (x1 - x0, y1 - y0)

    def to_dict(self) -> dict[str, Any]:
        return {"class_id": self.class_id, "class_name": self.class_name,
                "confidence": float(self.confidence),
                "bbox_xyxy": [float(v) for v in self.bbox_xyxy]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return cls(int(data["class_id"]), float(data["confidence"]),
                   tuple(float(v) for v in data["bbox_xyxy"]))


@dataclasses.dataclass(frozen=True)
class Segmentation:
    """Per-pixel classes of one image."""
    mask: np.ndarray                    # (H, W) uint8, index into schema.SEMANTIC_CLASSES
    probabilities: np.ndarray           # (C, H, W) float32, sums to 1 over C

    @property
    def field_mask(self) -> np.ndarray:
        return self.mask == schema.FIELD

    @property
    def line_mask(self) -> np.ndarray:
        return self.mask == schema.FIELD_LINE

    @property
    def line_confidence(self) -> np.ndarray:
        """(H, W) float32 probability that each pixel is line paint."""
        return self.probabilities[schema.FIELD_LINE]

    @property
    def field_confidence(self) -> np.ndarray:
        return self.probabilities[schema.FIELD]


@dataclasses.dataclass(frozen=True)
class VisionResult:
    """Everything the vision pipeline extracted from one image."""
    width: int
    height: int
    detections: tuple[Detection, ...] = ()
    segmentation: Segmentation | None = None
    timing_ms: dict[str, float] = dataclasses.field(default_factory=dict)
    source: str | None = None           # file name or frame id, for bookkeeping only

    def of_class(self, class_name: str) -> tuple[Detection, ...]:
        """Detections of one class, most confident first."""
        class_id = schema.DETECTION_CLASSES.index(class_name)
        return tuple(sorted((d for d in self.detections if d.class_id == class_id),
                            key=lambda d: -d.confidence))

    @property
    def ball(self) -> Detection | None:
        """The most confident ball detection (a match has one ball), or None."""
        balls = self.of_class("ball")
        return balls[0] if balls else None

    @property
    def goal_posts(self) -> tuple[Detection, ...]:
        return self.of_class("goal_post")

    @property
    def robots(self) -> tuple[Detection, ...]:
        return self.of_class("robot")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary. Masks are not embedded; `save` writes them as PNG."""
        data: dict[str, Any] = {
            "vision_result_version": VISION_RESULT_VERSION, "source": self.source,
            "width": self.width, "height": self.height,
            "detection_classes": list(schema.DETECTION_CLASSES),
            "detections": [d.to_dict() for d in self.detections],
            "timing_ms": {key: float(value) for key, value in self.timing_ms.items()},
            "segmentation": None,
        }
        if self.segmentation is not None:
            mask = self.segmentation.mask
            data["segmentation"] = {
                "semantic_classes": list(schema.SEMANTIC_CLASSES),
                "pixel_share": {name: float((mask == i).mean())
                                for i, name in enumerate(schema.SEMANTIC_CLASSES)},
                "mean_line_confidence_on_line_pixels": float(
                    self.segmentation.line_confidence[mask == schema.FIELD_LINE].mean())
                if (mask == schema.FIELD_LINE).any() else None,
            }
        return data

    def save(self, directory, stem: str) -> dict[str, Path]:
        """Write ``<stem>.json`` and, with a segmentation, ``<stem>_mask.png`` (class ids) and
        ``<stem>_line_confidence.png`` (0-255)."""
        import cv2

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"json": directory / f"{stem}.json"}
        data = self.to_dict()
        if self.segmentation is not None:
            paths["mask"] = directory / f"{stem}_mask.png"
            paths["line_confidence"] = directory / f"{stem}_line_confidence.png"
            cv2.imwrite(str(paths["mask"]), self.segmentation.mask)
            cv2.imwrite(str(paths["line_confidence"]),
                        np.rint(self.segmentation.line_confidence * 255).astype(np.uint8))
            data["segmentation"]["mask_file"] = paths["mask"].name
            data["segmentation"]["line_confidence_file"] = paths["line_confidence"].name
        paths["json"].write_text(json.dumps(data, indent=2))
        return paths
