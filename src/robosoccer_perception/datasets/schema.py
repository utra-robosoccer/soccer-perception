"""Label schema of the synthetic vision datasets: class ids, label policy, record layout.

These ids are used unchanged by the generator, the training code, the evaluation code and the
inference pipeline. Change them here and nowhere else. See docs/synthetic_data.md.
"""
from __future__ import annotations

import dataclasses
from typing import Any

DATASET_FORMAT_VERSION = "1.0"

# Object detection: class id = index. Written to the YOLO label files.
DETECTION_CLASSES: tuple[str, ...] = ("ball", "goal_post", "robot")
BALL, GOAL_POST, ROBOT = range(3)

# Semantic segmentation: class id = index = pixel value of the mask PNG.
SEMANTIC_CLASSES: tuple[str, ...] = ("background", "field", "field_line")
BACKGROUND, FIELD, FIELD_LINE = range(3)
IGNORE_INDEX = 255          # never written by the generator; accepted by the training loss

SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclasses.dataclass(frozen=True)
class LabelPolicy:
    """Rules that turn rendered geometry into labels. Recorded in every dataset manifest."""
    supersample: int = 2
    """Segmentation is rendered on a grid this many times finer than the image, so that box
    edges are known to 1/supersample px and thin lines are labelled by coverage."""
    line_coverage_threshold: float = 0.5
    """A pixel is ``field_line`` when at least this share of it shows line paint."""
    min_visible_pixels: float = 6.0
    """Objects showing fewer image pixels than this get no box (reason ``too_small``)."""
    min_box_side_px: float = 2.0
    """... and so do objects whose visible box is narrower or lower than this."""
    min_visible_fraction: float = 0.15
    """Objects with less than this share of their in-image silhouette un-occluded get no box
    (reason ``occluded``)."""

    def __post_init__(self) -> None:
        if self.supersample < 1:
            raise ValueError("supersample must be >= 1")
        if not 0.0 < self.line_coverage_threshold <= 1.0:
            raise ValueError("line_coverage_threshold must be in (0, 1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LabelPolicy":
        return cls(**(data or {}))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ObjectAnnotation:
    """One object of a detection class in one frame, whether or not it received a box.

    ``bbox_xyxy`` is the tight box around the object's VISIBLE pixels, in pixel-edge
    coordinates: the image spans x in [0, width] and y in [0, height], and pixel ``(v, u)``
    covers ``[u, u + 1] x [v, v + 1]``. Subtract 0.5 to get the pixel-centre convention of
    `geometry.camera`.
    """
    instance: str                       # "ball", "goal_pos_x/post_neg_y", "robot1"
    class_id: int
    labeled: bool                       # True = has a line in the YOLO label file
    reason: str                         # ok | out_of_view | occluded | too_small
    bbox_xyxy: tuple[float, float, float, float] | None
    visible_pixels: float               # image pixels (area) showing the object
    amodal_pixels: float                # the same if nothing stood in front of it
    visible_fraction: float             # visible / amodal; 0 when out of view
    truncated: bool                     # the un-occluded silhouette touches the image border
    distance_m: float                   # horizontal distance camera -> object's ground point
    position_field: tuple[float, float, float]
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def class_name(self) -> str:
        return DETECTION_CLASSES[self.class_id]

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["class_name"] = self.class_name
        return data


def yolo_line(annotation: ObjectAnnotation, width: int, height: int) -> str:
    """``class cx cy w h``, normalised to [0, 1], as Ultralytics expects."""
    x0, y0, x1, y1 = annotation.bbox_xyxy
    return (f"{annotation.class_id} {(x0 + x1) / (2 * width):.6f} {(y0 + y1) / (2 * height):.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")


def parse_yolo_labels(text: str, width: int, height: int) -> list[tuple[int, tuple[float, ...]]]:
    """Inverse of `yolo_line` for a whole label file: ``[(class_id, (x0, y0, x1, y1)), ...]``."""
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append((int(cls), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes
