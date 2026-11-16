"""Drawing boxes and segmentation masks on camera images (labels or predictions).

Plain OpenCV drawing on RGB uint8 arrays; used for dataset previews, evaluation examples and
the output of `scripts/run_vision.py`. Colour follows the class everywhere and every box also
carries its class name, so colour is never the only cue.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from robosoccer_perception.datasets import schema

# RGB. Chosen to stay apart from turf green and line white and from each other.
DETECTION_COLOURS = {schema.BALL: (255, 122, 26), schema.GOAL_POST: (255, 214, 10),
                     schema.ROBOT: (64, 156, 255)}
LINE_COLOUR = (255, 45, 210)
FIELD_COLOUR = (40, 200, 230)
GROUND_TRUTH_COLOUR = (255, 255, 255)
ERROR_COLOUR = (255, 40, 40)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, line_alpha: float = 0.75,
                 field_alpha: float = 0.22) -> np.ndarray:
    """Tint ``field`` pixels lightly and ``field_line`` pixels strongly; dim the background."""
    out = image_rgb.astype(np.float32)
    background = mask == schema.BACKGROUND
    out[background] *= 0.55
    for class_id, colour, alpha in ((schema.FIELD, FIELD_COLOUR, field_alpha),
                                    (schema.FIELD_LINE, LINE_COLOUR, line_alpha)):
        selected = mask == class_id
        out[selected] = (1 - alpha) * out[selected] + alpha * np.asarray(colour, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes(image_rgb: np.ndarray, boxes: Iterable[Sequence[float]], class_ids: Iterable[int],
               scores: Iterable[float] | None = None, colour=None, thickness: int = 2,
               tags: Iterable[str] | None = None, dashed: bool = False) -> np.ndarray:
    """Draw boxes given in pixel-edge ``x0, y0, x1, y1``. Returns a new image.

    ``colour`` overrides the per-class colour (for example white for ground truth);
    ``tags`` replaces the default ``name score`` caption.
    """
    out = image_rgb.copy()
    boxes, class_ids = list(boxes), list(class_ids)
    scores = [None] * len(boxes) if scores is None else list(scores)
    tags = [None] * len(boxes) if tags is None else list(tags)
    for box, class_id, score, tag in zip(boxes, class_ids, scores, tags):
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x1, y1 = max(x1 - 1, x0), max(y1 - 1, y0)          # edge coordinates -> last pixel
        rgb = colour or DETECTION_COLOURS[int(class_id)]
        if dashed:
            _dashed_rectangle(out, (x0, y0), (x1, y1), rgb, thickness)
        else:
            cv2.rectangle(out, (x0, y0), (x1, y1), rgb, thickness, cv2.LINE_AA)
        caption = tag if tag is not None else schema.DETECTION_CLASSES[int(class_id)] + (
            "" if score is None else f" {score:.2f}")
        if caption:
            _caption(out, caption, x0, y0, rgb)
    return out


def _caption(image: np.ndarray, text: str, x: int, y: int, rgb) -> None:
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.4
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    top = y - height - baseline - 2
    if top < 0:
        top = y + 2
    x = int(np.clip(x, 0, max(0, image.shape[1] - width - 4)))
    cv2.rectangle(image, (x, top), (x + width + 4, top + height + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, top + height + 1), font, scale, rgb, 1, cv2.LINE_AA)


def _dashed_rectangle(image: np.ndarray, p0, p1, rgb, thickness: int, dash: int = 5) -> None:
    (x0, y0), (x1, y1) = p0, p1
    for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                       ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = int(np.hypot(end[0] - start[0], end[1] - start[1]))
        for offset in range(0, max(length, 1), 2 * dash):
            a = offset / max(length, 1)
            b = min(offset + dash, length) / max(length, 1)
            cv2.line(image,
                     (int(start[0] + a * (end[0] - start[0])), int(start[1] + a * (end[1] - start[1]))),
                     (int(start[0] + b * (end[0] - start[0])), int(start[1] + b * (end[1] - start[1]))),
                     rgb, thickness, cv2.LINE_AA)


def label_preview(image_rgb: np.ndarray, mask: np.ndarray, boxes, class_ids,
                  caption: str = "") -> np.ndarray:
    """The image, and next to it the image with its mask and boxes drawn on."""
    annotated = draw_boxes(overlay_mask(image_rgb, mask), boxes, class_ids)
    panel = np.concatenate([image_rgb, annotated], axis=1)
    if caption:
        _caption(panel, caption, 4, panel.shape[0] - 4, (255, 255, 255))
    return panel


def tile(images: Sequence[np.ndarray], columns: int = 2, gap: int = 4) -> np.ndarray:
    """Arrange equally sized images in a grid."""
    height, width = images[0].shape[:2]
    rows = -(-len(images) // columns)
    canvas = np.full((rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3),
                     20, np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y, x = row * (height + gap), column * (width + gap)
        canvas[y:y + height, x:x + width] = image
    return canvas


def save_image(path, image_rgb: np.ndarray) -> None:
    """Write an RGB image; the format follows the file extension."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {path}")
