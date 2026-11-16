"""Pixel-wise segmentation metrics in plain NumPy (needs OpenCV only for the tolerant scores).

Overall pixel accuracy is reported but should not be trusted on its own: lines are about 3 %
of the pixels, so a model that never predicts a line still scores about 97 %. The numbers to
look at are the ``field_line`` IoU, precision, recall and F1.

Two flavours of line scores:

* EXACT: a predicted line pixel is right only where the label says line.
* TOLERANT (within ``tolerance`` px): a predicted line pixel is right if a labelled line pixel
  lies within ``tolerance`` pixels of it, and a labelled line pixel is found if a predicted one
  lies within ``tolerance`` pixels. This forgives one-pixel disagreements about where a line's
  edge is, which dominate the exact scores of thin lines and matter little for localization.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from robosoccer_perception.datasets import schema


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int,
                     ignore_index: int = schema.IGNORE_INDEX) -> np.ndarray:
    """``matrix[p, t]`` = number of pixels predicted ``p`` whose label is ``t``."""
    valid = target != ignore_index
    index = prediction[valid].astype(np.int64) * num_classes + target[valid].astype(np.int64)
    return np.bincount(index, minlength=num_classes * num_classes).reshape(num_classes,
                                                                           num_classes)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def scores_from_confusion(matrix: np.ndarray, class_names=schema.SEMANTIC_CLASSES) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    predicted, actual = matrix.sum(axis=1), matrix.sum(axis=0)
    per_class = {}
    for index, name in enumerate(class_names):
        precision = _ratio(true_positive[index], predicted[index])
        recall = _ratio(true_positive[index], actual[index])
        per_class[name] = {
            "iou": _ratio(true_positive[index], predicted[index] + actual[index] - true_positive[index]),
            "precision": precision, "recall": recall,
            "f1": _ratio(2 * true_positive[index], predicted[index] + actual[index]),
            "pixels": int(actual[index])}
    ious = [entry["iou"] for entry in per_class.values() if entry["iou"] is not None]
    return {"mean_iou": float(np.mean(ious)) if ious else None,
            "pixel_accuracy": _ratio(true_positive.sum(), matrix.sum()),
            "per_class": per_class}


def tolerant_line_counts(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2,
                         line_class: int = schema.FIELD_LINE) -> dict[str, int]:
    """Counts behind the tolerant line precision and recall of one image."""
    import cv2

    predicted, actual = prediction == line_class, target == line_class
    size = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    near_actual = cv2.dilate(actual.astype(np.uint8), kernel).astype(bool)
    near_predicted = cv2.dilate(predicted.astype(np.uint8), kernel).astype(bool)
    return {"predicted": int(predicted.sum()), "predicted_near_label": int((predicted & near_actual).sum()),
            "labelled": int(actual.sum()), "labelled_near_prediction": int((actual & near_predicted).sum())}


def tolerant_line_scores(counts: dict[str, int]) -> dict[str, float | None]:
    precision = _ratio(counts["predicted_near_label"], counts["predicted"])
    recall = _ratio(counts["labelled_near_prediction"], counts["labelled"])
    f1 = None if precision is None or recall is None or precision + recall == 0 \
        else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


class SegmentationMeter:
    """Accumulates confusion and tolerant line counts over images."""

    def __init__(self, num_classes: int = len(schema.SEMANTIC_CLASSES), tolerance: int = 2):
        self.num_classes, self.tolerance = num_classes, tolerance
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.tolerant = dict.fromkeys(("predicted", "predicted_near_label", "labelled",
                                       "labelled_near_prediction"), 0)
        self.images = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        self.matrix += confusion_matrix(prediction, target, self.num_classes)
        for key, value in tolerant_line_counts(prediction, target, self.tolerance).items():
            self.tolerant[key] += value
        self.images += 1

    def compute(self) -> dict[str, Any]:
        result = scores_from_confusion(self.matrix)
        line = result["per_class"][schema.SEMANTIC_CLASSES[schema.FIELD_LINE]]
        result["line"] = {"iou": line["iou"], "precision": line["precision"],
                          "recall": line["recall"], "f1": line["f1"]}
        result["line_tolerant"] = {"tolerance_px": self.tolerance,
                                   **tolerant_line_scores(self.tolerant)}
        result["num_images"] = self.images
        result["confusion_matrix"] = {"rows_predicted": list(schema.SEMANTIC_CLASSES),
                                      "columns_true": list(schema.SEMANTIC_CLASSES),
                                      "counts": self.matrix.tolist()}
        return result
