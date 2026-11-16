"""Detection metrics in plain NumPy: IoU, matching, average precision, confusion matrix.

Independent of any detection framework, so the same numbers can be computed for any detector
(and tested without one). Boxes are ``x0, y0, x1, y1``.

Definitions
-----------
* A prediction MATCHES a ground-truth box of the same class when their IoU reaches the
  threshold; predictions are assigned greedily in order of decreasing score and each
  ground-truth box is used once.
* IGNORE boxes are objects that are in the image but deliberately unlabelled (too small or too
  occluded for a label). A prediction that matches no labelled box but overlaps an ignore box
  of its class is dropped: it is neither a true nor a false positive.
* AP is the area under the precision-recall curve after making precision monotone, sampled at
  101 recall points (the COCO convention). mAP50-95 averages AP over IoU 0.50, 0.55, ..., 0.95.
  A perfect detector scores exactly 1. (Ultralytics interpolates the curve slightly differently
  and tops out near 0.995, so its numbers can differ from these in the third decimal.)
"""
from __future__ import annotations

import dataclasses

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a, b) -> np.ndarray:
    """Pairwise IoU of boxes ``a`` (N, 4) and ``b`` (M, 4): an (N, M) array."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.clip(right_bottom - left_top, 0.0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)


@dataclasses.dataclass(frozen=True)
class FrameBoxes:
    """Predictions and ground truth of one image."""
    pred_boxes: np.ndarray              # (P, 4)
    pred_classes: np.ndarray            # (P,)
    pred_scores: np.ndarray             # (P,)
    gt_boxes: np.ndarray                # (G, 4) labelled objects
    gt_classes: np.ndarray              # (G,)
    ignore_boxes: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 4)))
    ignore_classes: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, int))

    @classmethod
    def build(cls, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes,
              ignore_boxes=(), ignore_classes=()) -> "FrameBoxes":
        return cls(np.asarray(pred_boxes, float).reshape(-1, 4),
                   np.asarray(pred_classes, int).reshape(-1),
                   np.asarray(pred_scores, float).reshape(-1),
                   np.asarray(gt_boxes, float).reshape(-1, 4),
                   np.asarray(gt_classes, int).reshape(-1),
                   np.asarray(ignore_boxes, float).reshape(-1, 4),
                   np.asarray(ignore_classes, int).reshape(-1))


@dataclasses.dataclass(frozen=True)
class FrameMatch:
    pred_gt: np.ndarray                 # (P,) index of the matched ground-truth box, or -1
    pred_ignored: np.ndarray            # (P,) bool: unmatched but explained by an ignore box
    gt_pred: np.ndarray                 # (G,) index of the matching prediction, or -1


def match_frame(frame: FrameBoxes, iou_threshold: float = 0.5,
                min_score: float = 0.0) -> FrameMatch:
    """Greedy same-class matching of one image's predictions to its ground truth."""
    num_pred, num_gt = len(frame.pred_boxes), len(frame.gt_boxes)
    pred_gt = np.full(num_pred, -1, dtype=int)
    gt_pred = np.full(num_gt, -1, dtype=int)
    ignored = np.zeros(num_pred, dtype=bool)
    if num_pred == 0:
        return FrameMatch(pred_gt, ignored, gt_pred)
    iou = box_iou(frame.pred_boxes, frame.gt_boxes)
    iou_ignore = box_iou(frame.pred_boxes, frame.ignore_boxes)
    for p in np.argsort(-frame.pred_scores, kind="stable"):
        if frame.pred_scores[p] < min_score:
            continue
        if num_gt:
            candidates = np.where((frame.gt_classes == frame.pred_classes[p]) & (gt_pred < 0),
                                  iou[p], -1.0)
            best = int(np.argmax(candidates))
            if candidates[best] >= iou_threshold:
                pred_gt[p], gt_pred[best] = best, p
                continue
        if len(frame.ignore_boxes):
            same = frame.ignore_classes == frame.pred_classes[p]
            ignored[p] = bool(np.any(iou_ignore[p][same] >= min(iou_threshold, 0.3)))
    return FrameMatch(pred_gt, ignored, gt_pred)


def average_precision(scores, is_true_positive, num_gt: int) -> float:
    """AP of one class at one IoU threshold from every prediction's score and outcome."""
    if num_gt == 0:
        return float("nan")
    scores = np.asarray(scores, dtype=float)
    hits = np.asarray(is_true_positive, dtype=bool)[np.argsort(-scores, kind="stable")]
    if hits.size == 0:
        return 0.0
    true_positives = np.cumsum(hits)
    recall = true_positives / num_gt
    precision = true_positives / np.arange(1, hits.size + 1)
    # Precision at recall r = the best precision reached at any recall >= r; 0 beyond the
    # highest recall the detector reaches.
    envelope = np.flip(np.maximum.accumulate(np.flip(precision)))
    index = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
    sampled = np.where(index < hits.size, envelope[np.minimum(index, hits.size - 1)], 0.0)
    return float(np.mean(sampled))


def evaluate_frames(frames: list[FrameBoxes], class_names, conf_threshold: float = 0.25,
                    iou_threshold: float = 0.5) -> dict:
    """Dataset-level metrics.

    AP uses every prediction (all scores). Precision, recall, F1, the confusion matrix and the
    false-positive counts use only predictions scoring at least ``conf_threshold``: they
    describe the detector at the operating point it would run at.
    """
    num_classes = len(class_names)
    per_class_ap = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan)
    num_gt = np.array([sum(int((f.gt_classes == c).sum()) for f in frames)
                       for c in range(num_classes)])
    for t, threshold in enumerate(IOU_THRESHOLDS):
        scores = [[] for _ in range(num_classes)]
        hits = [[] for _ in range(num_classes)]
        for frame in frames:
            match = match_frame(frame, threshold)
            for p in range(len(frame.pred_boxes)):
                if match.pred_ignored[p]:
                    continue
                scores[frame.pred_classes[p]].append(frame.pred_scores[p])
                hits[frame.pred_classes[p]].append(match.pred_gt[p] >= 0)
        for c in range(num_classes):
            per_class_ap[c, t] = average_precision(scores[c], hits[c], int(num_gt[c]))

    true_pos, false_pos = np.zeros(num_classes, int), np.zeros(num_classes, int)
    frames_with_fp = 0
    # Rows = predicted class (last row: nothing predicted), columns = true class (last column:
    # background). Class-agnostic matching, so a ball called "robot" shows up off the diagonal.
    confusion = np.zeros((num_classes + 1, num_classes + 1), int)
    for frame in frames:
        match = match_frame(frame, iou_threshold, conf_threshold)
        keep = (frame.pred_scores >= conf_threshold) & ~match.pred_ignored
        frame_fp = 0
        for p in np.flatnonzero(keep):
            if match.pred_gt[p] >= 0:
                true_pos[frame.pred_classes[p]] += 1
            else:
                false_pos[frame.pred_classes[p]] += 1
                frame_fp += 1
        frames_with_fp += frame_fp > 0
        agnostic = _match_any_class(frame, keep, iou_threshold)
        for p in np.flatnonzero(keep):
            column = frame.gt_classes[agnostic[p]] if agnostic[p] >= 0 else num_classes
            confusion[frame.pred_classes[p], column] += 1
        for g in set(range(len(frame.gt_boxes))) - set(agnostic[agnostic >= 0].tolist()):
            confusion[num_classes, frame.gt_classes[g]] += 1

    def ratio(a, b):
        return float(a / b) if b else float("nan")

    per_class = {}
    for c, name in enumerate(class_names):
        precision, recall = ratio(true_pos[c], true_pos[c] + false_pos[c]), ratio(true_pos[c], num_gt[c])
        per_class[name] = {
            "num_gt": int(num_gt[c]), "ap50": float(per_class_ap[c, 0]),
            "ap50_95": float(np.mean(per_class_ap[c])), "precision": precision, "recall": recall,
            "f1": ratio(2 * precision * recall, precision + recall)
            if precision == precision and recall == recall else float("nan"),
            "true_positives": int(true_pos[c]), "false_positives": int(false_pos[c]),
            "false_negatives": int(num_gt[c] - true_pos[c]),
            "false_positives_per_image": ratio(false_pos[c], len(frames))}
    present = num_gt > 0
    return {
        "num_images": len(frames), "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "map50": float(np.mean(per_class_ap[present, 0])) if present.any() else float("nan"),
        "map50_95": float(np.mean(per_class_ap[present])) if present.any() else float("nan"),
        "precision": ratio(true_pos.sum(), true_pos.sum() + false_pos.sum()),
        "recall": ratio(true_pos.sum(), num_gt.sum()),
        "false_positives_per_image": ratio(false_pos.sum(), len(frames)),
        "images_with_false_positive": ratio(frames_with_fp, len(frames)),
        "per_class": per_class,
        "confusion_matrix": {"rows_predicted": [*class_names, "(missed)"],
                             "columns_true": [*class_names, "(background)"],
                             "counts": confusion.tolist()},
    }


def _match_any_class(frame: FrameBoxes, keep: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy matching that ignores classes: which ground-truth box each kept prediction covers."""
    result = np.full(len(frame.pred_boxes), -1, dtype=int)
    if not len(frame.gt_boxes) or not keep.any():
        return result
    iou = box_iou(frame.pred_boxes, frame.gt_boxes)
    taken = np.zeros(len(frame.gt_boxes), bool)
    for p in np.argsort(-frame.pred_scores, kind="stable"):
        if not keep[p]:
            continue
        candidates = np.where(~taken, iou[p], -1.0)
        best = int(np.argmax(candidates))
        if candidates[best] >= iou_threshold:
            result[p], taken[best] = best, True
    return result
