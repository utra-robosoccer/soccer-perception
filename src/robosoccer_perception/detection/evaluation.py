"""Evaluating a detector on a generated dataset, beyond a single mAP number.

`evaluate_detector` runs a detector over one split and reports:

* mAP50, mAP50-95, per-class AP, precision, recall, confusion matrix (`detection.metrics`);
* recall broken down by what makes detection hard: distance, apparent size in pixels,
  occlusion, clipping by the image border, robot posture and size, and frame layout mode;
* false positives: per image, per class, and what they sit on (turf, line or background);
* inference latency and memory;
* example images of correct detections, false positives and false negatives.

The detector sees images only. Ground truth (labels and the manifest's object records) is used
after inference, to score it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.datasets.io import SoccerDataset
from robosoccer_perception.detection import metrics
from robosoccer_perception.detection.metrics import FrameBoxes
from robosoccer_perception.inference import profiling
from robosoccer_perception.inference.types import Detection

# (label, lower bound inclusive, upper bound exclusive)
DISTANCE_BINS = {
    "ball": (("near (<2 m)", 0.0, 2.0), ("medium (2-5 m)", 2.0, 5.0), ("far (>5 m)", 5.0, np.inf)),
    "goal_post": (("near (<3 m)", 0.0, 3.0), ("medium (3-6 m)", 3.0, 6.0),
                  ("far (>6 m)", 6.0, np.inf)),
    "robot": (("near (<2 m)", 0.0, 2.0), ("medium (2-5 m)", 2.0, 5.0), ("far (>5 m)", 5.0, np.inf)),
}
SIZE_BINS = (("<8 px", 0.0, 8.0), ("8-16 px", 8.0, 16.0), ("16-32 px", 16.0, 32.0),
             ("32-96 px", 32.0, 96.0), (">=96 px", 96.0, np.inf))
VISIBILITY_BINS = (("heavily occluded (<50 % visible)", 0.0, 0.5),
                   ("partly occluded (50-95 %)", 0.5, 0.95),
                   ("unoccluded (>=95 %)", 0.95, np.inf))
ROBOT_HEIGHT_BINS = (("small (<0.7 m)", 0.0, 0.7), ("medium (0.7-1.0 m)", 0.7, 1.0),
                     ("tall (>=1.0 m)", 1.0, np.inf))


def _bin(value: float, bins) -> str:
    for label, low, high in bins:
        if low <= value < high:
            return label
    return bins[-1][0]


def frame_ground_truth(record: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Split a manifest record's objects into labelled ones and ignore regions.

    Ignore regions are objects that are (partly) in the image but got no label because they
    are too small or too occluded; detecting one is not an error.
    """
    robots = record["frame"]["robots"]
    labelled, ignore = [], []
    for obj in record["objects"]:
        if obj["bbox_xyxy"] is None:
            continue
        obj = dict(obj)
        if obj["class_id"] == schema.ROBOT:
            state = robots[obj["attributes"]["appearance_index"]]
            obj["fallen"] = bool(state and state["fallen"])
        (labelled if obj["labeled"] else ignore).append(obj)
    return labelled, ignore


def to_frame_boxes(detections: list[Detection], record: dict[str, Any]) -> tuple[FrameBoxes, list[dict]]:
    labelled, ignore = frame_ground_truth(record)
    return FrameBoxes.build(
        [d.bbox_xyxy for d in detections], [d.class_id for d in detections],
        [d.confidence for d in detections],
        [o["bbox_xyxy"] for o in labelled], [o["class_id"] for o in labelled],
        [o["bbox_xyxy"] for o in ignore], [o["class_id"] for o in ignore]), labelled


class _Tally:
    """Detected / total counts per (group, bin)."""

    def __init__(self):
        self.counts: dict[tuple[str, str, str], list[int]] = {}

    def add(self, class_name: str, group: str, label: str, detected: bool) -> None:
        entry = self.counts.setdefault((class_name, group, label), [0, 0])
        entry[0] += int(detected)
        entry[1] += 1

    def rows(self) -> list[dict[str, Any]]:
        return [{"class": c, "breakdown": g, "bin": b, "detected": d, "total": n,
                 "recall": d / n if n else float("nan")}
                for (c, g, b), (d, n) in self.counts.items()]


def stratified_recall(frames: list[FrameBoxes], objects: list[list[dict]],
                      records: list[dict[str, Any]], conf_threshold: float,
                      iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    """Recall of labelled objects at the operating point, per class and difficulty bin."""
    tally = _Tally()
    for frame, labelled, record in zip(frames, objects, records):
        match = metrics.match_frame(frame, iou_threshold, conf_threshold)
        for index, obj in enumerate(labelled):
            name, hit = obj["class_name"], bool(match.gt_pred[index] >= 0)
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            tally.add(name, "all", "all", hit)
            tally.add(name, "distance", _bin(obj["distance_m"], DISTANCE_BINS[name]), hit)
            tally.add(name, "apparent size (sqrt of box area)",
                      _bin(float(np.sqrt((x1 - x0) * (y1 - y0))), SIZE_BINS), hit)
            tally.add(name, "occlusion", _bin(obj["visible_fraction"], VISIBILITY_BINS), hit)
            tally.add(name, "image border",
                      "cut by the border" if obj["truncated"] else "fully inside", hit)
            tally.add(name, "frame mode", record["frame"]["mode"], hit)
            if name == "robot":
                tally.add(name, "posture", "fallen" if obj["fallen"] else "upright", hit)
                tally.add(name, "robot height",
                          _bin(obj["attributes"]["height"], ROBOT_HEIGHT_BINS), hit)
    order = {name: i for i, name in enumerate(schema.DETECTION_CLASSES)}
    groups = ["all", "distance", "apparent size (sqrt of box area)", "occlusion", "image border",
              "posture", "robot height", "frame mode"]
    labels = [label for bins in (*DISTANCE_BINS.values(), SIZE_BINS, VISIBILITY_BINS,
                                 ROBOT_HEIGHT_BINS) for label, _, _ in bins]
    return sorted(tally.rows(), key=lambda r: (
        order[r["class"]], groups.index(r["breakdown"]),
        labels.index(r["bin"]) if r["bin"] in labels else 0, r["bin"]))


def false_positive_analysis(frames: list[FrameBoxes], records: list[dict[str, Any]],
                            masks: Callable[[dict], np.ndarray], conf_threshold: float,
                            iou_threshold: float = 0.5) -> dict[str, Any]:
    """Where false positives come from. Each one is put in exactly one category:

    * ``badly_placed``   it overlaps (IoU >= 0.1) an object of its own class: a real object was
                         found but the box is too loose, or it is a second box on the same object;
    * ``other_object``   it overlaps an object of another class: a class confusion;
    * ``nothing_there``  it overlaps no object at all. For these, ``nothing_there_centre_on``
                         says which semantic class lies under the box centre: a goal-post false
                         positive centred on ``background`` is the backdrop being taken for a
                         post, one centred on ``field_line`` is paint being taken for a post.

    Also reported: false positives on images that contain no object of the class at all.
    """
    names = schema.DETECTION_CLASSES
    result = {name: {"false_positives": 0, "badly_placed": 0, "other_object": 0,
                     "nothing_there": 0,
                     "nothing_there_centre_on": dict.fromkeys(schema.SEMANTIC_CLASSES, 0),
                     "images_without_this_class": 0,
                     "false_positives_on_images_without_this_class": 0} for name in names}
    for frame, record in zip(frames, records):
        match = metrics.match_frame(frame, iou_threshold, conf_threshold)
        wrong = np.flatnonzero((frame.pred_scores >= conf_threshold) & (match.pred_gt < 0)
                               & ~match.pred_ignored)
        present = {obj["class_id"] for obj in record["objects"] if obj["bbox_xyxy"] is not None}
        for class_id, name in enumerate(names):
            if class_id not in present:
                result[name]["images_without_this_class"] += 1
        if not len(wrong):
            continue
        object_boxes = np.concatenate([frame.gt_boxes, frame.ignore_boxes])
        object_classes = np.concatenate([frame.gt_classes, frame.ignore_classes])
        overlap = metrics.box_iou(frame.pred_boxes[wrong], object_boxes) >= 0.1
        mask = None
        for row, p in enumerate(wrong):
            entry = result[names[frame.pred_classes[p]]]
            entry["false_positives"] += 1
            if frame.pred_classes[p] not in present:
                entry["false_positives_on_images_without_this_class"] += 1
            touched = object_classes[overlap[row]]
            if (touched == frame.pred_classes[p]).any():
                entry["badly_placed"] += 1
            elif len(touched):
                entry["other_object"] += 1
            else:
                entry["nothing_there"] += 1
                mask = masks(record) if mask is None else mask
                x0, y0, x1, y1 = frame.pred_boxes[p]
                v = int(np.clip((y0 + y1) / 2, 0, mask.shape[0] - 1))
                u = int(np.clip((x0 + x1) / 2, 0, mask.shape[1] - 1))
                entry["nothing_there_centre_on"][schema.SEMANTIC_CLASSES[int(mask[v, u])]] += 1
    for entry in result.values():
        entry["false_positives_per_image"] = entry["false_positives"] / max(len(frames), 1)
        empty = entry["images_without_this_class"]
        entry["false_positives_per_image_without_this_class"] = (
            entry["false_positives_on_images_without_this_class"] / empty if empty else None)
    return result


def evaluate_detector(detector, dataset_root, split: str = "test", conf_threshold: float = 0.25,
                      iou_threshold: float = 0.5, output_dir=None, max_examples: int = 8,
                      latency_repeats: int = 100, limit: int | None = None) -> dict[str, Any]:
    """Run ``detector`` over a split and compute every metric above.

    ``detector`` is anything with ``predict(image_rgb, conf=...) -> list[Detection]`` (and,
    for the hardware summary, a ``device`` attribute).
    """
    dataset = SoccerDataset(dataset_root)
    records = dataset.frames(split)[:limit]
    if not records:
        raise ValueError(f"split '{split}' of {dataset_root} has no frames")
    device = getattr(detector, "device", "cpu")
    profiling.reset_gpu_peak(device)
    frames, objects, predictions = [], [], []
    for record in records:
        # Low threshold here so that AP sees the whole score range; the operating threshold is
        # applied afterwards.
        detections = detector.predict(dataset.image(record), conf=0.001)
        frame, labelled = to_frame_boxes(detections, record)
        frames.append(frame)
        objects.append(labelled)
        predictions.append(detections)
    memory = profiling.gpu_memory_mb(device)

    result = metrics.evaluate_frames(frames, schema.DETECTION_CLASSES, conf_threshold,
                                     iou_threshold)
    result["dataset"] = {"root": str(dataset.root), "name": dataset.header["name"],
                         "split": split, "manifest_sha256": dataset.header["manifest_sha256"],
                         "randomization": sorted({r["randomization"] for r in records})}
    result["stratified_recall"] = stratified_recall(frames, objects, records, conf_threshold,
                                                    iou_threshold)
    result["false_positive_analysis"] = false_positive_analysis(
        frames, records, dataset.mask, conf_threshold, iou_threshold)
    sample = [dataset.image(record) for record in records[:16]]
    result["latency"] = {
        "note": "one image per call, end to end (pre-processing, network, NMS), "
                f"operating threshold {conf_threshold}",
        **profiling.measure_latency(lambda image: detector.predict(image, conf=conf_threshold),
                                    sample, device, warmup=10, repeats=latency_repeats)}
    result["memory"] = {"gpu": memory, "process_rss_mb": profiling.process_memory_mb()}
    result["hardware"] = profiling.hardware_summary(device)

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2))
        _write_csv(output / "stratified_recall.csv", result["stratified_recall"])
        (output / "predictions.jsonl").write_text("".join(
            json.dumps({"frame_id": record["frame_id"], "detections": [
                d.to_dict() for d in detections if d.confidence >= conf_threshold]}) + "\n"
            for record, detections in zip(records, predictions)))
        (output / "summary.md").write_text(summary_markdown(result))
        save_examples(dataset, records, frames, output / "examples", conf_threshold,
                      iou_threshold, max_examples)
        try:
            from robosoccer_perception.visualization import vision_plots
            vision_plots.plot_detection_report(result, output)
        except ImportError:                 # matplotlib is optional
            pass
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_markdown(result: dict[str, Any]) -> str:
    def pct(value) -> str:
        return "-" if value is None or value != value else f"{100 * value:.1f} %"

    data = result["dataset"]
    lines = [f"# Detector evaluation: {data['name']} / {data['split']}", "",
             f"{result['num_images']} images, randomization {', '.join(data['randomization'])}. "
             f"Operating point: confidence >= {result['conf_threshold']}, "
             f"IoU >= {result['iou_threshold']}.", "",
             f"- mAP50: **{result['map50']:.3f}**, mAP50-95: **{result['map50_95']:.3f}**",
             f"- precision {pct(result['precision'])}, recall {pct(result['recall'])}",
             f"- false positives per image: {result['false_positives_per_image']:.3f} "
             f"({pct(result['images_with_false_positive'])} of images have at least one)",
             f"- latency: {result['latency']['mean_ms']:.1f} ms mean, "
             f"{result['latency']['p95_ms']:.1f} ms p95 on {result['hardware'].get('gpu', result['hardware']['device'])}",
             "", "| Class | Objects | AP50 | AP50-95 | Precision | Recall | FP / image |",
             "|---|---|---|---|---|---|---|"]
    for name, row in result["per_class"].items():
        lines.append(f"| {name} | {row['num_gt']} | {row['ap50']:.3f} | {row['ap50_95']:.3f} | "
                     f"{pct(row['precision'])} | {pct(row['recall'])} | "
                     f"{row['false_positives_per_image']:.3f} |")
    lines += ["", "## Recall by difficulty", "",
              "| Class | Breakdown | Bin | Detected / total | Recall |", "|---|---|---|---|---|"]
    for row in result["stratified_recall"]:
        if row["breakdown"] != "all":
            lines.append(f"| {row['class']} | {row['breakdown']} | {row['bin']} | "
                         f"{row['detected']} / {row['total']} | {pct(row['recall'])} |")
    lines += ["", "## False positives", "",
              "Badly placed = overlaps an object of its own class (loose or duplicate box). "
              "Other object = overlaps an object of another class. Nothing there = overlaps no "
              "object; the last column says what is under the centre of those boxes.", "",
              "| Class | Count | Per image | Badly placed | Other object | Nothing there | "
              "On images without the class | Nothing there: background / field / line |",
              "|---|---|---|---|---|---|---|---|"]
    for name, row in result["false_positive_analysis"].items():
        on = row["nothing_there_centre_on"]
        lines.append(f"| {name} | {row['false_positives']} | {row['false_positives_per_image']:.3f} | "
                     f"{row['badly_placed']} | {row['other_object']} | {row['nothing_there']} | "
                     f"{row['false_positives_on_images_without_this_class']} in "
                     f"{row['images_without_this_class']} images | "
                     f"{on['background']} / {on['field']} / {on['field_line']} |")
    return "\n".join(lines) + "\n"


def save_examples(dataset: SoccerDataset, records, frames: list[FrameBoxes], output: Path,
                  conf_threshold: float, iou_threshold: float, max_examples: int) -> None:
    """Annotated images: ``correct/`` (everything found, nothing extra), ``false_positive/``
    and ``false_negative/``. Solid class-coloured boxes are correct detections, red dashed
    boxes false positives, white dashed boxes missed objects."""
    from robosoccer_perception.visualization import vision_overlay as vo

    saved = {"correct": 0, "false_positive": 0, "false_negative": 0}
    for kind in saved:                      # examples of an earlier evaluation would mislead
        for stale in (output / kind).glob("*.jpg"):
            stale.unlink()
    for record, frame in zip(records, frames):
        match = metrics.match_frame(frame, iou_threshold, conf_threshold)
        keep = (frame.pred_scores >= conf_threshold) & ~match.pred_ignored
        hits = np.flatnonzero(keep & (match.pred_gt >= 0))
        wrong = np.flatnonzero(keep & (match.pred_gt < 0))
        missed = np.flatnonzero(match.gt_pred < 0)
        kinds = [kind for kind, flag in (("false_positive", len(wrong)), ("false_negative", len(missed)),
                                         ("correct", not len(wrong) and not len(missed) and len(hits)))
                 if flag and saved[kind] < max_examples]
        if not kinds:
            if all(count >= max_examples for count in saved.values()):
                break
            continue
        image = vo.draw_boxes(dataset.image(record), frame.pred_boxes[hits],
                              frame.pred_classes[hits], frame.pred_scores[hits])
        image = vo.draw_boxes(image, frame.pred_boxes[wrong], frame.pred_classes[wrong],
                              colour=vo.ERROR_COLOUR, dashed=True, tags=[
                                  f"FP {schema.DETECTION_CLASSES[c]} {s:.2f}" for c, s in
                                  zip(frame.pred_classes[wrong], frame.pred_scores[wrong])])
        image = vo.draw_boxes(image, frame.gt_boxes[missed], frame.gt_classes[missed],
                              colour=vo.GROUND_TRUTH_COLOUR, dashed=True, tags=[
                                  f"missed {schema.DETECTION_CLASSES[c]}"
                                  for c in frame.gt_classes[missed]])
        for kind in kinds:
            vo.save_image(output / kind / f"{record['frame_id']}.jpg", image)
            saved[kind] += 1
