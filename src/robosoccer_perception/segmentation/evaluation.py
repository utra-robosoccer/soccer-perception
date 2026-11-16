"""Evaluating a segmenter on a generated dataset.

Reports per-class IoU, mean IoU, exact and tolerant field-line precision / recall / F1
(`segmentation.metrics`), line recall and precision as a function of distance from the camera
(lines thin out with distance, so this shows how far out the lines are usable), latency,
memory, and example overlays including the frames with the worst line IoU.

The segmenter sees images only. Labels, and the camera pose used for the distance breakdown,
are used after inference, to score it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.datasets.io import SoccerDataset
from robosoccer_perception.geometry.camera import CameraIntrinsics
from robosoccer_perception.inference import geometry, profiling
from robosoccer_perception.segmentation.metrics import SegmentationMeter, scores_from_confusion, confusion_matrix

LINE_DISTANCE_BINS = (("<2 m", 0.0, 2.0), ("2-4 m", 2.0, 4.0), ("4-6 m", 4.0, 6.0),
                      ("6-9 m", 6.0, 9.0), (">9 m", 9.0, np.inf))


def dataset_intrinsics(dataset: SoccerDataset) -> CameraIntrinsics:
    camera = dataset.header["camera"]
    return CameraIntrinsics(int(camera["width"]), int(camera["height"]), float(camera["fx"]),
                            float(camera["fy"]), float(camera["cx"]), float(camera["cy"]))


def evaluate_segmenter(segmenter, dataset_root, split: str = "test", output_dir=None,
                       max_examples: int = 8, latency_repeats: int = 100,
                       limit: int | None = None, tolerance: int = 2) -> dict[str, Any]:
    """Run ``segmenter`` over a split. ``segmenter`` is anything with
    ``predict(image_rgb) -> Segmentation`` (and, for the hardware summary, ``device``)."""
    dataset = SoccerDataset(dataset_root)
    records = dataset.frames(split)[:limit]
    if not records:
        raise ValueError(f"split '{split}' of {dataset_root} has no frames")
    intrinsics = dataset_intrinsics(dataset)
    device = getattr(segmenter, "device", "cpu")
    meter = SegmentationMeter(tolerance=tolerance)
    by_distance = {label: np.zeros(3, dtype=np.int64) for label, _, _ in LINE_DISTANCE_BINS}
    per_frame = []
    profiling.reset_gpu_peak(device)
    for record in records:
        target = dataset.mask(record)
        prediction = segmenter.predict(dataset.image(record)).mask
        meter.update(prediction, target)
        predicted_line, labelled_line = prediction == schema.FIELD_LINE, target == schema.FIELD_LINE
        distance = geometry.ground_distance_map(
            intrinsics, np.asarray(record["T_field_left_optical"]))
        for label, low, high in LINE_DISTANCE_BINS:
            zone = (distance >= low) & (distance < high)
            by_distance[label] += [int((predicted_line & labelled_line & zone).sum()),
                                   int((predicted_line & zone).sum()),
                                   int((labelled_line & zone).sum())]
        frame_scores = scores_from_confusion(confusion_matrix(prediction, target, meter.num_classes))
        per_frame.append({"frame_id": record["frame_id"], "mode": record["frame"]["mode"],
                          "line_iou": frame_scores["per_class"]["field_line"]["iou"],
                          "mean_iou": frame_scores["mean_iou"],
                          "line_pixels": int(labelled_line.sum())})
    memory = profiling.gpu_memory_mb(device)

    result = meter.compute()
    result["classes"] = list(schema.SEMANTIC_CLASSES)
    result["dataset"] = {"root": str(dataset.root), "name": dataset.header["name"],
                         "split": split, "manifest_sha256": dataset.header["manifest_sha256"],
                         "randomization": sorted({r["randomization"] for r in records})}
    result["line_by_distance"] = [
        {"bin": label, "labelled_pixels": int(labelled),
         "recall": float(hit / labelled) if labelled else None,
         "precision": float(hit / predicted) if predicted else None}
        for label, (hit, predicted, labelled) in by_distance.items()]
    with_lines = [f["line_iou"] for f in per_frame if f["line_pixels"] > 0 and f["line_iou"] is not None]
    result["per_frame_line_iou"] = {
        "frames_with_lines": len(with_lines),
        "mean": float(np.mean(with_lines)) if with_lines else None,
        "p10": float(np.percentile(with_lines, 10)) if with_lines else None,
        "median": float(np.median(with_lines)) if with_lines else None}
    sample = [dataset.image(record) for record in records[:16]]
    result["latency"] = {"note": "one image per call, end to end (normalisation, network, "
                                 "softmax, argmax, copy to CPU)",
                         **profiling.measure_latency(segmenter.predict, sample, device,
                                                     warmup=10, repeats=latency_repeats)}
    result["memory"] = {"gpu": memory, "process_rss_mb": profiling.process_memory_mb()}
    result["hardware"] = profiling.hardware_summary(device)
    parameters = getattr(segmenter, "model", None)
    if parameters is not None:
        result["parameters"] = int(sum(p.numel() for p in parameters.parameters()))

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2))
        (output / "per_frame.json").write_text(json.dumps(per_frame, indent=2))
        (output / "summary.md").write_text(summary_markdown(result))
        save_examples(segmenter, dataset, records, per_frame, output / "examples", max_examples)
        try:
            from robosoccer_perception.visualization import vision_plots
            vision_plots.plot_segmentation_report(result, output)
        except ImportError:                 # matplotlib is optional
            pass
    return result


def summary_markdown(result: dict[str, Any]) -> str:
    def pct(value) -> str:
        return "-" if value is None else f"{100 * value:.1f} %"

    data, line, tolerant = result["dataset"], result["line"], result["line_tolerant"]
    lines = [f"# Segmentation evaluation: {data['name']} / {data['split']}", "",
             f"{result['num_images']} images, randomization {', '.join(data['randomization'])}.", "",
             f"- mean IoU: **{pct(result['mean_iou'])}** (pixel accuracy {pct(result['pixel_accuracy'])}; "
             "do not read accuracy alone, lines are about 3 % of the pixels)",
             f"- field-line IoU: **{pct(line['iou'])}**, precision {pct(line['precision'])}, "
             f"recall {pct(line['recall'])}, F1 {pct(line['f1'])}",
             f"- field-line within {tolerant['tolerance_px']} px: precision {pct(tolerant['precision'])}, "
             f"recall {pct(tolerant['recall'])}, F1 {pct(tolerant['f1'])}",
             f"- latency: {result['latency']['mean_ms']:.1f} ms mean, {result['latency']['p95_ms']:.1f} ms p95 on "
             f"{result['hardware'].get('gpu', result['hardware']['device'])}",
             "", "| Class | IoU | Precision | Recall | F1 | Share of pixels |", "|---|---|---|---|---|---|"]
    total = sum(row["pixels"] for row in result["per_class"].values())
    for name, row in result["per_class"].items():
        lines.append(f"| {name} | {pct(row['iou'])} | {pct(row['precision'])} | {pct(row['recall'])} | "
                     f"{pct(row['f1'])} | {pct(row['pixels'] / total)} |")
    lines += ["", "## Field-line pixels by distance from the camera", "",
              "| Distance | Labelled line pixels | Recall | Precision |", "|---|---|---|---|"]
    for row in result["line_by_distance"]:
        lines.append(f"| {row['bin']} | {row['labelled_pixels']} | {pct(row['recall'])} | "
                     f"{pct(row['precision'])} |")
    return "\n".join(lines) + "\n"


def error_overlay(image_rgb: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Line errors on a dimmed image: correct line pixels white, false line pixels red, missed
    line pixels yellow. Field / background disagreements are tinted blue."""
    out = (image_rgb.astype(np.float32) * 0.45)
    predicted, labelled = prediction == schema.FIELD_LINE, target == schema.FIELD_LINE
    other = (prediction != target) & ~predicted & ~labelled
    out[other] = 0.5 * out[other] + 0.5 * np.array([60, 110, 255], np.float32)
    out[predicted & labelled] = (255, 255, 255)
    out[predicted & ~labelled] = (255, 40, 40)
    out[~predicted & labelled] = (255, 220, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_examples(segmenter, dataset: SoccerDataset, records, per_frame, output: Path,
                  max_examples: int) -> None:
    """``typical/``: evenly spaced frames; ``worst_line_iou/``: the frames with the lowest line
    IoU among those with a meaningful amount of line. Each panel is image | prediction | errors."""
    from robosoccer_perception.visualization import vision_overlay as vo

    typical = list(np.linspace(0, len(records) - 1, min(max_examples, len(records))).astype(int))
    candidates = [i for i, f in enumerate(per_frame) if f["line_pixels"] >= 200 and f["line_iou"] is not None]
    worst = sorted(candidates, key=lambda i: per_frame[i]["line_iou"])[:max_examples]
    for folder, indices in (("typical", typical), ("worst_line_iou", worst)):
        for stale in (output / folder).glob("*.jpg"):       # from an earlier evaluation
            stale.unlink()
        for index in indices:
            record = records[index]
            image, target = dataset.image(record), dataset.mask(record)
            prediction = segmenter.predict(image).mask
            panel = np.concatenate([image, vo.overlay_mask(image, prediction),
                                    error_overlay(image, prediction, target)], axis=1)
            iou = per_frame[index]["line_iou"]
            vo._caption(panel, f"{record['frame_id']}  line IoU "
                               f"{'-' if iou is None else f'{iou:.2f}'}   image | prediction | "
                               "errors (white ok, red false, yellow missed)", 4, panel.shape[0] - 4,
                        (255, 255, 255))
            vo.save_image(output / folder / f"{record['frame_id']}.jpg", panel)
