#!/usr/bin/env python3
"""Run the offline vision pipeline (detector + segmenter) on images and save what it sees.

Three sources of images:

    # saved images (files or directories)
    python scripts/run_vision.py --images outputs/render/s_field_stereo_left.png

    # frames of a generated dataset (ground truth is NOT given to the models)
    python scripts/run_vision.py --dataset outputs/datasets/baseline --split test --count 12

    # freshly rendered simulated ZED Mini frames at the camera's full 1280x720
    python scripts/run_vision.py --render 8 --seed 123 --randomization strong
    python scripts/run_vision.py --pose -2 -1 20 --ball 0.5 0.2

Models default to the baseline runs; override with ``--detector`` / ``--segmenter``, or drop
one with ``--no-detector`` / ``--no-segmenter``.

For every image, in ``--output`` (default ``outputs/vision``):
    <name>_overlay.jpg            image with predicted mask and boxes
    <name>.json                   detections, confidences, timings (inference.types.VisionResult)
    <name>_mask.png               predicted class ids (0 background, 1 field, 2 field_line)
    <name>_line_confidence.png    predicted line probability, 0-255
    <name>_ground.png             with --ground: predicted line pixels projected onto the ground
and ``summary.json`` with end-to-end latency and memory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def collect_images(args) -> list[tuple[str, np.ndarray, dict]]:
    """``[(name, image_rgb, extra)]``; ``extra`` may hold camera geometry for --ground."""
    from robosoccer_perception.datasets.io import SoccerDataset, load_image

    items: list[tuple[str, np.ndarray, dict]] = []
    if args.images:
        for entry in args.images:
            path = Path(entry)
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) \
                if path.is_dir() else [path]
            items += [(file.stem, load_image(file), {}) for file in files]
    elif args.dataset:
        from robosoccer_perception.segmentation.evaluation import dataset_intrinsics

        dataset = SoccerDataset(args.dataset)
        records = dataset.frames(args.split)
        chosen = np.random.default_rng(args.seed).choice(
            len(records), size=min(args.count, len(records)), replace=False)
        intrinsics = dataset_intrinsics(dataset)
        for index in sorted(chosen):
            record = records[index]
            items.append((record["frame_id"], dataset.image(record), {
                "intrinsics": intrinsics, "field": dataset.header["field"]["name"],
                "T_field_camera": np.asarray(record["T_field_left_optical"])}))
    else:
        items = render_images(args)
    if not items:
        raise SystemExit("no images to process")
    return items


def render_images(args) -> list[tuple[str, np.ndarray, dict]]:
    """Simulated left-camera frames. Rendering and labelling state stay on this side of the
    pipeline: only the RGB image is handed to the models."""
    import dataclasses

    from robosoccer_perception.datasets.generator import DatasetConfig, ScenarioRenderer
    from robosoccer_perception.geometry.camera import StereoCameraConfig

    native = StereoCameraConfig.load(args.camera).intrinsics
    width, height = args.resolution or (native.width, native.height)
    cfg = DatasetConfig.from_dict({
        "name": "run_vision", "seed": args.seed, "field": args.field, "camera": args.camera,
        "resolution": [width, height], "randomization": args.randomization,
        "splits": {"test": {"scenarios": 1, "frames_per_scenario": 1}}})
    items = []
    if args.pose:
        from robosoccer_perception.geometry.field import FieldGeometry
        from robosoccer_perception.simulation.mujoco_scene import LEFT_CAMERA, MujocoScene

        camera_cfg = StereoCameraConfig.load(args.camera).with_resolution(width, height)
        with MujocoScene(FieldGeometry.load(args.field), camera_cfg) as scene:
            scene.teleport([args.pose[0], args.pose[1], math.radians(args.pose[2])])
            scene.set_rig(height=args.height,
                          pitch=None if args.pitch is None else math.radians(args.pitch))
            if args.ball:
                scene.set_ball(args.ball)
            items.append(("pose", scene.render(LEFT_CAMERA, show_platform=False), {
                "intrinsics": camera_cfg.intrinsics, "field": args.field,
                "T_field_camera": scene.camera_pose(LEFT_CAMERA)}))
        return items
    # One fresh scenario (scene appearance) per frame, from seeds no dataset preset uses.
    for index in range(args.render):
        scenario_cfg = dataclasses.replace(cfg, seed=args.seed + index)
        with ScenarioRenderer(scenario_cfg, "test", 0) as scenario:
            frame = scenario.render(0)
            items.append((f"render_{args.seed + index}_{frame.spec.mode}", frame.image, {
                "intrinsics": scenario.camera_cfg.intrinsics, "field": args.field,
                "T_field_camera": frame.T_field_camera}))
    return items


def ground_figure(result, extra: dict, path: Path) -> None:
    """Top view: predicted line pixels projected onto the ground with the TRUE camera pose,
    over the field drawing. An inspection plot of mask quality, not localization."""
    from robosoccer_perception.geometry.field import FieldGeometry
    from robosoccer_perception.inference import geometry
    from robosoccer_perception.visualization import style
    from robosoccer_perception.visualization.field_plot import draw_field

    field = FieldGeometry.load(extra["field"])
    pixels = geometry.sample_line_pixels(result.segmentation.line_mask, max_points=400)
    intrinsics = extra["intrinsics"]
    scale = result.width / intrinsics.width          # intrinsics are those of the source image
    assert abs(scale - 1.0) < 1e-6, "image size differs from the camera intrinsics"
    points, valid = geometry.pixels_to_ground(pixels, intrinsics, extra["T_field_camera"])
    points = points[valid]
    camera_xy = extra["T_field_camera"][:2, 3]
    near = np.hypot(*(points - camera_xy).T) < 12.0
    figure = style.new_figure(8.0, 5.8)
    ax = figure.subplots()
    draw_field(ax, field)
    ax.scatter(points[near, 0], points[near, 1], s=10, color=style.OBSERVATION,
               label=f"predicted line pixels on the ground ({int(near.sum())})", zorder=3)
    ax.scatter(*camera_xy, s=70, marker="^", color=style.TRUTH, label="camera (true pose)", zorder=4)
    error = field.distance_to_paint(points[near]) if near.any() else np.zeros(0)
    title = "Predicted line pixels projected with the true camera pose"
    if error.size:
        title += f"\nmedian distance to paint {100 * np.median(error):.1f} cm, " \
                 f"90th percentile {100 * np.percentile(error, 90):.1f} cm"
    ax.set_title(title, fontsize=10, loc="left", color=style.TEXT)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=style.SURFACE)
    figure.tight_layout()
    figure.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("image source (default: --render 4)")
    source.add_argument("--images", nargs="+", help="image files or directories")
    source.add_argument("--dataset", help="generated dataset directory")
    source.add_argument("--split", default="test")
    source.add_argument("--count", type=int, default=8, help="frames taken from --dataset")
    source.add_argument("--render", type=int, default=4, help="number of random simulated frames")
    source.add_argument("--pose", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                        help="render one frame of the default scene from this platform pose")
    source.add_argument("--ball", type=float, nargs=2, metavar=("X", "Y"))
    source.add_argument("--height", type=float, help="camera height for --pose, metres")
    source.add_argument("--pitch", type=float, help="camera pitch for --pose, degrees down")
    source.add_argument("--field", default="s_field")
    source.add_argument("--camera", default="zed_mini_sim")
    source.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="render size (default: the camera's native 1280x720)")
    source.add_argument("--randomization", default="mild", help="preset for --render")
    source.add_argument("--seed", type=int, default=9001)
    models = parser.add_argument_group("models")
    models.add_argument("--detector", default="outputs/detection/yolo11n/weights/best.pt")
    models.add_argument("--segmenter", default="outputs/segmentation/unet_small/best.pt")
    models.add_argument("--no-detector", action="store_true")
    models.add_argument("--no-segmenter", action="store_true")
    models.add_argument("--conf", type=float, default=0.25)
    models.add_argument("--imgsz", type=int, default=640)
    models.add_argument("--device", default="auto")
    models.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    parser.add_argument("--ground", action="store_true",
                        help="also save a top view of the predicted line pixels (rendered or "
                             "dataset frames only: needs the camera pose)")
    parser.add_argument("--output", default="outputs/vision")
    args = parser.parse_args()

    from robosoccer_perception.inference import profiling
    from robosoccer_perception.inference.pipeline import VisionPipeline, render_overlay
    from robosoccer_perception.visualization import vision_overlay

    items = collect_images(args)
    pipeline = VisionPipeline.from_checkpoints(
        None if args.no_detector else args.detector,
        None if args.no_segmenter else args.segmenter,
        device=args.device, conf=args.conf, imgsz=args.imgsz, half=args.half)
    output = Path(args.output)
    pipeline.warmup(items[0][1])
    profiling.reset_gpu_peak(pipeline.device)
    timings = {"detector": [], "segmenter": [], "total": []}
    for name, image, extra in items:
        result = pipeline.run(image, source=name)
        for key, value in result.timing_ms.items():
            timings[key].append(value)
        result.save(output, name)
        vision_overlay.save_image(output / f"{name}_overlay.jpg", render_overlay(image, result))
        if args.ground and result.segmentation is not None and "T_field_camera" in extra:
            ground_figure(result, extra, output / f"{name}_ground.png")
        boxes = ", ".join(f"{d.class_name} {d.confidence:.2f}" for d in result.detections) or "-"
        print(f"{name}: {result.timing_ms['total']:.1f} ms  [{boxes}]")
    summary = {"images": len(items), "image_size": [items[0][1].shape[1], items[0][1].shape[0]],
               "pipeline": pipeline.describe(),
               "latency_ms": {key: profiling.latency_stats(values)
                              for key, values in timings.items() if values},
               "memory": {"gpu": profiling.gpu_memory_mb(pipeline.device),
                          "process_rss_mb": profiling.process_memory_mb()},
               "hardware": profiling.hardware_summary(pipeline.device)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    total = summary["latency_ms"]["total"]
    print(f"end to end: {total['mean_ms']:.1f} ms mean, {total['p95_ms']:.1f} ms p95 over "
          f"{total['count']} images on {pipeline.device}")
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
