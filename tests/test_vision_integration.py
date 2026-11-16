"""End to end, small and offline: generate a dataset, validate its labels, train a detector and
a segmenter for a moment, run the combined pipeline, save annotated images, evaluate.

Slow (about a minute on CPU) and needs rendering plus the ``vision`` extra. Nothing is
downloaded: the detector starts from an architecture file with random weights, so its
predictions are meaningless. What is tested is that every stage runs and hands the next one
what it expects, not accuracy."""
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

pytestmark = [pytest.mark.vision, pytest.mark.render, pytest.mark.slow]


@pytest.fixture(scope="module")
def trained(tiny_dataset, tmp_path_factory):
    """A detector and a segmenter trained for one or two epochs on the tiny dataset."""
    from robosoccer_perception.detection import yolo
    from robosoccer_perception.segmentation.trainer import (
        load_segmentation_config, train_segmentation)

    root = tmp_path_factory.mktemp("runs")
    detection_cfg = yolo.load_detection_config("yolo11n_smoke", {
        "name": "tiny", "model": "yolo11n.yaml", "epochs": 1, "imgsz": 160, "batch": 4,
        "device": "cpu", "workers": 0})
    detector_run = yolo.train_detector(detection_cfg, tiny_dataset, root / "detection")
    segmentation_cfg = load_segmentation_config("unet_smoke", {
        "name": "tiny", "epochs": 2, "batch": 4, "device": "cpu", "workers": 0,
        "augment": {"crop_size": [160, 96]}})
    segmenter_run = train_segmentation(segmentation_cfg, tiny_dataset, root / "segmentation",
                                       progress=None)
    return {"root": root, "detector": detector_run, "segmenter": segmenter_run}


def test_training_runs_leave_loadable_checkpoints(trained):
    from pathlib import Path

    run = Path(trained["detector"]["run_dir"])
    assert Path(trained["detector"]["best"]).is_file() and Path(trained["detector"]["last"]).is_file()
    results = (run / "results.csv").read_text().splitlines()
    assert len(results) == 2 and "train/box_loss" in results[0], "one epoch of losses was logged"
    assert all(np.isfinite(float(v)) for v in results[1].split(",")[2:5]), "finite losses"
    assert Path(trained["segmenter"]["best"]).is_file()
    assert trained["segmenter"]["epochs_run"] == 2


def test_combined_pipeline_on_dataset_frames_saves_predictions_and_overlays(trained, tiny_dataset,
                                                                            tmp_path):
    from robosoccer_perception.datasets import schema
    from robosoccer_perception.datasets.io import SoccerDataset, load_image
    from robosoccer_perception.inference.pipeline import VisionPipeline, render_overlay
    from robosoccer_perception.visualization import vision_overlay

    pipeline = VisionPipeline.from_checkpoints(trained["detector"]["best"],
                                               trained["segmenter"]["best"], device="cpu",
                                               conf=0.001, imgsz=160)
    assert pipeline.detector.model.names == dict(enumerate(schema.DETECTION_CLASSES))
    dataset = SoccerDataset(tiny_dataset)
    for record in dataset.frames("test"):
        image = dataset.image(record)
        result = pipeline.run(image, source=record["frame_id"])       # the image is all it gets
        assert (result.width, result.height) == (320, 180)
        assert result.segmentation.mask.shape == (180, 320)
        assert set(np.unique(result.segmentation.mask)) <= {0, 1, 2}
        assert all(d.class_id in (0, 1, 2) and 0 <= d.confidence <= 1 for d in result.detections)
        assert result.timing_ms["total"] > 0
        paths = result.save(tmp_path, record["frame_id"])
        vision_overlay.save_image(tmp_path / f"{record['frame_id']}_overlay.jpg",
                                  render_overlay(image, result))
        saved = json.loads(paths["json"].read_text())
        assert saved["source"] == record["frame_id"] and saved["segmentation"]["mask_file"]
        overlay = load_image(tmp_path / f"{record['frame_id']}_overlay.jpg")
        assert overlay.shape == image.shape and np.abs(overlay.astype(int) - image).mean() > 1.0
    assert len(list(tmp_path.glob("*_overlay.jpg"))) == 4
    assert len(list(tmp_path.glob("*_mask.png"))) == 4


def test_pipeline_runs_on_a_full_resolution_simulated_camera_frame(trained, s_field):
    """A 1280x720 frame straight from the virtual ZED Mini: other size than the training data."""
    from robosoccer_perception.inference.pipeline import VisionPipeline
    from robosoccer_perception.simulation.mujoco_scene import MujocoScene
    from robosoccer_perception.simulation.stereo_camera import VirtualStereoCamera

    pipeline = VisionPipeline.from_checkpoints(trained["detector"]["best"],
                                               trained["segmenter"]["best"], device="cpu",
                                               imgsz=160)
    with MujocoScene(s_field) as scene:
        scene.teleport([-2.0, -1.0, 0.3])
        frame = VirtualStereoCamera(scene).capture()
    result = pipeline.run(frame.left)
    assert (result.width, result.height) == (1280, 720)
    assert result.segmentation.mask.shape == (720, 1280)
    assert result.segmentation.probabilities.shape == (3, 720, 1280)


def test_evaluations_produce_their_reports(trained, tiny_dataset, tmp_path):
    from robosoccer_perception.detection.evaluation import evaluate_detector
    from robosoccer_perception.detection.yolo import YoloDetector, validate_detector
    from robosoccer_perception.segmentation.evaluation import evaluate_segmenter
    from robosoccer_perception.segmentation.segmenter import Segmenter

    detector = YoloDetector(trained["detector"]["best"], device="cpu", imgsz=160)
    report = evaluate_detector(detector, tiny_dataset, "test", conf_threshold=0.25,
                               output_dir=tmp_path / "detection", max_examples=2,
                               latency_repeats=3)
    assert report["num_images"] == 4
    assert set(report["per_class"]) == {"ball", "goal_post", "robot"}
    assert {"map50", "map50_95", "precision", "recall", "confusion_matrix", "stratified_recall",
            "false_positive_analysis", "latency", "memory", "hardware"} <= set(report)
    assert any(row["breakdown"] == "distance" for row in report["stratified_recall"])
    for name in ("metrics.json", "summary.md", "stratified_recall.csv", "predictions.jsonl",
                 "confusion_matrix.png"):
        assert (tmp_path / "detection" / name).is_file(), name
    assert len((tmp_path / "detection" / "predictions.jsonl").read_text().splitlines()) == 4
    cross_check = validate_detector(trained["detector"]["best"], tiny_dataset, "test", imgsz=160,
                                    batch=4, device="cpu", workers=0)
    assert 0.0 <= cross_check["map50"] <= 1.0

    segmenter = Segmenter(trained["segmenter"]["best"], device="cpu")
    report = evaluate_segmenter(segmenter, tiny_dataset, "test", tmp_path / "segmentation",
                                max_examples=2, latency_repeats=3)
    assert report["num_images"] == 4 and report["parameters"] > 0
    assert {"mean_iou", "line", "line_tolerant", "line_by_distance", "latency"} <= set(report)
    for name in ("metrics.json", "summary.md", "per_frame.json", "confusion_matrix.png",
                 "line_by_distance.png"):
        assert (tmp_path / "segmentation" / name).is_file(), name


def test_models_export_to_onnx(trained, tmp_path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    from robosoccer_perception.segmentation.export import export_segmenter_onnx

    path = export_segmenter_onnx(trained["segmenter"]["best"], tmp_path / "segmenter.onnx",
                                 height=192, width=320)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits, = session.run(None, {"image": np.zeros((1, 3, 192, 320), np.float32)})
    assert logits.shape == (1, 3, 192, 320)
