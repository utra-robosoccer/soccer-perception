"""The offline inference pipeline, its output types, the geometric inspection utilities, and
the boundaries Phase 2 must respect (no ground truth into models, no localization coupling)."""
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from robosoccer_perception.datasets import schema
from robosoccer_perception.geometry.camera import RigState, StereoCameraConfig, StereoCameraModel
from robosoccer_perception.inference import geometry, profiling
from robosoccer_perception.inference.pipeline import VisionPipeline
from robosoccer_perception.inference.types import (
    VISION_RESULT_VERSION, Detection, Segmentation, VisionResult)

SOURCE = Path(__file__).resolve().parents[1] / "src" / "robosoccer_perception"


class FakeDetector:
    device = "cpu"

    def predict(self, image_rgb, conf=None):
        return [Detection(schema.BALL, 0.4, (10.0, 20.0, 30.0, 40.0)),
                Detection(schema.BALL, 0.9, (100.0, 50.0, 120.0, 70.0)),
                Detection(schema.GOAL_POST, 0.7, (200.0, 0.0, 210.0, 90.0)),
                Detection(schema.ROBOT, 0.8, (50.0, 30.0, 90.0, 110.0))]


class FakeSegmenter:
    device = "cpu"

    def predict(self, image_rgb):
        height, width = image_rgb.shape[:2]
        mask = np.full((height, width), schema.FIELD, np.uint8)
        mask[: height // 4] = schema.BACKGROUND
        mask[height // 2] = schema.FIELD_LINE
        probabilities = np.eye(3, dtype=np.float32)[mask].transpose(2, 0, 1)
        return Segmentation(mask, probabilities)


IMAGE = np.zeros((120, 240, 3), np.uint8)


# ------------------------------------------------------------------ output types
def test_detection_accessors_and_round_trip():
    detection = Detection(schema.GOAL_POST, 0.75, (200.0, 10.0, 210.0, 90.0))
    assert detection.class_name == "goal_post"
    assert detection.center == (205.0, 50.0) and detection.size == (10.0, 80.0)
    assert detection.bottom_center == (205.0, 90.0), "where a standing object meets the ground"
    assert Detection.from_dict(json.loads(json.dumps(detection.to_dict()))) == detection


def test_pipeline_output_schema(tmp_path):
    result = VisionPipeline(FakeDetector(), FakeSegmenter()).run(IMAGE, source="frame_7")
    assert isinstance(result, VisionResult) and (result.width, result.height) == (240, 120)
    assert result.ball.confidence == 0.9, "the most confident ball"
    assert len(result.goal_posts) == 1 and len(result.robots) == 1
    assert [d.confidence for d in result.of_class("ball")] == [0.9, 0.4]
    assert result.segmentation.mask.shape == (120, 240)
    assert result.segmentation.line_mask.sum() == 240 and result.segmentation.field_mask.any()
    assert result.segmentation.line_confidence.shape == (120, 240)
    assert set(result.timing_ms) == {"detector", "segmenter", "total"}
    assert result.timing_ms["total"] >= result.timing_ms["detector"] >= 0.0

    data = json.loads(json.dumps(result.to_dict()))
    assert data["vision_result_version"] == VISION_RESULT_VERSION and data["source"] == "frame_7"
    assert data["detection_classes"] == list(schema.DETECTION_CLASSES)
    assert {"class_id", "class_name", "confidence", "bbox_xyxy"} == set(data["detections"][0])
    assert data["segmentation"]["semantic_classes"] == list(schema.SEMANTIC_CLASSES)
    assert sum(data["segmentation"]["pixel_share"].values()) == pytest.approx(1.0)

    cv2 = pytest.importorskip("cv2")
    paths = result.save(tmp_path, "frame_7")
    saved = json.loads(paths["json"].read_text())
    assert saved["segmentation"]["mask_file"] == "frame_7_mask.png"
    np.testing.assert_array_equal(cv2.imread(str(paths["mask"]), cv2.IMREAD_UNCHANGED),
                                  result.segmentation.mask)
    confidence = cv2.imread(str(paths["line_confidence"]), cv2.IMREAD_UNCHANGED)
    assert confidence.dtype == np.uint8 and confidence.max() == 255


def test_pipeline_components_are_optional_and_inputs_are_checked():
    detections_only = VisionPipeline(detector=FakeDetector()).run(IMAGE)
    assert detections_only.segmentation is None and len(detections_only.detections) == 4
    assert detections_only.to_dict()["segmentation"] is None
    mask_only = VisionPipeline(segmenter=FakeSegmenter()).run(IMAGE)
    assert mask_only.detections == () and mask_only.ball is None
    assert "detector" not in mask_only.timing_ms
    with pytest.raises(ValueError):
        VisionPipeline()
    for bad in (IMAGE.astype(np.float32), IMAGE[..., 0], np.zeros((4, 4, 4), np.uint8)):
        with pytest.raises(ValueError, match="RGB"):
            VisionPipeline(FakeDetector()).run(bad)


def test_overlay_is_drawn_on_a_copy():
    pytest.importorskip("cv2")
    from robosoccer_perception.inference.pipeline import render_overlay

    image = np.full((120, 240, 3), 90, np.uint8)
    result = VisionPipeline(FakeDetector(), FakeSegmenter()).run(image)
    overlay = render_overlay(image, result)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8
    assert (image == 90).all() and (overlay != image).any()


# ------------------------------------------------------------------ geometric inspection tools
def test_skeleton_is_thin_connected_and_inside_the_line():
    mask = np.zeros((60, 80), bool)
    mask[20:27, 5:75] = True                          # a 7 px thick horizontal bar
    mask[5:55, 40:45] = True                          # crossed by a 5 px thick vertical bar
    skeleton = geometry.skeletonize(mask)
    assert skeleton.any() and not (skeleton & ~mask).any()
    assert skeleton.sum() < 0.25 * mask.sum()
    assert skeleton[:, 10:35].sum(axis=0).max() == 1, "one pixel thick across the bar"
    assert skeleton[23, 10:35].all(), "and it runs along the bar's middle"
    cv2 = pytest.importorskip("cv2")
    components, _ = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    assert components - 1 == 1, "thinning does not break the shape apart"
    assert not geometry.skeletonize(np.zeros((10, 10), bool)).any()


def test_line_pixel_sampling_is_bounded_spread_and_deterministic():
    mask = np.zeros((90, 160), bool)
    mask[40:43, :] = True
    mask[:, 80:83] = True
    points = geometry.sample_line_pixels(mask, max_points=40)
    assert 10 <= len(points) <= 40 and points.shape[1] == 2
    assert mask[points[:, 1], points[:, 0]].all(), "(u, v) points lie on the line"
    assert points[:, 0].max() - points[:, 0].min() > 120, "spread along the whole line"
    np.testing.assert_array_equal(points, geometry.sample_line_pixels(mask, max_points=40))
    assert geometry.sample_line_pixels(np.zeros((9, 16), bool)).shape == (0, 2)


def test_ground_projection_inverts_the_camera_model(rng):
    camera = StereoCameraModel(StereoCameraConfig.load().with_resolution(640, 360))
    rig = RigState(height=0.9, pitch=math.radians(25), pan=0.2, roll=math.radians(3))
    T_base_camera = camera.T_base_camera("left", rig)
    ground = np.column_stack([rng.uniform(0.8, 6.0, 50), rng.uniform(-2.0, 2.0, 50), np.zeros(50)])
    pixels, _, visible = camera.project_field_points([0.0, 0.0, 0.0], ground, "left", rig)
    assert visible.sum() > 20
    points, valid = geometry.pixels_to_ground(pixels[visible], camera.intrinsics, T_base_camera)
    assert valid.all()
    np.testing.assert_allclose(points, ground[visible, :2], atol=1e-9)
    # Above the horizon there is no ground to hit.
    sky, valid = geometry.pixels_to_ground([[320.0, 0.0]], camera.intrinsics,
                                           camera.T_base_camera("left", rig.replace(pitch=0.0, roll=0.0)))
    assert not valid[0] and np.isnan(sky).all()


def test_ground_distance_map_grows_toward_the_horizon():
    camera = StereoCameraModel(StereoCameraConfig.load().with_resolution(320, 180))
    T = camera.T_base_camera("left", RigState(height=0.9, pitch=math.radians(25)))
    distance = geometry.ground_distance_map(camera.intrinsics, T)
    assert distance.shape == (180, 320)
    column = distance[:, 160]
    assert np.isinf(column[0]), "the top row looks above the horizon"
    finite = column[np.isfinite(column)]
    assert (np.diff(finite) < 0).all(), "lower rows look at nearer ground"
    assert finite[-1] == pytest.approx(0.9 / math.tan(math.radians(25) + math.atan(89.5 / 160)), rel=1e-3)


# ------------------------------------------------------------------ profiling
def test_latency_statistics_and_device_resolution():
    stats = profiling.latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["mean_ms"] == 25.0 and stats["median_ms"] == 25.0 and stats["count"] == 4
    assert stats["fps_at_mean"] == pytest.approx(40.0) and stats["p95_ms"] > 35.0
    assert profiling.latency_stats([]) == {"count": 0}
    calls = []
    measured = profiling.measure_latency(calls.append, [1, 2, 3], "cpu", warmup=2, repeats=5)
    assert measured["count"] == 5 and len(calls) == 7
    assert profiling.resolve_device("cpu") == "cpu" and profiling.resolve_device(0) == "cuda:0"
    assert profiling.resolve_device("auto") in ("cpu", "cuda:0")
    assert profiling.gpu_memory_mb("cpu") is None
    assert "device" in profiling.hardware_summary("cpu")


# ------------------------------------------------------------------ Phase 2 boundaries
VISION_PACKAGES = ("datasets", "detection", "segmentation", "inference")


def _imports(path: Path) -> set[str]:
    return set(re.findall(r"^\s*(?:from|import)\s+(robosoccer_perception[\w.]*)", path.read_text(),
                          re.MULTILINE))


def test_vision_code_is_not_coupled_to_localization():
    """Phase 3 has not started: nothing in the vision packages builds observations for, or
    talks to, the particle filter."""
    for package in VISION_PACKAGES:
        for path in (SOURCE / package).glob("*.py"):
            for module in _imports(path):
                assert not module.startswith(("robosoccer_perception.localization",
                                              "robosoccer_perception.observations",
                                              "robosoccer_perception.evaluation")), (path.name, module)
            assert "ObservationFrame(" not in path.read_text(), path.name


def test_localization_code_does_not_depend_on_the_vision_stack():
    """Phase 1 keeps running with NumPy alone: no torch, no OpenCV, no vision packages."""
    for package in ("localization", "observations", "evaluation", "geometry"):
        for path in (SOURCE / package).glob("*.py"):
            text = path.read_text()
            assert not re.search(r"^\s*(import|from)\s+(torch|cv2|ultralytics)\b", text, re.MULTILINE)
            for module in _imports(path):
                assert not set(module.split(".")[1:2]) & set(VISION_PACKAGES), (path.name, module)


def test_models_never_receive_ground_truth():
    """The inference modules cannot see simulator state: they import neither the simulator nor
    the dataset reader, and a `VisionResult` has no field that could carry a label."""
    for name in ("pipeline.py", "types.py", "geometry.py", "profiling.py"):
        for module in _imports(SOURCE / "inference" / name):
            assert "simulation" not in module and not module.endswith(("datasets.io", "datasets.generator",
                                                                       "datasets.annotation")), (name, module)
    for name in ("segmenter.py", "model.py"):
        assert not any("simulation" in m or m.endswith("datasets.io") for m in _imports(SOURCE / "segmentation" / name))
    fields = {f for f in VisionResult.__dataclass_fields__}
    assert fields == {"width", "height", "detections", "segmentation", "timing_ms", "source"}
    assert set(Detection.__dataclass_fields__) == {"class_id", "confidence", "bbox_xyxy"}
