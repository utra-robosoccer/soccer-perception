"""Detection: metrics (pure NumPy), stratified evaluation, configuration, and the YOLO wrapper.

The metric tests need nothing but NumPy. Tests marked ``vision`` need PyTorch and Ultralytics;
none of them downloads anything (models are built from an architecture file with random
weights)."""
import json

import numpy as np
import pytest

from robosoccer_perception.datasets import schema
from robosoccer_perception.detection import metrics
from robosoccer_perception.detection.metrics import FrameBoxes

GT = [[10, 10, 50, 50], [100, 20, 140, 100], [200, 200, 260, 240]]
GT_CLASSES = [schema.BALL, schema.GOAL_POST, schema.ROBOT]


def shifted(box, dx):
    return [box[0] + dx, box[1], box[2] + dx, box[3]]


# ------------------------------------------------------------------ IoU and matching
def test_box_iou_known_values():
    iou = metrics.box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [10, 0, 20, 10],
                                             [2, 2, 8, 8]])
    np.testing.assert_allclose(iou, [[1.0, 1 / 3, 0.0, 0.36]])
    assert metrics.box_iou(np.zeros((0, 4)), GT).shape == (0, 3)
    assert metrics.box_iou([[5, 5, 5, 5]], [[5, 5, 5, 5]])[0, 0] == 0.0     # empty boxes


def test_matching_is_greedy_by_score_one_to_one_and_class_aware():
    frame = FrameBoxes.build(
        [GT[0], shifted(GT[0], 2), GT[1], GT[2]],
        [schema.BALL, schema.BALL, schema.ROBOT, schema.ROBOT], [0.6, 0.9, 0.8, 0.7],
        GT, GT_CLASSES)
    match = metrics.match_frame(frame)
    assert match.pred_gt.tolist() == [-1, 0, -1, 2], \
        "the higher-scoring duplicate wins; the wrong class does not match"
    assert match.gt_pred.tolist() == [1, -1, 3]
    strict = metrics.match_frame(frame, iou_threshold=0.95)
    assert strict.pred_gt.tolist() == [0, -1, -1, 2]
    thresholded = metrics.match_frame(frame, min_score=0.75)
    assert thresholded.pred_gt.tolist() == [-1, 0, -1, -1], "low scores are not considered"


def test_detections_of_unlabelled_objects_are_ignored_not_punished():
    """An object too occluded to be labelled is an ignore region: finding it is not an error."""
    frame = FrameBoxes.build([GT[0], [300, 300, 320, 320]], [schema.BALL, schema.BALL], [0.9, 0.8],
                             [], [], ignore_boxes=[GT[0]], ignore_classes=[schema.BALL])
    match = metrics.match_frame(frame)
    assert match.pred_ignored.tolist() == [True, False]
    result = metrics.evaluate_frames([frame], schema.DETECTION_CLASSES)
    assert result["per_class"]["ball"]["false_positives"] == 1
    wrong_class = FrameBoxes.build([GT[0]], [schema.ROBOT], [0.9], [], [],
                                   ignore_boxes=[GT[0]], ignore_classes=[schema.BALL])
    assert not metrics.match_frame(wrong_class).pred_ignored.any()


# ------------------------------------------------------------------ average precision
def test_average_precision_known_cases():
    assert metrics.average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert metrics.average_precision([0.9, 0.8], [False, False], 2) == 0.0
    assert metrics.average_precision([], [], 3) == 0.0
    assert np.isnan(metrics.average_precision([0.5], [False], 0))
    # One of two objects found, ranked first: precision 1 up to recall 0.5, nothing beyond.
    assert metrics.average_precision([0.9, 0.3], [True, False], 2) == pytest.approx(0.5, abs=0.01)
    # A false positive ranked above the true positive: precision 0.5 up to recall 1.
    assert metrics.average_precision([0.9, 0.3], [False, True], 1) == pytest.approx(0.5, abs=0.01)
    better = metrics.average_precision([0.9, 0.8, 0.1], [True, True, False], 2)
    worse = metrics.average_precision([0.9, 0.8, 0.1], [False, True, True], 2)
    assert better > worse


def test_perfect_predictions_score_one_and_missing_ones_cost_recall():
    perfect = FrameBoxes.build(GT, GT_CLASSES, [0.9, 0.8, 0.7], GT, GT_CLASSES)
    result = metrics.evaluate_frames([perfect, perfect], schema.DETECTION_CLASSES)
    assert result["map50"] == pytest.approx(1.0) and result["map50_95"] == pytest.approx(1.0)
    assert result["precision"] == 1.0 and result["recall"] == 1.0
    assert result["false_positives_per_image"] == 0.0
    np.testing.assert_array_equal(np.diag(result["confusion_matrix"]["counts"]), [2, 2, 2, 0])

    partial = FrameBoxes.build(GT[:1], GT_CLASSES[:1], [0.9], GT, GT_CLASSES)
    result = metrics.evaluate_frames([partial], schema.DETECTION_CLASSES)
    assert result["recall"] == pytest.approx(1 / 3) and result["precision"] == 1.0
    assert result["per_class"]["goal_post"]["false_negatives"] == 1
    assert result["confusion_matrix"]["counts"][3][:3] == [0, 1, 1], "two objects missed"
    json.dumps(result)


def test_loose_boxes_count_at_iou_50_but_lower_map_50_95():
    loose = FrameBoxes.build([shifted(box, 6) for box in GT], GT_CLASSES, [0.9, 0.8, 0.7],
                             GT, GT_CLASSES)
    result = metrics.evaluate_frames([loose], schema.DETECTION_CLASSES)
    assert result["map50"] == pytest.approx(1.0)
    assert 0.2 < result["map50_95"] < 0.9


def test_confusion_matrix_false_positive_rate_and_operating_threshold():
    frames = [
        FrameBoxes.build([GT[0], GT[1], [400, 300, 420, 330]],
                         [schema.ROBOT, schema.GOAL_POST, schema.GOAL_POST], [0.9, 0.8, 0.6],
                         GT[:2], GT_CLASSES[:2]),
        FrameBoxes.build([[5, 5, 30, 30]], [schema.BALL], [0.1], [], []),      # below threshold
    ]
    result = metrics.evaluate_frames(frames, schema.DETECTION_CLASSES, conf_threshold=0.25)
    counts = np.array(result["confusion_matrix"]["counts"])
    assert counts[schema.ROBOT, schema.BALL] == 1, "a ball called robot is off the diagonal"
    assert counts[schema.GOAL_POST, schema.GOAL_POST] == 1
    assert counts[schema.GOAL_POST, 3] == 1, "a post where there is only background"
    assert counts.sum() == 3, "the low-score box is not at the operating point"
    assert result["per_class"]["goal_post"]["false_positives"] == 1
    assert result["per_class"]["robot"]["false_positives"] == 1
    assert result["false_positives_per_image"] == pytest.approx(1.0)
    assert result["images_with_false_positive"] == pytest.approx(0.5)
    assert result["per_class"]["ball"]["recall"] == 0.0
    assert np.isnan(result["per_class"]["robot"]["ap50"]), "no robots in the ground truth"


# ------------------------------------------------------------------ stratified evaluation
def _record(objects, robots=(None, None, None), mode="random"):
    return {"frame_id": "f", "frame": {"mode": mode, "robots": list(robots)}, "objects": objects}


def _object(class_id, box, distance, visible=1.0, labeled=True, truncated=False, **attributes):
    return {"class_id": class_id, "class_name": schema.DETECTION_CLASSES[class_id],
            "bbox_xyxy": box, "distance_m": distance, "visible_fraction": visible,
            "labeled": labeled, "truncated": truncated, "reason": "ok" if labeled else "occluded",
            "attributes": attributes}


def test_stratified_recall_and_false_positive_analysis():
    from robosoccer_perception.detection import evaluation
    from robosoccer_perception.inference.types import Detection

    near_ball = _object(schema.BALL, [100, 100, 140, 140], 1.0)
    far_ball = _object(schema.BALL, [300, 50, 306, 56], 7.0)
    fallen = _object(schema.ROBOT, [200, 100, 300, 160], 3.0, visible=0.4, appearance_index=1,
                     height=1.2)
    hidden_post = _object(schema.GOAL_POST, [10, 10, 20, 90], 4.0, visible=0.05, labeled=False)
    records = [_record([near_ball, hidden_post]),
               _record([far_ball, fallen], robots=(None, {"fallen": "side"}, None), mode="ball_far")]
    detections = [[Detection(schema.BALL, 0.9, (100, 100, 140, 140)),
                   Detection(schema.GOAL_POST, 0.8, (10, 10, 20, 90)),        # the ignore region
                   Detection(schema.GOAL_POST, 0.7, (500, 0, 520, 200))],     # a real false positive
                  [Detection(schema.ROBOT, 0.6, (200, 100, 300, 160))]]
    frames, objects = zip(*(evaluation.to_frame_boxes(d, r) for d, r in zip(detections, records)))
    rows = evaluation.stratified_recall(list(frames), list(objects), records, 0.25)

    def recall(cls, breakdown, label):
        row, = [r for r in rows if (r["class"], r["breakdown"], r["bin"]) == (cls, breakdown, label)]
        return row["detected"], row["total"]

    assert recall("ball", "all", "all") == (1, 2)
    assert recall("ball", "distance", "near (<2 m)") == (1, 1)
    assert recall("ball", "distance", "far (>5 m)") == (0, 1)
    assert recall("ball", "apparent size (sqrt of box area)", "<8 px") == (0, 1)
    assert recall("robot", "posture", "fallen") == (1, 1)
    assert recall("robot", "occlusion", "heavily occluded (<50 % visible)") == (1, 1)
    assert recall("robot", "robot height", "tall (>=1.0 m)") == (1, 1)
    assert recall("ball", "frame mode", "ball_far") == (0, 1)
    assert not [r for r in rows if r["class"] == "goal_post"], "the unlabelled post is not scored"

    mask = np.full((360, 640), schema.BACKGROUND, np.uint8)
    analysis = evaluation.false_positive_analysis(list(frames), records, lambda record: mask, 0.25)
    assert analysis["goal_post"]["false_positives"] == 1 and analysis["goal_post"]["nothing_there"] == 1
    assert analysis["goal_post"]["nothing_there_centre_on"] == {"background": 1, "field": 0, "field_line": 0}
    # A loose box on the near ball and a "robot" drawn on it are different kinds of mistakes.
    sloppy = [[Detection(schema.BALL, 0.9, (100, 100, 190, 190)),
               Detection(schema.ROBOT, 0.9, (95, 95, 150, 150))], []]
    sloppy_frames = [evaluation.to_frame_boxes(d, r)[0] for d, r in zip(sloppy, records)]
    kinds = evaluation.false_positive_analysis(sloppy_frames, records, lambda record: mask, 0.25)
    assert (kinds["ball"]["badly_placed"], kinds["ball"]["nothing_there"]) == (1, 0)
    assert (kinds["robot"]["other_object"], kinds["robot"]["nothing_there"]) == (1, 0)
    assert analysis["ball"]["false_positives"] == 0 and analysis["robot"]["false_positives"] == 0
    assert analysis["robot"]["images_without_this_class"] == 1


# ------------------------------------------------------------------ configuration
def test_detection_configurations_load():
    pytest.importorskip("ultralytics")
    from robosoccer_perception.detection import yolo

    cfg = yolo.load_detection_config("yolo11n")
    assert cfg["model"] == "yolo11n.pt" and cfg["imgsz"] == 640 and cfg["batch"] <= 16
    assert cfg["patience"] > 0 and cfg["seed"] == 0 and cfg["amp"] is True
    assert cfg["augment"]["flipud"] == 0.0
    smoke = yolo.load_detection_config("yolo11n_smoke", {"epochs": 1})
    assert smoke["epochs"] == 1 and smoke["dataset"].endswith("smoke")
    assert yolo.resolve_model("yolo11n.yaml") == "yolo11n.yaml"
    with pytest.raises(FileNotFoundError, match="--download"):
        yolo.resolve_model("no_such_checkpoint.pt")
    with pytest.raises(ValueError, match="unknown checkpoint"):
        yolo.download_weights("definitely_not_a_model.pt")


@pytest.mark.vision
def test_untrained_detector_loads_and_returns_the_detection_schema(rng):
    """Model loading and the output schema, with a random-weight network (no download)."""
    pytest.importorskip("ultralytics")
    from robosoccer_perception.detection.yolo import YoloDetector
    from robosoccer_perception.inference.types import Detection

    detector = YoloDetector("yolo11n.yaml", device="cpu", conf=0.001, imgsz=160)
    image = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    detections = detector.predict(image)
    batch = detector.predict_batch([image, image])
    assert len(batch) == 2 and isinstance(detections, list)
    for detection in detections:
        assert isinstance(detection, Detection)
        assert 0.0 <= detection.confidence <= 1.0 and detection.class_id >= 0
        x0, y0, x1, y1 = detection.bbox_xyxy
        assert 0 <= x0 < x1 <= 160 and 0 <= y0 < y1 <= 90, "boxes are in input-image pixels"
    assert detector.predict(image, conf=0.999) == []


@pytest.mark.vision
def test_detector_frames_the_image_in_a_grey_margin(rng):
    """The margin is added at network scale, and boxes come back in image coordinates."""
    pytest.importorskip("ultralytics")
    from robosoccer_perception.detection.yolo import YoloDetector

    detector = YoloDetector("yolo11n.yaml", device="cpu", conf=0.001, imgsz=160, margin=16)
    image = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    framed, border = detector._frame(image)
    assert border == 16 and framed.shape == (90 + 32, 160 + 32, 3)
    np.testing.assert_array_equal(framed[16:-16, 16:-16], image[..., ::-1])       # BGR inside
    assert (framed[:16] == 114).all() and (framed[:, -16:] == 114).all()
    double = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)                    # 320 x 180
    assert detector._frame(double)[1] == 32, "the margin scales with the image"
    plain = YoloDetector("yolo11n.yaml", device="cpu", imgsz=160, margin=0)
    unframed, border = plain._frame(image)
    assert border == 0 and unframed.shape == image.shape
