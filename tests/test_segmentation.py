"""Segmentation: metrics (NumPy), model, loss, data pipeline, checkpoints and a training smoke
test. Everything except the metric tests needs PyTorch (``vision`` extra)."""
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.segmentation import metrics  # noqa: E402

B, F, L = schema.BACKGROUND, schema.FIELD, schema.FIELD_LINE


def field_with_line(shift: int = 0, height: int = 40, width: int = 60) -> np.ndarray:
    """Background on top, field below, and a 2 px wide horizontal line at row 25 + shift."""
    mask = np.full((height, width), F, np.uint8)
    mask[:10] = B
    mask[25 + shift:27 + shift] = L
    return mask


# ------------------------------------------------------------------ metrics
def test_confusion_matrix_and_iou_known_values():
    target = np.array([[B, B, F, F], [F, F, L, L]], np.uint8)
    prediction = np.array([[B, F, F, F], [F, L, L, F]], np.uint8)
    matrix = metrics.confusion_matrix(prediction, target, 3)
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [1, 3, 1], [0, 1, 1]])   # [predicted, true]
    scores = metrics.scores_from_confusion(matrix)
    assert scores["per_class"]["background"]["iou"] == pytest.approx(1 / 2)
    assert scores["per_class"]["field"]["iou"] == pytest.approx(3 / 6)
    assert scores["per_class"]["field_line"]["iou"] == pytest.approx(1 / 3)
    assert scores["per_class"]["field_line"]["precision"] == pytest.approx(1 / 2)
    assert scores["per_class"]["field_line"]["recall"] == pytest.approx(1 / 2)
    assert scores["mean_iou"] == pytest.approx((1 / 2 + 1 / 2 + 1 / 3) / 3)
    assert scores["pixel_accuracy"] == pytest.approx(5 / 8)


def test_ignored_pixels_are_not_scored():
    target = np.array([[B, F, schema.IGNORE_INDEX, schema.IGNORE_INDEX]], np.uint8)
    prediction = np.array([[B, F, L, L]], np.uint8)
    matrix = metrics.confusion_matrix(prediction, target, 3)
    assert matrix.sum() == 2 and np.trace(matrix) == 2


def test_pixel_accuracy_hides_missing_lines_and_line_metrics_do_not():
    """Why accuracy is not the headline number: never predicting a line still scores ~95 %."""
    target = field_with_line()
    blind = np.where(target == L, F, target)
    meter = metrics.SegmentationMeter()
    meter.update(blind, target)
    result = meter.compute()
    assert result["pixel_accuracy"] > 0.94
    assert result["line"]["iou"] == 0.0 and result["line"]["recall"] == 0.0
    assert result["line"]["precision"] is None, "no line pixel was predicted"
    assert result["mean_iou"] < 0.67
    json.dumps(result)


def test_tolerant_line_scores_forgive_a_one_pixel_shift_only():
    target = field_with_line()
    exact = metrics.SegmentationMeter(tolerance=2)
    exact.update(target, target)
    assert exact.compute()["line"]["f1"] == 1.0 and exact.compute()["line_tolerant"]["f1"] == 1.0
    near = metrics.SegmentationMeter(tolerance=2)
    near.update(field_with_line(shift=1), target)
    result = near.compute()
    assert result["line"]["iou"] == pytest.approx(1 / 3), "exact scores punish the shift"
    assert result["line_tolerant"]["precision"] == 1.0 and result["line_tolerant"]["recall"] == 1.0
    far = metrics.SegmentationMeter(tolerance=2)
    far.update(field_with_line(shift=8), target)
    result = far.compute()
    assert result["line_tolerant"]["precision"] == 0.0 and result["line_tolerant"]["recall"] == 0.0


def test_meter_accumulates_over_images():
    meter = metrics.SegmentationMeter()
    for shift in (0, 1, 0, 1):
        meter.update(field_with_line(shift), field_with_line())
    result = meter.compute()
    assert result["num_images"] == 4
    assert result["line"]["recall"] == pytest.approx(0.75)
    assert np.sum(result["confusion_matrix"]["counts"]) == 4 * 40 * 60


# ------------------------------------------------------------------ model, loss, data
torch = pytest.importorskip("torch")


@pytest.mark.vision
def test_small_unet_shapes_and_size():
    from robosoccer_perception.segmentation.model import SmallUNet, build_model, count_parameters

    model = build_model({"architecture": "small_unet", "num_classes": 3, "base_channels": 16,
                         "depth": 4}).eval()
    assert isinstance(model, SmallUNet) and model.stride == 16
    assert 1_000_000 < count_parameters(model) < 3_000_000, "a lightweight model"
    with torch.inference_mode():
        for height, width in ((64, 96), (368, 640)):
            logits = model(torch.zeros(1, 3, height, width))
            assert logits.shape == (1, 3, height, width), "one logit per class per pixel"
    with pytest.raises(ValueError, match="architecture"):
        build_model({"architecture": "segformer_b5"})


@pytest.mark.vision
def test_loss_handles_imbalance_and_ignored_pixels():
    from robosoccer_perception.segmentation.data import class_weights_from_counts
    from robosoccer_perception.segmentation.losses import SegmentationLoss

    weights = class_weights_from_counts(np.array([400, 570, 30]))
    assert weights[L] > 3 * weights[F] and weights[F] == pytest.approx(1.0)
    assert class_weights_from_counts(np.array([1, 10**9, 1]))[0] == 10.0, "weights are capped"
    target = torch.from_numpy(field_with_line().astype(np.int64))[None]
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 30.0
    blind = perfect.clone()
    blind[:, L] = -30.0                                  # never predicts a line
    loss = SegmentationLoss(weights, dice_weight=0.5)
    assert float(loss(perfect, target)) < 0.02
    assert float(loss(blind, target)) > 1.0
    unweighted = SegmentationLoss(None, dice_weight=0.0)
    assert float(loss(blind, target)) > 2 * float(unweighted(blind, target)), \
        "class weights and the Dice term make missing the lines expensive"
    # Ignored pixels change nothing, whatever is predicted there.
    padded_target = torch.nn.functional.pad(target, (0, 4), value=schema.IGNORE_INDEX)
    padded_logits = torch.nn.functional.pad(perfect, (0, 4), value=0.0)
    padded_logits[..., -4:] = torch.randn(1, 3, 40, 4) * 10
    assert float(loss(padded_logits, padded_target)) == pytest.approx(float(loss(perfect, target)), abs=1e-4)


@pytest.mark.vision
@pytest.mark.render
def test_segmentation_dataset_keeps_image_and_mask_aligned(tiny_dataset):
    """After random scale, crop and flip, the mask still describes the image it came with.

    A line pixel is bright in these renders, so brightness under the augmented mask is a direct
    check that both went through the same transform."""
    from robosoccer_perception.segmentation.data import SegmentationDataset

    plain = SegmentationDataset(tiny_dataset, "train", stride=16)
    sample = plain[0]
    assert sample["image"].shape == (3, 192, 320) and sample["mask"].shape == (192, 320)
    assert sample["image"].dtype == torch.float32 and sample["mask"].dtype == torch.int64
    assert (sample["mask"][180:] == schema.IGNORE_INDEX).all(), "padding rows are ignored"
    assert set(sample["mask"][:180].unique().tolist()) <= {0, 1, 2}

    augmented = SegmentationDataset(tiny_dataset, "train", augment=True, crop_size=(160, 96),
                                    scale_range=(0.8, 1.3), seed=3)
    a, b = augmented[1], augmented[1]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"]), "deterministic"
    augmented.set_epoch(1)
    assert not torch.equal(a["image"], augmented[1]["image"]), "a new epoch draws new crops"
    assert a["image"].shape == (3, 96, 160) and a["mask"].shape == (96, 160)

    rng = np.random.default_rng(0)
    checked = 0
    for index in range(len(augmented)):
        record = augmented.records[index]
        image, mask = augmented.dataset.image(record), augmented.dataset.mask(record)
        marked = image.copy()
        marked[mask == L] = 255                          # make "line" unmistakable in the image
        marked[mask != L] = np.minimum(marked[mask != L], 120)
        for _ in range(4):
            out_image, out_mask = augmented.augment_pair(marked, mask, rng)
            assert out_image.shape[:2] == out_mask.shape == (96, 160)
            lines = out_mask == L
            core = cv2.erode(lines.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            away = cv2.dilate(lines.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
            if core.sum() >= 5:
                assert out_image[core].mean() > 240, "line pixels of the mask are line in the image"
                assert (out_image[away] <= 125).mean() > 0.995, "and nothing else is"
                checked += 1
    assert checked >= 5
    with pytest.raises(ValueError, match="multiple"):
        SegmentationDataset(tiny_dataset, "train", augment=True, crop_size=(150, 100))
    with pytest.raises(ValueError, match="no frames"):
        SegmentationDataset(tiny_dataset, "train", limit=0)


@pytest.mark.vision
def test_checkpoint_round_trip_and_segmenter_output_schema(tmp_path, rng):
    from robosoccer_perception.inference.types import Segmentation
    from robosoccer_perception.segmentation.data import DEFAULT_NORMALIZATION
    from robosoccer_perception.segmentation.model import build_model
    from robosoccer_perception.segmentation.segmenter import (
        Segmenter, load_checkpoint, save_checkpoint)

    model_cfg = {"architecture": "small_unet", "num_classes": 3, "base_channels": 8, "depth": 3}
    torch.manual_seed(0)
    model = build_model(model_cfg).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, model_cfg, DEFAULT_NORMALIZATION, train_width=320,
                    extra={"epoch": 4})
    loaded, metadata = load_checkpoint(path)
    assert metadata["train_width"] == 320 and metadata["epoch"] == 4
    assert metadata["classes"] == list(schema.SEMANTIC_CLASSES)
    probe = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        assert torch.allclose(model(probe), loaded(probe))

    segmenter = Segmenter(path, device="cpu")
    for height, width in ((180, 320), (175, 320), (360, 640), (90, 150)):   # odd sizes, other scales
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        result = segmenter.predict(image)
        assert isinstance(result, Segmentation)
        assert result.mask.shape == (height, width) and result.mask.dtype == np.uint8
        assert result.probabilities.shape == (3, height, width)
        np.testing.assert_allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-4)
        np.testing.assert_array_equal(result.mask, result.probabilities.argmax(axis=0))
        assert result.line_mask.shape == result.field_mask.shape == (height, width)
        assert result.line_confidence.min() >= 0.0 and result.line_confidence.max() <= 1.0
    # The same picture at twice the size gives (nearly) the same answer: it is resized to the
    # training width first.
    small = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    large = cv2.resize(small, (640, 360), interpolation=cv2.INTER_NEAREST)
    agreement = (cv2.resize(segmenter.predict(large).mask, (320, 180), interpolation=cv2.INTER_NEAREST)
                 == segmenter.predict(small).mask).mean()
    assert agreement > 0.9

    wrong = tmp_path / "wrong.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["classes"] = ["sky", "grass", "paint"]
    torch.save(checkpoint, wrong)
    with pytest.raises(ValueError, match="classes"):
        Segmenter(wrong, device="cpu")


@pytest.mark.vision
def test_segmentation_configurations_load():
    from robosoccer_perception.segmentation.trainer import load_segmentation_config

    cfg = load_segmentation_config("unet_small")
    assert cfg["model"]["architecture"] == "small_unet" and cfg["model"]["num_classes"] == 3
    assert cfg["monitor"] == "line_iou" and cfg["class_weights"] == "auto" and cfg["amp"] is True
    assert cfg["batch"] <= 8 and all(side % 16 == 0 for side in cfg["augment"]["crop_size"])
    assert load_segmentation_config("unet_smoke", {"epochs": 1})["epochs"] == 1
    with pytest.raises(ValueError, match="monitor"):
        load_segmentation_config("unet_small", {"monitor": "pixel_accuracy"})


@pytest.mark.vision
@pytest.mark.render
@pytest.mark.slow
def test_segmentation_training_smoke(tiny_dataset, tmp_path):
    """Two epochs on 12 frames: the loop runs, the loss falls, checkpoints load and predict."""
    from robosoccer_perception.segmentation.evaluation import evaluate_segmenter
    from robosoccer_perception.segmentation.segmenter import Segmenter
    from robosoccer_perception.segmentation.trainer import (
        load_segmentation_config, train_segmentation)

    cfg = load_segmentation_config("unet_smoke", {
        "name": "tiny", "epochs": 4, "batch": 4, "device": "cpu", "workers": 0,
        "augment": {"crop_size": [160, 96], "scale_range": [0.9, 1.1]}})
    summary = train_segmentation(cfg, tiny_dataset, tmp_path, progress=None)
    run = tmp_path / "tiny"
    history = json.loads((run / "history.json").read_text())
    assert len(history) == summary["epochs_run"] == 4
    assert all(np.isfinite(row["train_loss"]) for row in history)
    assert history[-1]["train_loss"] < history[0]["train_loss"], "the loss decreases"
    for name in ("best.pt", "last.pt", "history.csv", "train_summary.json"):
        assert (run / name).is_file()
    segmenter = Segmenter(run / "best.pt", device="cpu")
    assert segmenter.train_width == 320 and segmenter.metadata["dataset_name"] == "tiny"
    result = evaluate_segmenter(segmenter, tiny_dataset, "test", tmp_path / "eval",
                                max_examples=2, latency_repeats=3)
    assert result["num_images"] == 4 and 0.0 <= result["mean_iou"] <= 1.0
    assert {"iou", "precision", "recall", "f1"} <= set(result["line"])
    assert [row["bin"] for row in result["line_by_distance"]][0] == "<2 m"
    assert result["latency"]["count"] == 3 and result["latency"]["mean_ms"] > 0
    assert (tmp_path / "eval" / "metrics.json").is_file()
    assert list((tmp_path / "eval" / "examples" / "typical").glob("*.jpg"))
