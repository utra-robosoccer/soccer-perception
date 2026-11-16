"""Image-space augmentation: photometric operations leave geometry alone, geometric operations
move the mask and the boxes together with the image."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from robosoccer_perception.datasets import image_aug, schema  # noqa: E402
from robosoccer_perception.datasets.randomization import load_preset  # noqa: E402

BOXES = np.array([[10.0, 20.0, 40.0, 60.0], [100.0, 5.0, 130.0, 25.0]], np.float32)


def scene_with_boxes(width=160, height=90):
    """An image, a mask and boxes that agree: each box is a filled rectangle of ``field_line``
    in the mask and a bright rectangle in the image."""
    image = np.full((height, width, 3), 40, np.uint8)
    mask = np.full((height, width), schema.FIELD, np.uint8)
    for x0, y0, x1, y1 in BOXES.astype(int):
        image[y0:y1, x0:x1] = 230
        mask[y0:y1, x0:x1] = schema.FIELD_LINE
    return image, mask


def boxes_from(mask):
    """Recover the boxes from a mask (connected components of the line class)."""
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask == schema.FIELD_LINE).astype(np.uint8))
    return sorted([float(x), float(y), float(x + w), float(y + h)] for x, y, w, h, _ in stats[1:count])


# ------------------------------------------------------------------ photometric
def test_no_parameters_means_no_change(rng):
    image = rng.integers(0, 256, (36, 64, 3), dtype=np.uint8)
    out = image_aug.apply_photometric(image, {})
    np.testing.assert_array_equal(out, image)
    assert out is not image
    assert image_aug.sample_photometric(rng, {"enabled": False}) == {}
    assert image_aug.sample_photometric(rng, None) == {}


@pytest.mark.parametrize("preset", ["mild", "strong"])
def test_photometric_is_deterministic_and_keeps_the_shape(rng, preset):
    cfg = load_preset(preset)["image"]
    image = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    for _ in range(10):
        params = image_aug.sample_photometric(rng, cfg)
        assert set(params) == set(image_aug.PHOTOMETRIC_KEYS)
        first, second = image_aug.apply_photometric(image, params), image_aug.apply_photometric(image, params)
        assert first.shape == image.shape and first.dtype == np.uint8
        np.testing.assert_array_equal(first, second)


def test_photometric_effects_do_what_they_say():
    image = np.full((60, 80, 3), 100, np.uint8)
    brighter = image_aug.apply_photometric(image, {"exposure_gain": 1.3})
    assert brighter.mean() == pytest.approx(130, abs=1)
    warm = image_aug.apply_photometric(image, {"white_balance": [1.1, 1.0, 0.9]})
    assert warm[..., 0].mean() > warm[..., 1].mean() > warm[..., 2].mean()
    noisy_a = image_aug.apply_photometric(image, {"noise_std": 5.0, "noise_seed": 1})
    noisy_b = image_aug.apply_photometric(image, {"noise_std": 5.0, "noise_seed": 2})
    assert 2.0 < noisy_a.astype(float).std() < 8.0 and not np.array_equal(noisy_a, noisy_b)
    vignetted = image_aug.apply_photometric(image, {"vignette": 0.3})
    assert vignetted[0, 0, 0] < vignetted[30, 40, 0] <= 100


def test_blur_and_compression_stay_local():
    """Mild blur and JPEG artefacts move no object: a bright square stays where it is."""
    image, mask = scene_with_boxes()
    params = {"defocus_sigma": 1.0, "motion_blur": {"length_px": 5, "angle_deg": 30.0},
              "jpeg_quality": 50}
    out = image_aug.apply_photometric(image, params)
    bright = out.mean(axis=-1) > 135
    recovered = boxes_from(np.where(bright, schema.FIELD_LINE, schema.FIELD).astype(np.uint8))
    np.testing.assert_allclose(recovered, sorted(BOXES.tolist()), atol=2.0)
    kernel = image_aug.motion_kernel(5, 30.0)
    assert kernel.sum() == pytest.approx(1.0) and kernel.shape == (5, 5)


# ------------------------------------------------------------------ geometric
def test_hflip_moves_mask_and_boxes_with_the_image():
    image, mask = scene_with_boxes()
    flipped_image, flipped_mask, flipped_boxes = image_aug.hflip(image, mask, BOXES)
    assert boxes_from(flipped_mask) == sorted(flipped_boxes.tolist())
    np.testing.assert_array_equal(flipped_image[..., 0] > 128, flipped_mask == schema.FIELD_LINE)
    np.testing.assert_allclose(flipped_boxes[0], [160 - 40, 20, 160 - 10, 60])
    back_image, back_mask, back_boxes = image_aug.hflip(flipped_image, flipped_mask, flipped_boxes)
    np.testing.assert_array_equal(back_image, image)
    np.testing.assert_array_equal(back_mask, mask)
    np.testing.assert_allclose(back_boxes, BOXES)


def test_crop_shifts_clips_and_drops_boxes():
    image, mask = scene_with_boxes()
    out_image, out_mask, boxes, kept = image_aug.crop(image, mask, BOXES, x0=20, y0=10, width=100, height=70)
    assert out_image.shape == (70, 100, 3) and out_mask.shape == (70, 100)
    # Box 0 loses its left 10 px; box 1 loses its top 5 px and all but 20 px of its width.
    np.testing.assert_allclose(boxes, [[0, 10, 20, 50], [80, 0, 100, 15]])
    assert kept.tolist() == [0, 1]
    assert boxes_from(out_mask) == sorted(boxes.tolist())
    np.testing.assert_array_equal(out_image[..., 0] > 128, out_mask == schema.FIELD_LINE)
    # A window that leaves a 1 px sliver of box 0 drops it.
    _, _, boxes, kept = image_aug.crop(image, mask, BOXES, x0=39, y0=0, width=60, height=90)
    assert kept.tolist() == []
    with pytest.raises(ValueError):
        image_aug.crop(image, mask, BOXES, x0=100, y0=0, width=100, height=50)


def test_resize_scales_boxes_and_never_blends_class_ids():
    image, mask = scene_with_boxes()
    mask[0:5, 0:5] = schema.BACKGROUND
    for width, height in ((320, 180), (80, 45), (200, 60)):
        out_image, out_mask, boxes = image_aug.resize(image, mask, BOXES, width, height)
        assert out_image.shape == (height, width, 3) and out_mask.shape == (height, width)
        assert set(np.unique(out_mask)) <= {schema.BACKGROUND, schema.FIELD, schema.FIELD_LINE}
        np.testing.assert_allclose(boxes, BOXES * np.array([width / 160, height / 90] * 2))
        np.testing.assert_allclose(boxes_from(out_mask), sorted(boxes.tolist()), atol=1.0)


def test_pad_to_multiple_marks_the_padding_as_ignore():
    image, mask = scene_with_boxes(width=150, height=90)
    out_image, out_mask, (bottom, right) = image_aug.pad_to_multiple(image, mask, 16)
    assert out_image.shape[:2] == out_mask.shape == (96, 160) and (bottom, right) == (6, 10)
    np.testing.assert_array_equal(out_image[:90, :150], image)
    np.testing.assert_array_equal(out_mask[:90, :150], mask)
    assert (out_mask[90:] == schema.IGNORE_INDEX).all() and (out_mask[:, 150:] == schema.IGNORE_INDEX).all()
    same_image, same_mask, pads = image_aug.pad_to_multiple(out_image, out_mask, 16)
    assert pads == (0, 0) and same_image is out_image


def test_chained_geometric_transforms_keep_labels_aligned(rng):
    """Random flips, crops and resizes in sequence: the mask's rectangles stay the boxes."""
    for _ in range(20):
        image, mask = scene_with_boxes()
        boxes = BOXES.copy()
        if rng.random() < 0.5:
            image, mask, boxes = image_aug.hflip(image, mask, boxes)
        x0, y0 = int(rng.integers(0, 30)), int(rng.integers(0, 15))
        image, mask, boxes, _ = image_aug.crop(image, mask, boxes, x0, y0, 120, 70)
        image, mask, boxes = image_aug.resize(image, mask, boxes, 240, 140)
        np.testing.assert_allclose(boxes_from(mask), sorted(boxes.tolist()), atol=1.0)
        agreement = ((image[..., 0] > 128) == (mask == schema.FIELD_LINE)).mean()
        assert agreement > 0.99
