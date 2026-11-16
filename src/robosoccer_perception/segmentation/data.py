"""Training data for segmentation: images and masks of a generated dataset as tensors.

Every geometric augmentation goes through `datasets.image_aug`, which moves the mask with the
image, so pixel alignment is preserved by construction (and checked in the tests).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from robosoccer_perception.datasets import image_aug, schema
from robosoccer_perception.datasets.io import SoccerDataset

DEFAULT_NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}


def image_to_tensor(image_rgb: np.ndarray, normalization: dict[str, Any] | None = None) -> torch.Tensor:
    """``(H, W, 3)`` uint8 RGB -> ``(3, H, W)`` float32, normalised."""
    normalization = normalization or DEFAULT_NORMALIZATION
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(normalization["mean"]).view(3, 1, 1)
    std = torch.tensor(normalization["std"]).view(3, 1, 1)
    return (tensor - mean) / std


class SegmentationDataset(Dataset):
    """One split of a generated dataset.

    With ``augment`` on, each sample is randomly rescaled, cropped to ``crop_size`` and
    mirrored. With it off the whole image is returned, padded at the bottom and right to a
    multiple of ``stride`` (padding is `schema.IGNORE_INDEX` in the mask, so it is not scored).
    """

    def __init__(self, dataset_root, split: str, augment: bool = False,
                 crop_size: tuple[int, int] = (512, 352), scale_range: tuple[float, float] = (0.8, 1.25),
                 flip_prob: float = 0.5, stride: int = 16,
                 normalization: dict[str, Any] | None = None, seed: int = 0,
                 limit: int | None = None):
        self.dataset = SoccerDataset(dataset_root)
        self.records = self.dataset.frames(split)[:limit]
        if not self.records:
            raise ValueError(f"split '{split}' of {dataset_root} has no frames")
        self.augment, self.crop_size, self.scale_range = augment, tuple(crop_size), tuple(scale_range)
        self.flip_prob, self.stride = flip_prob, stride
        self.normalization = normalization or DEFAULT_NORMALIZATION
        self.seed, self.epoch = seed, 0
        if augment and (self.crop_size[0] % stride or self.crop_size[1] % stride):
            raise ValueError(f"crop_size {self.crop_size} must be a multiple of {stride}")

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Changes the augmentation draws; call once per epoch for reproducible training."""
        self.epoch = int(epoch)

    def augment_pair(self, image: np.ndarray, mask: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        crop_w, crop_h = self.crop_size
        scale = float(rng.uniform(*self.scale_range))
        # Never scale below the crop window.
        scale = max(scale, crop_w / image.shape[1], crop_h / image.shape[0])
        width, height = int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))
        if (width, height) != (image.shape[1], image.shape[0]):
            image, mask, _ = image_aug.resize(image, mask, None, width, height)
        x0 = int(rng.integers(0, width - crop_w + 1))
        y0 = int(rng.integers(0, height - crop_h + 1))
        image, mask, _, _ = image_aug.crop(image, mask, None, x0, y0, crop_w, crop_h)
        if rng.random() < self.flip_prob:
            image, mask, _ = image_aug.hflip(image, mask)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, mask = self.dataset.image(record), self.dataset.mask(record)
        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            image, mask = self.augment_pair(image, mask, rng)
        else:
            image, mask, _ = image_aug.pad_to_multiple(image, mask, self.stride)
        return {"image": image_to_tensor(image, self.normalization),
                "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
                "index": index}


def class_pixel_counts(dataset: SoccerDataset, split: str = "train") -> np.ndarray:
    """Pixels per semantic class in a split, from the manifest (no image is opened)."""
    counts = np.zeros(len(schema.SEMANTIC_CLASSES), dtype=np.int64)
    for record in dataset.frames(split):
        for index, name in enumerate(schema.SEMANTIC_CLASSES):
            counts[index] += record["semantic_pixels"][name]
    return counts


def class_weights_from_counts(counts: np.ndarray, power: float = 0.5, cap: float = 10.0) -> list[float]:
    """Loss weights that grow as a class gets rarer: ``(max_share / share) ** power``, capped.

    ``power`` 0.5 is the usual compromise between ignoring the imbalance (0) and weighting by
    inverse frequency (1), which over-predicts thin lines.
    """
    share = counts / max(counts.sum(), 1)
    weights = (share.max() / np.maximum(share, 1e-6)) ** power
    return [float(min(w, cap)) for w in weights]
