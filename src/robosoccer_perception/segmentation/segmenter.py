"""A trained segmentation checkpoint behind the project's segmenter interface.

``Segmenter.predict(image_rgb)`` takes an ``(H, W, 3)`` uint8 RGB image of any size and
returns an `inference.types.Segmentation` at that image's resolution.

Scale handling: a fully convolutional network works at any size, but it has only learnt what
lines look like at the scale it was trained on. The checkpoint therefore records the image
width it was trained at, and wider or narrower inputs are resized to that width before the
network and the class probabilities are resized back afterwards. Pass ``native_scale=True`` to
skip this and run on the pixels as given.

A checkpoint is a plain ``torch.save`` dictionary of tensors, numbers, strings and lists
(loadable with ``weights_only=True``): model configuration, weights, normalisation, class
names, training width, and the metrics it reached.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from robosoccer_perception.datasets import image_aug, schema
from robosoccer_perception.inference.profiling import resolve_device
from robosoccer_perception.inference.types import Segmentation
from robosoccer_perception.segmentation.data import image_to_tensor
from robosoccer_perception.segmentation.model import build_model

CHECKPOINT_VERSION = 1


def save_checkpoint(path, model: torch.nn.Module, model_cfg: dict[str, Any],
                    normalization: dict[str, Any], train_width: int,
                    extra: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_version": CHECKPOINT_VERSION, "model": dict(model_cfg),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "normalization": normalization, "classes": list(schema.SEMANTIC_CLASSES),
                "train_width": int(train_width), **(extra or {})}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Returns the model (in eval mode, on ``device``) and the checkpoint's metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != schema.SEMANTIC_CLASSES:
        raise ValueError(f"{path} has classes {checkpoint['classes']}, but this project uses "
                         f"{schema.SEMANTIC_CLASSES}")
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint.pop("state_dict"))
    return model.to(device).eval(), checkpoint


class Segmenter:
    def __init__(self, checkpoint, device: str = "auto", half: bool = False,
                 native_scale: bool = False):
        self.device = resolve_device(device)
        self.model, self.metadata = load_checkpoint(checkpoint, self.device)
        self.checkpoint = str(checkpoint)
        self.normalization = self.metadata["normalization"]
        self.train_width = int(self.metadata["train_width"])
        self.native_scale = native_scale
        self.half = bool(half and self.device.startswith("cuda"))
        self.stride = getattr(self.model, "stride", 16)

    @torch.inference_mode()
    def predict_probabilities(self, image_rgb: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` float32 class probabilities at the input image's resolution."""
        height, width = image_rgb.shape[:2]
        work = image_rgb
        if not self.native_scale and width != self.train_width:
            scaled_height = max(self.stride, int(round(height * self.train_width / width)))
            work, _, _ = image_aug.resize(image_rgb, None, None, self.train_width, scaled_height)
        work_height, work_width = work.shape[:2]
        padded, _, _ = image_aug.pad_to_multiple(work, None, self.stride)
        batch = image_to_tensor(padded, self.normalization).unsqueeze(0).to(self.device)
        with torch.autocast(self.device.split(":")[0], enabled=self.half):
            logits = self.model(batch)
        probabilities = torch.softmax(logits.float(), dim=1)[0, :, :work_height, :work_width]
        probabilities = probabilities.cpu().numpy()
        if (work_height, work_width) != (height, width):
            probabilities = np.stack([cv2.resize(p, (width, height), interpolation=cv2.INTER_LINEAR)
                                      for p in probabilities])
            probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-6)
        return probabilities

    def predict(self, image_rgb: np.ndarray) -> Segmentation:
        probabilities = self.predict_probabilities(image_rgb)
        return Segmentation(probabilities.argmax(axis=0).astype(np.uint8), probabilities)
