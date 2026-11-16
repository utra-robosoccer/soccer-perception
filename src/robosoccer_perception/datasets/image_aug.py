"""Image-space augmentation.

Two kinds, kept apart on purpose:

* PHOTOMETRIC operations (`apply_photometric`) change pixel values only: exposure, gamma,
  white balance, saturation, vignetting, blur, sensor noise, JPEG artefacts. Labels are not
  touched. Blur is kept mild so that object extents stay within about a pixel of their boxes.
* GEOMETRIC operations (`hflip`, `crop`, `resize`, `pad_to_multiple`) move pixels, and every
  one of them takes the mask and the boxes along and returns them transformed.

Boxes are ``(N, 4)`` arrays of ``x0, y0, x1, y1`` in pixel-edge coordinates (the image spans
``[0, W] x [0, H]``), the convention of `datasets.schema.ObjectAnnotation`.

Needs OpenCV (the ``vision`` extra).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from robosoccer_perception.datasets import schema

PHOTOMETRIC_KEYS = ("vignette", "exposure_gain", "white_balance", "gamma", "saturation",
                    "defocus_sigma", "motion_blur", "noise_std", "noise_seed", "jpeg_quality")


# ------------------------------------------------------------------ photometric
def sample_photometric(rng: np.random.Generator, cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Draw one set of photometric parameters from the ``image`` section of a randomization
    preset. An empty or disabled section gives ``{}``, which `apply_photometric` treats as
    "leave the image alone"."""
    if not cfg or not cfg.get("enabled", True):
        return {}

    def uniform(key: str, default: float) -> float:
        low, high = cfg.get(key, (default, default))
        return float(rng.uniform(low, high))

    balance = float(cfg.get("white_balance", 0.0))
    params: dict[str, Any] = {
        "vignette": uniform("vignette", 0.0),
        "exposure_gain": uniform("exposure_gain", 1.0),
        "white_balance": [float(v) for v in 1.0 + rng.uniform(-balance, balance, 3)],
        "gamma": uniform("gamma", 1.0),
        "saturation": uniform("saturation", 1.0),
        "defocus_sigma": uniform("defocus_sigma", 0.0),
        "motion_blur": None,
        "noise_std": uniform("noise_std", 0.0),
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
        "jpeg_quality": None,
    }
    if rng.random() < float(cfg.get("motion_blur_prob", 0.0)):
        low, high = cfg.get("motion_blur_px", (3, 5))
        params["motion_blur"] = {"length_px": int(rng.integers(int(low), int(high) + 1)),
                                 "angle_deg": float(rng.uniform(0.0, 180.0))}
    if rng.random() < float(cfg.get("jpeg_prob", 0.0)):
        low, high = cfg.get("jpeg_quality", (60, 95))
        params["jpeg_quality"] = int(rng.integers(int(low), int(high) + 1))
    return params


def motion_kernel(length_px: int, angle_deg: float) -> np.ndarray:
    """Normalised line kernel: linear motion of ``length_px`` pixels at ``angle_deg``."""
    size = int(length_px) | 1                      # odd
    kernel = np.zeros((size, size), np.float32)
    centre = (size - 1) / 2.0
    dx, dy = math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))
    start = (int(round(centre - dx * centre)), int(round(centre - dy * centre)))
    end = (int(round(centre + dx * centre)), int(round(centre + dy * centre)))
    cv2.line(kernel, start, end, 1.0, 1)
    return kernel / kernel.sum()


def apply_photometric(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply photometric parameters to an RGB uint8 image. Returns a new uint8 image of the
    same shape. Deterministic: the noise is drawn from ``params['noise_seed']``."""
    if not params:
        return image.copy()
    height, width = image.shape[:2]
    out = image.astype(np.float32) / 255.0
    if params.get("vignette", 0.0) > 0.0:
        v, u = np.mgrid[0:height, 0:width].astype(np.float32)
        r2 = ((u - (width - 1) / 2) ** 2 + (v - (height - 1) / 2) ** 2) \
            / ((width / 2) ** 2 + (height / 2) ** 2)
        out *= (1.0 - float(params["vignette"]) * r2)[..., None]
    out *= float(params.get("exposure_gain", 1.0))
    out *= np.asarray(params.get("white_balance", (1.0, 1.0, 1.0)), np.float32)
    out = np.clip(out, 0.0, 1.0)
    gamma = float(params.get("gamma", 1.0))
    if gamma != 1.0:
        out = out ** gamma
    saturation = float(params.get("saturation", 1.0))
    if saturation != 1.0:
        grey = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = np.clip(grey[..., None] + saturation * (out - grey[..., None]), 0.0, 1.0)
    sigma = float(params.get("defocus_sigma", 0.0))
    if sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    blur = params.get("motion_blur")
    if blur:
        out = cv2.filter2D(out, -1, motion_kernel(blur["length_px"], blur["angle_deg"]),
                           borderType=cv2.BORDER_REPLICATE)
    noise_std = float(params.get("noise_std", 0.0))
    if noise_std > 0.0:
        noise_rng = np.random.default_rng(int(params.get("noise_seed", 0)))
        # Shot-noise-like: more noise in brighter pixels, plus a constant read-noise floor.
        scale = (noise_std / 255.0) * np.sqrt(0.25 + 0.75 * out)
        out = out + noise_rng.standard_normal(out.shape, dtype=np.float32) * scale
    result = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    quality = params.get("jpeg_quality")
    if quality:
        result = jpeg_roundtrip(result, int(quality))
    return result


def jpeg_roundtrip(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Compress to JPEG at ``quality`` and decode again (compression artefacts)."""
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ geometric
def _as_boxes(boxes) -> np.ndarray:
    return np.zeros((0, 4), np.float32) if boxes is None or len(boxes) == 0 \
        else np.asarray(boxes, np.float32).reshape(-1, 4)


def hflip(image: np.ndarray, mask: np.ndarray | None = None, boxes=None):
    """Mirror left-right. Returns ``(image, mask, boxes)``."""
    width = image.shape[1]
    boxes = _as_boxes(boxes)
    flipped = boxes.copy()
    flipped[:, 0], flipped[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    return (np.ascontiguousarray(image[:, ::-1]),
            None if mask is None else np.ascontiguousarray(mask[:, ::-1]), flipped)


def crop(image: np.ndarray, mask: np.ndarray | None, boxes, x0: int, y0: int,
         width: int, height: int, min_side_px: float = 2.0):
    """Cut out the window ``[x0, x0 + width) x [y0, y0 + height)``.

    Boxes are clipped to the window; boxes left narrower or lower than ``min_side_px`` are
    dropped. Returns ``(image, mask, boxes, kept)`` with ``kept`` the indices of surviving boxes.
    """
    if x0 < 0 or y0 < 0 or x0 + width > image.shape[1] or y0 + height > image.shape[0]:
        raise ValueError("crop window is not inside the image")
    boxes = _as_boxes(boxes) - np.array([x0, y0, x0, y0], np.float32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    kept = np.flatnonzero((boxes[:, 2] - boxes[:, 0] >= min_side_px)
                          & (boxes[:, 3] - boxes[:, 1] >= min_side_px))
    return (np.ascontiguousarray(image[y0:y0 + height, x0:x0 + width]),
            None if mask is None else np.ascontiguousarray(mask[y0:y0 + height, x0:x0 + width]),
            boxes[kept], kept)


def resize(image: np.ndarray, mask: np.ndarray | None, boxes, width: int, height: int):
    """Resize to ``width x height``. The image is interpolated (area when shrinking, linear when
    enlarging); the mask uses nearest neighbour so class ids are never blended.

    The mask uses ``INTER_NEAREST_EXACT``, which samples at pixel centres like the image
    interpolation does. Plain ``INTER_NEAREST`` samples at pixel corners and would leave the
    mask up to half a source pixel off the image."""
    sx, sy = width / image.shape[1], height / image.shape[0]
    interpolation = cv2.INTER_AREA if sx < 1.0 or sy < 1.0 else cv2.INTER_LINEAR
    boxes = _as_boxes(boxes) * np.array([sx, sy, sx, sy], np.float32)
    return (cv2.resize(image, (width, height), interpolation=interpolation),
            None if mask is None else cv2.resize(mask, (width, height),
                                                 interpolation=cv2.INTER_NEAREST_EXACT), boxes)


def pad_to_multiple(image: np.ndarray, mask: np.ndarray | None, multiple: int):
    """Pad the bottom and right edges so both sides are multiples of ``multiple``.

    The image is padded by edge replication and the mask with `schema.IGNORE_INDEX`. Boxes need
    no change (the origin does not move). Returns ``(image, mask, (pad_bottom, pad_right))``.
    """
    height, width = image.shape[:2]
    pad_bottom, pad_right = (-height) % multiple, (-width) % multiple
    if pad_bottom == 0 and pad_right == 0:
        return image, mask, (0, 0)
    image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    if mask is not None:
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT,
                                  value=schema.IGNORE_INDEX)
    return image, mask, (pad_bottom, pad_right)
