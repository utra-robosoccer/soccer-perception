"""Test doubles for Phase 3 that need neither MuJoCo nor trained networks.

`analytic_view` paints what the camera would see of the field's lines directly from the field
geometry and the camera model (white paint on green turf, grey elsewhere), and also returns
the matching "perfect" predictions. `ColourVision` is a tiny classical stand-in for the
networks: it sees ONLY the image and labels pixels by colour. Together they let the geometry
and the whole localization pipeline be tested end to end, deterministically, in a few
milliseconds per frame.

Everything a test knows about the true pose stays in the test: the pipeline under test is
handed the image and the camera estimate, exactly as in the real benchmark.
"""
from __future__ import annotations

import math

import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ProjectionLimits, project_to_ground)
from robosoccer_perception.inference.types import Detection, Segmentation, VisionResult

TURF_RGB, LINE_RGB, BACKGROUND_RGB = (30, 110, 40), (235, 235, 235), (80, 80, 85)
_EVERYTHING = ProjectionLimits(min_depression_deg=0.2, max_range=60.0, min_range=0.0,
                               image_margin_px=0.0)


def small_calibration(width: int = 320, height: int = 180, **uncertainty
                      ) -> EstimatedCameraCalibration:
    """The nominal camera at a small image size (same field of view)."""
    from robosoccer_perception.geometry.ground_projection import CalibrationUncertainty

    cfg = StereoCameraConfig.load("zed_mini_sim").with_resolution(width, height)
    trust = CalibrationUncertainty(**uncertainty) if uncertainty else None
    return EstimatedCameraCalibration.from_camera_config(cfg, uncertainty=trust)


def semantic_mask(field: FieldGeometry, base_pose, calibration: EstimatedCameraCalibration,
                  camera_pose: EstimatedCameraPose) -> np.ndarray:
    """Class of every pixel for a robot at ``base_pose`` whose camera really is at
    ``camera_pose``: line where the pixel centre's ray lands on paint, field where it lands on
    the carpet, background elsewhere (off the carpet, or above the horizon)."""
    k = calibration.intrinsics
    v, u = np.mgrid[0:k.height, 0:k.width]
    ground = project_to_ground(np.column_stack([u.ravel(), v.ravel()]), calibration,
                               camera_pose, _EVERYTHING)
    mask = np.full(k.height * k.width, schema.BACKGROUND, np.uint8)
    seen = ground.valid
    in_field = tf.se2_transform_points(np.asarray(base_pose, float), ground.points[seen])
    on_carpet = field.contains(in_field)
    classes = np.where(on_carpet, schema.FIELD, schema.BACKGROUND).astype(np.uint8)
    classes[on_carpet & (field.distance_to_paint(in_field) <= 0.0)] = schema.FIELD_LINE
    mask[seen] = classes
    return mask.reshape(k.height, k.width)


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    palette = np.array([BACKGROUND_RGB, TURF_RGB, LINE_RGB], np.uint8)
    return palette[mask]


def mask_to_vision(mask: np.ndarray, detections=()) -> VisionResult:
    """Perfect predictions for a mask: one-hot probabilities."""
    probabilities = np.eye(3, dtype=np.float32)[mask].transpose(2, 0, 1)
    return VisionResult(mask.shape[1], mask.shape[0], tuple(detections),
                        Segmentation(mask.copy(), probabilities), {}, "analytic")


def analytic_view(field: FieldGeometry, base_pose, calibration: EstimatedCameraCalibration,
                  camera_pose: EstimatedCameraPose) -> tuple[np.ndarray, VisionResult]:
    """``(image, perfect predictions)`` for a robot at ``base_pose``."""
    mask = semantic_mask(field, base_pose, calibration, camera_pose)
    return mask_to_image(mask), mask_to_vision(mask)


class ColourVision:
    """A classical "network": bright, colourless pixels are line; green-dominant pixels are
    field; the rest is background. It receives the image and nothing else."""
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def run(self, image_rgb: np.ndarray, source: str | None = None) -> VisionResult:
        if image_rgb.ndim != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("expected an (H, W, 3) uint8 RGB image")
        self.calls += 1
        rgb = image_rgb.astype(np.int16)
        bright = (rgb.min(axis=2) > 170) & (np.ptp(rgb, axis=2) < 40)
        green = (rgb[..., 1] > rgb[..., 0] + 15) & (rgb[..., 1] > rgb[..., 2] + 15)
        mask = np.full(image_rgb.shape[:2], schema.BACKGROUND, np.uint8)
        mask[green] = schema.FIELD
        mask[bright] = schema.FIELD_LINE
        result = mask_to_vision(mask)
        return VisionResult(result.width, result.height, (), result.segmentation,
                            {"detector": 0.01, "segmenter": 0.01, "total": 0.02}, source)

    def describe(self) -> dict:
        return {"detector": None, "segmenter": "colour thresholds (test double)", "device": "cpu"}


def post_detection(post_field, base_pose, calibration: EstimatedCameraCalibration,
                   camera_pose: EstimatedCameraPose, diameter: float = 0.10, height: float = 1.25,
                   confidence: float = 0.9) -> Detection | None:
    """The tight box around the visible part of a vertical post standing at ``post_field``
    (field frame), as a detector trained on rendered labels would return it. None if the post
    is out of view."""
    k = calibration.intrinsics
    post_base = tf.se2_inverse_transform_points(np.asarray(base_pose, float),
                                                np.asarray(post_field, float)[None])[0]
    angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    heights = np.linspace(0.0, height, 60)
    ring = 0.5 * diameter * np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.array([[post_base[0] + dx, post_base[1] + dy, z]
                       for z in heights for dx, dy in ring])
    in_camera = tf.transform_points(tf.invert_transform(camera_pose.T_base_camera), points)
    pixels, depth = k.project(in_camera)
    visible = (depth > 0.05) & k.in_image(pixels)
    if visible.sum() < 10:
        return None
    u, v = pixels[visible, 0], pixels[visible, 1]
    # Pixel-centre coordinates -> the pixel-edge box around the pixels touched.
    box = (float(np.floor(u.min() + 0.5)), float(np.floor(v.min() + 0.5)),
           float(np.floor(u.max() + 0.5) + 1.0), float(np.floor(v.max() + 0.5) + 1.0))
    return Detection(schema.GOAL_POST, confidence, box)
