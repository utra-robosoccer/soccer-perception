"""The MuJoCo side of the Phase 3 benchmark: renders what the camera would see.

This is the only Phase 3 module that touches the simulator. It hands out

* `capture(...)`: an RGB image, which is all the vision pipeline receives;
* `labels_as_vision()`: the RENDERED LABELS of the same frame packed as a `VisionResult`, for
  the explicitly named "perfect vision" diagnostic and nothing else.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.datasets import image_aug, schema
from robosoccer_perception.datasets.annotation import SceneLabeler
from robosoccer_perception.datasets.schema import LabelPolicy
from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.inference.types import Detection, Segmentation, VisionResult
from robosoccer_perception.simulation.mujoco_scene import (
    LEFT_CAMERA, MujocoScene, load_scene_config)
from robosoccer_perception.vision_benchmark.camera_truth import SimulationCameraTruth


class SimulatedCamera:
    """A MuJoCo scene with the robot's left camera, driven frame by frame."""

    def __init__(self, field: FieldGeometry, camera_cfg: StereoCameraConfig,
                 scene_overrides: dict[str, Any] | None = None, with_labels: bool = False,
                 photometric: dict[str, Any] | None = None, scene: str = "default"):
        self.field, self.camera_cfg = field, camera_cfg
        self.policy = LabelPolicy()
        overrides = dict(scene_overrides or {})
        if with_labels:
            overrides = config.deep_update(overrides, {"offscreen_scale": self.policy.supersample})
        self.scene = MujocoScene(field, camera_cfg, load_scene_config(scene, overrides))
        self.labeler = SceneLabeler(self.scene, self.policy) if with_labels else None
        self.photometric = photometric          # the ``image`` section of a randomization preset
        self.ball_radius = float(self.scene.scene_cfg["ball"]["radius"])

    def capture(self, base_pose, truth: SimulationCameraTruth, ball_xy=None, robots=(),
                rng: np.random.Generator | None = None) -> np.ndarray:
        """Place everything and render the left image (H, W, 3 uint8 RGB)."""
        if type(truth) is not SimulationCameraTruth:
            raise TypeError("the simulator is driven by SimulationCameraTruth")
        self.scene.teleport(base_pose, truth.true_rig)
        self.scene.set_ball(ball_xy)
        self.scene.set_robots(list(robots))
        image = self.scene.render(LEFT_CAMERA, show_platform=False)
        if self.photometric and self.photometric.get("enabled") and rng is not None:
            image = image_aug.apply_photometric(
                image, image_aug.sample_photometric(rng, self.photometric))
        return image

    def labels_as_vision(self) -> VisionResult:
        """GROUND-TRUTH labels of the current frame in the vision pipeline's output format:
        the rendered semantic mask with one-hot probabilities, and the label boxes with
        confidence 1. Diagnostic use only ("perfect vision")."""
        if self.labeler is None:
            raise RuntimeError("construct SimulatedCamera(with_labels=True) to render labels")
        labels = self.labeler.annotate()
        probabilities = np.eye(len(schema.SEMANTIC_CLASSES), dtype=np.float32)[
            labels.semantic].transpose(2, 0, 1)
        detections = tuple(Detection(obj.class_id, 1.0, tuple(float(v) for v in obj.bbox_xyxy))
                           for obj in labels.labeled)
        return VisionResult(labels.width, labels.height, detections,
                            Segmentation(labels.semantic, probabilities), {}, "ground_truth_labels")

    def close(self) -> None:
        self.scene.close()

    def __enter__(self) -> "SimulatedCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
