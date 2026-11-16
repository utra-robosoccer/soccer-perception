"""The virtual stereo camera: synchronized left/right RGB frames from the MuJoCo scene.

Both images of a frame are rendered from the same simulation state, so they are synchronized
exactly. Frames are stamped on the camera's own clock (``frame_index / frame_rate``).

This is an idealised sensor: global shutter, no lens distortion, no noise, no exposure or
white-balance effects, and no depth output. See docs/camera_model.md for what that means for
sim-to-real. Importing this module requires the ``mujoco`` package (the ``sim`` extra).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.simulation.ground_truth import GroundTruth
from robosoccer_perception.simulation.mujoco_scene import (
    LEFT_CAMERA, RIGHT_CAMERA, MujocoScene)


@dataclasses.dataclass(frozen=True)
class StereoFrame:
    frame_index: int
    timestamp: float                    # seconds, camera clock
    left: np.ndarray                    # (H, W, 3) uint8 RGB
    right: np.ndarray                   # (H, W, 3) uint8 RGB
    T_field_left_optical: np.ndarray    # ground-truth camera poses at capture time
    T_field_right_optical: np.ndarray
    metadata: dict[str, Any]            # intrinsics, baseline, frame rate (camera_cfg.metadata())
    ground_truth: GroundTruth           # for evaluation and dataset labels; never for localization


class VirtualStereoCamera:
    """Captures stereo frames from a `MujocoScene` at the configured frame rate."""

    def __init__(self, scene: MujocoScene):
        self.scene = scene
        self.config = scene.camera_cfg
        self.frame_index = 0

    @property
    def frame_period(self) -> float:
        return 1.0 / self.config.frame_rate

    def capture(self) -> StereoFrame:
        """Render the left and right images of the scene as it is now."""
        truth = self.scene.ground_truth()
        left = self.scene.render(LEFT_CAMERA, show_platform=False)
        right = self.scene.render(RIGHT_CAMERA, show_platform=False)
        frame = StereoFrame(
            frame_index=self.frame_index, timestamp=self.frame_index * self.frame_period,
            left=left, right=right,
            T_field_left_optical=truth.T_field_left_optical,
            T_field_right_optical=truth.T_field_right_optical,
            metadata=self.config.metadata(), ground_truth=truth)
        self.frame_index += 1
        return frame


def save_stereo_frame(frame: StereoFrame, directory, stem: str | None = None) -> dict[str, Path]:
    """Write ``<stem>_left.png``, ``<stem>_right.png`` and ``<stem>.json`` (needs matplotlib)."""
    import matplotlib.image as mpimg

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = stem or f"frame_{frame.frame_index:06d}"
    paths = {side: directory / f"{stem}_{side}.png" for side in ("left", "right")}
    mpimg.imsave(paths["left"], frame.left)
    mpimg.imsave(paths["right"], frame.right)
    paths["metadata"] = directory / f"{stem}.json"
    paths["metadata"].write_text(json.dumps({
        "frame_index": frame.frame_index, "timestamp": frame.timestamp,
        "camera": frame.metadata, "ground_truth": frame.ground_truth.to_dict()}, indent=2))
    return paths
