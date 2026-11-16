"""Geometric model of the stereo camera: intrinsics, extrinsics and projection.

Pure NumPy. The MuJoCo renderer (simulation/stereo_camera.py) and the oracle observation
generator both use this model, so what is rendered and what is "observed" agree by
construction. See docs/camera_model.md.

Image convention (OpenCV / ROS): ``u`` to the right, ``v`` down, integer coordinates at pixel
centres. A point ``(X, Y, Z)`` in an optical frame projects to
``u = fx * X / Z + cx``, ``v = fy * Y / Z + cy``.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.geometry import transforms as tf


@dataclasses.dataclass(frozen=True)
class CameraIntrinsics:
    """Ideal pinhole camera (no distortion)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def horizontal_fov(self) -> float:
        """Horizontal field of view in radians (pixel edge to pixel edge)."""
        return (math.atan((self.cx + 0.5) / self.fx)
                + math.atan((self.width - 0.5 - self.cx) / self.fx))

    @property
    def vertical_fov(self) -> float:
        return (math.atan((self.cy + 0.5) / self.fy)
                + math.atan((self.height - 0.5 - self.cy) / self.fy))

    def project(self, points_optical) -> tuple[np.ndarray, np.ndarray]:
        """Project optical-frame points (..., 3) to pixels (..., 2). Also returns depth Z.

        Points with Z <= 0 are behind the camera; their pixel coordinates are NaN.
        """
        points = np.asarray(points_optical, dtype=float)
        depth = points[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(depth > 0.0, self.fx * points[..., 0] / depth + self.cx, np.nan)
            v = np.where(depth > 0.0, self.fy * points[..., 1] / depth + self.cy, np.nan)
        return np.stack([u, v], axis=-1), depth

    def in_image(self, pixels, margin: float = 0.0) -> np.ndarray:
        """True where a pixel coordinate lies inside the image, ``margin`` pixels from its edge."""
        pixels = np.asarray(pixels, dtype=float)
        u, v = pixels[..., 0], pixels[..., 1]
        with np.errstate(invalid="ignore"):
            return ((u >= -0.5 + margin) & (u <= self.width - 0.5 - margin)
                    & (v >= -0.5 + margin) & (v <= self.height - 0.5 - margin))

    def ray(self, pixels) -> np.ndarray:
        """Viewing-ray directions (..., 3) in the optical frame, with Z = 1."""
        pixels = np.asarray(pixels, dtype=float)
        return np.stack([(pixels[..., 0] - self.cx) / self.fx,
                         (pixels[..., 1] - self.cy) / self.fy,
                         np.ones(pixels.shape[:-1])], axis=-1)


@dataclasses.dataclass(frozen=True)
class RigState:
    """Where the stereo rig sits on the robot. Adjustable at run time (head motion).

    The rig frame is the midpoint between the two optical centres with body-style axes.
    ``pitch`` is a right-handed rotation about the base +y axis, so positive pitch looks DOWN;
    ``pan`` is a rotation about base +z, so positive pan looks LEFT. Angles in radians.
    """
    height: float
    pitch: float
    pan: float = 0.0
    forward_offset: float = 0.0
    lateral_offset: float = 0.0

    def replace(self, **changes) -> "RigState":
        return dataclasses.replace(self, **changes)


@dataclasses.dataclass(frozen=True)
class StereoCameraConfig:
    """One ``configs/cameras/*.yaml`` file."""
    name: str
    approximate: bool
    baseline: float
    frame_rate: float
    intrinsics: CameraIntrinsics
    mount: RigState
    clip_near: float
    clip_far: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StereoCameraConfig":
        image, intr, mount = data["image"], data["intrinsics"], data["mount"]
        clip = data.get("clip", {})
        cfg = cls(
            name=str(data["name"]),
            approximate=bool(data.get("approximate", True)),
            baseline=float(data["baseline"]),
            frame_rate=float(data["frame_rate"]),
            intrinsics=CameraIntrinsics(int(image["width"]), int(image["height"]),
                                        float(intr["fx"]), float(intr["fy"]),
                                        float(intr["cx"]), float(intr["cy"])),
            mount=RigState(height=float(mount["height"]),
                           pitch=math.radians(float(mount.get("pitch_deg", 0.0))),
                           pan=math.radians(float(mount.get("pan_deg", 0.0))),
                           forward_offset=float(mount.get("forward_offset", 0.0)),
                           lateral_offset=float(mount.get("lateral_offset", 0.0))),
            clip_near=float(clip.get("near", 0.05)),
            clip_far=float(clip.get("far", 60.0)))
        if cfg.baseline <= 0 or cfg.frame_rate <= 0 or cfg.mount.height <= 0:
            raise ValueError("baseline, frame_rate and mount height must be positive")
        return cfg

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "zed_mini_sim") -> "StereoCameraConfig":
        return cls.from_dict(config.load_config("cameras", name_or_path))

    def with_resolution(self, width: int, height: int) -> "StereoCameraConfig":
        """The same field of view at another resolution (intrinsics scaled with the image)."""
        k = self.intrinsics
        sx, sy = width / k.width, height / k.height
        scaled = CameraIntrinsics(width, height, k.fx * sx, k.fy * sy,
                                  (k.cx + 0.5) * sx - 0.5, (k.cy + 0.5) * sy - 0.5)
        return dataclasses.replace(self, intrinsics=scaled)

    def metadata(self) -> dict[str, Any]:
        k = self.intrinsics
        return {
            "name": self.name, "approximate": self.approximate, "baseline_m": self.baseline,
            "frame_rate_hz": self.frame_rate, "width": k.width, "height": k.height,
            "fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy,
            "horizontal_fov_deg": math.degrees(k.horizontal_fov),
            "vertical_fov_deg": math.degrees(k.vertical_fov),
            "distortion_model": "none (ideal pinhole)",
            "image_convention": "u right, v down, integer coordinates at pixel centres",
        }


SIDES = ("left", "right")


class StereoCameraModel:
    """Extrinsics and projection of the two cameras. Both cameras share one set of intrinsics.

    Frame chain: ``field <- base <- rig <- {left,right}_optical``.
    """

    def __init__(self, cfg: StereoCameraConfig):
        self.config = cfg
        self.intrinsics = cfg.intrinsics
        self.baseline = cfg.baseline

    # ------------------------------------------------------------------ extrinsics
    def T_base_rig(self, rig: RigState | None = None) -> np.ndarray:
        rig = rig or self.config.mount
        rotation = tf.rot_z(rig.pan) @ tf.rot_y(rig.pitch)
        return tf.make_transform(rotation, [rig.forward_offset, rig.lateral_offset, rig.height])

    def T_rig_optical(self, side: str = "left") -> np.ndarray:
        """Optical frame of one camera in the rig frame.

        The left camera is half a baseline to the rig's +y (left), the right camera half a
        baseline to -y. The optical axes are parallel.
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        offset = 0.5 * self.baseline * (1.0 if side == "left" else -1.0)
        return tf.make_transform(tf.R_BODY_OPTICAL, [0.0, offset, 0.0])

    def T_base_camera(self, side: str = "left", rig: RigState | None = None) -> np.ndarray:
        """``T_base_<side>_optical``."""
        return self.T_base_rig(rig) @ self.T_rig_optical(side)

    def T_field_camera(self, base_pose, side: str = "left",
                       rig: RigState | None = None) -> np.ndarray:
        """``T_field_<side>_optical`` for a robot at the 3-DOF pose ``base_pose``."""
        return tf.T_field_base_from_pose(base_pose) @ self.T_base_camera(side, rig)

    def T_left_right(self) -> np.ndarray:
        """Right optical frame in the left optical frame: a pure +x shift by the baseline."""
        return tf.invert_transform(self.T_rig_optical("left")) @ self.T_rig_optical("right")

    # ------------------------------------------------------------------ projection
    def project_field_points(self, base_pose, points_field, side: str = "left",
                             rig: RigState | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project field-frame 3-D points (..., 3) into one camera.

        Returns ``(pixels, depth, visible)``: ``visible`` is True for points in front of the
        near plane that land inside the image.
        """
        T_camera_field = tf.invert_transform(self.T_field_camera(base_pose, side, rig))
        pixels, depth = self.intrinsics.project(tf.transform_points(T_camera_field, points_field))
        visible = (depth > self.config.clip_near) & self.intrinsics.in_image(pixels)
        return pixels, depth, visible

    def disparity(self, depth) -> np.ndarray:
        """Horizontal disparity ``u_left - u_right`` in pixels of a point at optical depth Z."""
        return self.intrinsics.fx * self.baseline / np.asarray(depth, dtype=float)

    def depth_from_disparity(self, disparity) -> np.ndarray:
        return self.intrinsics.fx * self.baseline / np.asarray(disparity, dtype=float)
