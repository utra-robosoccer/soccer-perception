"""Configurable error model of the oracle. See configs/observations/*.yaml for the meaning."""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import numpy as np

from robosoccer_perception import config


@dataclasses.dataclass(frozen=True)
class OracleConfig:
    name: str = "nominal"
    # visibility
    min_range: float = 0.3
    max_range: float = 8.0
    image_margin_px: float = 4.0
    min_circle_visible_fraction: float = 0.35
    occlusion: bool = True
    # field-line points
    line_point_spacing: float = 0.10
    max_line_points: int = 60
    line_point_dropout_prob: float = 0.10
    false_line_point_fraction: float = 0.05
    # measurement noise
    range_std_base: float = 0.02
    range_std_quadratic: float = 0.01
    bearing_std_deg: float = 1.0
    pitch_jitter_std_deg: float = 0.25
    # detection errors
    landmark_dropout_prob: float = 0.10
    frame_dropout_prob: float = 0.0
    false_landmark_rate: float = 0.05
    landmark_outlier_prob: float = 0.02
    class_confusion_prob: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown oracle configuration keys: {sorted(unknown)}")
        cfg = cls(**data)
        for name in ("line_point_dropout_prob", "landmark_dropout_prob", "frame_dropout_prob",
                     "landmark_outlier_prob", "class_confusion_prob"):
            if not 0.0 <= getattr(cfg, name) <= 1.0:
                raise ValueError(f"{name} must be a probability")
        return cfg

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "nominal",
             overrides: dict[str, Any] | None = None) -> "OracleConfig":
        return cls.from_dict({**config.load_config("observations", name_or_path),
                              **(overrides or {})})

    def replace(self, **changes) -> "OracleConfig":
        return dataclasses.replace(self, **changes)

    @property
    def bearing_std(self) -> float:
        return math.radians(self.bearing_std_deg)

    def range_std(self, distance) -> np.ndarray:
        """1-sigma range error at a given range.

        Quadratic growth is what both ways of ranging a ground feature give: stereo depth error
        grows as Z^2 / (f * baseline), and ground-plane projection error as r^2 / height.
        """
        distance = np.asarray(distance, dtype=float)
        return self.range_std_base + self.range_std_quadratic * distance ** 2


def apply_pitch_error(distance, camera_height: float, pitch_error: float) -> np.ndarray:
    """Range reported for a ground point when the assumed camera pitch is off by ``pitch_error``.

    A ground point at range r is seen at a depression angle atan(h / r). Projecting it back to
    the ground with a pitch that is wrong by delta gives h / tan(atan(h / r) + delta): the same
    small error moves far points much more than near ones, and all points move together.
    """
    distance = np.asarray(distance, dtype=float)
    depression = np.arctan2(camera_height, distance) + pitch_error
    return camera_height / np.tan(np.clip(depression, math.radians(1.0), math.pi / 2))
