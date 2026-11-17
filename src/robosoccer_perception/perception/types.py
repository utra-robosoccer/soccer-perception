"""Features extracted from one camera frame, before they become an `ObservationFrame`.

Everything is expressed in the robot's base frame (metres, +x forward, +y left) and carries its
own uncertainty and validity flags. These types are richer than `ObservationFrame` on purpose:
they keep what evaluation and debugging need (pixels, covariance parts, reasons for
rejection). `perception.observation_builder` reduces them to the localizer's schema.

None of these types holds a global pose or a landmark identity. A junction is "a T about 2 m
ahead", never "the T at the +x penalty area".
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from robosoccer_perception.geometry.landmarks import LandmarkType


@dataclasses.dataclass(frozen=True)
class MeasurementUncertainty:
    """Uncertainty of one point feature in the base frame.

    ``pixel_covariance`` is the part caused by not knowing the feature's pixel exactly;
    ``camera_covariance`` the part caused by the uncertain camera pose and focal length, which
    is shared by every feature of the frame. ``range_std`` / ``bearing_std`` are what the
    localizer receives; they already include any inflation (truncation, occlusion, confidence).
    """
    range_std: float                    # metres, along the viewing direction
    bearing_std: float                  # radians, across it
    pixel_covariance: np.ndarray        # (2, 2) metres^2
    camera_covariance: np.ndarray       # (2, 2) metres^2

    @property
    def covariance(self) -> np.ndarray:
        return self.pixel_covariance + self.camera_covariance


@dataclasses.dataclass(frozen=True)
class LinePointFeatures:
    """Points believed to lie on painted lines, thinned so that neighbours are not redundant."""
    pixels: np.ndarray                  # (N, 2) u, v in the input image (pixel centres)
    points: np.ndarray                  # (N, 2) metres in the base frame
    std: np.ndarray                     # (N,) metres: 1 sigma across the line, as reported
    confidence: np.ndarray              # (N,) segmentation line probability in [0, 1]
    pixel_std: np.ndarray               # (N,) metres: the pixel part alone
    camera_std: np.ndarray              # (N,) metres: the camera-pose part alone
    normals: np.ndarray                 # (N, 2) unit normal of the line through the point; NaN if unknown

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def empty(cls) -> "LinePointFeatures":
        zero = np.zeros(0)
        return cls(np.zeros((0, 2)), np.zeros((0, 2)), zero, zero, zero, zero, np.zeros((0, 2)))


@dataclasses.dataclass(frozen=True)
class LineSegmentFeature:
    """A straight piece of line fitted to ground points. Used to find junctions; not sent to
    the localizer."""
    start: np.ndarray                   # (2,) metres in the base frame
    end: np.ndarray
    num_points: int
    rms: float                          # metres: scatter of the points about the fitted line

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / max(self.length, 1e-9)


@dataclasses.dataclass(frozen=True)
class GoalPostFeature:
    """One detected goal post: where its base is, or only its bearing.

    ``range_valid`` is False when the foot of the post could not be trusted (cut off by the
    image border, hidden, or inconsistent with the post's known size). The position is then a
    point on the bearing ray at a rough range, and ``uncertainty.range_std`` is large enough
    that only the bearing constrains the localizer.
    """
    pixel: np.ndarray                   # (2,) u, v of the foot (bottom centre of the box)
    position: np.ndarray                # (2,) metres in the base frame
    uncertainty: MeasurementUncertainty
    confidence: float                   # detector score
    range_valid: bool
    flags: tuple[str, ...] = ()         # why it was downgraded or inflated
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclasses.dataclass(frozen=True)
class IntersectionFeature:
    """A junction of two fitted line segments. ``type`` is L, T or X, or None when the shape
    could not be established (a generic junction, which is not sent to the localizer)."""
    type: LandmarkType | None
    position: np.ndarray                # (2,) metres in the base frame
    uncertainty: MeasurementUncertainty
    confidence: float
    orientation: float | None = None    # radians in the base frame: T stem / L bisector direction
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CircleFeature:
    """A centre-circle candidate: a circle of the known radius fitted to ground points."""
    center: np.ndarray                  # (2,) metres in the base frame
    radius: float                       # the known radius it was fitted with
    uncertainty: MeasurementUncertainty
    confidence: float
    arc_coverage_deg: float             # how much of the circle the supporting points span
    num_inliers: int
    valid: bool                         # passed every consistency check
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class FeatureSet:
    """Everything geometric extraction got out of one frame."""
    line_points: LinePointFeatures = dataclasses.field(default_factory=LinePointFeatures.empty)
    goal_posts: tuple[GoalPostFeature, ...] = ()
    intersections: tuple[IntersectionFeature, ...] = ()
    circles: tuple[CircleFeature, ...] = ()
    segments: tuple[LineSegmentFeature, ...] = ()
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def typed_intersections(self) -> tuple[IntersectionFeature, ...]:
        return tuple(i for i in self.intersections if i.type is not None)

    @property
    def valid_circles(self) -> tuple[CircleFeature, ...]:
        return tuple(c for c in self.circles if c.valid)
