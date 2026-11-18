"""Configuration of geometric feature extraction and of the measurement-noise model.

Two files, ``configs/perception/feature_extraction.yaml`` and
``configs/perception/measurement_noise.yaml``, mirror the dataclasses below. The defaults here
and the values there are the same; the YAML files exist so that every number is documented in
one place and can be changed without editing code.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any

from robosoccer_perception import config
from robosoccer_perception.geometry.ground_projection import (
    CalibrationUncertainty, ProjectionLimits)
from robosoccer_perception.perception.observation_builder import ObservationSelection


def _build(cls, data: dict[str, Any] | None, nested: dict[str, Any] | None = None):
    """Construct a config dataclass from a mapping, refusing unknown keys."""
    data = dict(data or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    for key, builder in (nested or {}).items():
        data[key] = builder(data.get(key))
    for field in dataclasses.fields(cls):
        if field.name in data and isinstance(data[field.name], list):
            data[field.name] = tuple(data[field.name])
    return cls(**data)


# ------------------------------------------------------------------ feature extraction
@dataclasses.dataclass(frozen=True)
class LineConfig:
    min_confidence: float = 0.5         # line probability a pixel needs
    min_component_area_px: int = 12     # smaller blobs of line pixels are dropped
    pixel_cell_px: int = 4              # one centreline pixel per cell is kept as a candidate
    dense_ground_spacing: float = 0.06  # metres between candidates used for fitting
    ground_spacing: float = 0.20        # metres between the line points given to the localizer
    max_points: int = 60                # line points per frame at most


@dataclasses.dataclass(frozen=True)
class SegmentConfig:
    enabled: bool = True
    max_range: float = 4.5              # metres: farther points are too noisy to fit
    iterations: int = 200               # RANSAC hypotheses per line
    inlier_tolerance: float = 0.03      # metres, combined with each point's pixel noise
    max_tolerance: float = 0.12         # metres: cap of the per-point tolerance
    min_points: int = 6
    min_length: float = 0.5             # metres
    max_gap: float = 0.30               # metres: a larger gap splits a line into two segments
    max_segments: int = 10
    merge_angle_deg: float = 8.0        # collinear segments this close in direction ...
    merge_offset: float = 0.08          # ... and this close across ...
    merge_gap: float = 0.5              # ... and with a gap below this are joined


@dataclasses.dataclass(frozen=True)
class IntersectionConfig:
    enabled: bool = True
    max_range: float = 4.0              # metres from the robot
    angle_tolerance_deg: float = 20.0   # the two lines must meet within this of a right angle
    min_arm: float = 0.30               # metres a line must extend past the junction to count
    end_tolerance: float = 0.12         # metres the junction may lie inside a segment's end
    max_end_gap: float = 0.20           # metres the junction may lie beyond a segment's end
    probe_distances: tuple[float, ...] = (0.20, 0.35)   # metres past an end, checked for turf
    probe_min_field_confidence: float = 0.5
    image_margin_px: float = 6.0        # a probe this close to the border proves nothing
    merge_distance: float = 0.30        # metres: junctions closer than this are one junction


@dataclasses.dataclass(frozen=True)
class CircleConfig:
    enabled: bool = True
    max_center_range: float = 5.0       # metres from the robot to the fitted centre
    iterations: int = 200
    inlier_tolerance: float = 0.04      # metres, combined with each point's pixel noise
    max_tolerance: float = 0.12
    min_arc_deg: float = 150.0          # the supporting points must span at least this much
    min_inliers: int = 14
    min_density: float = 0.5            # share of the spanned arc that must actually hold points
    angular_bin_deg: float = 15.0
    purity_band: float = 0.15           # metres: band around the circle that is inspected
    min_purity: float = 0.75            # share of the points in that band that must support it


@dataclasses.dataclass(frozen=True)
class GoalPostConfig:
    enabled: bool = True
    min_confidence: float = 0.35
    border_margin_px: float = 3.0       # a box this close to a border is cut off there
    max_range: float = 8.0              # metres: farther feet give bearing only
    min_range: float = 0.3
    reject_range: float = 14.0          # metres: beyond this the detection is implausible
    width_ratio_limits: tuple[float, float] = (0.45, 2.2)   # observed / expected box width
    height_ratio_min: float = 0.6       # observed / expected box height when the top is in view
    foot_window_px: tuple[int, int] = (9, 5)    # width, height of the mask patch under the foot
    foot_min_ground_fraction: float = 0.4       # share of that patch that must be turf or line
    occluder_overlap: float = 0.25      # share of the foot strip covered by a robot / ball box


@dataclasses.dataclass(frozen=True)
class FeatureExtractionConfig:
    name: str = "default"
    max_work_width: int = 640           # wider masks are reduced to this before extraction
    projection: ProjectionLimits = dataclasses.field(
        default_factory=lambda: ProjectionLimits(4.0, 6.0, 0.25, 2.0))
    lines: LineConfig = dataclasses.field(default_factory=LineConfig)
    segments: SegmentConfig = dataclasses.field(default_factory=SegmentConfig)
    intersections: IntersectionConfig = dataclasses.field(default_factory=IntersectionConfig)
    circle: CircleConfig = dataclasses.field(default_factory=CircleConfig)
    goal_posts: GoalPostConfig = dataclasses.field(default_factory=GoalPostConfig)
    observations: ObservationSelection = dataclasses.field(default_factory=ObservationSelection)
    seed: int = 0                       # RANSAC sampling; fixed so that extraction is repeatable

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureExtractionConfig":
        return _build(cls, data, {
            "projection": lambda d: ProjectionLimits.from_dict(d) if d else
            ProjectionLimits(4.0, 6.0, 0.25, 2.0),
            "lines": lambda d: _build(LineConfig, d),
            "segments": lambda d: _build(SegmentConfig, d),
            "intersections": lambda d: _build(IntersectionConfig, d),
            "circle": lambda d: _build(CircleConfig, d),
            "goal_posts": lambda d: _build(GoalPostConfig, d),
            "observations": ObservationSelection.from_dict})

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "feature_extraction",
             overrides: dict[str, Any] | None = None) -> "FeatureExtractionConfig":
        return cls.from_dict(config.deep_update(
            config.load_config("perception", name_or_path), overrides))


# ------------------------------------------------------------------ measurement noise
@dataclasses.dataclass(frozen=True)
class MeasurementNoiseConfig:
    """How image-derived features are turned into standard deviations.

    See docs/geometric_perception.md, "Uncertainty model", for what each term stands for.
    """
    name: str = "default"
    calibration: CalibrationUncertainty = dataclasses.field(default_factory=CalibrationUncertainty)
    confidence_floor: float = 0.25          # confidences below this inflate as if they were this
    confidence_exponent: float = 0.5        # std is divided by confidence ** exponent
    # field-line points
    line_std_floor: float = 0.02            # metres
    line_std_max: float = 0.45              # points less certain than this are dropped
    # goal posts
    post_pixel_std: tuple[float, float] = (1.5, 3.0)    # px: column, row of the foot
    post_range_std_floor: float = 0.05      # metres
    post_bearing_std_floor_deg: float = 0.75
    post_inflation_flagged: float = 2.0     # range std factor when a check raised a flag
    bearing_only_range_std_fraction: float = 0.6        # of the rough range ...
    bearing_only_range_std_min: float = 1.5             # ... but at least this, metres
    # junctions
    junction_pixel_std: float = 1.5         # px
    junction_std_floor: float = 0.04        # metres
    junction_inflation: float = 1.5         # they reuse the line pixels: do not count twice
    # centre circle
    circle_pixel_std: float = 1.0           # px
    circle_std_floor: float = 0.05          # metres
    circle_inflation: float = 1.5
    # camera error shared by every feature of a frame (Phase 4; off = the Phase 3 behaviour)
    shared_camera_error: bool = False       # price the camera part as COMMON to the frame
    shared_error_point_weight: float = 0.3  # the localizer's exponent per line point ...
    shared_error_max_points: int = 40       # ... and the number of points it uses per update
    shared_error_max_inflation: float = 12.0    # cap of the variance factor

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MeasurementNoiseConfig":
        return _build(cls, data, {"calibration": CalibrationUncertainty.from_dict})

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "measurement_noise",
             overrides: dict[str, Any] | None = None) -> "MeasurementNoiseConfig":
        return cls.from_dict(config.deep_update(
            config.load_config("perception", name_or_path), overrides))
