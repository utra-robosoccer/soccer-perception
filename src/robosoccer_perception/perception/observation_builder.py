"""The bridge from image-derived features to the localizer's `ObservationFrame`.

    FeatureSet  ->  ObservationFrame(timestamp, landmarks, line_points)

The frame produced here is the same type, with the same fields and the same meaning, as the
one the oracle produces (`observations/oracle.py`); the particle filter cannot tell them apart
and needs no change. What does not fit the schema is mapped onto it, not added to it:

* a goal post known only by its BEARING is a landmark on the bearing ray with a range
  standard deviation of metres, so the localizer's range term is practically flat;
* a junction whose class could not be established has no `LandmarkType` and is left out;
* a circle candidate that failed a check is left out; an accepted one is a ``CENTER_MARK``.

`ObservationSelection` decides which kinds of feature are handed over at all.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.observations.types import (
    LandmarkObservation, LinePointsObservation, ObservationFrame)
from robosoccer_perception.perception.types import FeatureSet

_JUNCTION_NAMES = {"L": LandmarkType.L_INTERSECTION, "T": LandmarkType.T_INTERSECTION,
                   "X": LandmarkType.X_INTERSECTION}


@dataclasses.dataclass(frozen=True)
class ObservationSelection:
    """Which image-derived features reach the localizer."""
    line_points: bool = True
    goal_posts: bool = True
    bearing_only_goal_posts: bool = True
    intersections: tuple[str, ...] = ("L", "T", "X")    # classes to pass on
    center_circle: bool = True
    max_landmarks: int = 12

    def __post_init__(self) -> None:
        unknown = set(self.intersections) - set(_JUNCTION_NAMES)
        if unknown:
            raise ValueError(f"unknown intersection classes {sorted(unknown)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ObservationSelection":
        data = dict(data or {})
        if "intersections" in data:
            data["intersections"] = tuple(data["intersections"])
        return cls(**data)


def build_observation_frame(features: FeatureSet, timestamp: float,
                            selection: ObservationSelection | None = None) -> ObservationFrame:
    """Reduce a `FeatureSet` to the localizer's schema."""
    selection = selection or ObservationSelection()
    landmarks: list[LandmarkObservation] = []
    if selection.goal_posts:
        for post in features.goal_posts:
            if post.range_valid or selection.bearing_only_goal_posts:
                landmarks.append(LandmarkObservation(
                    LandmarkType.GOAL_POST, np.asarray(post.position, float),
                    float(post.uncertainty.range_std), float(post.uncertainty.bearing_std),
                    float(post.confidence)))
    wanted = {_JUNCTION_NAMES[name] for name in selection.intersections}
    for junction in features.intersections:
        if junction.type in wanted:
            landmarks.append(LandmarkObservation(
                junction.type, np.asarray(junction.position, float),
                float(junction.uncertainty.range_std), float(junction.uncertainty.bearing_std),
                float(junction.confidence)))
    if selection.center_circle:
        for circle in features.valid_circles:
            landmarks.append(LandmarkObservation(
                LandmarkType.CENTER_MARK, np.asarray(circle.center, float),
                float(circle.uncertainty.range_std), float(circle.uncertainty.bearing_std),
                float(circle.confidence)))
    if len(landmarks) > selection.max_landmarks:
        landmarks = sorted(landmarks, key=lambda o: -o.confidence)[:selection.max_landmarks]
    points = features.line_points if selection.line_points else None
    line_points = LinePointsObservation.empty() if points is None or len(points) == 0 else \
        LinePointsObservation(np.array(points.points, dtype=float), np.array(points.std, dtype=float))
    return ObservationFrame(float(timestamp), tuple(landmarks), line_points)
