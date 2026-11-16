"""Point landmarks of a field and their semantic classes. See docs/field_geometry.md.

Three ideas are kept apart on purpose:

* **Physical geometry**: where paint and posts are (`FieldGeometry`).
* **Landmark identity**: one specific point of the field, e.g. the T junction where the +x
  goal line meets the +y side of the penalty area (`Landmark.id`). Only the map and the
  evaluation code know identities.
* **Landmark class**: what a detector can tell from an image, e.g. "this is a T junction"
  (`LandmarkType`). Several landmarks share each class, and the field looks the same after a
  half turn, so a class never identifies a landmark. Observations carry a class and a position
  relative to the robot, never an identity; the localizer has to resolve the correspondence.
"""
from __future__ import annotations

import dataclasses
import enum

import numpy as np

from robosoccer_perception.geometry.field import SIDES, FieldGeometry, LineSegment


class LandmarkType(str, enum.Enum):
    """Classes a detector can assign to a point feature. All are ambiguous (see module doc)."""
    L_INTERSECTION = "L"            # two lines end in a corner
    T_INTERSECTION = "T"            # one line ends on another
    X_INTERSECTION = "X"            # two lines cross
    GOAL_POST = "goal_post"         # the base of a goal post
    PENALTY_MARK = "penalty_mark"
    CENTER_MARK = "center_mark"     # the centre of the centre circle


#: How many landmarks of each class a full field has (corner arcs add none).
EXPECTED_COUNTS = {
    LandmarkType.L_INTERSECTION: 12,    # 4 field corners + 2 x 2 goal area + 2 x 2 penalty area
    LandmarkType.T_INTERSECTION: 10,    # 2 halfway/touchline + 4 goal area + 4 penalty area
    LandmarkType.X_INTERSECTION: 2,     # halfway line x centre circle
    LandmarkType.GOAL_POST: 4,
    LandmarkType.PENALTY_MARK: 2,
    LandmarkType.CENTER_MARK: 1,
}


@dataclasses.dataclass(frozen=True)
class Landmark:
    id: str                         # unique identity, e.g. "T:penalty_area:pos_x:neg_y"
    type: LandmarkType
    position: tuple[float, float]   # field frame, metres, on the ground


class LandmarkMap:
    """All point landmarks of one field, grouped by class for the measurement model."""

    def __init__(self, landmarks: list[Landmark]):
        self.landmarks: tuple[Landmark, ...] = tuple(landmarks)
        ids = [lm.id for lm in self.landmarks]
        if len(set(ids)) != len(ids):
            raise ValueError("landmark ids must be unique")
        self._positions = {
            kind: np.array([lm.position for lm in self.landmarks if lm.type == kind],
                           dtype=float).reshape(-1, 2)
            for kind in LandmarkType}

    def positions(self, kind: LandmarkType) -> np.ndarray:
        """Positions (K, 2) of every landmark of one class."""
        return self._positions[kind]

    def of_type(self, kind: LandmarkType) -> list[Landmark]:
        return [lm for lm in self.landmarks if lm.type == kind]

    def by_id(self, landmark_id: str) -> Landmark:
        for landmark in self.landmarks:
            if landmark.id == landmark_id:
                return landmark
        raise KeyError(landmark_id)

    def __len__(self) -> int:
        return len(self.landmarks)

    def to_dict(self) -> list[dict]:
        return [{"id": lm.id, "type": lm.type.value, "position": list(lm.position)}
                for lm in self.landmarks]


def _centreline_intersection(a: LineSegment, b: LineSegment) -> tuple[float, float]:
    """Intersection of the (infinite) centrelines of two non-parallel segments."""
    p, r = np.asarray(a.start), np.asarray(a.end) - np.asarray(a.start)
    q, s = np.asarray(b.start), np.asarray(b.end) - np.asarray(b.start)
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-12:
        raise ValueError(f"segments {a.name} and {b.name} are parallel")
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / cross
    point = p + t * r
    return float(point[0]), float(point[1])


def build_landmark_map(field: FieldGeometry) -> LandmarkMap:
    """Derive every point landmark from the field's painted primitives and goals.

    Junction positions are the intersections of line centrelines, so they move with the
    outside-of-line convention automatically.
    """
    landmarks: list[Landmark] = []

    def junction(kind: LandmarkType, label: str, first: str, second: str) -> None:
        position = _centreline_intersection(field.segment(first), field.segment(second))
        landmarks.append(Landmark(f"{kind.value}:{label}", kind, position))

    for xl, _ in SIDES:
        for yl, _ in SIDES:
            junction(LandmarkType.L_INTERSECTION, f"corner:{xl}_x:{yl}_y",
                     f"touchline_{yl}_y", f"goal_line_{xl}_x")
            for area in ("goal_area", "penalty_area"):
                junction(LandmarkType.L_INTERSECTION, f"{area}:{xl}_x:{yl}_y",
                         f"{area}_front_{xl}_x", f"{area}_side_{xl}_x_{yl}_y")
                junction(LandmarkType.T_INTERSECTION, f"{area}:{xl}_x:{yl}_y",
                         f"goal_line_{xl}_x", f"{area}_side_{xl}_x_{yl}_y")
    for yl, sy in SIDES:
        junction(LandmarkType.T_INTERSECTION, f"halfway:{yl}_y",
                 "halfway_line", f"touchline_{yl}_y")
        # The halfway line runs through the circle's centre, so it crosses the circle at +-R.
        landmarks.append(Landmark(f"X:center_circle:{yl}_y", LandmarkType.X_INTERSECTION,
                                  (0.0, sy * field.center_circle_radius)))

    for goal in field.goals:
        for post, (yl, _) in zip(goal.posts, SIDES):
            landmarks.append(Landmark(f"goal_post:{goal.name}:{yl}_y", LandmarkType.GOAL_POST,
                                      post))
    for xl, sx in SIDES:
        landmarks.append(Landmark(f"penalty_mark:{xl}_x", LandmarkType.PENALTY_MARK,
                                  field.penalty_mark(sx)))
    landmarks.append(Landmark("center_mark", LandmarkType.CENTER_MARK, field.center_mark))
    return LandmarkMap(landmarks)
