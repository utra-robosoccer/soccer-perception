"""The single authoritative description of a RoboCup field. See docs/field_geometry.md.

`FieldDimensions` holds the numbers from a ``configs/fields/*.yaml`` file. `FieldGeometry`
turns them into painted primitives (line segments, arcs), goals and extents, in the field frame.
Everything else is derived from a `FieldGeometry`: the MuJoCo scene, the landmark map, the
line distance map, the oracle observations, the benchmarks and the plots. No other module
contains field dimensions.

Outside-of-line convention
--------------------------
The rulebook measures every dimension to the OUTSIDE of the lines. A line is stored as its
centreline plus its width, so a boundary line of the 9 m field has its centreline at
``9 / 2 - line_width / 2`` and its outer edge exactly at 4.5 m. The same half-width shift is
applied to the goal area, the penalty area, the centre circle and the corner arcs. The halfway
line is centred on the field's y axis; marks are located by their centres.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from typing import Any

import numpy as np

from robosoccer_perception import config

SIDES = (("pos", 1), ("neg", -1))


@dataclasses.dataclass(frozen=True)
class GoalDimensions:
    width: float            # inner distance between the two posts
    depth: float            # goal line to the back of the goal
    height: float           # ground to the lower edge of the crossbar
    post_diameter: float


@dataclasses.dataclass(frozen=True)
class FieldDimensions:
    """The contents of one field configuration file. Lengths in metres, outside-of-line."""
    name: str
    length: float
    width: float
    line_width: float
    border_strip_width: float
    goal_area_depth: float
    goal_area_width: float
    penalty_area_depth: float
    penalty_area_width: float
    penalty_mark_distance: float
    center_circle_diameter: float
    corner_arc_radius: float
    goal: GoalDimensions
    mark_size: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldDimensions":
        data = dict(data)
        goal = GoalDimensions(**{k: float(v) for k, v in data.pop("goal").items()})
        names = {f.name for f in dataclasses.fields(cls)} - {"name", "goal"}
        unknown = set(data) - names - {"name"}
        if unknown:
            raise ValueError(f"unknown field configuration keys: {sorted(unknown)}")
        missing = names - set(data)
        if missing:
            raise ValueError(f"missing field configuration keys: {sorted(missing)}")
        dims = cls(name=str(data["name"]), goal=goal, **{k: float(data[k]) for k in names})
        dims.validate()
        return dims

    def validate(self) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"field '{self.name}': {message}")

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and field.name != "corner_arc_radius":
                require(value > 0.0, f"{field.name} must be positive")
        require(self.corner_arc_radius >= 0.0, "corner_arc_radius must be >= 0")
        require(self.length > self.width, "length must exceed width")
        require(self.goal_area_width < self.penalty_area_width < self.width,
                "need goal_area_width < penalty_area_width < width")
        require(self.goal_area_depth < self.penalty_area_depth < self.length / 2,
                "need goal_area_depth < penalty_area_depth < length / 2")
        require(self.goal.width + 2 * self.goal.post_diameter < self.goal_area_width,
                "the goal must fit inside the goal area")
        require(self.center_circle_diameter < self.width, "centre circle wider than the field")
        require(self.penalty_mark_distance < self.length / 2, "penalty mark beyond halfway")
        require(self.line_width < 0.5 * min(self.goal_area_depth, self.center_circle_diameter),
                "line_width is implausibly large")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LineSegment:
    """A straight painted line: a rectangle ``width`` wide around the centreline start-end."""
    name: str
    kind: str                       # touchline, goal_line, halfway_line, goal_area, ...
    start: tuple[float, float]
    end: tuple[float, float]
    width: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


@dataclasses.dataclass(frozen=True)
class LineArc:
    """A painted circular arc: ``radius`` is the centreline radius, angles are CCW, radians."""
    name: str
    kind: str                       # center_circle, corner_arc
    center: tuple[float, float]
    radius: float
    start_angle: float
    end_angle: float
    width: float

    @property
    def span(self) -> float:
        return self.end_angle - self.start_angle

    @property
    def is_full_circle(self) -> bool:
        return self.span >= 2.0 * math.pi - 1e-9


@dataclasses.dataclass(frozen=True)
class Goal:
    """One goal. Post positions are the post centres on the ground, in the field frame."""
    name: str                       # goal_pos_x / goal_neg_x
    side: int                       # +1: the goal at +x, -1: the goal at -x
    posts: tuple[tuple[float, float], tuple[float, float]]
    dims: GoalDimensions

    @property
    def line_x(self) -> float:
        return self.posts[0][0]


class FieldGeometry:
    """Painted lines, marks, goals and extents of one field, in the field frame."""

    def __init__(self, dims: FieldDimensions):
        self.dims = dims
        self.segments: tuple[LineSegment, ...] = tuple(self._build_segments())
        self.arcs: tuple[LineArc, ...] = tuple(self._build_arcs())
        self.goals: tuple[Goal, ...] = tuple(self._build_goals())
        self._segments_by_name = {s.name: s for s in self.segments}
        self._arcs_by_name = {a.name: a for a in self.arcs}

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, name_or_path: str | os.PathLike) -> "FieldGeometry":
        """Load ``configs/fields/<name>.yaml`` (or an explicit YAML path)."""
        return cls(FieldDimensions.from_dict(config.load_config("fields", name_or_path)))

    @property
    def name(self) -> str:
        return self.dims.name

    @property
    def half_length(self) -> float:
        """Half the field length: x of the OUTER edge of the +x goal line."""
        return self.dims.length / 2.0

    @property
    def half_width(self) -> float:
        """Half the field width: y of the OUTER edge of the +y touchline."""
        return self.dims.width / 2.0

    @property
    def half_extent(self) -> tuple[float, float]:
        """Half-size (x, y) of the carpet: the field plus its border strip.

        The x border is widened if needed so that the goals stand on the carpet.
        """
        d = self.dims
        border_x = max(d.border_strip_width, d.goal.depth + 2.0 * d.goal.post_diameter)
        return self.half_length + border_x, self.half_width + d.border_strip_width

    @property
    def center_circle_radius(self) -> float:
        """Centreline radius of the centre circle."""
        return self.dims.center_circle_diameter / 2.0 - self.dims.line_width / 2.0

    def penalty_mark(self, side: int) -> tuple[float, float]:
        return (side * (self.half_length - self.dims.penalty_mark_distance), 0.0)

    @property
    def center_mark(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def segment(self, name: str) -> LineSegment:
        return self._segments_by_name[name]

    def arc(self, name: str) -> LineArc:
        return self._arcs_by_name[name]

    def content_hash(self) -> str:
        """Stable hash of the dimensions; anything cached per field is keyed on it."""
        payload = json.dumps(self.dims.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()

    def _build_segments(self) -> list[LineSegment]:
        d = self.dims
        w, hw = d.line_width, d.line_width / 2.0
        a, b = self.half_length, self.half_width
        segments: list[LineSegment] = []

        def add(name, kind, start, end):
            segments.append(LineSegment(name, kind, (float(start[0]), float(start[1])),
                                        (float(end[0]), float(end[1])), w))

        for label, s in SIDES:
            add(f"touchline_{label}_y", "touchline", (-a, s * (b - hw)), (a, s * (b - hw)))
        for label, s in SIDES:
            add(f"goal_line_{label}_x", "goal_line", (s * (a - hw), -b), (s * (a - hw), b))
        add("halfway_line", "halfway_line", (0.0, -b), (0.0, b))

        for area, depth, width in (("goal_area", d.goal_area_depth, d.goal_area_width),
                                   ("penalty_area", d.penalty_area_depth, d.penalty_area_width)):
            for xl, sx in SIDES:
                front_x = sx * (a - depth + hw)      # outer edge of this line is `depth` from x=a
                add(f"{area}_front_{xl}_x", area, (front_x, -width / 2), (front_x, width / 2))
                for yl, sy in SIDES:
                    side_y = sy * (width / 2 - hw)
                    add(f"{area}_side_{xl}_x_{yl}_y", area,
                        (sx * (a - depth), side_y), (sx * a, side_y))

        # Marks are painted crosses. The centre mark needs only the arm across the halfway line.
        arm = d.mark_size / 2.0
        add("center_mark_arm", "mark", (-arm, 0.0), (arm, 0.0))
        for xl, sx in SIDES:
            mx, my = sx * (a - d.penalty_mark_distance), 0.0
            add(f"penalty_mark_{xl}_x_arm_x", "mark", (mx - arm, my), (mx + arm, my))
            add(f"penalty_mark_{xl}_x_arm_y", "mark", (mx, my - arm), (mx, my + arm))
        return segments

    def _build_arcs(self) -> list[LineArc]:
        d = self.dims
        w, hw = d.line_width, d.line_width / 2.0
        arcs = [LineArc("center_circle", "center_circle", (0.0, 0.0),
                        d.center_circle_diameter / 2.0 - hw, 0.0, 2.0 * math.pi, w)]
        if d.corner_arc_radius > 0.0:
            # A quarter circle centred on each OUTER corner, opening into the field.
            quadrant_start = {(1, 1): math.pi, (-1, 1): 1.5 * math.pi,
                              (-1, -1): 0.0, (1, -1): 0.5 * math.pi}
            for xl, sx in SIDES:
                for yl, sy in SIDES:
                    start = quadrant_start[(sx, sy)]
                    arcs.append(LineArc(
                        f"corner_arc_{xl}_x_{yl}_y", "corner_arc",
                        (sx * self.half_length, sy * self.half_width),
                        d.corner_arc_radius - hw, start, start + 0.5 * math.pi, w))
        return arcs

    def _build_goals(self) -> list[Goal]:
        d = self.dims
        # ASSUMPTION: the posts are centred on the goal line's centreline.
        line_x = self.half_length - d.line_width / 2.0
        post_y = d.goal.width / 2.0 + d.goal.post_diameter / 2.0
        return [Goal(f"goal_{label}_x", s, ((s * line_x, post_y), (s * line_x, -post_y)), d.goal)
                for label, s in SIDES]

    # ------------------------------------------------------------------ queries
    def distance_to_paint(self, points) -> np.ndarray:
        """Distance (m) from each (x, y) point to the nearest painted surface; 0 on the paint.

        Exact for the straight lines (rectangles with square ends) and for the arcs. ``points``
        has shape (..., 2); the result has shape (...).
        """
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 2)
        best = np.full(len(flat), np.inf)
        for segment in self.segments:
            np.minimum(best, _distance_to_segment(flat, segment), out=best)
        for arc in self.arcs:
            np.minimum(best, _distance_to_arc(flat, arc), out=best)
        return best.reshape(points.shape[:-1])

    def is_on_paint(self, points) -> np.ndarray:
        return self.distance_to_paint(points) <= 1e-9

    def sample_line_points(self, spacing: float = 0.1, include_marks: bool = False) -> np.ndarray:
        """Points (M, 2) spaced about ``spacing`` metres apart along every line's centreline."""
        chunks = []
        for segment in self.segments:
            if segment.kind == "mark" and not include_marks:
                continue
            count = max(2, int(math.ceil(segment.length / spacing)) + 1)
            fraction = np.linspace(0.0, 1.0, count)[:, None]
            chunks.append(np.asarray(segment.start) * (1 - fraction)
                          + np.asarray(segment.end) * fraction)
        for arc in self.arcs:
            chunks.append(sample_arc(arc, spacing))
        return np.concatenate(chunks, axis=0)

    def contains(self, points, margin: float = 0.0) -> np.ndarray:
        """True for points inside the carpet (field plus border strip), shrunk by ``margin``."""
        points = np.asarray(points, dtype=float)
        half_x, half_y = self.half_extent
        return ((np.abs(points[..., 0]) <= half_x - margin)
                & (np.abs(points[..., 1]) <= half_y - margin))


def sample_arc(arc: LineArc, spacing: float) -> np.ndarray:
    """Points along an arc's centreline. A full circle does not repeat its first point."""
    count = max(3, int(math.ceil(arc.radius * arc.span / spacing)) + 1)
    angles = np.linspace(arc.start_angle, arc.end_angle, count, endpoint=not arc.is_full_circle)
    return np.stack([arc.center[0] + arc.radius * np.cos(angles),
                     arc.center[1] + arc.radius * np.sin(angles)], axis=1)


def _distance_to_segment(points: np.ndarray, segment: LineSegment) -> np.ndarray:
    start, end = np.asarray(segment.start), np.asarray(segment.end)
    half_length = 0.5 * np.linalg.norm(end - start)
    along = (end - start) / (2.0 * half_length)
    across = np.array([-along[1], along[0]])
    relative = points - 0.5 * (start + end)
    excess_along = np.maximum(np.abs(relative @ along) - half_length, 0.0)
    excess_across = np.maximum(np.abs(relative @ across) - segment.width / 2.0, 0.0)
    return np.hypot(excess_along, excess_across)


def _distance_to_arc(points: np.ndarray, arc: LineArc) -> np.ndarray:
    relative = points - np.asarray(arc.center)
    radial = np.abs(np.hypot(relative[:, 0], relative[:, 1]) - arc.radius)
    if arc.is_full_circle:
        return np.maximum(radial - arc.width / 2.0, 0.0)
    angle = (np.arctan2(relative[:, 1], relative[:, 0]) - arc.start_angle) % (2.0 * math.pi)
    ends = np.array([[arc.center[0] + arc.radius * math.cos(t),
                      arc.center[1] + arc.radius * math.sin(t)]
                     for t in (arc.start_angle, arc.end_angle)])
    to_end = np.minimum(np.linalg.norm(points - ends[0], axis=1),
                        np.linalg.norm(points - ends[1], axis=1))
    centreline = np.where(angle <= arc.span, radial, to_end)
    return np.maximum(centreline - arc.width / 2.0, 0.0)
