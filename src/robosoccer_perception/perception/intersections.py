"""L, T and X junctions from fitted line segments.

Two segments that meet at about a right angle form a junction at the crossing of their lines.
Its class follows from where that crossing lies on each segment:

* the segment continues on both sides of it  -> the line goes THROUGH the junction;
* the segment starts or stops at it          -> the line ENDS at the junction.

    end + end -> L        through + end -> T        through + through -> X

The hard part is not finding crossings but not inventing classes. A line that leaves the image,
disappears behind a robot or runs beyond the usable range also "ends", and would turn a T into
an L or an X into a T. So an end only counts when the ground just past it is positively seen
to be turf (`probe`). If that cannot be established the junction is reported without a class,
and the localizer does not use it.

Every length is in metres on the ground. Nothing here knows which junction of the field was
seen: only its class and where it is relative to the robot.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

import numpy as np

from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.perception.config import IntersectionConfig
from robosoccer_perception.perception.line_fitting import FittedSegment

#: `probe` results: what the image shows at a ground point.
TURF, NOT_TURF, UNSEEN = 1, 0, -1
Probe = Callable[[np.ndarray], np.ndarray]      # ground points (N, 2) -> (N,) of the above


@dataclasses.dataclass(frozen=True)
class JunctionCandidate:
    type: LandmarkType | None
    position: np.ndarray            # (2,) base frame
    orientation: float | None       # T: direction of the stem; L: bisector of the two arms
    arm_length: float               # metres: the shortest arm the class relies on
    num_points: int                 # support of the weaker of the two segments
    flags: tuple[str, ...] = ()


def _role(segment: FittedSegment, crossing: np.ndarray, cfg: IntersectionConfig
          ) -> tuple[str | None, np.ndarray | None, float]:
    """How a segment relates to a crossing point on its line.

    Returns ``(role, body_direction, arm)``: role is ``through``, ``end``, ``unclear`` or None
    (the crossing is not on this segment at all); ``body_direction`` points from the crossing
    into the segment for an ``end``; ``arm`` is the length the role rests on.
    """
    before = float((crossing - segment.start) @ segment.direction)     # segment before the crossing
    after = segment.length - before                                     # ... and after it
    if before >= cfg.min_arm and after >= cfg.min_arm:
        return "through", None, min(before, after)
    for near, far, body in ((before, after, segment.direction),
                            (after, before, -segment.direction)):
        if far >= cfg.min_arm and -cfg.max_end_gap <= near <= cfg.end_tolerance:
            return "end", body, far
    if -cfg.max_end_gap <= before <= segment.length + cfg.max_end_gap:
        return "unclear", None, min(max(before, 0.0), max(after, 0.0))
    return None, None, 0.0


def find_junctions(segments: list[FittedSegment], probe: Probe, cfg: IntersectionConfig
                   ) -> list[JunctionCandidate]:
    """Junction candidates of a set of segments (base frame)."""
    tolerance = math.radians(cfg.angle_tolerance_deg)
    candidates: list[JunctionCandidate] = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a, b = segments[i], segments[j]
            cross = float(a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0])
            if abs(math.asin(min(1.0, abs(cross))) - math.pi / 2) > tolerance:
                continue
            offset = b.start - a.start
            t = float(offset[0] * b.direction[1] - offset[1] * b.direction[0]) / cross
            crossing = a.start + t * a.direction
            if float(np.linalg.norm(crossing)) > cfg.max_range:
                continue
            roles = [_role(segment, crossing, cfg) for segment in (a, b)]
            if any(role is None for role, _, _ in roles):
                continue
            flags: list[str] = []
            if int(probe(crossing[None])[0]) == UNSEEN:
                continue                                    # extrapolated outside the image
            for role, body, _ in roles:
                if role == "unclear":
                    flags.append("short_arm")
                elif role == "end":
                    beyond = crossing - np.outer(cfg.probe_distances, body)
                    seen = probe(beyond)
                    if (seen == UNSEEN).any():
                        flags.append("end_not_observed")
                    elif (seen != TURF).any():
                        flags.append("end_not_on_turf")
            kinds = sorted(role for role, _, _ in roles)
            kind, orientation = None, None
            if not flags:
                if kinds == ["end", "end"]:
                    kind = LandmarkType.L_INTERSECTION
                    bisector = roles[0][1] + roles[1][1]
                    orientation = float(math.atan2(bisector[1], bisector[0]))
                elif kinds == ["end", "through"]:
                    kind = LandmarkType.T_INTERSECTION
                    stem = next(body for role, body, _ in roles if role == "end")
                    orientation = float(math.atan2(stem[1], stem[0]))
                else:
                    kind = LandmarkType.X_INTERSECTION
            candidates.append(JunctionCandidate(
                kind, crossing, orientation, min(arm for _, _, arm in roles),
                min(len(a.indices), len(b.indices)), tuple(flags)))
    return _merge(candidates, cfg.merge_distance)


def circle_line_junctions(center: np.ndarray, radius: float, segments: list[FittedSegment],
                          cfg: IntersectionConfig) -> list[JunctionCandidate]:
    """X junctions where a straight line crosses an accepted centre circle.

    The only X junctions of a field are the two points where the halfway line crosses the
    centre circle. Once a circle has been found, a fitted segment that passes through its
    centre and reaches past the circle on both sides places them: ``centre +- radius`` along
    the line. A segment that stops inside the circle places neither.
    """
    found: list[JunctionCandidate] = []
    for segment in segments:
        offset = center - segment.start
        across = abs(float(offset[0] * segment.direction[1] - offset[1] * segment.direction[0]))
        if across > 0.15 * radius:
            continue
        along = float(offset @ segment.direction)
        for sign in (-1.0, 1.0):
            position_along = along + sign * radius
            arm = min(position_along, segment.length - position_along)
            crossing = segment.start + position_along * segment.direction
            if arm >= cfg.min_arm and float(np.linalg.norm(crossing)) <= cfg.max_range:
                found.append(JunctionCandidate(LandmarkType.X_INTERSECTION, crossing, None, arm,
                                               len(segment.indices), ("circle_and_line",)))
    return found


def _merge(candidates: list[JunctionCandidate], distance: float) -> list[JunctionCandidate]:
    """One junction per place. Candidates that disagree about the class cancel to "unknown"."""
    merged: list[JunctionCandidate] = []
    used = [False] * len(candidates)
    for i, candidate in enumerate(candidates):
        if used[i]:
            continue
        group = [candidate]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if not used[j] and np.linalg.norm(candidates[j].position - candidate.position) <= distance:
                group.append(candidates[j])
                used[j] = True
        typed = [c for c in group if c.type is not None]
        if typed and len({c.type for c in typed}) == 1:
            merged.append(max(typed, key=lambda c: c.arm_length))
        elif typed:
            best = max(typed, key=lambda c: c.arm_length)
            merged.append(dataclasses.replace(best, type=None, orientation=None,
                                              flags=(*best.flags, "conflicting_classes")))
        else:
            merged.append(max(group, key=lambda c: c.arm_length))
    return merged
