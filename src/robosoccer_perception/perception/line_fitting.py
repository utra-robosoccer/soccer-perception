"""Straight segments and the centre circle, fitted to line points on the ground.

Fitting is done in the ground plane, not in the image: there field lines are straight, meet at
right angles, and the centre circle is a circle of known radius, so every threshold is a
length in metres.

* `fit_segments`: sequential RANSAC. Find the line with the most supporting points, refit it
  by total least squares, cut it where the support has a gap, keep the pieces that are long
  enough, remove its points, repeat.
* `fit_circle`: RANSAC for a circle of KNOWN radius (two points give two candidate centres),
  refined by Gauss-Newton on the centre.

Each point comes with its own tolerance (its pixel noise on the ground), so far points, which
are spread more by one pixel of error, are judged more leniently than near ones. Camera-pose
error is not part of the tolerance: it moves all points of a line together and leaves the line
straight.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from robosoccer_perception.perception.config import CircleConfig, SegmentConfig

_MIN_SAMPLE_BASELINE = 0.25     # metres between the two points that define a line hypothesis


@dataclasses.dataclass(frozen=True)
class FittedSegment:
    start: np.ndarray           # (2,)
    end: np.ndarray             # (2,)
    indices: np.ndarray         # indices of the supporting points
    rms: float                  # metres: scatter of the points across the line

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / max(self.length, 1e-9)

    @property
    def normal(self) -> np.ndarray:
        direction = self.direction
        return np.array([-direction[1], direction[0]])


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line: (a point on it, its unit direction)."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vt[0]


def _segment_from(points: np.ndarray, indices: np.ndarray) -> FittedSegment:
    centre, direction = _fit_line(points[indices])
    along = (points[indices] - centre) @ direction
    across = (points[indices] - centre) @ np.array([-direction[1], direction[0]])
    return FittedSegment(centre + along.min() * direction, centre + along.max() * direction,
                         np.asarray(indices), float(np.sqrt(np.mean(across ** 2))))


def fit_segments(points: np.ndarray, tolerance: np.ndarray, cfg: SegmentConfig,
                 rng: np.random.Generator, gap_allowance: np.ndarray | None = None
                 ) -> list[FittedSegment]:
    """Straight segments supported by ground points (M, 2).

    ``tolerance`` (M,) is how far each point may lie from a line and still support it.
    ``gap_allowance`` (M,) is added to ``cfg.max_gap`` at each point: far from the robot the
    points are naturally farther apart, because one image pixel covers more ground.
    """
    points = np.asarray(points, dtype=float)
    count = len(points)
    remaining = np.ones(count, dtype=bool)
    allowance = np.zeros(count) if gap_allowance is None else np.asarray(gap_allowance, float)
    segments: list[FittedSegment] = []
    for _ in range(2 * cfg.max_segments):
        pool = np.flatnonzero(remaining)
        if len(pool) < cfg.min_points or len(segments) >= cfg.max_segments:
            break
        candidates, tol = points[pool], tolerance[pool]
        first = rng.integers(len(pool), size=cfg.iterations)
        second = rng.integers(len(pool), size=cfg.iterations)
        delta = candidates[second] - candidates[first]
        baseline = np.linalg.norm(delta, axis=1)
        usable = baseline >= _MIN_SAMPLE_BASELINE
        if not usable.any():
            break
        first, delta, baseline = first[usable], delta[usable], baseline[usable]
        normals = np.stack([-delta[:, 1], delta[:, 0]], axis=1) / baseline[:, None]
        offsets = np.einsum("kj,kj->k", candidates[first], normals)
        distance = np.abs(candidates @ normals.T - offsets)           # (M, K)
        support = (distance <= tol[:, None]).sum(axis=0)
        best = int(np.argmax(support))
        if support[best] < cfg.min_points:
            break
        members = pool[distance[:, best] <= tol]
        for _ in range(2):                                            # refit, re-select
            centre, direction = _fit_line(points[members])
            normal = np.array([-direction[1], direction[0]])
            members = pool[np.abs((candidates - centre) @ normal) <= tol]
            if len(members) < cfg.min_points:
                break
        remaining[members] = False
        if len(members) < cfg.min_points:
            continue
        along = (points[members] - centre) @ direction
        order = np.argsort(along)
        sorted_along, sorted_members = along[order], members[order]
        allowed = cfg.max_gap + np.maximum(allowance[sorted_members][:-1],
                                           allowance[sorted_members][1:])
        breaks = np.flatnonzero(np.diff(sorted_along) > allowed) + 1
        for run in np.split(np.arange(len(sorted_members)), breaks):
            if len(run) >= cfg.min_points and \
                    sorted_along[run[-1]] - sorted_along[run[0]] >= cfg.min_length:
                segments.append(_segment_from(points, sorted_members[run]))
    return _merge_collinear(points, segments, cfg)


def _merge_collinear(points: np.ndarray, segments: list[FittedSegment],
                     cfg: SegmentConfig) -> list[FittedSegment]:
    """Join pieces of one physical line that RANSAC returned separately."""
    max_angle = math.radians(cfg.merge_angle_deg)
    merged = True
    while merged:
        merged = False
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                a, b = segments[i], segments[j]
                cosine = abs(float(a.direction @ b.direction))
                if math.acos(min(1.0, cosine)) > max_angle:
                    continue
                if max(abs(float((b.start - a.start) @ a.normal)),
                       abs(float((b.end - a.start) @ a.normal))) > cfg.merge_offset:
                    continue
                t = sorted(float((p - a.start) @ a.direction) for p in (b.start, b.end))
                gap = max(t[0] - a.length, -t[1], 0.0)        # 0 when they overlap
                if gap > cfg.merge_gap:
                    continue
                segments[i] = _segment_from(points, np.concatenate([a.indices, b.indices]))
                del segments[j]
                merged = True
                break
            if merged:
                break
    return segments


# --------------------------------------------------------------------------------------------
# Circle of known radius
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FittedCircle:
    center: np.ndarray          # (2,)
    radius: float
    indices: np.ndarray         # supporting points
    arc_coverage_deg: float     # occupied angular bins times the bin size
    density: float              # occupied bins / bins in the smallest arc holding all support
    rms: float                  # metres: radial scatter of the support
    purity: float               # support / all points within a wider band around the circle
    accepted: bool
    reasons: tuple[str, ...]    # why it was not accepted


def _coverage(angles: np.ndarray, bin_deg: float) -> tuple[float, float]:
    """(covered degrees, density) of a set of directions around a centre."""
    bins = max(4, int(round(360.0 / bin_deg)))
    occupied = np.zeros(bins, dtype=bool)
    occupied[np.floor((angles % (2.0 * math.pi)) / (2.0 * math.pi) * bins).astype(int) % bins] = True
    filled = int(occupied.sum())
    if filled == bins:
        return 360.0, 1.0
    # Longest run of empty bins (circular): the support spans everything outside it.
    doubled = np.concatenate([occupied, occupied])
    longest = run = 0
    for value in doubled:
        run = 0 if value else run + 1
        longest = max(longest, run)
    span = bins - min(longest, bins)
    return filled * 360.0 / bins, filled / max(span, 1)


def fit_circle(points: np.ndarray, tolerance: np.ndarray, radius: float, cfg: CircleConfig,
               rng: np.random.Generator) -> FittedCircle | None:
    """The best circle of the given radius through ground points (M, 2), or None if there are
    too few points to try. The result says whether it passed the acceptance checks."""
    points = np.asarray(points, dtype=float)
    count = len(points)
    if count < cfg.min_inliers:
        return None
    first = rng.integers(count, size=cfg.iterations)
    second = rng.integers(count, size=cfg.iterations)
    delta = points[second] - points[first]
    chord = np.linalg.norm(delta, axis=1)
    usable = (chord >= 0.3 * radius) & (chord <= 1.95 * radius)
    if not usable.any():
        return None
    first, delta, chord = first[usable], delta[usable], chord[usable]
    middle = points[first] + 0.5 * delta
    perpendicular = np.stack([-delta[:, 1], delta[:, 0]], axis=1) / chord[:, None]
    rise = np.sqrt(radius ** 2 - (0.5 * chord) ** 2)[:, None]
    centres = np.concatenate([middle + rise * perpendicular, middle - rise * perpendicular])
    radial = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)       # (M, 2K)
    inside = np.abs(radial - radius) <= tolerance[:, None]
    support = inside.sum(axis=0)
    best_score, best_centre = (-1.0, -1), None
    for index in np.argsort(-support)[:12]:
        if support[index] < 3:
            break
        offset = points[inside[:, index]] - centres[index]
        covered, _ = _coverage(np.arctan2(offset[:, 1], offset[:, 0]), cfg.angular_bin_deg)
        if (covered, int(support[index])) > best_score:
            best_score, best_centre = (covered, int(support[index])), centres[index]
    if best_centre is None:
        return None

    centre = best_centre.copy()
    members = np.flatnonzero(np.abs(np.linalg.norm(points - centre, axis=1) - radius) <= tolerance)
    for _ in range(4):                                  # Gauss-Newton on the centre
        if len(members) < 3:
            break
        offset = points[members] - centre
        distance = np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
        jacobian = -offset / distance[:, None]
        step, *_ = np.linalg.lstsq(jacobian, -(distance - radius), rcond=None)
        centre = centre + step
        members = np.flatnonzero(
            np.abs(np.linalg.norm(points - centre, axis=1) - radius) <= tolerance)
    if len(members) < 3:
        return None
    offset = points[members] - centre
    covered, density = _coverage(np.arctan2(offset[:, 1], offset[:, 0]), cfg.angular_bin_deg)
    rms = float(np.sqrt(np.mean((np.linalg.norm(offset, axis=1) - radius) ** 2)))
    reasons = []
    if len(members) < cfg.min_inliers:
        reasons.append("few_inliers")
    if covered < cfg.min_arc_deg:
        reasons.append("short_arc")
    if density < cfg.min_density:
        reasons.append("sparse_arc")
    if float(np.linalg.norm(centre)) > cfg.max_center_range:
        reasons.append("beyond_range")
    # A circle of this radius can be laid TANGENT to two or three straight lines (the nested
    # penalty and goal areas) and collect support where it touches them. Those lines carry on
    # past the touching point, just outside the tolerance; the paint of a real circle does
    # not. So ask how much of everything near the circle actually supports it.
    band = np.maximum(cfg.purity_band, 2.5 * tolerance)       # wider where points are noisier
    nearby = int(np.count_nonzero(
        np.abs(np.linalg.norm(points - centre, axis=1) - radius) <= band))
    purity = len(members) / max(nearby, 1)
    if purity < cfg.min_purity:
        reasons.append("tangent_lines")
    return FittedCircle(centre, float(radius), members, covered, density, rms, purity,
                        not reasons, tuple(reasons))
