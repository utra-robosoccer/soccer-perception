"""Field-line pixels: from the predicted line-probability map to a thinned set of centreline
pixels, and thinning of their ground points.

Method
------
1. Threshold the line probability and drop small connected blobs (isolated false positives).
2. Find the centreline with a distance transform: a line pixel is kept when no neighbour is
   farther from the line's edge than it is. For a painted line seen as a band several pixels
   wide this is the ridge along its middle; for a one-pixel-wide distant line it is the line.
   This replaces morphological thinning, which costs tens of milliseconds in NumPy.
3. Keep one centreline pixel per small image cell, the most central one.

The result is still dense where lines are close to the robot (a few centimetres apart on the
ground). `thin_on_ground` then enforces a minimum distance in metres between the points that
are kept, which is what matters for the localizer: neighbouring points carry almost the same
information.

Pixel coordinates are ``(u, v)`` of the INPUT image at pixel centres, whatever resolution the
extraction worked at.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from robosoccer_perception.perception.config import LineConfig

_KERNEL = np.ones((3, 3), np.uint8)


@dataclasses.dataclass(frozen=True)
class CentrelinePixels:
    pixels: np.ndarray          # (N, 2) float u, v in the input image
    confidence: np.ndarray      # (N,) line probability at the pixel
    half_width_px: np.ndarray   # (N,) distance to the edge of the line band, input-image pixels
    num_line_pixels: int        # pixels above the threshold (work resolution)
    num_removed_pixels: int     # of those, dropped as small blobs
    num_centreline_pixels: int  # ridge pixels before the one-per-cell reduction
    scale: float                # input pixels per work pixel

    def __len__(self) -> int:
        return len(self.pixels)


def reduce_for_work(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """``image`` reduced to at most ``max_width`` columns, and input pixels per work pixel."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image, 1.0
    scale = width / max_width
    reduced = cv2.resize(image, (max_width, max(1, int(round(height / scale)))),
                         interpolation=cv2.INTER_AREA)
    return reduced, scale


def clean_line_mask(line_confidence: np.ndarray, cfg: LineConfig) -> tuple[np.ndarray, int, int]:
    """Binary line mask (uint8) without small blobs; also the pixel counts before / removed."""
    binary = (line_confidence >= cfg.min_confidence).astype(np.uint8)
    total = int(binary.sum())
    removed = 0
    if total and cfg.min_component_area_px > 1:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < cfg.min_component_area_px
        small[0] = False                                    # label 0 is "not line"
        if small.any():
            drop = small[labels]
            removed = int(drop.sum())
            binary[drop] = 0
    return binary, total, removed


def centreline_pixels(line_confidence: np.ndarray, cfg: LineConfig | None = None,
                      max_work_width: int = 640) -> CentrelinePixels:
    """Thinned, spread-out centreline pixels of a line-probability map ``(H, W)`` in [0, 1]."""
    cfg = cfg or LineConfig()
    work, scale = reduce_for_work(np.ascontiguousarray(line_confidence, dtype=np.float32),
                                  max_work_width)
    binary, total, removed = clean_line_mask(work, cfg)
    empty = np.zeros(0)
    if total - removed == 0:
        return CentrelinePixels(np.zeros((0, 2)), empty, empty, total, removed, 0, scale)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    ridge = (binary > 0) & (distance >= cv2.dilate(distance, _KERNEL) - 1e-4)
    v, u = np.nonzero(ridge)
    cell = max(1, int(cfg.pixel_cell_px))
    key = (v // cell) * (work.shape[1] // cell + 1) + u // cell
    # Within a cell prefer the most central pixel, then the most confident one.
    order = np.lexsort((-work[v, u], -distance[v, u], key))
    chosen = order[np.unique(key[order], return_index=True)[1]]
    u, v = u[chosen], v[chosen]
    pixels = np.column_stack([(u + 0.5) * scale - 0.5, (v + 0.5) * scale - 0.5])
    return CentrelinePixels(pixels, work[v, u].astype(float), distance[v, u] * scale, total,
                            removed, int(ridge.sum()), scale)


def thin_to_grid(points: np.ndarray, priority: np.ndarray, cell: float) -> np.ndarray:
    """Indices of at most one ground point (N, 2) per ``cell`` x ``cell`` square: the one with
    the highest ``priority``. Fast, and enough to even out the density before fitting; two
    kept points can still be close together across a cell border."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=np.intp)
    index = np.floor(points / cell).astype(np.int64)
    key = (index[:, 0] - index[:, 0].min()) * (int(np.ptp(index[:, 1])) + 1) \
        + (index[:, 1] - index[:, 1].min())
    order = np.lexsort((-np.asarray(priority, dtype=float), key))
    return np.sort(order[np.unique(key[order], return_index=True)[1]])


def thin_on_ground(points: np.ndarray, priority: np.ndarray, spacing: float,
                   max_points: int | None = None) -> np.ndarray:
    """Indices of a subset of ground points (N, 2) in which no two are closer than ``spacing``.

    Points are taken in order of decreasing ``priority``; each one that is kept rules out
    every other point within ``spacing`` metres of it. With ``max_points`` the spacing grows
    until the subset fits. Deterministic; the indices are returned in ascending order.
    """
    points = np.asarray(points, dtype=float)
    count = len(points)
    if count == 0:
        return np.zeros(0, dtype=np.intp)
    order = np.argsort(-np.asarray(priority, dtype=float), kind="stable")
    delta = points[:, None, :] - points[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    while True:
        available = np.ones(count, dtype=bool)
        chosen = []
        limit = spacing * spacing
        for index in order:
            if available[index]:
                chosen.append(index)
                available &= squared[index] >= limit
        if max_points is None or len(chosen) <= max_points:
            return np.sort(np.asarray(chosen, dtype=np.intp))
        spacing *= 1.25
