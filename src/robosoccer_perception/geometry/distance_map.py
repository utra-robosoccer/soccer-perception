"""Field-line distance map: distance from any ground point to the nearest painted line.

The particle filter scores an observed line point by transforming it into the field frame with
a particle's pose and asking "how far is this from the nearest line?". Doing that analytically
for thousands of particles times dozens of points on every update would be slow, so the answer
is precomputed on a regular grid and looked up with bilinear interpolation.

The map is built from a `FieldGeometry` and cached in memory by the field's content hash and the
grid resolution, so changing any field dimension (or loading another field) gives a new map with
no manual step.

Memory and precision: values are float32 metres at grid nodes; the default 2.5 cm grid is
0.6 MB for the S field and 2.5 MB for the L field. The distance to paint is 1-Lipschitz, so
bilinear interpolation is at worst a fraction of the cell size away from the exact distance
(tests/test_field_geometry.py measures it).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from robosoccer_perception.geometry.field import FieldGeometry

DEFAULT_RESOLUTION = 0.025


@dataclasses.dataclass(frozen=True)
class DistanceMap:
    """Distances at grid nodes. Node (row j, column i) is at ``origin + resolution * (i, j)``."""
    origin: tuple[float, float]         # field-frame (x, y) of node (0, 0)
    resolution: float                   # metres between neighbouring nodes
    data: np.ndarray                    # float32, shape (ny, nx), metres
    field_hash: str

    @classmethod
    def from_field(cls, field: FieldGeometry,
                   resolution: float = DEFAULT_RESOLUTION) -> "DistanceMap":
        """Compute the map over the whole carpet (field plus border strip)."""
        half_x, half_y = field.half_extent
        nx = int(np.ceil(2.0 * half_x / resolution)) + 1
        ny = int(np.ceil(2.0 * half_y / resolution)) + 1
        xs = -half_x + resolution * np.arange(nx)
        ys = -half_y + resolution * np.arange(ny)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1)             # (ny, nx, 2)
        data = field.distance_to_paint(grid).astype(np.float32)
        data.setflags(write=False)
        return cls((-half_x, -half_y), float(resolution), data, field.content_hash())

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def memory_bytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def max_distance(self) -> float:
        return float(self.data.max())

    def lookup(self, points) -> np.ndarray:
        """Distance to the nearest line for field-frame points of shape (..., 2).

        Bilinear inside the map. A point outside the map gets the value at the nearest map
        edge plus its distance to that edge, so the result keeps growing away from the field.
        """
        points = np.asarray(points, dtype=float)
        return self.lookup_xy(points[..., 0], points[..., 1])

    def lookup_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`lookup` for separate x and y arrays of the same shape (the filter's hot path)."""
        ny, nx = self.data.shape
        gx = (x - self.origin[0]) / self.resolution
        gy = (y - self.origin[1]) / self.resolution
        cx = np.clip(gx, 0.0, nx - 1.0)
        cy = np.clip(gy, 0.0, ny - 1.0)
        x0 = np.minimum(cx.astype(np.intp), nx - 2)
        y0 = np.minimum(cy.astype(np.intp), ny - 2)
        tx = cx - x0
        ty = cy - y0
        flat = self.data.reshape(-1)
        index = y0 * nx + x0
        top = flat.take(index) * (1.0 - tx) + flat.take(index + 1) * tx
        index += nx
        bottom = flat.take(index) * (1.0 - tx) + flat.take(index + 1) * tx
        inside = top * (1.0 - ty) + bottom * ty
        return inside + np.hypot(gx - cx, gy - cy) * self.resolution


_CACHE: dict[tuple[str, float], DistanceMap] = {}


def get_distance_map(field: FieldGeometry,
                     resolution: float = DEFAULT_RESOLUTION) -> DistanceMap:
    """The distance map of a field, computed on first use and then reused.

    The cache key is the field's content hash, so a field with different dimensions never gets
    a stale map.
    """
    key = (field.content_hash(), float(resolution))
    if key not in _CACHE:
        _CACHE[key] = DistanceMap.from_field(field, resolution)
    return _CACHE[key]
