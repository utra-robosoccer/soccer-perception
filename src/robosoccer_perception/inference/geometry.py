"""Light geometric utilities on top of the vision outputs. INSPECTION TOOLS ONLY.

Phase 2 delivers images -> detections and masks. Turning those into the relative observations
the particle filter consumes (line points, junctions, the centre circle, goal-post bearings) is
Phase 3. What is here is the minimum needed to look at a mask geometrically and to check that
it is usable for that step:

* `skeletonize`           thin a line mask to one-pixel-wide centrelines;
* `sample_line_pixels`    pick a bounded, evenly spread set of line pixels;
* `pixels_to_ground`      intersect pixel rays with the ground plane (needs the camera pose
                          relative to the ground, which the robot knows from its kinematics);
* `ground_distance_map`   ground distance seen at every pixel (used to break metrics down by
                          distance).

Nothing here builds an `ObservationFrame`, associates features with landmarks or touches the
localizer. See docs/phase3_interfaces.md for how Phase 3 is expected to use these.

Pixel coordinates follow `geometry.camera`: ``(u, v)``, integer values at pixel centres.
"""
from __future__ import annotations

import numpy as np

from robosoccer_perception.geometry.camera import CameraIntrinsics


def skeletonize(mask: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Zhang-Suen thinning of a boolean mask to one-pixel-wide curves.

    Pure NumPy. Each pass peels one layer of boundary pixels without breaking connectivity, so
    a line ``w`` pixels wide needs about ``w / 2`` passes.
    """
    image = np.pad(np.asarray(mask, dtype=bool), 1).astype(np.uint8)

    def neighbours(img):
        # P2..P9: clockwise from the pixel above.
        return (img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:],
                img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2])

    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = neighbours(image)
            ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            count = sum(ring[:-1])
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
                              for i in range(8))
            if step == 0:
                rule = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                rule = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = ((image[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6)
                      & (transitions == 1) & rule)
            if remove.any():
                image[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1].astype(bool)


def sample_line_pixels(line_mask: np.ndarray, max_points: int = 200, thin: bool = True,
                       cell: int | None = None) -> np.ndarray:
    """Up to ``max_points`` line pixels ``(N, 2)`` as ``(u, v)``, spread over the image.

    The mask is thinned first (``thin``), then at most one pixel is kept per ``cell x cell``
    block; the block size grows until the count fits. Deterministic.
    """
    points_mask = skeletonize(line_mask) if thin else np.asarray(line_mask, dtype=bool)
    v, u = np.nonzero(points_mask)
    if len(u) == 0:
        return np.zeros((0, 2), dtype=int)
    cell = cell or 4
    while True:
        _, first = np.unique((v // cell) * (points_mask.shape[1] // cell + 1) + u // cell,
                             return_index=True)
        if len(first) <= max_points:
            break
        cell *= 2
    return np.column_stack([u[first], v[first]])


def pixels_to_ground(pixels_uv, intrinsics: CameraIntrinsics,
                     T_plane_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where the viewing rays of pixels meet the plane ``z = 0`` of a frame.

    ``T_plane_camera`` is the camera's optical frame expressed in a frame whose ``z = 0`` plane
    is the ground: `StereoCameraModel.T_base_camera` gives points in the robot's base frame
    (what localization wants), ``T_field_camera`` gives field coordinates (for evaluation).

    Returns ``(points, valid)``: ``points`` is ``(..., 2)`` ground coordinates and ``valid`` is
    False for pixels at or above the horizon, whose rays never reach the ground (their points
    are NaN).
    """
    rays = intrinsics.ray(np.asarray(pixels_uv, dtype=float)) @ T_plane_camera[:3, :3].T
    origin = T_plane_camera[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -origin[2] / rays[..., 2]
        valid = (rays[..., 2] < -1e-9) & (scale > 0)
        points = origin[:2] + scale[..., None] * rays[..., :2]
    points[~valid] = np.nan
    return points, valid


def ground_distance_map(intrinsics: CameraIntrinsics, T_plane_camera: np.ndarray) -> np.ndarray:
    """``(H, W)`` horizontal distance from the camera to the ground point seen at each pixel;
    ``inf`` at and above the horizon."""
    v, u = np.mgrid[0:intrinsics.height, 0:intrinsics.width]
    points, valid = pixels_to_ground(np.stack([u, v], axis=-1), intrinsics, T_plane_camera)
    distance = np.hypot(*np.moveaxis(points - T_plane_camera[:2, 3], -1, 0))
    return np.where(valid, distance, np.inf)
