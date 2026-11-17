"""Goal-post observations from detector boxes.

A box does not say where a post stands. Two facts about posts make it usable:

1. A post is VERTICAL. With the camera pitched down, vertical edges are not vertical in the
   image: they all point at the vertical vanishing point (where the camera's own plumb line
   would appear, far below the image), so a post away from the image centre leans outward and
   its box is much wider than the post. The foot is therefore at the INNER bottom corner of
   the box, not at the middle of its bottom edge. `foot_pixel` recovers the foot column and
   the post's apparent width from the box and the vanishing point, which follows from the
   robot's estimated camera orientation alone.
2. Every point of a vertical post has the same bearing. So even when the foot is cut off or
   hidden, the lowest visible point on the post's axis still gives its direction.

Projecting the foot onto the ground gives the post's position; when the foot cannot be
trusted only the bearing is kept. The foot is NOT trusted when

* the box touches the bottom image border (the post continues below the image);
* the projected foot is above the horizon or absurdly far away;
* the post's apparent width is far from what a post of the known diameter would show at that
  range, or, with the whole post inside the image, the box is much shorter than a post of the
  known height (the box shows a part of the post, or the foot is hidden);
* what the segmentation shows just under the foot is neither turf nor line (something stands
  in front of the foot).

A robot or ball box overlapping the foot inflates the range uncertainty. Boxes that are not
post-shaped at all, or whose foot lands closer than the camera can see, are rejected.

The post's known diameter and height come from the field description, which the robot has.
Which of the four posts was seen is not decided here and cannot be: the localizer treats every
goal-post observation as "one of the posts on the map".
"""
from __future__ import annotations

import math

import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ProjectionLimits, ProjectionStatus,
    project_to_ground)
from robosoccer_perception.inference.types import Detection
from robosoccer_perception.perception.config import GoalPostConfig, MeasurementNoiseConfig
from robosoccer_perception.perception.types import GoalPostFeature
from robosoccer_perception.perception.uncertainty import point_uncertainty

_FALLBACK_RANGE = 3.0       # metres: where a bearing-only post is drawn when nothing better is known


def _foot_strip_overlap(post_box, other_box) -> float:
    """Share of the post's foot strip (its lowest fifth, widened) that another box covers."""
    x0, y0, x1, y1 = post_box
    width, height = x1 - x0, y1 - y0
    strip = (x0 - 0.5 * width, y1 - 0.2 * height, x1 + 0.5 * width, y1 + 0.05 * height)
    ox0, oy0, ox1, oy1 = other_box
    overlap = max(0.0, min(strip[2], ox1) - max(strip[0], ox0)) \
        * max(0.0, min(strip[3], oy1) - max(strip[1], oy0))
    area = (strip[2] - strip[0]) * (strip[3] - strip[1])
    return overlap / area if area > 0 else 0.0


def _ground_fraction_below(mask: np.ndarray, u: float, v: float, window: tuple[int, int]
                           ) -> float | None:
    """Share of a small patch just under pixel (u, v) that is turf or line; None if the patch
    is outside the image."""
    height, width = mask.shape
    half = window[0] // 2
    row0, row1 = int(math.floor(v)) + 1, int(math.floor(v)) + 1 + window[1]
    col0, col1 = int(round(u)) - half, int(round(u)) + half + 1
    row1, col0, col1 = min(row1, height), max(col0, 0), min(col1, width)
    if row0 >= row1 or col0 >= col1:
        return None
    patch = mask[row0:row1, col0:col1]
    return float(np.isin(patch, (schema.FIELD, schema.FIELD_LINE)).mean())


def vertical_vanishing_point(calibration: EstimatedCameraCalibration,
                             pose: EstimatedCameraPose) -> np.ndarray | None:
    """Pixel where images of vertical lines converge (the image of "straight down"), or None
    when the camera does not look below the horizon enough for it to be finite."""
    down = pose.T_base_camera[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    if down[2] < 1e-3:
        return None
    k = calibration.intrinsics
    return np.array([k.fx * down[0] / down[2] + k.cx, k.fy * down[1] / down[2] + k.cy])


def foot_pixel(box, vanishing_point: np.ndarray | None, image_width: int, margin: float,
               expected_width: float | None = None) -> tuple[float, float, float]:
    """``(u, v, width)`` of a leaning vertical post inside its box: the pixel on the post's
    axis at the bottom edge of the box (pixel-centre coordinates) and the post's apparent
    width in pixels.

    The box of a leaning post spans its lean plus its width. With the vanishing point
    ``(u_vp, v_vp)`` and the box ``(x0, y0, x1, y1)``, and the foot on the side of the box
    nearer the vanishing point::

        lean  = (u_foot - u_vp) * (y1 - y0) / (v_vp - y1)
        x1 - x0 = lean + width,     u_foot = x0 + width / 2      (post right of the VP)

    which is solved for ``width`` and ``u_foot``. If the outer side of the box is cut by the
    image border the box no longer spans the whole lean; ``expected_width`` (from the post's
    known diameter) is then used for the width instead.
    """
    x0, y0, x1, y1 = box
    centre = 0.5 * (x0 + x1)
    if vanishing_point is None or vanishing_point[1] - y1 < 1.0:
        return centre - 0.5, y1 - 0.5, x1 - x0
    u_vp = vanishing_point[0] + 0.5                 # to pixel-edge coordinates, like the box
    slope = (y1 - y0) / (vanishing_point[1] + 0.5 - y1)
    right_of_vp = centre >= u_vp
    inner = x0 if right_of_vp else x1
    outer_cut = (x1 >= image_width - margin) if right_of_vp else (x0 <= margin)
    offset = abs(inner - u_vp)
    width = ((x1 - x0) - offset * slope) / (1.0 + 0.5 * slope)
    if outer_cut or width < 1.0:
        width = expected_width if expected_width else max(width, 1.0)
    width = float(np.clip(width, 1.0, x1 - x0))
    u_foot = inner + 0.5 * width if right_of_vp else inner - 0.5 * width
    return u_foot - 0.5, y1 - 0.5, width


def extract_goal_posts(detections, mask: np.ndarray | None, image_size: tuple[int, int],
                       calibration: EstimatedCameraCalibration, pose: EstimatedCameraPose,
                       post_diameter: float, post_height: float, cfg: GoalPostConfig,
                       noise: MeasurementNoiseConfig
                       ) -> tuple[list[GoalPostFeature], dict[str, int]]:
    """Goal-post features of one frame, and counts of what was rejected and why.

    ``detections`` is every detection of the frame (posts are picked out; robots and the ball
    are used as possible occluders); ``mask`` is the predicted semantic mask or None;
    ``image_size`` is ``(width, height)``.
    """
    width, height = image_size
    k = calibration.intrinsics
    camera_xy = pose.T_base_camera[:2, 3]
    T_camera_base = tf.invert_transform(pose.T_base_camera)
    occluders = [d.bbox_xyxy for d in detections if d.class_id != schema.GOAL_POST]
    vanishing_point = vertical_vanishing_point(calibration, pose)
    loose = ProjectionLimits(min_depression_deg=0.5, max_range=cfg.reject_range, min_range=0.0)
    features: list[GoalPostFeature] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for detection in detections:
        if detection.class_id != schema.GOAL_POST:
            continue
        if detection.confidence < cfg.min_confidence:
            reject("low_confidence")
            continue
        x0, y0, x1, y1 = detection.bbox_xyxy
        box_width, box_height = x1 - x0, y1 - y0
        margin = cfg.border_margin_px
        cut_left, cut_right = x0 <= margin, x1 >= width - margin
        cut_top, cut_bottom = y0 <= margin, y1 >= height - margin
        cut_side = cut_left or cut_right
        if box_width > 1.5 * box_height and not (cut_top or cut_bottom):
            reject("not_post_shaped")
            continue
        # Box edges are pixel-edge coordinates; the camera model uses pixel centres.
        u_foot, v_foot, post_width = foot_pixel(detection.bbox_xyxy, vanishing_point, width,
                                                margin)
        foot = np.array([u_foot, v_foot])
        ground = project_to_ground(foot[None], calibration, pose, loose)
        if cut_side and ground.valid[0]:
            # The lean is not fully inside the box: take the width a post would have there.
            depth = float((T_camera_base @ [*ground.points[0], 0.0, 1.0])[2])
            u_foot, v_foot, post_width = foot_pixel(
                detection.bbox_xyxy, vanishing_point, width, margin,
                expected_width=k.fx * post_diameter / max(depth, 1e-6))
            foot = np.array([u_foot, v_foot])
            ground = project_to_ground(foot[None], calibration, pose, loose)
        status = int(ground.status[0])
        flags: list[str] = []
        range_valid = True
        ray = (k.ray(foot) @ pose.T_base_camera[:3, :3].T)
        bearing = math.atan2(ray[1], ray[0])
        rough_range = _FALLBACK_RANGE

        if status & (ProjectionStatus.ABOVE_HORIZON | ProjectionStatus.NEAR_HORIZON
                     | ProjectionStatus.UNSTABLE):
            flags.append("foot_above_horizon")
            range_valid = False
            position = None
        elif status & ProjectionStatus.TOO_FAR:
            reject("implausibly_far")
            continue
        else:
            # The lowest visible point of a round post is its near side: the axis is one
            # radius farther along the viewing direction.
            contact = ground.points[0]
            outward = contact - camera_xy
            position = contact + 0.5 * post_diameter * outward / max(np.linalg.norm(outward), 1e-9)
            rough_range = float(np.linalg.norm(position))
            if rough_range < cfg.min_range:
                reject("implausibly_close")
                continue
            depth = float((T_camera_base @ [contact[0], contact[1], 0.0, 1.0])[2])   # optical z
            expected_width = k.fx * post_diameter / max(depth, 1e-6)
            if not cut_side:
                ratio = post_width / expected_width
                if not cfg.width_ratio_limits[0] <= ratio <= cfg.width_ratio_limits[1]:
                    flags.append("width_mismatch")
                    range_valid = False
            if not (cut_top or cut_bottom or cut_side):
                top_camera = T_camera_base @ [*position, post_height, 1.0]
                if top_camera[2] > 0.0:
                    expected_height = foot[1] - (k.fy * top_camera[1] / top_camera[2] + k.cy)
                    if expected_height > 4.0 and not cfg.height_ratio_min <= \
                            box_height / expected_height <= 1.0 / cfg.height_ratio_min:
                        flags.append("height_mismatch")
                        range_valid = False
            if rough_range > cfg.max_range:
                flags.append("beyond_range")
                range_valid = False

        if cut_bottom:
            flags.append("cut_bottom")
            range_valid = False
        if cut_side:
            flags.append("cut_side")
        if mask is not None and not cut_bottom:
            fraction = _ground_fraction_below(mask, foot[0], foot[1], cfg.foot_window_px)
            if fraction is not None and fraction < cfg.foot_min_ground_fraction:
                flags.append("foot_not_on_turf")
                range_valid = False
        occluded = any(_foot_strip_overlap(detection.bbox_xyxy, other) >= cfg.occluder_overlap
                       for other in occluders)
        if occluded:
            flags.append("occluder_near_foot")

        if not range_valid:
            # Bearing only: a point on the ray at a rough range, with a range uncertainty so
            # large that it does not constrain the localizer.
            if not cut_side and post_width >= 2.0:
                depth = k.fx * post_diameter / post_width
                rough_range = math.sqrt(max(depth ** 2 - pose.height ** 2, 0.25))
            rough_range = float(np.clip(rough_range, 0.5, cfg.max_range))
            position = camera_xy + rough_range * np.array([math.cos(bearing), math.sin(bearing)])

        uncertainty = point_uncertainty(
            position, calibration, pose, noise, pixel=foot if range_valid else None,
            pixel_std=tuple(noise.post_pixel_std),
            range_std_floor=noise.post_range_std_floor,
            bearing_std_floor=math.radians(noise.post_bearing_std_floor_deg),
            confidence=detection.confidence)
        range_std, bearing_std = uncertainty.range_std, uncertainty.bearing_std
        if cut_side:
            # The width had to be assumed: the axis may be off by a good part of it.
            bearing_std = math.hypot(bearing_std, 0.5 * post_diameter / max(rough_range, 0.5))
        if not range_valid:
            range_std = max(noise.bearing_only_range_std_min,
                            noise.bearing_only_range_std_fraction * rough_range)
        elif occluded:
            range_std *= noise.post_inflation_flagged
        uncertainty = type(uncertainty)(range_std, bearing_std, uncertainty.pixel_covariance,
                                        uncertainty.camera_covariance)
        features.append(GoalPostFeature(foot, np.asarray(position, float), uncertainty,
                                        float(detection.confidence), range_valid, tuple(flags),
                                        tuple(float(v) for v in detection.bbox_xyxy)))
    return features, rejected


def goal_post_detections(detections) -> list[Detection]:
    return [d for d in detections if d.class_id == schema.GOAL_POST]
