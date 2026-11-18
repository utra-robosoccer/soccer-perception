"""Geometric feature extraction: one `VisionResult` in, one `FeatureSet` out.

    VisionResult (boxes, masks)  +  the robot's camera calibration and estimated camera pose
        -> centreline pixels of the predicted line mask
        -> ground points with validity and covariance            (geometry.ground_projection)
        -> centre circle (known radius), straight segments       (perception.line_fitting)
        -> L / T / X junctions                                   (perception.intersections)
        -> thinned line points with across-line uncertainty      (perception.uncertainty)
        -> goal posts: position, or bearing only                 (perception.goal_features)

Inputs are image predictions, the robot's own belief about its camera, and the static field
description (circle radius, post size). There is no simulator, no ground truth and no pose in
the field frame anywhere in this module.
"""
from __future__ import annotations

import time

import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ProjectionStatus, ground_to_pixels,
    project_to_ground, projection_covariance, require_estimate)
from robosoccer_perception.inference.types import VisionResult
from robosoccer_perception.perception import intersections, line_features, line_fitting
from robosoccer_perception.perception.config import (
    FeatureExtractionConfig, MeasurementNoiseConfig)
from robosoccer_perception.perception.goal_features import extract_goal_posts
from robosoccer_perception.perception.types import (
    CircleFeature, FeatureSet, IntersectionFeature, LinePointFeatures, LineSegmentFeature)
from robosoccer_perception.perception.uncertainty import (
    line_point_std, point_uncertainty, shared_error_inflation)

_REJECTION_NAMES = {
    ProjectionStatus.ABOVE_HORIZON: "above_horizon", ProjectionStatus.NEAR_HORIZON: "near_horizon",
    ProjectionStatus.TOO_FAR: "too_far", ProjectionStatus.TOO_CLOSE: "too_close",
    ProjectionStatus.UNSTABLE: "unstable", ProjectionStatus.OUTSIDE_IMAGE: "image_border"}


class GeometricFeatureExtractor:
    """Turns vision predictions into metric features in the robot's base frame."""

    def __init__(self, field: FieldGeometry, cfg: FeatureExtractionConfig | None = None,
                 noise: MeasurementNoiseConfig | None = None):
        self.cfg = cfg or FeatureExtractionConfig()
        self.noise = noise or MeasurementNoiseConfig()
        # Static field knowledge: the same for every pose, so it says nothing about the pose.
        self.circle_radius = float(field.center_circle_radius)
        goal = field.goals[0].dims
        self.post_diameter = float(goal.post_diameter)
        self.post_height = float(goal.height + goal.post_diameter / 2.0)
        self.line_width = float(field.dims.line_width)

    def extract(self, vision: VisionResult, calibration: EstimatedCameraCalibration,
                pose: EstimatedCameraPose) -> FeatureSet:
        require_estimate(calibration, pose)
        k = calibration.intrinsics
        if (vision.width, vision.height) != (k.width, k.height):
            raise ValueError(f"the image is {vision.width}x{vision.height} but the calibration "
                             f"describes {k.width}x{k.height}; rescale the calibration")
        started = time.perf_counter()
        diagnostics: dict = {}
        segmentation = vision.segmentation
        line_points = LinePointFeatures.empty()
        segments: tuple[LineSegmentFeature, ...] = ()
        junctions: tuple[IntersectionFeature, ...] = ()
        circles: tuple[CircleFeature, ...] = ()
        inflation = 1.0             # variance factor of the camera part (shared-error model)
        if segmentation is not None:
            line_points, segments, junctions, circles, inflation = self._line_features(
                segmentation, calibration, pose, diagnostics)
        posts, rejected_posts = (), {}
        if self.cfg.goal_posts.enabled and vision.detections:
            found, rejected_posts = extract_goal_posts(
                vision.detections, None if segmentation is None else segmentation.mask,
                (vision.width, vision.height), calibration, pose, self.post_diameter,
                self.post_height, self.cfg.goal_posts, self.noise, inflation)
            posts = tuple(found)
        diagnostics["goal_posts"] = {
            "detections": sum(d.class_id == schema.GOAL_POST for d in vision.detections),
            "with_range": sum(p.range_valid for p in posts),
            "bearing_only": sum(not p.range_valid for p in posts),
            "rejected": rejected_posts}
        diagnostics["extraction_ms"] = (time.perf_counter() - started) * 1000.0
        return FeatureSet(line_points, posts, junctions, circles, segments, diagnostics)

    # ------------------------------------------------------------------ lines
    def _line_features(self, segmentation, calibration, pose, diagnostics):
        cfg, noise = self.cfg, self.noise
        rng = np.random.default_rng(cfg.seed)
        centre = line_features.centreline_pixels(segmentation.line_confidence, cfg.lines,
                                                 cfg.max_work_width)
        projection = project_to_ground(centre.pixels, calibration, pose, cfg.projection)
        rejected = {name: projection.count(flag) for flag, name in _REJECTION_NAMES.items()}
        valid = projection.valid
        diagnostics["lines"] = {
            "line_pixels": centre.num_line_pixels, "removed_small_blobs": centre.num_removed_pixels,
            "centreline_pixels": centre.num_centreline_pixels, "candidates": len(centre),
            "valid": int(valid.sum()), "rejected": int((~valid).sum()),
            "rejected_by_reason": {k: v for k, v in rejected.items() if v}}
        empty = (LinePointFeatures.empty(), (), (), (), 1.0)
        if not valid.any():
            diagnostics["lines"].update(dense=0, reported=0)
            return empty

        pixels, points = centre.pixels[valid], projection.points[valid]
        confidence = centre.confidence[valid]
        # Dense set: evenly spaced on the ground, for fitting.
        keep = line_features.thin_to_grid(points, confidence, cfg.lines.dense_ground_spacing)
        pixels, points, confidence = pixels[keep], points[keep], confidence[keep]
        covariance = projection_covariance(pixels, calibration, pose)
        pixel_rms = covariance.std_rms("pixel")
        distance = np.hypot(points[:, 0], points[:, 1])
        normals = np.full((len(points), 2), np.nan)
        # The points the localizer will get: chosen now, because how many of them share the
        # frame's camera error decides how that error is priced (1.0 = Phase 3 behaviour).
        sparse_keep = line_features.thin_on_ground(points, confidence, cfg.lines.ground_spacing,
                                                   cfg.lines.max_points)
        inflation = shared_error_inflation(len(sparse_keep), noise)

        # ---- centre circle (first, so that its arcs are not mistaken for short lines)
        circles: list[CircleFeature] = []
        on_circle = np.zeros(len(points), dtype=bool)
        if cfg.circle.enabled:
            pool = np.flatnonzero(distance <= cfg.circle.max_center_range + self.circle_radius)
            tolerance = np.minimum(np.hypot(cfg.circle.inlier_tolerance, pixel_rms[pool]),
                                   cfg.circle.max_tolerance)
            fitted = line_fitting.fit_circle(points[pool], tolerance, self.circle_radius,
                                             cfg.circle, rng)
            if fitted is not None and len(fitted.indices) >= 3:
                members = pool[fitted.indices]
                circles.append(self._circle_feature(fitted, confidence[members], calibration, pose,
                                                    inflation))
                if fitted.accepted:
                    on_circle[members] = True
                    radial = points[members] - fitted.center
                    normals[members] = radial / np.maximum(
                        np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)

        # ---- straight segments and junctions
        fitted_segments: list[line_fitting.FittedSegment] = []
        junction_features: list[IntersectionFeature] = []
        if cfg.segments.enabled:
            pool = np.flatnonzero(~on_circle & (distance <= cfg.segments.max_range))
            if len(pool) >= cfg.segments.min_points:
                tolerance = np.minimum(np.hypot(cfg.segments.inlier_tolerance, pixel_rms[pool]),
                                       cfg.segments.max_tolerance)
                # One image cell covers this much ground: the natural spacing of far points.
                allowance = cfg.lines.pixel_cell_px * centre.scale * pixel_rms[pool] \
                    / max(calibration.uncertainty.pixel_std, 1e-6)
                local = line_fitting.fit_segments(points[pool], tolerance, cfg.segments, rng,
                                                  allowance)
                for segment in local:
                    members = pool[segment.indices]
                    normals[members] = segment.normal
                    fitted_segments.append(segment)
            if cfg.intersections.enabled:
                candidates = []
                if len(fitted_segments) >= 2:
                    probe = self._turf_probe(segmentation, calibration, pose)
                    candidates = intersections.find_junctions(fitted_segments, probe,
                                                              cfg.intersections)
                for circle in circles:
                    if circle.valid:
                        candidates += intersections.circle_line_junctions(
                            circle.center, circle.radius, fitted_segments, cfg.intersections)
                for candidate in candidates:
                    junction_features.append(self._junction_feature(candidate, calibration, pose,
                                                                    inflation))

        # ---- line points for the localizer: sparser, with across-line uncertainty
        keep = sparse_keep
        sparse = type(covariance)(covariance.pixel[keep], covariance.camera[keep])
        std, pixel_std, camera_std = line_point_std(sparse, normals[keep], confidence[keep], noise,
                                                    inflation)
        certain = std <= noise.line_std_max
        diagnostics["lines"].update(dense=len(points), reported=int(certain.sum()),
                                    dropped_uncertain=int((~certain).sum()),
                                    segments=len(fitted_segments))
        if inflation != 1.0:
            diagnostics["lines"]["shared_error_inflation"] = inflation
        keep = keep[certain]
        line_points = LinePointFeatures(pixels[keep], points[keep], std[certain],
                                        confidence[keep], pixel_std[certain], camera_std[certain],
                                        normals[keep])
        segments = tuple(LineSegmentFeature(s.start, s.end, len(s.indices), s.rms)
                         for s in fitted_segments)
        return line_points, segments, tuple(junction_features), tuple(circles), inflation

    def _turf_probe(self, segmentation, calibration, pose) -> intersections.Probe:
        """What the predicted mask shows at ground points, seen through the robot's own
        calibration: turf, something else, or nothing (outside the image or its margin)."""
        cfg = self.cfg.intersections
        k = calibration.intrinsics
        field_confidence, mask = segmentation.field_confidence, segmentation.mask

        def probe(points_base: np.ndarray) -> np.ndarray:
            pixels, in_image = ground_to_pixels(points_base, calibration, pose)
            seen = in_image & k.in_image(pixels, cfg.image_margin_px)
            result = np.full(len(pixels), intersections.UNSEEN, dtype=int)
            if seen.any():
                u = np.rint(pixels[seen, 0]).astype(int)
                v = np.rint(pixels[seen, 1]).astype(int)
                turf = (mask[v, u] == schema.FIELD) \
                    & (field_confidence[v, u] >= cfg.probe_min_field_confidence)
                result[seen] = np.where(turf, intersections.TURF, intersections.NOT_TURF)
            return result
        return probe

    def _junction_feature(self, candidate, calibration, pose, camera_inflation: float = 1.0
                          ) -> IntersectionFeature:
        noise = self.noise
        confidence = float(np.clip(candidate.arm_length / (2.0 * self.cfg.intersections.min_arm),
                                   0.3, 1.0))
        uncertainty = point_uncertainty(
            candidate.position, calibration, pose, noise, pixel_std=noise.junction_pixel_std,
            range_std_floor=noise.junction_std_floor,
            bearing_std_floor=noise.junction_std_floor / max(np.linalg.norm(candidate.position), 0.5),
            confidence=confidence, inflation=noise.junction_inflation,
            camera_inflation=camera_inflation)
        return IntersectionFeature(candidate.type, candidate.position, uncertainty, confidence,
                                   candidate.orientation, candidate.flags)

    def _circle_feature(self, fitted, confidence, calibration, pose,
                        camera_inflation: float = 1.0) -> CircleFeature:
        noise = self.noise
        # A circle seen as a short arc constrains its centre poorly: scale the fit error up.
        arc_factor = 360.0 / max(fitted.arc_coverage_deg, 30.0)
        fit_variance = (max(fitted.rms, 0.01) * arc_factor) ** 2 / max(len(fitted.indices), 1)
        score = float(np.clip(fitted.arc_coverage_deg / 360.0, 0.0, 1.0) * np.mean(confidence))
        uncertainty = point_uncertainty(
            fitted.center, calibration, pose, noise, pixel_std=noise.circle_pixel_std,
            range_std_floor=noise.circle_std_floor,
            bearing_std_floor=noise.circle_std_floor / max(np.linalg.norm(fitted.center), 0.5),
            confidence=max(score, noise.confidence_floor) if fitted.accepted else 1.0,
            inflation=noise.circle_inflation, extra_variance=fit_variance,
            camera_inflation=camera_inflation)
        return CircleFeature(fitted.center, fitted.radius, uncertainty, score,
                             fitted.arc_coverage_deg, len(fitted.indices), fitted.accepted,
                             fitted.reasons)
