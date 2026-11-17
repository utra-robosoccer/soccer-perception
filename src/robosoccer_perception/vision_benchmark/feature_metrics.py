"""Scoring extracted features against simulator ground truth. EVALUATION ONLY.

Everything here needs the true robot pose and the true camera, so none of it may run inside
the localization pipeline; it is called by the benchmark after the pipeline has produced its
`FeatureSet`.

What is measured (definitions in docs/phase3_validation.md):

line points         each reported point is moved into the field frame with the TRUE pose; its
                    error is the distance to the nearest paint (0 on the paint itself).
goal posts          a post observation is matched to the true post nearest in bearing.
                    Bearing error is always scored; position error only when a range was claimed.
intersections       a classed junction is a true positive if a map junction of the SAME class
                    lies within `gate`; "wrong class" if only one of another class does;
                    otherwise false. Recall is given twice: over the map junctions whose POINT
                    is in view, and over those whose arms are in view too (0.35 m of each
                    arm, and the turf where there is no arm), which is what deciding the
                    class needs.
centre circle       "expected" when enough of the circle is in view; "found" when a reported
                    centre is within `gate` of the true one.

Normalised errors (error / reported standard deviation) check the uncertainty model: about
68 % should be below 1 and 95 % below 2 if the reported values are honest.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import StereoCameraConfig, StereoCameraModel
from robosoccer_perception.geometry.field import FieldGeometry, sample_arc
from robosoccer_perception.geometry.landmarks import LandmarkMap, LandmarkType
from robosoccer_perception.observations.noise import OracleConfig
from robosoccer_perception.observations.oracle import OracleObservationGenerator
from robosoccer_perception.perception.types import FeatureSet
from robosoccer_perception.vision_benchmark.camera_truth import SimulationCameraTruth

JUNCTION_TYPES = (LandmarkType.L_INTERSECTION, LandmarkType.T_INTERSECTION,
                  LandmarkType.X_INTERSECTION)
RANGE_BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 99.0))


@dataclasses.dataclass(frozen=True)
class ScoringConfig:
    junction_gate: float = 0.35         # metres: a junction this close to a map junction matches
    junction_max_range: float = 4.0     # metres: map junctions farther than this are not expected
    junction_image_margin_px: float = 6.0
    junction_arm: float = 0.35          # metres of each arm that must be in view to be "decidable"
    post_position_gate: float = 1.5     # metres
    post_bearing_gate_deg: float = 6.0
    circle_gate: float = 0.30           # metres
    circle_min_visible_fraction: float = 0.42   # 150 of 360 degrees
    circle_max_range: float = 5.0


@dataclasses.dataclass(frozen=True)
class FrameTruth:
    """Ground truth of one frame, for scoring."""
    base_pose: np.ndarray               # true [x, y, yaw]
    camera: SimulationCameraTruth
    occluders: np.ndarray | None = None     # (K, 4) cylinders x, y, radius, height in the field


class FeatureScorer:
    def __init__(self, field: FieldGeometry, landmarks: LandmarkMap,
                 camera_cfg: StereoCameraConfig, cfg: ScoringConfig | None = None):
        self.field, self.landmarks = field, landmarks
        self.cfg = cfg or ScoringConfig()
        self._posts = landmarks.positions(LandmarkType.GOAL_POST)
        self._circle = sample_arc(field.arc("center_circle"), 0.05)
        self._junctions = {kind: landmarks.positions(kind) for kind in JUNCTION_TYPES}
        # Around each map junction, the four points one arm length away along the field axes.
        # They are what has to be in view for the junction's class to be decidable at all:
        # the painted arms, and the turf where there is no arm.
        steps = self.cfg.junction_arm * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        self._junction_surround = {kind: (self._junctions[kind][:, None, :] + steps).reshape(-1, 2)
                                   for kind in JUNCTION_TYPES}
        visibility = OracleConfig(min_range=0.0, max_range=1e3, occlusion=True,
                                  image_margin_px=self.cfg.junction_image_margin_px)
        self._oracle = OracleObservationGenerator(field, landmarks, StereoCameraModel(camera_cfg),
                                                  visibility)

    def _visible(self, truth: FrameTruth, points_field: np.ndarray, max_range: float) -> np.ndarray:
        occluders = self._oracle._occluders(None, 0.0, truth.occluders)
        seen = self._oracle.visible(truth.base_pose, truth.camera.true_rig, points_field,
                                    occluders)
        distance = np.linalg.norm(points_field - truth.base_pose[:2], axis=1)
        return seen & (distance <= max_range)

    def score(self, features: FeatureSet, truth: FrameTruth) -> dict[str, Any]:
        """Raw scores of one frame (lists and counts; `FeatureStatistics` aggregates them)."""
        cfg, pose = self.cfg, np.asarray(truth.base_pose, dtype=float)
        out: dict[str, Any] = {}

        # ---- line points
        points = features.line_points
        if len(points):
            in_field = tf.se2_transform_points(pose, points.points)
            out["line_error"] = self.field.distance_to_paint(in_field)
            out["line_std"] = np.asarray(points.std)
            out["line_range"] = np.hypot(points.points[:, 0], points.points[:, 1])
        else:
            out["line_error"] = out["line_std"] = out["line_range"] = np.zeros(0)

        # ---- goal posts
        true_posts = tf.se2_inverse_transform_points(pose, self._posts)
        true_bearing = np.arctan2(true_posts[:, 1], true_posts[:, 0])
        posts = []
        for post in features.goal_posts:
            bearing = math.atan2(post.position[1], post.position[0])
            bearing_errors = np.abs(tf.angle_difference(bearing, true_bearing))
            if post.range_valid:
                index = int(np.argmin(np.linalg.norm(true_posts - post.position, axis=1)))
                position_error = float(np.linalg.norm(true_posts[index] - post.position))
                matched = position_error <= cfg.post_position_gate
                range_error = abs(float(np.linalg.norm(post.position))
                                  - float(np.linalg.norm(true_posts[index])))
            else:
                index = int(np.argmin(bearing_errors))
                matched = bearing_errors[index] <= math.radians(cfg.post_bearing_gate_deg)
                position_error = range_error = float("nan")
            posts.append({
                "range_valid": post.range_valid, "matched": bool(matched),
                "position_error": position_error, "range_error": range_error,
                "bearing_error_deg": float(np.degrees(bearing_errors[index])),
                "range_std": post.uncertainty.range_std,
                "bearing_std_deg": math.degrees(post.uncertainty.bearing_std),
                "true_range": float(np.linalg.norm(true_posts[index])), "flags": post.flags})
        out["posts"] = posts

        # ---- intersections
        visible = {kind: self._visible(truth, self._junctions[kind], cfg.junction_max_range)
                   for kind in JUNCTION_TYPES}
        decidable = {kind: visible[kind] & self._visible(
            truth, self._junction_surround[kind], 1e3).reshape(-1, 4).all(axis=1)
            for kind in JUNCTION_TYPES}
        found = {kind: np.zeros(len(self._junctions[kind]), dtype=bool) for kind in JUNCTION_TYPES}
        junctions = []
        for junction in features.intersections:
            if junction.type is None:
                junctions.append({"type": None, "outcome": "unclassed", "flags": junction.flags})
                continue
            in_field = tf.se2_transform_points(pose, junction.position[None])[0]
            same = np.linalg.norm(self._junctions[junction.type] - in_field, axis=1)
            nearest = int(np.argmin(same))
            if same[nearest] <= cfg.junction_gate:
                outcome = "true"
                found[junction.type][nearest] = True
            elif any(np.linalg.norm(self._junctions[kind] - in_field, axis=1).min()
                     <= cfg.junction_gate for kind in JUNCTION_TYPES if kind != junction.type):
                outcome = "wrong_class"
            else:
                outcome = "false"
            radial = math.hypot(*junction.position)
            junctions.append({
                "type": junction.type.value, "outcome": outcome,
                "error": float(same[nearest]) if outcome == "true" else float("nan"),
                "range": radial,
                "sigma": math.hypot(junction.uncertainty.range_std,
                                    radial * junction.uncertainty.bearing_std)})
        out["junctions"] = junctions
        out["junction_visible"] = {kind.value: int(visible[kind].sum()) for kind in JUNCTION_TYPES}
        out["junction_recalled"] = {kind.value: int((visible[kind] & found[kind]).sum())
                                    for kind in JUNCTION_TYPES}
        out["junction_decidable"] = {kind.value: int(decidable[kind].sum())
                                     for kind in JUNCTION_TYPES}
        out["junction_decidable_recalled"] = {
            kind.value: int((decidable[kind] & found[kind]).sum()) for kind in JUNCTION_TYPES}

        # ---- centre circle
        true_centre = tf.se2_inverse_transform_points(pose, np.zeros((1, 2)))[0]
        fraction = float(self._visible(truth, self._circle, 1e3).mean())
        expected = fraction >= cfg.circle_min_visible_fraction \
            and float(np.linalg.norm(true_centre)) <= cfg.circle_max_range
        errors = [float(np.linalg.norm(c.center - true_centre)) for c in features.valid_circles]
        out["circle"] = {
            "expected": bool(expected), "visible_fraction": fraction, "reported": len(errors),
            "correct": int(sum(e <= cfg.circle_gate for e in errors)),
            "false": int(sum(e > cfg.circle_gate for e in errors)),
            "errors": [e for e in errors if e <= cfg.circle_gate],
            "sigma": [math.hypot(c.uncertainty.range_std,
                                 np.linalg.norm(c.center) * c.uncertainty.bearing_std)
                      for c in features.valid_circles]}

        # ---- counts
        lines = features.diagnostics.get("lines", {})
        out["counts"] = {
            "line_candidates": lines.get("candidates", 0), "line_valid": lines.get("valid", 0),
            "line_rejected": lines.get("rejected", 0),
            "line_rejected_by_reason": lines.get("rejected_by_reason", {}),
            "line_points": len(points), "goal_posts": len(features.goal_posts),
            "junctions_classed": len(features.typed_intersections),
            "junctions_unclassed": len(features.intersections) - len(features.typed_intersections),
            "circles": len(features.valid_circles)}
        return out


def _stats(values, digits: int = 4) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "rmse": None}
    return {"count": int(len(values)), "mean": round(float(values.mean()), digits),
            "median": round(float(np.median(values)), digits),
            "p95": round(float(np.percentile(values, 95)), digits),
            "rmse": round(float(np.sqrt(np.mean(values ** 2))), digits)}


def _share(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else round(float(numerator) / float(denominator), 4)


class FeatureStatistics:
    """Accumulates per-frame scores of one run (or several) and summarises them."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def add(self, score: dict[str, Any]) -> None:
        self.frames.append(score)

    def extend(self, other: "FeatureStatistics") -> None:
        self.frames.extend(other.frames)

    def summary(self) -> dict[str, Any]:
        frames = self.frames
        if not frames:
            return {"frames": 0}
        error = np.concatenate([f["line_error"] for f in frames])
        std = np.concatenate([f["line_std"] for f in frames])
        distance = np.concatenate([f["line_range"] for f in frames])
        normalised = error / np.maximum(std, 1e-9)
        line: dict[str, Any] = {
            "error_m": _stats(error),
            "within_5cm": _share((error <= 0.05).sum(), len(error)),
            "within_1_sigma": _share((normalised <= 1.0).sum(), len(error)),
            "within_2_sigma": _share((normalised <= 2.0).sum(), len(error)),
            "reported_std_m": _stats(std),
            "by_range": {f"{low:g}-{high:g}m" if high < 90 else f">{low:g}m":
                         _stats(error[(distance >= low) & (distance < high)])
                         for low, high in RANGE_BINS}}

        posts = [p for f in frames for p in f["posts"]]
        ranged = [p for p in posts if p["range_valid"]]
        matched_ranged = [p for p in ranged if p["matched"]]
        bearing_only = [p for p in posts if not p["range_valid"]]
        matched = [p for p in posts if p["matched"]]
        flags: dict[str, int] = {}
        for post in posts:
            for flag in post["flags"]:
                flags[flag] = flags.get(flag, 0) + 1
        goal_posts = {
            "observations": len(posts), "with_range": len(ranged),
            "bearing_only": len(bearing_only),
            "unmatched": len(posts) - len(matched),
            "unmatched_share": _share(len(posts) - len(matched), len(posts)),
            "ground_point_error_m": _stats([p["position_error"] for p in matched_ranged]),
            "range_error_m": _stats([p["range_error"] for p in matched_ranged]),
            "bearing_error_deg": _stats([p["bearing_error_deg"] for p in matched]),
            "bearing_error_deg_bearing_only": _stats(
                [p["bearing_error_deg"] for p in bearing_only if p["matched"]]),
            "range_within_2_sigma": _share(
                sum(p["range_error"] <= 2 * p["range_std"] for p in matched_ranged),
                len(matched_ranged)),
            "bearing_within_2_sigma": _share(
                sum(p["bearing_error_deg"] <= 2 * p["bearing_std_deg"] for p in matched),
                len(matched)),
            "flags": flags}

        junctions = [j for f in frames for j in f["junctions"]]
        intersections: dict[str, Any] = {}
        for kind in (*[k.value for k in JUNCTION_TYPES], "all"):
            chosen = [j for j in junctions if j["type"] is not None
                      and (kind == "all" or j["type"] == kind)]
            true = [j for j in chosen if j["outcome"] == "true"]
            visible = sum(sum(v for key, v in f["junction_visible"].items()
                              if kind in ("all", key)) for f in frames)
            recalled = sum(sum(v for key, v in f["junction_recalled"].items()
                               if kind in ("all", key)) for f in frames)
            decidable = sum(sum(v for key, v in f["junction_decidable"].items()
                                if kind in ("all", key)) for f in frames)
            decidable_recalled = sum(sum(v for key, v in f["junction_decidable_recalled"].items()
                                         if kind in ("all", key)) for f in frames)
            intersections[kind] = {
                "reported": len(chosen), "true": len(true),
                "wrong_class": sum(j["outcome"] == "wrong_class" for j in chosen),
                "false": sum(j["outcome"] == "false" for j in chosen),
                "precision": _share(len(true), len(chosen)),
                "in_view": visible, "recalled": recalled, "recall": _share(recalled, visible),
                "arms_in_view": decidable,
                "recall_arms_in_view": _share(decidable_recalled, decidable),
                "position_error_m": _stats([j["error"] for j in true]),
                "within_2_sigma": _share(sum(j["error"] <= 2 * j["sigma"] for j in true),
                                         len(true))}
        intersections["unclassed"] = sum(j["type"] is None for j in junctions)

        circles = [f["circle"] for f in frames]
        expected = [c for c in circles if c["expected"]]
        circle = {
            "frames_expected": len(expected),
            "found_when_expected": sum(c["correct"] > 0 for c in expected),
            "detection_rate": _share(sum(c["correct"] > 0 for c in expected), len(expected)),
            "reported": sum(c["reported"] for c in circles),
            "correct": sum(c["correct"] for c in circles),
            "false": sum(c["false"] for c in circles),
            "precision": _share(sum(c["correct"] for c in circles),
                                sum(c["reported"] for c in circles)),
            "centre_error_m": _stats([e for c in circles for e in c["errors"]])}

        counts = [f["counts"] for f in frames]
        candidates = sum(c["line_candidates"] for c in counts)
        reasons: dict[str, int] = {}
        for count in counts:
            for reason, number in count["line_rejected_by_reason"].items():
                reasons[reason] = reasons.get(reason, 0) + number
        per_frame = {
            "line_points": round(float(np.mean([c["line_points"] for c in counts])), 2),
            "goal_posts": round(float(np.mean([c["goal_posts"] for c in counts])), 3),
            "junctions_classed": round(float(np.mean([c["junctions_classed"] for c in counts])), 3),
            "junctions_unclassed": round(float(np.mean([c["junctions_unclassed"] for c in counts])), 3),
            "circles": round(float(np.mean([c["circles"] for c in counts])), 3),
            "frames_without_any_observation": _share(
                sum(c["line_points"] + c["goal_posts"] + c["junctions_classed"] + c["circles"] == 0
                    for c in counts), len(counts))}
        projection = {
            "candidate_pixels": candidates,
            "valid_share": _share(sum(c["line_valid"] for c in counts), candidates),
            "rejected_share": _share(sum(c["line_rejected"] for c in counts), candidates),
            "rejected_by_reason": {k: _share(v, candidates) for k, v in sorted(reasons.items())}}
        return {"frames": len(frames), "line_points": line, "goal_posts": goal_posts,
                "intersections": intersections, "center_circle": circle,
                "observations_per_frame": per_frame, "projection": projection}
