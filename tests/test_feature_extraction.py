"""Geometric feature extraction: line pixels and their thinning, segment and circle fitting,
junction classification, goal posts, the uncertainty model, and conversion to an
`ObservationFrame`. No MuJoCo and no networks: views are painted analytically from the field
geometry (tests/phase3_helpers.py)."""
import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("cv2")

import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.geometry import ground_projection as gp  # noqa: E402
from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.landmarks import LandmarkType  # noqa: E402
from robosoccer_perception.inference.types import Detection, VisionResult  # noqa: E402
from robosoccer_perception.localization.particle_filter import ParticleFilter  # noqa: E402
from robosoccer_perception.observations.oracle import OracleObservationGenerator  # noqa: E402
from robosoccer_perception.observations.types import (  # noqa: E402
    LandmarkObservation, LinePointsObservation, ObservationFrame)
from robosoccer_perception.perception import intersections, line_features, line_fitting  # noqa: E402
from robosoccer_perception.perception.config import (  # noqa: E402
    CircleConfig, FeatureExtractionConfig, GoalPostConfig, IntersectionConfig, LineConfig,
    MeasurementNoiseConfig, SegmentConfig)
from robosoccer_perception.perception.extractor import GeometricFeatureExtractor  # noqa: E402
from robosoccer_perception.perception.goal_features import (  # noqa: E402
    extract_goal_posts, foot_pixel, vertical_vanishing_point)
from robosoccer_perception.perception.observation_builder import (  # noqa: E402
    ObservationSelection, build_observation_frame)
from robosoccer_perception.perception.types import FeatureSet, LinePointFeatures  # noqa: E402
from robosoccer_perception.perception.uncertainty import (  # noqa: E402
    confidence_factor, line_point_std, point_uncertainty)

JUNCTIONS = (LandmarkType.L_INTERSECTION, LandmarkType.T_INTERSECTION, LandmarkType.X_INTERSECTION)


@pytest.fixture(scope="module")
def calibration():
    return helpers.small_calibration()


@pytest.fixture(scope="module")
def extractor(s_field):
    return GeometricFeatureExtractor(s_field)


def _in_field(pose, points):
    return tf.se2_transform_points(np.asarray(pose, float), np.asarray(points, float))


def _extract(extractor, field, pose, calibration, camera_pose=None, believed=None):
    camera_pose = camera_pose or calibration.pose()
    _, vision = helpers.analytic_view(field, pose, calibration, camera_pose)
    return extractor.extract(vision, calibration, believed or camera_pose)


# ------------------------------------------------------------------ configuration
def test_configuration_files_match_the_documented_defaults():
    assert FeatureExtractionConfig.load() == FeatureExtractionConfig()
    assert MeasurementNoiseConfig.load() == MeasurementNoiseConfig()
    with pytest.raises(ValueError):
        FeatureExtractionConfig.from_dict({"lines": {"no_such_key": 1}})
    with pytest.raises(ValueError):
        ObservationSelection(intersections=("L", "Q"))
    changed = FeatureExtractionConfig.load(overrides={"lines": {"max_points": 20}})
    assert changed.lines.max_points == 20 and changed.circle == CircleConfig()


# ------------------------------------------------------------------ line pixels
def test_centreline_runs_along_the_middle_of_a_thick_line():
    confidence = np.zeros((120, 200), np.float32)
    confidence[50:59, 20:180] = 0.9                         # a band 9 px thick, centre row 54
    found = line_features.centreline_pixels(confidence, LineConfig(pixel_cell_px=4))
    assert 30 <= len(found) <= 45, "about one pixel per 4 px cell along 160 px"
    assert np.all(np.abs(found.pixels[:, 1] - 54.0) <= 1.0)
    assert found.pixels[:, 0].min() < 30 and found.pixels[:, 0].max() > 170
    assert np.allclose(found.confidence, 0.9) and np.all(found.half_width_px >= 4.0)
    assert found.num_line_pixels == 9 * 160 and found.num_removed_pixels == 0


def test_small_blobs_and_weak_pixels_are_not_lines():
    confidence = np.zeros((120, 200), np.float32)
    confidence[60:63, 10:190] = 0.95
    confidence[20:23, 30:33] = 0.99                         # a 9-pixel blob: a false positive
    confidence[100:104, 20:180] = 0.3                       # a real band, but below threshold
    found = line_features.centreline_pixels(confidence, LineConfig(min_component_area_px=12))
    assert found.num_removed_pixels == 9
    assert np.all(np.abs(found.pixels[:, 1] - 61.0) <= 1.0), "only the confident line remains"
    kept = line_features.centreline_pixels(confidence, LineConfig(min_component_area_px=1))
    assert len(kept) > len(found)
    nothing = line_features.centreline_pixels(np.zeros((50, 80), np.float32))
    assert len(nothing) == 0 and nothing.pixels.shape == (0, 2)


def test_line_pixel_extraction_is_deterministic_and_reports_input_image_coordinates():
    rng = np.random.default_rng(3)
    confidence = np.zeros((360, 640), np.float32)
    confidence[200:206, :] = 0.9
    confidence[:, 300:305] = 0.8
    confidence += rng.uniform(0.0, 0.05, confidence.shape).astype(np.float32)
    a = line_features.centreline_pixels(confidence)
    b = line_features.centreline_pixels(confidence.copy())
    np.testing.assert_array_equal(a.pixels, b.pixels)
    # The same picture at twice the size is reduced to the work width, and the pixels come
    # back in the big image's coordinates.
    import cv2
    big = cv2.resize(confidence, (1280, 720), interpolation=cv2.INTER_NEAREST)
    scaled = line_features.centreline_pixels(big, max_work_width=640)
    assert scaled.scale == 2.0
    horizontal = scaled.pixels[np.abs(scaled.pixels[:, 0] - 100.0) < 60]
    assert np.all(np.abs(horizontal[:, 1] - 405.0) <= 3.0), "row 202.5 of the small image"


def test_ground_thinning_enforces_a_minimum_distance_and_respects_the_cap(rng):
    points = np.vstack([np.column_stack([np.linspace(0.0, 4.0, 400), np.zeros(400)]),
                        rng.uniform([0.0, 0.5], [4.0, 2.0], size=(300, 2))])
    priority = rng.uniform(size=len(points))
    kept = line_features.thin_on_ground(points, priority, 0.2)
    chosen = points[kept]
    distance = np.linalg.norm(chosen[:, None] - chosen[None], axis=2)
    np.fill_diagonal(distance, 9.0)
    assert distance.min() >= 0.2, "no two kept points are closer than the spacing"
    on_line = chosen[chosen[:, 1] == 0.0]
    assert 14 <= len(on_line) <= 21, "4 m of line at >= 0.2 m spacing"
    assert kept[0] != kept[-1] and np.all(np.diff(kept) > 0), "indices in ascending order"
    assert int(np.argmax(priority)) in kept, "the best point is always kept"
    capped = line_features.thin_on_ground(points, priority, 0.2, max_points=8)
    assert 1 <= len(capped) <= 8
    np.testing.assert_array_equal(kept, line_features.thin_on_ground(points, priority, 0.2))
    assert len(line_features.thin_on_ground(np.zeros((0, 2)), np.zeros(0), 0.2)) == 0
    # The grid variant (used to even out density before fitting) keeps one point per cell.
    cells = line_features.thin_to_grid(points, priority, 0.2)
    index = np.floor(points / 0.2).astype(int)
    assert len(cells) == len({tuple(c) for c in index})
    for i in cells[:20]:
        same = np.all(index == index[i], axis=1)
        assert priority[i] == priority[same].max()
    assert len(line_features.thin_to_grid(np.zeros((0, 2)), np.zeros(0), 0.2)) == 0


# ------------------------------------------------------------------ segments and the circle
def _line(start, end, spacing=0.06, noise=0.0, rng=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = max(2, int(np.linalg.norm(end - start) / spacing) + 1)
    points = start + np.linspace(0.0, 1.0, count)[:, None] * (end - start)
    return points if rng is None else points + rng.normal(0.0, noise, points.shape)


def test_segments_are_fitted_split_at_gaps_and_merged_when_collinear(rng):
    points = np.vstack([
        _line([1.0, -1.0], [1.0, 1.5], noise=0.008, rng=rng),      # A
        _line([1.0, 1.5], [3.0, 1.5], noise=0.008, rng=rng),       # B, meets A at a corner
        _line([2.0, -1.0], [2.6, -1.0], noise=0.008, rng=rng),     # C1
        _line([2.75, -1.0], [3.5, -1.0], noise=0.008, rng=rng),    # C2: 15 cm gap, same line
        _line([2.0, -2.0], [2.6, -2.0], noise=0.008, rng=rng),     # D1
        _line([3.6, -2.0], [4.2, -2.0], noise=0.008, rng=rng),     # D2: 1 m gap, same line
        rng.uniform([0.5, -2.5], [4.5, 2.0], size=(12, 2)),        # clutter
    ])
    tolerance = np.full(len(points), 0.03)
    segments = line_fitting.fit_segments(points, tolerance, SegmentConfig(),
                                         np.random.default_rng(0))
    lengths = sorted(s.length for s in segments)
    # A stray clutter point next to a line can extend it a little; A consumed the corner
    # points, so B starts a few centimetres late.
    assert len(lengths) == 5 and lengths == pytest.approx([0.6, 0.6, 1.5, 2.0, 2.5], abs=0.25), \
        lengths
    for segment in segments:
        assert segment.rms < 0.02 and len(segment.indices) >= 6
        assert abs(float(segment.direction @ segment.normal)) < 1e-12
    vertical = max(segments, key=lambda s: s.length)
    assert abs(vertical.direction[1]) > 0.999 and abs(vertical.start[0] - 1.0) < 0.02
    assert line_fitting.fit_segments(points[:3], tolerance[:3], SegmentConfig(),
                                     np.random.default_rng(0)) == []


def _circle(centre, radius, start_deg=0.0, end_deg=360.0, spacing=0.06, noise=0.0, rng=None):
    count = max(3, int(radius * math.radians(end_deg - start_deg) / spacing))
    angles = np.radians(np.linspace(start_deg, end_deg, count, endpoint=False))
    points = np.asarray(centre) + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return points if rng is None else points + rng.normal(0.0, noise, points.shape)


def test_a_circle_of_the_known_radius_is_found_and_a_short_arc_is_refused(rng):
    radius, cfg = 0.725, CircleConfig()
    full = np.vstack([_circle([2.5, 0.4], radius, noise=0.01, rng=rng),
                      _line([2.5, -1.2], [2.5, 2.0])])              # the halfway line through it
    fitted = line_fitting.fit_circle(full, np.full(len(full), 0.04), radius, cfg,
                                     np.random.default_rng(0))
    assert fitted.accepted and fitted.reasons == ()
    assert np.linalg.norm(fitted.center - [2.5, 0.4]) < 0.02
    assert fitted.arc_coverage_deg == 360.0 and fitted.purity > 0.75

    half = _circle([2.5, 0.4], radius, 20.0, 220.0, noise=0.01, rng=rng)
    fitted = line_fitting.fit_circle(half, np.full(len(half), 0.04), radius, cfg,
                                     np.random.default_rng(0))
    assert fitted.accepted and 180.0 <= fitted.arc_coverage_deg <= 225.0
    assert np.linalg.norm(fitted.center - [2.5, 0.4]) < 0.03

    arc = _circle([2.5, 0.4], radius, 100.0, 200.0, noise=0.01, rng=rng)
    fitted = line_fitting.fit_circle(arc, np.full(len(arc), 0.04), radius, cfg,
                                     np.random.default_rng(0))
    assert not fitted.accepted and "short_arc" in fitted.reasons, \
        "a 100 degree arc does not pin down the centre"
    assert line_fitting.fit_circle(arc[:5], np.full(5, 0.04), radius, cfg,
                                   np.random.default_rng(0)) is None


def test_a_circle_laid_tangent_to_straight_lines_is_refused():
    """The nested goal and penalty areas offer three or four straight lines that a circle of
    the centre-circle radius can touch. That is not a centre circle."""
    radius = 0.725
    points = np.vstack([
        _line([0.0, 0.0], [4.0, 0.0]), _line([0.0, 1.45], [4.0, 1.45]),     # two parallels 2R apart
        _line([1.0, -1.0], [1.0, 2.5]),                                      # and a perpendicular
    ])
    tolerance = np.full(len(points), 0.05)
    for seed in range(5):
        fitted = line_fitting.fit_circle(points, tolerance, radius, CircleConfig(),
                                         np.random.default_rng(seed))
        assert fitted is None or not fitted.accepted, fitted
    loose = dataclasses.replace(CircleConfig(), min_purity=0.0, min_arc_deg=100.0, min_density=0.0)
    accepted = [line_fitting.fit_circle(points, tolerance, radius, loose,
                                        np.random.default_rng(seed)) for seed in range(5)]
    assert any(f is not None and f.accepted for f in accepted), \
        "without the purity test the tangent circle would have been accepted"


# ------------------------------------------------------------------ junctions
def _segment(start, end):
    points = _line(start, end, 0.05)
    return line_fitting._segment_from(points, np.arange(len(points)))


def _all_turf(points):
    return np.full(len(points), intersections.TURF)


def _classes(segments, probe=_all_turf, cfg=IntersectionConfig()):
    return [(j.type, j.flags) for j in intersections.find_junctions(segments, probe, cfg)]


def test_junction_class_follows_from_where_the_lines_end():
    cfg = IntersectionConfig()
    corner = [_segment([1.0, 0.0], [3.0, 0.0]), _segment([1.0, 0.0], [1.0, 1.5])]
    tee = [_segment([1.0, -1.0], [1.0, 1.5]), _segment([1.0, 0.2], [2.5, 0.2])]
    cross = [_segment([1.0, -1.0], [1.0, 1.5]), _segment([0.0, 0.2], [2.5, 0.2])]
    (kind, _), = _classes(corner)
    assert kind == LandmarkType.L_INTERSECTION
    (kind, _), = _classes(tee)
    assert kind == LandmarkType.T_INTERSECTION
    (kind, _), = _classes(cross)
    assert kind == LandmarkType.X_INTERSECTION
    found = intersections.find_junctions(tee, _all_turf, cfg)[0]
    np.testing.assert_allclose(found.position, [1.0, 0.2], atol=0.01)
    assert found.orientation == pytest.approx(0.0, abs=0.02), "the stem points along +x"
    corner_found = intersections.find_junctions(corner, _all_turf, cfg)[0]
    assert corner_found.orientation == pytest.approx(math.radians(45.0), abs=0.02)
    # A stem that stops 5 cm short of the bar (the bar's own width) still makes a T.
    short = [_segment([1.0, -1.0], [1.0, 1.5]), _segment([1.05, 0.2], [2.5, 0.2])]
    assert _classes(short)[0][0] == LandmarkType.T_INTERSECTION


def test_lines_that_do_not_meet_or_meet_obliquely_give_no_junction():
    apart = [_segment([1.0, 0.0], [3.0, 0.0]), _segment([1.0, 0.6], [1.0, 2.0])]     # 60 cm gap
    oblique = [_segment([1.0, 0.0], [3.0, 0.0]), _segment([1.0, 0.0], [2.2, 1.2])]   # 45 degrees
    parallel = [_segment([1.0, 0.0], [3.0, 0.0]), _segment([1.0, 0.5], [3.0, 0.5])]
    far = [_segment([5.0, 0.0], [7.0, 0.0]), _segment([5.0, 0.0], [5.0, 1.5])]       # beyond range
    for segments in (apart, oblique, parallel, far):
        assert _classes(segments) == []


def test_an_end_only_counts_when_turf_is_seen_beyond_it():
    """A line cut by the image border or by an occluder also "ends". Without positive evidence
    of turf past the end, the junction is reported without a class."""
    tee = [_segment([1.0, -1.0], [1.0, 1.5]), _segment([1.0, 0.2], [2.5, 0.2])]

    def beyond_is_unseen(points):                       # nothing visible left of the bar
        return np.where(points[:, 0] < 0.95, intersections.UNSEEN, intersections.TURF)

    def beyond_is_occluded(points):                     # a robot stands there
        return np.where(points[:, 0] < 0.95, intersections.NOT_TURF, intersections.TURF)

    (kind, flags), = _classes(tee, beyond_is_unseen)
    assert kind is None and "end_not_observed" in flags
    (kind, flags), = _classes(tee, beyond_is_occluded)
    assert kind is None and "end_not_on_turf" in flags
    # A crossing needs no such evidence: both lines are seen on both sides.
    cross = [_segment([1.0, -1.0], [1.0, 1.5]), _segment([0.0, 0.2], [2.5, 0.2])]
    assert _classes(cross, beyond_is_occluded)[0][0] == LandmarkType.X_INTERSECTION
    # A junction whose own position is out of view is not reported at all.
    assert _classes(tee, lambda p: np.full(len(p), intersections.UNSEEN)) == []


def test_a_stub_too_short_to_judge_leaves_the_junction_unclassed():
    stub = [_segment([1.0, -1.0], [1.0, 1.5]), _segment([0.8, 0.2], [2.5, 0.2])]     # 20 cm past
    (kind, flags), = _classes(stub)
    assert kind is None and "short_arm" in flags


def test_conflicting_classes_at_one_place_cancel():
    a = intersections.JunctionCandidate(LandmarkType.L_INTERSECTION, np.array([1.0, 0.0]), 0.0, 0.5, 9)
    b = intersections.JunctionCandidate(LandmarkType.T_INTERSECTION, np.array([1.1, 0.05]), 0.0, 0.8, 9)
    c = intersections.JunctionCandidate(LandmarkType.T_INTERSECTION, np.array([3.0, 0.0]), 0.0, 0.8, 9)
    merged = intersections._merge([a, b, c], 0.3)
    assert len(merged) == 2
    assert merged[0].type is None and "conflicting_classes" in merged[0].flags
    assert merged[1].type == LandmarkType.T_INTERSECTION
    same = intersections._merge([b, dataclasses.replace(b, arm_length=0.4)], 0.3)
    assert len(same) == 1 and same[0].arm_length == 0.8


def test_x_junctions_come_from_an_accepted_circle_and_a_line_through_its_centre():
    cfg = IntersectionConfig()
    centre, radius = np.array([2.0, 0.3]), 0.725
    through = _segment([2.0, -1.2], [2.0, 1.8])
    found = intersections.circle_line_junctions(centre, radius, [through], cfg)
    assert len(found) == 2 and all(j.type == LandmarkType.X_INTERSECTION for j in found)
    np.testing.assert_allclose(sorted(j.position[1] for j in found), [0.3 - radius, 0.3 + radius],
                               atol=0.01)
    beside = _segment([2.4, -1.2], [2.4, 1.8])                  # misses the centre by 40 cm
    stops_inside = _segment([2.0, -1.2], [2.0, 0.4])            # reaches only one crossing
    assert intersections.circle_line_junctions(centre, radius, [beside], cfg) == []
    assert len(intersections.circle_line_junctions(centre, radius, [stops_inside], cfg)) == 1


# ------------------------------------------------------------------ the extractor on views
@pytest.mark.parametrize("pose", [[-2.6, -1.4, 0.6], [1.5, 0.8, math.pi], [2.0, -0.5, -0.3],
                                  [-3.2, 1.6, 2.4], [-2.0, 0.0, 0.0], [3.5, 2.0, -2.2]])
def test_extracted_features_are_where_the_field_says(s_field, s_landmarks, extractor,
                                                     calibration, pose):
    features = _extract(extractor, s_field, pose, calibration)
    points = features.line_points
    assert 10 <= len(points) <= extractor.cfg.lines.max_points
    assert s_field.distance_to_paint(_in_field(pose, points.points)).max() < 0.03
    assert np.all(points.std > 0.02) and np.all(points.std <= extractor.noise.line_std_max)
    assert np.all(np.linalg.norm(points.points, axis=1) <= extractor.cfg.projection.max_range)
    spacing = np.linalg.norm(points.points[:, None] - points.points[None], axis=2)
    np.fill_diagonal(spacing, 9.0)
    assert spacing.min() >= extractor.cfg.lines.ground_spacing - 1e-9, \
        "no two reported points are closer than the configured spacing"
    for junction in features.intersections:
        if junction.type is None:
            continue
        distance = np.linalg.norm(
            s_landmarks.positions(junction.type) - _in_field(pose, junction.position[None])[0],
            axis=1).min()
        assert distance < 0.06, (junction.type, distance)
        assert 0.0 < junction.confidence <= 1.0 and junction.uncertainty.range_std > 0.04
    for circle in features.valid_circles:
        assert np.linalg.norm(_in_field(pose, circle.center[None])[0]) < 0.05
        assert circle.radius == pytest.approx(s_field.center_circle_radius)
    lines = features.diagnostics["lines"]
    assert lines["candidates"] == lines["valid"] + lines["rejected"]
    assert lines["reported"] == len(points) and features.diagnostics["extraction_ms"] > 0


def test_every_junction_class_and_the_circle_are_found_somewhere(s_field, extractor, calibration):
    seen = set()
    circles = 0
    for pose in ([-2.6, -1.4, 0.6], [2.0, -0.5, -0.3], [-2.0, 0.0, 0.0], [-3.2, 1.6, 2.4]):
        features = _extract(extractor, s_field, pose, calibration)
        seen |= {j.type for j in features.typed_intersections}
        circles += len(features.valid_circles)
    assert seen == set(JUNCTIONS)
    assert circles >= 2


def test_line_point_uncertainty_grows_with_range_and_knows_the_line_direction(s_field, extractor,
                                                                              calibration):
    features = _extract(extractor, s_field, [-2.0, 0.0, 0.0], calibration)
    points = features.line_points
    distance = np.linalg.norm(points.points, axis=1)
    near, far = distance < 2.0, distance > 3.5
    assert near.any() and far.any()
    assert np.median(points.std[far]) > 2.0 * np.median(points.std[near])
    np.testing.assert_allclose(points.std ** 2,
                               (points.pixel_std ** 2 + points.camera_std ** 2
                                + extractor.noise.line_std_floor ** 2)
                               * confidence_factor(points.confidence, extractor.noise) ** 2)
    known = np.isfinite(points.normals).all(axis=1)
    assert known.mean() > 0.7, "most points lie on a fitted line or on the circle"
    np.testing.assert_allclose(np.linalg.norm(points.normals[known], axis=1), 1.0, atol=1e-9)
    # A line running straight away from the robot is barely affected by a pitch error; a line
    # across the view takes the full hit. The normal-aware uncertainty reflects that.
    radial = points.points / distance[:, None]
    along = np.abs(np.einsum("ni,ni->n", points.normals[known], radial[known]))
    away, across = known.copy(), known.copy()
    away[known] = (along < 0.3) & (distance[known] > 2.0)
    across[known] = (along > 0.9) & (distance[known] > 2.0)
    if away.any() and across.any():
        assert np.median(points.camera_std[away] / distance[away] ** 2) \
            < 0.5 * np.median(points.camera_std[across] / distance[across] ** 2)


def test_a_wrong_camera_estimate_moves_the_extracted_points(s_field, extractor, calibration):
    """The extractor uses the pose it is GIVEN. With the true camera 2 deg lower than believed
    every point lands too far away, by what the pitch sensitivity predicts."""
    pose = [-2.0, 0.0, 0.0]
    truth = calibration.pose()
    believed = truth.perturbed(d_pitch=math.radians(-2.0))
    right = _extract(extractor, s_field, pose, calibration, truth, truth)
    wrong = _extract(extractor, s_field, pose, calibration, truth, believed)
    error_right = s_field.distance_to_paint(_in_field(pose, right.line_points.points))
    error_wrong = s_field.distance_to_paint(_in_field(pose, wrong.line_points.points))
    assert error_right.max() < 0.03
    distance = np.linalg.norm(wrong.line_points.points, axis=1)
    assert error_wrong[distance > 3.0].mean() > 0.15
    assert wrong.diagnostics["lines"]["rejected_by_reason"].get("too_far", 0) \
        > right.diagnostics["lines"]["rejected_by_reason"].get("too_far", 0), \
        "more pixels now project beyond the usable range"
    higher = _extract(extractor, s_field, pose, calibration, truth, truth.perturbed(d_height=0.09))
    camera_xy = truth.T_base_camera[:2, 3]
    ratio = np.linalg.norm(higher.line_points.points - camera_xy, axis=1).min() \
        / np.linalg.norm(right.line_points.points - camera_xy, axis=1).min()
    assert ratio == pytest.approx(1.10, abs=0.03), \
        "believing the camera is 10 % higher stretches every range by 10 %"


def test_nothing_in_view_and_missing_models_give_empty_features(s_field, extractor, calibration):
    blank = np.full((180, 320), schema.FIELD, np.uint8)
    features = extractor.extract(helpers.mask_to_vision(blank), calibration, calibration.pose())
    assert len(features.line_points) == 0 and not features.intersections and not features.circles
    assert build_observation_frame(features, 1.0).is_empty
    no_segmentation = VisionResult(320, 180, (), None)
    assert len(extractor.extract(no_segmentation, calibration, calibration.pose()).line_points) == 0
    sky = extractor.extract(helpers.mask_to_vision(np.full((180, 320), schema.FIELD_LINE, np.uint8)),
                            calibration, calibration.pose(pitch=math.radians(-30.0)))
    assert len(sky.line_points) == 0, "a camera looking at the sky has no ground points"
    assert sky.diagnostics["lines"]["rejected_by_reason"]["above_horizon"] > 0


def test_the_image_must_match_the_calibration(s_field, extractor, calibration):
    wrong_size = helpers.mask_to_vision(np.full((90, 160), schema.FIELD, np.uint8))
    with pytest.raises(ValueError, match="rescale the calibration"):
        extractor.extract(wrong_size, calibration, calibration.pose())
    with pytest.raises(TypeError):
        extractor.extract(helpers.mask_to_vision(np.zeros((180, 320), np.uint8)), calibration,
                          calibration.pose().T_base_camera)


def test_extraction_is_repeatable(s_field, extractor, calibration):
    a = _extract(extractor, s_field, [2.0, -0.5, -0.3], calibration)
    b = _extract(extractor, s_field, [2.0, -0.5, -0.3], calibration)
    np.testing.assert_array_equal(a.line_points.points, b.line_points.points)
    np.testing.assert_array_equal(a.line_points.std, b.line_points.std)
    assert [(j.type, tuple(j.position)) for j in a.intersections] == \
        [(j.type, tuple(j.position)) for j in b.intersections]


# ------------------------------------------------------------------ goal posts
POST_POSE = np.array([2.2, -0.4, 0.15])     # in front of the +x goal


def _posts(field, calibration, camera_pose, base_pose=POST_POSE, mask=None, confidence=0.9,
           believed=None, extra=(), cfg=GoalPostConfig(), noise=MeasurementNoiseConfig()):
    goal = field.goals[0]
    detections = [helpers.post_detection(post, base_pose, calibration, camera_pose,
                                         goal.dims.post_diameter,
                                         goal.dims.height + goal.dims.post_diameter / 2,
                                         confidence) for post in goal.posts]
    detections = [d for d in detections if d is not None] + list(extra)
    k = calibration.intrinsics
    return extract_goal_posts(detections, mask, (k.width, k.height), calibration,
                              believed or camera_pose, goal.dims.post_diameter,
                              goal.dims.height + goal.dims.post_diameter / 2, cfg, noise), detections


def test_the_foot_of_a_leaning_post_is_at_the_inner_corner_of_its_box(s_field):
    """With the camera pitched down a vertical post leans in the image; its box is several
    times wider than the post and the foot is NOT at the middle of the bottom edge."""
    calibration = helpers.small_calibration(640, 360)
    camera = calibration.pose(pan=math.radians(15.0))
    (features, rejected), detections = _posts(s_field, calibration, camera)
    assert len(features) == 2 and not rejected
    true_posts = tf.se2_inverse_transform_points(POST_POSE, np.array(s_field.goals[0].posts))
    for feature in features:
        assert feature.range_valid and feature.flags == ()
        error = np.linalg.norm(true_posts - feature.position, axis=1).min()
        assert error < 0.06, error
    leaning = max(detections, key=lambda d: d.size[0])
    assert leaning.size[0] > 20.0, "the box is far wider than a 10 cm post at 2.5 m"
    vanishing = vertical_vanishing_point(calibration, camera)
    assert vanishing[1] > 2 * calibration.intrinsics.height, "far below the image"
    u, v, width = foot_pixel(leaning.bbox_xyxy, vanishing, 640, 3.0)
    centre_u = 0.5 * (leaning.bbox_xyxy[0] + leaning.bbox_xyxy[2]) - 0.5
    assert abs(u - centre_u) > 5.0 and 4.0 < width < 0.6 * leaning.size[0]
    # The naive bottom-centre would put the post somewhere else.
    naive = gp.project_to_ground([[centre_u, v]], calibration, camera).points[0]
    assert np.linalg.norm(true_posts - naive, axis=1).min() > 0.10


def test_a_post_cut_by_the_bottom_border_gives_a_bearing_only(s_field):
    calibration = helpers.small_calibration(640, 360)
    close = np.array([3.9, -1.25, 0.2])                     # 60 cm from the -y post
    camera = calibration.pose(pitch=math.radians(5.0))      # looking up: the foot is below the image
    (features, _), detections = _posts(s_field, calibration, camera, close)
    cut = [f for f in features if "cut_bottom" in f.flags]
    assert cut, [f.flags for f in features]
    true_posts = tf.se2_inverse_transform_points(close, np.array(s_field.goals[0].posts))
    true_bearings = np.arctan2(true_posts[:, 1], true_posts[:, 0])
    for feature in cut:
        assert not feature.range_valid
        assert feature.uncertainty.range_std >= MeasurementNoiseConfig().bearing_only_range_std_min
        bearing = math.atan2(feature.position[1], feature.position[0])
        assert np.abs(tf.angle_difference(bearing, true_bearings)).min() < math.radians(4.0), \
            "every point of a vertical post has the same bearing"
        assert feature.uncertainty.bearing_std < math.radians(6.0)


def test_a_hidden_foot_or_an_implausible_box_is_not_given_a_range(s_field):
    calibration = helpers.small_calibration(640, 360)
    camera = calibration.pose()
    k = calibration.intrinsics
    turf = np.full((k.height, k.width), schema.FIELD, np.uint8)
    (clean, _), detections = _posts(s_field, calibration, camera, mask=turf)
    assert all(f.range_valid for f in clean)
    # Something that is not turf stands just under the foot (another robot's legs).
    occluded = turf.copy()
    for detection in detections:
        x0, _, x1, y1 = (int(v) for v in detection.bbox_xyxy)
        occluded[y1:y1 + 12, max(x0 - 10, 0):x1 + 10] = schema.BACKGROUND
    (hidden, _), _ = _posts(s_field, calibration, camera, mask=occluded)
    assert all(not f.range_valid and "foot_not_on_turf" in f.flags for f in hidden)
    # A box whose lower half is missing (the detector saw only the top of the post).
    top_half = [Detection(d.class_id, d.confidence,
                          (d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2],
                           0.5 * (d.bbox_xyxy[1] + d.bbox_xyxy[3]))) for d in detections
                if d.bbox_xyxy[1] > 3.0]
    if top_half:
        features, _ = extract_goal_posts(top_half, turf, (k.width, k.height), calibration, camera,
                                         0.10, 1.25, GoalPostConfig(), MeasurementNoiseConfig())
        assert all(not f.range_valid for f in features)
        assert all({"height_mismatch", "width_mismatch"} & set(f.flags) for f in features)
    # A wide box that claims to be a post standing 6 m away.
    fat = Detection(schema.GOAL_POST, 0.9, (300.0, 20.0, 340.0, 100.0))
    features, _ = extract_goal_posts([fat], turf, (k.width, k.height), calibration, camera,
                                     0.10, 1.25, GoalPostConfig(), MeasurementNoiseConfig())
    assert len(features) == 1 and not features[0].range_valid
    assert "width_mismatch" in features[0].flags


def test_implausible_and_weak_post_detections_are_rejected(s_field):
    calibration = helpers.small_calibration(640, 360)
    camera = calibration.pose()
    k = calibration.intrinsics
    detections = [
        Detection(schema.GOAL_POST, 0.2, (100.0, 50.0, 110.0, 200.0)),      # weak
        Detection(schema.GOAL_POST, 0.9, (100.0, 150.0, 300.0, 180.0)),     # lying down
        Detection(schema.BALL, 0.9, (10.0, 10.0, 30.0, 30.0)),              # not a post
    ]
    features, rejected = extract_goal_posts(detections, None, (k.width, k.height), calibration,
                                            camera, 0.10, 1.25, GoalPostConfig(),
                                            MeasurementNoiseConfig())
    assert features == [] and rejected == {"low_confidence": 1, "not_post_shaped": 1}
    # A foot high in the image, above the horizon: no range can come from it.
    floating = Detection(schema.GOAL_POST, 0.9, (300.0, 2.0, 308.0, 20.0))
    features, _ = extract_goal_posts([floating], None, (k.width, k.height), calibration, camera,
                                     0.10, 1.25, GoalPostConfig(), MeasurementNoiseConfig())
    assert len(features) == 1 and not features[0].range_valid
    assert "foot_above_horizon" in features[0].flags


def test_post_uncertainty_reflects_range_confidence_and_occluders(s_field):
    calibration = helpers.small_calibration(640, 360)
    camera = calibration.pose()
    near_pose, far_pose = np.array([2.8, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    (near, _), near_detections = _posts(s_field, calibration, camera, near_pose)
    (far, _), _ = _posts(s_field, calibration, camera, far_pose)
    assert near and far and all(f.range_valid for f in near + far)
    assert min(f.uncertainty.range_std for f in far) > 2.0 * max(f.uncertainty.range_std for f in near)
    (unsure, _), _ = _posts(s_field, calibration, camera, near_pose, confidence=0.4)
    assert unsure[0].uncertainty.range_std > near[0].uncertainty.range_std
    assert unsure[0].confidence == pytest.approx(0.4)
    # A robot box overlapping the foot: keep the range, trust it less.
    x0, _, x1, y1 = near_detections[0].bbox_xyxy
    robot = Detection(schema.ROBOT, 0.9, (x0 - 10.0, y1 - 25.0, x1 + 25.0, y1 + 30.0))
    (crowded, _), _ = _posts(s_field, calibration, camera, near_pose, extra=[robot])
    flagged = [f for f in crowded if "occluder_near_foot" in f.flags]
    assert flagged and flagged[0].range_valid
    assert flagged[0].uncertainty.range_std == pytest.approx(
        MeasurementNoiseConfig().post_inflation_flagged * near[0].uncertainty.range_std, rel=0.05)
    for feature in near:
        assert np.all(np.linalg.eigvalsh(feature.uncertainty.covariance) > 0)


def test_goal_post_position_follows_the_camera_estimate(s_field):
    calibration = helpers.small_calibration(640, 360)
    truth = calibration.pose()
    (right, _), _ = _posts(s_field, calibration, truth)
    (wrong, _), _ = _posts(s_field, calibration, truth,
                           believed=truth.perturbed(d_pitch=math.radians(-2.0)))
    assert np.linalg.norm(wrong[0].position) > np.linalg.norm(right[0].position) + 0.1


# ------------------------------------------------------------------ uncertainty helpers
def test_confidence_and_floors_shape_the_reported_uncertainty(calibration):
    noise = MeasurementNoiseConfig()
    factors = confidence_factor(np.array([1.0, 0.64, 0.25, 0.01]), noise)
    np.testing.assert_allclose(factors, [1.0, 1.25, 2.0, 2.0])
    pose = calibration.pose()
    point = np.array([2.0, 0.5])
    plain = point_uncertainty(point, calibration, pose, noise, pixel_std=1.0,
                              range_std_floor=0.0, bearing_std_floor=0.0)
    floored = point_uncertainty(point, calibration, pose, noise, pixel_std=1.0,
                                range_std_floor=0.3, bearing_std_floor=0.1)
    inflated = point_uncertainty(point, calibration, pose, noise, pixel_std=1.0,
                                 range_std_floor=0.0, bearing_std_floor=0.0, confidence=0.25,
                                 inflation=1.5, extra_variance=0.01)
    assert floored.range_std == pytest.approx(math.hypot(plain.range_std, 0.3))
    assert floored.bearing_std == pytest.approx(math.hypot(plain.bearing_std, 0.1))
    assert inflated.range_std > 3.0 * plain.range_std * 0.99
    np.testing.assert_allclose(plain.covariance, plain.pixel_covariance + plain.camera_covariance)
    covariance = gp.projection_covariance(
        gp.ground_to_pixels(np.array([[3.0, 0.0], [3.0, 0.0]]), calibration, pose)[0], calibration,
        pose)
    normals = np.array([[1.0, 0.0], [np.nan, np.nan]])
    std, pixel, camera = line_point_std(covariance, normals, np.ones(2), noise)
    assert std[0] > std[1], "across a line that faces the robot the full range error applies"
    assert camera[1] == pytest.approx(covariance.std_rms("camera")[1])


# ------------------------------------------------------------------ the bridge
def test_features_become_the_same_observation_frame_the_oracle_produces(s_field, s_landmarks,
                                                                        camera_model, extractor,
                                                                        calibration, rng):
    features = _extract(extractor, s_field, [2.0, -0.5, -0.3], calibration)
    frame = build_observation_frame(features, 3.5)
    oracle_frame, _ = OracleObservationGenerator(s_field, s_landmarks, camera_model).generate(
        [2.0, -0.5, -0.3], rng)
    assert type(frame) is type(oracle_frame) is ObservationFrame
    assert frame.timestamp == 3.5 and frame.frame_id == oracle_frame.frame_id
    assert type(frame.line_points) is LinePointsObservation
    assert frame.line_points.points.shape == (len(features.line_points), 2)
    assert frame.line_points.std.shape == (len(features.line_points),)
    assert all(type(o) is LandmarkObservation for o in frame.landmarks)
    assert {f.name for f in dataclasses.fields(frame)} == \
        {"timestamp", "landmarks", "line_points", "frame_id"}
    assert len(frame.landmarks) == len(features.typed_intersections) + len(features.valid_circles)
    assert {o.type for o in frame.landmarks} <= set(LandmarkType)
    # The Phase 1 filter takes it as it is.
    localizer = ParticleFilter(s_field, s_landmarks, seed=0)
    localizer.init_gaussian([2.0, -0.5, -0.3])
    info = localizer.update(frame)
    assert info.used and np.linalg.norm(localizer.estimate().pose[:2] - [2.0, -0.5]) < 0.3


def test_the_bridge_maps_what_does_not_fit_the_schema_instead_of_extending_it(s_field,
                                                                              calibration):
    calibration = helpers.small_calibration(640, 360)
    camera = calibration.pose(pitch=math.radians(5.0))
    (posts, _), _ = _posts(s_field, calibration, camera, np.array([3.9, -1.25, 0.2]))
    bearing_only = [p for p in posts if not p.range_valid]
    assert bearing_only
    junction_ok = intersections.JunctionCandidate(LandmarkType.T_INTERSECTION,
                                                  np.array([2.0, 0.0]), 0.0, 0.6, 9)
    extractor = GeometricFeatureExtractor(s_field)
    typed = extractor._junction_feature(junction_ok, calibration, calibration.pose())
    untyped = dataclasses.replace(typed, type=None)
    features = FeatureSet(LinePointFeatures.empty(), tuple(posts), (typed, untyped), ())
    frame = build_observation_frame(features, 0.0)
    kinds = [o.type for o in frame.landmarks]
    assert kinds.count(LandmarkType.T_INTERSECTION) == 1, "the unclassed junction is left out"
    assert kinds.count(LandmarkType.GOAL_POST) == len(posts)
    wide = [o for o in frame.landmarks if o.type == LandmarkType.GOAL_POST and o.range_std >= 1.5]
    assert len(wide) == len(bearing_only), "bearing only = a range the localizer cannot use"
    assert all(o.bearing_std < math.radians(6.0) for o in wide)

    without = build_observation_frame(features, 0.0, ObservationSelection(
        bearing_only_goal_posts=False, intersections=()))
    assert len(without.landmarks) == len(posts) - len(bearing_only)
    nothing = build_observation_frame(features, 0.0, ObservationSelection(
        goal_posts=False, intersections=(), center_circle=False, line_points=False))
    assert nothing.is_empty
    capped = build_observation_frame(features, 0.0, ObservationSelection(max_landmarks=1))
    assert len(capped.landmarks) == 1
