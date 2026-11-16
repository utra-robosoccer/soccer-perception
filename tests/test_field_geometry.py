"""Field geometry, landmarks and the line distance map."""
import math

import numpy as np
import pytest

from robosoccer_perception import config
from robosoccer_perception.geometry.distance_map import DistanceMap, get_distance_map
from robosoccer_perception.geometry.field import FieldDimensions, FieldGeometry
from robosoccer_perception.geometry.landmarks import (
    EXPECTED_COUNTS, LandmarkType, build_landmark_map)

# The numbers of the task specification, written out independently of the YAML files.
SPEC = {
    "s_field": dict(length=9.0, width=6.0, line_width=0.05, goal_area=(1.0, 3.0),
                    penalty_area=(2.0, 4.0), mark=1.5, circle=1.5, corner=0.0,
                    goal_width=(1.8, 1.9), goal_depth=(0.5, 1.0)),
    "m_field": dict(length=14.0, width=9.0, line_width=0.05, goal_area=(1.0, 4.0),
                    penalty_area=(3.0, 6.0), mark=2.0, circle=3.0, corner=0.5,
                    goal_width=(2.4, 2.6), goal_depth=(0.7, 1.2)),
    "l_field": dict(length=22.0, width=14.0, line_width=0.12, goal_area=(1.0, 5.0),
                    penalty_area=(3.5, 7.0), mark=2.5, circle=4.0, corner=1.0,
                    goal_width=(2.6, 3.1), goal_depth=(1.0, 2.0)),
}
EPS = 1e-6


def on(field, x, y):
    return bool(field.is_on_paint([x, y]))


def test_configs_match_the_rulebook_numbers(any_field):
    spec, d = SPEC[any_field.name], any_field.dims
    assert (d.length, d.width, d.line_width) == (spec["length"], spec["width"], spec["line_width"])
    assert (d.goal_area_depth, d.goal_area_width) == spec["goal_area"]
    assert (d.penalty_area_depth, d.penalty_area_width) == spec["penalty_area"]
    assert d.penalty_mark_distance == spec["mark"]
    assert d.center_circle_diameter == spec["circle"]
    assert d.corner_arc_radius == spec["corner"]
    assert d.border_strip_width >= 1.0
    assert spec["goal_width"][0] <= d.goal.width <= spec["goal_width"][1]
    assert spec["goal_depth"][0] <= d.goal.depth <= spec["goal_depth"][1]


def test_boundary_lines_are_measured_to_their_outer_edge(any_field):
    f, w = any_field, any_field.dims.line_width
    a, b = f.half_length, f.half_width
    y = 0.37 * b                                   # away from every other line
    for s in (1, -1):
        assert on(f, s * (a - EPS), y), "outer edge of the goal line is at length / 2"
        assert not on(f, s * (a + 1e-3), y), "nothing is painted beyond the field"
        assert on(f, s * (a - w + EPS), y) and not on(f, s * (a - w - 1e-3), y)
        x = 0.9 * (a - f.dims.penalty_area_depth)  # between the circle and the penalty area
        if abs(x) < f.dims.center_circle_diameter / 2:
            x = 0.5 * (a - f.dims.penalty_area_depth + f.dims.center_circle_diameter / 2)
        assert on(f, x, s * (b - EPS)) and not on(f, x, s * (b + 1e-3))
        assert on(f, x, s * (b - w + EPS)) and not on(f, x, s * (b - w - 1e-3))
    assert f.segment("goal_line_pos_x").start[0] == pytest.approx(a - w / 2)
    assert f.segment("touchline_pos_y").start[1] == pytest.approx(b - w / 2)


def test_halfway_line_is_centred(any_field):
    f, w = any_field, any_field.dims.line_width
    y = f.dims.center_circle_diameter / 2 + 0.2
    assert on(f, w / 2 - EPS, y) and on(f, -w / 2 + EPS, y)
    assert not on(f, w / 2 + 1e-3, y) and not on(f, -w / 2 - 1e-3, y)


@pytest.mark.parametrize("area", ["goal_area", "penalty_area"])
def test_area_depth_and_width_are_outside_measurements(any_field, area):
    f, d = any_field, any_field.dims
    w, a = d.line_width, any_field.half_length
    depth, width = getattr(d, f"{area}_depth"), getattr(d, f"{area}_width")
    for s in (1, -1):
        # Front line: its far edge is exactly `depth` from the outer edge of the goal line.
        assert on(f, s * (a - depth + EPS), 0.123)
        assert not on(f, s * (a - depth - 1e-3), 0.123)
        assert on(f, s * (a - depth + w - EPS), 0.123)
        # Side lines: outer edges are `width` apart.
        x = s * (a - depth / 2 if area == "goal_area" else a - (depth + d.goal_area_depth) / 2)
        for sy in (1, -1):
            assert on(f, x, sy * (width / 2 - EPS))
            assert not on(f, x, sy * (width / 2 + 1e-3))
            assert on(f, x, sy * (width / 2 - w + EPS))
            assert not on(f, x, sy * (width / 2 - w - 1e-3))


def test_center_circle_radius(any_field):
    f, d = any_field, any_field.dims
    w, outer = d.line_width, d.center_circle_diameter / 2
    assert f.center_circle_radius == pytest.approx(outer - w / 2)
    for angle in np.linspace(0.3, 2 * math.pi, 7):     # avoid the halfway line at +-90 deg
        if abs(math.cos(angle)) < 0.2:
            continue
        c, s = math.cos(angle), math.sin(angle)
        assert on(f, (outer - EPS) * c, (outer - EPS) * s), "outer edge at diameter / 2"
        assert not on(f, (outer + 1e-3) * c, (outer + 1e-3) * s)
        assert on(f, (outer - w + EPS) * c, (outer - w + EPS) * s)
        assert not on(f, (outer - w - 1e-3) * c, (outer - w - 1e-3) * s)


def test_marks(any_field):
    f, d = any_field, any_field.dims
    assert f.center_mark == (0.0, 0.0) and on(f, 0.0, 0.0)
    for s in (1, -1):
        mx, my = f.penalty_mark(s)
        assert mx == pytest.approx(s * (f.half_length - d.penalty_mark_distance)) and my == 0.0
        assert on(f, mx, my)
        assert d.mark_size > 2 * d.line_width, "the mark must read as a cross, not a blob"
        assert on(f, mx + 0.4 * d.mark_size, my) and on(f, mx, my + 0.4 * d.mark_size)
        assert not on(f, mx + 0.4 * d.mark_size, my + 0.4 * d.mark_size)


def test_corner_arcs_only_where_configured(any_field):
    f, r = any_field, any_field.dims.corner_arc_radius
    arcs = [a for a in f.arcs if a.kind == "corner_arc"]
    if r == 0.0:
        assert arcs == []
        return
    assert len(arcs) == 4
    w = f.dims.line_width
    for arc in arcs:
        assert abs(arc.center[0]) == f.half_length and abs(arc.center[1]) == f.half_width
        assert arc.radius == pytest.approx(r - w / 2)
        mid = 0.5 * (arc.start_angle + arc.end_angle)
        direction = np.array([math.cos(mid), math.sin(mid)])
        inside = np.asarray(arc.center) + (r - w / 2) * direction
        assert abs(inside[0]) < f.half_length and abs(inside[1]) < f.half_width, "opens inward"
        assert on(f, *(np.asarray(arc.center) + (r - EPS) * direction))
        assert not on(f, *(np.asarray(arc.center) + (r + 1e-3) * direction))


def test_goals(any_field):
    f, d = any_field, any_field.dims
    assert [g.side for g in f.goals] == [1, -1]
    for goal in f.goals:
        (x1, y1), (x2, y2) = goal.posts
        assert x1 == x2 == pytest.approx(goal.side * (f.half_length - d.line_width / 2))
        assert y1 == -y2 > 0
        inner_gap = (y1 - y2) - d.goal.post_diameter
        assert inner_gap == pytest.approx(d.goal.width), "goal width is the inner post distance"
    half_x, half_y = f.half_extent
    assert half_x >= f.half_length + d.goal.depth, "the goals stand on the carpet"
    assert half_y == pytest.approx(f.half_width + d.border_strip_width)


def test_painted_geometry_is_point_symmetric_and_mirror_symmetric(any_field, rng):
    half_x, half_y = any_field.half_extent
    points = rng.uniform([-half_x, -half_y], [half_x, half_y], size=(4000, 2))
    reference = any_field.distance_to_paint(points)
    for flip in ([-1, -1], [-1, 1], [1, -1]):
        np.testing.assert_allclose(any_field.distance_to_paint(points * flip), reference,
                                   atol=1e-9)


def test_landmark_counts_and_classes(any_field):
    landmarks = build_landmark_map(any_field)
    for kind, count in EXPECTED_COUNTS.items():
        assert len(landmarks.positions(kind)) == count, kind
    assert len(landmarks) == sum(EXPECTED_COUNTS.values())
    assert len({lm.id for lm in landmarks.landmarks}) == len(landmarks)


def test_intersection_locations(s_field, s_landmarks):
    # S field, line width 0.05: boundary centrelines at 4.475 and 2.975.
    by_id = {lm.id: lm.position for lm in s_landmarks.landmarks}
    assert by_id["L:corner:pos_x:pos_y"] == pytest.approx((4.475, 2.975))
    assert by_id["L:corner:neg_x:neg_y"] == pytest.approx((-4.475, -2.975))
    assert by_id["T:halfway:pos_y"] == pytest.approx((0.0, 2.975))
    assert by_id["T:goal_area:pos_x:pos_y"] == pytest.approx((4.475, 1.475))
    assert by_id["L:goal_area:pos_x:neg_y"] == pytest.approx((3.525, -1.475))
    assert by_id["T:penalty_area:neg_x:pos_y"] == pytest.approx((-4.475, 1.975))
    assert by_id["L:penalty_area:neg_x:pos_y"] == pytest.approx((-2.525, 1.975))
    assert by_id["X:center_circle:pos_y"] == pytest.approx((0.0, 0.725))
    assert by_id["penalty_mark:pos_x"] == pytest.approx((3.0, 0.0))
    assert by_id["goal_post:goal_pos_x:pos_y"] == pytest.approx((4.475, 0.95))
    assert by_id["center_mark"] == (0.0, 0.0)


def test_every_junction_lies_on_paint_and_landmarks_are_symmetric(any_field):
    landmarks = build_landmark_map(any_field)
    for kind in LandmarkType:
        positions = landmarks.positions(kind)
        if kind != LandmarkType.GOAL_POST:
            assert np.all(any_field.is_on_paint(positions)), kind
        # A half turn of the field maps each class onto itself: the built-in ambiguity.
        as_set = {tuple(np.round(p, 9)) for p in positions}
        assert {tuple(np.round(-p, 9) + 0.0) for p in positions} == \
            {tuple(np.asarray(p) + 0.0) for p in as_set}, kind


def test_distance_map_matches_exact_distances(any_field, rng):
    dmap = get_distance_map(any_field)
    half_x, half_y = any_field.half_extent
    points = rng.uniform([-half_x, -half_y], [half_x, half_y], size=(20000, 2))
    error = np.abs(dmap.lookup(points) - any_field.distance_to_paint(points))
    assert error.max() < 0.75 * dmap.resolution
    assert error.mean() < 0.1 * dmap.resolution
    assert dmap.data.dtype == np.float32
    assert dmap.memory_bytes < 4e6


def test_distance_map_values(s_field):
    dmap = get_distance_map(s_field)
    assert dmap.lookup([0.0, 2.0]) == pytest.approx(0.0, abs=1e-6)        # on the halfway line
    # 0.45 m inside the inner edge (y = 2.95) of the +y touchline, clear of every other line.
    assert dmap.lookup([1.5, 2.5]) == pytest.approx(0.45, abs=0.01)
    # On the x axis the nearest paint is the tip of the centre mark's arm.
    arm_tip = s_field.dims.mark_size / 2
    assert dmap.lookup([0.3625, 0.0]) == pytest.approx(0.3625 - arm_tip, abs=0.01)
    # 0.5 m beyond the outer edge of the +x goal line, away from other lines.
    assert dmap.lookup([5.0, 2.5]) == pytest.approx(0.5, abs=0.01)


def test_distance_map_outside_the_carpet_keeps_growing(s_field):
    dmap = get_distance_map(s_field)
    half_x, _ = s_field.half_extent
    edge = dmap.lookup([half_x, 0.3])
    assert dmap.lookup([half_x + 2.0, 0.3]) == pytest.approx(edge + 2.0, abs=1e-6)
    assert np.isfinite(dmap.lookup([[1e3, -1e3]])).all()


def test_distance_map_vectorised_shapes(s_field, rng):
    dmap = get_distance_map(s_field)
    points = rng.uniform(-3, 3, size=(50, 7, 2))
    assert dmap.lookup(points).shape == (50, 7)


def test_distance_map_is_regenerated_when_the_field_changes(s_field):
    first = get_distance_map(s_field)
    assert get_distance_map(FieldGeometry.load("s_field")) is first, "same field: cached"
    changed = dict(config.load_config("fields", "s_field"))
    changed["center_circle_diameter"] = 2.0
    wider = FieldGeometry(FieldDimensions.from_dict(changed))
    assert wider.content_hash() != s_field.content_hash()
    second = get_distance_map(wider)
    assert second is not first
    assert second.lookup([0.975, 0.0]) == pytest.approx(0.0, abs=1e-6)     # new circle
    assert first.lookup([0.975, 0.0]) > 0.2                                # old circle


def test_field_size_switching():
    sizes = {}
    for name in SPEC:
        field = FieldGeometry.load(name)
        dmap = DistanceMap.from_field(field, 0.05)
        sizes[name] = (field.half_extent, dmap.shape)
        assert dmap.field_hash == field.content_hash()
    assert sizes["s_field"][1] < sizes["m_field"][1] < sizes["l_field"][1]
    assert sizes["l_field"][0][0] >= 11.0 + 1.0


def test_invalid_configuration_is_rejected():
    data = dict(config.load_config("fields", "s_field"))
    with pytest.raises(ValueError, match="penalty_area_width"):
        FieldDimensions.from_dict({**data, "penalty_area_width": 2.0})
    with pytest.raises(ValueError, match="unknown"):
        FieldDimensions.from_dict({**data, "lenght": 9.0})
    with pytest.raises(FileNotFoundError):
        FieldGeometry.load("xl_field")


def test_line_point_sampling(any_field):
    points = any_field.sample_line_points(0.1)
    assert np.all(any_field.is_on_paint(points))
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    assert gaps.min(axis=1).max() <= 0.1 + 1e-9, "no point is farther than the spacing from a neighbour"
