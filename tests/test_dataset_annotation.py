"""Ground-truth labels: boxes around visible pixels, clipping, occlusion, and the semantic mask.

Tests marked ``render`` need an OpenGL context."""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.datasets.annotation import (  # noqa: E402
    SceneLabeler, mask_box, reduce_semantic)
from robosoccer_perception.datasets.schema import LabelPolicy, ObjectAnnotation  # noqa: E402
from robosoccer_perception.geometry.camera import StereoCameraConfig  # noqa: E402
from robosoccer_perception.inference import geometry  # noqa: E402
from robosoccer_perception.simulation.mujoco_scene import (  # noqa: E402
    LEFT_CAMERA, MujocoScene, load_scene_config)
from robosoccer_perception.simulation.scene_objects import RobotState  # noqa: E402

WIDTH, HEIGHT = 640, 360
POSE = [-1.5, 0.2, math.radians(-10)]


@pytest.fixture(scope="module")
def scene(s_field):
    cfg = load_scene_config(overrides={"robots": [{}, {"height": 1.3, "build": 1.3}],
                                       "offscreen_scale": 2, "lighting": {"shadows": False}})
    camera_cfg = StereoCameraConfig.load().with_resolution(WIDTH, HEIGHT)
    with MujocoScene(s_field, camera_cfg, cfg) as scene:
        yield scene


@pytest.fixture
def labeler(scene, gl_context):
    scene.teleport(POSE, scene.camera_cfg.mount)
    scene.set_robots([None, None])
    scene.set_ball(None)
    return SceneLabeler(scene)


def find(labels, instance) -> ObjectAnnotation:
    return next(obj for obj in labels.objects if obj.instance == instance)


# ------------------------------------------------------------------ pure functions
def test_reduce_semantic_labels_by_coverage():
    B, F, L = schema.BACKGROUND, schema.FIELD, schema.FIELD_LINE
    fine = np.array([[L, L, F, F, B, B, L, F],
                     [F, F, F, F, B, F, B, B]], dtype=np.uint8)
    np.testing.assert_array_equal(reduce_semantic(fine, 2, 0.5), [[L, F, B, B]])
    np.testing.assert_array_equal(reduce_semantic(fine, 2, 0.25), [[L, F, B, L]])
    np.testing.assert_array_equal(reduce_semantic(fine, 1, 0.5), fine)


def test_mask_box_uses_pixel_edge_coordinates():
    mask = np.zeros((8, 12), bool)
    assert mask_box(mask, 1) is None
    mask[2:5, 3:9] = True
    assert mask_box(mask, 1) == (3.0, 2.0, 9.0, 5.0)
    assert mask_box(mask, 2) == (1.5, 1.0, 4.5, 2.5)        # the same mask on a 2x finer grid


def test_yolo_lines_round_trip():
    annotation = ObjectAnnotation("ball", schema.BALL, True, "ok", (100.0, 50.0, 140.0, 80.0),
                                  900.0, 900.0, 1.0, False, 2.0, (1.0, 0.0, 0.07))
    line = schema.yolo_line(annotation, WIDTH, HEIGHT)
    assert line.split()[0] == "0" and len(line.split()) == 5
    (class_id, box), = schema.parse_yolo_labels(line + "\n", WIDTH, HEIGHT)
    assert class_id == schema.BALL
    np.testing.assert_allclose(box, annotation.bbox_xyxy, atol=1e-3)
    assert schema.parse_yolo_labels("", WIDTH, HEIGHT) == []


def test_label_policy_validation_and_class_ids():
    assert schema.DETECTION_CLASSES == ("ball", "goal_post", "robot")
    assert schema.SEMANTIC_CLASSES == ("background", "field", "field_line")
    assert LabelPolicy.from_dict({"supersample": 3}).supersample == 3
    assert LabelPolicy.from_dict(None) == LabelPolicy()
    with pytest.raises(ValueError):
        LabelPolicy(supersample=0)
    with pytest.raises(ValueError):
        LabelPolicy(line_coverage_threshold=0.0)


# ------------------------------------------------------------------ object boxes
@pytest.mark.render
@pytest.mark.parametrize("ball_xy", [(0.0, 0.0), (1.5, 0.6), (4.0, -1.0)])
def test_ball_box_is_the_projected_ball_not_an_invented_size(scene, labeler, ball_xy):
    scene.set_ball(ball_xy)
    ball = find(labeler.annotate(), "ball")
    assert ball.labeled and ball.reason == "ok" and not ball.truncated
    assert ball.visible_fraction == pytest.approx(1.0)
    radius = scene.scene_cfg["ball"]["radius"]
    centre, depth, _ = scene.camera_model.project_field_points(POSE, [*ball_xy, radius])
    x0, y0, x1, y1 = ball.bbox_xyxy
    # Pixel-edge box centre versus pixel-centre projection: half a pixel apart by convention.
    np.testing.assert_allclose([(x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5], centre, atol=1.0)
    expected_diameter = 2 * scene.camera_cfg.intrinsics.fx * radius / depth
    assert x1 - x0 == pytest.approx(expected_diameter, rel=0.12, abs=1.0)
    assert y1 - y0 == pytest.approx(expected_diameter, rel=0.12, abs=1.0)
    assert ball.distance_m == pytest.approx(
        np.hypot(*(np.array(ball_xy) - scene.camera_pose(LEFT_CAMERA)[:2, 3])), abs=1e-6)


@pytest.mark.render
def test_goal_post_box_spans_the_post(scene, labeler, s_field):
    scene.teleport([1.0, 0.0, 0.0], scene.camera_cfg.mount)
    labels = labeler.annotate()
    posts = [obj for obj in labels.objects if obj.class_id == schema.GOAL_POST]
    assert len(posts) == 4
    near = [p for p in posts if p.attributes["goal"] == "goal_pos_x"]
    assert all(p.labeled for p in near) and not any(p.labeled for p in posts if p not in near)
    assert {p.reason for p in posts if p not in near} == {"out_of_view"}
    for post, xy in zip(near, s_field.goals[0].posts):
        foot, _, _ = scene.camera_model.project_field_points([1.0, 0.0, 0.0], [*xy, 0.0])
        x0, y0, x1, y1 = post.bbox_xyxy
        assert x0 - 1 <= foot[0] + 0.5 <= x1 + 1, "the post's foot is inside the box, sideways"
        assert y1 == pytest.approx(foot[1] + 0.5, abs=3.0), "the box ends where the post meets the turf"
        assert (y1 - y0) > 3 * (x1 - x0), "a post is tall and thin"


@pytest.mark.render
def test_box_is_clipped_at_the_image_border(scene, labeler):
    k = scene.camera_cfg.intrinsics
    scene.set_ball([0.5, 0.2])
    last_full_width, ball = None, None
    for _ in range(160):                             # turn right until the border cuts the ball
        scene.rotate_in_place(-math.radians(0.5))
        ball = find(labeler.annotate(), "ball")
        if ball.bbox_xyxy is None:
            break
        width = ball.bbox_xyxy[2] - ball.bbox_xyxy[0]
        if not ball.truncated:
            last_full_width = width
        elif width < 0.7 * last_full_width:
            break
    assert ball.truncated and ball.labeled and last_full_width is not None
    x0, y0, x1, y1 = ball.bbox_xyxy
    assert x0 == 0.0, "the box stops at the image border"
    assert 0 < x1 - x0 < 0.7 * last_full_width, "only the visible part is boxed"
    assert 0 <= y0 < y1 <= k.height
    line = schema.yolo_line(ball, k.width, k.height).split()
    cx, w = float(line[1]), float(line[3])
    assert cx - w / 2 == pytest.approx(0.0, abs=1e-6)


@pytest.mark.render
def test_fully_occluded_object_is_not_labelled(scene, labeler, s_field):
    """A ball hidden behind a goal post: in the frustum, zero visible pixels, no label."""
    post = np.array(s_field.goals[0].posts[0])
    camera = post + [-1.0, 0.0]
    scene.teleport([camera[0] - scene.camera_cfg.mount.forward_offset, camera[1], 0.0],
                   scene.camera_cfg.mount)
    eye = scene.camera_pose(LEFT_CAMERA)[:2, 3]
    direction = (post - eye) / np.linalg.norm(post - eye)
    scene.set_ball(post + 1.2 * direction)           # straight behind the post, as seen by the left eye
    hidden = find(labeler.annotate(), "ball")
    assert not hidden.labeled and hidden.reason == "occluded"
    assert hidden.amodal_pixels > 20 and hidden.visible_fraction < 0.15
    labels = labeler.annotate()
    assert "0 " not in {line[:2] for line in labels.yolo_text().splitlines()}, "no ball line"
    scene.set_ball(post + 1.2 * direction + [0.0, 0.5])      # step aside: visible again
    assert find(labeler.annotate(), "ball").labeled


@pytest.mark.render
def test_partial_occlusion_boxes_only_the_visible_part(scene, labeler):
    scene.set_ball([1.0, 0.0])
    free = find(labeler.annotate(), "ball")
    # A robot's leg in front of the ball.
    eye = scene.camera_pose(LEFT_CAMERA)[:2, 3]
    ball_xy = np.array([1.0, 0.0])
    spot = eye + 0.7 * (ball_xy - eye)
    best = None
    for offset in np.linspace(-0.25, 0.25, 21):      # slide the robot sideways across the view line
        across = np.array([-(ball_xy - eye)[1], (ball_xy - eye)[0]]) / np.linalg.norm(ball_xy - eye)
        scene.set_robots([None, RobotState(tuple(spot + offset * across), 0.0)])
        ball = find(labeler.annotate(), "ball")
        if 0.2 < ball.visible_fraction < 0.8:
            best = ball
            break
    assert best is not None, "some robot position half-hides the ball"
    assert best.labeled and best.amodal_pixels == pytest.approx(free.amodal_pixels, rel=0.02)
    assert best.visible_pixels < 0.85 * free.visible_pixels
    fx0, fy0, fx1, fy1 = free.bbox_xyxy
    x0, y0, x1, y1 = best.bbox_xyxy
    assert fx0 - 1e-6 <= x0 and x1 <= fx1 + 1e-6 and fy0 - 1e-6 <= y0 and y1 <= fy1 + 1e-6
    assert (x1 - x0) * (y1 - y0) < (fx1 - fx0) * (fy1 - fy0)
    robot = find(labeler.annotate(), "robot1")
    assert robot.labeled and robot.visible_fraction > 0.95, "the robot in front is fully visible"


@pytest.mark.render
def test_reasons_for_unlabelled_objects(scene, labeler):
    labels = labeler.annotate()                                  # ball hidden, robots hidden
    assert find(labels, "ball").reason == "hidden" and find(labels, "robot0").reason == "hidden"
    assert labels.yolo_text().count("\n") == len(labels.labeled)
    scene.set_ball([-4.0, 0.2])                                  # behind the camera
    behind = find(labeler.annotate(), "ball")
    assert behind.reason == "out_of_view" and behind.bbox_xyxy is None and not behind.labeled
    scene.set_ball([3.0, 0.0])
    strict = SceneLabeler(scene, LabelPolicy(min_visible_pixels=5000.0))
    small = find(strict.annotate(), "ball")
    assert small.reason == "too_small" and not small.labeled and small.bbox_xyxy is not None


# ------------------------------------------------------------------ semantic mask
@pytest.mark.render
def test_semantic_mask_matches_the_rendered_colours(scene, labeler):
    """Pixel alignment of mask and image: line pixels are white, field pixels are green."""
    import cv2

    scene.set_ball([1.0, 0.3])
    labels = labeler.annotate()
    image = scene.render(LEFT_CAMERA, show_platform=False).astype(int)
    mask = labels.semantic
    assert mask.shape == (HEIGHT, WIDTH) and mask.dtype == np.uint8
    assert set(np.unique(mask)) == {schema.BACKGROUND, schema.FIELD, schema.FIELD_LINE}
    kernel = np.ones((3, 3), np.uint8)
    line_core = cv2.erode((mask == schema.FIELD_LINE).astype(np.uint8), kernel).astype(bool)
    field_core = cv2.erode((mask == schema.FIELD).astype(np.uint8), kernel).astype(bool)
    assert line_core.sum() > 300 and field_core.sum() > 10000
    r, g, b = (image[..., i] for i in range(3))
    white = (np.minimum(np.minimum(r, g), b) > 150)
    green = (g > 1.4 * r) & (g > 1.4 * b)
    assert white[line_core].mean() > 0.98
    assert green[field_core].mean() > 0.98
    assert not white[field_core].mean() > 0.01


@pytest.mark.render
def test_line_pixels_lie_on_the_painted_lines(scene, labeler, s_field):
    """Back-project every line-labelled pixel onto the ground: it lands on paint."""
    labels = labeler.annotate()
    v, u = np.nonzero(labels.semantic == schema.FIELD_LINE)
    T_field_camera = scene.camera_pose(LEFT_CAMERA)
    points, valid = geometry.pixels_to_ground(np.column_stack([u, v]),
                                              scene.camera_cfg.intrinsics, T_field_camera)
    assert valid.all() and len(points) > 1000
    distance = np.hypot(*(points - T_field_camera[:2, 3]).T)
    # One pixel covers more ground the farther away it looks; allow that footprint.
    footprint = distance ** 2 / (scene.camera_cfg.intrinsics.fx * 0.9)
    error = s_field.distance_to_paint(points)
    assert (error <= footprint + 0.01).mean() > 0.99
    assert np.median(error[distance < 4.0]) < 0.005
    # And turf labelled "field" well inside the image is not paint.
    v, u = np.nonzero(labels.semantic == schema.FIELD)
    points, valid = geometry.pixels_to_ground(np.column_stack([u, v])[::7],
                                              scene.camera_cfg.intrinsics, T_field_camera)
    near = valid & (np.hypot(*(points - T_field_camera[:2, 3]).T) < 4.0)
    assert (s_field.distance_to_paint(points[near]) > 0).mean() > 0.97


@pytest.mark.render
def test_objects_standing_on_lines_are_not_labelled_as_lines(scene, labeler, s_field):
    """Occluder policy: pixels showing a robot, the ball or a post are ``background``."""
    scene.teleport([-2.5, 0.0, 0.0], scene.camera_cfg.mount)
    before = labeler.annotate().semantic
    scene.set_robots([RobotState((0.0, 0.0), 1.0), None])          # on the halfway line
    scene.set_ball([-0.75, 0.3])
    labels = labeler.annotate()
    geom_image = scene.render_segmentation(LEFT_CAMERA)
    robot_pixels = np.isin(geom_image, scene.robot_geom_ids(0))
    ball_pixels = geom_image == scene.model.geom("ball").id
    assert robot_pixels.sum() > 500 and ball_pixels.sum() > 30
    core = np.zeros_like(robot_pixels)
    core[1:-1, 1:-1] = (robot_pixels[1:-1, 1:-1] & robot_pixels[:-2, 1:-1] & robot_pixels[2:, 1:-1]
                        & robot_pixels[1:-1, :-2] & robot_pixels[1:-1, 2:])
    assert (before[core] == schema.FIELD_LINE).sum() > 20, "the robot does cover line pixels"
    assert (labels.semantic[core] == schema.BACKGROUND).all()
    assert (labels.semantic[ball_pixels] == schema.FIELD_LINE).sum() == 0
    untouched = ~(robot_pixels | ball_pixels)
    untouched[1:-1, 1:-1] &= ~(robot_pixels[:-2, 1:-1] | robot_pixels[2:, 1:-1]
                               | robot_pixels[1:-1, :-2] | robot_pixels[1:-1, 2:])
    assert (labels.semantic[untouched] == before[untouched]).mean() > 0.999
    robot = find(labels, "robot0")
    v, u = np.nonzero(robot_pixels)
    np.testing.assert_allclose(robot.bbox_xyxy, [u.min(), v.min(), u.max() + 1, v.max() + 1], atol=1.0)
