"""The virtual stereo camera and the MuJoCo scene.

Tests marked ``render`` need an OpenGL context and are skipped when none is available. The
others only compile the model and read its kinematics.
"""
import dataclasses
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.camera import (  # noqa: E402
    CameraIntrinsics, RigState, StereoCameraConfig, StereoCameraModel)
from robosoccer_perception.geometry.field import FieldGeometry  # noqa: E402
from robosoccer_perception.geometry.landmarks import LandmarkType, build_landmark_map  # noqa: E402
from robosoccer_perception.simulation import trajectories  # noqa: E402
from robosoccer_perception.simulation.ground_truth import compute_ground_truth  # noqa: E402
from robosoccer_perception.simulation.mujoco_scene import (  # noqa: E402
    LEFT_CAMERA, RIGHT_CAMERA, MujocoScene, load_scene_config, overview_intrinsics,
    overview_pixel_to_field)
from robosoccer_perception.simulation.stereo_camera import (  # noqa: E402
    VirtualStereoCamera, save_stereo_frame)

NO_SHADOWS = {"lighting": {"shadows": False}}


@pytest.fixture(scope="module")
def scene(s_field):
    with MujocoScene(s_field, scene_cfg=load_scene_config(overrides=NO_SHADOWS),
                     scenario_id="test") as scene:
        yield scene


@pytest.fixture(scope="module")
def gl(scene):
    """Skip rendering tests when no OpenGL context can be created."""
    try:
        scene.render(LEFT_CAMERA)
    except Exception as error:  # noqa: BLE001 - any context failure means "cannot render here"
        pytest.skip(f"MuJoCo offscreen rendering unavailable: {error!r}")


def changed_pixels(scene, camera_name, ball_xy):
    """Pixels (N, 2 as u, v) that change when the ball is put at ``ball_xy``."""
    scene.set_ball([40.0, 40.0])
    empty = scene.render(camera_name, show_platform=False).astype(int)
    scene.set_ball(ball_xy)
    with_ball = scene.render(camera_name, show_platform=False).astype(int)
    rows, cols = np.nonzero(np.abs(with_ball - empty).sum(axis=-1) > 30)
    return np.column_stack([cols, rows])


# ------------------------------------------------------------------ geometry only
def test_disparity_is_consistent_with_the_baseline(camera_model, rng):
    rig = RigState(height=0.9, pitch=math.radians(25))
    pose = [0.5, -1.0, 0.7]
    T_field_left = camera_model.T_field_camera(pose, "left", rig)
    points_left = np.column_stack([rng.uniform(-1, 1, 50), rng.uniform(-0.5, 0.5, 50),
                                   rng.uniform(0.5, 12.0, 50)])
    points_field = tf.transform_points(T_field_left, points_left)
    px_left, z_left, _ = camera_model.project_field_points(pose, points_field, "left", rig)
    px_right, z_right, _ = camera_model.project_field_points(pose, points_field, "right", rig)
    np.testing.assert_allclose(z_left, z_right, atol=1e-9)           # parallel optical axes
    np.testing.assert_allclose(px_left[:, 1], px_right[:, 1], atol=1e-9)   # same image row
    disparity = px_left[:, 0] - px_right[:, 0]
    assert np.all(disparity > 0), "a point is further right in the left image"
    np.testing.assert_allclose(disparity, camera_model.intrinsics.fx * 0.063 / z_left, atol=1e-9)
    np.testing.assert_allclose(camera_model.depth_from_disparity(disparity), z_left, atol=1e-9)


def test_camera_profile_is_labelled_approximate_and_matches_the_datasheet_fov():
    cfg = StereoCameraConfig.load("zed_mini_sim")
    assert cfg.approximate and cfg.baseline == pytest.approx(0.063)
    meta = cfg.metadata()
    assert meta["horizontal_fov_deg"] == pytest.approx(90.0, abs=0.1)
    assert meta["vertical_fov_deg"] == pytest.approx(58.7, abs=0.1)
    half = cfg.with_resolution(640, 360)
    assert math.degrees(half.intrinsics.horizontal_fov) == pytest.approx(90.0, abs=0.1)
    assert half.intrinsics.fx == pytest.approx(320.0)


# ------------------------------------------------------------------ model and kinematics
def test_scene_contents(scene, s_field):
    names = {scene.model.geom(i).name for i in range(scene.model.ngeom)}
    for segment in s_field.segments:
        assert f"line/{segment.name}" in names
    assert any(name.startswith("line/center_circle/") for name in names)
    for goal in s_field.goals:
        assert {f"{goal.name}/post_pos_y", f"{goal.name}/post_neg_y",
                f"{goal.name}/crossbar"} <= names
        for label, post in zip(("pos_y", "neg_y"), goal.posts):
            geom = scene.data.geom(f"{goal.name}/post_{label}")
            np.testing.assert_allclose(geom.xpos[:2], post, atol=1e-9)
            assert scene.model.geom(geom.id).size[0] == pytest.approx(
                goal.dims.post_diameter / 2)
    assert {"ball", "turf", "surround"} <= names
    turf = scene.model.geom("turf")
    np.testing.assert_allclose(turf.size[:2], s_field.half_extent)
    assert scene.model.nlight >= 1


@pytest.mark.parametrize("name", ["m_field", "l_field"])
def test_other_fields_compile_without_code_changes(name):
    field = FieldGeometry.load(name)
    with MujocoScene(field) as other:
        arcs = [other.model.geom(i).name for i in range(other.model.ngeom)
                if other.model.geom(i).name.startswith("line/corner_arc")]
        assert arcs, "M and L fields have corner arcs"
        np.testing.assert_allclose(other.model.geom("turf").size[:2], field.half_extent)
        post = other.data.geom("goal_pos_x/post_pos_y")
        np.testing.assert_allclose(post.xpos[:2], field.goals[0].posts[0], atol=1e-9)


def test_teleport_to_arbitrary_poses_and_ground_truth(scene, rng):
    model = scene.camera_model
    for _ in range(25):
        pose = rng.uniform([-5, -3.5, -math.pi], [5, 3.5, math.pi])
        rig = RigState(height=rng.uniform(0.3, 1.6), pitch=rng.uniform(-0.3, 1.2),
                       pan=rng.uniform(-1.5, 1.5), forward_offset=0.05)
        scene.teleport(pose, rig)
        truth = scene.ground_truth()                       # read back from MuJoCo
        expected = compute_ground_truth(model, pose, rig)  # analytic
        np.testing.assert_allclose(truth.base_pose, expected.base_pose, atol=1e-9)
        np.testing.assert_allclose(truth.T_field_left_optical, expected.T_field_left_optical,
                                   atol=1e-9)
        np.testing.assert_allclose(truth.T_field_right_optical, expected.T_field_right_optical,
                                   atol=1e-9)
        # Stereo extrinsics as MuJoCo sees them: parallel axes, 63 mm apart along optical +x.
        T_left_right = tf.invert_transform(truth.T_field_left_optical) @ truth.T_field_right_optical
        np.testing.assert_allclose(T_left_right, model.T_left_right(), atol=1e-9)
    assert truth.scenario_id == "test"
    assert set(truth.to_dict()) >= {"timestamp", "base_pose_xy_yaw", "ball_position",
                                    "T_field_left_optical", "scenario_id"}


def test_rig_controls(scene):
    scene.teleport([1.0, 1.0, 0.0], scene.camera_cfg.mount)
    scene.set_rig(height=1.3, pitch=math.radians(40))
    truth = scene.ground_truth()
    centre = 0.5 * (truth.T_field_left_optical[:3, 3] + truth.T_field_right_optical[:3, 3])
    assert centre[2] == pytest.approx(1.3)
    forward = truth.T_field_left_optical[:3, :3] @ [0, 0, 1]
    assert math.degrees(math.asin(-forward[2])) == pytest.approx(40.0)
    scene.rotate_in_place(math.pi / 2)
    assert scene.ground_truth().base_pose[2] == pytest.approx(math.pi / 2)
    scene.set_ball([2.0, -1.0])
    np.testing.assert_allclose(scene.ground_truth().ball_position[:2], [2.0, -1.0])
    scene.set_rig(height=scene.camera_cfg.mount.height, pitch=scene.camera_cfg.mount.pitch)


def test_scripted_trajectory_is_followed_without_any_policy(scene):
    path = trajectories.rectangle([-1.0, -1.0, 0.0], 2.0, 1.0).with_head_scan(0.8, 4.0)
    for step in (0, 17, len(path) // 2, len(path) - 1):
        scene.follow(path, step)
        truth = scene.ground_truth()
        np.testing.assert_allclose(truth.base_pose[:2], path.poses[step][:2], atol=1e-9)
        assert abs(tf.angle_difference(truth.base_pose[2], path.poses[step][2])) < 1e-9
        assert truth.timestamp == pytest.approx(step * path.dt) and truth.step == step
        assert truth.rig.pan == pytest.approx(path.pan[step])


def test_trajectories_are_repeatable_and_well_formed():
    first = trajectories.random_waypoints([0, 0, 0], (3.0, 2.0), 5, np.random.default_rng(7))
    again = trajectories.random_waypoints([0, 0, 0], (3.0, 2.0), 5, np.random.default_rng(7))
    other = trajectories.random_waypoints([0, 0, 0], (3.0, 2.0), 5, np.random.default_rng(8))
    np.testing.assert_array_equal(first.poses, again.poses)
    assert first.poses.shape != other.poses.shape or not np.allclose(first.poses, other.poses)
    assert np.all(np.abs(first.poses[:, 0]) <= 3.0 + 1e-9)

    loop = trajectories.rectangle([-1.0, -1.0, 0.0], 2.0, 1.0)
    np.testing.assert_allclose(loop.poses[-1][:2], loop.poses[0][:2], atol=1e-9)
    ring = trajectories.circle([1.5, 0.0, math.pi / 2], radius=1.5)
    np.testing.assert_allclose(np.hypot(ring.poses[:, 0], ring.poses[:, 1]), 1.5, atol=1e-9)
    np.testing.assert_allclose(ring.poses[-1][:2], ring.poses[0][:2], atol=1e-9)
    spin = trajectories.rotate_in_place([0.5, 0.5, 0.0], turns=1.0)
    assert np.ptp(spin.poses[:, 0]) == 0 and abs(tf.wrap_angle(spin.poses[-1, 2])) < 1e-9
    steps = np.linalg.norm(np.diff(loop.poses[:, :2], axis=0), axis=1)
    assert steps.max() <= 0.3 * loop.dt + 1e-9, "never faster than the commanded speed"

    jump = trajectories.PathBuilder([0, 0, 0]).forward(1.0).teleport([3, 2, 1]).hold(1.0).build("k")
    assert jump.kidnap_steps == (np.argmax(np.all(jump.poses == [3, 2, 1], axis=1)),)


# ------------------------------------------------------------------ rendering
@pytest.mark.render
def test_stereo_frames(scene, gl, tmp_path):
    scene.teleport([-2.0, -1.0, math.radians(20)], scene.camera_cfg.mount)
    scene.set_ball([1.0, 0.5])
    camera = VirtualStereoCamera(scene)
    first, second = camera.capture(), camera.capture()
    k = scene.camera_cfg.intrinsics
    for image in (first.left, first.right):
        assert image.shape == (k.height, k.width, 3) and image.dtype == np.uint8
        assert image.std() > 10, "the image is not blank"
    assert not np.array_equal(first.left, first.right), "the two views differ"
    np.testing.assert_array_equal(first.left, second.left)       # same state, same image
    assert (first.frame_index, second.frame_index) == (0, 1)
    assert second.timestamp - first.timestamp == pytest.approx(1 / scene.camera_cfg.frame_rate)
    assert first.metadata["baseline_m"] == pytest.approx(0.063)
    assert first.metadata["approximate"] is True
    np.testing.assert_allclose(first.T_field_left_optical,
                               scene.camera_model.T_field_camera(scene.base_pose, "left"),
                               atol=1e-9)
    paths = save_stereo_frame(first, tmp_path)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


@pytest.mark.render
@pytest.mark.parametrize("ball_xy", [(1.5, 0.3), (3.0, -1.2), (0.2, 1.0)])
def test_rendered_ball_matches_the_projection_model(scene, gl, ball_xy):
    """The rendered ball is where the pinhole model says, in both images, with the right disparity."""
    pose = [-1.5, 0.2, math.radians(-10)]
    scene.teleport(pose, scene.camera_cfg.mount)
    radius = scene.scene_cfg["ball"]["radius"]
    centre = np.array([*ball_xy, radius])
    measured = {}
    for name, side in ((LEFT_CAMERA, "left"), (RIGHT_CAMERA, "right")):
        pixels = changed_pixels(scene, name, ball_xy)
        assert len(pixels) > 30, "the ball is visible"
        expected, depth, visible = scene.camera_model.project_field_points(pose, centre, side)
        assert visible
        measured[side] = pixels.mean(axis=0)
        np.testing.assert_allclose(measured[side], expected, atol=1.5)
    disparity = measured["left"][0] - measured["right"][0]
    assert disparity == pytest.approx(float(scene.camera_model.disparity(depth)), abs=1.0)
    assert abs(measured["left"][1] - measured["right"][1]) < 1.0, "rectified: same row"


@pytest.mark.render
def test_asymmetric_intrinsics_are_honoured_by_the_renderer(s_field, gl):
    """fx != fy and an off-centre principal point render exactly as the model projects."""
    base = StereoCameraConfig.load("zed_mini_sim")
    odd = dataclasses.replace(base, intrinsics=CameraIntrinsics(960, 540, 520.0, 470.0, 430.0, 300.0))
    assert odd.intrinsics.cx != (960 - 1) / 2 and odd.intrinsics.cy != (540 - 1) / 2
    with MujocoScene(s_field, odd, load_scene_config(overrides=NO_SHADOWS)) as scene:
        pose = [0.0, 0.0, 0.3]
        scene.teleport(pose)
        for ball_xy in [(2.0, 0.2), (1.5, 1.4), (3.5, -0.5)]:
            pixels = changed_pixels(scene, LEFT_CAMERA, ball_xy)
            centre = [*ball_xy, scene.scene_cfg["ball"]["radius"]]
            expected, _, visible = StereoCameraModel(odd).project_field_points(pose, centre)
            assert visible and len(pixels) > 30
            np.testing.assert_allclose(pixels.mean(axis=0), expected, atol=1.5)


@pytest.mark.render
def test_rendered_lines_are_where_the_landmark_map_says(scene, gl, s_field):
    """Project known field points through T_field_camera and check the rendered colour there."""
    scene.set_ball([40.0, 40.0])
    landmarks = build_landmark_map(s_field)
    checked = 0
    for pose in ([-2.0, -1.0, 0.3], [1.0, 1.5, math.radians(-150)], [0.0, -2.0, math.pi / 2]):
        scene.teleport(pose, scene.camera_cfg.mount)
        image = scene.render(LEFT_CAMERA, show_platform=False).astype(int)
        on_paint = np.vstack([landmarks.positions(kind) for kind in
                              (LandmarkType.L_INTERSECTION, LandmarkType.T_INTERSECTION,
                               LandmarkType.X_INTERSECTION, LandmarkType.PENALTY_MARK)])
        off_paint = on_paint + [0.35, 0.3]
        off_paint = off_paint[s_field.distance_to_paint(off_paint) > 0.15]
        for points, expect_white in ((on_paint, True), (off_paint, False)):
            points3 = np.column_stack([points, np.zeros(len(points))])
            pixels, depth, visible = scene.camera_model.project_field_points(pose, points3)
            near = visible & (depth < 4.0) & scene.camera_cfg.intrinsics.in_image(pixels, 4)
            for u, v in np.round(pixels[near]).astype(int):
                r, g, b = image[v, u]
                is_white = min(r, g, b) > 150
                is_green = g > 1.5 * r and g > 1.5 * b
                assert is_white if expect_white else is_green, (pose, u, v, (r, g, b))
                checked += 1
    assert checked >= 15


@pytest.mark.render
def test_overview_render_matches_field_geometry(scene, gl, s_field):
    """Top-down render vs the analytic paint mask: same lines, same places, same widths."""
    scene.teleport([0.0, 0.0, 0.0], scene.camera_cfg.mount)
    scene.set_ball([40.0, 40.0])
    image = scene.render_overview(show_platform=False).astype(int)
    k = overview_intrinsics(s_field, scene.scene_cfg)
    assert image.shape[:2] == (k.height, k.width)
    v, u = np.mgrid[0:k.height, 0:k.width]
    xy = overview_pixel_to_field(s_field, scene.scene_cfg, np.stack([u, v], axis=-1))
    white = image.min(axis=-1) > 150
    paint = s_field.distance_to_paint(xy) <= 0.0
    inside = np.abs(xy[..., 0]) < s_field.half_length - 0.3        # goals are 3-D, leave them out
    iou = (white & paint & inside).sum() / ((white | paint) & inside).sum()
    assert iou > 0.95
    turf = image[~white & ~paint & inside]
    assert turf[:, 1].mean() > 1.5 * turf[:, 0].mean(), "the playing surface is green"
    # Where a post stands the image shows the post (seen from above, so mostly its shaded
    # side): a neutral grey-white, not turf green.
    for goal in s_field.goals:
        for post in goal.posts:
            pu = int(round(post[0] * k.fx / 40.0 + k.cx))
            pv = int(round(-post[1] * k.fy / 40.0 + k.cy))
            r, g, b = image[pv, pu]
            assert min(r, g, b) > 80 and abs(r - g) < 20 and abs(g - b) < 20, "a white post"
