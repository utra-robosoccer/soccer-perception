"""Coordinate conventions: SE(2), SE(3), the optical frame and the stereo extrinsics."""
import math

import numpy as np
import pytest

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import RigState

HALF_PI = math.pi / 2


def test_wrap_angle_and_difference():
    assert tf.wrap_angle(0.0) == 0.0
    assert tf.wrap_angle(math.pi) == pytest.approx(-math.pi)
    assert tf.wrap_angle(3 * math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert tf.wrap_angle(-3 * math.pi - 0.1) == pytest.approx(math.pi - 0.1)
    # 350 deg and 10 deg are 20 deg apart, not 340.
    assert tf.angle_difference(math.radians(10), math.radians(350)) == pytest.approx(
        math.radians(20))
    assert tf.angle_difference(math.radians(350), math.radians(10)) == pytest.approx(
        math.radians(-20))
    wrapped = tf.wrap_angle(np.linspace(-20, 20, 1001))
    assert np.all((wrapped >= -math.pi) & (wrapped < math.pi))


def test_circular_mean_across_the_wrap():
    mean = tf.circular_mean([math.radians(179), math.radians(-179)])
    assert abs(tf.angle_difference(mean, math.pi)) < 1e-9


def test_se2_90_degree_rotation():
    # A robot at (1, 2) facing +y: a point 1 m ahead of it is at (1, 3) in the field frame,
    # a point 1 m to its left is at (0, 2).
    pose = np.array([1.0, 2.0, HALF_PI])
    np.testing.assert_allclose(tf.se2_transform_points(pose, [[1.0, 0.0], [0.0, 1.0]]),
                               [[1.0, 3.0], [0.0, 2.0]], atol=1e-12)


def test_se2_180_degree_rotation():
    pose = np.array([1.0, 2.0, math.pi])
    np.testing.assert_allclose(tf.se2_transform_points(pose, [[1.0, 0.5]]), [[0.0, 1.5]],
                               atol=1e-12)
    np.testing.assert_allclose(tf.se2_compose(pose, pose), [0.0, 0.0, 0.0], atol=1e-12)


def test_se2_inverse_and_round_trip(rng):
    poses = rng.uniform([-5, -5, -math.pi], [5, 5, math.pi], size=(200, 3))
    identity = tf.se2_compose(poses, tf.se2_inverse(poses))
    np.testing.assert_allclose(identity[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(tf.wrap_angle(identity[:, 2]), 0.0, atol=1e-12)
    points = rng.uniform(-3, 3, size=(17, 2))
    in_field = tf.se2_transform_points(poses, points)
    assert in_field.shape == (200, 17, 2)
    for i in (0, 57, 199):
        np.testing.assert_allclose(tf.se2_inverse_transform_points(poses[i], in_field[i]),
                                   points, atol=1e-12)


def test_se2_relative_recovers_the_step(rng):
    start = rng.uniform([-5, -5, -math.pi], [5, 5, math.pi], size=(100, 3))
    step = rng.uniform([-1, -1, -1], [1, 1, 1], size=(100, 3))
    end = tf.se2_compose(start, step)
    np.testing.assert_allclose(tf.se2_relative(start, end), step, atol=1e-12)


def test_se2_agrees_with_se3(rng):
    for pose in rng.uniform([-5, -5, -math.pi], [5, 5, math.pi], size=(20, 3)):
        T_field_base = tf.T_field_base_from_pose(pose)
        np.testing.assert_allclose(tf.pose_from_T_field_base(T_field_base), pose, atol=1e-12)
        points = rng.uniform(-2, 2, size=(5, 2))
        expected = tf.transform_points(T_field_base, np.column_stack([points, np.zeros(5)]))
        np.testing.assert_allclose(tf.se2_transform_points(pose, points), expected[:, :2],
                                   atol=1e-12)


@pytest.mark.parametrize("rotation", [tf.rot_x, tf.rot_y, tf.rot_z])
def test_elementary_rotations_are_right_handed(rotation):
    quarter = rotation(HALF_PI)
    assert np.linalg.det(quarter) == pytest.approx(1.0)
    np.testing.assert_allclose(quarter @ quarter, rotation(math.pi), atol=1e-12)
    np.testing.assert_allclose(quarter.T @ quarter, np.eye(3), atol=1e-12)


def test_rotation_directions():
    # Right-hand rule: +z turns x into y, +x turns y into z, +y turns z into x.
    np.testing.assert_allclose(tf.rot_z(HALF_PI) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(tf.rot_x(HALF_PI) @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(tf.rot_y(HALF_PI) @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    # So a positive pitch (about +y) tips the forward axis DOWN.
    assert (tf.rot_y(math.radians(20)) @ [1, 0, 0])[2] < 0


def test_invert_transform_and_composition(rng):
    T_a_b = tf.make_transform(tf.rot_z(0.7) @ tf.rot_y(-0.3) @ tf.rot_x(1.1), [1.0, -2.0, 0.5])
    T_b_c = tf.make_transform(tf.rot_x(0.2) @ tf.rot_z(2.9), [0.3, 0.1, -4.0])
    np.testing.assert_allclose(T_a_b @ tf.invert_transform(T_a_b), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(tf.invert_transform(T_a_b), np.linalg.inv(T_a_b), atol=1e-12)
    points_c = rng.normal(size=(10, 3))
    via_b = tf.transform_points(T_a_b, tf.transform_points(T_b_c, points_c))
    np.testing.assert_allclose(tf.transform_points(T_a_b @ T_b_c, points_c), via_b, atol=1e-12)
    back = tf.transform_points(tf.invert_transform(T_a_b @ T_b_c), via_b)
    np.testing.assert_allclose(back, points_c, atol=1e-12)


def test_optical_frame_axes():
    R = tf.R_BODY_OPTICAL
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0])     # optical forward = body forward
    np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0])    # optical right   = body right
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1])    # optical down    = body down


def test_mujoco_camera_frame_differs_from_the_optical_frame():
    R = tf.R_OPTICAL_MUJOCO
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert not np.allclose(R, np.eye(3))
    np.testing.assert_allclose(R @ [0, 0, -1], [0, 0, 1])    # MuJoCo looks along -z
    np.testing.assert_allclose(R @ [0, 1, 0], [0, -1, 0])    # MuJoCo +y is up, optical +y down
    np.testing.assert_allclose(R @ [1, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(R, tf.rot_x(math.pi), atol=1e-12)


def test_quaternion_round_trip(rng):
    for _ in range(50):
        angles = rng.uniform(-math.pi, math.pi, size=3)
        R = tf.rot_z(angles[0]) @ tf.rot_y(angles[1]) @ tf.rot_x(angles[2])
        quat = tf.quat_wxyz_from_matrix(R)
        assert quat[0] >= 0 and np.linalg.norm(quat) == pytest.approx(1.0)
        np.testing.assert_allclose(tf.matrix_from_quat_wxyz(quat), R, atol=1e-9)
    np.testing.assert_allclose(tf.quat_wxyz_from_matrix(np.eye(3)), [1, 0, 0, 0])
    np.testing.assert_allclose(tf.quat_wxyz_from_matrix(tf.rot_z(math.pi)), [0, 0, 0, 1],
                               atol=1e-12)


# ------------------------------------------------------------------ camera extrinsics
def test_level_camera_looks_along_base_x(camera_model):
    rig = RigState(height=1.0, pitch=0.0)
    T_base_cam = camera_model.T_base_camera("left", rig)
    np.testing.assert_allclose(T_base_cam[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(T_base_cam[:3, 3], [0, camera_model.baseline / 2, 1.0],
                               atol=1e-12)


def test_pitch_and_pan_directions(camera_model):
    down = camera_model.T_base_camera("left", RigState(height=1.0, pitch=math.radians(30)))
    forward = down[:3, :3] @ [0, 0, 1]
    np.testing.assert_allclose(forward, [math.cos(math.radians(30)), 0,
                                         -math.sin(math.radians(30))], atol=1e-12)
    left = camera_model.T_base_camera("left", RigState(height=1.0, pitch=0.0, pan=HALF_PI))
    np.testing.assert_allclose(left[:3, :3] @ [0, 0, 1], [0, 1, 0], atol=1e-12)


def test_stereo_relative_geometry(camera_model):
    T_left_right = camera_model.T_left_right()
    np.testing.assert_allclose(T_left_right[:3, :3], np.eye(3), atol=1e-12)   # parallel axes
    np.testing.assert_allclose(T_left_right[:3, 3], [camera_model.baseline, 0, 0], atol=1e-12)
    assert camera_model.baseline == pytest.approx(0.063)
    # Independent of where the rig points, and measured between the optical centres.
    rig = RigState(height=0.7, pitch=0.4, pan=-1.2, forward_offset=0.1, lateral_offset=0.02)
    left = camera_model.T_field_camera([1.0, -2.0, 2.5], "left", rig)
    right = camera_model.T_field_camera([1.0, -2.0, 2.5], "right", rig)
    np.testing.assert_allclose(tf.invert_transform(left) @ right, T_left_right, atol=1e-12)
    assert np.linalg.norm(left[:3, 3] - right[:3, 3]) == pytest.approx(0.063)
    midpoint = 0.5 * (left[:3, 3] + right[:3, 3])
    assert midpoint[2] == pytest.approx(0.7), "mount height is the rig centre height"
    # The left camera really is on the robot's left: base y of left > base y of right.
    y_left = camera_model.T_base_camera("left", rig.replace(pan=0.0))[1, 3]
    y_right = camera_model.T_base_camera("right", rig.replace(pan=0.0))[1, 3]
    assert y_left > y_right


def test_camera_to_field_projection(camera_model):
    # Robot at (1, 2) facing +y, camera 1 m up, level. A point 3 m ahead on the optical axis
    # is at field (1, 5, 1); a ground point 3 m ahead and 1 m to the robot's right is (2, 5, 0).
    rig = RigState(height=1.0, pitch=0.0)
    pose = [1.0, 2.0, HALF_PI]
    k = camera_model.intrinsics
    left_x = -camera_model.baseline / 2      # the left camera is 31.5 mm left of the rig centre
    T_field_cam = camera_model.T_field_camera(pose, "left", rig)
    np.testing.assert_allclose(tf.transform_points(T_field_cam, [0.0, 0.0, 3.0]),
                               [1.0 + left_x, 5.0, 1.0], atol=1e-12)
    pixels, depth, visible = camera_model.project_field_points(
        pose, [[1.0 + left_x, 5.0, 1.0], [2.0 + left_x, 5.0, 0.0]], "left", rig)
    np.testing.assert_allclose(depth, [3.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(pixels[0], [k.cx, k.cy], atol=1e-9)
    # 1 m to the right at 3 m depth -> +fx/3 pixels in u; 1 m below the axis -> +fy/3 in v.
    np.testing.assert_allclose(pixels[1], [k.cx + k.fx / 3, k.cy + k.fy / 3], atol=1e-9)
    assert visible.all()


def test_points_behind_the_camera_are_not_visible(camera_model):
    rig = RigState(height=1.0, pitch=0.0)
    pixels, depth, visible = camera_model.project_field_points(
        [0, 0, 0], [[-3.0, 0.0, 1.0], [3.0, 0.0, 1.0]], "left", rig)
    assert depth[0] < 0 and not visible[0] and np.isnan(pixels[0]).all()
    assert visible[1]


def test_projection_round_trip_through_rays(camera_model, rng):
    k = camera_model.intrinsics
    pixels = rng.uniform([0, 0], [k.width - 1, k.height - 1], size=(100, 2))
    depth = rng.uniform(0.5, 10, size=100)
    points = k.ray(pixels) * depth[:, None]
    back, z = k.project(points)
    np.testing.assert_allclose(back, pixels, atol=1e-9)
    np.testing.assert_allclose(z, depth)
