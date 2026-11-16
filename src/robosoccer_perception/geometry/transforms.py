"""Coordinate frames and rigid transforms. See docs/coordinate_frames.md.

Naming convention
-----------------
``T_a_b`` is the pose of frame ``b`` expressed in frame ``a``. It maps coordinates from ``b``
to ``a``::

    p_a = T_a_b @ p_b            (homogeneous 4x4)
    T_a_c = T_a_b @ T_b_c        (adjacent frame names cancel)
    T_b_a = inverse(T_a_b)

Rotations follow the same rule: ``R_a_b`` has the axes of ``b``, written in ``a``, as columns.

Frames (all right-handed)
-------------------------
field           origin at the centre of the playing field, +x along the field length toward the
                reference goal, +y toward the reference touchline, +z up. Fixed for the match.
base            the robot's footprint frame: on the ground under the robot, +x forward, +y left,
                +z up (ROS REP-103). A 3-DOF pose (x, y, yaw) is ``T_field_base``.
rig             the stereo camera centre (midpoint of the two optical centres), body-style axes:
                +x along the viewing direction, +y left, +z up.
*_optical       ROS optical frame of one camera: +z forward (viewing direction), +x right,
                +y down.
mujoco camera   MuJoCo's own camera frame: +x right, +y up, looking along -z.

2-D poses are NumPy arrays ``[x, y, yaw]`` (metres, radians). Every SE(2) function broadcasts
over leading dimensions so it can be applied to a whole particle set at once.
"""
from __future__ import annotations

import numpy as np

# Rotation of an optical frame relative to a body-style frame that looks along the same
# direction. Columns are the optical axes written in the body frame:
#   optical +x (right)   = body -y
#   optical +y (down)    = body -z
#   optical +z (forward) = body +x
R_BODY_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])

# Rotation of MuJoCo's camera frame relative to the ROS optical frame of the same camera.
# MuJoCo cameras look along -z with +y up; optical frames look along +z with +y down. Both have
# +x to the right, so the two differ by a half turn about x.
R_OPTICAL_MUJOCO = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------------------------
# Angles
# --------------------------------------------------------------------------------------------
def wrap_angle(angle):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def angle_difference(a, b):
    """Smallest signed rotation that takes heading ``b`` to heading ``a``, in [-pi, pi)."""
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def circular_mean(angles, weights=None) -> float:
    """Weighted mean direction of a set of angles."""
    angles = np.asarray(angles, dtype=float)
    weights = np.ones_like(angles) if weights is None else np.asarray(weights, dtype=float)
    return float(np.arctan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles))))


# --------------------------------------------------------------------------------------------
# SE(2): poses [x, y, yaw]
# --------------------------------------------------------------------------------------------
def se2_compose(pose_a_b, pose_b_c) -> np.ndarray:
    """``T_a_c = T_a_b * T_b_c`` for 2-D poses. Broadcasts over leading dimensions."""
    a = np.asarray(pose_a_b, dtype=float)
    b = np.asarray(pose_b_c, dtype=float)
    cos, sin = np.cos(a[..., 2]), np.sin(a[..., 2])
    x = a[..., 0] + cos * b[..., 0] - sin * b[..., 1]
    y = a[..., 1] + sin * b[..., 0] + cos * b[..., 1]
    yaw = wrap_angle(a[..., 2] + b[..., 2])
    return np.stack(np.broadcast_arrays(x, y, yaw), axis=-1)


def se2_inverse(pose_a_b) -> np.ndarray:
    """``T_b_a`` from ``T_a_b``."""
    a = np.asarray(pose_a_b, dtype=float)
    cos, sin = np.cos(a[..., 2]), np.sin(a[..., 2])
    x = -(cos * a[..., 0] + sin * a[..., 1])
    y = -(-sin * a[..., 0] + cos * a[..., 1])
    return np.stack([x, y, wrap_angle(-a[..., 2])], axis=-1)


def se2_relative(pose_w_a, pose_w_b) -> np.ndarray:
    """``T_a_b`` from two poses in a common frame: where ``b`` is, as seen from ``a``."""
    return se2_compose(se2_inverse(pose_w_a), pose_w_b)


def se2_transform_points(pose_a_b, points_b) -> np.ndarray:
    """Map 2-D points from frame ``b`` to frame ``a``.

    ``pose_a_b`` has shape (..., 3) and ``points_b`` shape (M, 2); the result has shape
    (..., M, 2). With N particle poses this gives every point as seen by every particle.
    """
    pose = np.asarray(pose_a_b, dtype=float)
    points = np.asarray(points_b, dtype=float)
    cos = np.cos(pose[..., 2])[..., None]
    sin = np.sin(pose[..., 2])[..., None]
    x = pose[..., 0][..., None] + cos * points[..., 0] - sin * points[..., 1]
    y = pose[..., 1][..., None] + sin * points[..., 0] + cos * points[..., 1]
    return np.stack([x, y], axis=-1)


def se2_inverse_transform_points(pose_a_b, points_a) -> np.ndarray:
    """Map 2-D points from frame ``a`` to frame ``b`` (the inverse of `se2_transform_points`)."""
    pose = np.asarray(pose_a_b, dtype=float)
    points = np.asarray(points_a, dtype=float)
    cos = np.cos(pose[..., 2])[..., None]
    sin = np.sin(pose[..., 2])[..., None]
    dx = points[..., 0] - pose[..., 0][..., None]
    dy = points[..., 1] - pose[..., 1][..., None]
    return np.stack([cos * dx + sin * dy, -sin * dx + cos * dy], axis=-1)


# --------------------------------------------------------------------------------------------
# SE(3): homogeneous 4x4 matrices
# --------------------------------------------------------------------------------------------
def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(rotation=None, translation=None) -> np.ndarray:
    """Homogeneous transform from a 3x3 rotation and a translation (both optional)."""
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def invert_transform(transform_a_b: np.ndarray) -> np.ndarray:
    """``T_b_a`` from ``T_a_b``, using the rigid-transform closed form (no general inverse)."""
    rotation = transform_a_b[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform_a_b[:3, 3]
    return inverse


def transform_points(transform_a_b: np.ndarray, points_b) -> np.ndarray:
    """Map 3-D points of shape (..., 3) from frame ``b`` to frame ``a``."""
    points = np.asarray(points_b, dtype=float)
    return points @ transform_a_b[:3, :3].T + transform_a_b[:3, 3]


def T_field_base_from_pose(pose) -> np.ndarray:
    """4x4 ``T_field_base`` of a 3-DOF pose ``[x, y, yaw]`` (the base frame is on the ground)."""
    x, y, yaw = (float(v) for v in pose)
    return make_transform(rot_z(yaw), [x, y, 0.0])


def pose_from_T_field_base(transform_field_base: np.ndarray) -> np.ndarray:
    """3-DOF pose ``[x, y, yaw]`` of a 4x4 ``T_field_base``; yaw is the heading of base +x."""
    yaw = np.arctan2(transform_field_base[1, 0], transform_field_base[0, 0])
    return np.array([transform_field_base[0, 3], transform_field_base[1, 3], yaw])


# --------------------------------------------------------------------------------------------
# Quaternions (w, x, y, z), the order MuJoCo uses
# --------------------------------------------------------------------------------------------
def quat_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix, with w >= 0."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        quat = np.empty(4)
        quat[0] = (m[k, j] - m[j, k]) / s
        quat[1 + i] = 0.25 * s
        quat[1 + j] = (m[j, i] + m[i, j]) / s
        quat[1 + k] = (m[k, i] + m[i, k]) / s
    quat /= np.linalg.norm(quat)
    return -quat if quat[0] < 0.0 else quat


def matrix_from_quat_wxyz(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
