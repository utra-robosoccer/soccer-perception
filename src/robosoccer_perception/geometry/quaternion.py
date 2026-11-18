"""Unit quaternions (w, x, y, z) and the roll / pitch / yaw convention. Pure NumPy.

Convention
----------
A quaternion ``q_a_b`` is the orientation of frame ``b`` in frame ``a``: it rotates coordinates
from ``b`` to ``a``, like ``R_a_b`` in `geometry.transforms`. Products chain the same way,
``q_a_c = multiply(q_a_b, q_b_c)``.

Roll, pitch and yaw are the intrinsic Z-Y-X angles of a body-style frame (+x forward, +y left,
+z up): ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``. With these axes a POSITIVE PITCH TIPS THE NOSE
DOWN (the camera looks further down) and a positive roll tips the left side up. The same
convention is used by `geometry.camera.RigState`.

Roll and pitch together are the TILT: they are fully determined by where "up" points in the
body frame, which is what an accelerometer at rest measures. Yaw is not, and nothing in this
module pretends that an inertial sensor can observe it.
"""
from __future__ import annotations

import math

import numpy as np

from robosoccer_perception.geometry import transforms as tf

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def normalize(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    return quat / np.linalg.norm(quat)


def multiply(a, b) -> np.ndarray:
    """Hamilton product ``a * b``."""
    aw, ax, ay, az = np.asarray(a, dtype=float)
    bw, bx, by, bz = np.asarray(b, dtype=float)
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def conjugate(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def from_rotation_vector(vector) -> np.ndarray:
    """Quaternion of the rotation ``exp(vector)``: angle ``|vector|`` about ``vector``."""
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-9:                    # second-order series: exact enough and never divides by 0
        return normalize(np.array([1.0 - angle * angle / 8.0, *(0.5 * vector)]))
    return np.array([math.cos(0.5 * angle), *(math.sin(0.5 * angle) / angle * vector)])


def to_rotation_vector(quat) -> np.ndarray:
    """The rotation vector (axis times angle, angle in [0, pi]) of a unit quaternion."""
    quat = np.asarray(quat, dtype=float)
    if quat[0] < 0.0:
        quat = -quat
    sine = float(np.linalg.norm(quat[1:]))
    if sine < 1e-9:
        return 2.0 * quat[1:]
    return 2.0 * math.atan2(sine, quat[0]) / sine * quat[1:]


def from_matrix(rotation) -> np.ndarray:
    return tf.quat_wxyz_from_matrix(np.asarray(rotation, dtype=float))


def to_matrix(quat) -> np.ndarray:
    return tf.matrix_from_quat_wxyz(quat)


def rotate(quat, vector) -> np.ndarray:
    """Coordinates in frame ``a`` of a vector given in frame ``b``, for ``quat = q_a_b``."""
    return to_matrix(quat) @ np.asarray(vector, dtype=float)


def slerp(q0, q1, fraction: float) -> np.ndarray:
    """Spherical interpolation from ``q0`` (fraction 0) to ``q1`` (fraction 1) along the shorter
    arc. ``fraction`` outside [0, 1] extrapolates at the same angular rate."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    step = to_rotation_vector(multiply(conjugate(q0), q1))
    return normalize(multiply(q0, from_rotation_vector(fraction * step)))


def angle_between(q0, q1) -> float:
    """Angle in radians of the rotation that takes ``q0`` to ``q1``."""
    return float(np.linalg.norm(to_rotation_vector(multiply(conjugate(q0), q1))))


# ------------------------------------------------------------------ roll / pitch / yaw
def matrix_from_rpy(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    return tf.rot_z(yaw) @ tf.rot_y(pitch) @ tf.rot_x(roll)


def rpy_from_matrix(rotation) -> tuple[float, float, float]:
    """``(roll, pitch, yaw)`` of ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``; pitch in (-pi/2, pi/2)."""
    m = np.asarray(rotation, dtype=float)
    pitch = -math.asin(float(np.clip(m[2, 0], -1.0, 1.0)))
    return float(math.atan2(m[2, 1], m[2, 2])), pitch, float(math.atan2(m[1, 0], m[0, 0]))


def tilt_from_up(up_in_body) -> tuple[float, float]:
    """``(roll, pitch)`` from the world's up direction written in the body frame.

    For ``R_world_body = Rz(yaw) Ry(pitch) Rx(roll)`` the third ROW of the matrix is
    ``[-sin(pitch), sin(roll) cos(pitch), cos(roll) cos(pitch)]``, whatever the yaw is.
    """
    x, y, z = np.asarray(up_in_body, dtype=float) / np.linalg.norm(up_in_body)
    return float(math.atan2(y, z)), -math.asin(float(np.clip(x, -1.0, 1.0)))


def tilt_from_quat(quat_world_body) -> tuple[float, float]:
    """``(roll, pitch)`` of a body orientation, independent of its yaw."""
    return tilt_from_up(to_matrix(quat_world_body)[2, :])


def up_from_tilt(roll: float, pitch: float) -> np.ndarray:
    """The inverse of `tilt_from_up`: the world's up direction in the body frame."""
    return np.array([-math.sin(pitch), math.sin(roll) * math.cos(pitch),
                     math.cos(roll) * math.cos(pitch)])


def tilt_angle(roll: float, pitch: float) -> float:
    """Total angle between the body's +z axis and the vertical, radians."""
    return float(math.acos(np.clip(math.cos(roll) * math.cos(pitch), -1.0, 1.0)))
