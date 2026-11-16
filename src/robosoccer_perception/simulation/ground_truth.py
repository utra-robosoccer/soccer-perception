"""Simulator ground truth: the true state of the world at one instant.

Ground truth is for the oracle observation generator, the evaluation code and the plots. The
localizer never receives one of these objects; it only sees `OdometryDelta` and
`ObservationFrame` (tests/test_observations.py enforces that those types carry no global pose).

`compute_ground_truth` derives the frame analytically from the commanded state.
`MujocoScene.ground_truth` reads the same quantities back from MuJoCo; the two are tested
against each other, which is what ties the geometric model to the rendered scene.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import RigState, StereoCameraModel
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import LandmarkMap


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    scenario_id: str
    step: int
    timestamp: float                        # simulation time, seconds
    base_pose: np.ndarray                   # [x, y, yaw]: T_field_base as a 3-DOF pose
    rig: RigState
    T_field_base: np.ndarray                # 4x4
    T_field_left_optical: np.ndarray        # 4x4
    T_field_right_optical: np.ndarray       # 4x4
    ball_position: np.ndarray               # (3,) ball centre in the field frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id, "step": self.step, "timestamp": self.timestamp,
            "base_pose_xy_yaw": self.base_pose.tolist(),
            "rig": dataclasses.asdict(self.rig),
            "T_field_base": self.T_field_base.tolist(),
            "T_field_left_optical": self.T_field_left_optical.tolist(),
            "T_field_right_optical": self.T_field_right_optical.tolist(),
            "ball_position": self.ball_position.tolist(),
        }


def compute_ground_truth(camera: StereoCameraModel, base_pose, rig: RigState | None = None,
                         ball_position=(0.0, 0.0, 0.0), timestamp: float = 0.0, step: int = 0,
                         scenario_id: str = "unnamed") -> GroundTruth:
    """Ground truth of a kinematic world: everything follows from the commanded state."""
    rig = rig or camera.config.mount
    base_pose = np.array(base_pose, dtype=float)
    base_pose[2] = tf.wrap_angle(base_pose[2])
    return GroundTruth(
        scenario_id=scenario_id, step=int(step), timestamp=float(timestamp),
        base_pose=base_pose, rig=rig,
        T_field_base=tf.T_field_base_from_pose(base_pose),
        T_field_left_optical=camera.T_field_camera(base_pose, "left", rig),
        T_field_right_optical=camera.T_field_camera(base_pose, "right", rig),
        ball_position=np.array(ball_position, dtype=float))


def static_ground_truth(field: FieldGeometry, landmarks: LandmarkMap) -> dict[str, Any]:
    """The parts of the truth that never change during a run: the field and what stands on it."""
    return {
        "field": field.dims.to_dict(),
        "field_hash": field.content_hash(),
        "goal_posts": [{"goal": goal.name, "position": list(post), "height": goal.dims.height,
                        "diameter": goal.dims.post_diameter}
                       for goal in field.goals for post in goal.posts],
        "landmarks": landmarks.to_dict(),
        "line_segments": [dataclasses.asdict(s) for s in field.segments],
        "line_arcs": [dataclasses.asdict(a) for a in field.arcs],
    }
