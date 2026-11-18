"""The configurable camera mount: where the stereo rig sits on the robot's body. Pure NumPy.

THE HEAD OF THE ROBOT HAS NOT BEEN DESIGNED. Nothing in this module is a mechanical
specification. A `CameraMount` is a rigid transform from a robot body frame to the rig, read
from ``configs/mounts/*.yaml``; every preset that ships with the repository is an EXPERIMENTAL
ASSUMPTION and says so. When the real head exists, its measured transform goes into one more
YAML file and nothing else changes.

Frame chain (see docs/humanoid_integration.md)::

    field <- footprint <- body <- rig <- {left, right}_optical

    footprint   on the ground under the body origin, +x along the body's heading, +z up. Its
                3-DOF pose (x, y, yaw) is what the localizer estimates. (`geometry.transforms`
                calls this frame "base".)
    body        the robot's own reference frame: the PELVIS of humanoid_prototype_1 (midway
                between the hip-pitch axes, +x forward, +y left, +z up). It moves with the
                torso: ``T_footprint_body = Trans(0, 0, height) Ry(pitch) Rx(roll)``.
    rig         midpoint of the two optical centres, +x along the viewing direction.

Four different things that are easy to confuse, and that this module keeps apart:

1. BODY ORIENTATION      roll / pitch of the body in the footprint frame. Changes all the time.
                         An IMU on the body can estimate it.
2. MOUNT ORIENTATION     the fixed rotation of the rig in the body frame (`CameraMount.rpy`).
                         A property of the mechanics; known only as well as it was calibrated.
3. CAMERA ORIENTATION    the product of the two. This is what ground projection needs.
4. MOUNT CALIBRATION ERROR  the difference between the true mount and the robot's belief of
                         it. An IMU cannot see it: the IMU measures the body, not the camera.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.geometry import quaternion as quat
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.camera import RigState

REFERENCES = ("body", "torso_top", "ground")
PARENT_FRAME = "pelvis"


# --------------------------------------------------------------------------------------------
# What is known about the robot's body
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class BodyGeometry:
    """The few numbers of the robot's body that camera geometry needs.

    For humanoid_prototype_1 they are MEASURED from the model's meshes and kinematics
    (`simulation.humanoid.measure_body_geometry`) and stored next to the imported model; they
    are not typed in by hand. All lengths in metres.
    """
    model: str
    body_frame: str                         # name of the body frame in the model ("pelvis")
    standing_height: float                  # body origin above the ground in the nominal stance
    torso_top: tuple[float, float, float]   # top-centre of the torso, in the body frame
    imu_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)       # IMU in the body frame
    imu_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    placeholder: bool = False               # True for the capsule platform of Phases 1 to 3
    note: str = ""

    def __post_init__(self) -> None:
        if self.standing_height <= 0.0:
            raise ValueError("standing_height must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyGeometry":
        return cls(model=str(data["model"]), body_frame=str(data.get("body_frame", PARENT_FRAME)),
                   standing_height=float(data["standing_height_m"]),
                   torso_top=tuple(float(v) for v in data["torso_top_m"]),
                   imu_xyz=tuple(float(v) for v in data.get("imu_xyz_m", (0.0, 0.0, 0.0))),
                   imu_rpy_deg=tuple(float(v) for v in data.get("imu_rpy_deg", (0.0, 0.0, 0.0))),
                   placeholder=bool(data.get("placeholder", False)),
                   note=str(data.get("note", "")))

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "body_frame": self.body_frame,
                "standing_height_m": self.standing_height, "torso_top_m": list(self.torso_top),
                "imu_xyz_m": list(self.imu_xyz), "imu_rpy_deg": list(self.imu_rpy_deg),
                "placeholder": self.placeholder, "note": self.note}

    @property
    def torso_top_height(self) -> float:
        """Top of the torso above the ground in the nominal stance."""
        return self.standing_height + self.torso_top[2]

    @property
    def T_body_imu(self) -> np.ndarray:
        roll, pitch, yaw = (math.radians(v) for v in self.imu_rpy_deg)
        return tf.make_transform(quat.matrix_from_rpy(roll, pitch, yaw), self.imu_xyz)


def T_footprint_body(height: float, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Pose of the body frame in the footprint frame: ``Trans(0, 0, height) Ry(pitch) Rx(roll)``.

    There is no yaw and no horizontal offset by definition: the footprint is the point on the
    ground under the body origin, turned to the body's heading.
    """
    return tf.make_transform(quat.matrix_from_rpy(roll, pitch), [0.0, 0.0, height])


# --------------------------------------------------------------------------------------------
# The mount
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CameraMount:
    """A rigid camera mount: the rig frame in the robot's body frame.

    ``offset`` is measured from ``reference``:

    * ``body``       the origin of the body frame;
    * ``torso_top``  the top-centre of the torso mesh (so "8 cm above the torso" stays 8 cm if
                     the torso is redesigned);
    * ``ground``     the point on the ground under the body origin in the nominal stance (for
                     "a camera 0.90 m above the floor", the placeholder rig of Phases 1 to 3).

    ``rpy`` is the rig's orientation in the body frame, radians, ``Rz(yaw) Ry(pitch) Rx(roll)``:
    positive pitch looks down, positive yaw looks left, positive roll tips the right side of
    the image down.
    """
    name: str
    offset: tuple[float, float, float]              # metres: forward, left, up from the reference
    rpy: tuple[float, float, float]                 # radians: roll, pitch, yaw
    reference: str = "torso_top"
    parent_frame: str = PARENT_FRAME
    experimental: bool = True                       # True = an assumption, not a measurement
    head_extension: bool = True                     # draw a mast from the torso to the camera
    note: str = ""

    def __post_init__(self) -> None:
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got '{self.reference}'")
        if len(self.offset) != 3 or len(self.rpy) != 3:
            raise ValueError("offset and rpy need three values each")
        if not all(math.isfinite(v) for v in (*self.offset, *self.rpy)):
            raise ValueError("the mount offset and angles must be finite numbers; a mount "
                             "without a height is not a mount (derive it from the body "
                             "geometry with reference: torso_top)")

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraMount":
        offset = data["xyz_m"]
        if any(v is None for v in offset):
            raise ValueError(f"mount '{data.get('name')}': xyz_m has an unset value. The camera "
                             "height is never assumed: set the offset from a reference that the "
                             "body geometry defines (reference: torso_top)")
        return cls(name=str(data["name"]), offset=tuple(float(v) for v in offset),
                   rpy=tuple(math.radians(float(v)) for v in data.get("rpy_deg", (0, 0, 0))),
                   reference=str(data.get("reference", "torso_top")),
                   parent_frame=str(data.get("parent_frame", PARENT_FRAME)),
                   experimental=bool(data.get("experimental", True)),
                   head_extension=bool(data.get("head_extension", True)),
                   note=str(data.get("note", "")))

    @classmethod
    def load(cls, name_or_path: str | os.PathLike) -> "CameraMount":
        return cls.from_dict(config.load_config("mounts", name_or_path))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parent_frame": self.parent_frame,
                "reference": self.reference, "xyz_m": list(self.offset),
                "rpy_deg": [math.degrees(v) for v in self.rpy],
                "experimental": self.experimental, "head_extension": self.head_extension,
                "note": self.note}

    # ------------------------------------------------------------------ geometry
    def position_in_body(self, body: BodyGeometry) -> np.ndarray:
        """The rig centre in the body frame, metres."""
        offset = np.asarray(self.offset, dtype=float)
        if self.reference == "body":
            return offset
        if self.reference == "torso_top":
            return np.asarray(body.torso_top, dtype=float) + offset
        return offset - np.array([0.0, 0.0, body.standing_height])

    def T_body_rig(self, body: BodyGeometry) -> np.ndarray:
        """The camera-to-body transform (rig frame in the body frame)."""
        return tf.make_transform(quat.matrix_from_rpy(*self.rpy), self.position_in_body(body))

    def standing_height(self, body: BodyGeometry) -> float:
        """Height of the rig centre above the ground with the body upright in its stance."""
        return float(body.standing_height + self.position_in_body(body)[2])

    def shifted(self, *, forward: float = 0.0, left: float = 0.0, up: float = 0.0,
                roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                name: str | None = None) -> "CameraMount":
        """This mount moved by the given amounts (metres along the body axes, radians added
        to roll / pitch / yaw). Used for "what if the head were 5 cm taller" and for the hidden
        difference between the real mount and the calibrated one."""
        return dataclasses.replace(
            self, name=name or self.name,
            offset=(self.offset[0] + forward, self.offset[1] + left, self.offset[2] + up),
            rpy=(self.rpy[0] + roll, self.rpy[1] + pitch, self.rpy[2] + yaw))

    def describe(self, body: BodyGeometry) -> str:
        """One line for a diagnostics panel."""
        x, y, z = self.position_in_body(body)
        roll, pitch, yaw = (math.degrees(v) for v in self.rpy)
        flag = "EXPERIMENTAL " if self.experimental else ""
        return (f"{flag}{self.name}: {body.body_frame} + [{x:+.3f} {y:+.3f} {z:+.3f}] m, "
                f"rpy [{roll:+.1f} {pitch:+.1f} {yaw:+.1f}] deg")


def available_mounts() -> list[str]:
    return sorted(p.stem for p in (config.config_dir() / "mounts").glob("*.yaml"))


# --------------------------------------------------------------------------------------------
# The chain down to the optical frame
# --------------------------------------------------------------------------------------------
def T_rig_optical(baseline: float, side: str = "left") -> np.ndarray:
    """Optical frame of one camera in the rig frame (the same rule as `StereoCameraModel`)."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    offset = 0.5 * baseline * (1.0 if side == "left" else -1.0)
    return tf.make_transform(tf.R_BODY_OPTICAL, [0.0, offset, 0.0])


def T_footprint_rig(mount: CameraMount, body: BodyGeometry, height: float, roll: float = 0.0,
                    pitch: float = 0.0) -> np.ndarray:
    """Rig frame in the footprint frame for a body at ``height`` tilted by ``roll``, ``pitch``."""
    return T_footprint_body(height, roll, pitch) @ mount.T_body_rig(body)


def rig_state_from_transform(transform_footprint_rig: np.ndarray) -> RigState:
    """The `RigState` (height, pitch, pan, roll, offsets) that reproduces a rig pose.

    `RigState` composes ``Trans(forward, lateral, height) Rz(pan) Ry(pitch) Rx(roll)``, which
    is a general rigid transform, so a rig on a tilted, swaying body can be handed to the
    Phase 1 to 3 scene and oracle unchanged.
    """
    transform = np.asarray(transform_footprint_rig, dtype=float)
    roll, pitch, pan = quat.rpy_from_matrix(transform[:3, :3])
    return RigState(height=float(transform[2, 3]), pitch=pitch, pan=pan,
                    forward_offset=float(transform[0, 3]), lateral_offset=float(transform[1, 3]),
                    roll=roll)


def transform_from_rig_state(rig: RigState) -> np.ndarray:
    """``T_footprint_rig`` of a `RigState`; the inverse of `rig_state_from_transform`."""
    rotation = quat.matrix_from_rpy(rig.roll, rig.pitch, rig.pan)
    return tf.make_transform(rotation, [rig.forward_offset, rig.lateral_offset, rig.height])
