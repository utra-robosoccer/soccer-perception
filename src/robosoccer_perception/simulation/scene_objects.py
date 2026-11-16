"""Optional scene content for vision datasets: humanoid placeholders, ball looks, backdrop.

`mujoco_scene.build_spec` calls into this module only for the configuration keys that Phase 1
never set (``robots``, ``background``, a non-default ball ``style``), so the Phase 1 scene is
unchanged when those keys are absent.

The humanoid is a PLACEHOLDER: a box torso, a head, and capsule limbs with six hinge joints
(shoulders, hips, knees). It has roughly the silhouette and proportions of a small humanoid and
nothing else: no meshes, no hands, no cables, no team markings beyond flat colours. A detector
trained on it learns "upright limbed thing on the turf", not what a real robot looks like.

Importing this module requires the ``mujoco`` package (the ``sim`` extra).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import mujoco
import numpy as np

from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry

HUMANOID_JOINTS = ("shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r")
FALLEN_POSES = ("front", "back", "side")
PARKED_POSITION = (0.0, 0.0, -50.0)     # below the floor: invisible from every camera

DEFAULT_ROBOT_APPEARANCE: dict[str, Any] = {
    "height": 0.80,                     # metres, standing
    "build": 1.0,                       # width multiplier
    "head_shape": "sphere",             # sphere | box
    "torso_rgb": [0.16, 0.16, 0.18],
    "limb_rgb": [0.55, 0.55, 0.58],
    "head_rgb": [0.20, 0.20, 0.22],
    "foot_rgb": [0.10, 0.10, 0.10],
    "marker_rgb": [0.85, 0.10, 0.60],   # team-colour band on the torso
}


@dataclasses.dataclass(frozen=True)
class RobotState:
    """Where one placeholder robot stands in a frame. Angles in radians."""
    xy: tuple[float, float]
    yaw: float = 0.0
    joints: tuple[float, ...] = (0.0,) * len(HUMANOID_JOINTS)   # order of HUMANOID_JOINTS
    fallen: str | None = None           # None (upright) or one of FALLEN_POSES

    def to_dict(self) -> dict[str, Any]:
        return {"xy": [float(v) for v in self.xy], "yaw": float(self.yaw),
                "joints": [float(v) for v in self.joints], "fallen": self.fallen}


def _matte(body, name: str, geom_type, rgb, **kwargs) -> None:
    body.add_geom(name=name, type=geom_type, rgba=[*rgb, 1.0], material="matte",
                  contype=0, conaffinity=0, **kwargs)


def add_humanoid(world, name: str, appearance: dict[str, Any] | None = None) -> None:
    """Add a kinematic humanoid placeholder called ``name`` (a mocap body), parked out of sight.

    The body origin is on the ground under the pelvis, +x forward, +z up.
    """
    look = {**DEFAULT_ROBOT_APPEARANCE, **(appearance or {})}
    height, build = float(look["height"]), float(look["build"])
    hip_z, shoulder_z = 0.50 * height, 0.78 * height
    foot_height = 0.035 * height
    leg_radius, arm_radius = 0.042 * height * build, 0.032 * height * build
    segment = 0.5 * (hip_z - foot_height)               # thigh and shank length
    hip_y, shoulder_y = 0.07 * height * build, 0.165 * height * build
    capsule, box, sphere = (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX,
                            mujoco.mjtGeom.mjGEOM_SPHERE)
    hinge = mujoco.mjtJoint.mjJNT_HINGE

    root = world.add_body(name=name, mocap=True, pos=list(PARKED_POSITION))
    torso_half = [0.075 * height * build, 0.125 * height * build,
                  0.5 * (shoulder_z - hip_z) + 0.02 * height]
    torso_z = 0.5 * (hip_z + shoulder_z) + 0.01 * height
    _matte(root, f"{name}/torso", box, look["torso_rgb"], pos=[0, 0, torso_z], size=torso_half)
    _matte(root, f"{name}/marker", box, look["marker_rgb"],
           pos=[0, 0, torso_z + 0.3 * torso_half[2]],
           size=[torso_half[0] + 0.003, torso_half[1] + 0.003, 0.18 * torso_half[2]])
    head_z, head_radius = 0.905 * height, 0.075 * height
    if look["head_shape"] == "box":
        _matte(root, f"{name}/head", box, look["head_rgb"], pos=[0, 0, head_z],
               size=[head_radius, 1.1 * head_radius, 0.9 * head_radius])
    else:
        _matte(root, f"{name}/head", sphere, look["head_rgb"], pos=[0, 0, head_z],
               size=[head_radius, 0, 0])
    for side, sign in (("l", 1.0), ("r", -1.0)):
        arm = root.add_body(name=f"{name}/arm_{side}", pos=[0, sign * shoulder_y, shoulder_z])
        arm.add_joint(name=f"{name}/shoulder_{side}", type=hinge, axis=[0, 1, 0])
        _matte(arm, f"{name}/arm_{side}", capsule, look["limb_rgb"], size=[arm_radius, 0, 0],
               fromto=[0, 0, 0, 0, 0, -0.30 * height])
        thigh = root.add_body(name=f"{name}/thigh_{side}", pos=[0, sign * hip_y, hip_z])
        thigh.add_joint(name=f"{name}/hip_{side}", type=hinge, axis=[0, 1, 0])
        _matte(thigh, f"{name}/thigh_{side}", capsule, look["limb_rgb"], size=[leg_radius, 0, 0],
               fromto=[0, 0, 0, 0, 0, -segment])
        shank = thigh.add_body(name=f"{name}/shank_{side}", pos=[0, 0, -segment])
        shank.add_joint(name=f"{name}/knee_{side}", type=hinge, axis=[0, 1, 0])
        _matte(shank, f"{name}/shank_{side}", capsule, look["limb_rgb"], size=[leg_radius, 0, 0],
               fromto=[0, 0, 0, 0, 0, -segment])
        _matte(shank, f"{name}/foot_{side}", box, look["foot_rgb"],
               pos=[0.035 * height, 0, -segment - 0.5 * foot_height],
               size=[0.085 * height, 0.045 * height * build, 0.5 * foot_height])


def fallen_rotation(yaw: float, fallen: str | None) -> np.ndarray:
    """Body orientation of an upright (``None``) or fallen robot heading ``yaw``."""
    rotation = tf.rot_z(yaw)
    if fallen is None:
        return rotation
    if fallen not in FALLEN_POSES:
        raise ValueError(f"fallen must be None or one of {FALLEN_POSES}, got {fallen!r}")
    tip = {"front": tf.rot_y(0.5 * math.pi), "back": tf.rot_y(-0.5 * math.pi),
           "side": tf.rot_x(0.5 * math.pi)}[fallen]
    return rotation @ tip


def lowest_point_z(model, data, geom_ids) -> float:
    """Height of the lowest point of a set of box / sphere / capsule / cylinder geoms."""
    lowest = math.inf
    for gid in geom_ids:
        z = float(data.geom_xpos[gid][2])
        up = np.abs(np.asarray(data.geom_xmat[gid]).reshape(3, 3)[2])   # world z of local axes
        size, kind = model.geom_size[gid], int(model.geom_type[gid])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            reach = float(up @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            reach = float(size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            reach = float(size[0] + up[2] * size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            reach = float(up[2] * size[1] + math.hypot(up[0], up[1]) * size[0])
        else:
            raise ValueError(f"unsupported geom type {kind}")
        lowest = min(lowest, z - reach)
    return lowest


# ------------------------------------------------------------------ ball
def _spots_cube_texture(size: int, rgb1, rgb2, count: int, seed: int) -> np.ndarray:
    """Six stacked cube faces (6*size, size, 3 uint8): ``rgb1`` with round ``rgb2`` patches."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:size, 0:size]
    faces = []
    for _ in range(6):
        face = np.ones((size, size, 3)) * np.asarray(rgb1, float)
        for _ in range(count):
            cu, cv = rng.uniform(0, size, 2)
            radius = rng.uniform(0.12, 0.24) * size
            face[np.hypot(u - cu, v - cv) < radius] = rgb2
        faces.append(face)
    return (np.clip(np.concatenate(faces, axis=0), 0.0, 1.0) * 255).astype(np.uint8)


def add_ball_material(spec, ball_cfg: dict[str, Any]) -> None:
    """Texture and material named ``ball``.

    ``style``: ``checker`` (the Phase 1 look and the default), ``spots`` (light ball with dark
    round patches, loosely like a classic football) or ``flat`` (one colour).
    """
    style = ball_cfg.get("style", "checker")
    rgb1 = list(ball_cfg.get("rgb1", [0.95, 0.95, 0.95]))
    rgb2 = list(ball_cfg.get("rgb2", [0.08, 0.08, 0.08]))
    cube = mujoco.mjtTexture.mjTEXTURE_CUBE
    if style == "checker":
        texture = spec.add_texture(name="ball", type=cube,
                                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                                   width=64, height=64)
        texture.rgb1, texture.rgb2 = rgb1, rgb2
    elif style == "flat":
        texture = spec.add_texture(name="ball", type=cube,
                                   builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, width=16, height=16)
        texture.rgb1, texture.rgb2 = rgb1, rgb1
    elif style == "spots":
        size = 64
        image = _spots_cube_texture(size, rgb1, rgb2, int(ball_cfg.get("spot_count", 3)),
                                    int(ball_cfg.get("texture_seed", 0)))
        texture = spec.add_texture(name="ball", type=cube, width=size, height=6 * size,
                                   nchannel=3)
        texture.data = image.tobytes()
    else:
        raise ValueError(f"unknown ball style {style!r} (checker, spots, flat)")
    material = spec.add_material(name="ball", reflectance=0.0, specular=0.1, shininess=0.1)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ball"


# ------------------------------------------------------------------ backdrop
def add_background(world, field: FieldGeometry, background: dict[str, Any] | None) -> None:
    """Walls around the carpet and loose boxes on the floor outside it.

    Everything here is unlabelled clutter: it renders into the image and is ``background`` in
    the semantic mask. Its purpose is to stop a model from learning that anything that is not
    turf is a flat grey floor.

    ``walls``: list of ``{side: pos_x|neg_x|pos_y|neg_y, distance, height, rgb}``, where
    ``distance`` is measured outward from the carpet edge. ``clutter``: list of
    ``{position: [x, y], size: [sx, sy, sz] (full extents), yaw, rgb}``.
    """
    if not background:
        return
    half_x, half_y = field.half_extent
    box = mujoco.mjtGeom.mjGEOM_BOX
    for index, wall in enumerate(background.get("walls", [])):
        distance, height = float(wall["distance"]), float(wall["height"])
        side, thickness = wall["side"], 0.05
        if side in ("pos_x", "neg_x"):
            sign = 1.0 if side == "pos_x" else -1.0
            position = [sign * (half_x + distance), 0.0, 0.5 * height]
            size = [thickness, half_y + distance + 12.0, 0.5 * height]
        elif side in ("pos_y", "neg_y"):
            sign = 1.0 if side == "pos_y" else -1.0
            position = [0.0, sign * (half_y + distance), 0.5 * height]
            size = [half_x + distance + 12.0, thickness, 0.5 * height]
        else:
            raise ValueError(f"unknown wall side {side!r}")
        _matte(world, f"bg/wall_{index}_{side}", box, wall["rgb"], pos=position, size=size)
    for index, item in enumerate(background.get("clutter", [])):
        sx, sy, sz = (float(v) for v in item["size"])
        x, y = (float(v) for v in item["position"])
        _matte(world, f"bg/clutter_{index}", box, item["rgb"], pos=[x, y, 0.5 * sz],
               size=[0.5 * sx, 0.5 * sy, 0.5 * sz],
               quat=tf.quat_wxyz_from_matrix(tf.rot_z(float(item.get("yaw", 0.0)))))
