"""UTRA's humanoid_prototype_1 as a kinematic, visual-only body in the perception scene.

What is taken from the official model (soccer-software, ``humanoid_description``), unchanged:

* the kinematic tree: every body, joint axis, joint range and frame offset of ``robot.xml``;
* the visual meshes of all 21 links, triangle for triangle (stored losslessly);
* the ``imu`` site and the ankle linkage's pin sites.

What is deliberately NOT taken, because perception does not need it and the model's own
documentation calls it provisional:

* collision geometry (243 convex pieces), contacts and contact exclusions;
* actuators, joint dynamics and the equality constraints that close the ankle four-bars;
* sensors (the IMU is simulated by `simulation.imu` from the body's motion instead).

The body is POSED, never simulated: its pelvis is placed at a given 6-DOF pose, its joints are
set to the stand configuration plus a cosmetic stepping motion, and only forward kinematics is
evaluated. It cannot fall, collide or balance, and the stepping is an animation, not a gait.
Each ankle's four-bar is closed geometrically (`AnkleLinkage`), so the horn and the rod stay
connected as the ankle pitches.

`import_model` builds the asset bundle in ``assets/robots/<name>/`` from a checkout of the
official model; it reads the source and never writes to it. Importing this module requires
``mujoco``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from robosoccer_perception.geometry import quaternion
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.mount import BodyGeometry, CameraMount
from robosoccer_perception.geometry.robot_description import (
    BODY_GEOMETRY_FILE, RobotDescription)

SKELETON_FILE = "skeleton.xml"
MESH_FILE = "visual_meshes.npz"
MANIFEST_FILE = "manifest.json"
PREFIX = "robot/"
ROOT_BODY = "robot_root"            # the mocap body that carries the pelvis
VISUAL_GROUP = 2                    # geom group of visual meshes in the official MJCF
COLLISION_GROUP = 3
TOP_SLAB = 0.004                    # metres: vertices this close to the top define "torso top"


# --------------------------------------------------------------------------------------------
# Import: official model -> asset bundle
# --------------------------------------------------------------------------------------------
def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (V, 3) float32, faces (F, 3) int32)`` of a binary STL, with the duplicated
    corner vertices merged. No vertex is moved: the triangles are exactly the file's."""
    raw = Path(path).read_bytes()
    count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path}: not a binary STL (size does not match its triangle count)")
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    corners = np.frombuffer(raw, dtype=record, count=count, offset=84)["corners"].reshape(-1, 3)
    vertices, inverse = np.unique(corners, axis=0, return_inverse=True)
    return vertices.astype(np.float32), inverse.reshape(-1, 3).astype(np.int32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(source: Path, *args: str) -> str | None:
    """Output of a read-only git query on the source checkout, or None outside a repository."""
    try:
        result = subprocess.run(["git", "-C", str(source), *args], capture_output=True,
                                text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def strip_to_skeleton(robot_xml: str) -> tuple[str, list[dict[str, str]]]:
    """The kinematic skeleton of an official ``robot.xml``, and its visual geoms.

    Kept: compiler angle unit, bodies, inertials, hinge joints, sites. Removed: the floating
    base joint, every geom, mesh assets, contacts, equalities, actuators, sensors. Returns the
    skeleton as XML text and one ``{"body", "geom", "mesh"}`` entry per visual geom.
    """
    root = ET.fromstring(robot_xml)
    visuals: list[dict[str, str]] = []

    def clean(body: ET.Element) -> None:
        for child in list(body):
            if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
                body.remove(child)
            elif child.tag == "geom":
                group = int(child.get("group", "0"))
                if group == VISUAL_GROUP and child.get("type") == "mesh":
                    visuals.append({"body": body.get("name"), "geom": child.get("name"),
                                    "mesh": child.get("mesh")})
                body.remove(child)
            elif child.tag == "body":
                clean(child)

    worldbody = root.find("worldbody")
    for body in worldbody.findall("body"):
        clean(body)
    skeleton = ET.Element("mujoco", {"model": root.get("model", "robot")})
    compiler = root.find("compiler")
    ET.SubElement(skeleton, "compiler", {
        "angle": compiler.get("angle", "radian") if compiler is not None else "radian",
        "inertiafromgeom": "false", "boundmass": "0", "boundinertia": "0"})
    skeleton.append(worldbody)
    ET.indent(skeleton, space="  ")
    return ET.tostring(skeleton, encoding="unicode"), visuals


def import_model(source: str | Path, destination: str | Path, name: str = "humanoid_prototype_1",
                 stance: dict[str, float] | None = None) -> dict[str, Any]:
    """Build ``destination`` (the asset bundle) from ``source``, the official model directory
    that holds ``mjcf/robot.xml`` and ``meshes/``. Returns the manifest that was written.

    The source is only read. ``stance`` (joint name -> radians) is the pose in which the body
    geometry is measured; it defaults to the ``home`` keyframe of ``mjcf/scene.xml``.
    """
    source, destination = Path(source).resolve(), Path(destination)
    robot_xml = source / "mjcf" / "robot.xml"
    if not robot_xml.is_file():
        raise FileNotFoundError(f"{robot_xml} not found: point --source at the model directory "
                                "that contains mjcf/robot.xml and meshes/")
    skeleton, visuals = strip_to_skeleton(robot_xml.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    (destination / SKELETON_FILE).write_text(skeleton + "\n")

    arrays: dict[str, np.ndarray] = {}
    mesh_records = {}
    for mesh in sorted({v["mesh"] for v in visuals}):
        path = source / "meshes" / f"{mesh}.stl"
        vertices, faces = read_binary_stl(path)
        arrays[f"{mesh}__v"], arrays[f"{mesh}__f"] = vertices, faces
        mesh_records[mesh] = {"file": path.name, "sha256": _sha256(path),
                              "triangles": int(len(faces)), "vertices": int(len(vertices))}
    np.savez_compressed(destination / MESH_FILE, **arrays)

    # ---- the stand pose the official scene defines, by joint name
    home: dict[str, float] = {}
    scene_xml = source / "mjcf" / "scene.xml"
    if scene_xml.is_file():
        official = mujoco.MjModel.from_xml_path(str(scene_xml))
        key = mujoco.mj_name2id(official, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            for joint in range(official.njnt):
                if official.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                    value = float(official.key_qpos[key][official.jnt_qposadr[joint]])
                    if value != 0.0:
                        home[official.joint(joint).name] = value
    stance = dict(home if stance is None else stance)

    manifest = {
        "name": name, "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"directory": source.name, "robot_xml_sha256": _sha256(robot_xml),
                   "git_commit": _git(source, "rev-parse", "HEAD"),
                   "git_dirty": bool(_git(source, "status", "--porcelain", "--", ".") or "")},
        "kept": ["bodies", "inertials", "hinge joints", "sites", "visual meshes"],
        "removed": ["floating-base joint", "collision geoms and meshes", "contact exclusions",
                    "equality constraints", "actuators", "sensors"],
        "visual_geoms": visuals, "meshes": mesh_records,
        "triangles": int(sum(r["triangles"] for r in mesh_records.values())),
        "home_stance": home}
    (destination / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")

    description = RobotDescription(name=name, kind="mjcf",
                                   body=BodyGeometry(name, "pelvis", 1.0, (0.0, 0.0, 0.0)),
                                   asset_dir=destination, stance=stance)
    measured = measure_body_geometry(description)
    header = (
        "# MEASURED from the imported model by scripts/import_humanoid_model.py; do not edit.\n"
        "# standing_height_m: the pelvis origin above the lowest point of the feet in the stand\n"
        "#   pose. torso_top_m: the centre of the topmost surface of the torso (pelvis link)\n"
        "#   mesh, in the pelvis frame. The robot has NO HEAD yet: this is the top of the torso.\n")
    (destination / BODY_GEOMETRY_FILE).write_text(header + yaml.safe_dump(
        {"body_geometry": measured.to_dict(), "stance": stance}, sort_keys=False))
    manifest["body_geometry"] = measured.to_dict()
    return manifest


# --------------------------------------------------------------------------------------------
# Loading the bundle
# --------------------------------------------------------------------------------------------
def load_manifest(description: RobotDescription) -> dict[str, Any]:
    return json.loads((description.asset_dir / MANIFEST_FILE).read_text())


def build_robot_spec(description: RobotDescription, group: int = VISUAL_GROUP) -> mujoco.MjSpec:
    """The robot as a stand-alone `MjSpec`: skeleton plus visual meshes, the pelvis at the
    origin of the world. Nothing in it collides."""
    if description.asset_dir is None:
        raise ValueError(f"robot '{description.name}' has no imported model")
    spec = mujoco.MjSpec.from_string((description.asset_dir / SKELETON_FILE).read_text())
    manifest = load_manifest(description)
    appearance = description.appearance
    rgba = [float(v) for v in appearance.get("rgba", (0.2, 0.21, 0.23, 1.0))]
    foot_rgba = [float(v) for v in appearance.get("foot_rgba", rgba)]
    material = spec.add_material(name="printed", specular=0.15, shininess=0.2, reflectance=0.0)
    del material
    with np.load(description.asset_dir / MESH_FILE) as meshes:
        for mesh in sorted({v["mesh"] for v in manifest["visual_geoms"]}):
            spec.add_mesh(name=mesh, uservert=meshes[f"{mesh}__v"].astype(np.float32).ravel(),
                          userface=meshes[f"{mesh}__f"].astype(np.int32).ravel())
    for visual in manifest["visual_geoms"]:
        colour = foot_rgba if "foot" in visual["body"] else rgba
        spec.body(visual["body"]).add_geom(
            name=visual["geom"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=visual["mesh"],
            rgba=colour, material="printed", group=group, contype=0, conaffinity=0)
    spec.body("pelvis").pos = [0.0, 0.0, 0.0]
    return spec


def _mesh_points(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """World coordinates of the vertices of a mesh geom."""
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T \
        + data.geom_xpos[geom]


def measure_body_geometry(description: RobotDescription) -> BodyGeometry:
    """Standing height and torso top of the imported model, from its meshes and kinematics."""
    model = build_robot_spec(description).compile()
    data = mujoco.MjData(model)
    for joint, value in description.stance.items():
        data.qpos[model.joint(joint).qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    mesh_geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    lowest = min(float(_mesh_points(model, data, g)[:, 2].min()) for g in mesh_geoms)
    torso = _mesh_points(model, data, model.geom("pelvis_visual").id)
    top = float(torso[:, 2].max())
    slab = torso[torso[:, 2] >= top - TOP_SLAB]
    centre = 0.5 * (slab.min(axis=0) + slab.max(axis=0))
    site = model.site("imu")
    quat = np.array(model.site_quat[site.id], dtype=float)
    roll, pitch, yaw = quaternion.rpy_from_matrix(tf.matrix_from_quat_wxyz(quat))
    return BodyGeometry(
        model=description.name, body_frame="pelvis", standing_height=round(-lowest, 6),
        torso_top=(round(float(centre[0]), 6), round(float(centre[1]), 6), round(top, 6)),
        imu_xyz=tuple(round(float(v), 6) for v in model.site_pos[site.id]),
        imu_rpy_deg=tuple(round(math.degrees(v), 6) + 0.0 for v in (roll, pitch, yaw)),
        note="Measured from the imported meshes in the stand pose. The IMU is the model's "
             "placeholder site at the pelvis origin; the robot has no head.")


# --------------------------------------------------------------------------------------------
# Posing
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class BodyPose:
    """How the robot's body is placed in one frame. SIMULATOR SIDE (the true body pose)."""
    T_footprint_body: np.ndarray                    # 4x4: pelvis in the footprint frame
    gait_phase: float = 0.0                         # radians, 2 pi per step
    activity: float = 0.0                           # 0 standing ... 1 walking

    @property
    def roll_pitch(self) -> tuple[float, float]:
        roll, pitch, _ = quaternion.rpy_from_matrix(self.T_footprint_body[:3, :3])
        return roll, pitch


@dataclasses.dataclass(frozen=True)
class AnkleLinkage:
    """The ankle's four-bar, closed geometrically: for an ankle pitch, the horn (motor) and
    rod angles at which the rod's pin meets the horn's pin. Tabulated once per leg."""
    ankle: np.ndarray           # (K,) ankle pitch, ascending
    horn: np.ndarray            # (K,) motor joint angle
    rod: np.ndarray             # (K,) rod joint angle
    residual: float             # metres: worst pin distance left over the table

    def solve(self, ankle_pitch: float) -> tuple[float, float]:
        return (float(np.interp(ankle_pitch, self.ankle, self.horn)),
                float(np.interp(ankle_pitch, self.ankle, self.rod)))


def _tabulate_linkage(model: mujoco.MjModel, side: str, prefix: str, samples: int = 41
                      ) -> AnkleLinkage:
    """Solve the pin closure of one ankle over the ankle's range by Gauss-Newton on the two
    passive angles (kinematics only; no dynamics, no constraint solver)."""
    data = mujoco.MjData(model)
    address = {name: model.joint(f"{prefix}{side}_{name}").qposadr[0]
               for name in ("ankle_pitch_joint", "ankle_motor_joint", "ankle_rod_joint")}
    coupler = model.site(f"{prefix}{side}_ankle_pitch_linkage_coupler_pin").id
    pin = model.site(f"{prefix}{side}_ankle_pitch_linkage_input_pin").id
    low, high = model.jnt_range[model.joint(f"{prefix}{side}_ankle_pitch_joint").id]

    def gap(ankle: float, passive: np.ndarray) -> np.ndarray:
        data.qpos[address["ankle_pitch_joint"]] = ankle
        data.qpos[address["ankle_motor_joint"]] = passive[0]
        data.qpos[address["ankle_rod_joint"]] = passive[1]
        mujoco.mj_kinematics(model, data)
        delta = data.site_xpos[coupler] - data.site_xpos[pin]
        return np.array([delta[0], delta[2]])       # the linkage moves in the x-z plane

    angles = np.linspace(low, high, samples)
    order = np.argsort(np.abs(angles))              # start at zero and continue outwards
    solution = np.zeros((samples, 2))
    worst, solved = 0.0, {}
    for index in order:
        neighbours = [i for i in solved if abs(i - index) == 1]
        passive = solved[neighbours[0]].copy() if neighbours else np.zeros(2)
        for _ in range(30):
            residual = gap(angles[index], passive)
            if np.linalg.norm(residual) < 1e-9:
                break
            jacobian = np.column_stack([(gap(angles[index], passive + step) - residual) / 1e-6
                                        for step in (np.array([1e-6, 0.0]), np.array([0.0, 1e-6]))])
            passive = passive - np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        worst = max(worst, float(np.linalg.norm(gap(angles[index], passive))))
        solved[index] = passive
        solution[index] = passive
    return AnkleLinkage(angles, solution[:, 0], solution[:, 1], worst)


class HumanoidBody:
    """The imported humanoid inside a compiled scene: posing and cosmetic stepping."""

    SIDES = ("left", "right")

    def __init__(self, model: mujoco.MjModel, description: RobotDescription,
                 prefix: str = PREFIX, root: str = ROOT_BODY):
        self.model, self.description, self.prefix = model, description, prefix
        self.mocap_id = int(model.body_mocapid[model.body(root).id])
        self._address = {
            model.joint(j).name[len(prefix):]: int(model.jnt_qposadr[j])
            for j in range(model.njnt) if model.joint(j).name.startswith(prefix)}
        self._range = {
            model.joint(j).name[len(prefix):]: tuple(model.jnt_range[j])
            for j in range(model.njnt)
            if model.joint(j).name.startswith(prefix) and model.jnt_limited[j]}
        body_ids = {b for b in range(model.nbody) if model.body(b).name.startswith(prefix)}
        self.geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, list(body_ids)))
        self.linkage = {side: _tabulate_linkage(model, side, prefix) for side in self.SIDES}

        # Leg geometry for the cosmetic stepping, read from the model's own frame offsets.
        def offset(child: str) -> np.ndarray:
            return np.array(model.body(prefix + child).pos, dtype=float)

        self._thigh = float(-(offset("left_hip_roll_link") + offset("left_hip_yaw_link")
                              + offset("left_knee_link"))[2])
        self._shin = float(-offset("left_foot_link")[2])
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        foot = model.geom(prefix + "left_foot_link_visual").id
        self._sole = float(data.xpos[model.body(prefix + "left_foot_link").id][2]
                           - _mesh_points(model, data, foot)[:, 2].min())
        stance = description.stance
        hip, knee = stance.get("left_hip_pitch_joint", 0.0), stance.get("left_knee_joint", 0.0)
        self._stance_x = -self._thigh * math.sin(hip) - self._shin * math.sin(hip + knee)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._address)

    def stance_joints(self) -> dict[str, float]:
        return dict(self.description.stance)

    # ------------------------------------------------------------------ cosmetic stepping
    def _leg(self, forward: float, down: float) -> tuple[float, float]:
        """Hip pitch and knee angle that put the ankle ``forward`` of and ``down`` below the
        hip-pitch axis (planar two-link inverse kinematics; the knee bends forward)."""
        reach = self._thigh + self._shin
        distance = min(math.hypot(forward, down), 0.999 * reach)
        cosine = (distance ** 2 - self._thigh ** 2 - self._shin ** 2) \
            / (2.0 * self._thigh * self._shin)
        knee = math.acos(max(-1.0, min(1.0, cosine)))
        hip = math.atan2(-forward, down) - math.atan2(self._shin * math.sin(knee),
                                                      self._thigh + self._shin * math.cos(knee))
        return hip, knee

    def joints_for(self, pose: BodyPose) -> dict[str, float]:
        """Joint angles for a body pose: the stand configuration with the feet kept under the
        body, and a stepping motion in proportion to ``pose.activity``. COSMETIC."""
        gait = self.description.gait
        joints = self.stance_joints()
        roll, pitch = pose.roll_pitch
        activity = float(np.clip(pose.activity, 0.0, 1.0))
        stride = 0.5 * pose.gait_phase
        hip_height = float(pose.T_footprint_body[2, 3]) - self._sole
        for side, shift in (("left", 0.0), ("right", math.pi)):
            phase = stride + shift
            forward = self._stance_x + activity * 0.5 * gait.get("step_length_m", 0.0) \
                * math.cos(phase)
            lift = activity * gait.get("step_height_m", 0.0) * max(0.0, -math.sin(phase))
            hip, knee = self._leg(forward, hip_height - lift)
            hip -= pitch                             # the pelvis pitched: keep the leg plumb
            joints[f"{side}_hip_pitch_joint"] = hip
            joints[f"{side}_knee_joint"] = knee
            joints[f"{side}_ankle_pitch_joint"] = -(hip + knee) - pitch     # sole flat
            joints[f"{side}_hip_roll_joint"] = -roll
            swing = activity * gait.get("arm_swing_rad", 0.0) * math.cos(phase)
            joints[f"{side}_shoulder_pitch_joint"] = swing
        return joints

    # ------------------------------------------------------------------ placement
    def apply(self, data: mujoco.MjData, T_field_body: np.ndarray,
              joints: dict[str, float] | None = None) -> None:
        """Write the body's pose and joint angles into ``data`` (kinematics is the caller's)."""
        data.mocap_pos[self.mocap_id] = T_field_body[:3, 3]
        data.mocap_quat[self.mocap_id] = tf.quat_wxyz_from_matrix(T_field_body[:3, :3])
        joints = self.stance_joints() if joints is None else dict(joints)
        for name, address in self._address.items():
            value = joints.get(name, 0.0)
            if name in self._range:
                value = float(np.clip(value, *self._range[name]))
                joints[name] = value
            data.qpos[address] = value
        for side in self.SIDES:
            horn, rod = self.linkage[side].solve(joints.get(f"{side}_ankle_pitch_joint", 0.0))
            data.qpos[self._address[f"{side}_ankle_motor_joint"]] = horn
            data.qpos[self._address[f"{side}_ankle_rod_joint"]] = rod

    def park(self, data: mujoco.MjData, position) -> None:
        data.mocap_pos[self.mocap_id] = position


# --------------------------------------------------------------------------------------------
# Adding the robot to a scene description
# --------------------------------------------------------------------------------------------
def attach_to_scene(spec: mujoco.MjSpec, description: RobotDescription, body_group: int,
                    prefix: str = PREFIX, root: str = ROOT_BODY):
    """Add the robot to a scene `MjSpec`, hanging from a mocap body named ``root``. Returns
    the body that represents the robot's body frame (the pelvis)."""
    robot = build_robot_spec(description, group=body_group)
    carrier = spec.worldbody.add_body(name=root, mocap=True)
    carrier.add_frame().attach_body(robot.body("pelvis"), prefix, "")
    return spec.body(prefix + "pelvis")


def add_mount_markers(parent, body: BodyGeometry, mount: CameraMount, group: int) -> None:
    """Visual aids fixed to the body frame ``parent``: the frame's axes, the mount offset as a
    line from the body origin to the camera, a mast for the virtual head, and the viewing
    direction.

    MARKERS ONLY. They have no mass and do not collide; the mast is not a head design. They
    live in their own geom group so that the robot's camera never sees them.
    """
    camera = mount.position_in_body(body)
    T_body_rig = mount.T_body_rig(body)

    def stick(name: str, start, end, radius: float, rgba) -> None:
        start, end = np.asarray(start, float), np.asarray(end, float)
        if np.linalg.norm(end - start) < 1e-6:
            return
        parent.add_geom(name=f"marker/{name}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[radius, 0, 0], fromto=[*start, *end], rgba=rgba, group=group,
                        contype=0, conaffinity=0)

    for axis, rgba in ((0, [0.9, 0.15, 0.15, 1.0]), (1, [0.15, 0.75, 0.2, 1.0]),
                       (2, [0.2, 0.35, 0.95, 1.0])):
        tip = np.zeros(3)
        tip[axis] = 0.35                    # long enough to stick out of the torso
        stick(f"body_axis_{'xyz'[axis]}", np.zeros(3), tip, 0.006, rgba)
    stick("mount_offset", np.zeros(3), camera, 0.003, [1.0, 0.85, 0.1, 1.0])
    if mount.head_extension:
        top = np.asarray(body.torso_top, dtype=float)
        stick("head_extension", top, [camera[0] - 0.02, camera[1], camera[2] - 0.02], 0.012,
              [0.75, 0.75, 0.78, 0.55])
    stick("view_direction", camera, camera + 0.30 * T_body_rig[:3, 0], 0.004,
          [1.0, 0.35, 0.85, 1.0])
