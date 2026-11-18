"""The MuJoCo soccer scene: field, goals, ball, lighting and a kinematic camera platform.

The scene is generated from a `FieldGeometry`; it contains no field dimensions of its own.

The camera platform is purely kinematic. The robot base, the stereo rig and the ball are MuJoCo
mocap bodies that are placed directly, so nothing has to balance or walk and no policy is
involved. Only kinematics are evaluated (``mj_forward``); physics is never stepped. The base
body carries a simple placeholder shape; a humanoid model can replace it later without touching
the cameras, because the rig is positioned from ``T_field_base @ T_base_rig`` on every update.

For vision datasets the scene configuration may also list other robots (kinematic humanoid
placeholders, ``robots``), a backdrop (``background``) and a ball look; see
`simulation/scene_objects.py`. `MujocoScene.render_segmentation` returns the geom id seen at
every pixel, which is what dataset labels are derived from (`datasets/annotation.py`).

Phase 4 can put a ROBOT BODY in place of the placeholder shape: `MujocoScene(robot=...,
mount=...)` attaches UTRA's humanoid_prototype_1 (`simulation/humanoid.py`) or a tilting
version of the capsule, adds markers for the body frame and the camera mount, and a third-
person ``chase`` camera. The body is posed with `set_body`; the stereo rig is still placed
from a `RigState`, so the cameras, the oracle and the labels work exactly as before. Without
those two arguments the scene is the Phase 1 to 3 scene, unchanged.

Importing this module requires the ``mujoco`` package (the ``sim`` extra).
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")      # must be set before mujoco creates a context

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from robosoccer_perception import config  # noqa: E402
from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.camera import (  # noqa: E402
    CameraIntrinsics, RigState, StereoCameraConfig, StereoCameraModel)
from robosoccer_perception.geometry.field import FieldGeometry, sample_arc  # noqa: E402
from robosoccer_perception.simulation import scene_objects  # noqa: E402
from robosoccer_perception.simulation.ground_truth import GroundTruth  # noqa: E402

PLATFORM_GROUP = 1          # geom group of the placeholder robot; hidden in its own cameras
NET_GROUP = 2               # goal nets: drawn in colour images, ignored by segmentation renders
MARKER_GROUP = 4            # body-frame and mount markers: drawn only when asked for
ISOLATION_GROUP = 5         # scratch group used to render one object on its own
PIXEL_PITCH = 3.0e-6        # metres; only sets the scale of MuJoCo's metric sensor description
LEFT_CAMERA, RIGHT_CAMERA, OVERVIEW_CAMERA = "zed_left", "zed_right", "overview"
CHASE_CAMERA = "chase"      # third-person view of the robot (Phase 4 scenes only)
CHASE_SIZE = (480, 270)     # width, height in pixels
CHASE_OFFSET = (-0.95, -1.25, 0.90)     # camera position relative to the robot, field axes
CHASE_TARGET = (0.20, 0.0, 0.42)        # the point it looks at, relative to the robot
BODY_ROOT = "robot_root"    # mocap body that carries the robot's body frame


def load_scene_config(name_or_path: str = "default",
                      overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return config.deep_update(config.load_config("scenes", name_or_path), overrides)


def overview_intrinsics(field: FieldGeometry, scene_cfg: dict[str, Any]) -> CameraIntrinsics:
    """Intrinsics of the top-down overview camera: the whole carpet, at a whole-pixel scale."""
    half_x, half_y = field.half_extent
    height_m = float(scene_cfg["overview"]["height_m"])
    px_per_m = math.floor(float(scene_cfg["overview"]["max_width_px"]) / (2.0 * half_x))
    width, height = int(round(2 * half_x * px_per_m)), int(round(2 * half_y * px_per_m))
    focal = height_m * px_per_m          # one metre on the ground spans px_per_m pixels
    return CameraIntrinsics(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2)


def overview_pixel_to_field(field: FieldGeometry, scene_cfg: dict[str, Any], pixels) -> np.ndarray:
    """Ground-plane field coordinates (x, y) seen at overview pixel coordinates (u, v)."""
    k = overview_intrinsics(field, scene_cfg)
    height_m = float(scene_cfg["overview"]["height_m"])
    pixels = np.asarray(pixels, dtype=float)
    # The overview camera looks straight down with image right = field +x, image up = field +y.
    return np.stack([(pixels[..., 0] - k.cx) * height_m / k.fx,
                     -(pixels[..., 1] - k.cy) * height_m / k.fy], axis=-1)


def _turf_texture(turf: dict[str, Any]) -> tuple[np.ndarray, float]:
    """A square, tileable turf image (H, W, 3 uint8) and the side length in metres it covers."""
    stripe = float(turf["stripe_width"])
    tile_m = 2.0 * stripe if stripe > 0 else 2.0
    size = int(round(tile_m * float(turf["texture_px_per_m"])))
    rng = np.random.default_rng(int(turf["seed"]))
    base, alt = np.asarray(turf["base_rgb"], float), np.asarray(turf["stripe_rgb"], float)
    columns = (np.arange(size) * 2 // size) % 2 if stripe > 0 else np.zeros(size, int)
    image = np.where(columns[None, :, None] == 0, base, alt) * np.ones((size, size, 1))
    grain = rng.normal(0.0, float(turf["noise"]), size=(size, size))
    grain = 0.5 * grain + 0.25 * (np.roll(grain, 1, 0) + np.roll(grain, 1, 1))   # soften, tileable
    image = image * (1.0 + grain[..., None])
    blotch = float(turf.get("blotch", 0.0))
    if blotch > 0.0:
        # Large soft patches (wear, uneven pile): a few low spatial frequencies, tileable.
        v, u = np.mgrid[0:size, 0:size] * (2.0 * np.pi / size)
        patches = np.zeros((size, size))
        for _ in range(4):
            fu, fv = rng.integers(1, 4, 2)
            patches += np.sin(fu * u + rng.uniform(0, 2 * np.pi)) \
                * np.sin(fv * v + rng.uniform(0, 2 * np.pi))
        image = image * (1.0 + blotch * patches[..., None] / 2.0)
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), tile_m


def build_spec(field: FieldGeometry, camera_cfg: StereoCameraConfig,
               scene_cfg: dict[str, Any], robot=None, mount=None) -> mujoco.MjSpec:
    """Generate the MuJoCo model description of the whole scene.

    ``robot`` (a `RobotDescription`) and ``mount`` (a `CameraMount`) switch the Phase 4
    additions on: a robot body that can tilt, mount markers and the chase camera.
    """
    spec = mujoco.MjSpec()
    spec.modelname = f"robosoccer_{field.name}"
    world = spec.worldbody
    half_x, half_y = field.half_extent
    extent = 2.0 * math.hypot(half_x, half_y)

    # ---- global visual settings
    lighting = scene_cfg["lighting"]
    k = camera_cfg.intrinsics
    k_overview = overview_intrinsics(field, scene_cfg)
    # ``offscreen_scale`` > 1 leaves room for supersampled segmentation renders (dataset labels).
    offscreen_scale = int(scene_cfg.get("offscreen_scale", 1))
    chase = CHASE_SIZE if robot is not None else (0, 0)
    spec.visual.global_.offwidth = max(k.width * offscreen_scale, k_overview.width, chase[0])
    spec.visual.global_.offheight = max(k.height * offscreen_scale, k_overview.height, chase[1])
    spec.visual.headlight.ambient = [float(lighting["ambient"])] * 3
    spec.visual.headlight.diffuse = [0.0] * 3
    spec.visual.headlight.specular = [0.0] * 3
    spec.visual.quality.shadowsize = int(lighting["shadow_map_size"])
    spec.visual.quality.offsamples = 4
    spec.stat.extent = extent
    spec.stat.center = [0.0, 0.0, 0.5]
    spec.visual.map.znear = camera_cfg.clip_near / extent
    spec.visual.map.zfar = max(camera_cfg.clip_far, 2.0 * scene_cfg["overview"]["height_m"]) / extent

    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=1536)
    sky.rgb1 = scene_cfg["sky_rgb"]
    sky.rgb2 = [min(1.0, c + 0.2) for c in scene_cfg["sky_rgb"]]

    # ---- carpet (field + border strip) and the floor around it
    image, tile_m = _turf_texture(scene_cfg["turf"])
    turf_tex = spec.add_texture(name="turf", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                width=image.shape[1], height=image.shape[0], nchannel=3)
    turf_tex.data = image.tobytes()
    turf_mat = spec.add_material(name="turf", texuniform=True, reflectance=0.0, specular=0.0,
                                 shininess=0.0)
    turf_mat.texrepeat = [1.0 / tile_m, 1.0 / tile_m]
    turf_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "turf"
    matte = spec.add_material(name="matte", reflectance=0.0, specular=0.0, shininess=0.0)
    del matte
    world.add_geom(name="surround", type=mujoco.mjtGeom.mjGEOM_PLANE, material="matte",
                   size=[half_x + 15.0, half_y + 15.0, 0.1], pos=[0, 0, -0.01],
                   rgba=[*scene_cfg["surround_rgb"], 1.0], contype=0, conaffinity=0)
    world.add_geom(name="turf", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[half_x, half_y, 0.1],
                   material="turf", contype=0, conaffinity=0)

    # ---- painted lines: thin boxes lying on the turf
    line_rgba = [*scene_cfg["line_rgb"], 1.0]
    half_thickness = 0.5 * float(scene_cfg["line_thickness"])

    def add_line_box(name: str, start, end, width: float) -> None:
        start, end = np.asarray(start, float), np.asarray(end, float)
        centre, delta = 0.5 * (start + end), end - start
        yaw = math.atan2(delta[1], delta[0])
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, material="matte",
                       size=[0.5 * np.linalg.norm(delta), 0.5 * width, half_thickness],
                       pos=[centre[0], centre[1], half_thickness],
                       quat=tf.quat_wxyz_from_matrix(tf.rot_z(yaw)), rgba=line_rgba,
                       contype=0, conaffinity=0)

    for segment in field.segments:
        add_line_box(f"line/{segment.name}", segment.start, segment.end, segment.width)
    for arc in field.arcs:
        points = sample_arc(arc, float(scene_cfg["arc_segment_length"]))
        if arc.is_full_circle:
            points = np.vstack([points, points[:1]])
        for i, (p, q) in enumerate(zip(points[:-1], points[1:])):
            # Lengthen each piece slightly so neighbours overlap and leave no wedge-shaped gaps.
            direction = (q - p) / np.linalg.norm(q - p)
            pad = 0.5 * arc.width * math.tan(0.5 * arc.span / (len(points) - 1))
            add_line_box(f"line/{arc.name}/{i}", p - pad * direction, q + pad * direction,
                         arc.width)

    # ---- goals
    goal_rgba = [*scene_cfg["goal"]["rgb"], 1.0]
    net_rgba = list(scene_cfg["goal"]["net_rgba"])
    for goal in field.goals:
        g, radius = goal.dims, goal.dims.post_diameter / 2.0
        (x, y_pos), (_, y_neg) = goal.posts
        bar_z = g.height + radius
        back_x = x + goal.side * g.depth
        for label, y in (("pos_y", y_pos), ("neg_y", y_neg)):
            world.add_geom(name=f"{goal.name}/post_{label}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[radius, 0, 0], fromto=[x, y, 0, x, y, bar_z], rgba=goal_rgba,
                           material="matte", contype=0, conaffinity=0)
        world.add_geom(name=f"{goal.name}/crossbar", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       size=[radius, 0, 0], fromto=[x, y_neg, bar_z, x, y_pos, bar_z],
                       rgba=goal_rgba, material="matte", contype=0, conaffinity=0)
        if net_rgba[3] > 0.0:
            t = 0.004
            panels = {
                "back": ([back_x, 0, bar_z / 2], [t, y_pos, bar_z / 2]),
                "top": ([(x + back_x) / 2, 0, bar_z], [g.depth / 2, y_pos, t]),
                "side_pos_y": ([(x + back_x) / 2, y_pos, bar_z / 2], [g.depth / 2, t, bar_z / 2]),
                "side_neg_y": ([(x + back_x) / 2, y_neg, bar_z / 2], [g.depth / 2, t, bar_z / 2]),
            }
            for label, (pos, size) in panels.items():
                world.add_geom(name=f"{goal.name}/net_{label}", type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=pos, size=size, rgba=net_rgba, material="matte",
                               group=NET_GROUP, contype=0, conaffinity=0)

    # ---- ball
    ball_cfg = scene_cfg["ball"]
    scene_objects.add_ball_material(spec, ball_cfg)
    ball = world.add_body(name="ball", mocap=True,
                          pos=[*ball_cfg["position"], float(ball_cfg["radius"])])
    ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, material="ball",
                  size=[float(ball_cfg["radius"]), 0, 0], contype=0, conaffinity=0)

    # ---- lights
    for i, light_cfg in enumerate(lighting["lights"]):
        direction = np.asarray(light_cfg["direction"], float)
        direction /= np.linalg.norm(direction)
        light = world.add_light(name=f"light_{i}", pos=(-20.0 * direction).tolist(),
                                dir=direction.tolist())
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        tint = light_cfg.get("rgb", [1.0, 1.0, 1.0])
        light.diffuse = [float(light_cfg["diffuse"]) * float(c) for c in tint]
        light.specular = [float(light_cfg.get("specular", 0.0))] * 3
        light.castshadow = bool(light_cfg.get("cast_shadow", False) and lighting["shadows"])

    # ---- other robots (kinematic humanoid placeholders) and the backdrop; both optional
    for index, appearance in enumerate(scene_cfg.get("robots") or []):
        scene_objects.add_humanoid(world, f"robot{index}", appearance)
    scene_objects.add_background(world, field, scene_cfg.get("background"))

    # ---- camera platform: placeholder body, stereo rig, cameras
    rig_mount = camera_cfg.mount
    base = world.add_body(name="base", mocap=True)
    if robot is None:
        torso_top = max(0.25, rig_mount.height - 0.12)
        base.add_geom(name="platform/torso", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      size=[0.09, 0, 0], fromto=[0, 0, 0.09, 0, 0, torso_top],
                      rgba=[0.25, 0.3, 0.8, 1.0], group=PLATFORM_GROUP, material="matte",
                      contype=0, conaffinity=0)
    else:
        _add_robot_body(spec, robot, mount)
    base.add_geom(name="platform/heading", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.12, 0.02, 0.005], pos=[0.12, 0, 0.006], rgba=[0.9, 0.2, 0.2, 1.0],
                  group=PLATFORM_GROUP, material="matte", contype=0, conaffinity=0)

    rig = world.add_body(name="stereo_rig", mocap=True, pos=[0, 0, rig_mount.height])
    # Housing with the ZED Mini's outer dimensions (124.5 x 30.5 x 26.5 mm), behind the lenses.
    rig.add_geom(name="platform/camera_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[0.013, 0.062, 0.015], pos=[-0.02, 0, 0], rgba=[0.15, 0.15, 0.15, 1.0],
                 group=PLATFORM_GROUP, material="matte", contype=0, conaffinity=0)
    model_camera = StereoCameraModel(camera_cfg)
    for name, side in ((LEFT_CAMERA, "left"), (RIGHT_CAMERA, "right")):
        T_rig_optical = model_camera.T_rig_optical(side)
        # MuJoCo's camera frame is not the optical frame: convert explicitly.
        R_rig_mujoco = T_rig_optical[:3, :3] @ tf.R_OPTICAL_MUJOCO
        _add_pinhole_camera(rig, name, k, T_rig_optical[:3, 3], R_rig_mujoco)
    _add_pinhole_camera(world, OVERVIEW_CAMERA, k_overview,
                        [0.0, 0.0, float(scene_cfg["overview"]["height_m"])], np.eye(3))
    if robot is not None:
        chase_rig = world.add_body(name="chase_rig", mocap=True)
        camera = chase_rig.add_camera(name=CHASE_CAMERA, fovy=42.0)
        camera.resolution = list(CHASE_SIZE)
    return spec


def _add_robot_body(spec: mujoco.MjSpec, robot, mount) -> None:
    """The Phase 4 robot body under the mocap body `BODY_ROOT`, with its mount markers."""
    from robosoccer_perception.simulation import humanoid

    if robot.is_placeholder:
        body = robot.body
        carrier = spec.worldbody.add_body(name=BODY_ROOT, mocap=True)
        carrier.add_geom(name="platform/torso", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                         size=[0.09, 0, 0],
                         fromto=[0, 0, 0.09 - body.standing_height, 0, 0, body.torso_top[2]],
                         rgba=[0.25, 0.3, 0.8, 1.0], group=PLATFORM_GROUP, material="matte",
                         contype=0, conaffinity=0)
    else:
        carrier = humanoid.attach_to_scene(spec, robot, PLATFORM_GROUP, root=BODY_ROOT)
    if mount is not None:
        humanoid.add_mount_markers(carrier, robot.body, mount, MARKER_GROUP)


def _add_pinhole_camera(body, name: str, k: CameraIntrinsics, position, R_parent_mujoco) -> None:
    """Add a camera whose projection is exactly the pinhole model ``k``.

    MuJoCo takes the principal point as an offset from the image centre in pixels. Its
    horizontal sign is opposite to OpenCV's ``cx`` and its vertical sign matches ``cy``; this was
    determined by rendering and is pinned by
    tests/test_stereo_camera.py::test_asymmetric_intrinsics_are_honoured_by_the_renderer.
    """
    camera = body.add_camera(name=name, pos=np.asarray(position, float).tolist(),
                             quat=tf.quat_wxyz_from_matrix(R_parent_mujoco))
    camera.resolution = [k.width, k.height]
    camera.sensor_size = [k.width * PIXEL_PITCH, k.height * PIXEL_PITCH]
    camera.focal_pixel = [k.fx, k.fy]
    camera.principal_pixel = [(k.width - 1) / 2.0 - k.cx, k.cy - (k.height - 1) / 2.0]


@dataclasses.dataclass
class _PlatformState:
    base_pose: np.ndarray
    rig: RigState
    ball_position: np.ndarray
    ball_quat: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    robots: list = dataclasses.field(default_factory=list)      # RobotState | None per robot
    T_footprint_body: np.ndarray | None = None                  # Phase 4: the body's pose
    joints: dict | None = None                                  # Phase 4: humanoid joint angles


@dataclasses.dataclass(frozen=True)
class _RobotHandle:
    name: str
    mocap_id: int
    qpos_addresses: np.ndarray          # one per scene_objects.HUMANOID_JOINTS
    geom_ids: np.ndarray


class MujocoScene:
    """A compiled scene plus the scripted-control API of the camera platform."""

    def __init__(self, field: FieldGeometry, camera_cfg: StereoCameraConfig | None = None,
                 scene_cfg: dict[str, Any] | None = None, scenario_id: str = "unnamed",
                 robot=None, mount=None):
        self.field = field
        self.camera_cfg = camera_cfg or StereoCameraConfig.load()
        self.camera_model = StereoCameraModel(self.camera_cfg)
        self.scene_cfg = scene_cfg or load_scene_config()
        self.scenario_id = scenario_id
        self.robot, self.mount = robot, mount
        self.model = build_spec(field, self.camera_cfg, self.scene_cfg, robot, mount).compile()
        self.data = mujoco.MjData(self.model)
        self.step_index = 0
        self.time = 0.0
        names = ("base", "stereo_rig", "ball") + (
            (BODY_ROOT, "chase_rig") if robot is not None else ())
        self._mocap = {name: int(self.model.body_mocapid[self.model.body(name).id])
                       for name in names}
        self.humanoid = None                # the posed humanoid body, when one is attached
        if robot is not None and not robot.is_placeholder:
            from robosoccer_perception.simulation.humanoid import HumanoidBody

            self.humanoid = HumanoidBody(self.model, robot, root=BODY_ROOT)
        ball_cfg = self.scene_cfg["ball"]
        self._state = _PlatformState(
            np.zeros(3), self.camera_cfg.mount,
            np.array([*ball_cfg["position"], float(ball_cfg["radius"])], dtype=float))
        self._robots: list[_RobotHandle] = []
        for index in range(len(self.scene_cfg.get("robots") or [])):
            name = f"robot{index}"
            body_ids = [b for b in range(self.model.nbody)
                        if self.model.body(b).name == name
                        or self.model.body(b).name.startswith(f"{name}/")]
            self._robots.append(_RobotHandle(
                name, int(self.model.body_mocapid[self.model.body(name).id]),
                np.array([self.model.joint(f"{name}/{joint}").qposadr[0]
                          for joint in scene_objects.HUMANOID_JOINTS]),
                np.flatnonzero(np.isin(self.model.geom_bodyid, body_ids))))
        self._state.robots = [None] * len(self._robots)
        self._renderers: dict[tuple, mujoco.Renderer] = {}
        self._apply()

    # ------------------------------------------------------------------ scripted control
    @property
    def base_pose(self) -> np.ndarray:
        return self._state.base_pose.copy()

    @property
    def rig(self) -> RigState:
        return self._state.rig

    def teleport(self, base_pose, rig: RigState | None = None) -> None:
        """Place the platform at an arbitrary ``[x, y, yaw]`` (and optionally move the rig)."""
        pose = np.array(base_pose, dtype=float)
        pose[2] = tf.wrap_angle(pose[2])
        self._state.base_pose = pose
        if rig is not None:
            self._state.rig = rig
        self._apply()

    def set_rig(self, *, height: float | None = None, pitch: float | None = None,
                pan: float | None = None) -> None:
        """Change the camera height (m), pitch or pan (rad). Positive pitch looks down."""
        changes = {key: float(value) for key, value in
                   (("height", height), ("pitch", pitch), ("pan", pan)) if value is not None}
        self._state.rig = self._state.rig.replace(**changes)
        self._apply()

    def rotate_in_place(self, angle: float) -> None:
        self.teleport(self._state.base_pose + [0.0, 0.0, angle])

    def set_body(self, T_footprint_body=None, joints: dict | None = None) -> None:
        """Pose the robot's body (Phase 4 scenes): its body frame in the footprint frame, and
        for the humanoid its joint angles (None = the stand configuration). ``None`` for the
        transform stands the body upright at its nominal height."""
        if self.robot is None:
            raise RuntimeError("this scene has no robot body; construct MujocoScene(robot=...)")
        self._state.T_footprint_body = None if T_footprint_body is None \
            else np.array(T_footprint_body, dtype=float)
        self._state.joints = None if joints is None else dict(joints)
        self._apply()

    def body_pose(self) -> np.ndarray:
        """``T_field_body`` of the robot's body frame, read back from MuJoCo."""
        name = "robot/pelvis" if self.humanoid is not None else BODY_ROOT
        body = self.data.body(name)
        return tf.make_transform(np.array(body.xmat).reshape(3, 3), np.array(body.xpos))

    def set_ball(self, position, quat=None) -> None:
        """Move the ball. A 2-D position puts it on the ground. ``None`` hides it.

        ``quat`` (w, x, y, z) optionally turns the ball, which changes the visible pattern.
        """
        if position is None:
            position = scene_objects.PARKED_POSITION
        position = np.asarray(position, dtype=float)
        if position.shape == (2,):
            position = np.array([*position, float(self.scene_cfg["ball"]["radius"])])
        self._state.ball_position = position
        if quat is not None:
            self._state.ball_quat = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        self._apply()

    @property
    def num_robots(self) -> int:
        """Number of other robots (humanoid placeholders) built into this scene."""
        return len(self._robots)

    def set_robots(self, states) -> None:
        """Place the other robots. ``states`` has one `scene_objects.RobotState` or ``None``
        (hidden) per robot; a shorter list hides the remaining robots."""
        states = list(states)
        if len(states) > len(self._robots):
            raise ValueError(f"this scene has {len(self._robots)} robots, got {len(states)} states")
        self._state.robots = states + [None] * (len(self._robots) - len(states))
        self._apply()

    def robot_geom_ids(self, index: int) -> np.ndarray:
        return self._robots[index].geom_ids.copy()

    def follow(self, trajectory, step: int) -> None:
        """Jump to step ``step`` of a `Trajectory`, including its head motion and time."""
        rig = self.camera_cfg.mount
        if trajectory.pan is not None:
            rig = rig.replace(pan=float(trajectory.pan[step]))
        if trajectory.pitch is not None:
            rig = rig.replace(pitch=float(trajectory.pitch[step]))
        self.step_index, self.time = int(step), float(step * trajectory.dt)
        self.teleport(trajectory.poses[step], rig)

    def advance(self, dt: float) -> None:
        """Advance the simulation clock (the scene is kinematic; nothing else changes)."""
        self.step_index += 1
        self.time += float(dt)

    def _apply(self) -> None:
        state = self._state
        T_field_base = tf.T_field_base_from_pose(state.base_pose)
        T_field_rig = T_field_base @ self.camera_model.T_base_rig(state.rig)
        for name, transform in (("base", T_field_base), ("stereo_rig", T_field_rig)):
            self.data.mocap_pos[self._mocap[name]] = transform[:3, 3]
            self.data.mocap_quat[self._mocap[name]] = tf.quat_wxyz_from_matrix(transform[:3, :3])
        self.data.mocap_pos[self._mocap["ball"]] = state.ball_position
        self.data.mocap_quat[self._mocap["ball"]] = state.ball_quat
        if self.robot is not None:
            T_footprint_body = state.T_footprint_body
            if T_footprint_body is None:
                T_footprint_body = tf.make_transform(
                    None, [0.0, 0.0, self.robot.body.standing_height])
            T_field_body = T_field_base @ T_footprint_body
            if self.humanoid is not None:
                self.humanoid.apply(self.data, T_field_body, state.joints)
            else:
                self.data.mocap_pos[self._mocap[BODY_ROOT]] = T_field_body[:3, 3]
                self.data.mocap_quat[self._mocap[BODY_ROOT]] = tf.quat_wxyz_from_matrix(
                    T_field_body[:3, :3])
            # The chase camera keeps a fixed offset in FIELD axes and looks at the robot, so
            # the robot is seen to turn and to walk across the field.
            eye = np.array([state.base_pose[0], state.base_pose[1], 0.0]) + CHASE_OFFSET
            target = np.array([state.base_pose[0], state.base_pose[1], 0.0]) + CHASE_TARGET
            backward = (eye - target) / np.linalg.norm(eye - target)
            right = np.cross([0.0, 0.0, 1.0], backward)
            right /= np.linalg.norm(right)
            R_field_camera = np.column_stack([right, np.cross(backward, right), backward])
            self.data.mocap_pos[self._mocap["chase_rig"]] = eye
            self.data.mocap_quat[self._mocap["chase_rig"]] = tf.quat_wxyz_from_matrix(
                R_field_camera)
        self.data.time = self.time
        standing = []
        for handle, robot in zip(self._robots, state.robots):
            if robot is None:
                self.data.mocap_pos[handle.mocap_id] = scene_objects.PARKED_POSITION
                continue
            self.data.mocap_pos[handle.mocap_id] = [robot.xy[0], robot.xy[1], 0.0]
            self.data.mocap_quat[handle.mocap_id] = tf.quat_wxyz_from_matrix(
                scene_objects.fallen_rotation(robot.yaw, robot.fallen))
            self.data.qpos[handle.qpos_addresses] = robot.joints
            standing.append(handle)
        mujoco.mj_forward(self.model, self.data)
        if standing:
            # Whatever the limb pose or fall direction, rest the lowest point on the ground.
            for handle in standing:
                self.data.mocap_pos[handle.mocap_id][2] -= scene_objects.lowest_point_z(
                    self.model, self.data, handle.geom_ids)
            mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ ground truth
    def camera_pose(self, camera_name: str) -> np.ndarray:
        """``T_field_optical`` of a camera, read back from MuJoCo's own kinematics."""
        cam = self.data.cam(camera_name)
        R_field_mujoco = np.array(cam.xmat, dtype=float).reshape(3, 3)
        return tf.make_transform(R_field_mujoco @ tf.R_OPTICAL_MUJOCO.T, np.array(cam.xpos))

    def ground_truth(self) -> GroundTruth:
        """The true state, read from MuJoCo (not from the commanded values)."""
        base = self.data.body("base")
        T_field_base = tf.make_transform(np.array(base.xmat).reshape(3, 3), np.array(base.xpos))
        return GroundTruth(
            scenario_id=self.scenario_id, step=self.step_index, timestamp=self.time,
            base_pose=tf.pose_from_T_field_base(T_field_base), rig=self._state.rig,
            T_field_base=T_field_base,
            T_field_left_optical=self.camera_pose(LEFT_CAMERA),
            T_field_right_optical=self.camera_pose(RIGHT_CAMERA),
            ball_position=np.array(self.data.body("ball").xpos))

    # ------------------------------------------------------------------ rendering
    def render(self, camera_name: str, show_platform: bool = True,
               show_markers: bool = False) -> np.ndarray:
        """RGB image (H, W, 3 uint8) from a named camera, at that camera's own resolution.
        ``show_markers`` also draws the body-frame and mount markers of a Phase 4 scene."""
        width, height = (int(v) for v in self.model.cam_resolution[self.model.camera(camera_name).id])
        renderer = self._renderers.get((height, width))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[(height, width)] = renderer
        option = mujoco.MjvOption()
        option.geomgroup[PLATFORM_GROUP] = int(show_platform)
        option.geomgroup[MARKER_GROUP] = int(show_markers)
        renderer.update_scene(self.data, camera=camera_name, scene_option=option)
        return renderer.render().copy()

    def render_chase(self, show_markers: bool = True) -> np.ndarray:
        """Third-person view of the robot with its body-frame and mount markers."""
        if self.robot is None:
            raise RuntimeError("this scene has no chase camera; construct MujocoScene(robot=...)")
        return self.render(CHASE_CAMERA, show_platform=True, show_markers=show_markers)

    def render_segmentation(self, camera_name: str, scale: int = 1,
                            only_geoms=None) -> np.ndarray:
        """The id of the geom seen at every pixel, ``(H * scale, W * scale)`` int32.

        Pixels that show no geom (sky) are -1. The camera platform and the goal nets are left
        out: the platform never appears in its own cameras, and the nets are see-through, so
        what stands behind a net counts as seen. No anti-aliasing is applied; each pixel is
        the geom at its centre. With ``scale`` > 1 the same view is sampled on a finer grid
        (pixel ``(v, u)`` of the camera covers the block ``[v*scale:(v+1)*scale,
        u*scale:(u+1)*scale]``), which needs ``offscreen_scale >= scale`` in the scene config.

        ``only_geoms`` renders just those geoms, as if nothing else existed: the un-occluded
        (amodal) silhouette of an object.
        """
        width, height = (int(v) * scale for v in
                         self.model.cam_resolution[self.model.camera(camera_name).id])
        if (width > self.model.vis.global_.offwidth
                or height > self.model.vis.global_.offheight):
            raise ValueError(f"segmentation at scale {scale} needs an offscreen buffer of "
                             f"{width}x{height}; set 'offscreen_scale: {scale}' in the scene "
                             "configuration")
        renderer = self._renderers.get((height, width, "segmentation"))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.enable_segmentation_rendering()
            self._renderers[(height, width, "segmentation")] = renderer
        option = mujoco.MjvOption()
        option.geomgroup[PLATFORM_GROUP] = 0
        option.geomgroup[NET_GROUP] = 0
        if only_geoms is None:
            renderer.update_scene(self.data, camera=camera_name, scene_option=option)
            return renderer.render()[..., 0].copy()
        only_geoms = np.asarray(only_geoms, dtype=int)
        saved = self.model.geom_group[only_geoms].copy()
        option.geomgroup[:] = 0
        option.geomgroup[ISOLATION_GROUP] = 1
        try:
            self.model.geom_group[only_geoms] = ISOLATION_GROUP
            renderer.update_scene(self.data, camera=camera_name, scene_option=option)
            return renderer.render()[..., 0].copy()
        finally:
            self.model.geom_group[only_geoms] = saved

    def render_overview(self, show_platform: bool = True) -> np.ndarray:
        """Top-down image of the whole carpet: image right = field +x, image up = field +y."""
        return self.render(OVERVIEW_CAMERA, show_platform)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> "MujocoScene":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
