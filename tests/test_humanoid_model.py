"""UTRA's humanoid_prototype_1 in the perception simulator: the imported bundle, kinematic
posing, the configurable camera mount on the real body, and the placeholder fallback."""
import json
import math
import struct

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robosoccer_perception.geometry import mount as mg  # noqa: E402
from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.camera import StereoCameraConfig  # noqa: E402
from robosoccer_perception.geometry.field import FieldGeometry  # noqa: E402
from robosoccer_perception.geometry.mount import CameraMount  # noqa: E402
from robosoccer_perception.geometry.robot_description import (  # noqa: E402
    RobotDescription, robot_is_installed)
from robosoccer_perception.simulation import humanoid  # noqa: E402
from robosoccer_perception.simulation import mujoco_scene as ms  # noqa: E402
from robosoccer_perception.simulation.humanoid import BodyPose, HumanoidBody  # noqa: E402

ROBOT = "humanoid_prototype_1"
pytestmark = pytest.mark.skipif(not robot_is_installed(ROBOT),
                                reason="humanoid model not imported (scripts/import_humanoid_model.py)")
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_motor", "ankle_pitch",
              "ankle_rod")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")


@pytest.fixture(scope="module")
def robot() -> RobotDescription:
    return RobotDescription.load(ROBOT)


@pytest.fixture(scope="module")
def model(robot):
    return humanoid.build_robot_spec(robot).compile()


@pytest.fixture(scope="module")
def body(robot):
    spec = mujoco.MjSpec()
    humanoid.attach_to_scene(spec, robot, body_group=1)
    return HumanoidBody(spec.compile(), robot)


# ------------------------------------------------------------------ the imported bundle
def test_bundle_records_where_it_came_from_and_what_was_left_out(robot):
    manifest = humanoid.load_manifest(robot)
    assert manifest["name"] == ROBOT
    assert len(manifest["source"]["robot_xml_sha256"]) == 64
    assert len(manifest["meshes"]) == 21 == len(manifest["visual_geoms"])
    assert manifest["triangles"] == sum(m["triangles"] for m in manifest["meshes"].values())
    assert manifest["triangles"] > 300_000                       # nothing was simplified
    assert all(len(m["sha256"]) == 64 for m in manifest["meshes"].values())
    for removed in ("collision geoms and meshes", "actuators", "equality constraints", "sensors",
                    "floating-base joint"):
        assert removed in manifest["removed"]
    skeleton = (robot.asset_dir / humanoid.SKELETON_FILE).read_text()
    for tag in ("<geom", "<actuator", "<equality", "<sensor", "<contact", "freejoint", "<mesh"):
        assert tag not in skeleton, tag


def test_the_actual_humanoid_model_loads_with_its_kinematic_tree(model):
    joints = [model.joint(j).name for j in range(model.njnt)]
    expected = {f"{side}_{name}_joint" for side in ("left", "right")
                for name in LEG_JOINTS + ARM_JOINTS}
    assert set(joints) == expected and model.njnt == 20 == model.nq   # no floating base
    assert all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in range(model.njnt))
    assert model.nmesh == 21 and model.nmeshface > 300_000
    assert model.nu == 0 and model.neq == 0 and model.nsensor == 0
    assert not model.geom_contype.any() and not model.geom_conaffinity.any()   # visual only
    assert model.body_mass.sum() == pytest.approx(13.94, abs=0.05)             # inertials kept
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu") >= 0
    assert np.allclose(model.body("pelvis").pos, 0.0)
    # documented axes: hip pitch about +y, hip roll about +x, hip yaw about +z
    for name, axis in (("left_hip_pitch_joint", [0, 1, 0]), ("left_hip_roll_joint", [1, 0, 0]),
                       ("left_hip_yaw_joint", [0, 0, 1]), ("right_knee_joint", [0, 1, 0])):
        assert np.allclose(model.jnt_axis[model.joint(name).id], axis)


def test_stored_body_geometry_is_what_the_model_measures(robot):
    measured = humanoid.measure_body_geometry(robot)
    stored = robot.body
    assert measured.standing_height == pytest.approx(stored.standing_height, abs=1e-6)
    assert np.allclose(measured.torso_top, stored.torso_top, atol=1e-6)
    assert np.allclose(measured.imu_xyz, stored.imu_xyz) and not stored.placeholder
    # The stand pose is the official "home" keyframe, and the pelvis height it implies is the
    # official one (0.405699 m in the model's scene.xml).
    manifest = humanoid.load_manifest(robot)
    assert {k: v for k, v in robot.stance.items() if v} == pytest.approx(manifest["home_stance"])
    assert stored.standing_height == pytest.approx(0.4057, abs=2e-4)
    assert stored.torso_top_height == pytest.approx(0.679, abs=2e-3)   # the torso; no head yet


def test_import_helpers_strip_and_read_without_changing_geometry(tmp_path):
    xml = """<mujoco model="toy"><compiler angle="radian" meshdir="../meshes"/>
      <asset><mesh name="a" file="a.stl"/><mesh name="a_collision_0" file="c.stl"/></asset>
      <worldbody><body name="pelvis" pos="0 0 0.4"><freejoint name="root"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom name="a_visual" type="mesh" mesh="a" contype="0" conaffinity="0" group="2"/>
        <geom name="a_collision" type="mesh" mesh="a_collision_0" group="3"/>
        <site name="imu" pos="0 0 0"/>
        <body name="leg" pos="0 0.1 0"><joint name="hip" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <geom name="foot_box" type="box" size="0.1 0.1 0.1" group="3"/></body></body></worldbody>
      <actuator><general joint="hip"/></actuator>
      <sensor><gyro site="imu"/></sensor></mujoco>"""
    skeleton, visuals = humanoid.strip_to_skeleton(xml)
    assert visuals == [{"body": "pelvis", "geom": "a_visual", "mesh": "a"}]
    assert 'joint name="hip"' in skeleton and 'site name="imu"' in skeleton
    for gone in ("freejoint", "<geom", "actuator", "sensor", "<asset"):
        assert gone not in skeleton
    assert mujoco.MjSpec.from_string(skeleton).compile().nq == 1

    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                          [[1, 0, 0], [1, 1, 0], [0, 1, 0]]], dtype="<f4")
    path = tmp_path / "quad.stl"
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80 + struct.pack("<I", len(triangles)))
        for triangle in triangles:
            handle.write(struct.pack("<3f", 0, 0, 1) + triangle.tobytes() + b"\0\0")
    vertices, faces = humanoid.read_binary_stl(path)
    assert len(vertices) == 4 and faces.shape == (2, 3)          # shared corners merged ...
    assert np.array_equal(vertices[faces], triangles)            # ... and no corner moved
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="not a binary STL"):
        humanoid.read_binary_stl(path)
    with pytest.raises(FileNotFoundError):
        humanoid.import_model(tmp_path, tmp_path / "out")


# ------------------------------------------------------------------ posing
def test_ankle_four_bars_stay_closed_over_the_whole_ankle_range(body):
    data = mujoco.MjData(body.model)
    for side in ("left", "right"):
        linkage = body.linkage[side]
        assert linkage.residual < 1e-6
        pin = body.model.site(f"robot/{side}_ankle_pitch_linkage_input_pin").id
        coupler = body.model.site(f"robot/{side}_ankle_pitch_linkage_coupler_pin").id
        for angle in np.linspace(linkage.ankle[0], linkage.ankle[-1], 23):
            body.apply(data, np.eye(4), {f"{side}_ankle_pitch_joint": float(angle)})
            mujoco.mj_kinematics(body.model, data)
            assert np.linalg.norm(data.site_xpos[pin] - data.site_xpos[coupler]) < 2e-4


def _lowest(body: HumanoidBody, data) -> float:
    mujoco.mj_kinematics(body.model, data)
    return min(float(humanoid._mesh_points(body.model, data, g)[:, 2].min())
               for g in body.geom_ids if body.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)


def test_the_body_is_posed_kinematically_and_its_feet_stay_on_the_ground(body, robot):
    data = mujoco.MjData(body.model)
    height = robot.body.standing_height
    body.apply(data, tf.make_transform(None, [1.0, -2.0, height]))       # the stand pose
    assert _lowest(body, data) == pytest.approx(0.0, abs=1e-4)
    pelvis = data.body("robot/pelvis")
    assert np.allclose(pelvis.xpos, [1.0, -2.0, height])
    # Tilted and stepping: the cosmetic leg motion keeps the feet within 2 cm of the ground
    # and never drives a joint past its documented range.
    for roll, pitch, phase in ((0.05, 0.08, 0.5), (-0.06, 0.02, 2.5), (0.0, -0.05, 4.0),
                               (0.04, 0.10, 5.5)):
        pose = BodyPose(mg.T_footprint_body(height - 0.005, roll, pitch), phase, 1.0)
        joints = body.joints_for(pose)
        body.apply(data, pose.T_footprint_body, joints)
        assert abs(_lowest(body, data)) < 0.02
        for name, (low, high) in body._range.items():
            value = data.qpos[body._address[name]]
            assert low - 1e-9 <= value <= high + 1e-9, name
        assert pose.roll_pitch == pytest.approx((roll, pitch))
    still = body.joints_for(BodyPose(mg.T_footprint_body(height), 1.0, 0.0))
    assert still["left_knee_joint"] == pytest.approx(robot.stance["left_knee_joint"], abs=0.02)
    assert still["left_shoulder_pitch_joint"] == 0.0             # no arm swing when standing


# ------------------------------------------------------------------ in the field scene
@pytest.fixture(scope="module")
def scene(gl_context, robot):
    camera = StereoCameraConfig.load("zed_mini_sim").with_resolution(320, 180)
    with ms.MujocoScene(FieldGeometry.load("s_field"), camera, robot=robot,
                        mount=CameraMount.load("head_mid")) as built:
        yield built


def _place(scene, mount, pose, roll=0.0, pitch=0.0, height=None):
    body = scene.robot.body
    height = body.standing_height if height is None else height
    rig = mg.rig_state_from_transform(mg.T_footprint_rig(mount, body, height, roll, pitch))
    scene.teleport(pose, rig)
    scene.set_body(mg.T_footprint_body(height, roll, pitch))


@pytest.mark.render
def test_camera_and_body_are_where_the_mount_chain_says(scene, robot):
    mount = CameraMount.load("head_mid")
    for pose, roll, pitch in (([0.0, 0.0, 0.0], 0.0, 0.0), ([-1.5, 0.8, 0.7], 0.06, -0.04),
                              ([2.0, -1.0, -2.2], -0.03, 0.09)):
        _place(scene, mount, pose, roll, pitch)
        T_field_footprint = tf.T_field_base_from_pose(pose)
        T_footprint_body = mg.T_footprint_body(robot.body.standing_height, roll, pitch)
        assert np.allclose(scene.body_pose(), T_field_footprint @ T_footprint_body, atol=1e-9)
        expected = T_field_footprint @ T_footprint_body @ mount.T_body_rig(robot.body) \
            @ mg.T_rig_optical(scene.camera_cfg.baseline, "left")
        truth = scene.ground_truth()
        assert np.allclose(truth.T_field_left_optical, expected, atol=1e-9)
        # the footprint pose that ground truth exports is the 3-DOF pose that was commanded
        assert np.allclose(truth.base_pose[:2], pose[:2]) and np.isclose(
            tf.angle_difference(truth.base_pose[2], pose[2]), 0.0)
    _place(scene, mount, [0.0, 0.0, 0.0])
    assert scene.ground_truth().T_field_left_optical[2, 3] == pytest.approx(
        mount.standing_height(robot.body))


@pytest.mark.render
def test_the_humanoid_is_rendered_and_markers_stay_out_of_its_own_camera(scene):
    mount = CameraMount.load("head_mid")
    _place(scene, mount, [-1.0, 0.5, 0.3])
    with_markers, without = scene.render_chase(True), scene.render_chase(False)
    assert with_markers.shape == (ms.CHASE_SIZE[1], ms.CHASE_SIZE[0], 3)
    assert (with_markers != without).any(axis=-1).sum() > 50      # markers are drawn on request
    hidden = scene.render(ms.CHASE_CAMERA, show_platform=False)
    assert (without != hidden).any(axis=-1).mean() > 0.01         # the robot fills part of it
    # The robot's own camera: markers never, the body only if it is in view (it is not, from
    # a head above the torso looking 25 deg down).
    own = scene.render(ms.LEFT_CAMERA, show_platform=True)
    assert np.array_equal(own, scene.render(ms.LEFT_CAMERA, show_platform=True,
                                            show_markers=False))
    assert np.array_equal(own, scene.render(ms.LEFT_CAMERA, show_platform=False))


@pytest.mark.render
def test_the_camera_sees_its_own_body_when_the_mount_points_at_it(gl_context, robot):
    """Nothing hides the robot from itself: a camera that looks at its feet sees them."""
    camera = StereoCameraConfig.load("zed_mini_sim").with_resolution(320, 180)
    down = CameraMount("look_down", (0.10, 0.0, 0.10), (0.0, math.radians(80.0), 0.0))
    with ms.MujocoScene(FieldGeometry.load("s_field"), camera, robot=robot, mount=down) as scene:
        _place(scene, down, [0.0, 0.0, 0.0])
        shown = scene.render(ms.LEFT_CAMERA, show_platform=True)
        hidden = scene.render(ms.LEFT_CAMERA, show_platform=False)
        assert (shown != hidden).any(axis=-1).mean() > 0.02


@pytest.mark.render
def test_virtual_head_presets_change_the_rendered_camera_height(scene, robot):
    heights = []
    for name in ("head_low", "head_mid", "head_high", "head_tall"):
        mount = CameraMount.load(name)
        _place(scene, mount, [0.5, 0.0, 0.0])
        heights.append(scene.ground_truth().T_field_left_optical[2, 3])
        assert heights[-1] == pytest.approx(mount.standing_height(robot.body), abs=1e-9)
    assert np.all(np.diff(heights) > 0.03) and 0.72 < heights[0] < heights[-1] < 0.92


@pytest.mark.render
def test_placeholder_platform_is_still_available_and_can_tilt(gl_context):
    camera = StereoCameraConfig.load("zed_mini_sim").with_resolution(320, 180)
    placeholder = RobotDescription.load("placeholder")
    mount = CameraMount.load("legacy_rig")
    with ms.MujocoScene(FieldGeometry.load("s_field"), camera, robot=placeholder,
                        mount=mount) as scene:
        assert scene.humanoid is None
        _place(scene, mount, [0.0, 0.0, 0.0])
        level = scene.ground_truth().T_field_left_optical
        assert level[2, 3] == pytest.approx(0.90)                # the Phase 1 to 3 rig height
        _place(scene, mount, [0.0, 0.0, 0.0], pitch=0.1)
        assert scene.ground_truth().T_field_left_optical[0, 3] > level[0, 3] + 0.03
        assert np.allclose(scene.body_pose()[:3, :3], tf.rot_y(0.1))
        assert scene.render_chase().shape[:2] == ms.CHASE_SIZE[::-1]


@pytest.mark.render
def test_scenes_without_a_robot_are_the_phase_3_scenes(gl_context):
    camera = StereoCameraConfig.load("zed_mini_sim").with_resolution(320, 180)
    with ms.MujocoScene(FieldGeometry.load("s_field"), camera) as scene:
        assert scene.robot is None and scene.humanoid is None
        assert scene.model.ncam == 3                             # no chase camera
        names = {scene.model.body(b).name for b in range(scene.model.nbody)}
        assert ms.BODY_ROOT not in names and "chase_rig" not in names
        assert scene.model.geom("platform/torso").bodyid == scene.model.body("base").id
        with pytest.raises(RuntimeError):
            scene.set_body(np.eye(4))
        with pytest.raises(RuntimeError):
            scene.render_chase()


def test_manifest_is_valid_json_with_no_absolute_paths(robot):
    text = (robot.asset_dir / humanoid.MANIFEST_FILE).read_text()
    json.loads(text)
    assert "/home/" not in text and "Desktop" not in text       # nothing machine-specific
