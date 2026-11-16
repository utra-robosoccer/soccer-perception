"""Phase 2 additions to the MuJoCo scene: other robots, ball looks, backdrop, camera roll and
segmentation (geom-id) rendering. Tests marked ``render`` need an OpenGL context."""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robosoccer_perception.geometry.camera import RigState, StereoCameraConfig  # noqa: E402
from robosoccer_perception.simulation import scene_objects  # noqa: E402
from robosoccer_perception.simulation.mujoco_scene import (  # noqa: E402
    LEFT_CAMERA, NET_GROUP, MujocoScene, load_scene_config, overview_intrinsics)
from robosoccer_perception.simulation.scene_objects import RobotState  # noqa: E402

ROBOTS = [{}, {"height": 1.2, "build": 1.2, "head_shape": "box"}]


@pytest.fixture(scope="module")
def camera_cfg():
    return StereoCameraConfig.load().with_resolution(320, 180)


@pytest.fixture(scope="module")
def scene(s_field, camera_cfg):
    cfg = load_scene_config(overrides={
        "robots": ROBOTS, "offscreen_scale": 2, "lighting": {"shadows": False},
        "background": {"walls": [{"side": "pos_x", "distance": 2.0, "height": 2.0,
                                  "rgb": [0.5, 0.4, 0.3]}],
                       "clutter": [{"position": [6.5, 1.0], "size": [0.5, 0.5, 1.0],
                                    "rgb": [0.9, 0.9, 0.9], "yaw": 0.3}]}})
    with MujocoScene(s_field, camera_cfg, cfg) as scene:
        yield scene


# ------------------------------------------------------------------ Phase 1 scene is unchanged
def test_default_scene_has_no_phase2_content(s_field):
    with MujocoScene(s_field) as plain:
        names = [plain.model.geom(i).name for i in range(plain.model.ngeom)]
        assert not any(n.startswith(("robot", "bg/")) for n in names)
        assert plain.num_robots == 0 and plain.model.nq == 0
        overview = overview_intrinsics(s_field, plain.scene_cfg)
        assert plain.model.vis.global_.offwidth == max(1280, overview.width)   # as in Phase 1
        for name in names:
            if "/net_" in name:
                assert plain.model.geom(name).group[0] == NET_GROUP


def test_zero_roll_leaves_the_camera_pose_untouched(camera_model):
    rig = RigState(height=0.9, pitch=math.radians(25), pan=0.3)
    np.testing.assert_array_equal(camera_model.T_base_rig(rig),
                                  camera_model.T_base_rig(rig.replace(roll=0.0)))


def test_roll_turns_the_camera_about_its_viewing_direction(camera_model):
    level = RigState(height=0.9, pitch=math.radians(25), pan=0.3)
    rolled = level.replace(roll=math.radians(10))
    T0, T1 = camera_model.T_base_camera("left", level), camera_model.T_base_camera("left", rolled)
    np.testing.assert_allclose(T1[:3, 2], T0[:3, 2], atol=1e-12)        # same optical axis
    assert math.degrees(math.acos(T0[:3, 0] @ T1[:3, 0])) == pytest.approx(10.0)
    assert T1[2, 0] < T0[2, 0], "positive roll tips the camera's right side down"


# ------------------------------------------------------------------ robots
def test_robots_are_placed_and_stand_on_the_ground(scene):
    assert scene.num_robots == 2
    states = [RobotState((1.0, 0.5), 0.7, (0.6, -0.4, 0.5, -0.5, 0.7, 0.1)),
              RobotState((-2.0, 1.0), -2.0)]
    scene.set_robots(states)
    for index, state in enumerate(states):
        body = scene.data.body(f"robot{index}")
        np.testing.assert_allclose(body.xpos[:2], state.xy, atol=1e-9)
        lowest = scene_objects.lowest_point_z(scene.model, scene.data, scene.robot_geom_ids(index))
        assert lowest == pytest.approx(0.0, abs=1e-6), "feet rest on the turf whatever the pose"
        head = scene.data.geom(f"robot{index}/head")
        assert head.xpos[2] > 0.6 * ROBOTS[index].get("height", 0.8)


@pytest.mark.parametrize("fallen", scene_objects.FALLEN_POSES)
def test_fallen_robots_lie_on_the_ground(scene, fallen):
    scene.set_robots([RobotState((0.0, 0.0), 0.3, fallen=fallen)])
    geoms = scene.robot_geom_ids(0)
    assert scene_objects.lowest_point_z(scene.model, scene.data, geoms) == pytest.approx(0.0, abs=1e-6)
    assert scene.data.geom("robot0/head").xpos[2] < 0.3, "the head is near the ground"


def test_hidden_robots_are_parked_out_of_sight(scene):
    scene.set_robots([RobotState((0.0, 0.0))])                 # the second robot is not listed
    assert scene.data.body("robot1").xpos[2] < -10
    scene.set_robots([None, None])
    assert scene.data.body("robot0").xpos[2] < -10
    with pytest.raises(ValueError):
        scene.set_robots([None, None, None])
    with pytest.raises(ValueError):
        scene_objects.fallen_rotation(0.0, "sideways")


def test_ball_can_be_hidden_and_turned(scene):
    scene.set_ball(None)
    assert scene.ground_truth().ball_position[2] < -10
    scene.set_ball([1.0, 0.5], quat=[0.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(scene.ground_truth().ball_position[:2], [1.0, 0.5])
    np.testing.assert_allclose(scene.data.body("ball").xquat, [0, 1, 0, 0], atol=1e-9)


@pytest.mark.parametrize("style", ["checker", "spots", "flat"])
def test_ball_styles_compile(s_field, camera_cfg, style):
    cfg = load_scene_config(overrides={"ball": {"style": style, "radius": 0.11}})
    with MujocoScene(s_field, camera_cfg, cfg) as other:
        assert other.model.geom("ball").size[0] == pytest.approx(0.11)
        assert other.ground_truth().ball_position[2] == pytest.approx(0.11)


def test_unknown_ball_style_is_rejected(s_field, camera_cfg):
    with pytest.raises(ValueError, match="ball style"):
        MujocoScene(s_field, camera_cfg, load_scene_config(overrides={"ball": {"style": "plaid"}}))


def test_backdrop_geoms_exist_outside_the_carpet(scene, s_field):
    wall, clutter = scene.data.geom("bg/wall_0_pos_x"), scene.data.geom("bg/clutter_0")
    assert wall.xpos[0] == pytest.approx(s_field.half_extent[0] + 2.0)
    assert not s_field.contains(clutter.xpos[:2])


# ------------------------------------------------------------------ segmentation rendering
@pytest.mark.render
def test_segmentation_render_names_the_geom_at_every_pixel(scene, gl_context, camera_cfg):
    pose = [-1.5, 0.2, math.radians(-10)]
    scene.teleport(pose, camera_cfg.mount)
    scene.set_robots([None, None])
    scene.set_ball([1.5, 0.3])
    ids = scene.render_segmentation(LEFT_CAMERA)
    k = camera_cfg.intrinsics
    assert ids.shape == (k.height, k.width) and ids.dtype == np.int32
    assert ids.min() >= -1 and ids.max() < scene.model.ngeom
    names = {scene.model.geom(int(i)).name for i in np.unique(ids) if i >= 0}
    assert {"turf", "ball"} <= names and any(n.startswith("line/") for n in names)
    assert not any("/net_" in n or n.startswith("platform/") for n in names)
    # The ball's pixels are where the camera model projects the ball.
    v, u = np.nonzero(ids == scene.model.geom("ball").id)
    expected, _, visible = scene.camera_model.project_field_points(
        pose, [1.5, 0.3, scene.scene_cfg["ball"]["radius"]])
    assert visible and len(u) > 20
    np.testing.assert_allclose([u.mean(), v.mean()], expected, atol=1.0)


@pytest.mark.render
def test_supersampled_segmentation_is_the_same_view_on_a_finer_grid(scene, gl_context, camera_cfg):
    scene.teleport([-1.5, 0.2, 0.1], camera_cfg.mount)
    scene.set_ball([1.0, 0.0])
    coarse, fine = scene.render_segmentation(LEFT_CAMERA), scene.render_segmentation(LEFT_CAMERA, 2)
    assert fine.shape == (2 * coarse.shape[0], 2 * coarse.shape[1])
    blocks = fine.reshape(coarse.shape[0], 2, coarse.shape[1], 2)
    uniform = (blocks == blocks[:, :1, :, :1]).all(axis=(1, 3))        # blocks of a single geom
    assert uniform.mean() > 0.9
    assert (blocks[:, 0, :, 0][uniform] == coarse[uniform]).mean() > 0.99
    with pytest.raises(ValueError, match="offscreen_scale"):
        scene.render_segmentation(LEFT_CAMERA, 8)


@pytest.mark.render
def test_isolated_render_shows_one_object_without_occluders(scene, gl_context, camera_cfg):
    scene.teleport([0.0, 0.0, 0.0], camera_cfg.mount)
    scene.set_ball([2.4, 0.1])
    scene.set_robots([None, RobotState((2.0, 0.0), 0.0)])      # its leg stands in front of the ball
    ball = scene.model.geom("ball").id
    groups_before = scene.model.geom_group.copy()
    alone = scene.render_segmentation(LEFT_CAMERA, only_geoms=[ball])
    np.testing.assert_array_equal(scene.model.geom_group, groups_before)
    assert set(np.unique(alone)) == {-1, ball}
    seen = scene.render_segmentation(LEFT_CAMERA) == ball
    assert (alone == ball).sum() > seen.sum(), "part of the ball is behind the robot"
    assert not (seen & (alone != ball)).any(), "visible pixels are a subset of the silhouette"
