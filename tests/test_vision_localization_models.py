"""The real thing: rendered camera images -> the TRAINED detector and segmenter -> geometric
features -> `ObservationFrame` -> particle filter -> pose, compared with ground truth by the
evaluator.

Needs MuJoCo rendering, the ``vision`` extra and the trained Phase 2 checkpoints
(``outputs/detection/yolo11n/weights/best.pt`` and ``outputs/segmentation/unet_small/best.pt``,
see README). Skipped when any of them is missing; nothing is downloaded and nothing is trained.
"""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from robosoccer_perception.geometry import ground_projection as gp  # noqa: E402
from robosoccer_perception.geometry import transforms as tf  # noqa: E402
from robosoccer_perception.geometry.landmarks import LandmarkType  # noqa: E402
from robosoccer_perception.localization.particle_filter import ParticleFilter  # noqa: E402
from robosoccer_perception.observations.types import ObservationFrame  # noqa: E402
from robosoccer_perception.perception.observation_builder import build_observation_frame  # noqa: E402
from robosoccer_perception.pipeline.vision_localization import VisionLocalizationPipeline  # noqa: E402
from robosoccer_perception.simulation import trajectories as tj  # noqa: E402
from robosoccer_perception.vision_benchmark import runner, scenarios, setup  # noqa: E402
from robosoccer_perception.vision_benchmark.camera_truth import (  # noqa: E402
    HeadJitter, MountError, TrueCameraModel)
from robosoccer_perception.vision_benchmark.feature_evaluation import evaluate_features  # noqa: E402
from robosoccer_perception.vision_benchmark.sim_camera import SimulatedCamera  # noqa: E402

pytestmark = [pytest.mark.vision, pytest.mark.render, pytest.mark.slow]


@pytest.fixture(scope="module")
def resources(gl_context):
    settings = setup.load_settings()
    missing = setup.missing_checkpoints(settings)
    if missing:
        pytest.skip("trained checkpoints not found: " + ", ".join(str(p) for p in missing))
    return setup.load_resources(settings)


def test_one_frame_goes_from_pixels_to_a_pose_update(resources, s_field, s_landmarks):
    """Every stage once, by hand: render, run both networks, extract, build the observation
    frame, update the Phase 1 filter, and only then look at the truth."""
    camera_cfg = resources.camera_config()
    calibration = gp.EstimatedCameraCalibration.from_camera_config(
        camera_cfg, uncertainty=resources.noise.calibration)
    true_pose = np.array([2.3, -0.6, -0.2])                 # in front of the +x goal
    with SimulatedCamera(s_field, camera_cfg) as sim:
        image = sim.capture(true_pose, TrueCameraModel(camera_cfg, MountError(),
                                                       HeadJitter.none()).sample(), (1.0, 0.5))
    assert image.shape == (camera_cfg.intrinsics.height, camera_cfg.intrinsics.width, 3)

    predictions = resources.vision.run(image)               # the image is all the models get
    assert predictions.segmentation.line_mask.sum() > 500
    assert len(predictions.goal_posts) == 2 and all(d.confidence > 0.5 for d in predictions.goal_posts)

    from robosoccer_perception.perception.extractor import GeometricFeatureExtractor
    features = GeometricFeatureExtractor(s_field, resources.extraction, resources.noise).extract(
        predictions, calibration, calibration.pose())
    frame = build_observation_frame(features, 0.0, resources.extraction.observations)
    assert type(frame) is ObservationFrame and len(frame.line_points) > 20
    kinds = {o.type for o in frame.landmarks}
    assert LandmarkType.GOAL_POST in kinds and kinds & {LandmarkType.L_INTERSECTION,
                                                         LandmarkType.T_INTERSECTION}

    localizer = ParticleFilter(s_field, s_landmarks, seed=0)
    localizer.init_gaussian(true_pose + [0.25, -0.2, math.radians(8.0)])
    for _ in range(5):
        info = localizer.update(frame)
    assert info.used
    estimate = localizer.estimate()

    # ---- evaluation: ground truth appears here for the first time
    assert np.linalg.norm(estimate.pose[:2] - true_pose[:2]) < 0.10
    assert abs(math.degrees(tf.angle_difference(estimate.pose[2], true_pose[2]))) < 3.0
    in_field = tf.se2_transform_points(true_pose, features.line_points.points)
    assert np.percentile(s_field.distance_to_paint(in_field), 95) < 0.10
    posts = tf.se2_inverse_transform_points(true_pose, s_landmarks.positions(LandmarkType.GOAL_POST))
    for post in features.goal_posts:
        assert post.range_valid, post.flags
        assert np.linalg.norm(posts - post.position, axis=1).min() < 0.15
    for junction in features.typed_intersections:
        on_map = s_landmarks.positions(junction.type)
        located = tf.se2_transform_points(true_pose, junction.position[None])[0]
        assert np.linalg.norm(on_map - located, axis=1).min() < 0.12, junction.type


def test_trained_models_give_accurate_features_on_random_views(resources):
    summary = evaluate_features(resources, frames=40, seed=11)
    assert summary["frames"] == 40
    assert summary["line_points"]["error_m"]["mean"] < 0.04
    assert summary["line_points"]["within_2_sigma"] > 0.95, "reported uncertainty is not optimistic"
    assert summary["projection"]["valid_share"] > 0.75
    junctions = summary["intersections"]["all"]
    assert junctions["reported"] >= 20 and junctions["precision"] >= 0.95
    assert junctions["position_error_m"]["mean"] < 0.08
    posts = summary["goal_posts"]
    assert posts["observations"] >= 10 and posts["unmatched_share"] <= 0.1
    assert posts["ground_point_error_m"]["median"] < 0.15
    assert posts["bearing_error_deg"]["median"] < 0.5
    assert summary["center_circle"]["false"] <= 1
    assert summary["geometry_latency_ms"]["mean_ms"] < 60.0


def _walk(field, rng):
    return tj.PathBuilder([-2.6, -1.3, 0.2]).forward(1.5, 0.3).turn(math.radians(60.0)) \
        .forward(0.9, 0.3).build("walk").with_head_scan(math.radians(40.0), 4.0)


def test_the_vision_arm_localizes_about_as_well_as_the_oracle_on_a_clean_walk(resources):
    scenario = scenarios.VisionScenario("m_walk", "A short walk", "Model test.", _walk)
    run = runner.run_vision_scenario(scenario, 0, resources, (
        runner.Arm("vision", "vision"), runner.Arm("oracle", "oracle"),
        runner.Arm("odometry", "odometry")))
    vision, oracle, odometry = (run.arms[name].result for name in ("vision", "oracle", "odometry"))
    assert vision.position_error[-30:].max() < 0.15 and vision.yaw_error_deg[-30:].max() < 4.0
    assert oracle.position_error[-30:].max() < 0.15
    assert vision.position_error[-30:].mean() < odometry.position_error[-30:].mean()
    assert vision.num_line_points.mean() > 20 and vision.filter_converged[-30:].all()
    timing = run.arms["vision"].timing_ms
    for key in ("detector", "segmenter", "geometry", "filter"):
        assert np.all(timing[key][5:] > 0.0), key
    np.testing.assert_allclose(
        timing["total"][5:], sum(timing[k][5:] for k in ("detector", "segmenter", "geometry",
                                                          "filter")), rtol=0.25)
    features = run.arms["vision"].features.summary()
    assert features["intersections"]["all"]["precision"] in (None, 1.0) \
        or features["intersections"]["all"]["precision"] > 0.9


def test_pipeline_loads_from_checkpoints_and_steps_at_full_camera_resolution(resources, s_field):
    """1280x720, the simulated camera's native size: the calibration must describe the image
    that is actually passed in."""
    from robosoccer_perception.geometry.camera import StereoCameraConfig

    settings = setup.load_settings()
    pipeline = VisionLocalizationPipeline.from_checkpoints(
        setup.checkpoint_path(settings["detector"]), setup.checkpoint_path(settings["segmenter"]))
    camera_cfg = StereoCameraConfig.load("zed_mini_sim")            # 1280x720
    calibration = gp.EstimatedCameraCalibration.from_camera_config(camera_cfg)
    true_pose = np.array([-2.0, -1.0, 0.4])
    with SimulatedCamera(s_field, camera_cfg) as sim:
        image = sim.capture(true_pose, TrueCameraModel(camera_cfg, MountError(),
                                                       HeadJitter.none()).sample(), (1.0, 0.5))
    assert image.shape == (720, 1280, 3)
    pipeline.init_gaussian(true_pose + [0.2, 0.2, 0.1])
    for step in range(6):
        out = pipeline.step(image, 0.1 * step, None, calibration, calibration.pose())
    assert np.linalg.norm(out.pose[:2] - true_pose[:2]) < 0.12
    assert out.vision.segmentation.mask.shape == (720, 1280)
    assert out.timing_ms["total"] < 500.0
    small = gp.EstimatedCameraCalibration.from_camera_config(camera_cfg.with_resolution(640, 360))
    with pytest.raises(ValueError, match="rescale the calibration"):
        pipeline.step(image, 1.0, None, small, small.pose())
