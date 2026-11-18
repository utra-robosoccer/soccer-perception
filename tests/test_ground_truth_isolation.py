"""Ground truth must not reach the vision-driven localizer. This file is the tripwire.

Three layers:

1. STRUCTURE. The pipeline's modules do not import the simulator, the dataset reader, the
   oracle or the benchmark; importing and running the pipeline works with MuJoCo blocked.
2. TYPES. `step` takes an image, a timestamp, an odometry delta and the robot's own camera
   estimate. Simulator objects of every kind are rejected, and the true camera is a different
   type that cannot stand in for an estimate.
3. BEHAVIOUR. Everything the benchmark hands to the pipeline during a real run is recorded and
   replayed into a fresh pipeline with no simulator in sight. If the replay reproduces the
   estimates exactly, nothing else can have influenced them. The recorded camera estimate is
   also checked to be the NOMINAL one while the simulator's camera is deliberately elsewhere.
"""
import dataclasses
import inspect
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.geometry import ground_projection as gp  # noqa: E402
from robosoccer_perception.geometry.camera import RigState, StereoCameraConfig  # noqa: E402
from robosoccer_perception.localization.motion_model import OdometryDelta  # noqa: E402
from robosoccer_perception.observations.types import (  # noqa: E402
    LandmarkObservation, LinePointsObservation, ObservationFrame)
from robosoccer_perception.perception import types as feature_types  # noqa: E402
from robosoccer_perception.pipeline.vision_localization import (  # noqa: E402
    VisionLocalizationInput, VisionLocalizationPipeline)

SOURCE = Path(__file__).resolve().parents[1] / "src" / "robosoccer_perception"
#: Modules that make up the runtime pipeline (everything between the image and the pose).
RUNTIME_MODULES = sorted([*(SOURCE / "perception").glob("*.py"), *(SOURCE / "pipeline").glob("*.py"),
                          SOURCE / "geometry" / "ground_projection.py",
                          # Phase 4: IMU orientation, synchronisation, camera pose, mount
                          # calibration monitoring and the geometry they stand on
                          *(SOURCE / "estimation").glob("*.py"),
                          SOURCE / "geometry" / "mount.py", SOURCE / "geometry" / "quaternion.py",
                          SOURCE / "geometry" / "robot_description.py"])
FORBIDDEN_IMPORTS = (
    "robosoccer_perception.simulation",         # the simulator and its ground truth
    "robosoccer_perception.vision_benchmark",   # the evaluator
    "robosoccer_perception.evaluation",         # Phase 1's evaluator
    "robosoccer_perception.observations.oracle", "robosoccer_perception.observations.noise",
    "robosoccer_perception.datasets.io", "robosoccer_perception.datasets.generator",
    "robosoccer_perception.datasets.annotation", "robosoccer_perception.datasets.sampling",
    "robosoccer_perception.datasets.randomization",
)


def _imports(path: Path) -> set[str]:
    return set(re.findall(r"^\s*(?:from|import)\s+([\w.]+)", path.read_text(), re.MULTILINE))


# ------------------------------------------------------------------ 1. structure
def test_runtime_modules_exist_and_are_scanned():
    names = {p.name for p in RUNTIME_MODULES}
    assert {"extractor.py", "line_features.py", "goal_features.py", "intersections.py",
            "line_fitting.py", "uncertainty.py", "observation_builder.py",
            "vision_localization.py", "ground_projection.py"} <= names
    assert {"orientation.py", "sync.py", "camera_pose.py", "calibration_monitor.py",
            "imu_types.py", "mount.py", "quaternion.py", "robot_description.py"} <= names


@pytest.mark.parametrize("path", RUNTIME_MODULES, ids=lambda p: p.name)
def test_runtime_modules_do_not_import_ground_truth_sources(path):
    for module in _imports(path):
        assert module != "mujoco" and not module.startswith("mujoco."), (path.name, module)
        assert not module.startswith(FORBIDDEN_IMPORTS), (path.name, module)
    text = path.read_text()
    for word in ("GroundTruth", "SimulationCameraTruth", "MujocoScene", "base_pose"):
        assert word not in text.replace("vision_benchmark.camera_truth.SimulationCameraTruth", ""), \
            (path.name, word)


def test_the_pipeline_imports_and_runs_with_mujoco_blocked():
    """A fresh interpreter in which `import mujoco` fails: the pipeline still localizes."""
    script = textwrap.dedent("""
        import sys
        sys.modules["mujoco"] = None                      # any import of it now raises
        sys.path.insert(0, sys.argv[1])
        import numpy as np
        import phase3_helpers as helpers
        from robosoccer_perception.geometry.field import FieldGeometry
        from robosoccer_perception.pipeline.vision_localization import VisionLocalizationPipeline

        field = FieldGeometry.load("s_field")
        calibration = helpers.small_calibration()
        image, _ = helpers.analytic_view(field, [-2.0, -1.0, 0.3], calibration, calibration.pose())
        pipeline = VisionLocalizationPipeline(helpers.ColourVision(), field, seed=0)
        pipeline.init_gaussian([-2.1, -0.9, 0.35])
        for step in range(5):
            out = pipeline.step(image, 0.1 * step, None, calibration, calibration.pose())
        loaded = [m for m in sys.modules if m.startswith("robosoccer_perception.")]
        bad = [m for m in loaded if ".simulation" in m or ".vision_benchmark" in m
               or m.endswith((".oracle", ".datasets.io", ".datasets.generator",
                              ".datasets.annotation"))]
        assert not bad, bad
        assert "torch" not in sys.modules and "ultralytics" not in sys.modules
        print("error %.3f" % np.linalg.norm(out.pose[:2] - [-2.0, -1.0]))
    """)
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).parent)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert float(result.stdout.split()[-1]) < 0.15


# ------------------------------------------------------------------ 2. types
def test_step_takes_exactly_the_permitted_inputs():
    parameters = list(inspect.signature(VisionLocalizationPipeline.step).parameters)
    assert parameters == ["self", "image", "timestamp", "odometry_delta", "camera_calibration",
                          "estimated_camera_pose", "imu_roll", "imu_pitch"]
    assert [f.name for f in dataclasses.fields(VisionLocalizationInput)] == [
        "image", "timestamp", "odometry_delta", "camera_calibration", "estimated_camera_pose",
        "imu_roll", "imu_pitch"]
    constructor = set(inspect.signature(VisionLocalizationPipeline.__init__).parameters)
    assert constructor == {"self", "vision", "field", "extraction", "noise", "selection",
                           "filter_config", "landmarks", "seed"}


def test_nothing_that_crosses_the_boundary_can_carry_a_global_pose_or_an_identity():
    assert {f.name for f in dataclasses.fields(OdometryDelta)} == \
        {"dx", "dy", "dyaw", "timestamp", "dt", "covariance"}
    assert {f.name for f in dataclasses.fields(gp.EstimatedCameraPose)} == {"T_base_camera"}
    assert {f.name for f in dataclasses.fields(ObservationFrame)} == \
        {"timestamp", "landmarks", "line_points", "frame_id"}
    assert {f.name for f in dataclasses.fields(LandmarkObservation)} == \
        {"type", "position", "range_std", "bearing_std", "confidence"}
    assert {f.name for f in dataclasses.fields(LinePointsObservation)} == {"points", "std"}
    for cls in (feature_types.FeatureSet, feature_types.LinePointFeatures,
                feature_types.GoalPostFeature, feature_types.IntersectionFeature,
                feature_types.CircleFeature, feature_types.LineSegmentFeature):
        for field in dataclasses.fields(cls):
            assert not re.search(r"field|global|truth|identity|landmark_id|world", field.name), \
                (cls.__name__, field.name)


def test_simulator_objects_are_rejected_at_the_boundary(s_field, camera_model):
    pytest.importorskip("mujoco")
    from robosoccer_perception.simulation.ground_truth import compute_ground_truth
    from robosoccer_perception.vision_benchmark.camera_truth import (
        SimulationCameraTruth, TrueCameraModel)

    calibration = helpers.small_calibration()
    pose = calibration.pose()
    image, _ = helpers.analytic_view(s_field, [0.0, 0.0, 0.0], calibration, pose)
    pipeline = VisionLocalizationPipeline(helpers.ColourVision(), s_field, seed=0)
    truth = compute_ground_truth(camera_model, [1.0, 2.0, 0.3])
    camera_truth = TrueCameraModel(StereoCameraConfig.load("zed_mini_sim")).sample()
    assert type(camera_truth) is SimulationCameraTruth
    for sneaky in (truth, truth.base_pose, truth.T_field_base, truth.T_field_left_optical,
                   truth.rig, camera_truth, camera_truth.true_rig,
                   camera_truth.true_T_base_camera(), RigState(0.9, 0.4)):
        with pytest.raises(TypeError):
            pipeline.step(image, 0.0, None, calibration, sneaky)
        with pytest.raises(TypeError):
            pipeline.step(image, 0.0, sneaky, calibration, pose)
        with pytest.raises(TypeError):
            pipeline.step(image, 0.0, None, sneaky, pose)
        with pytest.raises(TypeError):
            gp.project_to_ground([[10.0, 10.0]], calibration, sneaky)
    with pytest.raises(TypeError):
        pipeline.step(truth, 0.0, None, calibration, pose)
    assert pipeline.steps == 0


def test_true_and_estimated_camera_are_different_types_with_different_names():
    pytest.importorskip("mujoco")
    from robosoccer_perception.vision_benchmark import camera_truth

    truth_fields = {f.name for f in dataclasses.fields(camera_truth.SimulationCameraTruth)}
    estimate_fields = {f.name for f in dataclasses.fields(gp.EstimatedCameraPose)} \
        | {f.name for f in dataclasses.fields(gp.EstimatedCameraCalibration)}
    assert not truth_fields & estimate_fields, "no shared attribute to mix up"
    assert all(name.startswith("true_") or name == "baseline" for name in truth_fields)
    assert not issubclass(camera_truth.SimulationCameraTruth, gp.EstimatedCameraPose)
    assert not issubclass(gp.EstimatedCameraPose, camera_truth.SimulationCameraTruth)
    # Nothing on the truth side manufactures an estimate.
    source = Path(camera_truth.__file__).read_text()
    assert "EstimatedCameraPose(" not in source and "EstimatedCameraCalibration(" not in source


def test_the_true_camera_follows_commands_plus_hidden_errors():
    pytest.importorskip("mujoco")
    from robosoccer_perception.vision_benchmark.camera_truth import (
        HeadJitter, MountError, TrueCameraModel)

    cfg = StereoCameraConfig.load("zed_mini_sim")
    exact = TrueCameraModel(cfg, MountError(), HeadJitter.none()).sample(0.3, math.radians(40.0))
    assert exact.true_rig.pan == pytest.approx(0.3)
    assert exact.true_rig.pitch == pytest.approx(math.radians(40.0))
    assert exact.true_rig.height == pytest.approx(cfg.mount.height) and exact.true_rig.roll == 0.0
    off = TrueCameraModel(cfg, MountError(height=0.05, pitch_deg=2.0, roll_deg=-1.0, pan_deg=3.0),
                          HeadJitter.none()).sample(0.3)
    assert off.true_rig.height == pytest.approx(cfg.mount.height + 0.05)
    assert math.degrees(off.true_rig.pitch) == pytest.approx(27.0)
    assert math.degrees(off.true_rig.roll) == pytest.approx(-1.0)
    assert math.degrees(off.true_rig.pan) == pytest.approx(math.degrees(0.3) + 3.0)
    shaky = TrueCameraModel(cfg, MountError(), HeadJitter(0.25, 0.25), np.random.default_rng(0))
    samples = np.array([[math.degrees(s.true_rig.pitch), math.degrees(s.true_rig.roll)]
                        for s in (shaky.sample() for _ in range(2000))])
    np.testing.assert_allclose(samples.mean(axis=0), [25.0, 0.0], atol=0.03)
    np.testing.assert_allclose(samples.std(axis=0), [0.25, 0.25], atol=0.02)


# ------------------------------------------------------------------ 3. behaviour
@pytest.mark.render
@pytest.mark.slow
def test_only_images_odometry_and_the_robots_own_camera_estimate_reach_the_pipeline(
        gl_context, s_field, monkeypatch):
    """Record what the benchmark gives the vision arm, replay it without the benchmark."""
    from robosoccer_perception.simulation import trajectories as tj
    from robosoccer_perception.vision_benchmark import runner
    from robosoccer_perception.vision_benchmark.camera_truth import MountError
    from robosoccer_perception.vision_benchmark.scenarios import VisionScenario

    def walk(field, rng):
        return tj.PathBuilder([-2.4, -1.2, 0.2]).forward(0.9, 0.3).turn(math.radians(40.0)) \
            .build("walk").with_head_scan(math.radians(30.0), 2.0)

    scenario = VisionScenario("iso_walk", "isolation", "a short walk with a hidden mount error",
                              walk, mount_error=MountError(pitch_deg=2.0, height=0.04))
    resources = runner.BenchmarkResources(
        helpers.ColourVision(), resolution=(320, 180),
        filter_overrides={"num_particles": 300})
    arms = (runner.Arm("vision", "vision"), runner.Arm("oracle", "oracle"),
            runner.Arm("perfect_vision", "perfect_vision"))

    recorded = {"construct": [], "init": [], "step": [], "predictions": []}
    originals = {name: getattr(VisionLocalizationPipeline, name) for name in (
        "__init__", "init_gaussian", "init_uniform", "step", "step_with_predictions")}

    def construct(self, *args, **kwargs):
        recorded["construct"].append((self, args, kwargs))
        originals["__init__"](self, *args, **kwargs)

    def spy(name, bucket):
        def wrapper(self, *args, **kwargs):
            recorded[bucket].append((self, name, args, kwargs))
            return originals[name](self, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(VisionLocalizationPipeline, "__init__", construct)
    monkeypatch.setattr(VisionLocalizationPipeline, "init_gaussian", spy("init_gaussian", "init"))
    monkeypatch.setattr(VisionLocalizationPipeline, "init_uniform", spy("init_uniform", "init"))
    monkeypatch.setattr(VisionLocalizationPipeline, "step", spy("step", "step"))
    monkeypatch.setattr(VisionLocalizationPipeline, "step_with_predictions",
                        spy("step_with_predictions", "predictions"))
    run = runner.run_vision_scenario(scenario, 3, resources, arms)
    monkeypatch.undo()

    result = run.arms["vision"].result
    steps = len(result.time)
    vision_pipeline = recorded["step"][0][0]
    assert all(entry[0] is vision_pipeline for entry in recorded["step"])
    assert len(recorded["step"]) == steps, "the vision arm is driven through step() only"
    assert all(entry[0] is not vision_pipeline for entry in recorded["predictions"]), \
        "rendered labels never enter the vision arm"
    assert len(recorded["predictions"]) == steps        # the perfect-vision diagnostic arm

    # ---- what crossed the boundary, call by call
    trajectory = runner.build_trajectory(scenario, 3)
    nominal = gp.EstimatedCameraCalibration.from_camera_config(
        resources.camera_config(), uncertainty=resources.noise.calibration)
    for step, (_, _, args, kwargs) in enumerate(recorded["step"]):
        assert not kwargs and len(args) == 5
        image, timestamp, delta, calibration, camera_pose = args
        assert type(image) is np.ndarray and image.dtype == np.uint8 and image.shape == (180, 320, 3)
        assert isinstance(timestamp, float)
        assert delta is None or type(delta) is OdometryDelta
        assert type(calibration) is gp.EstimatedCameraCalibration
        assert type(camera_pose) is gp.EstimatedCameraPose
        # The estimate is the nominal mount with the COMMANDED head pan: 25 deg down, 0.90 m
        # up. The simulator's camera is 27 deg down and 0.94 m up, and shaking.
        assert calibration == nominal
        expected = nominal.pose(pan=float(trajectory.pan[step]))
        np.testing.assert_array_equal(camera_pose.T_base_camera, expected.T_base_camera)
        assert math.degrees(camera_pose.depression) == pytest.approx(25.0, abs=1e-9)
        assert camera_pose.height == pytest.approx(0.90)
    deltas = np.array([args[2].as_array() for _, _, args, _ in recorded["step"] if args[2]])
    assert np.abs(deltas[:, :2]).max() < 0.1 and np.abs(deltas[:, 2]).max() < 0.2, \
        "odometry is relative and small: it cannot encode where on the field the robot is"

    # ---- replay: a fresh pipeline, no simulator, no scenario, no trajectory
    (_, args, kwargs), = [c for c in recorded["construct"] if c[0] is vision_pipeline]
    replay = VisionLocalizationPipeline(*args, **kwargs)
    (_, name, init_args, init_kwargs), = [c for c in recorded["init"] if c[0] is vision_pipeline]
    getattr(replay, name)(*init_args, **init_kwargs)
    estimates = [replay.step(*args).pose for _, _, args, _ in recorded["step"]]
    np.testing.assert_array_equal(np.array(estimates), result.estimate)

    # The hidden mount error is visible in the result, as it must be if the true camera was
    # never used: the same run without it is more accurate.
    clean = dataclasses.replace(scenario, mount_error=MountError())
    exact = runner.run_vision_scenario(clean, 3, resources, arms[:1]).arms["vision"].result
    assert result.position_error[-10:].mean() > 1.5 * exact.position_error[-10:].mean()


@pytest.mark.render
def test_the_simulator_is_driven_by_truth_and_refuses_estimates(gl_context, s_field):
    from robosoccer_perception.vision_benchmark.camera_truth import TrueCameraModel
    from robosoccer_perception.vision_benchmark.sim_camera import SimulatedCamera

    cfg = StereoCameraConfig.load("zed_mini_sim").with_resolution(320, 180)
    calibration = gp.EstimatedCameraCalibration.from_camera_config(cfg)
    with SimulatedCamera(s_field, cfg, with_labels=True) as sim:
        with pytest.raises(TypeError):
            sim.capture([0.0, 0.0, 0.0], calibration.pose())
        image = sim.capture([-2.0, -1.0, 0.3], TrueCameraModel(cfg).sample(), (1.0, 0.5))
        assert image.shape == (180, 320, 3) and image.dtype == np.uint8
        labels = sim.labels_as_vision()
        assert labels.source == "ground_truth_labels", "labels are marked as what they are"
        assert labels.segmentation.mask.shape == (180, 320)
        assert all(d.confidence == 1.0 for d in labels.detections)
    plain = SimulatedCamera(s_field, cfg)
    with pytest.raises(RuntimeError):
        plain.labels_as_vision()
    plain.close()


# ------------------------------------------------------------------ Phase 4: the IMU chain
def test_orientation_estimation_runs_with_mujoco_and_the_simulator_blocked():
    """The tilt filter, the synchronisation and the camera-pose estimator in a fresh
    interpreter where neither MuJoCo nor the simulation package can be imported."""
    script = textwrap.dedent("""
        import sys
        sys.modules["mujoco"] = None
        sys.modules["robosoccer_perception.simulation"] = None
        import math
        import numpy as np
        from robosoccer_perception.estimation.camera_pose import CameraPoseEstimator, FusionConfig
        from robosoccer_perception.estimation.calibration_monitor import CalibrationMonitor
        from robosoccer_perception.estimation.imu_types import GRAVITY, ImuSample
        from robosoccer_perception.geometry import quaternion as quat
        from robosoccer_perception.geometry.camera import StereoCameraConfig
        from robosoccer_perception.geometry.field import FieldGeometry
        from robosoccer_perception.geometry.ground_projection import EstimatedCameraCalibration
        from robosoccer_perception.geometry.mount import CameraMount
        from robosoccer_perception.geometry.robot_description import RobotDescription

        body = RobotDescription.load("placeholder").body
        calibration = EstimatedCameraCalibration.from_camera_config(
            StereoCameraConfig.load("zed_mini_sim"))
        estimator = CameraPoseEstimator(calibration, CameraMount.load("legacy_rig"), body, 0.063,
                                        FusionConfig(use_uncertainty=True))
        up = quat.up_from_tilt(0.0, math.radians(5.0))
        for index in range(1200):
            estimator.add_imu(ImuSample(0.005 * index, np.zeros(3), GRAVITY * up))
        estimate = estimator.estimate(5.5)
        CalibrationMonitor(FieldGeometry.load("s_field"))
        bad = [m for m, loaded in sys.modules.items()
               if loaded is not None and m.startswith("robosoccer_perception.")
               and (".simulation" in m or ".vision_benchmark" in m or ".humanoid_benchmark" in m)]
        assert not bad, bad
        print("pitch %.3f" % math.degrees(estimate.body_pitch))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert float(result.stdout.split()[-1]) == pytest.approx(5.0, abs=0.1)


def test_estimators_take_sensor_types_only():
    from robosoccer_perception.estimation import camera_pose, imu_types, orientation, sync

    assert [f.name for f in dataclasses.fields(imu_types.ImuSample)] == [
        "timestamp", "gyro", "accel"]
    assert list(inspect.signature(orientation.TiltEstimator.update).parameters) == [
        "self", "sample"]
    assert list(inspect.signature(camera_pose.CameraPoseEstimator.add_imu).parameters) == [
        "self", "sample"]
    assert list(inspect.signature(camera_pose.CameraPoseEstimator.estimate).parameters) == [
        "self", "image_timestamp"]
    assert set(inspect.signature(camera_pose.CameraPoseEstimator.__init__).parameters) == {
        "self", "calibration", "mount", "body", "baseline", "cfg", "side"}
    for cls in (orientation.OrientationEstimate, sync.SyncedTilt, camera_pose.CameraPoseEstimate):
        for field in dataclasses.fields(cls):
            assert not re.search(r"truth|true|field|global|world", field.name), (cls, field.name)
    # the simulated IMU's truth arrays exist, and are not part of what a sample carries
    from robosoccer_perception.simulation.imu import ImuStream

    truth_fields = {f.name for f in dataclasses.fields(ImuStream) if "tru" in f.name}
    assert truth_fields == {"true_time", "truth_roll", "truth_pitch"}
    assert not truth_fields & {f.name for f in dataclasses.fields(imu_types.ImuSample)}
