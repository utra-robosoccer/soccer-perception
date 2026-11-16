"""Shared fixtures. Every randomized test draws from a seeded generator."""
import numpy as np
import pytest

from robosoccer_perception.geometry.camera import StereoCameraConfig, StereoCameraModel
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import build_landmark_map

FIELD_NAMES = ("s_field", "m_field", "l_field")


@pytest.fixture(scope="session")
def s_field() -> FieldGeometry:
    return FieldGeometry.load("s_field")


@pytest.fixture(scope="session", params=FIELD_NAMES)
def any_field(request) -> FieldGeometry:
    return FieldGeometry.load(request.param)


@pytest.fixture(scope="session")
def s_landmarks(s_field):
    return build_landmark_map(s_field)


@pytest.fixture(scope="session")
def camera_model() -> StereoCameraModel:
    return StereoCameraModel(StereoCameraConfig.load("zed_mini_sim"))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


# ------------------------------------------------------------------ Phase 2 (vision) fixtures
TINY_DATASET = {
    "name": "tiny", "version": "test", "seed": 11, "field": "s_field", "camera": "zed_mini_sim",
    "resolution": [320, 180], "randomization": "mild",
    "splits": {"train": {"scenarios": 3, "frames_per_scenario": 4},
               "val": {"scenarios": 1, "frames_per_scenario": 4},
               "test": {"scenarios": 1, "frames_per_scenario": 4}},
}


@pytest.fixture(scope="session")
def gl_context():
    """Skip when MuJoCo cannot render offscreen here (no display / no OpenGL)."""
    pytest.importorskip("mujoco")
    from robosoccer_perception.simulation.mujoco_scene import LEFT_CAMERA, MujocoScene
    try:
        with MujocoScene(FieldGeometry.load("s_field")) as scene:
            scene.render(LEFT_CAMERA)
    except Exception as error:  # noqa: BLE001 - any context failure means "cannot render here"
        pytest.skip(f"MuJoCo offscreen rendering unavailable: {error!r}")


@pytest.fixture(scope="session")
def tiny_dataset_config():
    pytest.importorskip("cv2")
    from robosoccer_perception.datasets.generator import DatasetConfig
    return DatasetConfig.from_dict(TINY_DATASET)


@pytest.fixture(scope="session")
def tiny_dataset(gl_context, tiny_dataset_config, tmp_path_factory):
    """A 20-frame dataset at 320x180, generated once per test session."""
    from robosoccer_perception.datasets.generator import generate_dataset
    return generate_dataset(tiny_dataset_config, tmp_path_factory.mktemp("datasets") / "tiny")
