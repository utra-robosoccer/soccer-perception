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
