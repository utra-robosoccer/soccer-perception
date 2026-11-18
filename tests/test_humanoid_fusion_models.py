"""Phase 4 with the TRAINED networks: localization from the humanoid's virtual camera while
the body sways, and the viewer session against the benchmark. Skipped when the checkpoints or
the imported humanoid model are missing."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from robosoccer_perception.geometry.robot_description import robot_is_installed  # noqa: E402
from robosoccer_perception.humanoid_benchmark import runner, scenarios  # noqa: E402
from robosoccer_perception.humanoid_benchmark import setup as fusion_setup  # noqa: E402
from robosoccer_perception.viewer.humanoid_session import (  # noqa: E402
    HumanoidLiveSession, SessionOptions)
from robosoccer_perception.vision_benchmark import setup  # noqa: E402

pytestmark = [pytest.mark.vision, pytest.mark.render, pytest.mark.slow,
              pytest.mark.skipif(not robot_is_installed("humanoid_prototype_1"),
                                 reason="humanoid model not imported")]
STEPS = 90


@pytest.fixture(scope="module")
def resources(gl_context):
    settings = fusion_setup.load_settings()
    missing = setup.missing_checkpoints(settings)
    if missing:
        pytest.skip("trained checkpoints not found: " + ", ".join(str(p) for p in missing))
    return setup.load_resources(settings)


@pytest.fixture(scope="module")
def sway(resources):
    return runner.run_humanoid_scenario(
        scenarios.get_scenario("h02_sway_pitch"), 0, resources,
        (runner.IMU_UNCERTAINTY, runner.FIXED, runner.ORACLE, runner.IMU), max_steps=STEPS)


def _rmse(values, skip=25) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(values, float)[skip:] ** 2)))


def test_trained_networks_localize_from_the_humanoids_camera_under_sway(sway):
    """The real YOLO11n and U-Net on images rendered from the humanoid's virtual head while
    its body pitches +-4 deg: with the IMU the pose holds, without it the pose is lost."""
    arms = sway.arms
    assert sway.true_camera_height.mean() == pytest.approx(0.80, abs=0.02)
    fixed, imu, aware, oracle = (_rmse(arms[name].result.position_error) for name in (
        "fixed", "imu", "imu_uncertainty", "oracle_orientation"))
    assert _rmse(arms["fixed"].camera_pitch_error_deg) > 2.0
    assert _rmse(arms["imu"].camera_pitch_error_deg) < 0.9
    assert imu < 0.25 and aware < 0.25 and oracle < 0.15
    assert fixed > 2.0 * max(imu, aware)
    assert np.nanmean(arms["imu"].projection_error[25:]) \
        < 0.4 * np.nanmean(arms["fixed"].projection_error[25:])
    timing = arms["imu_uncertainty"].timing_ms
    assert np.median(timing["orientation"]) < 5.0            # ms: the IMU chain is cheap
    assert np.median(timing["total"]) < 150.0


def test_viewer_session_with_trained_networks_matches_the_benchmark(resources, sway):
    scenario = scenarios.get_scenario("h02_sway_pitch")
    with HumanoidLiveSession(scenario, 0, resources, max_steps=30,
                             options=SessionOptions(fusion="imu_uncertainty")) as session:
        frames = [session.step() for _ in range(30)]
    assert np.array_equal(np.array([f.pose for f in frames]),
                          sway.arms["imu_uncertainty"].result.estimate[:30])
    assert frames[-1].robot_view is not None and frames[-1].fusion.source == "imu"
    assert frames[-1].image.shape == (360, 640, 3)
