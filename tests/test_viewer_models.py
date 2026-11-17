"""The viewer with the TRAINED networks: three benchmark scenarios through the live session,
compared with the benchmark itself. Skipped when the checkpoints are missing."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.viewer import headless  # noqa: E402
from robosoccer_perception.viewer.recording import Recording  # noqa: E402
from robosoccer_perception.viewer.session import LiveSession  # noqa: E402
from robosoccer_perception.vision_benchmark import runner, setup  # noqa: E402
from robosoccer_perception.vision_benchmark.scenarios import get_scenario  # noqa: E402

pytestmark = [pytest.mark.vision, pytest.mark.render, pytest.mark.slow]
STEPS = 25


def _first_steps(scenario, steps):
    """The same scenario (same id, so the same random streams) cut to its first frames, so
    that the benchmark does not have to run all of it for the comparison."""
    def build(field, rng):
        full = scenario.build_trajectory(field, rng)
        return dataclasses.replace(
            full, poses=full.poses[:steps],
            pan=None if full.pan is None else full.pan[:steps],
            pitch=None if full.pitch is None else full.pitch[:steps],
            kidnap_steps=tuple(s for s in full.kidnap_steps if s < steps))
    return dataclasses.replace(scenario, build_trajectory=build)


@pytest.fixture(scope="module")
def resources(gl_context):
    settings = setup.load_settings()
    missing = setup.missing_checkpoints(settings)
    if missing:
        pytest.skip("trained checkpoints not found: " + ", ".join(str(p) for p in missing))
    return setup.load_resources(settings)


@pytest.mark.parametrize("scenario_id", ["v04_rectangle", "v11_goal_post_occlusion",
                                         "v13a_kidnapped_same_side"])
def test_a_session_with_the_trained_models_matches_the_benchmark(resources, scenario_id):
    scenario = get_scenario(scenario_id)
    with LiveSession(scenario, 0, resources, max_steps=STEPS) as session:
        frames = [session.step() for _ in range(STEPS)]
    arms = tuple(arm for arm in runner.DEFAULT_ARMS if arm.name == "vision")
    run = runner.run_vision_scenario(_first_steps(scenario, STEPS), 0, resources, arms,
                                     keep_frames=(STEPS - 1,), score_features=False)
    result = run.arms["vision"].result
    # The same images through the same networks on the same device: the same estimates.
    assert len(result.estimate) == STEPS
    np.testing.assert_allclose([f.pose for f in frames], result.estimate, atol=1e-9)
    np.testing.assert_array_equal(frames[-1].image, run.frames[STEPS - 1].image)
    np.testing.assert_array_equal(frames[-1].mask,
                                  run.frames[STEPS - 1].vision.segmentation.mask)
    assert all(f.image.shape == (360, 640, 3) for f in frames)
    assert np.mean([len(f.line_points) for f in frames]) > 15
    assert frames[-1].truth.position_error < 0.15 and frames[-1].converged
    assert all(0.0 < f.inference_ms < 2000.0 for f in frames)
    if scenario_id == "v11_goal_post_occlusion":
        classes = {d.class_id for f in frames for d in f.detections}
        assert schema.ROBOT in classes and schema.GOAL_POST in classes


def test_the_headless_demonstration_runs_end_to_end(resources, tmp_path):
    summary = headless.run_headless(
        resources, get_scenario("v04_rectangle"), 0, max_steps=STEPS,
        record=tmp_path / "demo.npz", video=tmp_path / "demo.mp4", screenshots=1,
        screenshot_dir=tmp_path)
    assert summary["frames"] == STEPS and summary["position_rmse_m"] < 0.2
    assert (tmp_path / "demo.mp4").stat().st_size > 10_000
    with Recording(summary["recording"]) as recording:
        frame = recording.frame(STEPS - 1)
        assert frame.step == STEPS - 1 and len(frame.detections) + len(frame.line_points) > 0
        assert recording.meta["models"]["device"] == recording.meta["device"]
