"""The debugging viewer without its window: frame packets, recordings, the four panels, the
step-by-step session and the worker thread. The window is tested in test_viewer_gui.py.

The session tests are the ones that matter for trust: a viewer session reproduces the
benchmark's vision arm bit for bit, the image in a frame is the image that frame's estimate was
computed from, and the estimator is handed sensor data only.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mujoco")

import cv2  # noqa: E402
import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.geometry.field import FieldGeometry  # noqa: E402
from robosoccer_perception.geometry.ground_projection import (  # noqa: E402
    EstimatedCameraCalibration, EstimatedCameraPose)
from robosoccer_perception.inference.types import Detection  # noqa: E402
from robosoccer_perception.localization.motion_model import OdometryDelta  # noqa: E402
from robosoccer_perception.simulation import trajectories as tj  # noqa: E402
from robosoccer_perception.viewer import headless, render  # noqa: E402
from robosoccer_perception.viewer import recording as recording_module  # noqa: E402
from robosoccer_perception.viewer.frame import (  # noqa: E402
    LANDMARK_TYPES, UNCLASSED, Timeline, TruthView, ViewerFrame, pack, unpack)
from robosoccer_perception.viewer.recording import Recording, save_recording  # noqa: E402
from robosoccer_perception.viewer.render import Layers, PanelRenderer, mosaic  # noqa: E402
from robosoccer_perception.viewer.session import LiveSession, _Sensors  # noqa: E402
from robosoccer_perception.viewer.worker import SimulationWorker  # noqa: E402
from robosoccer_perception.vision_benchmark import runner, scenarios  # noqa: E402

META = {"scenario": "t_fake", "title": "A made-up run", "description": "", "seed": 3,
        "field": "s_field", "dt": 0.1, "steps": 6, "resolution": [320, 180],
        "kidnap_steps": [], "device": "cpu", "has_truth": True}


def fake_frame(step: int = 2, truth: bool = True) -> ViewerFrame:
    """A frame with something in every layer, made without the simulator or any model."""
    rng = np.random.default_rng(step)
    image = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    mask = np.zeros((180, 320), np.uint8)
    mask[60:] = schema.FIELD
    mask[100:104, 20:300] = schema.FIELD_LINE
    angles = np.linspace(0.0, 2.0 * math.pi, 64)
    outline = np.array([[160.0 + 60 * np.cos(angles), 120.0 + 25 * np.sin(angles)]]) \
        .transpose(0, 2, 1)
    pose = np.array([-1.0 + 0.1 * step, 0.5, 0.3])
    particles = (pose + rng.normal(size=(200, 3)) * [0.3, 0.3, 0.2]).astype(np.float32)
    return ViewerFrame(
        step=step, timestamp=0.1 * step, image=image, mask=mask,
        detections=(Detection(schema.BALL, 0.91, (40.0, 90.0, 60.0, 110.0)),
                    Detection(schema.GOAL_POST, 0.83, (200.0, 20.0, 212.0, 100.0))),
        line_pixels=np.array([[30.0, 101.5], [120.0, 101.5], [250.0, 101.5]]),
        line_points=np.array([[1.0, 0.8], [1.5, 0.0], [2.0, -0.9]]),
        line_std=np.array([0.03, 0.04, 0.06]),
        segment_pixels=np.array([[[30.0, 101.5], [250.0, 101.5]]]),
        segments=np.array([[[1.0, 0.8], [2.0, -0.9]]]),
        junction_pixels=np.array([[120.0, 101.5], [60.0, 140.0]]),
        junctions=np.array([[1.5, 0.0], [1.0, 0.5]]),
        junction_codes=np.array([1, UNCLASSED]),
        circle_centers=np.array([[2.0, 0.2]]), circle_center_pixels=np.array([[160.0, 120.0]]),
        circle_valid=np.array([True]), circle_radius=0.7, circle_outlines=outline,
        post_pixels=np.array([[206.0, 99.5], [280.0, 90.0]]),
        posts=np.array([[3.0, -0.6], [3.2, -1.4]]),
        post_range_valid=np.array([True, False]),
        landmarks=np.array([[3.0, -0.6], [1.5, 0.0]]),
        landmark_codes=np.array([LANDMARK_TYPES.index(lt) for lt in LANDMARK_TYPES[3:4]] + [1]),
        observed_line_points=3, pose=pose, covariance=np.diag([0.04, 0.01, 0.002]),
        position_std=0.22, yaw_std_deg=2.6, confidence=0.7, mode_weight=0.6, converged=False,
        ambiguous=True,
        modes=np.array([[*pose, 0.6], [-pose[0], -pose[1], pose[2] + math.pi, 0.4]]),
        ess_fraction=0.55, resampled=True, injected=3, particles=particles,
        particle_weights=np.full(200, 1 / 200, np.float32), num_particles=1000,
        odometry_lost=False,
        timing_ms={"detector": 8.0, "segmenter": 21.0, "geometry": 10.0, "filter": 2.5,
                   "total": 42.0, "render": 7.0},
        sim_fps=12.5,
        truth=TruthView(pose + [0.15, -0.1, 0.05], 0.18, 2.9, 2.4, 177.1, step == 3)
        if truth else None)


def fake_timeline(steps: int = 6, upto: int = 4) -> Timeline:
    timeline = Timeline(steps, 0.1)
    for step in range(upto + 1):
        timeline.record(fake_frame(step))
    return timeline


# ------------------------------------------------------------------ frame packets
def test_a_packed_frame_unpacks_to_the_same_frame():
    frame = fake_frame()
    restored = unpack(pack(frame))
    for field in dataclasses.fields(ViewerFrame):
        before, after = getattr(frame, field.name), getattr(restored, field.name)
        if field.name == "image":
            # stored as JPEG: close, not identical
            assert after.shape == before.shape and after.dtype == np.uint8
        elif field.name == "truth":
            np.testing.assert_allclose(after.pose, before.pose)
            assert (after.position_error, after.yaw_error_deg, after.kidnapped) == \
                (before.position_error, before.yaw_error_deg, before.kidnapped)
        elif field.name == "detections":
            assert [d.class_id for d in after] == [d.class_id for d in before]
            np.testing.assert_allclose([d.bbox_xyxy for d in after],
                                       [d.bbox_xyxy for d in before])
            np.testing.assert_allclose([d.confidence for d in after],
                                       [d.confidence for d in before])
        elif isinstance(before, np.ndarray):
            np.testing.assert_allclose(after, before, rtol=1e-6)
            assert after.dtype.kind == before.dtype.kind, field.name
        elif isinstance(before, dict):
            assert after == pytest.approx(before)
        else:
            assert after == pytest.approx(before) and type(after) is type(before), field.name
    np.testing.assert_array_equal(restored.mask, frame.mask)        # PNG is lossless


def test_a_smooth_image_survives_jpeg_closely():
    frame = fake_frame()
    smooth = cv2.GaussianBlur(frame.image, (0, 0), 6)
    restored = unpack(pack(dataclasses.replace(frame, image=smooth)))
    assert np.abs(restored.image.astype(int) - smooth.astype(int)).mean() < 2.0


def test_a_frame_without_truth_packs_without_truth():
    packed = pack(fake_frame(truth=False))
    assert "truth" not in packed and unpack(packed).truth is None
    assert fake_frame().without_truth().truth is None


def test_an_empty_frame_packs_too():
    frame = fake_frame()
    empty = dataclasses.replace(
        frame, detections=(), line_pixels=np.zeros((0, 2)), line_points=np.zeros((0, 2)),
        line_std=np.zeros(0), segment_pixels=np.zeros((0, 2, 2)), segments=np.zeros((0, 2, 2)),
        junction_pixels=np.zeros((0, 2)), junctions=np.zeros((0, 2)),
        junction_codes=np.zeros(0, int), circle_centers=np.zeros((0, 2)),
        circle_center_pixels=np.zeros((0, 2)), circle_valid=np.zeros(0, bool),
        circle_outlines=np.zeros((0, 64, 2)), post_pixels=np.zeros((0, 2)),
        posts=np.zeros((0, 2)), post_range_valid=np.zeros(0, bool),
        landmarks=np.zeros((0, 2)), landmark_codes=np.zeros(0, int))
    restored = unpack(pack(empty))
    assert restored.detections == () and len(restored.line_points) == 0
    panels = PanelRenderer(FieldGeometry.load("s_field"), META).panels(
        restored, Layers(segments=True), fake_timeline())
    assert set(panels) == {"camera", "segmentation", "map", "diagnostics"}


def test_the_timeline_keeps_one_value_per_step():
    timeline = fake_timeline(6, upto=3)
    assert timeline.last == 3 and timeline.steps == 6
    assert np.isfinite(timeline.values["position_error"][:4]).all()
    assert np.isnan(timeline.values["position_error"][4:]).all()
    assert timeline.values["kidnapped"][3] == 1.0
    np.testing.assert_allclose(timeline.time, 0.1 * np.arange(6))
    restored = Timeline.from_arrays(timeline.to_arrays(), 0.1)
    assert restored.last == 3
    np.testing.assert_array_equal(restored.values["est_x"], timeline.values["est_x"])
    timeline.record(dataclasses.replace(fake_frame(), step=99))     # out of range: ignored
    timeline.clear()
    assert timeline.last == -1 and np.isnan(timeline.values["confidence"]).all()


# ------------------------------------------------------------------ recordings
def test_a_recording_replays_what_was_saved(tmp_path):
    frames = [fake_frame(step) for step in range(5)]
    timeline = fake_timeline(6, upto=4)
    path = save_recording(tmp_path / "run", META, [pack(f) for f in frames], timeline)
    assert path.suffix == ".npz" and path.is_file()
    with Recording(path) as recording:
        assert len(recording) == 5
        assert recording.meta["scenario"] == "t_fake" and recording.meta["frames"] == 5
        for index in (4, 0, 2):                                     # any order: lazy access
            frame = recording.frame(index)
            assert frame.step == index
            np.testing.assert_allclose(frame.pose, frames[index].pose)
            np.testing.assert_array_equal(frame.mask, frames[index].mask)
            np.testing.assert_allclose(frame.truth.pose, frames[index].truth.pose)
        np.testing.assert_array_equal(recording.timeline.values["position_error"],
                                      timeline.values["position_error"])
        assert recording.timeline.last == 4


def test_other_files_are_not_mistaken_for_recordings(tmp_path):
    np.savez(tmp_path / "other.npz", values=np.arange(3))
    with pytest.raises(ValueError, match="not a viewer recording"):
        Recording(tmp_path / "other.npz")


def test_generated_files_go_to_the_ignored_outputs_folder():
    path = recording_module.default_path("recordings", META, ".npz")
    assert path.parent == recording_module.REPOSITORY_ROOT / "outputs" / "viewer" / "recordings"
    assert path.name.startswith("t_fake_seed3_")
    ignored = (recording_module.REPOSITORY_ROOT / ".gitignore").read_text().split()
    assert "outputs/" in ignored and "*.npz" in ignored and "*.mp4" in ignored


# ------------------------------------------------------------------ the panels
@pytest.fixture(scope="module")
def renderer():
    return PanelRenderer(FieldGeometry.load("s_field"), META, (640, 360))


def test_panels_have_one_size_and_a_title_bar(renderer):
    panels = renderer.panels(fake_frame(), Layers(), fake_timeline(), "TEST")
    assert list(panels) == ["camera", "segmentation", "map", "diagnostics"]
    for image in panels.values():
        assert image.shape == (360 + render.TITLE_BAR, 640, 3) and image.dtype == np.uint8
    picture = mosaic(panels)
    assert picture.shape[0] == 2 * panels["camera"].shape[0] + 6
    assert picture.shape[1] == 2 * 640 + 6


@pytest.mark.parametrize("attribute,panel", [(a, p) for a, p, _ in Layers.CATALOGUE])
def test_every_toggle_changes_its_own_panel(renderer, attribute, panel):
    frame, timeline = fake_frame(), fake_timeline()
    on = Layers(**{field.name: True for field in dataclasses.fields(Layers)})
    off = dataclasses.replace(on, **{attribute: False})
    with_layer = renderer.panels(frame, on, timeline)
    without = renderer.panels(frame, off, timeline)
    own = panel.lower()
    assert not np.array_equal(with_layer[own], without[own]), f"{attribute} draws nothing"
    for name in with_layer:
        if name != own and not (attribute == "ground_truth" and name == "diagnostics"):
            np.testing.assert_array_equal(with_layer[name], without[name])


def test_hiding_ground_truth_leaves_no_trace_of_it(renderer):
    """With the layer off, the picture is the one of a frame that never had a truth."""
    frame, timeline = fake_frame(), fake_timeline()
    hidden = renderer.panels(frame, Layers(ground_truth=False), timeline)
    never_had = renderer.panels(frame.without_truth(), Layers(ground_truth=True), timeline)
    for name in hidden:
        np.testing.assert_array_equal(hidden[name], never_had[name])
    shown = renderer.panels(frame, Layers(ground_truth=True), timeline)
    assert not np.array_equal(shown["map"], hidden["map"])
    assert not np.array_equal(shown["diagnostics"], hidden["diagnostics"])


def test_the_map_says_ambiguous_and_shows_the_other_hypothesis(renderer):
    frame, timeline = fake_frame(), fake_timeline()
    ambiguous = renderer.map(frame, Layers(), timeline)
    single = renderer.map(dataclasses.replace(frame, ambiguous=False, converged=True,
                                              modes=frame.modes[:1]), Layers(), timeline)
    assert not np.array_equal(ambiguous, single)
    bar = slice(0, render.TITLE_BAR)
    warning = (ambiguous[bar] == render.WARNING).all(axis=2).sum()
    good = (single[bar] == render.GOOD).all(axis=2).sum()
    assert warning > 20 and good > 20, "the state is written in the title bar"
    assert (single[bar] == render.WARNING).all(axis=2).sum() == 0


def test_diagnostics_can_stop_at_the_cursor(renderer):
    timeline = fake_timeline(6, upto=5)
    frame = fake_frame(2)
    everything = renderer.diagnostics(frame, Layers(), timeline)
    upto_cursor = renderer.diagnostics(frame, Layers(), timeline, reveal_all=False)
    assert not np.array_equal(everything, upto_cursor)
    at_end = fake_frame(5)
    np.testing.assert_array_equal(
        renderer.diagnostics(at_end, Layers(), timeline),
        renderer.diagnostics(at_end, Layers(), timeline, reveal_all=False))


def test_panels_adapt_to_other_sizes_and_image_shapes():
    field = FieldGeometry.load("s_field")
    frame, timeline = fake_frame(), fake_timeline()
    for size in ((520, 300), (661, 403), (960, 540)):
        panels = PanelRenderer(field, META, size).panels(frame, Layers(segments=True), timeline)
        assert all(p.shape == (size[1] + render.TITLE_BAR, size[0], 3)
                   for p in panels.values())
    square = dataclasses.replace(frame, image=np.zeros((200, 200, 3), np.uint8),
                                 mask=np.zeros((200, 200), np.uint8))
    assert PanelRenderer(field, META, (640, 360)).camera(square, Layers()).shape[1] == 640


def test_nice_axis_limits():
    assert render._nice_ceiling(0.31) == 0.5 and render._nice_ceiling(6.0) == 10.0
    assert render._nice_ceiling(42.0) == 50.0 and render._nice_ceiling(0.0) == 1.0
    assert render._nice_ceiling(1.0) == 1.0 and render._nice_ceiling(2.4) == 2.5


def test_video_export_writes_every_frame(tmp_path, renderer):
    frames = [fake_frame(step) for step in range(5)]
    seen = []
    result = headless.export_video(tmp_path / "clip.mp4", renderer, iter(frames),
                                   fake_timeline(6, 4), Layers(), fps=10.0,
                                   progress=lambda done: seen.append(done))
    assert result["frames"] == 5 and seen == [1, 2, 3, 4, 5] and not result["cancelled"]
    capture = cv2.VideoCapture(str(tmp_path / "clip.mp4"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        ok, picture = capture.read()
        assert ok and picture.shape[:2] == (result["size"][1], result["size"][0])
    finally:
        capture.release()
    cancelled = headless.export_video(tmp_path / "short.mp4", renderer, iter(frames),
                                      fake_timeline(6, 4), Layers(), fps=10.0,
                                      progress=lambda done: done < 2)
    assert cancelled["cancelled"] and cancelled["frames"] == 2
    with pytest.raises(ValueError, match="no frames"):
        headless.export_video(tmp_path / "none.mp4", renderer, iter(()), fake_timeline(),
                              Layers(), fps=10.0)


def test_video_falls_back_to_opencv_without_ffmpeg(tmp_path):
    with headless.VideoSink(tmp_path / "plain.mp4", (64, 48), 10.0,
                            prefer_ffmpeg=False) as sink:
        for _ in range(3):
            sink.write(np.zeros((48, 64, 3), np.uint8))
    assert sink.encoder == "OpenCV mp4v" and (tmp_path / "plain.mp4").stat().st_size > 0


# ------------------------------------------------------------------ the live session
def _walk(field, rng):
    return tj.PathBuilder([-2.4, -1.2, 0.2]).forward(1.2, 0.3).turn(math.radians(50.0)) \
        .forward(0.6, 0.3).build("walk").with_head_scan(math.radians(30.0), 3.0)


WALK = scenarios.VisionScenario("t_viewer_walk", "A short walk", "Test scenario.", _walk)
VISION_ARM = tuple(arm for arm in runner.DEFAULT_ARMS if arm.name == "vision")


@pytest.fixture(scope="module")
def resources():
    return runner.BenchmarkResources(helpers.ColourVision(), resolution=(320, 180),
                                     filter_overrides={"num_particles": 300})


@pytest.fixture(scope="module")
def walk_frames(gl_context, resources):
    with LiveSession(WALK, 0, resources) as session:
        frames = []
        while not session.finished:
            frames.append(session.step())
        assert session.step() is None, "past the end there is nothing more"
        return frames, session.meta()


@pytest.mark.render
@pytest.mark.slow
def test_a_session_reproduces_the_benchmark_vision_arm_bit_for_bit(walk_frames, resources):
    """Same scenario, seed and settings: the viewer shows exactly the run the benchmark
    scores, not a look-alike."""
    frames, meta = walk_frames
    kept = (0, 7, len(frames) - 1)
    run = runner.run_vision_scenario(WALK, 0, resources, VISION_ARM, keep_frames=kept,
                                     score_features=False)
    result = run.arms["vision"].result
    assert len(frames) == len(result.time) == meta["steps"]
    np.testing.assert_array_equal([f.pose for f in frames], result.estimate)
    np.testing.assert_array_equal([f.timestamp for f in frames], result.time)
    np.testing.assert_array_equal([f.truth.pose for f in frames], result.truth)
    np.testing.assert_array_equal([f.truth.position_error for f in frames],
                                  result.position_error)
    np.testing.assert_array_equal([f.truth.yaw_error_deg for f in frames],
                                  result.yaw_error_deg)
    np.testing.assert_array_equal([f.converged for f in frames], result.filter_converged)
    np.testing.assert_array_equal([f.ambiguous for f in frames], result.symmetric_ambiguity)
    np.testing.assert_allclose([f.ess_fraction for f in frames], result.ess_fraction)
    # ... and frame by frame the picture is the one that estimate was computed from.
    for step in kept:
        record, frame = run.frames[step], frames[step]
        np.testing.assert_array_equal(frame.image, record.image)
        np.testing.assert_array_equal(frame.mask, record.vision.segmentation.mask)
        np.testing.assert_array_equal(frame.pose, record.estimate_pose)
        np.testing.assert_array_equal(frame.line_points, record.features.line_points.points)
        assert frame.step == step and frame.timestamp == pytest.approx(step * meta["dt"])


@pytest.mark.render
@pytest.mark.slow
def test_frames_are_complete_and_consistent(walk_frames):
    frames, meta = walk_frames
    assert [f.step for f in frames] == list(range(len(frames)))
    assert meta["scenario"] == "t_viewer_walk" and meta["resolution"] == [320, 180]
    for frame in frames:
        assert frame.image.shape == (180, 320, 3) and frame.mask.shape == (180, 320)
        assert len(frame.line_pixels) == len(frame.line_points) == len(frame.line_std)
        assert len(frame.junctions) == len(frame.junction_pixels) == len(frame.junction_codes)
        assert len(frame.particles) == len(frame.particle_weights) == 300
        assert frame.particle_weights.sum() == pytest.approx(1.0, abs=1e-4)
        assert 0.0 < frame.ess_fraction <= 1.0 and 0.0 <= frame.confidence <= 1.0
        assert set(frame.timing_ms) == {"detector", "segmenter", "geometry", "filter",
                                        "total", "render"}
        # extracted pixels are the features' own pixels: they lie on the line class
        if len(frame.line_pixels):
            u, v = np.round(frame.line_pixels).astype(int).T
            assert (frame.mask[v, u] == schema.FIELD_LINE).mean() > 0.9
    assert sum(len(f.line_points) for f in frames) > 10 * len(frames)
    assert frames[-1].truth.position_error < 0.25
    assert frames[-1].sim_fps > 0.0


class _Spy:
    """Stands where the pipeline's `step` was and writes down everything it is given."""

    def __init__(self, step):
        self._step, self.calls = step, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._step(*args, **kwargs)


@pytest.mark.render
@pytest.mark.slow
def test_the_estimator_is_given_sensor_data_and_nothing_else(gl_context, resources,
                                                             walk_frames):
    """Everything that reaches `VisionLocalizationPipeline.step` in a viewer session, by type
    and by value: an image, a time, an odometry increment and the robot's own camera estimate.
    The true pose is in none of them, and showing or hiding ground truth changes nothing."""
    reference, _ = walk_frames
    assert {f.name for f in dataclasses.fields(_Sensors)} == {
        "image", "timestamp", "odometry", "commanded_pan", "commanded_pitch"}
    with LiveSession(WALK, 0, resources, with_truth=False) as session:
        spy = _Spy(session._pipeline.step)
        session._pipeline.step = spy
        blind = [session.step() for _ in range(12)]
        nominal = session._calibration
    assert all(frame.truth is None for frame in blind), "no truth was put in the frames"
    np.testing.assert_array_equal([f.pose for f in blind], [f.pose for f in reference[:12]])
    assert len(spy.calls) == 12
    for index, (args, kwargs) in enumerate(spy.calls):
        image, timestamp, delta, calibration, camera_pose = args
        assert kwargs == {}
        assert type(image) is np.ndarray and image.dtype == np.uint8 and image.ndim == 3
        assert isinstance(timestamp, float)
        assert delta is None or type(delta) is OdometryDelta
        assert (delta is None) == (index == 0)
        assert type(calibration) is EstimatedCameraCalibration and calibration is nominal
        assert type(camera_pose) is EstimatedCameraPose
        np.testing.assert_array_equal(image, reference[index].image)
        # numbers that could smuggle a field-frame pose: there are none of that size
        truth = reference[index].truth.pose
        if delta is not None:
            assert abs(delta.dx) < 0.2 and abs(delta.dy) < 0.2, "an increment, not a pose"
            assert not np.allclose([delta.dx, delta.dy], truth[:2], atol=1e-3)
        assert camera_pose.height == pytest.approx(calibration.nominal_height)


@pytest.mark.render
@pytest.mark.slow
def test_reset_repeats_the_run_exactly(gl_context, resources, walk_frames):
    reference, _ = walk_frames
    with LiveSession(WALK, 0, resources, max_steps=9) as session:
        assert session.total_steps == 9
        first = [session.step() for _ in range(6)]
        assert session.next_step == 6 and not session.finished
        session.reset()
        assert session.next_step == 0
        again = [session.step() for _ in range(9)]
        assert session.finished and session.step() is None
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a.pose, b.pose)
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.particles, b.particles)
    np.testing.assert_array_equal([f.pose for f in again], [f.pose for f in reference[:9]])
    with pytest.raises(RuntimeError, match="closed"):
        session.step()


@pytest.mark.render
@pytest.mark.slow
def test_another_seed_is_another_run(gl_context, resources, walk_frames):
    reference, _ = walk_frames
    with LiveSession(WALK, 1, resources, max_steps=3) as session:
        other = [session.step() for _ in range(3)]
    assert not np.array_equal(other[0].pose, reference[0].pose)
    assert not np.array_equal(other[2].truth.pose, reference[2].truth.pose)


@pytest.mark.render
@pytest.mark.slow
def test_a_headless_run_writes_a_recording_a_video_and_pictures(gl_context, resources,
                                                                walk_frames, tmp_path):
    reference, _ = walk_frames
    progress = []
    summary = headless.run_headless(
        resources, WALK, 0, max_steps=12, record=tmp_path / "walk.npz",
        video=tmp_path / "walk.mp4", screenshots=2, screenshot_dir=tmp_path / "shots",
        progress=lambda done, total: progress.append((done, total)))
    assert summary["frames"] == 12 and progress[-1] == (12, 12)
    assert summary["position_rmse_m"] < 0.5 and summary["pipeline_ms_mean"] > 0.0
    assert len(summary["screenshots"]) == 2 and summary["video"]["frames"] == 12
    assert all((tmp_path / "shots").glob("t_viewer_walk_seed0_step*.png"))
    assert (tmp_path / "walk.mp4").stat().st_size > 1000
    with Recording(summary["recording"]) as recording:
        assert len(recording) == 12 and recording.meta["scenario"] == "t_viewer_walk"
        for index in (0, 5, 11):
            np.testing.assert_allclose(recording.frame(index).pose, reference[index].pose)
            np.testing.assert_array_equal(recording.frame(index).mask, reference[index].mask)
        assert recording.timeline.last == 11


# ------------------------------------------------------------------ the worker thread
class _Collector:
    """What a window would do with the worker's callbacks, minus the window."""

    def __init__(self, acknowledge: bool = True):
        self.frames: list[tuple[int, ViewerFrame]] = []
        self.events: list[tuple[str, object]] = []
        self.acknowledge = acknowledge
        self.worker: SimulationWorker | None = None
        self._lock = threading.Lock()

    def on_frame(self, run_id, frame, packed):
        assert set(packed) >= {"image_jpg", "mask_png", "scalars"}
        with self._lock:
            self.frames.append((run_id, frame))
        if self.acknowledge:
            self.worker.ack()

    def on_event(self, kind, payload):
        with self._lock:
            self.events.append((kind, payload))

    def steps(self, run_id):
        with self._lock:
            return [frame.step for rid, frame in self.frames if rid == run_id]

    def kinds(self):
        with self._lock:
            return [kind for kind, _ in self.events]

    def wait(self, condition, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.01)
        return False


def _start_worker(resources, **options):
    collector = _Collector(options.pop("acknowledge", True))
    worker = SimulationWorker(lambda: resources, collector.on_frame, collector.on_event,
                              **options)
    collector.worker = worker
    worker.start()
    return worker, collector


@pytest.mark.render
@pytest.mark.slow
def test_the_worker_steps_runs_pauses_and_resets(gl_context, resources):
    worker, seen = _start_worker(resources, max_steps=40)
    try:
        run = worker.load("v03_straight_line", 0)
        assert seen.wait(lambda: seen.steps(run) == [0]), "the first frame comes on its own"
        assert "loaded" in seen.kinds()
        time.sleep(0.2)
        assert seen.steps(run) == [0], "a loaded scenario waits: it does not start running"

        for expected in (1, 2, 3):                              # stepping: one frame each
            worker.step_once()
            assert seen.wait(lambda: seen.steps(run)[-1] == expected)
        time.sleep(0.2)
        assert seen.steps(run) == [0, 1, 2, 3]

        worker.set_speed(math.inf)                              # running
        worker.resume()
        assert seen.wait(lambda: len(seen.steps(run)) >= 15)
        worker.pause()                                          # pausing
        assert seen.wait(lambda: seen.events[-1] == ("running", False))
        paused_at = len(seen.steps(run))
        time.sleep(0.3)
        assert len(seen.steps(run)) == paused_at, "nothing is produced while paused"
        assert seen.steps(run) == list(range(paused_at)), "no frame skipped or repeated"
        first_pose = seen.frames[0][1].pose.copy()

        again = worker.reset()                                  # resetting
        assert again != run
        assert seen.wait(lambda: seen.steps(again) == [0])
        np.testing.assert_array_equal(
            [f.pose for rid, f in seen.frames if rid == again][0], first_pose)

        worker.resume()                                         # to the end
        assert seen.wait(lambda: ("finished", again) in seen.events, 60.0)
        assert seen.steps(again) == list(range(40))
        worker.resume()                                         # nothing left: says so again
        assert seen.wait(lambda: seen.events.count(("finished", again)) >= 2)
        assert seen.steps(again) == list(range(40))
        assert "error" not in seen.kinds()
    finally:
        worker.shutdown()
    assert not worker.is_alive()


@pytest.mark.render
@pytest.mark.slow
def test_the_worker_waits_for_a_slow_consumer(gl_context, resources):
    """Frames nobody has taken yet are not piled up: the simulation waits instead."""
    worker, seen = _start_worker(resources, max_steps=40, backlog=3, acknowledge=False)
    try:
        run = worker.load("v03_straight_line", 0)
        worker.set_speed(math.inf)
        worker.resume()
        assert seen.wait(lambda: len(seen.steps(run)) == 3)
        time.sleep(0.4)
        assert len(seen.steps(run)) == 3, "three frames out, none acknowledged: it waits"
        worker.pause()                                          # still obeys commands
        assert seen.wait(lambda: ("running", False) in seen.events)
        for _ in range(3):
            worker.ack()
        worker.resume()
        assert seen.wait(lambda: len(seen.steps(run)) == 6)
        assert seen.steps(run) == [0, 1, 2, 3, 4, 5]
    finally:
        worker.shutdown()


@pytest.mark.render
@pytest.mark.slow
def test_playback_speed_sets_the_pace(gl_context, resources):
    worker, seen = _start_worker(resources, max_steps=200)
    try:
        run = worker.load("v03_straight_line", 0)
        assert seen.wait(lambda: seen.steps(run) == [0])

        def frames_in(seconds, speed):
            worker.set_speed(speed)
            before = len(seen.steps(run))
            worker.resume()
            time.sleep(seconds)
            worker.pause()
            seen.wait(lambda: seen.events[-1] == ("running", False))
            return len(seen.steps(run)) - before

        slow = frames_in(1.5, 0.5)                              # 5 frames per second
        normal = frames_in(1.5, 1.0)                            # 10 frames per second
        assert 5 <= slow <= 10 and 11 <= normal <= 18
        assert frames_in(1.0, math.inf) > 20                    # as fast as it can
    finally:
        worker.shutdown()


@pytest.mark.render
@pytest.mark.slow
def test_a_failure_in_the_worker_is_reported_not_swallowed(gl_context, resources):
    worker, seen = _start_worker(resources)
    try:
        worker.load("no_such_scenario", 0)
        assert seen.wait(lambda: "error" in seen.kinds())
        message = [payload for kind, payload in seen.events if kind == "error"][0]
        assert "no_such_scenario" in message
        run = worker.load("v03_straight_line", 0)               # and it carries on
        assert seen.wait(lambda: seen.steps(run) == [0])
    finally:
        worker.shutdown()
