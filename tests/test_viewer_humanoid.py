"""The viewer on Phase 4 scenarios: the humanoid session, the frame's fusion and truth fields,
the orientation diagnostics, recordings, and the window's body / mount / motion controls."""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mujoco")

import phase3_helpers as helpers  # noqa: E402
from test_viewer import META, fake_frame  # noqa: E402

from robosoccer_perception.geometry.field import FieldGeometry  # noqa: E402
from robosoccer_perception.geometry.robot_description import robot_is_installed  # noqa: E402
from robosoccer_perception.humanoid_benchmark import runner, scenarios  # noqa: E402
from robosoccer_perception.viewer import catalog, headless  # noqa: E402
from robosoccer_perception.viewer.frame import (  # noqa: E402
    FusionView, Timeline, TruthView, pack, unpack)
from robosoccer_perception.viewer.humanoid_session import (  # noqa: E402
    FUSION_CHOICES, HumanoidLiveSession, SessionOptions)
from robosoccer_perception.viewer.recording import Recording  # noqa: E402
from robosoccer_perception.viewer.render import Layers, PanelRenderer  # noqa: E402
from robosoccer_perception.viewer.session import LiveSession  # noqa: E402
from robosoccer_perception.vision_benchmark.runner import BenchmarkResources  # noqa: E402

needs_humanoid = pytest.mark.skipif(not robot_is_installed("humanoid_prototype_1"),
                                    reason="humanoid model not imported")
SIM = [pytest.mark.render, pytest.mark.slow, needs_humanoid]
PHASE4_META = {**META, "phase4": True, "robot": "humanoid_prototype_1", "mount": "head_mid",
               "fusion_label": "IMU + uncertainty",
               "mount_description": "EXPERIMENTAL head_mid: pelvis + [...]"}


def _resources() -> BenchmarkResources:
    return BenchmarkResources(helpers.ColourVision(), resolution=(320, 180),
                              filter_overrides={"num_particles": 300})


def fusion_frame(step: int = 2, truth: bool = True):
    """`fake_frame` with the Phase 4 fields filled in (no simulator involved)."""
    frame = fake_frame(step, truth)
    fusion = FusionView("imu", 1.5, 3.25, 0.6, 1.4, 28.3, 0.797, "interpolated", 1.2, 20,
                        "correcting", (-0.3, 2.1), (-0.2, 1.9), 0.04)
    shown = None
    if truth:
        shown = dataclasses.replace(frame.truth, body_roll_deg=1.1, body_pitch_deg=3.9,
                                    camera_roll_deg=0.9, camera_pitch_deg=30.9,
                                    camera_height=0.791)
    view = np.random.default_rng(step).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    return dataclasses.replace(frame, fusion=fusion, truth=shown, robot_view=view)


# ------------------------------------------------------------------ frames and recordings
def test_fusion_and_truth_fields_survive_packing():
    frame = fusion_frame()
    back = unpack(pack(frame))
    assert back.fusion == frame.fusion
    assert back.truth.camera_pitch_deg == 30.9 and back.truth.body_roll_deg == 1.1
    assert back.robot_view.shape == (90, 160, 3)
    plain = unpack(pack(fake_frame()))                           # a Phase 3 frame: unchanged
    assert plain.fusion is None and plain.robot_view is None
    assert math.isnan(plain.truth.camera_pitch_deg)
    assert unpack(pack(frame.without_truth())).truth is None


def test_recordings_made_before_phase_4_still_open():
    packed = pack(fake_frame())
    packed["truth"] = packed["truth"][:8]                        # the old, shorter layout
    old = unpack(packed)
    assert isinstance(old.truth, TruthView) and math.isnan(old.truth.camera_height)
    timeline = Timeline(4, 0.1)
    timeline.record(fake_frame(1))
    arrays = {k: v for k, v in timeline.to_arrays().items()
              if k.split("__")[1] not in Timeline.OPTIONAL}
    loaded = Timeline.from_arrays(arrays, 0.1)
    assert loaded.last == 1 and np.isnan(loaded.values["est_camera_pitch"]).all()


def test_timeline_keeps_estimated_and_true_camera_orientation_apart():
    timeline = Timeline(6, 0.1)
    timeline.record(fusion_frame(2))
    timeline.record(fusion_frame(3, truth=False))
    values = timeline.values
    assert values["est_camera_pitch"][2] == values["est_camera_pitch"][3] == 28.3
    assert values["true_camera_pitch"][2] == 30.9 and np.isnan(values["true_camera_pitch"][3])
    assert values["tilt_std"][2] == 0.6


def test_panels_show_orientation_diagnostics_and_hide_truth_on_request():
    renderer = PanelRenderer(FieldGeometry.load("s_field"), PHASE4_META)
    timeline = Timeline(6, 0.1)
    frames = [fusion_frame(step) for step in range(4)]
    for frame in frames:
        timeline.record(frame)
    shown = renderer.panels(frames[3], Layers(), timeline)
    assert {k: v.shape for k, v in shown.items()} == {
        k: (540 + 28, 960, 3) for k in ("camera", "segmentation", "map", "diagnostics")}
    hidden = renderer.panels(frames[3], Layers(ground_truth=False), timeline)
    assert not np.array_equal(shown["diagnostics"], hidden["diagnostics"])
    assert np.array_equal(shown["camera"], hidden["camera"])
    # the third-person inset is one of the camera panel's layers
    without = renderer.camera(frames[3], Layers(robot_view=False))
    assert (shown["camera"] != without).any(axis=-1).mean() > 0.03
    # a Phase 3 frame is drawn by the same renderer as before
    plain = PanelRenderer(FieldGeometry.load("s_field"), META)
    assert plain.panels(fake_frame(), Layers(), timeline)["diagnostics"].shape == (568, 960, 3)
    assert ("robot_view", "Camera", "Robot view (3rd person)") in Layers.CATALOGUE


def test_options_are_checked_and_the_catalog_serves_both_phases():
    assert SessionOptions().fusion in FUSION_CHOICES and "oracle" not in " ".join(FUSION_CHOICES)
    for bad in (dict(fusion="oracle_orientation"), dict(robot="atlas"), dict(mount="head_x"),
                dict(motion="moonwalk")):
        with pytest.raises(ValueError):
            SessionOptions(**bad)
    scenario = scenarios.get_scenario("h05_walk_rectangle")
    changed = SessionOptions(robot="placeholder", mount="legacy_rig",
                             motion="sway_roll").apply(scenario)
    assert (changed.robot, changed.mount, changed.motion) == ("placeholder", "legacy_rig",
                                                              "sway_roll")
    assert SessionOptions().apply(scenario) is scenario
    ids = [s.id for s in catalog.all_scenarios()]
    assert "h05_walk_rectangle" in ids and "v04_rectangle" in ids and len(ids) == len(set(ids))
    assert catalog.is_phase4(catalog.find_scenario("h02"))
    assert not catalog.is_phase4(catalog.find_scenario("v04"))
    with pytest.raises(KeyError):
        catalog.find_scenario("z99")


# ------------------------------------------------------------------ the live session
class TestSession:
    pytestmark = SIM

    @pytest.fixture(scope="class")
    def frames(self, gl_context):
        scenario = scenarios.get_scenario("h02_sway_pitch")
        with HumanoidLiveSession(scenario, 0, _resources(), max_steps=40) as session:
            meta = session.meta()
            first = [session.step() for _ in range(40)]
            assert session.finished and session.step() is None
            session.reset()
            again = [session.step() for _ in range(40)]
        return meta, first, again

    def test_a_session_reproduces_the_benchmark_arm_exactly(self, frames):
        _, first, again = frames
        scenario = scenarios.get_scenario("h02_sway_pitch")
        run = runner.run_humanoid_scenario(scenario, 0, _resources(),
                                           (runner.IMU_UNCERTAINTY,), max_steps=40)
        estimates = np.array([frame.pose for frame in first])
        assert np.array_equal(estimates, run.arms["imu_uncertainty"].result.estimate)
        assert np.array_equal(estimates, np.array([frame.pose for frame in again]))   # reset
        assert np.array_equal(first[7].image, again[7].image)

    def test_frames_carry_the_robots_belief_and_separately_the_truth(self, frames):
        meta, first, _ = frames
        assert meta["phase4"] and meta["robot"] == "humanoid_prototype_1"
        assert meta["mount"] == "head_mid" and meta["mount_experimental"] is True
        assert "EXPERIMENTAL" in meta["mount_description"] and "pelvis" in meta["mount_description"]
        assert meta["camera_height_standing"] == pytest.approx(0.799, abs=0.01)
        late = first[30]
        assert late.fusion.source == "imu" and late.fusion.sync_method == "interpolated"
        assert late.fusion.imu_samples >= 15 and 0.1 < late.fusion.tilt_std_deg < 3.0
        assert late.fusion.camera_height == pytest.approx(0.799, abs=0.02)
        assert late.robot_view.shape == (270, 480, 3)
        # the estimated camera pitch follows the true one; a fixed model would be 25 deg flat
        errors = [f.fusion.camera_pitch_deg - f.truth.camera_pitch_deg for f in first[15:]]
        true_pitch = [f.truth.camera_pitch_deg for f in first[15:]]
        assert np.ptp(true_pitch) > 4.0 and np.sqrt(np.mean(np.square(errors))) < 1.0
        assert late.truth.body_pitch_deg == pytest.approx(late.truth.camera_pitch_deg - 25.0,
                                                          abs=0.5)
        assert "orientation" not in late.timing_ms and late.timing_ms["total"] > 0

    def test_ground_truth_display_does_not_change_the_estimates(self, frames, gl_context):
        _, first, _ = frames
        scenario = scenarios.get_scenario("h02_sway_pitch")
        with HumanoidLiveSession(scenario, 0, _resources(), max_steps=25, with_truth=False,
                                 options=SessionOptions(robot_view=False)) as session:
            blind = [session.step() for _ in range(25)]
        assert all(frame.truth is None and frame.robot_view is None for frame in blind)
        assert np.array_equal(np.array([f.pose for f in blind]),
                              np.array([f.pose for f in first[:25]]))
        assert blind[20].fusion == first[20].fusion          # the belief needs no truth

    def test_controls_switch_body_mount_motion_and_estimator(self, gl_context):
        scenario = scenarios.get_scenario("h05_walk_rectangle")
        options = SessionOptions(robot="placeholder", mount="legacy_rig", motion="sway_pitch",
                                 fusion="fixed")
        with catalog.open_session(scenario, 0, _resources(), options=options,
                                  max_steps=30) as session:
            assert isinstance(session, HumanoidLiveSession)
            meta = session.meta()
            frames = [session.step() for _ in range(30)]
        assert (meta["robot"], meta["mount"], meta["motion"], meta["fusion"]) == (
            "placeholder", "legacy_rig", "sway_pitch", "fixed")
        assert meta["camera_height_standing"] == pytest.approx(0.90)
        assert all(f.fusion.source == "nominal" for f in frames)
        assert all(f.fusion.camera_pitch_deg == pytest.approx(25.0) for f in frames)
        assert math.isnan(frames[10].fusion.tilt_std_deg) and frames[10].fusion.sync_method == ""
        assert np.ptp([f.truth.camera_pitch_deg for f in frames]) > 3.0   # the body does sway
        with catalog.open_session(catalog.find_scenario("v03"), 0, _resources(),
                                  max_steps=2) as plain:
            assert isinstance(plain, LiveSession) and plain.step().fusion is None

    def test_headless_run_records_and_replays_a_phase_4_scenario(self, gl_context, tmp_path):
        scenario = scenarios.get_scenario("h10_mount_pitch_error")
        summary = headless.run_headless(
            _resources(), scenario, 1, max_steps=30, record=tmp_path / "run.npz",
            video=tmp_path / "run.mp4", screenshots=1, screenshot_dir=tmp_path,
            options=SessionOptions(fusion="imu_selfcal"))
        assert summary["fusion"] == "imu_selfcal" and summary["frames"] == 30
        assert summary["camera_pitch_rmse_deg"] > 0.5            # the mount error is visible
        assert os.path.getsize(tmp_path / "run.mp4") > 10_000 and len(summary["screenshots"]) == 1
        with Recording(tmp_path / "run.npz") as recording:
            assert recording.meta["phase4"] and recording.meta["mount"] == "head_mid"
            frame = recording.frame(29)
            assert frame.fusion.monitor_status in ("collecting", "ok", "bias_suspected",
                                                   "correcting")
            assert frame.robot_view is not None and frame.truth.camera_height > 0.7
            assert np.isfinite(recording.timeline.values["est_camera_pitch"][:30]).all()


# ------------------------------------------------------------------ the window
class TestWindow:
    pytestmark = SIM

    @pytest.fixture
    def window(self, gl_context):
        pytest.importorskip("PySide6")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from robosoccer_perception.viewer import qt_env

        qt_env.prepare()
        from PySide6 import QtWidgets

        from robosoccer_perception.viewer.app import MainWindow, apply_dark_theme, wait_until

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
        apply_dark_theme(app)
        opened = []

        def open_window(**options):
            options.setdefault("max_steps", 25)
            options.setdefault("speed", float("inf"))
            win = MainWindow(_resources, **options)
            win.show()
            opened.append(win)
            assert wait_until(lambda: win.count >= 1, 60), win._last_error
            return win

        yield open_window, wait_until
        for win in opened:
            win.close()

    def test_humanoid_scenario_with_controls(self, window, tmp_path):
        open_window, wait_until = window
        win = open_window(scenario="h02_sway_pitch")
        assert win.meta["phase4"] and win.meta["robot"] == "humanoid_prototype_1"
        assert all(widget.isEnabled() for widget in win.phase4_widgets)
        assert win.session_options() == SessionOptions()
        frame = win.current_frame()
        assert frame.fusion is not None and frame.robot_view is not None
        win.play()
        assert wait_until(lambda: win.finished, 90), win._last_error
        assert win.count == 25 and np.isfinite(win.timeline.values["est_camera_pitch"]).all()
        # switch the body, the mount and the estimator: a new run starts with them
        win.robot_box.setCurrentIndex(win.robot_box.findData("placeholder"))
        win.mount_box.setCurrentIndex(win.mount_box.findData("legacy_rig"))
        win.fusion_box.setCurrentIndex(win.fusion_box.findData("fixed"))
        win.load_scenario()
        assert wait_until(lambda: win.count >= 1 and win.meta.get("robot") == "placeholder", 60)
        assert win.meta["mount"] == "legacy_rig" and win.meta["fusion"] == "fixed"
        assert win.current_frame().fusion.source == "nominal"
        shot = win.save_screenshot(tmp_path / "placeholder.png")
        assert shot.is_file() and shot.stat().st_size > 20_000

    def test_phase_3_scenarios_disable_the_body_controls(self, window):
        open_window, wait_until = window
        win = open_window(scenario="v03_straight_line")
        assert not win.meta.get("phase4")
        assert not any(widget.isEnabled() for widget in win.phase4_widgets)
        assert win.current_frame().fusion is None
