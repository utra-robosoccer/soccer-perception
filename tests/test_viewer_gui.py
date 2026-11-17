"""The viewer's window, driven without a screen (Qt's ``offscreen`` platform).

A colour-threshold stand-in replaces the networks, so these tests exercise the real window,
the real worker thread and the real simulator in a few seconds. Skipped when PySide6 (the
``viewer`` extra) or OpenGL is missing.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mujoco")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import phase3_helpers as helpers  # noqa: E402

from robosoccer_perception.viewer import qt_env  # noqa: E402

qt_env.prepare()

from PySide6 import QtCore, QtWidgets  # noqa: E402

from robosoccer_perception.viewer.app import MainWindow, apply_dark_theme, wait_until  # noqa: E402
from robosoccer_perception.viewer.recording import Recording  # noqa: E402
from robosoccer_perception.viewer.render import mosaic  # noqa: E402
from robosoccer_perception.vision_benchmark import runner  # noqa: E402

pytestmark = [pytest.mark.render, pytest.mark.slow]


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    apply_dark_theme(application)
    return application


def _resources(vision=None):
    return runner.BenchmarkResources(vision or helpers.ColourVision(), resolution=(320, 180),
                                     filter_overrides={"num_particles": 300})


@pytest.fixture
def window(app, gl_context):
    opened = []

    def open_window(**options):
        options.setdefault("scenario", "v03_straight_line")
        options.setdefault("max_steps", 40)
        options.setdefault("speed", float("inf"))
        vision = options.pop("vision", None)
        win = MainWindow(lambda: _resources(vision), **options)
        win.show()
        opened.append(win)
        assert wait_until(lambda: win.count >= 1 and win.renderer is not None, 60.0), \
            win._last_error or "no first frame"
        return win

    yield open_window
    for win in opened:
        win.close()
    app.processEvents()


def _settle(win, timeout=10.0):
    """Wait until the worker is idle and the newest state is on screen."""
    assert wait_until(lambda: not win.live_running, timeout)
    win._dirty = True
    assert wait_until(lambda: not win._dirty, timeout)


def test_the_window_opens_paused_on_the_first_frame(window):
    win = window()
    _settle(win)
    assert win.source == "live" and not win.playing and win.count == 1
    assert win.current_frame().step == 0 and win._shown_step == 0
    assert win.meta["scenario"] == "v03_straight_line" and win.meta["steps"] == 40
    assert win.run_button.text() == "Run"
    assert set(win._panels) == {"camera", "segmentation", "map", "diagnostics"}
    time.sleep(0.3)
    QtWidgets.QApplication.processEvents()
    assert win.count == 1, "nothing runs until Run is pressed"


def test_run_pause_step_and_reset(window):
    win = window()
    win.run_button.click()                                      # run
    assert win.playing and win.run_button.text() == "Pause"
    assert wait_until(lambda: win.count >= 12, 30.0)
    win.run_button.click()                                      # pause
    assert not win.playing
    _settle(win)
    paused = win.count
    wait_until(lambda: False, 0.3)
    assert win.count == paused, "paused means no new frames"
    assert win.cursor == paused - 1 and win._shown_step == paused - 1

    for extra in (1, 2, 3):                                     # step forward: one frame each
        win.step_button.click()
        assert wait_until(lambda: win.count == paused + extra, 10.0)
        _settle(win)
        assert win._shown_step == paused + extra - 1
    win.back_button.click()                                     # step back: from the history
    _settle(win)
    assert win.count == paused + 3 and win._shown_step == paused + 1

    first_pose = win.frame_at(0).pose.copy()
    win.reset_button.click()                                    # reset
    assert wait_until(lambda: win.count == 1 and win.current_frame() is not None
                      and win.current_frame().step == 0, 30.0)
    _settle(win)
    assert not win.playing and win.timeline.last == 0
    np.testing.assert_allclose(win.current_frame().pose, first_pose)

    win.play()                                                  # and on to the end
    assert wait_until(lambda: win.finished and not win.playing, 60.0)
    assert win.count == 40 and win.cursor == 39
    assert [win.frame_at(i).step for i in (0, 17, 39)] == [0, 17, 39]


def test_all_panels_show_the_frame_under_the_cursor(window):
    """Scrubbing: whatever frame the slider points at, the four panels are that frame's
    image, features, estimate and diagnostics, and the labels say its time."""
    win = window()
    win.play()
    assert wait_until(lambda: win.count >= 25, 30.0)
    win.pause()
    _settle(win)
    for target in (3, 19, 0, win.count - 1):
        win.slider.setValue(target)                             # as dragging the slider does
        _settle(win)
        frame = win.frame_at(target)
        assert win.cursor == target and win._shown_step == frame.step == target
        expected = win.renderer.panels(frame, win.layers, win.timeline, win._status())
        for name in expected:
            np.testing.assert_array_equal(win._panels[name], expected[name])
        assert f"frame {target + 1} of 40" in win.position_label.text()
        assert f"{frame.timestamp:6.2f} s" in win.position_label.text()
    # the newest frame is shown as it arrived; older ones come back from the history
    assert win.frame_at(win.count - 1) is win._latest
    assert win.frame_at(3) is not win._latest and win.frame_at(3).step == 3


def test_replay_from_the_history_then_live_again(window):
    win = window()
    win.play()
    assert wait_until(lambda: win.count >= 15, 30.0)
    win.pause()
    _settle(win)
    produced = win.count
    win.speed_box.setCurrentIndex(win.speed_box.findText("4x"))
    win.slider.setValue(2)
    _settle(win)
    assert "REPLAY of this run" in win._status()
    win.play()                                                  # replays 2 .. end, then new
    assert wait_until(lambda: win.cursor > 6, 10.0)
    assert win.count == produced, "replaying the history simulates nothing"
    assert wait_until(lambda: win.count > produced + 3, 30.0), "then the simulation resumes"
    win.pause()
    _settle(win)
    assert "LIVE" in win._status()
    assert [win.frame_at(i).step for i in range(win.count)] == list(range(win.count))


def test_overlay_toggles_and_the_ground_truth_switch(window):
    win = window()
    win.play()
    assert wait_until(lambda: win.count >= 10, 30.0)
    win.pause()
    _settle(win)
    before = {name: image.copy() for name, image in win._panels.items()}
    win.layer_boxes["line_points"].setChecked(False)
    _settle(win)
    assert not win.layers.line_points
    assert not np.array_equal(win._panels["segmentation"], before["segmentation"])
    np.testing.assert_array_equal(win._panels["camera"], before["camera"])
    win.layer_boxes["line_points"].setChecked(True)
    _settle(win)
    np.testing.assert_array_equal(win._panels["segmentation"], before["segmentation"])

    index, count = win.cursor, win.count
    estimate = win.current_frame().pose.copy()
    win.truth_box.setChecked(False)                             # hide ground truth
    _settle(win)
    assert not win.layers.ground_truth
    assert not np.array_equal(win._panels["map"], before["map"])
    assert not np.array_equal(win._panels["diagnostics"], before["diagnostics"])
    win.step_forward()                                          # the run goes on unchanged
    assert wait_until(lambda: win.count == count + 1, 10.0)
    np.testing.assert_array_equal(win.frame_at(index).pose, estimate)


def test_the_ground_truth_switch_does_not_change_the_estimates(window):
    shown = window(show_truth=True, max_steps=15)
    shown.play()
    assert wait_until(lambda: shown.finished, 60.0)
    with_truth = [shown.frame_at(i).pose for i in range(shown.count)]
    hidden = window(show_truth=False, max_steps=15)
    assert not hidden.truth_box.isChecked()
    hidden.play()
    assert wait_until(lambda: hidden.finished, 60.0)
    np.testing.assert_array_equal([hidden.frame_at(i).pose for i in range(hidden.count)],
                                  with_truth)


def test_screenshot_recording_replay_and_video(window, tmp_path):
    win = window(max_steps=20)
    win.play()
    assert wait_until(lambda: win.finished and not win.playing, 60.0)
    _settle(win)
    poses = [win.frame_at(i).pose.copy() for i in range(win.count)]

    shot = win.save_screenshot(tmp_path / "shot.png")
    picture = cv2.imread(str(shot))
    assert picture is not None and picture.shape[0] > 1000 and picture.shape[1] > 1900
    expected = mosaic(win._output_panels(win.current_frame(), win._status()))
    assert picture.shape == expected.shape, "screenshots have a fixed size, not the window's"

    saved = win.save_recording(tmp_path / "run.npz")
    with Recording(saved) as recording:
        assert len(recording) == 20 and recording.meta["scenario"] == "v03_straight_line"

    video = win.export_video(tmp_path / "run.mp4")
    assert video["frames"] == 20 and not video["cancelled"]
    capture = cv2.VideoCapture(str(tmp_path / "run.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    capture.release()

    win.open_recording(saved)                                   # replay it
    _settle(win)
    assert win.source == "recording" and win.count == 20 and win.cursor == 0
    assert not win.live_button.isHidden() and "REPLAY of a recording" in win._status()
    np.testing.assert_allclose([win.frame_at(i).pose for i in range(20)], poses)
    win.speed_box.setCurrentIndex(win.speed_box.findText("4x"))
    win.play()
    assert wait_until(lambda: win.cursor == 19 and not win.playing, 20.0), "plays to the end"
    win.slider.setValue(7)
    _settle(win)
    assert win._shown_step == 7
    assert win.save_recording(tmp_path / "again.npz") is None, "a recording is not re-recorded"
    win.reset()
    assert win.cursor == 0
    win.step_forward()
    assert win.cursor == 1 and win.count == 20

    win.live_button.click()                                     # back to the simulation
    assert wait_until(lambda: win.source == "live" and win.count == 1
                      and win.renderer is not None and win.current_frame() is not None
                      and win.current_frame().step == 0, 30.0)
    assert win.live_button.isHidden()


def test_a_recording_can_be_opened_at_start_up(window, tmp_path, app):
    win = window(max_steps=8)
    win.play()
    assert wait_until(lambda: win.finished, 60.0)
    saved = win.save_recording(tmp_path / "short.npz")
    replay = MainWindow(lambda: pytest.fail("a replay must not load the models"),
                        replay=str(saved))
    try:
        replay.show()
        _settle(replay)
        assert replay.source == "recording" and replay.count == 8 and replay._shown_step == 0
        replay.seek(5)
        _settle(replay)
        assert replay._shown_step == 5
    finally:
        replay.close()
        app.processEvents()


@pytest.mark.parametrize("scenario,ambiguous", [("v04_rectangle", False),
                                               ("v12a_symmetry_no_prior", True),
                                               ("v13a_kidnapped_same_side", False),
                                               ("v11_goal_post_occlusion", False)])
def test_benchmark_scenarios_can_be_selected(window, scenario, ambiguous):
    win = window(max_steps=12)
    win.scenario_box.setCurrentIndex(win.scenario_box.findData(scenario))
    win.load_scenario()
    assert wait_until(lambda: win.meta.get("scenario") == scenario and win.count == 1, 60.0), \
        win._last_error
    win.play()
    assert wait_until(lambda: win.finished and not win.playing, 60.0), win._last_error
    _settle(win)
    assert win.count == 12 and not win._last_error
    frames = [win.frame_at(i) for i in range(12)]
    assert all(np.isfinite(f.pose).all() for f in frames)
    assert any(f.ambiguous for f in frames) == ambiguous
    assert win.scenario_box.count() == 16


def test_changing_the_seed_restarts_with_another_run(window):
    win = window(max_steps=6)
    first = win.frame_at(0).truth.pose.copy()
    win.seed_box.setValue(3)
    assert wait_until(lambda: win.meta.get("seed") == 3 and win.count == 1
                      and win.current_frame() is not None, 60.0)
    assert not np.allclose(win.frame_at(0).truth.pose, first)


class _SlowVision(helpers.ColourVision):
    """A network that takes 150 ms per image: slower than any display refresh."""

    def run(self, image_rgb, source=None):
        time.sleep(0.15)
        return super().run(image_rgb, source)


def test_the_window_stays_responsive_when_inference_is_slow(window, app):
    win = window(vision=_SlowVision(), max_steps=60, speed=float("inf"))
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(10)
    start_count, started = win.count, time.monotonic()
    win.play()
    wait_until(lambda: False, 2.0)
    elapsed = time.monotonic() - started
    timer.stop()
    produced = win.count - start_count
    assert 5 <= produced <= elapsed / 0.15 + 1, "inference really was the bottleneck"
    assert len(ticks) > 100, "the GUI thread kept running its own timers"
    assert max(np.diff(ticks)) < 0.12, "and was never blocked for long"
    win.pause()                                                 # a command is obeyed promptly
    assert wait_until(lambda: not win.live_running, 2.0)
    stopped = win.count
    wait_until(lambda: False, 0.4)
    assert win.count == stopped


def test_keyboard_shortcuts_exist(window):
    win = window()
    from PySide6 import QtGui
    keys = {shortcut.key().toString() for shortcut in win.findChildren(QtGui.QShortcut)}
    assert {"Space", "Right", "Left", "R", "S", "G", "Home", "End"} <= keys


def test_qt_starts_next_to_opencv(monkeypatch):
    """`import cv2` points Qt at OpenCV's own Qt 5 plugins; `prepare` undoes that."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/somewhere/site-packages/cv2/qt/plugins")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/somewhere/site-packages/cv2/qt/fonts")
    assert qt_env.prepare() == "offscreen"
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ
    assert "QT_QPA_FONTDIR" not in os.environ
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/opt/my-own-qt/plugins")
    qt_env.prepare()
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/opt/my-own-qt/plugins", \
        "a path the user set is left alone"


def test_a_missing_display_or_library_is_explained(monkeypatch):
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(qt_env.sys, "platform", "linux")
    with pytest.raises(qt_env.QtUnavailable, match="--headless"):
        qt_env.prepare()
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(qt_env.ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(qt_env, "LOCAL_LIBRARY_DIR", qt_env.Path("/nonexistent"))
    with pytest.raises(qt_env.QtUnavailable, match="libxcb-cursor0"):
        qt_env.prepare()
