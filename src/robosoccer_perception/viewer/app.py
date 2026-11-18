"""The viewer's window (PySide6). See docs/viewer.md.

    +----------------------+----------------------+-----------+
    | camera               | segmentation         | overlays  |
    +----------------------+----------------------+ (toggles) |
    | localization map     | diagnostics          |           |
    +----------------------+----------------------+-----------+
    | scenario  seed  run / pause  step  reset  speed   slider   screenshot  record  video |
    | robot  camera mount  body motion  camera pose          (Phase 4 scenarios, "h...")   |

How it stays consistent and responsive:

* The simulation, the networks and the filter run in `worker.SimulationWorker`. The window
  only receives finished `ViewerFrame`s and draws them.
* All four panels are drawn from the one frame under the cursor, in one go, so they always
  show the same timestamp.
* Frames are drawn by a 30 Hz timer that takes the newest state. If frames arrive faster than
  they can be drawn, intermediate ones are skipped on screen (they are still kept in the
  history); if inference is slow, the window simply redraws less often.
* There is one transport. Every frame of the run is kept (compressed) in a history; the slider
  moves a cursor through it. "Run" plays the history forward from the cursor and, when the
  cursor reaches the newest frame of a live run, asks the worker for new frames.

Ground truth is drawn from `ViewerFrame.truth` and nowhere else; the checkbox hides it.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.viewer import headless
from robosoccer_perception.viewer import recording as recording_module
from robosoccer_perception.viewer.frame import Timeline, ViewerFrame, unpack
from robosoccer_perception.viewer.recording import Recording
from robosoccer_perception.viewer.render import (
    PANEL_SIZE, TITLE_BAR, Layers, PanelRenderer, mosaic)
from robosoccer_perception.geometry.mount import available_mounts
from robosoccer_perception.geometry.robot_description import available_robots
from robosoccer_perception.simulation.body_motion import available_profiles
from robosoccer_perception.viewer import catalog
from robosoccer_perception.viewer.humanoid_session import (
    DEFAULT_FUSION, FUSION_LABELS, SessionOptions)
from robosoccer_perception.viewer.worker import SimulationWorker

SPEEDS = (("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0),
          ("max", math.inf))
DISPLAY_INTERVAL_MS = 33
PANELS = ("camera", "segmentation", "map", "diagnostics")


class _Bridge(QtCore.QObject):
    """Carries the worker's callbacks into the GUI thread (queued signal delivery)."""
    frame = QtCore.Signal(int, object, object)
    event = QtCore.Signal(str, object)


class ImagePanel(QtWidgets.QWidget):
    """Shows one RGB array, scaled to the widget with its aspect ratio kept."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self.setMinimumSize(320, 190)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_image(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb)
        self._image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                                   QtGui.QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event) -> None:                        # noqa: N802 - Qt name
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(22, 24, 28))
        if self._image is None:
            return
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        painter.drawImage(QtCore.QRectF((self.width() - width) / 2.0,
                                        (self.height() - height) / 2.0, width, height),
                          self._image)


class MainWindow(QtWidgets.QMainWindow):
    """The viewer. ``resources_factory`` is called once, in the worker thread, to load the
    models; tests pass a factory that returns a stand-in."""

    def __init__(self, resources_factory: Callable[[], Any], *, scenario: str = "v04_rectangle",
                 seed: int = 0, speed: float = 1.0, max_particles: int = 600,
                 max_steps: int | None = None, show_truth: bool = True,
                 autoplay: bool = False, replay: str | None = None,
                 options: SessionOptions | None = None):
        super().__init__()
        self.setWindowTitle("RoboSoccer vision-localization viewer")
        self.layers = Layers(ground_truth=show_truth)
        self.scenarios = catalog.all_scenarios()
        self._initial_options = options or SessionOptions()
        self._autoplay = autoplay

        # ---- what is being shown
        self.source = "live"                    # "live" or "recording"
        self.history: list[dict] = []           # packed frames of the live run
        self.recording: Recording | None = None
        self.timeline: Timeline | None = None
        self.renderer: PanelRenderer | None = None
        self.meta: dict[str, Any] = {}
        self.cursor = -1
        self.playing = False                    # the transport is moving
        self.live_running = False               # the worker is producing
        self.finished = False                   # the live run reached its last frame
        self.run_id = -1
        self._latest: ViewerFrame | None = None     # newest live frame, undecoded
        self._cache: tuple[int, ViewerFrame] | None = None
        self._dirty = False
        self._speed = speed
        self._panels: dict[str, np.ndarray] = {}
        self._export_renderer: PanelRenderer | None = None
        self._shown_step = -1
        self._last_error = ""
        self._drawn = 0

        self._bridge = _Bridge()
        self._bridge.frame.connect(self._on_frame)
        self._bridge.event.connect(self._on_event)
        self.worker = SimulationWorker(resources_factory, self._bridge.frame.emit,
                                       self._bridge.event.emit, max_particles=max_particles,
                                       max_steps=max_steps)
        self._build_ui(scenario, seed, speed)
        self.worker.set_speed(speed)
        self.worker.start()

        self._display_timer = QtCore.QTimer(self)
        self._display_timer.timeout.connect(self._refresh)
        self._display_timer.start(DISPLAY_INTERVAL_MS)
        self._replay_timer = QtCore.QTimer(self)
        self._replay_timer.timeout.connect(self._replay_tick)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._fit_renderer)

        if replay:
            self.open_recording(replay)
        else:
            self.load_scenario()

    # ------------------------------------------------------------------ construction
    def _build_ui(self, scenario: str, seed: int, speed: float) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ---- top bar: what to run
        top = QtWidgets.QHBoxLayout()
        self.scenario_box = QtWidgets.QComboBox()
        for item in self.scenarios:
            self.scenario_box.addItem(f"{item.id}  -  {item.title}", item.id)
        index = self.scenario_box.findData(scenario)
        if index < 0:
            matches = [i for i, item in enumerate(self.scenarios)
                       if item.id.startswith(scenario)]
            index = matches[0] if matches else 0
        self.scenario_box.setCurrentIndex(index)
        self.scenario_box.activated.connect(lambda _=None: self.load_scenario())
        self.seed_box = QtWidgets.QSpinBox()
        self.seed_box.setRange(0, 9999)
        self.seed_box.setValue(seed)
        self.seed_box.setKeyboardTracking(False)
        self.seed_box.valueChanged.connect(lambda _=None: self.load_scenario())
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setToolTip("Run / pause (Space)")
        self.run_button.clicked.connect(self.toggle_play)
        self.back_button = QtWidgets.QPushButton("< Step")
        self.back_button.setToolTip("One frame back (Left)")
        self.back_button.clicked.connect(self.step_back)
        self.step_button = QtWidgets.QPushButton("Step >")
        self.step_button.setToolTip("One frame forward (Right)")
        self.step_button.clicked.connect(self.step_forward)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Restart the run from its first frame (R)")
        self.reset_button.clicked.connect(self.reset)
        self.speed_box = QtWidgets.QComboBox()
        for label, value in SPEEDS:
            self.speed_box.addItem(label, value)
        values = [value for _, value in SPEEDS]
        self.speed_box.setCurrentIndex(values.index(speed) if speed in values
                                       else values.index(1.0))
        self.speed_box.currentIndexChanged.connect(self._speed_changed)
        self.truth_box = QtWidgets.QCheckBox("Ground truth")
        self.truth_box.setToolTip("Show the simulator's true pose and the errors (G). "
                                  "Display only: the estimator never receives it.")
        self.truth_box.setChecked(self.layers.ground_truth)
        self.truth_box.toggled.connect(lambda on: self.set_layer("ground_truth", on))
        self.live_button = QtWidgets.QPushButton("Back to live")
        self.live_button.setToolTip("Close the recording and return to the live simulation")
        self.live_button.clicked.connect(self.load_scenario)
        self.live_button.setVisible(False)
        for widget in (QtWidgets.QLabel("Scenario"), self.scenario_box,
                       QtWidgets.QLabel("Seed"), self.seed_box, self.run_button,
                       self.back_button, self.step_button, self.reset_button,
                       QtWidgets.QLabel("Speed"), self.speed_box, self.truth_box,
                       self.live_button):
            top.addWidget(widget)
        top.addStretch(1)
        outer.addLayout(top)

        # ---- second bar: the body, its camera mount, its motion and the camera-pose
        # estimator. These apply to the Phase 4 scenarios; "scenario" keeps a scenario's own.
        body = QtWidgets.QHBoxLayout()
        initial = self._initial_options

        def choice(tip: str, entries, current) -> QtWidgets.QComboBox:
            box = QtWidgets.QComboBox()
            box.setToolTip(tip)
            for label, value in entries:
                box.addItem(label, value)
            box.setCurrentIndex(max(box.findData(current), 0))
            box.activated.connect(lambda _=None: self.load_scenario())
            return box

        keep = [("as the scenario", None)]
        self.robot_box = choice(
            "The body that carries the camera: UTRA's humanoid_prototype_1 or the placeholder "
            "rig of Phases 1 to 3", keep + [(name, name) for name in available_robots()],
            initial.robot)
        self.mount_box = choice(
            "Virtual camera mount preset. Every preset is an experimental assumption: the "
            "robot's head is not designed", keep + [(name, name) for name in available_mounts()],
            initial.mount)
        self.motion_box = choice(
            "Synthetic body motion: standing, sway, walking-like, disturbances",
            keep + [(name, name) for name in available_profiles()], initial.motion)
        self.fusion_box = choice(
            "How the robot estimates its camera pose. 'Fixed' ignores the IMU, as Phase 3 did",
            [(label, name) for name, label in FUSION_LABELS.items()], initial.fusion)
        self.phase4_widgets = [QtWidgets.QLabel("Robot"), self.robot_box,
                               QtWidgets.QLabel("Camera mount"), self.mount_box,
                               QtWidgets.QLabel("Body motion"), self.motion_box,
                               QtWidgets.QLabel("Camera pose"), self.fusion_box]
        for widget in self.phase4_widgets:
            body.addWidget(widget)
        body.addStretch(1)
        outer.addLayout(body)

        # ---- the four panels and the overlay toggles
        middle = QtWidgets.QHBoxLayout()
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        self.panels = {name: ImagePanel() for name in PANELS}
        for index, name in enumerate(PANELS):
            grid.addWidget(self.panels[name], index // 2, index % 2)
        middle.addLayout(grid, 1)
        side = QtWidgets.QVBoxLayout()
        self.layer_boxes: dict[str, QtWidgets.QCheckBox] = {}
        groups: dict[str, QtWidgets.QVBoxLayout] = {}
        for attribute, panel, label in Layers.CATALOGUE:
            if attribute == "ground_truth":
                continue                        # has its own checkbox in the top bar
            if panel not in groups:
                box = QtWidgets.QGroupBox(panel)
                groups[panel] = QtWidgets.QVBoxLayout(box)
                side.addWidget(box)
            check = QtWidgets.QCheckBox(label)
            check.setChecked(getattr(self.layers, attribute))
            check.toggled.connect(lambda on, name=attribute: self.set_layer(name, on))
            groups[panel].addWidget(check)
            self.layer_boxes[attribute] = check
        output = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout(output)
        for label, tip, action in (
                ("Screenshot", "Save the four panels as a PNG (S)", self.save_screenshot),
                ("Save recording ...", "Save this run for later replay", self._save_dialog),
                ("Open recording ...", "Replay a saved run", self._open_dialog),
                ("Export video ...", "Write the run as an .mp4", self._video_dialog)):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, call=action: call())
            output_layout.addWidget(button)
        side.addWidget(output)
        side.addStretch(1)
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(215)
        middle.addWidget(side_widget)
        outer.addLayout(middle, 1)

        # ---- bottom: the scrubber
        bottom = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setToolTip("Scrub through the frames of this run")
        self.slider.valueChanged.connect(self._slider_moved)
        self.position_label = QtWidgets.QLabel("no frame yet")
        self.position_label.setMinimumWidth(300)
        bottom.addWidget(self.slider, 1)
        bottom.addWidget(self.position_label)
        outer.addLayout(bottom)
        self.statusBar().showMessage("starting ...")

        for key, action in (("Space", self.toggle_play), ("Right", self.step_forward),
                            ("Left", self.step_back), ("R", self.reset),
                            ("S", self.save_screenshot), ("G", self.truth_box.toggle),
                            ("Home", lambda: self.seek(0)),
                            ("End", lambda: self.seek(self.count - 1))):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            shortcut.activated.connect(action)
        self.resize(1560, 960)

    # ------------------------------------------------------------------ frames
    @property
    def count(self) -> int:
        """Frames available under the slider."""
        return len(self.recording) if self.source == "recording" else len(self.history)

    def frame_at(self, index: int) -> ViewerFrame:
        if self.source == "live" and self._latest is not None and self._latest.step == index \
                and index == len(self.history) - 1:
            return self._latest
        if self._cache is not None and self._cache[0] == index:
            return self._cache[1]
        frame = self.recording.frame(index) if self.source == "recording" \
            else unpack(self.history[index])
        self._cache = (index, frame)
        return frame

    def current_frame(self) -> ViewerFrame | None:
        return self.frame_at(self.cursor) if 0 <= self.cursor < self.count else None

    def _on_frame(self, run_id: int, frame: ViewerFrame, packed: dict) -> None:
        self.worker.ack()
        if run_id != self.run_id or self.source != "live":
            return                              # a frame of a run that was reset or replaced
        at_end = self.cursor >= len(self.history) - 1
        self.history.append(packed)
        self.timeline.record(frame)
        self._latest = frame
        if at_end:
            self.cursor = len(self.history) - 1
        self._dirty = True

    def _on_event(self, kind: str, payload: Any) -> None:
        if kind == "status":
            self.statusBar().showMessage(str(payload))
        elif kind == "loaded":
            if payload["run_id"] != self.run_id:
                return
            self._begin_run(payload["meta"], payload["field"])
            self.statusBar().showMessage(
                f"{self.meta['scenario']} seed {self.meta['seed']}: {self.meta['description']}")
            if self._autoplay:
                self.play()
        elif kind == "running":
            self.live_running = bool(payload)
            self._update_controls()
        elif kind == "finished":
            if payload == self.run_id and self.source == "live":
                self.finished = True
                if self.cursor >= self.count - 1:
                    self._stop_transport()
                self.statusBar().showMessage("the run is complete: scrub, replay, record or "
                                             "export it, or press Reset")
        elif kind == "error":
            self._last_error = str(payload)
            self._stop_transport()
            self.statusBar().showMessage("error: " + self._last_error.strip().splitlines()[-1])
            print(self._last_error, file=sys.stderr)

    def _begin_run(self, meta: dict[str, Any], field: FieldGeometry) -> None:
        self.meta = dict(meta)
        self.timeline = Timeline(int(meta["steps"]), float(meta["dt"]))
        self.renderer = PanelRenderer(field, meta, self._panel_size())
        self._export_renderer = None
        self.history.clear()
        self.cursor, self._latest, self._cache = -1, None, None
        self.finished = False
        self._dirty = True

    # ------------------------------------------------------------------ sources
    def load_scenario(self) -> None:
        """(Re)start the live simulation of the selected scenario and seed, paused."""
        self._stop_transport()
        if self.recording is not None:
            self.recording.close()
            self.recording = None
        self.source = "live"
        self.history.clear()
        self.cursor, self._latest, self._cache = -1, None, None
        self.finished = False
        self.run_id = self.worker.load(self.scenario_box.currentData(), self.seed_box.value(),
                                       self.session_options())
        self._update_controls()

    def session_options(self) -> SessionOptions:
        """The Phase 4 controls as they are set now."""
        return SessionOptions(robot=self.robot_box.currentData(),
                              mount=self.mount_box.currentData(),
                              motion=self.motion_box.currentData(),
                              fusion=self.fusion_box.currentData() or DEFAULT_FUSION,
                              robot_view=self._initial_options.robot_view)

    def _is_phase4_selected(self) -> bool:
        return catalog.is_phase4(catalog.find_scenario(self.scenario_box.currentData()))

    def reset(self) -> None:
        """Back to the first frame. A live run is simulated again from the start."""
        self._stop_transport()
        if self.source == "recording":
            self.seek(0)
            return
        self.history.clear()
        self.cursor, self._latest, self._cache = -1, None, None
        self.finished = False
        if self.timeline is not None:
            self.timeline.clear()
        self.run_id = self.worker.reset()
        self._update_controls()

    def open_recording(self, path) -> None:
        self._stop_transport()
        recording = Recording(path)
        if self.recording is not None:
            self.recording.close()
        self.run_id = -1                        # frames still coming from the worker are dropped
        self.source, self.recording = "recording", recording
        self.meta = dict(recording.meta)
        self.timeline = recording.timeline
        self.renderer = PanelRenderer(FieldGeometry.load(self.meta["field"]), self.meta,
                                      self._panel_size())
        self._export_renderer = None
        self.history.clear()
        self._latest, self._cache = None, None
        self.cursor = 0 if len(recording) else -1
        self.finished = True
        self._dirty = True
        self._update_controls()
        self.statusBar().showMessage(f"replaying {Path(path).name}: {len(recording)} frames of "
                                     f"{self.meta['scenario']} seed {self.meta['seed']}")

    # ------------------------------------------------------------------ transport
    def _interval_ms(self) -> int:
        dt = float(self.meta.get("dt", 0.1))
        return 0 if math.isinf(self._speed) else max(1, round(1000.0 * dt / self._speed))

    def play(self) -> None:
        if self.count == 0 and self.source != "live":
            return
        self.playing = True
        if self.cursor >= self.count - 1 and (self.source == "recording" or self.finished):
            self.cursor = 0                     # at the end of a complete run: play it again
            self._dirty = True
        self._continue()

    def _continue(self) -> None:
        """Play from the history if the cursor is behind, else ask for new frames."""
        if self.cursor < self.count - 1:
            self.worker.pause()
            self._replay_timer.start(self._interval_ms())
        elif self.source == "live" and not self.finished:
            self._replay_timer.stop()
            self.worker.resume()
        else:
            self._stop_transport()
        self._update_controls()

    def _replay_tick(self) -> None:
        if self.cursor < self.count - 1:
            self.cursor += 1
            self._dirty = True
        else:
            self._continue()

    def _stop_transport(self) -> None:
        self.playing = False
        self._replay_timer.stop()
        self.worker.pause()
        self._update_controls()

    def pause(self) -> None:
        self._stop_transport()

    def toggle_play(self) -> None:
        self.pause() if self.playing else self.play()

    def step_forward(self) -> None:
        self._stop_transport()
        if self.cursor < self.count - 1:
            self.seek(self.cursor + 1)
        elif self.source == "live" and not self.finished:
            self.worker.step_once()

    def step_back(self) -> None:
        self._stop_transport()
        self.seek(self.cursor - 1)

    def seek(self, index: int) -> None:
        if self.count:
            self.cursor = int(np.clip(index, 0, self.count - 1))
            self._dirty = True

    def _slider_moved(self, value: int) -> None:
        if value != self.cursor:                # moved by the user, not by `_refresh`
            self._stop_transport()
            self.seek(value)

    def _speed_changed(self) -> None:
        self._speed = float(self.speed_box.currentData())
        self.worker.set_speed(self._speed)
        if self._replay_timer.isActive():
            self._replay_timer.start(self._interval_ms())

    def set_layer(self, name: str, enabled: bool) -> None:
        setattr(self.layers, name, bool(enabled))
        if name == "ground_truth" and self.truth_box.isChecked() != bool(enabled):
            self.truth_box.setChecked(bool(enabled))
        self._dirty = True

    # ------------------------------------------------------------------ drawing
    def _panel_size(self) -> tuple[int, int]:
        """The size the panels are drawn at: the size of a panel widget, so that text and
        lines are not rescaled on screen."""
        widget = self.panels["camera"]
        return (max(widget.width(), 520), max(widget.height() - TITLE_BAR, 300))

    def _fit_renderer(self) -> None:
        if self.renderer is not None and self.renderer.size != self._panel_size():
            self.renderer = PanelRenderer(self.renderer.field, self.meta, self._panel_size())
            self._dirty = True

    def resizeEvent(self, event) -> None:                       # noqa: N802 - Qt name
        super().resizeEvent(event)
        self._resize_timer.start(120)

    def _output_panels(self, frame: ViewerFrame, status: str) -> dict[str, np.ndarray]:
        """The panels at the fixed output size, whatever the window's size is."""
        return self._output_renderer().panels(frame, self.layers, self.timeline, status)

    def _output_renderer(self) -> PanelRenderer:
        if self._export_renderer is None:
            self._export_renderer = PanelRenderer(self.renderer.field, self.meta, PANEL_SIZE)
        return self._export_renderer

    def _status(self) -> str:
        if self.source == "recording":
            mode = "REPLAY of a recording"
        elif self.cursor < self.count - 1:
            mode = "REPLAY of this run"
        else:
            mode = "LIVE"
        state = "running" if self.playing else "paused"
        speed = self.speed_box.currentText()
        return f"{mode}  |  {state}  |  speed {speed}"

    def _refresh(self) -> None:
        """Draw the frame under the cursor if anything changed. Called by the display timer."""
        if not self._dirty or self.renderer is None:
            return
        self._dirty = False
        frame = self.current_frame()
        if frame is None:
            return
        self._panels = self.renderer.panels(frame, self.layers, self.timeline, self._status())
        for name, image in self._panels.items():
            self.panels[name].set_image(image)
        self._shown_step = frame.step
        self._drawn += 1
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(self.count - 1, 0))
        self.slider.setValue(self.cursor)
        self.slider.blockSignals(False)
        total = int(self.meta.get("steps", self.count))
        self.position_label.setText(
            f"t = {frame.timestamp:6.2f} s   frame {frame.step + 1} of {total}"
            f"   ({self.count} available)")

    def _update_controls(self) -> None:
        self.run_button.setText("Pause" if self.playing else "Run")
        self.live_button.setVisible(self.source == "recording")
        phase4 = self.source == "live" and self._is_phase4_selected()
        for widget in self.phase4_widgets:
            widget.setEnabled(phase4)
        self._dirty = True

    # ------------------------------------------------------------------ output
    def save_screenshot(self, path=None) -> Path | None:
        """The four panels of the frame on screen, as one PNG."""
        frame = self.current_frame()
        if frame is None:
            return None
        panels = self._output_panels(frame, self._status())
        path = Path(path) if path else recording_module.default_path(
            "screenshots", self.meta, ".png", frame.step)
        headless.save_png(path, mosaic(panels))
        self.statusBar().showMessage(f"screenshot saved: {path}")
        return path

    def save_recording(self, path=None) -> Path | None:
        """Everything simulated so far in this run, for later replay."""
        if self.source != "live" or not self.history:
            self.statusBar().showMessage("nothing to record: this is already a recording"
                                         if self.source != "live" else "no frames yet")
            return None
        self._stop_transport()
        path = Path(path) if path else recording_module.default_path(
            "recordings", self.meta, ".npz")
        written = recording_module.save_recording(path, self.meta, list(self.history),
                                                  self.timeline)
        self.statusBar().showMessage(f"recording saved ({len(self.history)} frames): {written}")
        return written

    def export_video(self, path=None, first: int = 0, last: int | None = None
                     ) -> dict[str, Any] | None:
        """The frames ``first`` to ``last`` with the overlays as they are set now."""
        if not self.count:
            return None
        self._stop_transport()
        last = self.count - 1 if last is None else min(last, self.count - 1)
        path = Path(path) if path else recording_module.default_path("videos", self.meta, ".mp4")
        total = last - first + 1
        dialog = QtWidgets.QProgressDialog("Exporting video ...", "Cancel", 0, total, self)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setMinimumDuration(400)

        def progress(done: int) -> bool:
            dialog.setValue(done)
            QtWidgets.QApplication.processEvents()
            return not dialog.wasCanceled()

        fps = (1.0 if math.isinf(self._speed) else self._speed) / float(self.meta["dt"])
        try:
            result = headless.export_video(
                path, self._output_renderer(),
                (self.frame_at(i) for i in range(first, last + 1)),
                self.timeline, self.layers, fps, "RECORDED RUN", progress)
        finally:
            dialog.close()
        self.statusBar().showMessage(
            f"video saved ({result['frames']} frames, {result['encoder']}): {result['path']}")
        return result

    def _save_dialog(self) -> None:
        default = recording_module.default_path("recordings", self.meta, ".npz")
        default.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save recording", str(default),
                                                        "Viewer recordings (*.npz)")
        if path:
            self.save_recording(path)

    def _open_dialog(self) -> None:
        folder = recording_module.OUTPUT_ROOT / "recordings"
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open recording", str(folder),
                                                        "Viewer recordings (*.npz)")
        if path:
            try:
                self.open_recording(path)
            except Exception as error:          # noqa: BLE001 - shown to the user
                QtWidgets.QMessageBox.warning(self, "Cannot open the recording", str(error))

    def _video_dialog(self) -> None:
        default = recording_module.default_path("videos", self.meta, ".mp4")
        default.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", str(default),
                                                        "Video (*.mp4)")
        if path:
            try:
                self.export_video(path)
            except Exception as error:          # noqa: BLE001 - shown to the user
                QtWidgets.QMessageBox.warning(self, "Video export failed", str(error))

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event) -> None:                        # noqa: N802 - Qt name
        self._display_timer.stop()
        self._replay_timer.stop()
        self.worker.shutdown()
        if self.recording is not None:
            self.recording.close()
        super().closeEvent(event)


def wait_until(condition: Callable[[], bool], timeout: float = 30.0) -> bool:
    """Run the Qt event loop until ``condition()`` holds. For tests and scripted checks."""
    deadline = time.monotonic() + timeout
    app = QtWidgets.QApplication.instance()
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if condition():
            return True
        time.sleep(0.005)
    return False


def apply_dark_theme(app) -> None:
    """Fusion style with a dark palette, so the controls match the panels."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    colour = QtGui.QColor
    for role, value in ((QtGui.QPalette.Window, colour(34, 37, 43)),
                        (QtGui.QPalette.WindowText, colour(232, 232, 228)),
                        (QtGui.QPalette.Base, colour(22, 24, 28)),
                        (QtGui.QPalette.AlternateBase, colour(34, 37, 43)),
                        (QtGui.QPalette.Text, colour(232, 232, 228)),
                        (QtGui.QPalette.Button, colour(48, 52, 60)),
                        (QtGui.QPalette.ButtonText, colour(232, 232, 228)),
                        (QtGui.QPalette.ToolTipBase, colour(48, 52, 60)),
                        (QtGui.QPalette.ToolTipText, colour(232, 232, 228)),
                        (QtGui.QPalette.Highlight, colour(57, 135, 229)),
                        (QtGui.QPalette.HighlightedText, colour(255, 255, 255)),
                        (QtGui.QPalette.PlaceholderText, colour(150, 152, 148))):
        palette.setColor(role, value)
    app.setPalette(palette)


def run_app(resources_factory: Callable[[], Any], **options: Any) -> int:
    """Open the window and run until it is closed."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    apply_dark_theme(app)
    window = MainWindow(resources_factory, **options)
    window.show()
    return app.exec()
