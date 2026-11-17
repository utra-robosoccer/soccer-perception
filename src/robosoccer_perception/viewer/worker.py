"""The thread that runs the simulation and the pipeline, so the window never waits for them.

The worker owns the `LiveSession` (and with it the MuJoCo renderer, which must stay in one
thread) and the models. The window talks to it through a command queue and is told about new
frames through two callbacks, which are called IN THE WORKER THREAD:

    on_frame(run_id, frame, packed)     one step is done; ``packed`` is `frame.pack(frame)`
    on_event(kind, payload)             "status", "loaded", "finished", "running", "error"

Three rules keep the window responsive and the frames consistent:

* a frame is produced by one `session.step()` and handed over whole, so the image, the
  predictions and the estimate in it always belong to the same step;
* the worker stops producing when ``backlog`` frames have not been acknowledged yet
  (`ack()`), so a slow consumer makes the simulation wait instead of piling frames up;
* every `load` and `reset` starts a new ``run_id``; frames of an older run that are still
  on their way can be recognised and dropped.

This module does not import a GUI toolkit.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable

from robosoccer_perception.viewer.frame import ViewerFrame, pack
from robosoccer_perception.viewer.session import LiveSession
from robosoccer_perception.vision_benchmark.scenarios import get_scenario

FrameCallback = Callable[[int, ViewerFrame, dict], None]
EventCallback = Callable[[str, Any], None]


class SimulationWorker(threading.Thread):
    def __init__(self, resources_factory: Callable[[], Any], on_frame: FrameCallback,
                 on_event: EventCallback, *, max_particles: int = 600, backlog: int = 4,
                 max_steps: int | None = None):
        super().__init__(name="viewer-simulation", daemon=True)
        self._resources_factory = resources_factory
        self._on_frame, self._on_event = on_frame, on_event
        self._max_particles, self._max_steps = max_particles, max_steps
        self._commands: queue.Queue[tuple] = queue.Queue()
        self._slots = threading.Semaphore(backlog)
        self._run_counter = 0
        self._counter_lock = threading.Lock()
        # ---- touched by the worker thread only
        self._resources = None
        self._session: LiveSession | None = None
        self._run_id = 0
        self._running = False
        self._speed = 1.0
        self._next_due = 0.0

    # ------------------------------------------------------------------ called by the window
    def _new_run(self) -> int:
        with self._counter_lock:
            self._run_counter += 1
            return self._run_counter

    def load(self, scenario_id: str, seed: int) -> int:
        """Start a scenario from its first frame (paused). Returns the new run id."""
        run_id = self._new_run()
        self._commands.put(("load", run_id, scenario_id, int(seed)))
        return run_id

    def reset(self) -> int:
        """Restart the loaded scenario (paused). Returns the new run id."""
        run_id = self._new_run()
        self._commands.put(("reset", run_id))
        return run_id

    def run(self) -> None:                                      # noqa: D401 - see `resume`
        """Thread entry point. Use `resume()` to start the simulation."""
        self._loop()

    def resume(self) -> None:
        self._commands.put(("resume",))

    def pause(self) -> None:
        self._commands.put(("pause",))

    def step_once(self) -> None:
        """Pause and produce exactly one more frame."""
        self._commands.put(("step",))

    def set_speed(self, speed: float) -> None:
        """Simulated seconds per wall-clock second; ``float('inf')`` = as fast as possible."""
        self._commands.put(("speed", float(speed)))

    def ack(self) -> None:
        """The consumer has taken one frame."""
        self._slots.release()

    def shutdown(self, timeout: float = 10.0) -> None:
        self._commands.put(("quit",))
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    # ------------------------------------------------------------------ the worker thread
    def _loop(self) -> None:
        try:
            while True:
                timeout = None
                if self._running:
                    timeout = max(0.0, self._next_due - time.perf_counter())
                try:
                    command = self._commands.get(timeout=timeout) if timeout != 0.0 \
                        else self._commands.get_nowait()
                except queue.Empty:
                    command = None
                if command is not None:
                    if command[0] == "quit":
                        break
                    self._handle(command)
                    continue
                if self._running:
                    self._produce()
        finally:
            if self._session is not None:
                self._session.close()
                self._session = None

    def _handle(self, command: tuple) -> None:
        kind = command[0]
        try:
            if kind == "load":
                _, run_id, scenario_id, seed = command
                self._set_running(False)
                if self._resources is None:
                    self._on_event("status", "loading the vision models ...")
                    self._resources = self._resources_factory()
                self._on_event("status", f"building the scene for {scenario_id} ...")
                if self._session is not None:
                    self._session.close()
                    self._session = None
                self._session = LiveSession(get_scenario(scenario_id), seed, self._resources,
                                            max_particles=self._max_particles,
                                            max_steps=self._max_steps)
                self._begin(run_id)
            elif kind == "reset":
                self._set_running(False)
                if self._session is not None:
                    self._session.reset()
                    self._begin(command[1])
            elif kind == "resume":
                if self._session is not None and not self._session.finished:
                    self._next_due = time.perf_counter()
                    self._set_running(True)
                elif self._session is not None:
                    self._on_event("finished", self._run_id)
            elif kind == "pause":
                self._set_running(False)
            elif kind == "step":
                self._set_running(False)
                self._produce()
            elif kind == "speed":
                self._speed = max(command[1], 1e-3)
                self._next_due = time.perf_counter()
        except Exception:                                       # noqa: BLE001 - reported to the window
            self._set_running(False)
            self._on_event("error", traceback.format_exc())

    def _begin(self, run_id: int) -> None:
        self._run_id = run_id
        self._on_event("loaded", {"run_id": run_id, "meta": self._session.meta(),
                                  "field": self._session.field})
        self._produce()                                         # show the first frame at once

    def _set_running(self, running: bool) -> None:
        if running != self._running:
            self._running = running
            self._on_event("running", running)

    def _produce(self) -> None:
        session = self._session
        if session is None:
            return
        if session.finished:
            self._set_running(False)
            self._on_event("finished", self._run_id)
            return
        # Wait for the consumer, but keep listening for commands while waiting.
        while not self._slots.acquire(timeout=0.02):
            if not self._commands.empty():
                return
        try:
            frame = session.step()
            packed = pack(frame)
        except Exception:                                       # noqa: BLE001
            self._slots.release()
            self._set_running(False)
            self._on_event("error", traceback.format_exc())
            return
        self._on_frame(self._run_id, frame, packed)
        period = session.dt / self._speed
        now = time.perf_counter()
        self._next_due = max(self._next_due + period, now - period) if period > 0.0 else now
        if session.finished:
            self._set_running(False)
            self._on_event("finished", self._run_id)
