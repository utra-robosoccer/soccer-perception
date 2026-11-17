"""The viewer without a window: record a run, export a video, save panel pictures.

Used by ``scripts/visualize_pipeline.py --headless``, by the window's "Export video" action
and by the tests. Needs OpenCV; uses ``ffmpeg`` for H.264 when it is installed and falls back
to OpenCV's MPEG-4 writer when it is not.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np

from robosoccer_perception.viewer import recording as recording_module
from robosoccer_perception.viewer.frame import Timeline, ViewerFrame, pack
from robosoccer_perception.viewer.render import Layers, PanelRenderer, mosaic
from robosoccer_perception.viewer.session import LiveSession


def save_png(path, image_rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {path}")
    return path


class VideoSink:
    """Writes RGB frames of one size to an ``.mp4``."""

    def __init__(self, path, size: tuple[int, int], fps: float, prefer_ffmpeg: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)      # codecs need even sizes
        self.frames = 0
        self._process = None
        self._writer = None
        ffmpeg = shutil.which("ffmpeg") if prefer_ffmpeg else None
        if ffmpeg:
            self._process = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:.4f}", "-i", "-", "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 str(self.path)], stdin=subprocess.PIPE)
            self.encoder = "ffmpeg libx264"
        else:
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                           float(fps), self.size)
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open a video writer for {self.path}")
            self.encoder = "OpenCV mp4v"

    def write(self, image_rgb: np.ndarray) -> None:
        image = np.ascontiguousarray(image_rgb[:self.size[1], :self.size[0]])
        if self._process is not None:
            try:
                self._process.stdin.write(image.tobytes())
            except BrokenPipeError as error:
                raise RuntimeError("ffmpeg stopped while encoding the video") from error
        else:
            self._writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            if self._process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed to write {self.path}")
            self._process = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        if exc_type is None:
            self.close()
        elif self._process is not None:
            self._process.kill()
        elif self._writer is not None:
            self._writer.release()


def export_video(path, renderer: PanelRenderer, frames: Iterable[ViewerFrame],
                 timeline: Timeline, layers: Layers, fps: float, status: str = "",
                 progress: Callable[[int], bool | None] | None = None) -> dict[str, Any]:
    """Render the four-panel picture of every frame into a video.

    The diagnostics curves grow with the video (they stop at the frame being shown).
    ``progress(count)`` is called after every frame; returning False cancels the export.
    """
    sink: VideoSink | None = None
    cancelled = False
    try:
        for frame in frames:
            picture = mosaic(renderer.panels(frame, layers, timeline, status,
                                             reveal_all=False))
            if sink is None:
                sink = VideoSink(path, (picture.shape[1], picture.shape[0]), fps)
            sink.write(picture)
            if progress is not None and progress(sink.frames) is False:
                cancelled = True
                break
    except BaseException:
        if sink is not None:
            sink.__exit__(RuntimeError, None, None)
        raise
    if sink is None:
        raise ValueError("no frames to export")
    sink.close()
    return {"path": str(sink.path), "frames": sink.frames, "size": list(sink.size),
            "fps": float(fps), "encoder": sink.encoder, "cancelled": cancelled}


def run_headless(resources, scenario, seed: int = 0, *, max_steps: int | None = None,
                 record=None, video=None, screenshots: int = 0, screenshot_dir=None,
                 layers: Layers | None = None, with_truth: bool = True,
                 max_particles: int = 600, video_speed: float = 1.0,
                 progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Run one scenario to its end without a window.

    Optionally writes a recording (``record``), a video (``video``) and ``screenshots``
    evenly spaced four-panel pictures. Returns a summary with the files written.
    """
    layers = layers or Layers()
    packed: list[Mapping[str, np.ndarray]] = []
    written: dict[str, Any] = {"screenshots": []}
    with LiveSession(scenario, seed, resources, max_particles=max_particles,
                     with_truth=with_truth, max_steps=max_steps) as session:
        meta = session.meta()
        timeline = Timeline(session.total_steps, session.dt)
        renderer = PanelRenderer(session.field, meta)
        wanted = set(np.unique(np.linspace(0, session.total_steps - 1, screenshots + 2)[1:-1]
                               .round().astype(int)).tolist()) if screenshots > 0 else set()
        sink: VideoSink | None = None
        try:
            while True:
                frame = session.step()
                if frame is None:
                    break
                timeline.record(frame)
                if record is not None:
                    packed.append(pack(frame))
                picture = None
                if video is not None or frame.step in wanted:
                    picture = mosaic(renderer.panels(frame, layers, timeline, "HEADLESS",
                                                     reveal_all=False))
                if video is not None:
                    if sink is None:
                        sink = VideoSink(video, (picture.shape[1], picture.shape[0]),
                                         video_speed / session.dt)
                    sink.write(picture)
                if frame.step in wanted:
                    target = Path(screenshot_dir) if screenshot_dir is not None else \
                        recording_module.OUTPUT_ROOT / "screenshots"
                    written["screenshots"].append(str(save_png(
                        target / f"{meta['scenario']}_seed{meta['seed']}_step{frame.step:04d}.png",
                        picture)))
                if progress is not None:
                    progress(frame.step + 1, session.total_steps)
        finally:
            if sink is not None:
                sink.close()
                written["video"] = {"path": str(sink.path), "frames": sink.frames,
                                    "encoder": sink.encoder}
    if record is not None:
        written["recording"] = str(recording_module.save_recording(record, meta, packed,
                                                                   timeline))
    v = timeline.values
    done = slice(0, timeline.last + 1)
    summary = {"scenario": meta["scenario"], "seed": meta["seed"], "frames": timeline.last + 1,
               "pipeline_ms_mean": float(np.nanmean(v["total_ms"][done])),
               "final_confidence": float(v["confidence"][timeline.last])}
    if with_truth:
        summary["position_rmse_m"] = float(np.sqrt(np.nanmean(v["position_error"][done] ** 2)))
        summary["yaw_rmse_deg"] = float(np.sqrt(np.nanmean(v["yaw_error_deg"][done] ** 2)))
    return {**summary, **written}
