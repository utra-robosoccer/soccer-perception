"""Recorded runs: one ``.npz`` file per run, written from packed frames and read lazily.

Layout of the archive (plain arrays only; it is loaded with ``allow_pickle=False``):

    meta                    JSON text: scenario, seed, dt, steps, resolution, models, ...
    timeline__<series>      one value per step, for the diagnostics panel
    f000123__<name>         the arrays of `frame.pack` for step 123

Images are stored as JPEG and masks as PNG, so a 60 s run at 640x360 is 30 to 40 MB. A
recording holds everything the four panels show, including ground truth when the run had it;
replaying one needs neither the simulator nor the models.

Recordings are generated files: ``outputs/`` and ``*.npz`` are in ``.gitignore``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robosoccer_perception.viewer.frame import Timeline, ViewerFrame, unpack

FORMAT_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "viewer"


def default_path(kind: str, meta: Mapping[str, Any], suffix: str, step: int | None = None
                 ) -> Path:
    """``outputs/viewer/<kind>/<scenario>_seed<k>_<time>[_stepNNNN]<suffix>``."""
    name = f"{meta.get('scenario', 'run')}_seed{meta.get('seed', 0)}_{time.strftime('%Y%m%d-%H%M%S')}"
    if step is not None:
        name += f"_step{step:04d}"
    return OUTPUT_ROOT / kind / (name + suffix)


def save_recording(path, meta: Mapping[str, Any], frames: Sequence[Mapping[str, np.ndarray]],
                   timeline: Timeline) -> Path:
    """Write packed frames (see `frame.pack`) to ``path``. Returns the path written."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps({**meta, "format": FORMAT_VERSION,
                                     "frames": len(frames)})),
        **timeline.to_arrays()}
    for index, packed in enumerate(frames):
        for name, value in packed.items():
            arrays[f"f{index:06d}__{name}"] = value
    np.savez(path, **arrays)            # not compressed: the bulk is already JPEG and PNG
    return path


class _FrameArrays(Mapping):
    """The arrays of one frame inside an open archive."""

    def __init__(self, archive, prefix: str, names: frozenset[str]):
        self._archive, self._prefix, self._names = archive, prefix, names

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._names:
            raise KeyError(name)
        return self._archive[self._prefix + name]

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class Recording:
    """A recorded run opened for replay. Frames are decoded when asked for."""

    def __init__(self, path):
        self.path = Path(path)
        self._archive = np.load(self.path, allow_pickle=False)
        if "meta" not in self._archive.files:
            raise ValueError(f"{self.path} is not a viewer recording")
        self.meta: dict[str, Any] = json.loads(str(self._archive["meta"]))
        if self.meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"{self.path}: unsupported recording format "
                             f"{self.meta.get('format')!r}")
        names: dict[int, set[str]] = {}
        for key in self._archive.files:
            if key.startswith("f") and "__" in key and key[1:7].isdigit():
                names.setdefault(int(key[1:7]), set()).add(key.split("__", 1)[1])
        self._names = [frozenset(names[i]) for i in sorted(names)]
        if sorted(names) != list(range(len(names))):
            raise ValueError(f"{self.path}: frames are missing from the recording")
        self.timeline = Timeline.from_arrays(self._archive, float(self.meta["dt"]))

    def __len__(self) -> int:
        return len(self._names)

    def packed(self, index: int) -> Mapping[str, np.ndarray]:
        return _FrameArrays(self._archive, f"f{index:06d}__", self._names[index])

    def frame(self, index: int) -> ViewerFrame:
        return unpack(self.packed(index))

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
