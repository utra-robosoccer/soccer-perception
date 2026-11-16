"""The synthetic dataset generator. See docs/synthetic_data.md.

A dataset is a set of SCENARIOS. A scenario is one look of the world (turf, lighting, backdrop,
ball, robots: one draw of the ``scene`` section of a randomization preset) and a handful of
frames taken in it from unrelated viewpoints. Splits are made of whole scenarios: no scenario
contributes frames to two splits, so a validation or test image never shares its turf texture,
lighting or robot looks with a training image.

Every random draw descends from the dataset seed through `numpy.random.SeedSequence` keys
``[seed, split, scenario, ...]``, so a scenario's content depends only on the seed and on its
own position, not on how many other scenarios or splits are generated, nor in which order.

Layout written under the output directory::

    dataset.json            header: versions, configurations, seed, label policy, class ids
    manifest.jsonl          one line per frame: paths, split, scenario, frame setup, objects
    scenarios.jsonl         one line per scenario: its seed and scene overrides
    data.yaml               Ultralytics dataset description
    images/<split>/*.jpg    camera images (left eye)
    labels/<split>/*.txt    YOLO boxes: class cx cy w h, normalised
    masks/<split>/*.png     semantic masks, pixel value = class id

Needs MuJoCo with a working OpenGL context and OpenCV.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from robosoccer_perception import config
from robosoccer_perception.datasets import image_aug, randomization, schema
from robosoccer_perception.datasets.annotation import FrameLabels, SceneLabeler
from robosoccer_perception.datasets.sampling import (
    DEFAULT_CONTENT, DEFAULT_MODE_WEIGHTS, FrameSampler, FrameSpec)
from robosoccer_perception.datasets.schema import LabelPolicy
from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.simulation.mujoco_scene import (
    LEFT_CAMERA, MujocoScene, load_scene_config)

HEADER_FILE, MANIFEST_FILE, SCENARIOS_FILE, YOLO_FILE = (
    "dataset.json", "manifest.jsonl", "scenarios.jsonl", "data.yaml")
_SCENE_STREAM, _FRAME_STREAM = 0, 1


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    scenarios: int
    frames_per_scenario: int
    randomization: str              # preset name or path


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """One ``configs/datasets/*.yaml`` file."""
    name: str
    version: str
    seed: int
    field: str
    camera: str
    scene: str
    width: int
    height: int
    splits: dict[str, SplitConfig]
    image_format: str
    jpeg_quality: int
    labels: LabelPolicy
    content: dict[str, Any]
    modes: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        data = dict(data)
        default_preset = str(data.get("randomization", "mild"))
        splits = {}
        for split, entry in data["splits"].items():
            if split not in schema.SPLITS:
                raise ValueError(f"unknown split {split!r}; expected one of {schema.SPLITS}")
            splits[split] = SplitConfig(int(entry["scenarios"]),
                                        int(entry["frames_per_scenario"]),
                                        str(entry.get("randomization", default_preset)))
        image_format = str(data.get("image_format", "jpg")).lower()
        if image_format not in ("jpg", "png"):
            raise ValueError("image_format must be 'jpg' or 'png'")
        width, height = (int(v) for v in data.get("resolution", (640, 360)))
        return cls(
            name=str(data["name"]), version=str(data.get("version", "0")),
            seed=int(data.get("seed", 0)), field=str(data.get("field", "s_field")),
            camera=str(data.get("camera", "zed_mini_sim")),
            scene=str(data.get("scene", "default")), width=width, height=height,
            splits=splits, image_format=image_format,
            jpeg_quality=int(data.get("jpeg_quality", 95)),
            labels=LabelPolicy.from_dict(data.get("labels")),
            content={**DEFAULT_CONTENT, **(data.get("content") or {})},
            modes=dict(data.get("modes") or DEFAULT_MODE_WEIGHTS))

    @classmethod
    def load(cls, name_or_path: str, overrides: dict[str, Any] | None = None) -> "DatasetConfig":
        return cls.from_dict(config.deep_update(config.load_config("datasets", name_or_path),
                                                overrides))

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["resolution"] = [data.pop("width"), data.pop("height")]
        return data

    @property
    def num_frames(self) -> int:
        return sum(s.scenarios * s.frames_per_scenario for s in self.splits.values())


def scenario_id(split: str, index: int) -> str:
    return f"{split}_{index:04d}"


def scenario_seed_key(seed: int, split: str, index: int) -> list[int]:
    """The `SeedSequence` entropy that everything in one scenario descends from."""
    return [int(seed), schema.SPLITS.index(split), int(index)]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent, timeout=5, check=False)
        return result.stdout.strip() or None
    except OSError:
        return None


def apply_frame(scene: MujocoScene, spec: FrameSpec) -> None:
    """Put a `MujocoScene` into the state a `FrameSpec` describes."""
    scene.teleport(spec.base_pose, spec.rig)
    scene.set_ball(spec.ball_xy, spec.ball_quat)
    scene.set_robots(spec.robots)


@dataclasses.dataclass(frozen=True)
class RenderedFrame:
    image: np.ndarray               # (H, W, 3) uint8 RGB, after photometric augmentation
    labels: FrameLabels
    spec: FrameSpec
    photometric: dict[str, Any]
    T_field_camera: np.ndarray      # left optical frame in the field frame, ground truth


class ScenarioRenderer:
    """One scenario: builds its scene once, then renders labelled frames on request.

    This is the unit the dataset is made of, and it is usable on its own (tests, notebooks,
    `scripts/run_vision.py --render`).
    """

    def __init__(self, cfg: DatasetConfig, split: str, index: int):
        self.cfg, self.split, self.index = cfg, split, index
        self.id = scenario_id(split, index)
        self.seed_key = scenario_seed_key(cfg.seed, split, index)
        self.preset = randomization.load_preset(cfg.splits[split].randomization)
        self.field = FieldGeometry.load(cfg.field)
        self.camera_cfg = StereoCameraConfig.load(cfg.camera).with_resolution(cfg.width,
                                                                              cfg.height)
        scene_rng = np.random.default_rng([*self.seed_key, _SCENE_STREAM])
        self.scene_overrides = randomization.sample_scene_overrides(
            scene_rng, self.preset, self.field, int(cfg.content["max_robots"]))
        scene_cfg = load_scene_config(cfg.scene, config.deep_update(
            self.scene_overrides, {"offscreen_scale": cfg.labels.supersample}))
        self.scene = MujocoScene(self.field, self.camera_cfg, scene_cfg, scenario_id=self.id)
        self.labeler = SceneLabeler(self.scene, cfg.labels)
        self.sampler = FrameSampler(self.field, self.camera_cfg, self.preset, cfg.content,
                                    cfg.modes, ball_radius=float(scene_cfg["ball"]["radius"]))

    def frame_rng(self, frame_index: int) -> np.random.Generator:
        return np.random.default_rng([*self.seed_key, _FRAME_STREAM, int(frame_index)])

    def render(self, frame_index: int, mode: str | None = None) -> RenderedFrame:
        rng = self.frame_rng(frame_index)
        spec = self.sampler.sample(rng, mode)
        apply_frame(self.scene, spec)
        clean = self.scene.render(LEFT_CAMERA, show_platform=False)
        labels = self.labeler.annotate()
        photometric = image_aug.sample_photometric(rng, self.preset.get("image"))
        return RenderedFrame(image_aug.apply_photometric(clean, photometric), labels, spec,
                             photometric, self.scene.camera_pose(LEFT_CAMERA))

    def record(self) -> dict[str, Any]:
        return {"scenario_id": self.id, "split": self.split, "index": self.index,
                "seed_key": self.seed_key, "randomization": self.preset["name"],
                "scene_overrides": self.scene_overrides}

    def close(self) -> None:
        self.scene.close()

    def __enter__(self) -> "ScenarioRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def encode_image(image_rgb: np.ndarray, image_format: str, jpeg_quality: int) -> bytes:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if image_format == "jpg":
        ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("image encoding failed")
    return buffer.tobytes()


def generate_dataset(cfg: DatasetConfig, output_dir, overwrite: bool = False,
                     progress: Callable[[str], None] | None = None) -> Path:
    """Generate the whole dataset described by ``cfg`` into ``output_dir``.

    An existing directory is replaced only when ``overwrite`` is set AND it already holds a
    dataset written by this function (it has a ``dataset.json``); anything else is refused.
    """
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass overwrite=True (--overwrite) "
                                  "to regenerate it")
        if not (output / HEADER_FILE).is_file():
            raise FileExistsError(f"{output} is not empty and is not a generated dataset "
                                  f"(no {HEADER_FILE}); refusing to delete it")
        shutil.rmtree(output)
    for folder in ("images", "labels", "masks"):
        for split in cfg.splits:
            (output / folder / split).mkdir(parents=True, exist_ok=True)

    started = time.time()
    counts = {split: {"frames": 0, "boxes": dict.fromkeys(schema.DETECTION_CLASSES, 0)}
              for split in cfg.splits}
    digest = hashlib.sha256()
    presets: dict[str, Any] = {}
    camera_metadata: dict[str, Any] = {}
    with open(output / MANIFEST_FILE, "w", encoding="utf-8") as manifest, \
            open(output / SCENARIOS_FILE, "w", encoding="utf-8") as scenarios:
        for split, split_cfg in cfg.splits.items():
            for index in range(split_cfg.scenarios):
                with ScenarioRenderer(cfg, split, index) as scenario:
                    presets[scenario.preset["name"]] = scenario.preset
                    camera_metadata = {**scenario.camera_cfg.metadata(),
                                       "mount": dataclasses.asdict(scenario.camera_cfg.mount)}
                    scenarios.write(json.dumps(scenario.record(), sort_keys=True) + "\n")
                    for frame_index in range(split_cfg.frames_per_scenario):
                        frame = scenario.render(frame_index)
                        stem = f"{scenario.id}_{frame_index:03d}"
                        paths = {"image": f"images/{split}/{stem}.{cfg.image_format}",
                                 "label": f"labels/{split}/{stem}.txt",
                                 "mask": f"masks/{split}/{stem}.png"}
                        image_bytes = encode_image(frame.image, cfg.image_format,
                                                   cfg.jpeg_quality)
                        (output / paths["image"]).write_bytes(image_bytes)
                        (output / paths["label"]).write_text(frame.labels.yolo_text())
                        if not cv2.imwrite(str(output / paths["mask"]), frame.labels.semantic):
                            raise RuntimeError(f"could not write {paths['mask']}")
                        semantic = np.bincount(frame.labels.semantic.ravel(),
                                               minlength=len(schema.SEMANTIC_CLASSES))
                        record = {
                            "frame_id": stem, "split": split, "scenario_id": scenario.id,
                            "frame_index": frame_index, **paths,
                            "image_sha1": hashlib.sha1(image_bytes).hexdigest(),
                            "width": cfg.width, "height": cfg.height,
                            "randomization": scenario.preset["name"],
                            "frame": frame.spec.to_dict(),
                            "T_field_left_optical": frame.T_field_camera.tolist(),
                            "photometric": frame.photometric,
                            "objects": [obj.to_dict() for obj in frame.labels.objects],
                            "semantic_pixels": {name: int(semantic[i]) for i, name in
                                                enumerate(schema.SEMANTIC_CLASSES)},
                        }
                        line = json.dumps(record, sort_keys=True)
                        manifest.write(line + "\n")
                        digest.update(line.encode())
                        counts[split]["frames"] += 1
                        for obj in frame.labels.labeled:
                            counts[split]["boxes"][obj.class_name] += 1
                if progress:
                    progress(f"{split}: scenario {index + 1}/{split_cfg.scenarios} "
                             f"({counts[split]['frames']} frames)")

    import mujoco

    field = FieldGeometry.load(cfg.field)
    header = {
        "format_version": schema.DATASET_FORMAT_VERSION,
        "name": cfg.name, "dataset_version": cfg.version, "seed": cfg.seed,
        "config": cfg.to_dict(),
        "field": {"name": field.name, "hash": field.content_hash(),
                  "dimensions": field.dims.to_dict()},
        "camera": camera_metadata,
        "scene": cfg.scene,
        "detection_classes": list(schema.DETECTION_CLASSES),
        "semantic_classes": list(schema.SEMANTIC_CLASSES),
        "label_policy": cfg.labels.to_dict(),
        "bbox_convention": "x0 y0 x1 y1 in pixel-edge coordinates; image spans [0,W]x[0,H]",
        "split_strategy": "by scenario: one scene appearance per scenario, each scenario in "
                          "exactly one split, seeds derived as [seed, split, scenario]",
        "randomization_presets": presets,
        "counts": counts,
        "manifest_sha256": digest.hexdigest(),
        "generator": {"git_commit": _git_commit(), "mujoco": mujoco.__version__,
                      "numpy": np.__version__, "opencv": cv2.__version__},
        "generation_seconds": round(time.time() - started, 1),
    }
    (output / HEADER_FILE).write_text(json.dumps(header, indent=2))
    # `path` is left out on purpose: Ultralytics then resolves the splits relative to this
    # file, so the dataset directory can be moved or copied.
    (output / YOLO_FILE).write_text(yaml.safe_dump({
        **{split: f"images/{split}" for split in cfg.splits},
        "names": dict(enumerate(schema.DETECTION_CLASSES))}, sort_keys=False))
    return output
