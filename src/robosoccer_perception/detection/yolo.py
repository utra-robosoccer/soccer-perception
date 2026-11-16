"""The Ultralytics YOLO baseline: weights, training, validation, export and inference.

Everything that touches the ``ultralytics`` package is in this module; the rest of the code
sees only `inference.types.Detection`. Swapping the detector means writing another class with
`YoloDetector`'s ``predict`` method.

Licensing: the ``ultralytics`` package and the pretrained YOLO11 / YOLO26 weights are released
under AGPL-3.0 (commercial licences are sold separately). See docs/object_detection.md before
distributing a model or building it into a product.

Network access: Ultralytics can download weights, fonts and missing Python packages on its own
and can report anonymous usage statistics. This module switches all of that off by setting
``YOLO_OFFLINE=1`` before the import (unless the variable is already set), so runs are
reproducible and nothing leaves the machine. Weights are fetched explicitly, once, with
`download_weights` (``scripts/train_detector.py --download``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")          # must precede the ultralytics import

from robosoccer_perception import config  # noqa: E402
from robosoccer_perception.datasets import schema  # noqa: E402
from robosoccer_perception.inference.profiling import resolve_device  # noqa: E402
from robosoccer_perception.inference.types import Detection  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR_ENV = "ROBOSOCCER_PERCEPTION_WEIGHTS_DIR"
# Pretrained checkpoints this project knows about: file -> (approximate size, what it is).
KNOWN_WEIGHTS = {
    "yolo11n.pt": ("5.4 MB", "YOLO11 nano detector, COCO-pretrained, AGPL-3.0"),
    "yolo26n.pt": ("5.3 MB", "YOLO26 nano detector, COCO-pretrained, AGPL-3.0"),
}


def weights_dir() -> Path:
    """Where pretrained checkpoints are kept: ``weights/`` in the repository (not tracked)."""
    return Path(os.environ.get(WEIGHTS_DIR_ENV) or REPOSITORY_ROOT / "weights")


def load_detection_config(name_or_path: str = "yolo11n",
                          overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config.deep_update(config.load_config("detection", name_or_path), overrides)
    for key in ("name", "model", "imgsz", "epochs", "batch"):
        if key not in cfg:
            raise ValueError(f"detection configuration is missing '{key}'")
    return cfg


def resolve_model(model: str) -> str:
    """Turn a configuration's ``model`` entry into something ``ultralytics.YOLO`` can load
    without touching the network.

    * ``*.yaml``  an architecture shipped with Ultralytics (``yolo11n.yaml``): random weights.
    * ``*.pt``    a checkpoint: an existing path, or a file name looked up in `weights_dir()`.
    """
    if model.endswith((".yaml", ".yml")):
        return model
    path = Path(model)
    if path.is_file():
        return str(path)
    candidate = weights_dir() / path.name
    if candidate.is_file():
        return str(candidate)
    size, purpose = KNOWN_WEIGHTS.get(path.name, ("unknown size", "unknown checkpoint"))
    raise FileNotFoundError(
        f"checkpoint '{model}' not found (looked in {weights_dir()}). It is: {purpose}, {size}. "
        f"Fetch it once with:  python scripts/train_detector.py --download {path.name}")


def download_weights(name: str) -> Path:
    """Download one known pretrained checkpoint from the Ultralytics release assets into
    `weights_dir()`. This is the only function in the project that uses the network."""
    if name not in KNOWN_WEIGHTS:
        raise ValueError(f"unknown checkpoint {name!r}; known: {sorted(KNOWN_WEIGHTS)}")
    target = weights_dir() / name
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils import downloads

    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
    downloads.safe_download(url, file=target, min_bytes=1e6)
    if not target.is_file():
        raise RuntimeError(f"download of {url} failed")
    return target


def _check_classes(names: dict[int, str], source: str) -> None:
    found = tuple(names[i] for i in sorted(names))
    if found != schema.DETECTION_CLASSES:
        raise ValueError(f"{source} has classes {found}, but this project uses "
                         f"{schema.DETECTION_CLASSES}; it was not trained on these datasets")


def train_detector(cfg: dict[str, Any], dataset_root, output_dir=None) -> dict[str, Any]:
    """Train (fine-tune) the detector described by ``cfg`` on a generated dataset.

    Returns the run directory, the checkpoint paths and the final validation metrics that
    Ultralytics reports. Checkpoints are ``weights/best.pt`` and ``weights/last.pt`` in the run
    directory; ``results.csv`` there has the per-epoch losses and metrics.
    """
    from ultralytics import YOLO

    from robosoccer_perception.datasets.io import SoccerDataset

    dataset = SoccerDataset(dataset_root)
    if tuple(dataset.header["detection_classes"]) != schema.DETECTION_CLASSES:
        raise ValueError("the dataset's detection classes differ from datasets.schema")
    output = Path(output_dir or cfg.get("output", "outputs/detection")).resolve()
    device = resolve_device(cfg.get("device", "auto"))
    model = YOLO(resolve_model(str(cfg["model"])))
    model.train(
        data=str(dataset.yolo_yaml), project=str(output), name=str(cfg["name"]), exist_ok=True,
        epochs=int(cfg["epochs"]), patience=int(cfg.get("patience", 20)),
        imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]), workers=int(cfg.get("workers", 4)),
        device=device, amp=bool(cfg.get("amp", True)), seed=int(cfg.get("seed", 0)),
        deterministic=bool(cfg.get("deterministic", True)),
        optimizer=str(cfg.get("optimizer", "auto")), cache=cfg.get("cache", False),
        save_period=int(cfg.get("save_period", -1)), resume=bool(cfg.get("resume", False)),
        fraction=float(cfg.get("fraction", 1.0)), plots=bool(cfg.get("plots", True)),
        pretrained=True, verbose=bool(cfg.get("verbose", True)), **(cfg.get("augment") or {}))
    run_dir = Path(model.trainer.save_dir)
    metrics = getattr(model.trainer, "metrics", None) or {}
    return {"run_dir": str(run_dir), "best": str(run_dir / "weights" / "best.pt"),
            "last": str(run_dir / "weights" / "last.pt"), "device": device,
            "final_metrics": {key: float(value) for key, value in metrics.items()}}


def validate_detector(weights, dataset_root, split: str = "val", imgsz: int = 640,
                      batch: int = 16, device: str = "auto", output_dir=None,
                      workers: int = 2) -> dict[str, Any]:
    """Ultralytics' own validation (mAP, per-class AP, its plots) on one split of a dataset.

    The project's primary numbers come from `detection.evaluation`; this is the framework's
    independent computation of the same quantities, kept as a cross-check.
    """
    from ultralytics import YOLO

    from robosoccer_perception.datasets.io import SoccerDataset

    dataset = SoccerDataset(dataset_root)
    model = YOLO(str(weights))
    _check_classes(model.names, str(weights))
    arguments = dict(data=str(dataset.yolo_yaml), split=split, imgsz=imgsz, batch=batch,
                     device=resolve_device(device), workers=workers, verbose=False,
                     plots=output_dir is not None)
    if output_dir is not None:
        output = Path(output_dir).resolve()
        arguments.update(project=str(output.parent), name=output.name, exist_ok=True)
    result = model.val(**arguments)
    per_class = {schema.DETECTION_CLASSES[int(c)]: {
        "precision": float(result.box.p[i]), "recall": float(result.box.r[i]),
        "ap50": float(result.box.ap50[i]), "ap50_95": float(result.box.ap[i])}
        for i, c in enumerate(result.box.ap_class_index)}
    return {"map50": float(result.box.map50), "map50_95": float(result.box.map),
            "precision": float(result.box.mp), "recall": float(result.box.mr),
            "per_class": per_class, "speed_ms_per_image": dict(result.speed)}


def export_detector(weights, export_format: str = "onnx", imgsz: int = 640,
                    half: bool = False) -> Path:
    """Export a trained detector (``onnx`` and ``torchscript`` need no extra packages)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    _check_classes(model.names, str(weights))
    options = {"half": True} if half else {}
    return Path(model.export(format=export_format, imgsz=imgsz, simplify=False, **options))


class YoloDetector:
    """A trained YOLO checkpoint behind the project's detector interface.

    ``predict(image_rgb)`` takes an ``(H, W, 3)`` uint8 RGB image of any size and returns
    `Detection`s in that image's pixel coordinates. Ultralytics letterboxes the image to
    ``imgsz`` internally and maps the boxes back.

    The grey margin (``margin``). During training and in Ultralytics' validator an image never
    fills the network input edge to edge: mosaic, random shifts and batch-shape rounding leave
    grey (value 114) around it. The network therefore learns that an object cut off by a
    visible image border ends there. Fed an image with no margin, it instead guesses the hidden
    part: on this project's data a goal post whose foot alone is in view got a full-height box,
    and recall on border-cut posts was 90 % without the margin against 97 % with it. So this class surrounds every image
    with ``margin`` pixels of grey (measured at network scale; 16 reproduces the validator's
    framing for 16:9 images at 640), runs the network at ``imgsz + 2 * margin`` so that the
    image itself keeps its scale, and shifts and clips the boxes back. ``margin=0`` turns it
    off. See docs/object_detection.md.
    """

    PAD_VALUE = 114

    def __init__(self, weights, device: str = "auto", conf: float = 0.25, iou: float = 0.5,
                 imgsz: int = 640, half: bool = False, max_det: int = 100, margin: int = 16):
        from ultralytics import YOLO

        self.weights = str(weights)
        self.device = resolve_device(device)
        self.conf, self.iou, self.imgsz, self.max_det = conf, iou, imgsz, max_det
        self.margin = int(margin)
        self.half = bool(half and self.device.startswith("cuda"))
        self.model = YOLO(self.weights)
        if Path(self.weights).suffix != ".yaml":            # an untrained architecture has COCO names
            _check_classes(self.model.names, self.weights)
        self.last_speed_ms: dict[str, float] = {}

    def predict(self, image_rgb: np.ndarray, conf: float | None = None) -> list[Detection]:
        return self.predict_batch([image_rgb], conf)[0]

    def _frame(self, image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
        """BGR copy of the image inside its grey margin, and the margin in image pixels."""
        bgr = np.ascontiguousarray(image_rgb[..., ::-1])    # Ultralytics expects OpenCV-style BGR
        border = int(round(self.margin * max(image_rgb.shape[:2]) / self.imgsz))
        if border <= 0:
            return bgr, 0
        import cv2
        return cv2.copyMakeBorder(bgr, border, border, border, border, cv2.BORDER_CONSTANT,
                                  value=(self.PAD_VALUE,) * 3), border

    def predict_batch(self, images_rgb: list[np.ndarray],
                      conf: float | None = None) -> list[list[Detection]]:
        framed = [self._frame(image) for image in images_rgb]
        options = {"half": True} if self.half else {}
        results = self.model.predict(
            [image for image, _ in framed], conf=self.conf if conf is None else conf,
            iou=self.iou, imgsz=self.imgsz + 2 * self.margin, device=self.device,
            max_det=self.max_det, verbose=False, **options)
        self.last_speed_ms = dict(results[0].speed) if results else {}
        output = []
        for result, image, (_, border) in zip(results, images_rgb, framed):
            height, width = image.shape[:2]
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(float) - border
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, width)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, height)
            output.append([
                Detection(int(c), float(s), tuple(float(v) for v in box))
                for box, s, c in zip(xyxy, boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
                if box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0])    # not swallowed by the margin
        return output
