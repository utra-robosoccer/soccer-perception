"""Loading the Phase 3 benchmark settings (``configs/evaluation/vision_localization.yaml``)
and the trained models they name."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from robosoccer_perception import config
from robosoccer_perception.perception.config import (
    FeatureExtractionConfig, MeasurementNoiseConfig)
from robosoccer_perception.vision_benchmark.runner import BenchmarkResources

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_settings(name_or_path: str | os.PathLike = "vision_localization",
                  overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return config.deep_update(config.load_config("evaluation", name_or_path), overrides)


def checkpoint_path(entry: str) -> Path:
    """A checkpoint path from the settings: absolute, or relative to the repository root."""
    path = Path(entry)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def missing_checkpoints(settings: dict[str, Any]) -> list[Path]:
    return [p for p in (checkpoint_path(settings["detector"]),
                        checkpoint_path(settings["segmenter"])) if not p.is_file()]


def load_resources(settings: dict[str, Any] | None = None, vision=None) -> BenchmarkResources:
    """Build `BenchmarkResources`: load the models (unless ``vision`` is given) and the
    extraction, noise and filter configurations the settings name."""
    settings = settings or load_settings()
    if vision is None:
        missing = missing_checkpoints(settings)
        if missing:
            raise FileNotFoundError(
                "trained checkpoints not found: " + ", ".join(str(p) for p in missing)
                + ". Train them first (README, 'Vision') or point the settings at yours.")
        from robosoccer_perception.inference.pipeline import VisionPipeline

        vision = VisionPipeline.from_checkpoints(
            checkpoint_path(settings["detector"]), checkpoint_path(settings["segmenter"]),
            device=settings.get("device", "auto"),
            conf=float(settings.get("detector_confidence", 0.25)),
            imgsz=int(settings.get("detector_imgsz", 640)), half=bool(settings.get("half", False)))
    return BenchmarkResources(
        vision=vision, camera=str(settings.get("camera", "zed_mini_sim")),
        resolution=tuple(int(v) for v in settings.get("resolution", (640, 360))),
        extraction=FeatureExtractionConfig.load(settings.get("feature_extraction",
                                                             "feature_extraction")),
        noise=MeasurementNoiseConfig.load(settings.get("measurement_noise", "measurement_noise")),
        filter_config=str(settings.get("filter_config", "particle_filter")),
        filter_overrides=settings.get("filter_overrides") or None,
        vision_filter_overrides=settings.get("vision_filter_overrides") or None)
