"""Loading the Phase 4 settings (``configs/evaluation/humanoid_fusion.yaml``)."""
from __future__ import annotations

import os
from typing import Any

from robosoccer_perception import config
from robosoccer_perception.estimation.calibration_monitor import MonitorConfig
from robosoccer_perception.estimation.camera_pose import FusionConfig
from robosoccer_perception.vision_benchmark import setup as vision_setup


def load_settings(name_or_path: str | os.PathLike = "humanoid_fusion",
                  overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Phase 4 settings on top of the Phase 3 settings they name as ``base``."""
    own = config.load_config("evaluation", name_or_path)
    base = vision_setup.load_settings(own.get("base", "vision_localization"))
    return config.deep_update(config.deep_update(base, own), overrides)


def fusion_config(settings: dict[str, Any]) -> FusionConfig:
    return FusionConfig.from_dict(settings.get("fusion"))


def monitor_config(settings: dict[str, Any]) -> MonitorConfig:
    return MonitorConfig.from_dict(settings.get("monitor"))
