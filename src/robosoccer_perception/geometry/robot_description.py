"""What the perception code needs to know about a robot body: one ``configs/robots/*.yaml``.

Two kinds exist:

* ``placeholder``  the capsule platform of Phases 1 to 3 (a camera on a stick);
* ``mjcf``         a real robot model imported into ``assets/robots/<name>/`` by
                   ``scripts/import_humanoid_model.py``: UTRA's humanoid_prototype_1.

This module is pure data and NumPy. It reads the body geometry that the importer MEASURED from
the model (``body_geometry.yaml``); it does not load meshes or MuJoCo. The simulator's side of
a robot lives in `simulation.humanoid`.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from robosoccer_perception import config
from robosoccer_perception.geometry.mount import BodyGeometry

KINDS = ("placeholder", "mjcf")
BODY_GEOMETRY_FILE = "body_geometry.yaml"


@dataclasses.dataclass(frozen=True)
class RobotDescription:
    name: str
    kind: str
    body: BodyGeometry
    asset_dir: Path | None = None               # where the imported model lives (mjcf only)
    stance: dict[str, float] = dataclasses.field(default_factory=dict)      # joint -> radians
    appearance: dict[str, Any] = dataclasses.field(default_factory=dict)
    gait: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def is_placeholder(self) -> bool:
        return self.kind == "placeholder"

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "humanoid_prototype_1") -> "RobotDescription":
        data = config.load_config("robots", name_or_path)
        kind = str(data.get("kind", "placeholder"))
        if kind not in KINDS:
            raise ValueError(f"robot kind must be one of {KINDS}, got '{kind}'")
        asset_dir = None
        if kind == "mjcf":
            asset_dir = Path(data["asset_dir"])
            if not asset_dir.is_absolute():
                asset_dir = config.assets_dir() / asset_dir
            geometry_file = asset_dir / BODY_GEOMETRY_FILE
            if not geometry_file.is_file():
                raise FileNotFoundError(
                    f"{geometry_file} not found: the robot model '{data['name']}' has not been "
                    "imported. Run scripts/import_humanoid_model.py (docs/humanoid_integration.md)")
            body = BodyGeometry.from_dict(config.load_yaml(geometry_file)["body_geometry"])
        else:
            body = BodyGeometry.from_dict(data["body_geometry"])
        return cls(name=str(data["name"]), kind=kind, body=body, asset_dir=asset_dir,
                   stance={str(k): float(v) for k, v in (data.get("stance") or {}).items()},
                   appearance=dict(data.get("appearance") or {}),
                   gait={str(k): float(v) for k, v in (data.get("gait") or {}).items()})


def available_robots() -> list[str]:
    return sorted(p.stem for p in (config.config_dir() / "robots").glob("*.yaml"))


def robot_is_installed(name: str) -> bool:
    """False when a robot's imported model is missing (so tests can skip instead of failing)."""
    try:
        RobotDescription.load(name)
    except FileNotFoundError:
        return False
    return True
