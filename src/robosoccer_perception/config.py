"""Locating and loading the YAML configuration files.

Configuration lives in the repository's ``configs/`` directory, in one sub-directory per kind
(``fields``, ``cameras``, ``localization``, ``observations``, ``scenes``). Code refers to a
configuration either by name (``"s_field"``) or by an explicit path.

The directory is found, in order, from:

1. the ``ROBOSOCCER_PERCEPTION_CONFIG_DIR`` environment variable,
2. ``configs/`` next to ``src/`` (the layout of an editable install of this repository).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR_ENV = "ROBOSOCCER_PERCEPTION_CONFIG_DIR"


def config_dir() -> Path:
    """Directory that holds the ``fields/``, ``cameras/``, ... configuration sub-directories."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "configs"


def resolve_config(kind: str, name_or_path: str | os.PathLike) -> Path:
    """Path of a configuration file given its kind and either a name or a path.

    ``resolve_config("fields", "s_field")`` returns ``<configs>/fields/s_field.yaml``. A value
    that names an existing file is returned unchanged.
    """
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") or candidate.is_file():
        if not candidate.is_file():
            raise FileNotFoundError(f"configuration file not found: {candidate}")
        return candidate
    path = config_dir() / kind / f"{name_or_path}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (config_dir() / kind).glob("*.yaml"))
        raise FileNotFoundError(
            f"no '{kind}' configuration named '{name_or_path}' in {config_dir() / kind} "
            f"(available: {available})")
    return path


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def load_config(kind: str, name_or_path: str | os.PathLike) -> dict[str, Any]:
    return load_yaml(resolve_config(kind, name_or_path))


def deep_update(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``base`` with ``overrides`` merged in, recursing into nested mappings."""
    result = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
