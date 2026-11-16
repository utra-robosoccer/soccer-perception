"""Domain randomization: turning a preset (``configs/randomization/*.yaml``) into concrete
scene appearances and camera-mount perturbations.

A preset has three sections, sampled at different rates:

* ``scene``  once per scenario; becomes overrides of the scene configuration and is baked into
  the MuJoCo model (turf, lines, lighting, shadows, backdrop, goal, ball and robot looks);
* ``camera`` once per frame; perturbs the head-camera mount (height, pitch, roll, pan);
* ``image``  once per frame; photometric image-space effects (`datasets.image_aug`).

Everything is drawn from the generator that is passed in, so a seed fixes the result. All of
it is appearance only: no sampled value moves a field line or changes a goal's dimensions. The
one geometric quantity that does vary is the ball radius, and the labels follow it because
they are rendered, not computed.
"""
from __future__ import annotations

import colorsys
import math
from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.geometry.camera import RigState
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.simulation.scene_objects import DEFAULT_ROBOT_APPEARANCE

PRESETS = ("none", "mild", "strong")
MARKER_COLOURS = ([0.85, 0.10, 0.60], [0.05, 0.75, 0.85])      # magenta, cyan team markers


def load_preset(name_or_path: str) -> dict[str, Any]:
    return config.load_config("randomization", name_or_path)


def _uniform(rng: np.random.Generator, bounds) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _tinted(rng: np.random.Generator, level: float, tint: float) -> list[float]:
    """A near-neutral colour: grey ``level`` with each channel moved by up to ``tint``."""
    return [float(np.clip(level + rng.uniform(-tint, tint), 0.0, 1.0)) for _ in range(3)]


def _hsv(hue: float, saturation: float, value: float) -> list[float]:
    return [float(c) for c in colorsys.hsv_to_rgb(hue % 1.0, saturation, value)]


# ------------------------------------------------------------------ camera mount
def sample_rig(rng: np.random.Generator, preset: dict[str, Any], mount: RigState) -> RigState:
    """A physically plausible head-camera pose around the configured mount."""
    cam = preset["camera"]
    height = float(np.clip(mount.height + rng.normal(0.0, 1.0) * float(cam["height_std"]),
                           *cam["height_limits"]))
    pitch_deg = float(np.clip(
        math.degrees(mount.pitch) + rng.normal(0.0, 1.0) * float(cam["pitch_std_deg"]),
        *cam["pitch_limits_deg"]))
    limit = float(cam["roll_limit_deg"])
    roll_deg = float(np.clip(rng.normal(0.0, 1.0) * float(cam["roll_std_deg"]), -limit, limit))
    pan_deg = _uniform(rng, cam["pan_range_deg"])
    return mount.replace(height=height, pitch=math.radians(pitch_deg),
                         roll=math.radians(roll_deg), pan=math.radians(pan_deg))


# ------------------------------------------------------------------ scene appearance
def _sample_robot(rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, Any]:
    tint = float(cfg["tint"])
    look = {
        "height": _uniform(rng, cfg["height"]),
        "build": _uniform(rng, cfg["build"]),
        "head_shape": str(rng.choice(["sphere", "box"])),
        "torso_rgb": _tinted(rng, _uniform(rng, cfg["grey"]), tint),
        "limb_rgb": _tinted(rng, _uniform(rng, cfg["grey"]), tint),
        "head_rgb": _tinted(rng, _uniform(rng, cfg["grey"]), tint),
        "foot_rgb": _tinted(rng, rng.uniform(0.05, 0.4), tint),
        "marker_rgb": list(MARKER_COLOURS[int(rng.integers(len(MARKER_COLOURS)))]),
    }
    if rng.random() < float(cfg.get("colorful_prob", 0.0)):
        look["torso_rgb"] = _hsv(rng.uniform(), rng.uniform(0.5, 1.0), rng.uniform(0.3, 0.9))
    return look


def _sample_background(rng: np.random.Generator, cfg: dict[str, Any], field: FieldGeometry,
                       turf_rgb: list[float]) -> dict[str, Any]:
    half_x, half_y = field.half_extent

    def colour() -> list[float]:
        roll = rng.random()
        if roll < 0.5 * float(cfg["distractor_prob"]):
            return _tinted(rng, 0.92, 0.03)                         # white, like posts and lines
        if roll < float(cfg["distractor_prob"]):
            return [float(np.clip(c * rng.uniform(0.8, 1.2), 0, 1)) for c in turf_rgb]   # turf-like
        if rng.random() < 0.5:
            return _tinted(rng, rng.uniform(0.1, 0.9), 0.05)
        return _hsv(rng.uniform(), rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.9))

    walls = [{"side": side, "distance": _uniform(rng, cfg["wall_distance"]),
              "height": _uniform(rng, cfg["wall_height"]), "rgb": colour()}
             for side in ("pos_x", "neg_x", "pos_y", "neg_y")
             if rng.random() < float(cfg["wall_prob"])]
    clutter = []
    low, high = cfg["clutter_count"]
    for _ in range(int(rng.integers(int(low), int(high) + 1))):
        # A spot on the floor 0.3 to 4 m outside the carpet, on a random side.
        outward = rng.uniform(0.3, 4.0)
        if rng.random() < 0.5:
            position = [float(rng.choice([-1, 1])) * (half_x + outward),
                        float(rng.uniform(-half_y, half_y))]
        else:
            position = [float(rng.uniform(-half_x, half_x)),
                        float(rng.choice([-1, 1])) * (half_y + outward)]
        clutter.append({"position": position, "yaw": float(rng.uniform(0, math.pi)),
                        "size": [float(rng.uniform(0.15, 1.2)), float(rng.uniform(0.15, 1.2)),
                                 float(rng.uniform(0.3, 2.0))],
                        "rgb": colour()})
    return {"walls": walls, "clutter": clutter}


def sample_scene_overrides(rng: np.random.Generator, preset: dict[str, Any],
                           field: FieldGeometry, num_robots: int) -> dict[str, Any]:
    """Scene-configuration overrides for one scenario (merge with `config.deep_update`).

    With randomization disabled the only override is the list of robots, all with the default
    look; the rest of the scene stays exactly as ``configs/scenes/default.yaml`` describes it.
    """
    cfg = preset.get("scene") or {}
    if not cfg.get("enabled", False):
        return {"robots": [dict(DEFAULT_ROBOT_APPEARANCE) for _ in range(num_robots)]}

    turf = cfg["turf"]
    base = _hsv(_uniform(rng, turf["hue"]), _uniform(rng, turf["saturation"]),
                _uniform(rng, turf["value"]))
    contrast = _uniform(rng, turf["stripe_contrast"])
    lighting = cfg["lighting"]
    elevation = math.radians(_uniform(rng, lighting["key_elevation_deg"]))
    azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
    # A light's direction is where the light travels: from the lamp down onto the field.
    key_direction = [-math.cos(elevation) * math.cos(azimuth),
                     -math.cos(elevation) * math.sin(azimuth), -math.sin(elevation)]
    fill_azimuth = azimuth + math.pi + float(rng.uniform(-1.0, 1.0))
    fill_direction = [-0.5 * math.cos(fill_azimuth), -0.5 * math.sin(fill_azimuth), -1.0]
    light_tint = float(lighting["tint"])
    ball, goal = cfg["ball"], cfg["goal"]
    styles = list(ball["styles"])
    weights = np.array([float(ball["styles"][s]) for s in styles])
    dark = _tinted(rng, _uniform(rng, ball["dark_value"]), 0.02)
    if rng.random() < float(ball["colored_prob"]):
        dark = _hsv(rng.uniform(), rng.uniform(0.6, 1.0), rng.uniform(0.3, 0.9))
    return {
        "turf": {"base_rgb": base,
                 "stripe_rgb": [float(min(1.0, c * contrast)) for c in base],
                 "stripe_width": float(rng.choice(turf["stripe_width_choices"])),
                 "noise": _uniform(rng, turf["noise"]),
                 "blotch": _uniform(rng, turf["blotch"]),
                 "seed": int(rng.integers(0, 2**31 - 1))},
        "line_rgb": _tinted(rng, _uniform(rng, cfg["lines"]["brightness"]),
                            float(cfg["lines"]["tint"])),
        "surround_rgb": _tinted(rng, _uniform(rng, cfg["surround"]["brightness"]),
                                float(cfg["surround"]["tint"])),
        "sky_rgb": _tinted(rng, _uniform(rng, cfg["sky"]["brightness"]),
                           float(cfg["sky"]["tint"])),
        "lighting": {
            "ambient": _uniform(rng, lighting["ambient"]),
            "shadows": bool(rng.random() < float(lighting["shadow_prob"])),
            "lights": [
                {"direction": key_direction, "diffuse": _uniform(rng, lighting["key_diffuse"]),
                 "specular": 0.1, "cast_shadow": True, "rgb": _tinted(rng, 1.0, light_tint)},
                {"direction": fill_direction, "diffuse": _uniform(rng, lighting["fill_diffuse"]),
                 "specular": 0.0, "cast_shadow": False, "rgb": _tinted(rng, 1.0, light_tint)},
            ]},
        "goal": {"rgb": _tinted(rng, _uniform(rng, goal["brightness"]), float(goal["tint"])),
                 "net_rgba": [1.0, 1.0, 1.0, _uniform(rng, goal["net_alpha"])]},
        "ball": {"style": str(rng.choice(styles, p=weights / weights.sum())),
                 "radius": _uniform(rng, ball["radius"]),
                 "rgb1": _tinted(rng, _uniform(rng, ball["light_brightness"]), 0.02),
                 "rgb2": dark,
                 "spot_count": int(rng.integers(2, 5)),
                 "texture_seed": int(rng.integers(0, 2**31 - 1))},
        "robots": [_sample_robot(rng, cfg["robots"]) for _ in range(num_robots)],
        "background": _sample_background(rng, cfg["background"], field, base),
    }
