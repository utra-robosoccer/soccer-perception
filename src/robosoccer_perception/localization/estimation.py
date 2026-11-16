"""Turning a particle set into one pose estimate, without averaging across hypotheses.

A particle filter often holds several separate hypotheses: a RoboCup field looks identical after
a half turn, so right after a global start there are (at least) two equally good clusters. The
plain weighted mean of all particles then lands between them, typically at the centre of the
field, which no particle supports.

Method (dominant mode):

1. Bin the particles into a coarse 3-D histogram over (x, y, yaw), weighted by particle weight.
2. Sum each cell with its neighbours (yaw wraps around) and take the heaviest neighbourhood.
3. The estimate is the weighted mean of the particles in that neighbourhood, with a circular
   mean for yaw. Its covariance and its share of the total weight come from the same particles.
4. Remove those particles and repeat to list the remaining modes.

`symmetric_ambiguity` is raised when another mode sits near the half-turn mirror image of the
estimate; the filter then knows where it is only up to the field's symmetry.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.geometry import transforms as tf


@dataclasses.dataclass(frozen=True)
class EstimatorConfig:
    cell_size: float = 0.4
    yaw_bin_deg: float = 30.0
    max_modes: int = 3
    min_mode_weight: float = 0.05
    converged_min_mode_weight: float = 0.7
    converged_max_position_std: float = 0.25
    converged_max_yaw_std_deg: float = 10.0
    confidence_position_scale: float = 0.30
    confidence_yaw_scale_deg: float = 15.0
    mirror_position_tolerance: float = 0.8
    mirror_yaw_tolerance_deg: float = 45.0
    mirror_weight_ratio: float = 0.25

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EstimatorConfig":
        return cls(**(data or {}))


@dataclasses.dataclass(frozen=True)
class PoseMode:
    """One cluster of particles: a pose hypothesis."""
    pose: np.ndarray            # [x, y, yaw]
    weight: float               # share of the total particle weight, in [0, 1]
    covariance: np.ndarray      # 3x3 over (x, y, yaw) of the cluster's particles
    position_std: float         # sqrt(var_x + var_y), metres
    yaw_std: float              # radians


@dataclasses.dataclass(frozen=True)
class PoseEstimate:
    pose: np.ndarray            # [x, y, yaw] of the dominant mode
    covariance: np.ndarray
    position_std: float
    yaw_std: float
    mode_weight: float          # how much of the belief the dominant mode holds
    confidence: float           # in [0, 1]: mode weight discounted by the mode's spread
    converged: bool             # the filter's own verdict (no ground truth involved)
    symmetric_ambiguity: bool   # a second mode sits at the mirrored pose
    modes: tuple[PoseMode, ...]  # all modes found, heaviest first; modes[0] is this estimate

    @property
    def x(self) -> float:
        return float(self.pose[0])

    @property
    def y(self) -> float:
        return float(self.pose[1])

    @property
    def yaw(self) -> float:
        return float(self.pose[2])


def mirror_pose(pose) -> np.ndarray:
    """The pose that sees exactly the same field: a half turn about the field centre."""
    pose = np.asarray(pose, dtype=float)
    return np.stack([-pose[..., 0], -pose[..., 1], tf.wrap_angle(pose[..., 2] + math.pi)], axis=-1)


def find_modes(particles: np.ndarray, weights: np.ndarray, cfg: EstimatorConfig) -> list[PoseMode]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    yaw_bins = max(4, int(round(360.0 / cfg.yaw_bin_deg)))
    origin = particles[:, :2].min(axis=0)
    cell = np.floor((particles[:, :2] - origin) / cfg.cell_size).astype(np.intp) + 1  # 1 = padding
    yaw_cell = np.floor((tf.wrap_angle(particles[:, 2]) + math.pi)
                        / (2.0 * math.pi) * yaw_bins).astype(np.intp) % yaw_bins
    shape = (cell[:, 0].max() + 2, cell[:, 1].max() + 2, yaw_bins)
    flat = np.ravel_multi_index((cell[:, 0], cell[:, 1], yaw_cell), shape)

    remaining = weights.copy()
    modes: list[PoseMode] = []
    for _ in range(cfg.max_modes):
        if modes and remaining.sum() < cfg.min_mode_weight:
            break
        smooth = np.bincount(flat, weights=remaining, minlength=int(np.prod(shape)))
        smooth = smooth.reshape(shape)
        for axis in range(3):
            # Sum each cell with its two neighbours along this axis. The outer ring of position
            # cells is empty padding, so np.roll never wraps real mass across the map; yaw is
            # meant to wrap.
            smooth = smooth + np.roll(smooth, 1, axis=axis) + np.roll(smooth, -1, axis=axis)
        peak = np.unravel_index(int(np.argmax(smooth)), shape)
        yaw_offset = (yaw_cell - peak[2] + yaw_bins // 2) % yaw_bins - yaw_bins // 2
        members = ((np.abs(cell[:, 0] - peak[0]) <= 1) & (np.abs(cell[:, 1] - peak[1]) <= 1)
                   & (np.abs(yaw_offset) <= 1) & (remaining > 0.0))
        weight = float(remaining[members].sum())
        if weight <= 0.0 or (modes and weight < cfg.min_mode_weight):
            break
        modes.append(_summarise(particles[members], remaining[members], weight))
        remaining[members] = 0.0
    return modes


def _summarise(particles: np.ndarray, weights: np.ndarray, weight: float) -> PoseMode:
    w = weights / weights.sum()
    mean_xy = w @ particles[:, :2]
    mean_yaw = tf.circular_mean(particles[:, 2], w)
    deviation = np.column_stack([particles[:, :2] - mean_xy,
                                 tf.angle_difference(particles[:, 2], mean_yaw)])
    covariance = (deviation * w[:, None]).T @ deviation
    return PoseMode(np.array([mean_xy[0], mean_xy[1], mean_yaw]), weight, covariance,
                    float(math.sqrt(covariance[0, 0] + covariance[1, 1])),
                    float(math.sqrt(covariance[2, 2])))


def _is_mirror(mode: PoseMode, of: PoseMode, cfg: EstimatorConfig) -> bool:
    mirrored = mirror_pose(of.pose)
    return bool(
        np.linalg.norm(mode.pose[:2] - mirrored[:2]) <= cfg.mirror_position_tolerance
        and abs(tf.angle_difference(mode.pose[2], mirrored[2]))
        <= math.radians(cfg.mirror_yaw_tolerance_deg))


def estimate_pose(particles: np.ndarray, weights: np.ndarray,
                  cfg: EstimatorConfig | None = None, previous_pose=None) -> PoseEstimate:
    """Dominant-mode pose estimate of a weighted particle set (N, 3).

    When the dominant mode has a mirror-image twin of comparable weight, the two are equally
    good and which one is "heavier" is noise. ``previous_pose`` (the last reported pose) then
    decides: the twin nearer to it is reported, so the estimate does not flicker between them.
    """
    cfg = cfg or EstimatorConfig()
    modes = find_modes(particles, weights, cfg)
    best = modes[0]
    twins = [mode for mode in modes[1:] if _is_mirror(mode, best, cfg)
             and mode.weight >= cfg.mirror_weight_ratio * best.weight]
    ambiguous = bool(twins)
    if ambiguous and previous_pose is not None:
        previous_xy = np.asarray(previous_pose, dtype=float)[:2]
        if (np.linalg.norm(twins[0].pose[:2] - previous_xy)
                < np.linalg.norm(best.pose[:2] - previous_xy)):
            best = twins[0]
            modes = [best, *[mode for mode in modes if mode is not best]]
    confidence = best.weight * math.exp(
        -0.5 * (best.position_std / cfg.confidence_position_scale) ** 2
        - 0.5 * (best.yaw_std / math.radians(cfg.confidence_yaw_scale_deg)) ** 2)
    converged = (not ambiguous
                 and best.weight >= cfg.converged_min_mode_weight
                 and best.position_std <= cfg.converged_max_position_std
                 and best.yaw_std <= math.radians(cfg.converged_max_yaw_std_deg))
    return PoseEstimate(best.pose, best.covariance, best.position_std, best.yaw_std, best.weight,
                        float(confidence), bool(converged), bool(ambiguous), tuple(modes))
