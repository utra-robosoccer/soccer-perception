"""3-DOF Monte Carlo localization: a particle filter over the robot pose (x, y, yaw).

Read docs/localization.md first if particle filters are new to you. In one paragraph: the
filter keeps N guesses ("particles") of where the robot might be. Each cycle it

1. **predicts**: moves every particle by the reported odometry plus random noise;
2. **updates**: scores every particle by how well it explains the current observations and
   multiplies its weight by that score;
3. **resamples** (only when the weights have become too uneven): draws a new set of particles
   in proportion to weight, so good guesses multiply and bad ones die out;
4. **estimates**: reports the dominant cluster of particles as the pose.

Inputs are `OdometryDelta` and `ObservationFrame` only. The filter has no access to the
simulator or to ground truth; a global pose enters only through `init_gaussian`.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import LandmarkMap, build_landmark_map
from robosoccer_perception.localization import motion_model
from robosoccer_perception.localization.estimation import (
    EstimatorConfig, PoseEstimate, estimate_pose, mirror_pose)
from robosoccer_perception.localization.measurement_model import (
    MeasurementModel, MeasurementModelConfig)
from robosoccer_perception.localization.motion_model import MotionModelConfig, OdometryDelta
from robosoccer_perception.observations.types import ObservationFrame


@dataclasses.dataclass(frozen=True)
class RecoveryConfig:
    alpha_slow: float = 0.02
    alpha_fast: float = 0.10
    max_injection_fraction: float = 0.25
    injection_trigger: float = 0.10
    background_injection_fraction: float = 0.01
    sensor_reset_fraction: float = 0.7
    injected_particle_weight: float = 0.1
    symmetry_prior: str = "same_side"
    anchor_hold_updates: int = 20
    maintain_symmetry: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RecoveryConfig":
        cfg = cls(**(data or {}))
        if cfg.symmetry_prior not in ("same_side", "none"):
            raise ValueError("symmetry_prior must be 'same_side' or 'none'")
        return cfg


@dataclasses.dataclass(frozen=True)
class ParticleFilterConfig:
    num_particles: int = 1000
    resample_ess_fraction: float = 0.5
    init_position_std: float = 0.30
    init_yaw_std_deg: float = 10.0
    recovery: RecoveryConfig = dataclasses.field(default_factory=RecoveryConfig)
    motion: MotionModelConfig = dataclasses.field(default_factory=MotionModelConfig)
    measurement: MeasurementModelConfig = dataclasses.field(default_factory=MeasurementModelConfig)
    estimator: EstimatorConfig = dataclasses.field(default_factory=EstimatorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParticleFilterConfig":
        data = dict(data)
        cfg = cls(
            recovery=RecoveryConfig.from_dict(data.pop("recovery", None)),
            motion=MotionModelConfig.from_dict(data.pop("motion", None)),
            measurement=MeasurementModelConfig.from_dict(data.pop("measurement", None)),
            estimator=EstimatorConfig.from_dict(data.pop("estimator", None)),
            **data)
        if cfg.num_particles < 10:
            raise ValueError("num_particles must be at least 10")
        return cfg

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "particle_filter",
             overrides: dict[str, Any] | None = None) -> "ParticleFilterConfig":
        """Load ``configs/localization/<name>.yaml``, optionally with nested overrides."""
        return cls.from_dict(config.deep_update(
            config.load_config("localization", name_or_path), overrides))


@dataclasses.dataclass(frozen=True)
class UpdateInfo:
    """What one measurement update did. For diagnostics and plots."""
    effective_sample_size: float    # after reweighting, before any resampling
    resampled: bool
    injected: int                   # particles replaced by fresh hypotheses
    agreement: float                # in (0, 1]: how well the belief explains this frame
    used: bool                      # False when the frame held no usable evidence


class ParticleFilter:
    def __init__(self, field: FieldGeometry, landmarks: LandmarkMap | None = None,
                 cfg: ParticleFilterConfig | None = None, seed: int | None = 0):
        self.cfg = cfg or ParticleFilterConfig()
        self.field = field
        self.landmarks = landmarks or build_landmark_map(field)
        self.measurement_model = MeasurementModel(field, self.landmarks, self.cfg.measurement)
        self.rng = np.random.default_rng(seed)
        self.particles = np.zeros((self.cfg.num_particles, 3))
        self.weights = np.full(self.cfg.num_particles, 1.0 / self.cfg.num_particles)
        self._agreement_slow: float | None = None
        self._agreement_fast: float | None = None
        self._anchor: np.ndarray | None = None      # see _reset
        self._converged_streak = 0
        self._last_pose: np.ndarray | None = None
        self.init_uniform()

    # ------------------------------------------------------------------ initialisation
    @property
    def num_particles(self) -> int:
        return len(self.particles)

    def _reset(self, particles: np.ndarray, anchor: np.ndarray | None) -> None:
        particles[:, 2] = tf.wrap_angle(particles[:, 2])
        self.particles = particles
        self.weights = np.full(len(particles), 1.0 / len(particles))
        self._agreement_slow = self._agreement_fast = None
        # The anchor is the last position the filter had reason to believe in: the initial
        # prior, then every sustained confident estimate. Only `symmetry_prior` uses it.
        self._anchor = anchor
        self._converged_streak = 0
        self._last_pose = None

    def init_gaussian(self, pose, position_std: float | None = None,
                      yaw_std: float | None = None) -> None:
        """Start (or restart) from an approximately known pose ``[x, y, yaw]``.

        ``position_std`` (m, per axis) and ``yaw_std`` (rad) default to the configured values.
        """
        position_std = self.cfg.init_position_std if position_std is None else position_std
        yaw_std = math.radians(self.cfg.init_yaw_std_deg) if yaw_std is None else yaw_std
        noise = self.rng.normal(size=(self.cfg.num_particles, 3))
        pose = np.asarray(pose, dtype=float)
        self._reset(pose + noise * [position_std, position_std, yaw_std], anchor=pose[:2].copy())

    def init_uniform(self, x_range: tuple[float, float] | None = None,
                     y_range: tuple[float, float] | None = None,
                     yaw_range: tuple[float, float] | None = None) -> None:
        """Start (or restart) with no idea of the pose: uniform over the carpet and all headings.

        Ranges restrict the prior, e.g. ``x_range=(-half_length, 0)`` for "somewhere in our
        own half", which is what breaks the field's symmetry.
        """
        particles = self._sample_uniform(self.cfg.num_particles, x_range, y_range, yaw_range)
        centre = particles[:, :2].mean(axis=0) if (x_range or y_range) else None
        # A prior that is not centred on the field tells the two symmetric halves apart.
        informative = centre is not None and np.linalg.norm(centre) > 0.25
        self._reset(particles, anchor=centre if informative else None)

    def resolve_symmetry(self, reference_xy) -> None:
        """Use outside information to choose between the two mirror-image hypotheses.

        ``reference_xy`` is any field position known to be on the robot's side of the field
        centre, for example the middle of its own half at kick-off. Every particle that is
        farther from it than its mirror image is flipped, and the reference becomes the side
        anchor. This is the only way a filter started with `init_uniform()` (no prior) can
        leave the ambiguous state: no observation of a symmetric field can do it.
        """
        reference = np.asarray(reference_xy, dtype=float)[:2]
        flipped = mirror_pose(self.particles)
        nearer = (np.linalg.norm(flipped[:, :2] - reference, axis=1)
                  < np.linalg.norm(self.particles[:, :2] - reference, axis=1))
        self.particles[nearer] = flipped[nearer]
        self._anchor = reference.copy()
        self._last_pose = None

    def _sample_uniform(self, count: int, x_range=None, y_range=None, yaw_range=None) -> np.ndarray:
        half_x, half_y = self.field.half_extent
        x_range = x_range or (-half_x, half_x)
        y_range = y_range or (-half_y, half_y)
        yaw_range = yaw_range or (-math.pi, math.pi)
        low = [x_range[0], y_range[0], yaw_range[0]]
        high = [x_range[1], y_range[1], yaw_range[1]]
        return self.rng.uniform(low, high, size=(count, 3))

    # ------------------------------------------------------------------ the filter cycle
    def predict(self, delta: OdometryDelta | None) -> None:
        """Motion update. Pass ``None`` when an odometry message was lost."""
        self.particles = motion_model.predict(self.particles, delta, self.cfg.motion, self.rng)

    def update(self, frame: ObservationFrame) -> UpdateInfo:
        """Measurement update, followed by resampling when it is needed."""
        count = self.num_particles
        likelihood = self.measurement_model.log_likelihood(self.particles, frame, self.rng)
        if likelihood.evidence_weight <= 0.0:
            return UpdateInfo(self.effective_sample_size, False, 0,
                              self._agreement_fast or 1.0, used=False)

        log_weights = np.log(np.maximum(self.weights, 1e-300)) + likelihood.log_likelihood
        log_weights -= log_weights.max()                    # avoid underflow
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()

        # Agreement: the posterior-weighted mean log-likelihood per unit of evidence, mapped
        # back to (0, 1]. Near 1 when the surviving hypotheses explain the frame well, small
        # when not even the best of them does (the signature of a kidnapping).
        agreement = float(np.exp(self.weights @ likelihood.log_likelihood
                                 / likelihood.evidence_weight))
        recovery = self.cfg.recovery
        if self._agreement_slow is None:
            self._agreement_slow = self._agreement_fast = agreement
        else:
            self._agreement_slow += recovery.alpha_slow * (agreement - self._agreement_slow)
            self._agreement_fast += recovery.alpha_fast * (agreement - self._agreement_fast)
        adaptive = float(np.clip(1.0 - self._agreement_fast / self._agreement_slow,
                                 0.0, recovery.max_injection_fraction))

        ess = self.effective_sample_size
        forced = adaptive > recovery.injection_trigger
        if not forced and ess >= self.cfg.resample_ess_fraction * count:
            return UpdateInfo(ess, False, 0, agreement, used=True)

        fraction = adaptive if forced else recovery.background_injection_fraction
        injected = self._propose(int(round(fraction * count)), frame)
        keep = count - len(injected)
        survivors = self.particles[self._systematic_resample(keep)]
        if recovery.maintain_symmetry and self._anchor is None:
            # Nothing has told the two halves of the field apart yet, so the true posterior is
            # exactly symmetric. Keep the particle set symmetric too, instead of letting
            # resampling noise pick a side and then be confident about it.
            flip = self.rng.random(keep) < 0.5
            survivors[flip] = mirror_pose(survivors[flip])
        self.particles = np.vstack([survivors, injected])
        # A fresh hypothesis starts with less weight than a survivor: being moved is a priori
        # unlikely, so it has to explain several frames better before it can take over. After
        # a real kidnapping the survivors explain nothing and it takes over within a few frames.
        self.weights = np.concatenate([np.ones(keep),
                                       np.full(len(injected), recovery.injected_particle_weight)])
        self.weights /= self.weights.sum()
        return UpdateInfo(ess, True, len(injected), agreement, used=True)

    def estimate(self) -> PoseEstimate:
        """Dominant-mode pose estimate (see localization/estimation.py)."""
        estimate = estimate_pose(self.particles, self.weights, self.cfg.estimator,
                                 self._last_pose)
        self._last_pose = estimate.pose.copy()
        # Move the anchor only after the filter has stayed converged for a while: a brief
        # false convergence on a look-alike pose must not redefine "our side of the field".
        self._converged_streak = self._converged_streak + 1 if estimate.converged else 0
        if self._converged_streak >= self.cfg.recovery.anchor_hold_updates:
            self._anchor = estimate.pose[:2].copy()
        return estimate

    def step(self, delta: OdometryDelta | None, frame: ObservationFrame | None
             ) -> tuple[PoseEstimate, UpdateInfo]:
        """One full cycle: predict, update (if there is a frame), estimate."""
        self.predict(delta)
        info = self.update(frame) if frame is not None else UpdateInfo(
            self.effective_sample_size, False, 0, self._agreement_fast or 1.0, used=False)
        return self.estimate(), info

    @property
    def effective_sample_size(self) -> float:
        """1 / sum(w^2): N when all weights are equal, 1 when one particle holds everything."""
        return float(1.0 / np.sum(self.weights ** 2))

    # ------------------------------------------------------------------ internals
    def _systematic_resample(self, count: int) -> np.ndarray:
        """Low-variance resampling: one random offset, then ``count`` evenly spaced picks."""
        positions = (self.rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions)

    def _propose(self, count: int, frame: ObservationFrame) -> np.ndarray:
        """Fresh pose hypotheses for recovery. Never uses ground truth.

        Part of them are drawn from the current observations ("sensor resetting"): pick an
        observed landmark, pick a map landmark of the same class, and place the robot at the
        observed range from it with the heading that puts it at the observed bearing. Such a
        pose explains at least one observation, which makes recovery far faster than uniform
        guessing over three dimensions. The rest are uniform over the carpet.
        """
        if count <= 0:
            return np.zeros((0, 3))
        proposals = self._sample_uniform(count)
        observations = [o for o in frame.landmarks if len(self.landmarks.positions(o.type))]
        from_sensor = int(round(self.cfg.recovery.sensor_reset_fraction * count)) \
            if observations else 0
        for i in range(from_sensor):
            observation = observations[self.rng.integers(len(observations))]
            candidates = self.landmarks.positions(observation.type)
            landmark = candidates[self.rng.integers(len(candidates))]
            distance = max(0.05, observation.range
                           + self.rng.normal(0.0, max(observation.range_std, 0.05)))
            direction = self.rng.uniform(-math.pi, math.pi)     # robot -> landmark, field frame
            x = landmark[0] - distance * math.cos(direction)
            y = landmark[1] - distance * math.sin(direction)
            if self.field.contains([x, y]):
                yaw = direction - observation.bearing + self.rng.normal(
                    0.0, max(observation.bearing_std, math.radians(1.0)))
                proposals[i] = [x, y, tf.wrap_angle(yaw)]

        if self.cfg.recovery.symmetry_prior == "same_side" and self._anchor is not None:
            # A pose and its half-turn mirror explain the observations equally well. Keep each
            # proposal on the side of the field centre where the filter was last confident.
            flipped = mirror_pose(proposals)
            nearer = (np.linalg.norm(flipped[:, :2] - self._anchor, axis=1)
                      < np.linalg.norm(proposals[:, :2] - self._anchor, axis=1))
            proposals[nearer] = flipped[nearer]
        return proposals
