"""Roll and pitch of the robot's body from gyroscope and accelerometer samples.

A Mahony-style complementary filter on a unit quaternion, with the additions that a walking
robot needs: the accelerometer is not treated as a perfect gravity sensor, and the filter
reports how uncertain its tilt is. Pure Python / NumPy; it knows nothing about MuJoCo, the
camera or the field.

How it works
------------
* GYROSCOPE. The orientation is integrated from the bias-corrected angular velocity. This is
  accurate over a fraction of a second and drifts over many seconds.
* ACCELEROMETER. At rest the specific force ``f`` points straight up, so its direction gives
  roll and pitch (never yaw). The disagreement ``e = (f x up_predicted) / g`` is fed back:
  ``omega <- omega - bias + kp_eff * e`` and ``bias <- bias - ki_eff * e * dt``.
  ``e`` is deliberately NOT normalised by ``|f|``: written this way it is exactly the body's
  horizontal acceleration divided by g, which averages to zero over a stride, so a stride's
  accelerations cancel in the estimate instead of leaving a bias.
* GAIN. The feedback is a low-pass with time constant ``1 / kp_eff``. A step's accelerations
  (5 to 10 degrees of apparent tilt at 1 to 2 Hz) pass through attenuated by ``kp_eff / omega``,
  so the gain must be well below the gait frequency; but it must stay high enough to hold the
  gyro bias down, whose effect on the tilt is ``bias_error / kp_eff``. ``kp_eff = kp * w``:
  ``w`` in (0, 1] falls with the recent LEVEL of disagreement (averaged over more than a
  stride, so it does not favour one half of the stride over the other), and the correction
  from a single sample that is plainly not gravity (an impact, a push) is capped. ``ki_eff``
  scales with the square of the level, which keeps the loop's damping the same at every gain. For the first
  seconds after start-up the gain is boosted, because the first tilt (taken from a few
  accelerometer samples, possibly while the robot was already walking) can be degrees off.
* UNCERTAINTY. The error of this filter obeys a two-state linear system (tilt error, gyro-bias
  error) driven by gyro noise, bias drift and the accelerometer's error. Its 2 x 2 covariance
  is propagated with the SAME gains the filter uses, so the reported standard deviation rises
  during lost samples and fast motion and settles when the robot stands still. The
  accelerometer's error is the recent size of ``e``: its white part (sensor noise) is taken
  as it is, the rest (the body's own accelerations) is treated as a band around
  ``dynamics_frequency_hz``. The accelerometer BIAS cannot be observed by this filter at all;
  it is added as a constant floor.

What it cannot do: observe yaw; tell a constant accelerometer bias from a real lean; or say
anything about how the CAMERA is mounted on the body. See `geometry.mount`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.estimation.imu_types import GRAVITY, ImuSample
from robosoccer_perception.geometry import quaternion as quat


@dataclasses.dataclass(frozen=True)
class TiltFilterConfig:
    """Gains and the estimator's BELIEFS about its sensor (not the simulator's truth)."""
    kp: float = 1.0                     # 1/s: accelerometer feedback; time constant 1 / kp
    ki: float = 0.25                    # 1/s^2: gyro-bias learning (kp^2 / 4: no overshoot)
    boost_kp: float = 4.0               # 1/s: extra feedback right after start-up ...
    boost_time_s: float = 1.5           # ... decaying with this time constant
    dynamics_reference: float = 0.10    # rad: RMS disagreement at which the gain is halved
    dynamics_window_s: float = 1.0      # averaging time of that RMS (longer than a stride)
    dynamics_frequency_hz: float = 1.5  # where the body's own accelerations are assumed to lie
    accel_gate: float = 0.35            # |f| may differ from g by this share before a sample
    innovation_gate: float = 0.20       # ... or disagree by this many rad before its pull is capped
    min_weight: float = 0.02            # the accelerometer is never ignored completely
    gyro_noise_std: float = 0.003       # rad/s per sample
    gyro_bias_std: float = 0.015        # rad/s: uncertainty of the bias at start-up
    gyro_bias_walk: float = 3.0e-4      # rad/s per sqrt(s)
    accel_noise_std: float = 0.05       # m/s^2 per sample
    accel_bias_std: float = 0.05        # m/s^2: unobservable, becomes a floor of the tilt std
    max_dt: float = 0.03                # seconds: a longer gap between samples is a dropout
    reboost_gap_s: float = 0.08         # seconds lost beyond which the start-up boost returns
    init_samples: int = 10              # accelerometer samples averaged for the first tilt
    init_accel_tolerance: float = 0.10  # ... each within this share of g in magnitude
    init_patience: int = 100            # after this many refusals any sample is accepted
    init_tilt_std_deg: float = 2.0      # uncertainty of that first tilt

    def __post_init__(self) -> None:
        if self.kp <= 0.0 or self.ki < 0.0:
            raise ValueError("kp must be positive and ki non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TiltFilterConfig":
        data = dict(data or {})
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown TiltFilterConfig keys: {sorted(unknown)}")
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class OrientationEstimate:
    """The filter's state after one sample."""
    timestamp: float            # the sample's own timestamp
    quat: np.ndarray            # (4,) w, x, y, z: body in a gravity-aligned frame of unknown yaw
    roll: float                 # radians
    pitch: float
    tilt_std: float             # radians, 1 sigma of roll and of pitch
    rate: np.ndarray            # (3,) bias-corrected angular velocity in the body frame, rad/s
    accel_weight: float         # 0 .. 1: how much the accelerometer was trusted
    gap: float                  # seconds since the previous sample


class TiltEstimator:
    """Feed `ImuSample`s in the order of their timestamps; read `latest` or keep the returned
    `OrientationEstimate`s (`estimation.sync.OrientationHistory` does)."""

    def __init__(self, cfg: TiltFilterConfig | None = None,
                 R_body_sensor: np.ndarray | None = None):
        self.cfg = cfg or TiltFilterConfig()
        self.R_body_sensor = np.eye(3) if R_body_sensor is None \
            else np.asarray(R_body_sensor, dtype=float)
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        self._quat = quat.IDENTITY.copy()
        self._bias = np.zeros(3)
        self._covariance = np.diag([math.radians(cfg.init_tilt_std_deg) ** 2,
                                    cfg.gyro_bias_std ** 2])
        self._innovation_var = (cfg.accel_noise_std / GRAVITY) ** 2
        self._rate_rms = 0.0
        self._last_time: float | None = None
        self._init_sum = np.zeros(3)
        self._init_count = 0
        self._init_refused = 0
        self._elapsed = 0.0
        self.latest: OrientationEstimate | None = None
        self.samples = 0
        self.rejected = 0               # samples that were not newer than the previous one
        self.dropouts = 0               # gaps longer than ``max_dt``
        self.longest_gap = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def initialized(self) -> bool:
        return self._init_count >= self.cfg.init_samples

    @property
    def gyro_bias(self) -> np.ndarray:
        """The learned gyro bias in the body frame, rad/s (z is not observable)."""
        return self._bias.copy()

    @property
    def rate_rms(self) -> float:
        """Recent RMS angular rate of the body, rad/s (used to price timing errors)."""
        return math.sqrt(self._rate_rms)

    # ------------------------------------------------------------------ one sample
    def update(self, sample: ImuSample) -> OrientationEstimate | None:
        """Process one sample. Returns None while initialising and for a sample that is not
        newer than the previous one (it is dropped: the filter does not run backwards)."""
        if type(sample) is not ImuSample:
            raise TypeError("the orientation filter takes ImuSample objects")
        cfg = self.cfg
        gyro = self.R_body_sensor @ sample.gyro
        force = self.R_body_sensor @ sample.accel
        norm = float(np.linalg.norm(force))
        if not (np.isfinite(gyro).all() and math.isfinite(norm)) or norm < 1e-6:
            self.rejected += 1
            return None
        if self._last_time is not None and sample.timestamp <= self._last_time:
            self.rejected += 1
            return None
        up_measured = force / norm

        if not self.initialized:
            # Average the first few accelerometer readings whose magnitude is close to g: the
            # robot is assumed not to be accelerating hard when it is switched on.
            self._last_time = sample.timestamp
            steady = abs(norm - GRAVITY) <= cfg.init_accel_tolerance * GRAVITY
            if not steady and self._init_refused < cfg.init_patience:
                self._init_refused += 1
                return None
            self._init_sum += force
            self._init_count += 1
            if self.initialized:
                roll, pitch = quat.tilt_from_up(self._init_sum)
                self._quat = quat.from_matrix(quat.matrix_from_rpy(roll, pitch))
            return None

        dt = sample.timestamp - self._last_time
        self._last_time = sample.timestamp
        self.samples += 1
        self.longest_gap = max(self.longest_gap, dt)
        variance = self._covariance
        gap = dt
        if dt > cfg.max_dt:
            # Samples were lost. What the body did in the gap is unknown, and holding the last
            # rate across it would turn one fast sample into a large rotation. The estimate
            # is advanced by one ordinary interval only; the tilt variance grows by how much
            # the body typically turns in the time that is unaccounted for, and the feedback
            # is boosted for a moment so that the accelerometer can pull the estimate back.
            self.dropouts += 1
            lost = dt - cfg.max_dt
            dt = cfg.max_dt
            unknown = max(self.rate_rms, 0.05) * lost
            variance = variance + np.diag([unknown * unknown, 0.0])
            if lost > cfg.reboost_gap_s:
                self._elapsed = min(self._elapsed, cfg.boost_time_s)

        # ---- how far the accelerometer can be trusted right now
        rate = gyro - self._bias
        up_predicted = quat.to_matrix(self._quat)[2, :]
        error = np.cross(force, up_predicted) / GRAVITY
        disagreement = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(rate))
        blend = min(dt / cfg.dynamics_window_s, 1.0)
        self._innovation_var += blend * (disagreement * disagreement - self._innovation_var)
        self._rate_rms += blend * (speed * speed - self._rate_rms)
        level = 1.0 / (1.0 + self._innovation_var / cfg.dynamics_reference ** 2)
        # Gross outliers only: inside the gates a sample counts in full, whatever the phase
        # of the stride.
        sample_weight = 1.0
        excess = abs(norm - GRAVITY) / GRAVITY - cfg.accel_gate
        if excess > 0.0:
            sample_weight *= math.exp(-0.5 * (excess / cfg.accel_gate) ** 2)
        if disagreement > cfg.innovation_gate:
            # Saturate, do not redescend: the correction of a large disagreement is capped at
            # the gate's, never made SMALLER than a moderate one's. A rule that shrinks the
            # correction as the disagreement grows turns into positive feedback when the
            # disagreement swings widely every cycle (a fast sway about the ankles), and the
            # estimate walks away. For the same reason the cap enters both feedback paths
            # exactly once.
            sample_weight *= cfg.innovation_gate / disagreement
        weight = max(level * sample_weight, cfg.min_weight)

        # ---- complementary update
        self._elapsed += dt
        boost = math.exp(-self._elapsed / cfg.boost_time_s)
        kp = cfg.kp * weight + cfg.boost_kp * boost
        # No bias learning while the start-up boost is still pulling a rough first tilt into
        # place: that transient is not gyro bias, and a bias learned from it takes a long
        # time to unlearn once the gain is low.
        ki = cfg.ki * level * weight * (1.0 - boost) ** 4
        alpha = min(kp * dt, 1.0)
        self._bias = self._bias - ki * dt * error
        self._bias[2] = 0.0                         # yaw-rate bias is unobservable: do not drift
        step = rate * dt + alpha * error
        self._quat = quat.normalize(quat.multiply(self._quat, quat.from_rotation_vector(step)))

        # ---- uncertainty: the same linear system, with the same gains
        white = (cfg.accel_noise_std / GRAVITY) ** 2
        dynamic = max(self._innovation_var - white, 0.0)
        omega = 2.0 * math.pi * cfg.dynamics_frequency_hz
        band = 2.0 * kp / ((kp * kp + omega * omega) * dt)      # see the module docstring
        beta = ki * dt
        transition = np.array([[1.0 - alpha, dt], [-beta, 1.0]])
        gain = np.array([alpha, beta])
        process = np.diag([(cfg.gyro_noise_std * dt) ** 2, cfg.gyro_bias_walk ** 2 * dt])
        variance = transition @ variance @ transition.T \
            + (white + dynamic * band) * np.outer(gain, gain) + process
        self._covariance = 0.5 * (variance + variance.T)

        roll, pitch = quat.tilt_from_quat(self._quat)
        tilt_std = math.sqrt(max(self._covariance[0, 0], 0.0)
                             + (cfg.accel_bias_std / GRAVITY) ** 2)
        self.latest = OrientationEstimate(sample.timestamp, self._quat.copy(), roll, pitch,
                                          tilt_std, rate, weight, gap)
        return self.latest
