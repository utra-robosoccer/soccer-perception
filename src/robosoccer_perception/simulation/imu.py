"""A simulated inertial measurement unit on the robot's body. NumPy only.

The IMU is derived from the body's true motion, not from its true orientation:

1. IDEAL MEASUREMENTS. The body pose is an analytic function of time (`BodyMotion`). At every
   sample instant the sensor's position is differentiated twice and its orientation once
   (central differences over +-1 ms), which gives the angular velocity and the SPECIFIC FORCE
   ``R_world_sensor^T (a_world - g_world)`` that a perfect sensor at that point would report.
   A lever arm from the body origin and all centripetal and tangential terms are included
   automatically, because it is the sensor's own position that is differentiated.
2. IMPERFECTIONS. A turn-on bias and a slowly drifting bias on each axis, white noise, a
   sampling interval that wanders, transport latency, single lost samples, longer outages,
   and a clock offset between the IMU's timestamps and true time.

What leaves this module for the robot is `ImuSample` (timestamp, gyro, accel) and nothing else.
The true roll and pitch are kept in `ImuStream.truth_*` for scoring; no estimator reads them
(tests/test_ground_truth_isolation.py).

SYNTHETIC. The noise figures in ``configs/imu/*.yaml`` are typical of a consumer MEMS IMU after
a basic bench calibration. They are not measurements of the robot's sensor: the prototype has
no IMU in its CAD yet, only a placeholder site at the pelvis origin.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Iterator

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.estimation.imu_types import GRAVITY, ImuSample
from robosoccer_perception.geometry import quaternion as quat
from robosoccer_perception.simulation.body_motion import BodyMotion

DIFFERENCE_STEP = 1.0e-3        # seconds: half-width of the central differences


@dataclasses.dataclass(frozen=True)
class ImuConfig:
    """One ``configs/imu/*.yaml`` file. Everything here is the SIMULATOR's truth about the
    sensor; the estimator has its own, separate beliefs (`estimation.orientation`)."""
    name: str = "nominal"
    rate_hz: float = 200.0
    rate_jitter: float = 0.02               # 1 sigma of each interval, as a share of the period
    gyro_noise_std: float = 0.002           # rad/s per sample
    accel_noise_std: float = 0.03           # m/s^2 per sample
    gyro_bias_std: float = 0.010            # rad/s per axis, drawn once per run
    accel_bias_std: float = 0.05            # m/s^2 per axis, drawn once per run
    gyro_bias_walk: float = 2.0e-4          # rad/s per sqrt(s): slow drift of the gyro bias
    accel_bias_walk: float = 1.0e-3         # m/s^2 per sqrt(s)
    latency_s: float = 0.004                # delay between a sample and its arrival
    latency_jitter_s: float = 0.001
    dropout_prob: float = 0.0               # each sample is lost with this probability
    outage_rate_hz: float = 0.0             # mean number of longer outages per second
    outage_duration_s: tuple[float, float] = (0.10, 0.40)
    # ---- hidden from the robot
    timestamp_offset_s: float = 0.0         # reported timestamp = true time + this
    misalignment_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # sensor axes vs. believed

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in [0, 1)")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ImuConfig":
        data = dict(data or {})
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown ImuConfig keys: {sorted(unknown)}")
        for key in ("outage_duration_s", "misalignment_deg"):
            if key in data:
                data[key] = tuple(float(v) for v in data[key])
        return cls(**data)

    @classmethod
    def load(cls, name_or_path: str | os.PathLike = "nominal",
             overrides: dict[str, Any] | None = None) -> "ImuConfig":
        return cls.from_dict(config.deep_update(config.load_config("imu", name_or_path),
                                                overrides))

    def replace(self, **changes) -> "ImuConfig":
        return dataclasses.replace(self, **changes)


@dataclasses.dataclass(frozen=True)
class ImuStream:
    """Every sample of one run that the robot receives, in the order of arrival.

    ``timestamp`` is what the sensor reports, ``arrival`` when the robot gets the sample (both
    on the robot's clock). ``true_time`` and the ``truth_*`` arrays are simulator ground truth
    for scoring and are not part of an `ImuSample`.
    """
    timestamp: np.ndarray       # (N,)
    arrival: np.ndarray         # (N,) non-decreasing
    gyro: np.ndarray            # (N, 3) rad/s, sensor frame
    accel: np.ndarray           # (N, 3) m/s^2, sensor frame
    true_time: np.ndarray       # (N,)
    truth_roll: np.ndarray      # (N,) true body roll at ``true_time``
    truth_pitch: np.ndarray
    generated: int              # samples the sensor took, including the lost ones

    def __len__(self) -> int:
        return len(self.timestamp)

    def sample(self, index: int) -> ImuSample:
        return ImuSample(float(self.timestamp[index]), self.gyro[index].copy(),
                         self.accel[index].copy())

    def arrived_between(self, after: float, until: float) -> Iterator[ImuSample]:
        """Samples that arrive in ``(after, until]``, oldest arrival first."""
        first = int(np.searchsorted(self.arrival, after, side="right"))
        last = int(np.searchsorted(self.arrival, until, side="right"))
        for index in range(first, last):
            yield self.sample(index)

    @property
    def lost_fraction(self) -> float:
        return 1.0 - len(self) / max(self.generated, 1)


def _rotation_vectors(relative: np.ndarray) -> np.ndarray:
    """Rotation vectors (N, 3) of rotation matrices (N, 3, 3) close to the identity."""
    skew = 0.5 * np.stack([relative[:, 2, 1] - relative[:, 1, 2],
                           relative[:, 0, 2] - relative[:, 2, 0],
                           relative[:, 1, 0] - relative[:, 0, 1]], axis=1)
    sine = np.linalg.norm(skew, axis=1)
    angle = np.arcsin(np.clip(sine, 0.0, 1.0))
    scale = np.where(sine > 1e-9, angle / np.maximum(sine, 1e-12), 1.0)
    return skew * scale[:, None]


def ideal_measurements(motion: BodyMotion, times, T_body_sensor: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """What a perfect IMU reads at ``times``: ``(gyro (N, 3), specific force (N, 3))`` in the
    sensor frame, for a sensor at ``T_body_sensor`` (identity: at the body origin)."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    T_body_sensor = np.eye(4) if T_body_sensor is None else np.asarray(T_body_sensor, float)
    offset, R_body_sensor = T_body_sensor[:3, 3], T_body_sensor[:3, :3]
    h = DIFFERENCE_STEP
    before, now, after = (motion.sample(times + shift) for shift in (-h, 0.0, h))

    def sensor_position(sample) -> np.ndarray:
        return sample.position + sample.rotation @ offset

    acceleration = (sensor_position(after) - 2.0 * sensor_position(now)
                    + sensor_position(before)) / (h * h)
    acceleration[:, 2] += GRAVITY                   # a - g, with g = [0, 0, -9.81]
    R_field_sensor = now.rotation @ R_body_sensor
    force = np.einsum("nji,nj->ni", R_field_sensor, acceleration)
    relative = np.einsum("nji,njk->nik", before.rotation, after.rotation)
    rate_body = _rotation_vectors(relative) / (2.0 * h)
    return rate_body @ R_body_sensor, force


def simulate_imu(motion: BodyMotion, cfg: ImuConfig, rng: np.random.Generator,
                 T_body_sensor: np.ndarray | None = None, start: float = 0.0,
                 end: float | None = None) -> ImuStream:
    """Generate the whole IMU stream of one run. Deterministic for a given generator state."""
    end = motion.duration if end is None else float(end)
    period = 1.0 / cfg.rate_hz
    count = int(math.ceil((end - start) / period)) + 2
    intervals = period * np.clip(1.0 + cfg.rate_jitter * rng.normal(size=count), 0.2, 3.0)
    true_time = start + np.concatenate([[0.0], np.cumsum(intervals[:-1])])
    true_time = true_time[true_time <= end + period]
    count = len(true_time)

    # The sensor's true axes may differ slightly from where the robot believes they are.
    T_true = np.eye(4) if T_body_sensor is None else np.array(T_body_sensor, dtype=float)
    roll, pitch, yaw = (math.radians(v) for v in cfg.misalignment_deg)
    T_true[:3, :3] = T_true[:3, :3] @ quat.matrix_from_rpy(roll, pitch, yaw)
    gyro, accel = ideal_measurements(motion, true_time, T_true)

    dt = np.diff(true_time, prepend=true_time[0])
    for values, bias_std, walk, noise_std in (
            (gyro, cfg.gyro_bias_std, cfg.gyro_bias_walk, cfg.gyro_noise_std),
            (accel, cfg.accel_bias_std, cfg.accel_bias_walk, cfg.accel_noise_std)):
        bias = bias_std * rng.normal(size=3) + np.cumsum(
            walk * np.sqrt(dt)[:, None] * rng.normal(size=(count, 3)), axis=0)
        values += bias + noise_std * rng.normal(size=(count, 3))

    # ---- delivery: latency, lost samples, outages
    latency = np.maximum(cfg.latency_s + cfg.latency_jitter_s * rng.normal(size=count), 0.0)
    arrival = np.maximum.accumulate(true_time + latency)        # a serial link keeps the order
    delivered = rng.random(count) >= cfg.dropout_prob
    if cfg.outage_rate_hz > 0.0:
        time = start
        while True:
            time += float(rng.exponential(1.0 / cfg.outage_rate_hz))
            if time >= end:
                break
            length = float(rng.uniform(*cfg.outage_duration_s))
            delivered &= ~((true_time >= time) & (true_time < time + length))
            time += length
    truth = motion.sample(true_time)
    keep = np.flatnonzero(delivered)
    return ImuStream(timestamp=true_time[keep] + cfg.timestamp_offset_s, arrival=arrival[keep],
                     gyro=gyro[keep], accel=accel[keep], true_time=true_time[keep],
                     truth_roll=truth.roll[keep], truth_pitch=truth.pitch[keep],
                     generated=count)
