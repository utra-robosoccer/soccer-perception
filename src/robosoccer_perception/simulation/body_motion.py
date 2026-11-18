"""Synthetic motion of the robot's body: what a walking, swaying or pushed humanoid does to a
camera that is bolted to it. Pure NumPy; no physics, no policy, no MuJoCo.

SYNTHETIC. The amplitudes and frequencies in ``configs/motion/*.yaml`` are plausible for a
70 cm humanoid and nothing more: they have not been fitted to measured motion of
humanoid_prototype_1, which cannot walk yet. Replace them when IMU logs of the real robot exist.

What is modelled
----------------
A Phase 1 `Trajectory` gives the smooth path of the robot over the field (x, y, heading every
``dt``). `BodyMotion` turns it into the six-degree-of-freedom motion of the BODY frame (the
pelvis) as a smooth function of time, by adding

* GAIT terms, active while the path moves: lateral shift, roll and yaw at the stride
  frequency; vertical bob, pitch and a speed ripple at the step frequency (two steps per
  stride); a constant forward lean;
* POSTURAL SWAY: slow pitch and roll of the whole body about the ankles, like an inverted
  pendulum, so the body also moves forward when it pitches forward;
* WOBBLE: band-limited random roll, pitch and height;
* DISTURBANCES: brief tilts with a fast rise and a slower recovery (a push, a stumble), again
  about the ankles.

Everything is an analytic function of time (sums of sinusoids and smooth pulses over a C1
spline of the path), so the pose can be evaluated at any instant: at the camera's frame times
and, much more often, at the IMU's sample times. The simulated IMU differentiates exactly this
motion (`simulation.imu`), which is what makes camera and IMU physically consistent.

Frames: see `geometry.mount`. The FOOTPRINT (the point on the ground under the body origin,
turned to the body's heading) is what the localizer estimates and what ground truth reports;
it includes the gait's lateral shift and yaw, as the pelvis of a real robot does.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any

import numpy as np

from robosoccer_perception import config
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.simulation.trajectories import Trajectory


@dataclasses.dataclass(frozen=True)
class MotionProfile:
    """One ``configs/motion/*.yaml`` file. Angles in degrees, lengths in metres (1-sided
    amplitudes), frequencies in hertz."""
    name: str
    description: str = ""
    # ---- gait: scaled by how fast the path moves (or always, for stepping in place)
    step_frequency_hz: float = 2.2
    gait_roll_deg: float = 0.0
    gait_pitch_deg: float = 0.0
    gait_yaw_deg: float = 0.0
    gait_lateral_m: float = 0.0
    gait_vertical_m: float = 0.0
    gait_surge_m: float = 0.0
    gait_lean_pitch_deg: float = 0.0
    gait_always: bool = False
    walk_speed_reference: float = 0.15          # m/s at which the gait terms are fully on
    turn_rate_reference_deg: float = 20.0       # deg/s, likewise for turning in place
    # ---- postural sway about the ankles
    sway_pitch_deg: float = 0.0
    sway_pitch_hz: float = 0.5
    sway_roll_deg: float = 0.0
    sway_roll_hz: float = 0.4
    # ---- band-limited wobble (RMS values)
    noise_roll_deg: float = 0.0
    noise_pitch_deg: float = 0.0
    noise_height_m: float = 0.0
    noise_bandwidth_hz: float = 3.0
    # ---- disturbances
    disturbance_rate_hz: float = 0.0            # mean number per second
    disturbance_deg: tuple[float, float] = (4.0, 8.0)   # peak tilt, drawn uniformly
    disturbance_rise_s: float = 0.12
    disturbance_recover_s: float = 0.7
    disturbance_start_s: float = 2.0            # none before this time

    def __post_init__(self) -> None:
        if self.step_frequency_hz <= 0.0 or self.noise_bandwidth_hz <= 0.0:
            raise ValueError("frequencies must be positive")
        if self.disturbance_rise_s <= 0.0 or self.disturbance_recover_s <= 0.0:
            raise ValueError("disturbance times must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionProfile":
        data = dict(data)
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown MotionProfile keys: {sorted(unknown)}")
        if "disturbance_deg" in data:
            data["disturbance_deg"] = tuple(float(v) for v in data["disturbance_deg"])
        return cls(**data)

    @classmethod
    def load(cls, name_or_path: str | os.PathLike) -> "MotionProfile":
        return cls.from_dict(config.load_config("motion", name_or_path))

    def replace(self, **changes) -> "MotionProfile":
        return dataclasses.replace(self, **changes)

    @property
    def is_still(self) -> bool:
        """True when the profile adds no motion at all (a rigid, level body)."""
        return not any((self.gait_roll_deg, self.gait_pitch_deg, self.gait_yaw_deg,
                        self.gait_lateral_m, self.gait_vertical_m, self.gait_surge_m,
                        self.gait_lean_pitch_deg, self.sway_pitch_deg, self.sway_roll_deg,
                        self.noise_roll_deg, self.noise_pitch_deg, self.noise_height_m,
                        self.disturbance_rate_hz))


def available_profiles() -> list[str]:
    return sorted(p.stem for p in (config.config_dir() / "motion").glob("*.yaml"))


# --------------------------------------------------------------------------------------------
# Smooth functions of time
# --------------------------------------------------------------------------------------------
class UniformSpline:
    """C1 cubic (Catmull-Rom) interpolation of samples taken every ``dt`` from time 0.

    It passes through every sample, so the path at the trajectory's own times is exactly the
    trajectory. Outside the sampled interval the curve continues in a straight line at its end
    slope (``extrapolate``) or holds the end value. A path is extrapolated: a robot that is
    walking at the first sample was walking just before it too, and stopping it dead there
    would be an acceleration that never happened.
    """

    def __init__(self, values: np.ndarray, dt: float, extrapolate: bool = False):
        self.extrapolate = bool(extrapolate)
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        self.dt = float(dt)
        count = len(self.values)
        tangent = np.zeros_like(self.values)
        if count > 2:
            tangent[1:-1] = 0.5 * (self.values[2:] - self.values[:-2])
        if count > 1:
            tangent[0] = self.values[1] - self.values[0]
            tangent[-1] = self.values[-1] - self.values[-2]
        self._tangent = tangent                 # per unit of the normalised parameter

    def __call__(self, times) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times, dtype=float))
        count = len(self.values)
        if count == 1:
            return np.repeat(self.values, len(times), axis=0)
        raw = times / self.dt
        u = np.clip(raw, 0.0, count - 1.0)
        index = np.minimum(u.astype(int), count - 2)
        s = (u - index)[:, None]
        p0, p1 = self.values[index], self.values[index + 1]
        m0, m1 = self._tangent[index], self._tangent[index + 1]
        s2, s3 = s * s, s * s * s
        value = ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0
                 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1)
        if self.extrapolate:
            outside = (raw - u)[:, None]            # negative before the start, positive after
            value = value + outside * np.where(outside < 0.0, self._tangent[0],
                                               self._tangent[-1])
        return value


def smoothstep(u) -> np.ndarray:
    """0 below 0, 1 above 1, and a C2 quintic in between."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


@dataclasses.dataclass(frozen=True)
class _Tones:
    """A sum of sinusoids with fixed random frequencies and phases, scaled to a target RMS."""
    amplitude: np.ndarray
    omega: np.ndarray
    phase: np.ndarray

    @classmethod
    def random(cls, rng: np.random.Generator, rms: float, bandwidth_hz: float,
               count: int = 6) -> "_Tones":
        frequency = rng.uniform(0.15, 1.0, count) * bandwidth_hz
        weights = rng.uniform(0.5, 1.0, count)
        amplitude = weights * rms / math.sqrt(0.5 * float(np.sum(weights ** 2)))
        return cls(amplitude, 2.0 * math.pi * frequency, rng.uniform(0.0, 2.0 * math.pi, count))

    def __call__(self, times: np.ndarray) -> np.ndarray:
        return (self.amplitude * np.sin(np.outer(times, self.omega) + self.phase)).sum(axis=1)


@dataclasses.dataclass(frozen=True)
class Disturbance:
    """One brief tilt: time it starts, peak roll and pitch (radians)."""
    start: float
    roll: float
    pitch: float


@dataclasses.dataclass(frozen=True)
class BodySample:
    """The body's true motion at one or more instants. SIMULATOR SIDE: ground truth."""
    times: np.ndarray           # (N,)
    position: np.ndarray        # (N, 3) body origin in the field frame
    rotation: np.ndarray        # (N, 3, 3) R_field_body
    roll: np.ndarray            # (N,) radians, of ``Rz(yaw) Ry(pitch) Rx(roll)``
    pitch: np.ndarray
    yaw: np.ndarray
    gait_phase: np.ndarray      # (N,) radians; advances by 2 pi per step
    activity: np.ndarray        # (N,) 0 = standing, 1 = walking at full amplitude

    def __len__(self) -> int:
        return len(self.times)

    @property
    def height(self) -> np.ndarray:
        """Body origin above the ground, metres."""
        return self.position[:, 2]

    @property
    def footprint(self) -> np.ndarray:
        """(N, 3) true ``[x, y, yaw]`` of the footprint frame: the localizer's ground truth."""
        return np.column_stack([self.position[:, 0], self.position[:, 1],
                                tf.wrap_angle(self.yaw)])

    def T_field_body(self, index: int = 0) -> np.ndarray:
        return tf.make_transform(self.rotation[index], self.position[index])


class BodyMotion:
    """The true motion of the body for one path, one profile and one seed."""

    def __init__(self, trajectory: Trajectory, profile: MotionProfile, standing_height: float,
                 rng: np.random.Generator | None = None):
        if trajectory.kidnap_steps:
            raise ValueError("body motion needs a continuous path; this trajectory teleports")
        rng = rng or np.random.default_rng(0)
        self.trajectory, self.profile = trajectory, profile
        self.standing_height = float(standing_height)
        self.dt = float(trajectory.dt)
        poses = np.asarray(trajectory.poses, dtype=float)
        unwrapped = np.unwrap(poses[:, 2])
        self._path = UniformSpline(np.column_stack([poses[:, 0], poses[:, 1], unwrapped]), self.dt,
                                   extrapolate=True)

        # How much the robot is walking at each instant: 0 standing ... 1 full gait.
        if profile.gait_always or len(poses) < 2:
            activity = np.ones(len(poses))
        else:
            speed = np.hypot(*np.diff(poses[:, :2], axis=0).T) / self.dt
            turn = np.abs(np.diff(unwrapped)) / self.dt
            level = np.clip(np.maximum(speed / profile.walk_speed_reference,
                                       turn / math.radians(profile.turn_rate_reference_deg)),
                            0.0, 1.0)
            level = np.concatenate([level[:1], level])
            window = max(1, int(round(0.4 / self.dt)))
            padded = np.pad(level, window, mode="edge")
            activity = np.convolve(padded, np.ones(2 * window + 1) / (2 * window + 1),
                                   mode="valid")
        self._activity = UniformSpline(activity, self.dt)

        # Draw every random quantity up front, in a fixed order, so a seed fixes the motion.
        self._gait_phase0 = float(rng.uniform(0.0, 2.0 * math.pi))
        self._sway_phase = rng.uniform(0.0, 2.0 * math.pi, 2)
        bandwidth = profile.noise_bandwidth_hz
        self._noise_roll = _Tones.random(rng, math.radians(profile.noise_roll_deg), bandwidth)
        self._noise_pitch = _Tones.random(rng, math.radians(profile.noise_pitch_deg), bandwidth)
        self._noise_height = _Tones.random(rng, profile.noise_height_m, bandwidth)
        self.disturbances = self._draw_disturbances(rng, trajectory.duration)

    def _draw_disturbances(self, rng: np.random.Generator, duration: float
                           ) -> tuple[Disturbance, ...]:
        profile = self.profile
        events = []
        if profile.disturbance_rate_hz > 0.0:
            time = profile.disturbance_start_s
            spacing = profile.disturbance_rise_s + profile.disturbance_recover_s
            while True:
                time += float(rng.exponential(1.0 / profile.disturbance_rate_hz))
                if time + spacing > duration:
                    break
                peak = math.radians(float(rng.uniform(*profile.disturbance_deg)))
                direction = float(rng.uniform(0.0, 2.0 * math.pi))
                events.append(Disturbance(time, peak * math.sin(direction),
                                          peak * math.cos(direction)))
                time += spacing                 # one at a time
        return tuple(events)

    @property
    def duration(self) -> float:
        return self.trajectory.duration

    # ------------------------------------------------------------------ evaluation
    def sample(self, times) -> BodySample:
        """The body's pose at ``times`` (seconds; scalar or array)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        profile, height = self.profile, self.standing_height
        path = self._path(times)
        activity = np.clip(self._activity(times)[:, 0], 0.0, 1.0)
        step_phase = 2.0 * math.pi * profile.step_frequency_hz * times + self._gait_phase0
        stride_phase = 0.5 * step_phase

        # ---- pendulum terms: the whole body turning about the ankles
        pendulum_pitch = math.radians(profile.sway_pitch_deg) * np.sin(
            2.0 * math.pi * profile.sway_pitch_hz * times + self._sway_phase[0])
        pendulum_roll = math.radians(profile.sway_roll_deg) * np.sin(
            2.0 * math.pi * profile.sway_roll_hz * times + self._sway_phase[1])
        for event in self.disturbances:
            since = times - event.start
            pulse = smoothstep(since / profile.disturbance_rise_s) * (
                1.0 - smoothstep((since - profile.disturbance_rise_s)
                                 / profile.disturbance_recover_s))
            pendulum_roll = pendulum_roll + event.roll * pulse
            pendulum_pitch = pendulum_pitch + event.pitch * pulse

        # ---- gait terms
        roll = pendulum_roll + self._noise_roll(times) \
            + activity * math.radians(profile.gait_roll_deg) * np.sin(stride_phase)
        pitch = pendulum_pitch + self._noise_pitch(times) + activity * np.radians(
            profile.gait_lean_pitch_deg + profile.gait_pitch_deg * np.sin(step_phase + 0.6))
        yaw_wobble = activity * math.radians(profile.gait_yaw_deg) * np.cos(stride_phase)
        forward = height * np.sin(pendulum_pitch) \
            + activity * profile.gait_surge_m * np.sin(step_phase)
        left = -height * np.sin(pendulum_roll) \
            + activity * profile.gait_lateral_m * np.sin(stride_phase)
        up = height * np.cos(pendulum_roll) * np.cos(pendulum_pitch) \
            - activity * profile.gait_vertical_m * np.cos(step_phase) + self._noise_height(times)

        cos, sin = np.cos(path[:, 2]), np.sin(path[:, 2])
        position = np.column_stack([path[:, 0] + cos * forward - sin * left,
                                    path[:, 1] + sin * forward + cos * left, up])
        yaw = path[:, 2] + yaw_wobble
        return BodySample(times, position, _rotations(roll, pitch, yaw), roll, pitch, yaw,
                          step_phase, activity)

    def footprint_trajectory(self, name: str | None = None) -> Trajectory:
        """The true footprint poses at the path's own sample times, as a `Trajectory`: what
        odometry is derived from and what the localizer's estimate is scored against."""
        sample = self.sample(self.trajectory.times)
        return dataclasses.replace(self.trajectory, name=name or self.trajectory.name,
                                   poses=sample.footprint)


def _rotations(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """``Rz(yaw) Ry(pitch) Rx(roll)`` for arrays of angles: (N, 3, 3)."""
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.empty((len(roll), 3, 3))
    rotation[:, 0, 0] = cy * cp
    rotation[:, 0, 1] = cy * sp * sr - sy * cr
    rotation[:, 0, 2] = cy * sp * cr + sy * sr
    rotation[:, 1, 0] = sy * cp
    rotation[:, 1, 1] = sy * sp * sr + cy * cr
    rotation[:, 1, 2] = sy * sp * cr - cy * sr
    rotation[:, 2, 0] = -sp
    rotation[:, 2, 1] = cp * sr
    rotation[:, 2, 2] = cp * cr
    return rotation
