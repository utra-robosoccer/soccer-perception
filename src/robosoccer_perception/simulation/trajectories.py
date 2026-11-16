"""Scripted, repeatable motion of the camera platform. Pure NumPy; no physics, no policy.

A `Trajectory` is a sequence of true base poses sampled every ``dt`` seconds, plus optional
head motion. `PathBuilder` assembles one from simple moves (hold, turn, walk, arc, teleport).
Any randomness comes from a `numpy.random.Generator` supplied by the caller, so a fixed seed
gives the same trajectory every time.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from robosoccer_perception.geometry import transforms as tf


@dataclasses.dataclass(frozen=True)
class Trajectory:
    name: str
    dt: float
    poses: np.ndarray                       # (T, 3) true [x, y, yaw] in the field frame
    pan: np.ndarray | None = None           # (T,) head pan in radians; None = the mount default
    pitch: np.ndarray | None = None         # (T,) camera pitch in radians; None = mount default
    kidnap_steps: tuple[int, ...] = ()      # steps at which the pose jumps (robot was moved)

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def times(self) -> np.ndarray:
        return self.dt * np.arange(len(self.poses))

    @property
    def duration(self) -> float:
        return self.dt * (len(self.poses) - 1)

    def with_head_scan(self, amplitude: float, period: float) -> "Trajectory":
        """Add a sinusoidal left-right head pan (radians, seconds)."""
        pan = amplitude * np.sin(2.0 * math.pi * self.times / period)
        return dataclasses.replace(self, pan=pan)

    def with_pitch(self, pitch: float) -> "Trajectory":
        return dataclasses.replace(self, pitch=np.full(len(self.poses), float(pitch)))


def _steps(exact: float) -> int:
    """Whole number of steps for a move, rounded up so the commanded speed is never exceeded."""
    return max(1, int(math.ceil(exact - 1e-9)))


class PathBuilder:
    """Build a trajectory from moves. Speeds are in m/s and rad/s; angles in radians."""

    def __init__(self, start_pose, dt: float = 0.1):
        self.dt = float(dt)
        self._poses = [np.array(start_pose, dtype=float)]
        self._kidnaps: list[int] = []

    @property
    def pose(self) -> np.ndarray:
        return self._poses[-1]

    def _append(self, pose) -> None:
        pose = np.array(pose, dtype=float)
        pose[2] = tf.wrap_angle(pose[2])
        self._poses.append(pose)

    def hold(self, duration: float) -> "PathBuilder":
        for _ in range(int(round(duration / self.dt))):
            self._append(self.pose)
        return self

    def turn(self, angle: float, rate: float = math.radians(45)) -> "PathBuilder":
        """Rotate in place by ``angle`` (positive = counter-clockwise)."""
        steps = _steps(abs(angle) / (rate * self.dt))
        for _ in range(steps):
            self._append(self.pose + [0.0, 0.0, angle / steps])
        return self

    def turn_to(self, heading: float, rate: float = math.radians(45)) -> "PathBuilder":
        delta = float(tf.angle_difference(heading, self.pose[2]))
        return self.turn(delta, rate) if abs(delta) > 1e-9 else self

    def forward(self, distance: float, speed: float = 0.3) -> "PathBuilder":
        """Walk straight along the current heading (negative distance walks backward)."""
        steps = _steps(abs(distance) / (speed * self.dt))
        step = distance / steps
        for _ in range(steps):
            x, y, yaw = self.pose
            self._append([x + step * math.cos(yaw), y + step * math.sin(yaw), yaw])
        return self

    def sidestep(self, distance: float, speed: float = 0.15) -> "PathBuilder":
        """Walk sideways (positive = to the robot's left) without turning."""
        steps = _steps(abs(distance) / (speed * self.dt))
        step = distance / steps
        for _ in range(steps):
            x, y, yaw = self.pose
            self._append([x - step * math.sin(yaw), y + step * math.cos(yaw), yaw])
        return self

    def arc(self, angle: float, radius: float, speed: float = 0.3) -> "PathBuilder":
        """Walk a circular arc, turning by ``angle`` in total (positive = to the left)."""
        steps = _steps(abs(angle) * radius / (speed * self.dt))
        step = tf.se2_compose([0.0, 0.0, angle / (2 * steps)],
                              [2.0 * radius * abs(math.sin(angle / (2 * steps))), 0.0,
                               angle / (2 * steps)])
        for _ in range(steps):
            self._append(tf.se2_compose(self.pose, step))
        return self

    def go_to(self, x: float, y: float, speed: float = 0.3,
              turn_rate: float = math.radians(45)) -> "PathBuilder":
        """Turn toward a point, then walk to it."""
        dx, dy = x - self.pose[0], y - self.pose[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return self
        return self.turn_to(math.atan2(dy, dx), turn_rate).forward(distance, speed)

    def teleport(self, pose) -> "PathBuilder":
        """Jump to another pose in one step: a kidnapping. Odometry does not see it."""
        self._append(pose)
        self._kidnaps.append(len(self._poses) - 1)
        return self

    def build(self, name: str) -> Trajectory:
        return Trajectory(name, self.dt, np.array(self._poses), kidnap_steps=tuple(self._kidnaps))


# ------------------------------------------------------------------ ready-made trajectories
def stationary(pose, duration: float, dt: float = 0.1) -> Trajectory:
    return PathBuilder(pose, dt).hold(duration).build("stationary")


def rotate_in_place(pose, turns: float = 1.0, rate: float = math.radians(30),
                    dt: float = 0.1) -> Trajectory:
    return PathBuilder(pose, dt).turn(2.0 * math.pi * turns, rate).build("rotate_in_place")


def straight_line(start_pose, distance: float, speed: float = 0.3, dt: float = 0.1) -> Trajectory:
    return PathBuilder(start_pose, dt).forward(distance, speed).build("straight_line")


def rectangle(corner_pose, size_x: float, size_y: float, laps: int = 1, speed: float = 0.3,
              turn_rate: float = math.radians(45), dt: float = 0.1) -> Trajectory:
    """Walk a rectangle counter-clockwise, turning in place at each corner."""
    path = PathBuilder(corner_pose, dt)
    for _ in range(laps):
        for side in (size_x, size_y, size_x, size_y):
            path.forward(side, speed).turn(math.pi / 2, turn_rate)
    return path.build("rectangle")


def circle(start_pose, radius: float, turns: float = 1.0, speed: float = 0.3,
           dt: float = 0.1) -> Trajectory:
    """Walk a circle to the left, starting tangent to it at ``start_pose``."""
    return PathBuilder(start_pose, dt).arc(2.0 * math.pi * turns, radius, speed).build("circle")


def random_waypoints(start_pose, bounds: tuple[float, float], count: int,
                     rng: np.random.Generator, speed: float = 0.3,
                     turn_rate: float = math.radians(45), dt: float = 0.1) -> Trajectory:
    """Visit ``count`` uniformly random points with |x| <= bounds[0], |y| <= bounds[1]."""
    path = PathBuilder(start_pose, dt)
    for _ in range(count):
        x, y = rng.uniform([-bounds[0], -bounds[1]], [bounds[0], bounds[1]])
        path.go_to(float(x), float(y), speed, turn_rate)
    return path.build("random_waypoints")


def random_pose(bounds: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """A uniformly random pose with |x| <= bounds[0], |y| <= bounds[1] and any heading."""
    x, y = rng.uniform([-bounds[0], -bounds[1]], [bounds[0], bounds[1]])
    return np.array([x, y, rng.uniform(-math.pi, math.pi)])
