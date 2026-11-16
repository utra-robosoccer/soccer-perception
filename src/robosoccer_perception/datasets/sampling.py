"""Where the camera, the ball and the other robots are in each frame.

Camera poses are not arbitrary: the camera is a humanoid's head camera, so it stands on the
carpet at head height, pitched down, panned within neck limits (`randomization.sample_rig`).
What varies freely is where the robot stands and which way it faces.

Each frame is drawn from one of several MODES. ``random`` gives the bulk of ordinary views; the
others are deliberate edge cases that plain random sampling would produce too rarely (a distant
ball, a ball on a line, a half-hidden ball, a goal post at the image border, ...). The mode is
recorded with every frame so results can be broken down by it.

A mode only arranges the scene. Whether an object ends up visible, clipped or hidden is decided
by rendering, and the labels are read from the render (`datasets.annotation`), never from what a
mode intended.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from robosoccer_perception.datasets import randomization
from robosoccer_perception.geometry.camera import RigState, StereoCameraConfig
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.simulation.scene_objects import (
    FALLEN_POSES, HUMANOID_JOINTS, RobotState)

DEFAULT_MODE_WEIGHTS: dict[str, float] = {
    "random": 0.30,             # anywhere, facing anywhere
    "ball_far": 0.08,           # ball 4 m or more away: a few pixels
    "ball_near_line": 0.08,     # ball on or touching a painted line
    "ball_occluded": 0.08,      # a robot between the camera and the ball
    "ball_image_edge": 0.05,    # ball cut by the image border
    "goal_view": 0.10,          # looking toward a goal
    "goal_post_edge": 0.05,     # a goal post at the left or right image border
    "robot_on_line": 0.07,      # a robot standing on a painted line
    "robot_close": 0.05,        # a robot within a metre: large and cut off
    "empty_turf": 0.04,         # nothing but turf (and maybe lines): a negative example
    "midfield": 0.05,           # camera near the centre circle
    "touchline": 0.05,          # camera at a touchline
}

DEFAULT_CONTENT: dict[str, Any] = {
    "max_robots": 3,            # humanoid placeholders built into every scenario
    "ball_present_prob": 0.85,  # in modes that do not need the ball
    "robot_present_prob": 0.55, # per robot, in modes that do not need it
    "robot_fallen_prob": 0.12,
    "in_view_prob": 0.65,       # share of freely placed objects that are aimed into the view
    "camera_margin": 0.25,      # metres the camera keeps from the carpet edge
    "max_range": 10.0,          # metres; farthest deliberate object placement
}


@dataclasses.dataclass(frozen=True)
class FrameSpec:
    """Everything needed to set up one frame in a `MujocoScene`."""
    mode: str
    base_pose: tuple[float, float, float]           # camera platform [x, y, yaw]
    rig: RigState
    ball_xy: tuple[float, float] | None             # None = no ball in this frame
    ball_quat: tuple[float, float, float, float]
    robots: tuple[RobotState | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "base_pose_xy_yaw": [float(v) for v in self.base_pose],
                "rig": dataclasses.asdict(self.rig),
                "ball_xy": None if self.ball_xy is None else [float(v) for v in self.ball_xy],
                "ball_quat_wxyz": [float(v) for v in self.ball_quat],
                "robots": [None if robot is None else robot.to_dict() for robot in self.robots]}


class FrameSampler:
    """Draws `FrameSpec`s for one scenario."""

    def __init__(self, field: FieldGeometry, camera_cfg: StereoCameraConfig,
                 preset: dict[str, Any], content: dict[str, Any] | None = None,
                 mode_weights: dict[str, float] | None = None, num_robots: int | None = None,
                 ball_radius: float = 0.07):
        self.field, self.camera_cfg, self.preset = field, camera_cfg, preset
        self.content = {**DEFAULT_CONTENT, **(content or {})}
        weights = dict(DEFAULT_MODE_WEIGHTS if mode_weights is None else mode_weights)
        unknown = set(weights) - set(DEFAULT_MODE_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown frame modes: {sorted(unknown)}")
        self.modes = [mode for mode, weight in weights.items() if weight > 0]
        total = sum(weights[mode] for mode in self.modes)
        self.mode_probabilities = np.array([weights[mode] / total for mode in self.modes])
        self.num_robots = (int(self.content["max_robots"]) if num_robots is None
                           else int(num_robots))
        self.ball_radius = float(ball_radius)
        self.half_hfov = 0.5 * camera_cfg.intrinsics.horizontal_fov
        self.half_vfov = 0.5 * camera_cfg.intrinsics.vertical_fov
        self._line_points = field.sample_line_points(0.05, include_marks=True)
        self._posts = np.array([post for goal in field.goals for post in goal.posts])

    # ------------------------------------------------------------------ geometry helpers
    def _camera_xy(self, rng, region: str = "anywhere") -> np.ndarray:
        half_x, half_y = self.field.half_extent
        margin = float(self.content["camera_margin"])
        for _ in range(100):
            if region == "midfield":
                xy = rng.uniform(-1.0, 1.0, 2) * [0.18 * self.field.dims.length,
                                                  0.25 * self.field.dims.width]
            elif region == "touchline":
                xy = np.array([rng.uniform(-half_x + margin, half_x - margin),
                               rng.choice([-1, 1]) * (self.field.half_width
                                                      + rng.uniform(-0.5, 0.5))])
                xy[1] = np.clip(xy[1], -half_y + margin, half_y - margin)
            else:
                xy = rng.uniform(-1.0, 1.0, 2) * [half_x - margin, half_y - margin]
            if np.min(np.hypot(*(self._posts - xy).T)) > 0.35:      # not inside a goal post
                return xy
        return np.zeros(2)

    def _ground_range(self, rig: RigState) -> tuple[float, float]:
        """Nearest and farthest ground distance the camera sees straight ahead."""
        steep = rig.pitch + 0.85 * self.half_vfov
        shallow = rig.pitch - 0.85 * self.half_vfov
        near = rig.height / math.tan(steep) if steep < math.radians(88) else 0.05
        far = rig.height / math.tan(shallow) if shallow > math.radians(2) else math.inf
        return max(near, 0.05), far

    def _point_in_view(self, rng, camera_xy, heading: float, rig: RigState,
                       distance: tuple[float, float],
                       bearing_fraction=(-0.9, 0.9)) -> np.ndarray | None:
        """A point on the carpet that the camera can see, within ``distance`` if the view
        allows it. ``None`` when the camera looks off the carpet."""
        near, far = self._ground_range(rig)
        low = min(max(distance[0], near), max(far, near))
        high = max(min(distance[1], far), low)
        for _ in range(12):
            d = rng.uniform(low, high)
            bearing = rng.uniform(*bearing_fraction) * self.half_hfov
            xy = camera_xy + d * np.array([math.cos(heading + bearing),
                                           math.sin(heading + bearing)])
            if self.field.contains(xy, margin=0.1):
                return xy
        return None

    def _aim_at_carpet(self, rng, camera_xy, yaw: float, rig: RigState,
                       distance: tuple[float, float], bearing_fraction=(-0.9, 0.9)):
        """Turn (and if need be move) the camera until a point at ``distance`` in its view lies
        on the carpet. Returns ``(camera_xy, yaw, point)``."""
        for attempt in range(24):
            point = self._point_in_view(rng, camera_xy, yaw + rig.pan, rig, distance,
                                        bearing_fraction)
            if point is not None:
                return camera_xy, yaw, point
            yaw = float(rng.uniform(-math.pi, math.pi))
            if attempt >= 8:
                camera_xy = self._camera_xy(rng)
        return camera_xy, yaw, self._on_field(rng)

    def _on_field(self, rng) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, 2) * [self.field.half_length, self.field.half_width]

    def _yaw_toward(self, camera_xy, target_xy, rig: RigState, bearing: float = 0.0) -> float:
        """Base yaw that puts ``target_xy`` at horizontal angle ``bearing`` (positive = left of
        the image centre)."""
        direction = math.atan2(target_xy[1] - camera_xy[1], target_xy[0] - camera_xy[0])
        return direction - rig.pan - bearing

    def _pitch_toward(self, rig: RigState, distance: float, target_height: float,
                      image_fraction: float) -> RigState:
        """Pitch that puts a point at ``distance`` at a vertical image position
        (``image_fraction`` -1 = top edge, 0 = centre, +1 = bottom edge), within the preset's
        pitch limits."""
        depression = math.atan2(rig.height - target_height, max(distance, 1e-3))
        low, high = (math.radians(v) for v in self.preset["camera"]["pitch_limits_deg"])
        if low == high:
            return rig
        pitch = depression - image_fraction * self.half_vfov
        return rig.replace(pitch=float(np.clip(pitch, low, high)))

    def _camera_around(self, rng, target_xy, distance: tuple[float, float]) -> np.ndarray:
        """A camera position on the carpet at about ``distance`` from a target."""
        margin = float(self.content["camera_margin"])
        for _ in range(50):
            angle, d = rng.uniform(0, 2 * math.pi), rng.uniform(*distance)
            xy = np.asarray(target_xy) + d * np.array([math.cos(angle), math.sin(angle)])
            if (self.field.contains(xy, margin=margin)
                    and np.min(np.hypot(*(self._posts - xy).T)) > 0.35):
                return xy
        return self._camera_xy(rng)

    def _robot_state(self, rng, xy, yaw: float | None = None, allow_fall: bool = True) -> RobotState:
        if rng.random() < 0.3:
            joints = rng.normal(0.0, 0.05, len(HUMANOID_JOINTS))        # standing
            joints[4:] = np.abs(joints[4:])
        else:
            stride = rng.uniform(-0.55, 0.55)
            joints = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9), stride, -stride,
                               rng.uniform(0.0, 0.7), rng.uniform(0.0, 0.7)])
        fallen = None
        if allow_fall and rng.random() < float(self.content["robot_fallen_prob"]):
            fallen = str(rng.choice(FALLEN_POSES))
        return RobotState((float(xy[0]), float(xy[1])),
                          float(rng.uniform(-math.pi, math.pi) if yaw is None else yaw),
                          tuple(float(v) for v in joints), fallen)

    # ------------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator, mode: str | None = None) -> FrameSpec:
        mode = mode or str(rng.choice(self.modes, p=self.mode_probabilities))
        if mode not in DEFAULT_MODE_WEIGHTS:
            raise ValueError(f"unknown frame mode {mode!r}")
        content = self.content
        rig = randomization.sample_rig(rng, self.preset, self.camera_cfg.mount)
        max_range = float(content["max_range"])
        region = mode if mode in ("midfield", "touchline") else "anywhere"
        camera_xy = self._camera_xy(rng, region)
        yaw = float(rng.uniform(-math.pi, math.pi))
        ball_xy: np.ndarray | None = None
        forced: dict[int, RobotState] = {}          # robots a mode places deliberately
        free_ball = free_robots = True

        if mode == "touchline":
            side = math.copysign(1.0, camera_xy[1])
            if rng.random() < 0.5:                  # look along the line
                yaw = float(rng.choice([0.0, math.pi]) + rng.normal(0, 0.25)) - rig.pan
            else:                                   # look into the field
                yaw = -side * 0.5 * math.pi + float(rng.normal(0, 0.5)) - rig.pan
        elif mode == "ball_far":
            if self._ground_range(rig)[1] < 5.0:
                rig = self._pitch_toward(rig, 8.0, 0.0, rng.uniform(-0.6, 0.2))
            camera_xy, yaw, ball_xy = self._aim_at_carpet(rng, camera_xy, yaw, rig,
                                                          (4.0, max_range))
            free_ball = False
        elif mode == "ball_near_line":
            point = self._line_points[rng.integers(len(self._line_points))]
            ball_xy = point + rng.uniform(-1, 1, 2) * (0.5 * self.field.dims.line_width
                                                       + self.ball_radius)
            camera_xy = self._camera_around(rng, ball_xy, (0.7, 5.0))
            distance = float(np.hypot(*(ball_xy - camera_xy)))
            rig = self._pitch_toward(rig, distance, self.ball_radius, rng.uniform(-0.7, 0.7))
            yaw = self._yaw_toward(camera_xy, ball_xy, rig,
                                   rng.uniform(-0.8, 0.8) * self.half_hfov)
            free_ball = False
        elif mode == "ball_occluded":
            camera_xy, yaw, ball_xy = self._aim_at_carpet(rng, camera_xy, yaw, rig, (1.2, 6.0),
                                                          (-0.6, 0.6))
            free_ball = False
            if self.num_robots:
                along = ball_xy - camera_xy
                distance = float(np.hypot(*along))
                across = np.array([-along[1], along[0]]) / max(distance, 1e-6)
                spot = (camera_xy + along * rng.uniform(0.45, 0.85)
                        + across * rng.uniform(-0.12, 0.12))
                forced[0] = self._robot_state(rng, spot, allow_fall=False)
        elif mode == "ball_image_edge":
            near, _ = self._ground_range(rig)
            if rng.random() < 0.6:                  # cut by the left or right border
                bearing = float(rng.choice([-1, 1])) * self.half_hfov * rng.uniform(0.93, 1.03)
                d = rng.uniform(max(near, 0.6) * 1.3, 4.0)
            else:                                   # cut by the bottom border
                bearing = rng.uniform(-0.6, 0.6) * self.half_hfov
                steep = rig.pitch + self.half_vfov * rng.uniform(0.94, 1.04)
                d = rig.height / math.tan(min(steep, math.radians(85))) * math.cos(bearing)
            # A ray at image-plane angle `bearing` leaves the camera at ground azimuth
            # atan(tan(bearing) / cos(pitch)) when the camera is pitched (roll neglected).
            azimuth = math.atan2(math.tan(bearing), math.cos(rig.pitch))
            for _ in range(24):                     # face a direction that keeps it on the carpet
                heading = yaw + rig.pan
                ball_xy = camera_xy + d * np.array([math.cos(heading + azimuth),
                                                    math.sin(heading + azimuth)])
                if self.field.contains(ball_xy, margin=0.1):
                    break
                yaw = float(rng.uniform(-math.pi, math.pi))
            free_ball = False
        elif mode in ("goal_view", "goal_post_edge"):
            goal = self.field.goals[int(rng.integers(len(self.field.goals)))]
            if mode == "goal_view":
                target = np.array([goal.line_x, rng.uniform(-1, 1) * 0.5 * goal.dims.width])
                camera_xy = self._camera_around(rng, target, (1.0, max_range))
                bearing = rng.uniform(-0.6, 0.6) * self.half_hfov
            else:
                target = np.array(goal.posts[int(rng.integers(2))])
                camera_xy = self._camera_around(rng, target, (0.8, 7.0))
                bearing = float(rng.choice([-1, 1])) * self.half_hfov * rng.uniform(0.9, 1.04)
                bearing = math.atan2(math.tan(bearing), math.cos(rig.pitch))
            yaw = self._yaw_toward(camera_xy, target, rig, bearing)
        elif mode == "robot_on_line" and self.num_robots:
            point = self._line_points[rng.integers(len(self._line_points))]
            camera_xy = self._camera_around(rng, point, (1.0, 6.0))
            distance = float(np.hypot(*(point - camera_xy)))
            rig = self._pitch_toward(rig, distance, 0.3, rng.uniform(-0.5, 0.6))
            yaw = self._yaw_toward(camera_xy, point, rig, rng.uniform(-0.7, 0.7) * self.half_hfov)
            forced[0] = self._robot_state(rng, point)
        elif mode == "robot_close" and self.num_robots:
            heading = yaw + rig.pan + rng.uniform(-0.8, 0.8) * self.half_hfov
            spot = camera_xy + rng.uniform(0.4, 1.0) * np.array([math.cos(heading),
                                                                 math.sin(heading)])
            forced[0] = self._robot_state(rng, spot, allow_fall=False)
        elif mode == "empty_turf":
            low, high = (math.radians(v) for v in self.preset["camera"]["pitch_limits_deg"])
            if low != high:
                rig = rig.replace(pitch=float(rng.uniform(max(low, min(high, math.radians(32))),
                                                          high)))
            free_ball = free_robots = False

        heading = float(yaw + rig.pan)

        def free_spot(near: float) -> np.ndarray:
            point = None
            if rng.random() < float(content["in_view_prob"]):
                point = self._point_in_view(rng, camera_xy, heading, rig, (near, max_range))
            return self._on_field(rng) if point is None else point

        if free_ball and rng.random() < float(content["ball_present_prob"]):
            ball_xy = free_spot(0.4)
        robots: list[RobotState | None] = []
        occupied = [camera_xy] + [np.asarray(state.xy) for state in forced.values()]
        for index in range(self.num_robots):
            if index in forced:
                robots.append(forced[index])
                continue
            state = None
            if free_robots and rng.random() < float(content["robot_present_prob"]):
                for _ in range(10):
                    xy = free_spot(0.6)
                    clear = min(np.hypot(*(xy - other)) for other in occupied) > 0.45
                    if clear and (ball_xy is None or np.hypot(*(xy - ball_xy)) > 0.30):
                        state = self._robot_state(rng, xy)
                        occupied.append(xy)
                        break
            robots.append(state)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        yaw = float((yaw + math.pi) % (2 * math.pi) - math.pi)
        return FrameSpec(mode, (float(camera_xy[0]), float(camera_xy[1]), yaw), rig,
                         None if ball_xy is None else (float(ball_xy[0]), float(ball_xy[1])),
                         tuple(float(v) for v in quat), tuple(robots))
