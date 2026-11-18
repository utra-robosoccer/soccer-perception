"""Bringing IMU orientation and camera frames to the same instant. Pure Python / NumPy.

The IMU runs at a few hundred hertz and the camera at ten to thirty; their samples never share
a timestamp, and an image usually reaches the pipeline tens of milliseconds after it was
exposed. A body that pitches at 30 deg/s turns 1 degree in 33 ms, so "the newest IMU estimate"
is not the orientation the camera had when it took the picture.

`OrientationHistory` keeps the recent `OrientationEstimate`s by timestamp and answers "what
was the tilt at time t":

* INTERPOLATED  t lies between two stored estimates: spherical interpolation of the
                quaternions (the normal case, because the IMU arrives sooner than the image);
* EXTRAPOLATED  t is newer than every estimate (the IMU is late): the newest estimate is
                advanced with its own angular rate, for at most ``max_extrapolation_s``;
* HELD          t is further ahead than that: the newest estimate is used as it is;
* STALE         t is older than everything stored: the oldest estimate is used;
* NONE          nothing is stored.

Every answer carries its timing (`SyncedTilt.offset`, ``gap``) and an uncertainty that grows
with the distance in time that had to be bridged. `SyncStatistics` counts what happened over a
run: these are the timestamp-mismatch diagnostics.

`OrientationHistory.latest` is the naive alternative (the newest estimate, whatever the
image's timestamp). It exists so that the benchmark can measure what interpolation is worth.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
import math

from robosoccer_perception.estimation.orientation import OrientationEstimate
from robosoccer_perception.geometry import quaternion as quat

INTERPOLATED, EXTRAPOLATED, HELD, STALE, NONE, LATEST = (
    "interpolated", "extrapolated", "held", "stale", "none", "latest")
METHODS = (INTERPOLATED, EXTRAPOLATED, HELD, STALE, NONE, LATEST)


@dataclasses.dataclass(frozen=True)
class SyncConfig:
    history_s: float = 2.0              # how much orientation history is kept
    max_extrapolation_s: float = 0.05   # beyond this the newest estimate is held, not advanced
    expected_period_s: float = 0.005    # the IMU's nominal sampling interval
    sparse_factor: float = 4.0          # a bracket wider than this many periods is "sparse"
    timestamp_std_s: float = 0.003      # how well camera and IMU clocks are believed to agree


@dataclasses.dataclass(frozen=True)
class SyncedTilt:
    """The body's tilt at one image timestamp, and how it was obtained."""
    roll: float                 # radians
    pitch: float
    tilt_std: float             # radians: the filter's own uncertainty plus the timing terms
    method: str                 # one of METHODS
    offset: float               # seconds from the image timestamp to the nearest IMU estimate
    gap: float                  # seconds between the two estimates that bracket the timestamp
    newest_age: float           # seconds from the newest IMU estimate back to the image time
    rate: float                 # rad/s: how fast the body was turning there

    @property
    def available(self) -> bool:
        return self.method != NONE


@dataclasses.dataclass
class SyncStatistics:
    """Counts and extremes over a run."""
    frames: int = 0
    by_method: dict = dataclasses.field(default_factory=lambda: {m: 0 for m in METHODS})
    sparse: int = 0             # interpolations across an unusually wide bracket
    max_gap: float = 0.0
    max_offset: float = 0.0
    sum_offset: float = 0.0

    def add(self, tilt: SyncedTilt, sparse: bool) -> None:
        self.frames += 1
        self.by_method[tilt.method] += 1
        self.sparse += int(sparse)
        if math.isfinite(tilt.gap):
            self.max_gap = max(self.max_gap, tilt.gap)
        if math.isfinite(tilt.offset):
            self.max_offset = max(self.max_offset, abs(tilt.offset))
            self.sum_offset += abs(tilt.offset)

    def summary(self) -> dict:
        frames = max(self.frames, 1)
        return {"frames": self.frames, "sparse_brackets": self.sparse,
                "max_bracket_gap_ms": 1000.0 * self.max_gap,
                "max_offset_ms": 1000.0 * self.max_offset,
                "mean_offset_ms": 1000.0 * self.sum_offset / frames,
                **{f"share_{name}": count / frames for name, count in self.by_method.items()}}


class OrientationHistory:
    """Recent orientation estimates, ordered by timestamp."""

    def __init__(self, cfg: SyncConfig | None = None):
        self.cfg = cfg or SyncConfig()
        self._times: collections.deque[float] = collections.deque()
        self._estimates: collections.deque[OrientationEstimate] = collections.deque()
        self.statistics = SyncStatistics()

    def __len__(self) -> int:
        return len(self._times)

    def clear(self) -> None:
        self._times.clear()
        self._estimates.clear()
        self.statistics = SyncStatistics()

    def add(self, estimate: OrientationEstimate) -> None:
        if type(estimate) is not OrientationEstimate:
            raise TypeError("the history stores OrientationEstimate objects")
        if self._times and estimate.timestamp <= self._times[-1]:
            return                              # the filter never emits these; be safe anyway
        self._times.append(estimate.timestamp)
        self._estimates.append(estimate)
        while self._times and estimate.timestamp - self._times[0] > self.cfg.history_s:
            self._times.popleft()
            self._estimates.popleft()

    @property
    def newest(self) -> OrientationEstimate | None:
        return self._estimates[-1] if self._estimates else None

    # ------------------------------------------------------------------ queries
    def at(self, timestamp: float, record: bool = True) -> SyncedTilt:
        """The tilt at ``timestamp``, by interpolation where the history brackets it."""
        cfg = self.cfg
        if not self._estimates:
            tilt = SyncedTilt(0.0, 0.0, math.inf, NONE, math.inf, math.inf, math.inf, 0.0)
            if record:
                self.statistics.add(tilt, False)
            return tilt
        times = self._times
        newest, oldest = self._estimates[-1], self._estimates[0]
        newest_age = newest.timestamp - timestamp
        sparse = False
        if timestamp >= newest.timestamp:
            ahead = timestamp - newest.timestamp
            speed = float((newest.rate @ newest.rate) ** 0.5)
            if ahead <= cfg.max_extrapolation_s:
                orientation = quat.multiply(newest.quat,
                                            quat.from_rotation_vector(newest.rate * ahead))
                # The rate may change over the horizon; charge half of it.
                extra = 0.5 * speed * ahead
                method = EXTRAPOLATED if ahead > 0.0 else INTERPOLATED
            else:
                orientation, extra, method = newest.quat, speed * ahead, HELD
            roll, pitch = quat.tilt_from_quat(orientation)
            tilt = self._result(roll, pitch, newest.tilt_std, extra, method, -ahead, math.inf,
                                newest_age, speed)
        elif timestamp <= oldest.timestamp:
            behind = oldest.timestamp - timestamp
            speed = float((oldest.rate @ oldest.rate) ** 0.5)
            tilt = self._result(oldest.roll, oldest.pitch, oldest.tilt_std, speed * behind,
                                STALE, behind, math.inf, newest_age, speed)
        else:
            index = bisect.bisect_right(times, timestamp)
            before, after = self._estimates[index - 1], self._estimates[index]
            gap = after.timestamp - before.timestamp
            fraction = (timestamp - before.timestamp) / gap
            roll, pitch = quat.tilt_from_quat(quat.slerp(before.quat, after.quat, fraction))
            speed = float((after.rate @ after.rate) ** 0.5)
            sparse = gap > cfg.sparse_factor * cfg.expected_period_s
            # Across a wide bracket the motion in between is unknown; a chord is assumed.
            extra = 0.25 * speed * gap if sparse else 0.0
            offset = min(timestamp - before.timestamp, after.timestamp - timestamp)
            tilt = self._result(roll, pitch, max(before.tilt_std, after.tilt_std), extra,
                                INTERPOLATED, offset, gap, newest_age, speed)
        if record:
            self.statistics.add(tilt, sparse)
        return tilt

    def latest(self, timestamp: float, record: bool = True) -> SyncedTilt:
        """The newest estimate, whatever ``timestamp`` is: the naive pairing."""
        newest = self.newest
        if newest is None:
            return self.at(timestamp, record)
        speed = float((newest.rate @ newest.rate) ** 0.5)
        age = newest.timestamp - timestamp
        tilt = SyncedTilt(newest.roll, newest.pitch, newest.tilt_std, LATEST, age, math.inf, age,
                          speed)
        if record:
            self.statistics.add(tilt, False)
        return tilt

    def _result(self, roll, pitch, filter_std, extra, method, offset, gap, newest_age, speed
                ) -> SyncedTilt:
        clock = speed * self.cfg.timestamp_std_s
        std = math.sqrt(filter_std * filter_std + extra * extra + clock * clock)
        return SyncedTilt(float(roll), float(pitch), std, method, float(offset), float(gap),
                          float(newest_age), speed)
