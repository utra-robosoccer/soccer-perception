"""What perception hands to localization. See docs/localization.md.

Every observation is expressed in the robot's base frame (on the ground, +x forward, +y left)
and describes what was seen relative to the robot. None of these types carries a global pose
or a landmark identity: a T junction is reported as "a T junction about 2.1 m ahead and 0.4 m
to the left", never as "the T junction at the +x goal line". Resolving which landmark it was is
the localizer's job.

Today the oracle (observations/oracle.py) fills these structures from simulator geometry.
Later the same structures are filled from real images:

* `LinePointsObservation`: pixels of a field-line segmentation mask, sub-sampled and
  back-projected onto the ground plane with the camera extrinsics (or with stereo depth).
* `LandmarkObservation`: L / T / X junctions from line fitting on that mask; goal-post bases,
  penalty marks and the centre circle from a detector (for example YOLO boxes -> bottom-centre
  pixel -> ground plane), with the range refined by stereo disparity.

The standard deviations are the producer's own estimate of its error and are what the
localizer uses to weight each measurement.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from robosoccer_perception.geometry.landmarks import LandmarkType

BASE_FRAME_ID = "base_footprint"


@dataclasses.dataclass(frozen=True)
class LandmarkObservation:
    """One detected point feature: its class and where it is relative to the robot."""
    type: LandmarkType
    position: np.ndarray        # (2,) metres in the base frame, on the ground
    range_std: float            # metres, 1 sigma along the viewing direction
    bearing_std: float          # radians, 1 sigma across it
    confidence: float = 1.0     # detector score in [0, 1]; not used by the localizer yet

    @property
    def range(self) -> float:
        return float(math.hypot(self.position[0], self.position[1]))

    @property
    def bearing(self) -> float:
        """Angle from the robot's forward axis, positive to the left, radians."""
        return float(math.atan2(self.position[1], self.position[0]))


@dataclasses.dataclass(frozen=True)
class LinePointsObservation:
    """Points believed to lie on painted field lines. Which line is unknown."""
    points: np.ndarray          # (M, 2) metres in the base frame
    std: np.ndarray             # (M,) metres, 1 sigma position uncertainty of each point

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def empty(cls) -> "LinePointsObservation":
        return cls(np.zeros((0, 2)), np.zeros(0))


@dataclasses.dataclass(frozen=True)
class ObservationFrame:
    """Everything perception extracted from one camera frame."""
    timestamp: float
    landmarks: tuple[LandmarkObservation, ...] = ()
    line_points: LinePointsObservation = dataclasses.field(
        default_factory=LinePointsObservation.empty)
    frame_id: str = BASE_FRAME_ID

    @property
    def is_empty(self) -> bool:
        return not self.landmarks and len(self.line_points) == 0


@dataclasses.dataclass(frozen=True)
class OracleDebug:
    """Side information from the oracle for tests, evaluation and plots.

    This is deliberately a separate object from `ObservationFrame`: it contains landmark
    identities, which the localizer must never see.
    """
    landmark_ids: tuple[str | None, ...]    # per reported landmark; None = false positive
    num_true_line_points: int
    num_false_line_points: int
    frame_dropped: bool = False
