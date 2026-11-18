"""What an inertial measurement unit delivers. Pure data; shared by the simulated IMU and the
orientation estimator, and the only thing that passes between them."""
from __future__ import annotations

import dataclasses

import numpy as np

GRAVITY = 9.80665           # m/s^2, standard gravity


@dataclasses.dataclass(frozen=True)
class ImuSample:
    """One IMU measurement, in the SENSOR's own frame.

    ``timestamp`` is the time the sensor says the sample was taken, on the robot's clock.
    ``gyro`` is angular velocity in rad/s. ``accel`` is SPECIFIC FORCE in m/s^2: what an
    accelerometer reads, which is acceleration minus gravity, so a sensor at rest and level
    reads ``[0, 0, +9.81]``. Neither contains an orientation.
    """
    timestamp: float
    gyro: np.ndarray        # (3,)
    accel: np.ndarray       # (3,)

    def __post_init__(self) -> None:
        for name in ("gyro", "accel"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,):
                raise ValueError(f"{name} must have three components")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "timestamp", float(self.timestamp))
