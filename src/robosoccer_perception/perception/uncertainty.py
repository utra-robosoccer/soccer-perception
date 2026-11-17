"""The measurement-noise model: from projection covariance to the standard deviations that the
localizer is given. See docs/geometric_perception.md, "Uncertainty model".

Every image-derived measurement gets its uncertainty from the same three ingredients:

1. the PIXEL part: how well the feature's pixel is known, mapped to the ground. It grows
   roughly with ``range^2 / height`` along the viewing direction and with ``range`` across it;
2. the CAMERA part: the effect of the uncertain camera height, pitch, roll, yaw and focal
   length at that pixel. It is common to all features of a frame;
3. feature-specific terms: a floor, an inflation for features that reuse the same pixels as
   others, a penalty for low model confidence, and penalties for truncation or occlusion.

Approximations (deliberate):

* first order (Jacobians), Gaussian;
* the camera part is added to every feature as if it were independent between features. It is
  not: the localizer compensates with likelihood tempering (`line_point_weight` < 1) and an
  outlier floor, and extraction compensates by spacing the line points out;
* the localizer's schema takes one number per line point and (range, bearing) per landmark, so
  off-diagonal covariance is dropped after rotating into those directions.
"""
from __future__ import annotations

import math

import numpy as np

from robosoccer_perception.geometry.ground_projection import (
    EstimatedCameraCalibration, EstimatedCameraPose, ProjectionCovariance, ground_to_pixels,
    projection_covariance)
from robosoccer_perception.perception.config import MeasurementNoiseConfig
from robosoccer_perception.perception.types import MeasurementUncertainty


def confidence_factor(confidence, noise: MeasurementNoiseConfig):
    """Factor (>= 1) by which a standard deviation grows for a model confidence in [0, 1]."""
    clipped = np.clip(np.asarray(confidence, dtype=float), noise.confidence_floor, 1.0)
    return clipped ** (-noise.confidence_exponent)


def line_point_std(covariance: ProjectionCovariance, normals: np.ndarray, confidence: np.ndarray,
                   noise: MeasurementNoiseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(reported std, pixel part, camera part) of line points, in metres.

    What the localizer scores is a point's distance to the nearest line, so the relevant error
    is the one ACROSS the line. Where the line's direction is known (``normals`` finite) the
    covariance is projected onto the normal; elsewhere the direction-free RMS is used.
    """
    known = np.isfinite(normals).all(axis=1)
    safe = np.where(known[:, None], normals, [1.0, 0.0])
    pixel = np.where(known, covariance.std_along(safe, "pixel"), covariance.std_rms("pixel"))
    camera = np.where(known, covariance.std_along(safe, "camera"), covariance.std_rms("camera"))
    reported = np.sqrt(pixel ** 2 + camera ** 2 + noise.line_std_floor ** 2) \
        * confidence_factor(confidence, noise)
    return reported, pixel, camera


def point_uncertainty(position: np.ndarray, calibration: EstimatedCameraCalibration,
                      pose: EstimatedCameraPose, noise: MeasurementNoiseConfig, *,
                      pixel_std, range_std_floor: float, bearing_std_floor: float,
                      confidence: float = 1.0, inflation: float = 1.0,
                      extra_variance: float = 0.0, pixel=None) -> MeasurementUncertainty:
    """Uncertainty of one point feature at ``position`` (2,) in the base frame.

    ``pixel`` is the feature's pixel if it has one; otherwise the point is projected into the
    image with the robot's own calibration. ``extra_variance`` (metres^2, isotropic) adds a
    fit error that does not come from a single pixel.
    """
    position = np.asarray(position, dtype=float)
    if pixel is None:
        pixel = ground_to_pixels(position[None], calibration, pose)[0][0]
    covariance = projection_covariance(np.asarray(pixel, dtype=float)[None], calibration, pose,
                                       pixel_std)
    pixel_part = covariance.pixel[0] + np.eye(2) * extra_variance
    camera_part = covariance.camera[0]
    combined = ProjectionCovariance(pixel_part[None], camera_part[None])
    range_std, bearing_std = (float(v[0]) for v in combined.polar(position[None]))
    factor = float(confidence_factor(confidence, noise)) * inflation
    return MeasurementUncertainty(
        range_std=math.hypot(range_std, range_std_floor) * factor,
        bearing_std=math.hypot(bearing_std, bearing_std_floor) * factor,
        pixel_covariance=pixel_part, camera_covariance=camera_part)
