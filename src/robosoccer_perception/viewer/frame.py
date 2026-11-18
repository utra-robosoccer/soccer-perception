"""One step of a run as the viewer sees it.

A `ViewerFrame` is produced once per simulation step, from ONE inference, and all four panels
are drawn from it. That is what keeps the camera image, the predictions, the features, the
pose estimate and the diagnostics of a panel set in step with each other: there is no second
source a panel could read from.

`pack` turns a frame into a flat dict of NumPy arrays (the image as JPEG, the mask as PNG) for
the scrubbing history and for recordings; `unpack` reverses it.

Ground truth appears here in exactly one place, the optional `truth` field, and is for display
only. Nothing in this module is read by the estimator. `fusion` (Phase 4) is the opposite: the
robot's own belief about its body tilt and camera pose, computed without ground truth.
"""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.inference.types import Detection

LANDMARK_TYPES = tuple(LandmarkType)
UNCLASSED = -1                          # junction whose shape could not be established
TIMING_KEYS = ("detector", "segmenter", "geometry", "filter", "total", "render")
JPEG_QUALITY = 90


def landmark_code(kind: LandmarkType | None) -> int:
    return UNCLASSED if kind is None else LANDMARK_TYPES.index(kind)


def landmark_label(code: int) -> str:
    """Short label for drawing: L, T, X, post, mark, centre; "?" for an unclassed junction."""
    if code == UNCLASSED:
        return "?"
    return {LandmarkType.GOAL_POST: "post", LandmarkType.PENALTY_MARK: "mark",
            LandmarkType.CENTER_MARK: "centre"}.get(LANDMARK_TYPES[code],
                                                    LANDMARK_TYPES[code].value)


@dataclasses.dataclass(frozen=True)
class TruthView:
    """What the simulator knows about this step. DISPLAY ONLY."""
    pose: np.ndarray                    # (3,) true [x, y, yaw] in the field frame
    position_error: float               # metres, estimate against truth
    yaw_error_deg: float
    mirror_position_error: float        # the same against the half-turn mirror of the truth
    mirror_yaw_error_deg: float
    kidnapped: bool = False             # the robot was moved at this step
    # ---- Phase 4 (NaN when the run has no moving body): degrees and metres
    body_roll_deg: float = float("nan")         # true tilt of the robot's body
    body_pitch_deg: float = float("nan")
    camera_roll_deg: float = float("nan")       # true orientation of the camera
    camera_pitch_deg: float = float("nan")
    camera_height: float = float("nan")


@dataclasses.dataclass(frozen=True)
class FusionView:
    """What the robot believes about its body and camera in this frame (Phase 4 runs). Every
    number here was computed WITHOUT ground truth; the true values are in `TruthView`."""
    source: str                         # nominal | imu | imu_latest | nominal_fallback
    body_roll_deg: float                # estimated body tilt (0 when the camera pose is fixed)
    body_pitch_deg: float
    tilt_std_deg: float                 # NaN when no IMU estimate was used
    camera_roll_deg: float              # estimated camera orientation used for projection
    camera_pitch_deg: float
    camera_height: float                # metres
    sync_method: str = ""               # interpolated | extrapolated | held | latest | none
    sync_offset_ms: float = float("nan")    # image timestamp to the nearest IMU estimate
    imu_samples: int = 0                # IMU samples delivered with this frame
    monitor_status: str = ""            # mount-calibration monitor; "" when it is not running
    monitor_bias_deg: tuple[float, float] = (float("nan"), float("nan"))    # roll, pitch
    monitor_correction_deg: tuple[float, float] = (0.0, 0.0)
    line_residual: float = float("nan")     # metres: field-line consistency residual


@dataclasses.dataclass(frozen=True)
class ViewerFrame:
    step: int
    timestamp: float                    # seconds of simulated time; shared by every panel
    # ---- camera and raw predictions
    image: np.ndarray                   # (H, W, 3) uint8 RGB, what the pipeline was given
    mask: np.ndarray                    # (H, W) uint8 semantic classes from the segmenter
    detections: tuple[Detection, ...]
    # ---- extracted features: pixels in the image, metres in the robot's base frame
    line_pixels: np.ndarray             # (N, 2)
    line_points: np.ndarray             # (N, 2)
    line_std: np.ndarray                # (N,)
    segment_pixels: np.ndarray          # (S, 2, 2) fitted line pieces, start and end
    segments: np.ndarray                # (S, 2, 2)
    junction_pixels: np.ndarray         # (J, 2)
    junctions: np.ndarray               # (J, 2)
    junction_codes: np.ndarray          # (J,) index into LANDMARK_TYPES or UNCLASSED
    circle_centers: np.ndarray          # (C, 2)
    circle_center_pixels: np.ndarray    # (C, 2)
    circle_valid: np.ndarray            # (C,) bool: accepted, or only a candidate
    circle_radius: float
    circle_outlines: np.ndarray         # (C, K, 2) pixels of the fitted circle; NaN where unseen
    post_pixels: np.ndarray             # (P, 2) foot of each goal post
    posts: np.ndarray                   # (P, 2)
    post_range_valid: np.ndarray        # (P,) bool: False = bearing only
    # ---- what the localizer was given (after selection), base frame
    landmarks: np.ndarray               # (L, 2)
    landmark_codes: np.ndarray          # (L,)
    observed_line_points: int
    # ---- what the localizer answered
    pose: np.ndarray                    # (3,) estimated [x, y, yaw]
    covariance: np.ndarray              # (3, 3)
    position_std: float
    yaw_std_deg: float
    confidence: float
    mode_weight: float
    converged: bool
    ambiguous: bool                     # a second mode sits at the mirrored pose
    modes: np.ndarray                   # (M, 4) x, y, yaw, weight; heaviest first
    ess_fraction: float                 # effective sample size / number of particles
    resampled: bool
    injected: int
    particles: np.ndarray               # (K, 3) float32: an evenly spaced subset
    particle_weights: np.ndarray        # (K,) float32, normalised over the subset
    num_particles: int
    odometry_lost: bool
    # ---- timing
    timing_ms: dict[str, float]         # TIMING_KEYS; ``render`` is the simulator, not the pipeline
    sim_fps: float                      # steps per wall-clock second when this frame was made
    # ---- display only
    truth: TruthView | None = None
    # ---- Phase 4: the robot's camera-pose estimate and a third-person picture of the robot
    fusion: FusionView | None = None
    robot_view: np.ndarray | None = None    # (h, w, 3) uint8 RGB from the chase camera

    @property
    def inference_ms(self) -> float:
        return self.timing_ms.get("detector", 0.0) + self.timing_ms.get("segmenter", 0.0)

    def without_truth(self) -> "ViewerFrame":
        return dataclasses.replace(self, truth=None)


# ------------------------------------------------------------------------------- packing
_ARRAYS = ("line_pixels", "line_points", "line_std", "segment_pixels", "segments",
           "junction_pixels", "junctions", "junction_codes", "circle_centers",
           "circle_center_pixels", "circle_valid",
           "circle_outlines", "post_pixels", "posts", "post_range_valid", "landmarks",
           "landmark_codes", "pose", "covariance", "modes", "particles", "particle_weights")
_SCALARS = ("step", "timestamp", "circle_radius", "observed_line_points", "position_std",
            "yaw_std_deg", "confidence", "mode_weight", "converged", "ambiguous",
            "ess_fraction", "resampled", "injected", "num_particles", "odometry_lost", "sim_fps")
_SCALAR_TYPES = {"step": int, "observed_line_points": int, "injected": int,
                 "num_particles": int, "converged": bool, "ambiguous": bool, "resampled": bool,
                 "odometry_lost": bool}


def pack(frame: ViewerFrame) -> dict[str, np.ndarray]:
    """A frame as a flat dict of arrays, small enough to keep hundreds of them in memory."""
    ok_image, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    ok_mask, png = cv2.imencode(".png", frame.mask)
    if not (ok_image and ok_mask):
        raise RuntimeError("could not encode the frame")
    out: dict[str, np.ndarray] = {"image_jpg": jpeg.ravel(), "mask_png": png.ravel()}
    for name in _ARRAYS:
        out[name] = np.asarray(getattr(frame, name))
    out["scalars"] = np.array([float(getattr(frame, name)) for name in _SCALARS])
    out["timing"] = np.array([float(frame.timing_ms.get(key, 0.0)) for key in TIMING_KEYS])
    out["detections"] = np.array(
        [[d.class_id, d.confidence, *d.bbox_xyxy] for d in frame.detections]).reshape(-1, 6)
    if frame.truth is not None:
        t = frame.truth
        out["truth"] = np.array([*t.pose, t.position_error, t.yaw_error_deg,
                                 t.mirror_position_error, t.mirror_yaw_error_deg,
                                 float(t.kidnapped), t.body_roll_deg, t.body_pitch_deg,
                                 t.camera_roll_deg, t.camera_pitch_deg, t.camera_height])
    if frame.fusion is not None:
        f = frame.fusion
        out["fusion"] = np.array([f.body_roll_deg, f.body_pitch_deg, f.tilt_std_deg,
                                  f.camera_roll_deg, f.camera_pitch_deg, f.camera_height,
                                  f.sync_offset_ms, float(f.imu_samples), *f.monitor_bias_deg,
                                  *f.monitor_correction_deg, f.line_residual])
        out["fusion_text"] = np.array("|".join([f.source, f.sync_method, f.monitor_status]))
    if frame.robot_view is not None:
        ok_view, view = cv2.imencode(".jpg", cv2.cvtColor(frame.robot_view, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok_view:
            raise RuntimeError("could not encode the robot view")
        out["robot_view_jpg"] = view.ravel()
    return out


def unpack(data) -> ViewerFrame:
    """The reverse of `pack`. ``data`` is any mapping from name to array."""
    image = cv2.cvtColor(cv2.imdecode(np.asarray(data["image_jpg"]), cv2.IMREAD_COLOR),
                         cv2.COLOR_BGR2RGB)
    mask = cv2.imdecode(np.asarray(data["mask_png"]), cv2.IMREAD_UNCHANGED)
    values = {name: np.asarray(data[name]) for name in _ARRAYS}
    for name in ("circle_valid", "post_range_valid"):
        values[name] = values[name].astype(bool)
    for name, value in zip(_SCALARS, np.asarray(data["scalars"])):
        values[name] = _SCALAR_TYPES.get(name, float)(value)
    timing = dict(zip(TIMING_KEYS, (float(v) for v in np.asarray(data["timing"]))))
    detections = tuple(Detection(int(row[0]), float(row[1]), tuple(float(v) for v in row[2:6]))
                       for row in np.asarray(data["detections"]).reshape(-1, 6))
    truth = None
    if "truth" in data:
        t = np.asarray(data["truth"])
        truth = TruthView(t[0:3].copy(), float(t[3]), float(t[4]), float(t[5]), float(t[6]),
                          bool(t[7]), *(float(v) for v in t[8:13]))    # older files stop at 8
    fusion = None
    if "fusion" in data:
        f = [float(v) for v in np.asarray(data["fusion"])]
        source, method, status = str(np.asarray(data["fusion_text"])).split("|")
        fusion = FusionView(source, f[0], f[1], f[2], f[3], f[4], f[5], method, f[6], int(f[7]),
                            status, (f[8], f[9]), (f[10], f[11]), f[12])
    robot_view = None
    if "robot_view_jpg" in data:
        robot_view = cv2.cvtColor(cv2.imdecode(np.asarray(data["robot_view_jpg"]),
                                               cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return ViewerFrame(image=image, mask=mask, detections=detections, timing_ms=timing,
                       truth=truth, fusion=fusion, robot_view=robot_view, **values)


# ------------------------------------------------------------------------------- timeline
class Timeline:
    """The per-step numbers the diagnostics panel plots, for a whole run.

    Preallocated for the run's length and filled as frames arrive; steps not reached yet are
    NaN and are simply not drawn.
    """
    SERIES = ("position_error", "yaw_error_deg", "confidence", "ess_fraction", "position_std",
              "yaw_std_deg", "inference_ms", "total_ms", "sim_fps", "ambiguous", "converged",
              "kidnapped", "est_x", "est_y", "true_x", "true_y",
              # Phase 4: camera orientation, estimated (no truth) and true (display only)
              "est_camera_pitch", "est_camera_roll", "tilt_std", "true_camera_pitch",
              "true_camera_roll")
    #: series a recording made before Phase 4 does not have; they load as "never recorded"
    OPTIONAL = ("est_camera_pitch", "est_camera_roll", "tilt_std", "true_camera_pitch",
                "true_camera_roll")

    def __init__(self, steps: int, dt: float):
        self.steps, self.dt = int(steps), float(dt)
        self.values = {name: np.full(self.steps, np.nan) for name in self.SERIES}
        self.last = -1                              # highest step recorded so far

    @property
    def time(self) -> np.ndarray:
        return self.dt * np.arange(self.steps)

    def record(self, frame: ViewerFrame) -> None:
        step = frame.step
        if not 0 <= step < self.steps:
            return
        v = self.values
        v["confidence"][step] = frame.confidence
        v["ess_fraction"][step] = frame.ess_fraction
        v["position_std"][step] = frame.position_std
        v["yaw_std_deg"][step] = frame.yaw_std_deg
        v["inference_ms"][step] = frame.inference_ms
        v["total_ms"][step] = frame.timing_ms.get("total", np.nan)
        v["sim_fps"][step] = frame.sim_fps
        v["ambiguous"][step] = float(frame.ambiguous)
        v["converged"][step] = float(frame.converged)
        v["est_x"][step], v["est_y"][step] = frame.pose[0], frame.pose[1]
        if frame.truth is not None:
            v["position_error"][step] = frame.truth.position_error
            v["yaw_error_deg"][step] = frame.truth.yaw_error_deg
            v["kidnapped"][step] = float(frame.truth.kidnapped)
            v["true_x"][step], v["true_y"][step] = frame.truth.pose[0], frame.truth.pose[1]
            v["true_camera_pitch"][step] = frame.truth.camera_pitch_deg
            v["true_camera_roll"][step] = frame.truth.camera_roll_deg
        if frame.fusion is not None:
            v["est_camera_pitch"][step] = frame.fusion.camera_pitch_deg
            v["est_camera_roll"][step] = frame.fusion.camera_roll_deg
            v["tilt_std"][step] = frame.fusion.tilt_std_deg
        self.last = max(self.last, step)

    def clear(self) -> None:
        for series in self.values.values():
            series[:] = np.nan
        self.last = -1

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {f"timeline__{name}": series for name, series in self.values.items()}

    @classmethod
    def from_arrays(cls, data, dt: float) -> "Timeline":
        steps = len(np.asarray(data[f"timeline__{cls.SERIES[0]}"]))
        timeline = cls(steps, dt)
        for name in cls.SERIES:
            key = f"timeline__{name}"
            if name in cls.OPTIONAL and key not in data:
                continue                            # an older recording: stays NaN
            timeline.values[name] = np.asarray(data[key], dtype=float).copy()
        recorded = np.flatnonzero(np.isfinite(timeline.values["confidence"]))
        timeline.last = int(recorded[-1]) if len(recorded) else -1
        return timeline
