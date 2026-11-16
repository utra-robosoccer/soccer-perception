# Camera model

The intended camera is a Stereolabs ZED Mini. Until the physical unit is calibrated, the
simulator uses an **approximate, idealised** profile: `configs/cameras/zed_mini_sim.yaml`. Its
metadata says `approximate: true`, and nothing in it is a factory calibration value.

## The simulated profile

| Parameter | Value | Where it comes from |
|---|---|---|
| Baseline | 0.063 m between optical centres | public datasheet |
| Resolution | 1280 x 720 | the camera's HD720 mode |
| `fx = fy` | 640 px | chosen to give a 90 deg horizontal field of view |
| Field of view | 90.0 deg x 58.7 deg | follows from the above; datasheet says about 90 x 60 |
| `cx, cy` | 639.5, 359.5 | image centre |
| Distortion | none | idealisation |
| Frame rate | 30 Hz | configurable |
| Mount | 0.90 m high, 5 cm forward, pitched 25 deg down | **placeholder**, the robot's head is not designed yet |

Everything is configurable. `StereoCameraConfig.with_resolution(w, h)` rescales the intrinsics
for another resolution with the same field of view.

## Geometry

`geometry/camera.py` (pure NumPy) defines the model used by both the renderer and the oracle:

- `CameraIntrinsics`: pinhole projection `u = fx X/Z + cx`, `v = fy Y/Z + cy`.
- `RigState`: height, pitch, pan and offsets of the rig centre on the robot. It can change
  every frame (head motion).
- `StereoCameraModel`: `T_base_rig`, `T_rig_optical(side)`, `T_field_camera`, `T_left_right`,
  `project_field_points`, and `disparity(depth) = fx * baseline / Z`.

The two cameras are identical, with parallel optical axes, separated by exactly one baseline
along the left camera's +x. Frames are documented in [coordinate_frames.md](coordinate_frames.md).

## Rendering

`simulation/mujoco_scene.py` builds two MuJoCo cameras from the same model, with MuJoCo's
pixel-unit intrinsics (`focalpixel`, `principalpixel`), so the rendered projection is the
pinhole model above, not an approximation of it. `simulation/stereo_camera.py` renders both
images from the same simulation state (exactly synchronized) and returns a `StereoFrame`:

- `left`, `right`: RGB, uint8, `(height, width, 3)`
- `timestamp`, `frame_index`: camera clock, `frame_index / frame_rate`
- `T_field_left_optical`, `T_field_right_optical`: true camera poses
- `metadata`: intrinsics, baseline, field of view, frame rate, the `approximate` flag
- `ground_truth`: the full `GroundTruth` record (for labels and evaluation)

`save_stereo_frame` writes the pair as PNG plus a JSON sidecar.

## What is verified

- Stereo extrinsics, analytically and as read back from MuJoCo: parallel axes, 63 mm between
  optical centres, independent of where the rig points.
- Disparity: a point at depth Z shows `u_left - u_right = fx * 0.063 / Z` and equal `v`.
- Rendered images against the model: the ball's rendered centroid is within 1.5 px of its
  projected centre in both images, with the predicted disparity to within 1 px; the same holds
  with `fx != fy` and an off-centre principal point; pixels at projected line junctions are
  white and nearby turf is green.
- A top-down render of the S field overlaps the analytic paint mask with IoU > 0.95.

## What is not modelled

| Not modelled | Consequence for sim-to-real |
|---|---|
| Lens distortion | Real ZED Mini images must be rectified (the SDK does this) before these intrinsics mean anything. |
| Rolling shutter, motion blur | Walking shakes the head; real frames will be blurrier and skewed. |
| Exposure, white balance, sensor noise, vignetting | Detectors trained on these renders alone will not transfer. |
| Stereo miscalibration | Real rigs have small rotation between the cameras; here it is exactly zero. |
| Depth | No depth image is produced and the ZED depth algorithm is not imitated. Disparity is available only as geometry. |
| Turf, lighting, crowd, other robots | The scene is clean: procedural turf, two directional lights, an empty hall. |

The ZED SDK is not a dependency.

## When the real camera arrives

1. Calibrate it (or read the SDK's rectified calibration) and write a new
   `configs/cameras/<name>.yaml` with `approximate: false`.
2. If you need raw (unrectified) images in simulation, add a distortion model to
   `CameraIntrinsics.project` and its inverse to `ray`.
3. Measure the mount (`T_base_rig`) on the robot. A pitch error of 0.25 deg already moves a
   ground point 6 m away by about 17 cm (`observations/noise.py::apply_pitch_error`), so the
   mount calibration matters more than the intrinsics for ground-plane ranging.
