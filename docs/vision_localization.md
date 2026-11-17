# Vision localization (Phase 3)

Phase 1 localized on oracle observations made from simulator geometry. Phase 2 trained
networks that turn images into boxes and masks. Phase 3 joins them: the particle filter is now
driven by measurements extracted from **rendered camera images**, and the result is compared
with the oracle on identical runs.

```
RGB image ─► VisionPipeline (YOLO11n + U-Net) ─► VisionResult            Phase 2, unchanged
          ─► GeometricFeatureExtractor         ─► FeatureSet              docs/geometric_perception.md
          ─► build_observation_frame           ─► ObservationFrame        same type as the oracle's
odometry  ─► ParticleFilter.predict / update / estimate ─► PoseEstimate   Phase 1, unchanged
```

Everything is offline and ROS-free. Results: [phase3_validation.md](phase3_validation.md).

**Everything here is synthetic.** Images come from this repository's simulator and the models
were trained on images from the same simulator. Nothing in Phase 3 says how the pipeline does
on a real ZED Mini.

## Using it

```python
from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.ground_projection import EstimatedCameraCalibration
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.pipeline.vision_localization import VisionLocalizationPipeline

pipeline = VisionLocalizationPipeline.from_checkpoints(
    "outputs/detection/yolo11n/weights/best.pt", "outputs/segmentation/unet_small/best.pt",
    field="s_field")

# The robot's calibration: intrinsics and nominal mount for the image size it will pass in.
camera = StereoCameraConfig.load("zed_mini_sim").with_resolution(640, 360)
calibration = EstimatedCameraCalibration.from_camera_config(camera)

pipeline.init_gaussian([-3.0, 0.0, 0.0])            # or init_uniform() for a global start

out = pipeline.step(
    image,                                          # (H, W, 3) uint8 RGB
    timestamp,                                      # seconds
    OdometryDelta(dx, dy, dyaw),                    # motion since the last step, or None if lost
    calibration,
    calibration.pose(pan=head_pan, pitch=head_pitch))   # where the robot believes its camera is

out.pose                    # [x, y, yaw] in the field frame
out.estimate.covariance     # 3x3; also position_std, yaw_std, converged, symmetric_ambiguity, modes
out.observations            # the ObservationFrame the filter was given
out.features                # everything extracted, with flags and diagnostics
out.timing_ms               # detector, segmenter, geometry, filter, total
```

`step_input(VisionLocalizationInput(...))` is the same call with one argument;
`imu_roll=` / `imu_pitch=` apply an IMU-measured torso tilt to the camera pose.

## The boundary

What crosses into `step`, and nothing else:

| Input | Type |
|---|---|
| RGB frame | `numpy.ndarray`, `(H, W, 3)` uint8 |
| Timestamp | number |
| Odometry increment | `OdometryDelta` or `None` |
| Camera calibration | `EstimatedCameraCalibration` |
| Estimated camera-to-base transform | `EstimatedCameraPose` |
| Optional IMU roll / pitch | numbers |

The static field map is given once, at construction. A pose in the field frame enters only
through `init_gaussian`, `init_uniform` and `resolve_symmetry`, exactly as in Phase 1.

This is enforced three ways (`tests/test_ground_truth_isolation.py`):

1. **Structure.** `perception/`, `pipeline/` and `geometry/ground_projection.py` import
   neither the simulator, nor the dataset reader, nor the oracle, nor the benchmark. In a fresh
   interpreter where `import mujoco` fails, the pipeline still imports and localizes.
2. **Types.** `step` raises `TypeError` for anything but the types above. A `GroundTruth`, a
   `RigState`, a 4x4 matrix or a `SimulationCameraTruth` cannot be passed as a camera pose.
   The true camera and the estimated camera are different types with no attribute name in
   common.
3. **Behaviour.** During a real benchmark run with a hidden 2 deg mount error, every call into
   the vision arm is recorded: the image, the odometry delta, the calibration and the camera
   pose. The recorded camera pose is exactly the nominal one (25 deg down, 0.90 m up) with
   the commanded head pan, while the simulator's camera is 27 deg down, 0.94 m up and shaking.
   Replaying the recording into a fresh pipeline, with no simulator, reproduces the estimates
   **bit for bit**.

`step_with_predictions(vision_result, ...)` runs the same cycle from predictions that already
exist. The benchmark uses it to share one inference between the arms of a calibration sweep,
and for the "perfect vision" diagnostic, which passes rendered labels under that name. The
claimed vision arm always goes through `step`.

## Camera truth and camera estimate

The real mount is not designed yet, so the benchmark keeps two cameras apart:

| | Type | Built from | Used for |
|---|---|---|---|
| What the robot believes | `EstimatedCameraCalibration`, `EstimatedCameraPose` | `configs/cameras/*.yaml` and the commanded head angles | the pipeline |
| What is true | `SimulationCameraTruth` | the same, **plus** a hidden `MountError` and per-frame `HeadJitter` | rendering, scoring |

By default the true head shakes by 0.25 deg (1 sigma) in pitch and roll every frame, the same
value Phase 1's oracle used for its pitch jitter. The robot is not told. Mount errors (height,
pitch, roll, pan) are set per scenario and are not told either. Nothing produces an estimate
from the truth.

Sign conventions: `MountError` is truth minus belief; the sweep's `EstimateError` is estimate
minus truth. A mount pitch error of +2 deg and an estimate error of -2 deg are the same
mismatch (a test checks that both give about the same localization error).

## Integration with the particle filter

The filter, its configuration file and its API are **unchanged**; the oracle arm and the
vision arm use the same `configs/localization/particle_filter.yaml`. The vision pipeline fills
the Phase 1 `ObservationFrame`; where something did not fit the schema it was mapped onto it:

- a bearing-only goal post is a landmark with a range standard deviation of metres;
- a junction without a class, and a circle that failed a check, are left out.

`LandmarkObservation.confidence` is filled in (the filter still ignores it; confidence acts
through the standard deviations). On the first `step`, `odometry_delta=None` means "no motion
yet"; later it means a lost message and the particles diffuse.

## Field symmetry

A RoboCup field looks the same after a half turn. Two poses, `(x, y, yaw)` and
`(-x, -y, yaw + 180 deg)`, produce the same image (a test checks it on analytic views: fewer
than 0.1 % of pixels differ, from rounding at paint edges). Nothing extracted from an image can tell them apart, and Phase 3 does not try:

- from a global start the filter keeps both hypotheses, reports `symmetric_ambiguity`, and
  does not report `converged`;
- `resolve_symmetry(reference_xy)` and `init_uniform(x_range=...)` apply **external**
  information (which half the robot is in, from the game state or a referee command);
- with a wrong prior the filter is confidently on the mirrored pose, and only the evaluator
  can see it.

Scenarios v12 and v13b measure this. No landmark identity is invented from image geometry, and
the simulator's pose is never used to choose a side.

## The benchmark

`vision_benchmark/` is the only Phase 3 package that touches MuJoCo. One pass over a trajectory
renders each frame once and feeds several **arms**: independent localizers with the same
trajectory, the same odometry stream, the same initial guess and the same filter seed.

| Arm | Observations come from | Role |
|---|---|---|
| `vision` | the trained networks on the rendered image | **the Phase 3 pipeline** |
| `oracle` | Phase 1's oracle (simulator geometry, noise model) | idealised reference |
| `odometry` | nothing | dead-reckoning baseline |
| `perfect_vision` | rendered **labels** through the same geometric extraction | diagnostic: isolates network errors from geometry errors |

Arms are never mixed. `perfect_vision` uses ground-truth masks and boxes by design and is
reported under that name.

Reading the comparison:

- `oracle` vs `perfect_vision`: what real projection geometry and the extraction rules cost
  compared with an idealised noise model.
- `perfect_vision` vs `vision`: what the networks' errors cost.
- anything vs `odometry`: what looking at all is worth.

### Scenarios

`vision_benchmark/scenarios.py`, S field, each with a fixed seed stream. Unlike Phase 1, the
path is shifted (up to 0.20 m) and turned (up to 5 deg) for every seed. A deterministic
renderer always shows the same pixels at the same pose; without the shift every seed would
repeat the same perception errors.

| Id | What it tests | What is changed |
|---|---|---|
| `v01_stationary` | stationary robot | |
| `v02_rotation` | rotation in place | |
| `v03_straight_line` | straight-line walk | |
| `v04_rectangle` | rectangular lap | |
| `v05_curve` | circular lap | |
| `v06_partial_visibility` | little in view | camera commanded 50 deg down: sees 0.2 to 2.4 m ahead |
| `v07_noisy_odometry` | bad odometry | errors x4, 15 % of messages lost |
| `v08_missing_false_features` | missed and invented features | predictions damaged: 10 % frames lost, half the detections missed, invented posts, mask patches erased, false strokes |
| `v09_camera_pitch_error` | camera pitch error | camera really 2 deg lower than believed |
| `v10_camera_height_error` | camera height error | camera really 5 cm higher than believed |
| `v11_goal_post_occlusion` | post truncation and occlusion | in front of the goal, three robots at the posts |
| `v12a_symmetry_no_prior` | 180 deg symmetry | global start, no side information |
| `v12b_symmetry_own_half_prior` | the same with external side information | own-half prior |
| `v13a_kidnapped_same_side` | pose discontinuity | moved within its side |
| `v13b_kidnapped_across_field` | pose discontinuity | moved across the centre |
| `v14_strong_randomization` | appearance shift | `strong` scene preset, new look per seed, photometric effects per frame |

The damage in `v08` is applied to the networks' output (`vision_benchmark/corruption.py`). It
sees only the image and the predictions. It stress-tests the pipeline after the networks; it
is not a model of how networks fail on real images.

### Calibration sweep

`run_calibration_sweep` runs one scenario with a set of deliberately wrong camera
**estimates**, one parameter at a time (pitch, height, roll, yaw, focal length), all on the
same inference per frame. The simulator's camera is untouched. A second pitch curve repeats
the sweep with the robot believing its pitch is uncertain to 2 deg instead of 0.5 deg.

### Metrics

Localization metrics are Phase 1's (`evaluation/metrics.py`), unchanged: position and heading
RMSE, MAE and 95th percentile, convergence time, kidnapped-recovery time, wrong-half and
divergence rates. Angles use circular differences; positions are metres, angles degrees.
"Tracking" statistics cover the steps after convergence; the `_all` columns cover every step,
and are the ones to use for an arm that never converges, such as dead reckoning.

Geometric perception is scored per frame against the simulator
(`vision_benchmark/feature_metrics.py`): line-point error (distance to paint after moving the
points with the true pose), goal-post ground-point and bearing error, junction precision and
recall, centre-circle detection, the share of valid projections with rejection reasons, and
observation counts. Normalised errors (error divided by reported sigma) check the uncertainty
model.

Latency is measured inside `step`: detector, segmenter, geometry, filter, total. Rendering is
not part of the pipeline and is excluded.

## Commands

```bash
# One scenario, one seed: overlays, ground view, particle cloud, error plots
.venv/bin/python scripts/run_vision_localization.py --list
.venv/bin/python scripts/run_vision_localization.py v04_rectangle
.venv/bin/python scripts/run_vision_localization.py v13a_kidnapped_same_side --seed 3 --frames 6
.venv/bin/python scripts/run_vision_localization.py v04_rectangle --resolution 1280 720

# The benchmark (16 scenarios, 5 seeds, 4 arms) and the calibration sweep
.venv/bin/python scripts/benchmark_vision_localization.py
.venv/bin/python scripts/benchmark_vision_localization.py --seeds 2 v01 v04 --no-sweep
.venv/bin/python scripts/benchmark_vision_localization.py --sweep-only

# Feature extraction alone, scored against the simulator on random views
.venv/bin/python scripts/evaluate_features.py
.venv/bin/python scripts/evaluate_features.py --source perfect_vision
.venv/bin/python scripts/evaluate_features.py --appearance strong
.venv/bin/python scripts/evaluate_features.py --pitch-error 2
```

Settings (models, image size, seeds, sweep values) are in
`configs/evaluation/vision_localization.yaml`; extraction and noise parameters in
`configs/perception/`. Outputs go to `outputs/vision_localization/` (not tracked by Git).

## Code map

| Module | Role | Needs |
|---|---|---|
| `geometry/ground_projection.py` | calibration and pose estimates, pixel to ground, covariance | NumPy |
| `perception/line_features.py`, `line_fitting.py`, `intersections.py`, `goal_features.py` | extraction | NumPy, OpenCV |
| `perception/uncertainty.py`, `config.py`, `types.py` | noise model, configuration, feature types | NumPy |
| `perception/extractor.py` | `GeometricFeatureExtractor` | NumPy, OpenCV |
| `perception/observation_builder.py` | `FeatureSet` to `ObservationFrame` | NumPy |
| `pipeline/vision_localization.py` | `VisionLocalizationPipeline` | the above; models are passed in |
| `vision_benchmark/camera_truth.py`, `sim_camera.py` | the true camera, rendering, labels | MuJoCo |
| `vision_benchmark/scenarios.py`, `corruption.py`, `runner.py` | scenarios, damage model, arms | MuJoCo |
| `vision_benchmark/feature_metrics.py`, `feature_evaluation.py`, `benchmark.py`, `setup.py` | scoring, reports, sweep | MuJoCo |
| `visualization/vision_localization_plots.py` | overlays, ground view, comparison and sensitivity plots | Matplotlib, OpenCV |

The benchmark lives in its own package rather than in `evaluation/` because Phase 1's
evaluator must stay free of the vision stack (a Phase 2 test enforces it).

## Toward ROS 2 and the real camera

Nothing here is a ROS node, and nothing uses the ZED SDK. What a later integration needs:

- **A node around `step`.** Subscribe to the rectified left image and to odometry; take
  `T_base_camera` from TF (head joints, plus IMU roll and pitch) and build an
  `EstimatedCameraPose` from it; publish `map -> odom` from the `PoseEstimate`, as sketched in
  [coordinate_frames.md](coordinate_frames.md).
- **A calibrated camera profile.** `configs/cameras/<name>.yaml` with `approximate: false`,
  intrinsics of the **rectified** image at the resolution that is passed in.
- **A measured mount**, and honest values in `configs/perception/measurement_noise.yaml`. The
  sweep shows that pitch is the parameter to get right.
- **Timestamps.** Image and odometry must be aligned; the offline pipeline assumes they are.
- **Side information** from the game controller, for `resolve_symmetry`.
