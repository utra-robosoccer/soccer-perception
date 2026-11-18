# soccer-perception

Perception and localization research for the UTRA RoboSoccer humanoid.

The repository was built in four phases:

- **Phase 1, localization on oracle observations.** Lets the team develop and **measure**
  localization before any neural perception model exists and before the robot can walk.
- **Phase 2, vision.** Generates labelled synthetic camera images from the same simulator,
  trains a small object detector (ball, goal posts, robots) and a small segmentation network
  (field, field lines), evaluates both, and runs them together on camera frames.
- **Phase 3, vision localization.** Joins the two: geometric features are extracted from the
  networks' predictions on rendered camera images, turned into the observations the particle
  filter already consumes, and the result is benchmarked against the oracle on identical runs.

- **Phase 4, the humanoid and a moving camera.** Puts UTRA's actual `humanoid_prototype_1`
  body under a configurable virtual camera mount, moves it the way a standing, swaying, walking
  or pushed humanoid moves, simulates an IMU on it, and estimates the camera's orientation for
  every image so that the Phase 3 pipeline keeps working when the torso does not hold still.

All four are **simulation only**. No real camera image, no measured robot motion and no real
IMU data has been used anywhere.

Phase 1 contains:

- a configurable RoboCup field (S, M and L sizes of the 2026 Humanoid League rulebook);
- a MuJoCo scene of that field with a movable virtual ZED Mini stereo camera;
- an **oracle** that turns simulator ground truth into noisy, anonymous landmark and
  field-line observations, standing in for detectors that are not trained yet;
- noisy synthetic odometry;
- a 3-DOF (x, y, yaw) particle-filter localizer;
- deterministic benchmark scenarios with metrics, plots and JSON/CSV results.

Phase 2 adds:

- other robots (humanoid placeholders), ball looks and a backdrop in the MuJoCo scene;
- a deterministic dataset generator with labels read from the renderer: YOLO boxes around the
  visible part of each object, and pixel-aligned field / line masks;
- domain randomization presets (`none`, `mild`, `strong`) and leakage-free splits;
- a YOLO11n detector and a small U-Net segmenter, with training, evaluation and export scripts;
- an offline inference pipeline that runs both on an image and saves what it sees.

Phase 3 adds:

- image-to-ground projection from the robot's **estimated** camera pose, with validity checks
  and uncertainty propagation;
- geometric feature extraction from predicted masks and boxes: field-line points, goal posts
  (position, or bearing only when the foot cannot be trusted), L / T / X junctions and the
  centre circle;
- a bridge to the Phase 1 `ObservationFrame`, and `VisionLocalizationPipeline.step(image, ...)`:
  camera frame in, pose estimate out, with no access to the simulator;
- a benchmark that runs the vision pipeline, the oracle, dead reckoning and a "perfect vision"
  diagnostic side by side, with camera-calibration error, occlusion, damaged predictions,
  appearance shift, field symmetry and kidnapping scenarios, and a calibration sensitivity
  sweep.

Phase 4 adds:

- UTRA's `humanoid_prototype_1`, imported from `soccer-software` as a kinematic, visual-only
  body (every link mesh, the joint tree, the IMU site), with the placeholder platform kept as a
  fallback;
- a **configurable virtual camera mount**: the head is not designed, so the camera's place on
  the body is a preset file, its height is derived from the body's geometry, and every preset
  is marked experimental;
- synthetic body motion (standing, sway, walking-like, turning, abrupt disturbances) and a
  simulated IMU derived from that motion, with bias, drift, noise, latency, lost samples and
  clock offsets;
- a lightweight roll / pitch filter with an uncertainty, interpolation of orientation to each
  image's timestamp, and a per-image camera pose for the unchanged Phase 3 pipeline;
- uncertainty-aware projection that treats a camera-orientation error as shared by the whole
  frame, and a monitor that finds a persistent mount pitch / roll bias in the field lines and
  applies a bounded correction;
- a benchmark of fixed-camera, oracle-orientation, IMU and uncertainty-aware localization on
  identical images, with calibration and timing sweeps;
- the viewer extended to show the humanoid, the mount and the estimated against the true
  camera orientation.

It is a standalone Python package (`robosoccer_perception`). It does not use ROS and does not
depend on `robosoccer-rl` or `soccer-software` at run time: the humanoid's meshes are imported
once into `assets/`.

Not here: the ZED SDK, ROS 2 nodes or messages, SLAM, humanoid dynamics or locomotion (the
humanoid is posed, not simulated, and its stepping is an animation), a head design, stereo
depth, and any validation on real camera footage or real IMU data.

## Setup

Requires Linux and Python 3.11 or newer (developed on Ubuntu with Python 3.12).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[all]"
```

`[all]` installs MuJoCo 3.4.0 (the version the other RoboSoccer repositories pin), Matplotlib
and pytest. The core needs only NumPy and PyYAML: `pip install -e .` is enough to use the
geometry, the oracle, the localizer and the benchmarks without plots.

For Phase 2 (datasets, training, inference) also install the `vision` extra. It pulls in
PyTorch with CUDA libraries, Ultralytics and OpenCV: several GB of downloads.

```bash
.venv/bin/python -m pip install -e ".[all,vision]"
.venv/bin/python scripts/train_detector.py --download yolo11n.pt    # pretrained weights, 5 MB
```

The interactive viewer needs one more extra, `viewer` (PySide6), and on Ubuntu the system
package `libxcb-cursor0`; see [docs/viewer.md](docs/viewer.md).

```bash
.venv/bin/python -m pip install -e ".[all,vision,viewer]"
sudo apt install libxcb-cursor0
```

**The humanoid model** used by Phase 4 is already in the repository
(`assets/robots/humanoid_prototype_1/`, 1.9 MB, imported from soccer-software commit
`bb4a3bb`). Nothing has to be done to use it. To refresh it after the official model changes:

```bash
.venv/bin/python scripts/import_humanoid_model.py \
    --source <soccer-software>/src/humanoid_description/robots/humanoid_prototype_1
```

The source is only read. See [docs/humanoid_integration.md](docs/humanoid_integration.md).

Phase 1 does not need it, and its tests run without it (the vision tests are skipped).
Ultralytics and the YOLO weights are AGPL-3.0; see
[docs/object_detection.md](docs/object_detection.md#licensing).

If your shell has a ROS workspace sourced, prefix commands with `env -u PYTHONPATH` so that
ROS's Python packages do not shadow the virtual environment.

**Rendering needs an OpenGL context.** MuJoCo's `glfw` backend is used by default and needs a
display (`DISPLAY` set). On a headless machine try `MUJOCO_GL=egl`. Only image rendering needs
this; localization, benchmarks and most tests do not.

## Run

All commands are run from the repository root. Outputs go to `outputs/`, which is not tracked
by Git.

### Tests

```bash
.venv/bin/python -m pytest                    # everything: 585 tests, about 5.5 minutes
.venv/bin/python -m pytest -m "not slow"      # skips full localization runs and training smoke tests
.venv/bin/python -m pytest -m "not vision"    # everything that needs no PyTorch
```

Tests that render images are skipped automatically when no OpenGL context is available, and
tests that need PyTorch or Ultralytics when the `vision` extra is not installed. No test uses
the network or needs downloaded weights. The Phase 3 tests that run the **trained** models
(`tests/test_vision_localization_models.py`) are skipped until the checkpoints exist; all
other Phase 3 tests use analytically painted views or a colour-threshold stand-in. The
viewer's window tests run without a screen and are skipped when PySide6 is not installed. The
Phase 4 tests that need the humanoid are skipped if `assets/robots/humanoid_prototype_1/` is
missing; the mount, body-motion, IMU and orientation tests need neither MuJoCo nor a display.

### Render the field and a stereo pair

```bash
.venv/bin/python scripts/render_field.py --overlay
.venv/bin/python scripts/render_field.py --field m_field --pose -4 -2 30
.venv/bin/python scripts/render_field.py --pose 1 0 180 --height 1.2 --pitch 35 --pan 20
```

Writes to `outputs/render/`:

| File | Content |
|---|---|
| `s_field_overview.png` | top-down render of the whole field |
| `s_field_stereo_left.png`, `s_field_stereo_right.png` | the stereo pair |
| `s_field_stereo.json` | timestamp, camera metadata and ground-truth poses of that frame |
| `s_field_left_overlay.png` | left image with the map's landmarks projected onto it (`--overlay`) |
| `s_field_static_ground_truth.json` | field dimensions, goal posts, landmarks, line geometry |

`--pose X Y YAW_DEG` places the camera platform anywhere; `--height`, `--pitch`, `--pan` and
`--ball X Y` change the rig and the ball.

### Run one localization scenario

```bash
.venv/bin/python scripts/run_localization.py --list
.venv/bin/python scripts/run_localization.py s04_rectangle
.venv/bin/python scripts/run_localization.py s10a_kidnapped_same_side --seed 3
.venv/bin/python scripts/run_localization.py s04_rectangle --mujoco --frames 5
```

Writes to `outputs/localization/`: the particle-cloud figure (`*_particles.png`), the error
plot (`*_errors.png`), the true and estimated paths (`*_trajectory.png`), the per-step time
series (`*.csv`) and the metrics (`*_metrics.json`). With `--mujoco` the platform is driven
through the MuJoCo scene and ground truth is read back from it; `--frames N` also saves N
stereo pairs along the run.

### Run the benchmark

```bash
.venv/bin/python scripts/benchmark.py                  # all scenarios, seeds 0-9, about 7 minutes
.venv/bin/python scripts/benchmark.py --seeds 3        # quicker
.venv/bin/python scripts/benchmark.py s01 s10          # only scenarios whose id starts with these
```

Writes to `outputs/benchmark/`:

| File | Content |
|---|---|
| `results.json` | everything: configuration, per-scenario aggregates, per-run metrics |
| `summary.md` | human-readable table and scenario notes |
| `summary.csv`, `runs.csv` | one row per scenario / per run |
| `timeseries/<scenario>_seed<k>.csv` | per-step truth, estimate, errors and filter state |
| `plots/<scenario>_{particles,errors,trajectory}.png` | figures for seed 0 |

Current results and their interpretation are in [docs/validation.md](docs/validation.md).

### Vision: dataset, training, evaluation, inference

The whole Phase 2 loop, in order. Each step is described in the document named next to it.

```bash
# 1. Generate a labelled dataset and look at the labels          docs/synthetic_data.md
.venv/bin/python scripts/generate_dataset.py baseline            # 2400 frames, about 6 min
.venv/bin/python scripts/inspect_dataset.py outputs/datasets/baseline

# 2. Train and evaluate the detector                             docs/object_detection.md
.venv/bin/python scripts/train_detector.py --download yolo11n.pt # once
.venv/bin/python scripts/train_detector.py yolo11n               # about 15 min on an RTX 2050
.venv/bin/python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt

# 3. Train and evaluate the segmenter                            docs/segmentation.md
.venv/bin/python scripts/train_segmentation.py unet_small        # about 35 min on an RTX 2050
.venv/bin/python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt

# 4. Run both on simulated ZED Mini frames (1280x720)            docs/phase3_interfaces.md
.venv/bin/python scripts/run_vision.py --render 8 --ground
.venv/bin/python scripts/run_vision.py --images path/to/image.png
```

To check that everything runs without waiting, use the smoke presets (a minute in total; the
resulting models are not usable):

```bash
.venv/bin/python scripts/generate_dataset.py smoke
.venv/bin/python scripts/train_detector.py yolo11n_smoke
.venv/bin/python scripts/train_segmentation.py unet_smoke
```

| Output | Where |
|---|---|
| Datasets | `outputs/datasets/<name>/` (`images/`, `labels/`, `masks/`, `manifest.jsonl`, `previews/`) |
| Detector runs | `outputs/detection/<name>/` (`weights/best.pt`, `results.csv`, `eval_<dataset>_<split>/`) |
| Segmenter runs | `outputs/segmentation/<name>/` (`best.pt`, `history.csv`, `eval_<dataset>_<split>/`) |
| Pipeline results | `outputs/vision/` (`*_overlay.jpg`, `*.json`, `*_mask.png`, `summary.json`) |

Training one model at a time is deliberate: on a 16 GB laptop, two trainings side by side can
exhaust the RAM. If a run is killed, lower `workers` or `batch` in its configuration.

Results and their limits are in [docs/vision_validation.md](docs/vision_validation.md).

### Vision localization (Phase 3)

Needs the trained detector and segmenter from the steps above.

```bash
# One scenario, one seed: what the camera saw, what was extracted, where the filter thinks it is
.venv/bin/python scripts/run_vision_localization.py --list
.venv/bin/python scripts/run_vision_localization.py v04_rectangle
.venv/bin/python scripts/run_vision_localization.py v13a_kidnapped_same_side --seed 3 --frames 6
.venv/bin/python scripts/run_vision_localization.py v04_rectangle --resolution 1280 720

# The benchmark: 16 scenarios x 5 seeds x 4 arms, then the calibration sweep (about 1 h 50 min)
.venv/bin/python scripts/benchmark_vision_localization.py
.venv/bin/python scripts/benchmark_vision_localization.py --seeds 2 v01 v04 --no-sweep   # quick
.venv/bin/python scripts/benchmark_vision_localization.py --sweep-only

# Feature extraction alone, scored against the simulator on random views
.venv/bin/python scripts/evaluate_features.py
.venv/bin/python scripts/evaluate_features.py --source perfect_vision   # rendered labels instead
.venv/bin/python scripts/evaluate_features.py --appearance strong       # strong domain randomization
.venv/bin/python scripts/evaluate_features.py --pitch-error 2           # camera really 2 deg lower
```

Every benchmark run compares four **arms** on the same trajectory, odometry and initial guess:
`vision` (the trained networks on the rendered image), `oracle` (Phase 1's observations),
`odometry` (dead reckoning) and `perfect_vision` (rendered labels through the same geometry, a
diagnostic).

| Output | Where |
|---|---|
| Benchmark | `outputs/vision_localization/benchmark/` (`summary.md`, `results.json`, `summary.csv`, `runs.csv`, `features.csv`, `timeseries/`, `plots/`, `frames/`) |
| Calibration sweep | `outputs/vision_localization/sweep/` (`sweep.json`, `sweep.csv`, `sensitivity.png`) |
| Single runs | `outputs/vision_localization/runs/<scenario>_seed<k>/` (`arms.png`, `vision_trajectory.png`, `vision_particles.png`, `<arm>.csv`, `metrics.json`, `frames/`) |
| Feature scores | `outputs/vision_localization/features/<name>.json` |

Per kept frame, `frames/` holds the rendered image (`*_rgb.jpg`), the YOLO boxes and U-Net mask
(`*_predictions.jpg`), the extracted line points, junctions and goal-post feet
(`*_features.jpg`), the same on the ground in the robot frame (`*_ground.png`), and the
particle cloud on the field (`*_particles.png`).

Settings are in `configs/evaluation/vision_localization.yaml` (models, image size, seeds,
sweep values) and `configs/perception/` (extraction and measurement-noise parameters).
Results and their limits are in [docs/phase3_validation.md](docs/phase3_validation.md).

### The humanoid, a moving camera and the IMU (Phase 4)

```bash
# Pictures of the robot on the field with each virtual camera mount, and what its camera sees
.venv/bin/python scripts/render_humanoid.py
.venv/bin/python scripts/render_humanoid.py head_mid --pitch 6 --roll -3     # a tilted body

# The benchmark: 18 scenarios x 3 seeds x 6 arms, then the calibration and timing sweeps
# (about 75 min on the RTX 2050 laptop; run it alone, its latency numbers are a result)
.venv/bin/python scripts/benchmark_humanoid_fusion.py
.venv/bin/python scripts/benchmark_humanoid_fusion.py h05 h10 --seeds 1 --no-sweeps   # a quick look
```

Every arm localizes from the same rendered images, IMU samples and odometry and differs only in
its camera-pose estimate: `fixed` (Phase 3: a level body in every frame),
`oracle_orientation` (the true camera orientation; an upper bound that uses ground truth),
`imu`, `imu_uncertainty`, `imu_latest` (no timestamp interpolation) and `imu_selfcal` (with the
mount-bias correction).

| Output | Where |
|---|---|
| Benchmark | `outputs/humanoid_fusion/benchmark/` (`summary.md`, `results.json`, `summary.csv`, `plots/overview.png`, `plots/camera_height.png`, `plots/run_<scenario>.png`) |
| Sweeps | `outputs/humanoid_fusion/sweeps/` (`calibration_sweep.json`, `calibration_sweep.csv`, `calibration_sensitivity.png`, `timing_sweep.json`, `timing_sensitivity.png`) |
| Mount pictures | `outputs/humanoid/` (`mount_<robot>_<preset>.png`, `mounts_<robot>.png`, `mounts_<robot>.json`) |

Settings: `configs/evaluation/humanoid_fusion.yaml`, `configs/mounts/`, `configs/motion/`,
`configs/imu/`, `configs/robots/`. What was integrated:
[docs/humanoid_integration.md](docs/humanoid_integration.md). The mount and what changes when
the head exists: [docs/camera_mount.md](docs/camera_mount.md). Methods:
[docs/humanoid_fusion.md](docs/humanoid_fusion.md). Results and limits:
[docs/phase4_validation.md](docs/phase4_validation.md).

### Watch the pipeline work (interactive viewer)

One window with four synchronized panels: the camera image with the detector's boxes, the
segmentation with the extracted features, the field from above with the particle cloud, the
estimate and the true pose, and plots of error, confidence and latency. It runs any benchmark
scenario live, pauses, steps, scrubs back through what it has shown, and replays recordings.
For the Phase 4 scenarios it shows the humanoid itself in a third-person inset, and the
estimated against the true camera orientation.

```bash
.venv/bin/python scripts/visualize_pipeline.py                      # the humanoid walking a rectangle
.venv/bin/python scripts/visualize_pipeline.py h02 --fusion fixed   # sway, with the Phase 3 camera model
.venv/bin/python scripts/visualize_pipeline.py h10 --fusion imu_selfcal --autoplay
.venv/bin/python scripts/visualize_pipeline.py h05 --robot placeholder --mount legacy_rig
.venv/bin/python scripts/visualize_pipeline.py v13a --autoplay      # a Phase 3 scenario, by prefix
.venv/bin/python scripts/visualize_pipeline.py --replay outputs/viewer/recordings/<file>.npz
.venv/bin/python scripts/visualize_pipeline.py h05 --headless --record --video   # no window
```

Keys: Space run / pause, Right / Left step, R reset, S screenshot, G ground truth. The second
control bar selects the robot body, the camera mount preset, the body motion and the
camera-pose estimator. Recordings, videos and screenshots go to `outputs/viewer/`. Ground truth
is displayed, never given to the estimator. Controls, panels and limits:
[docs/viewer.md](docs/viewer.md).

### Use it from Python

```python
import numpy as np
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.localization.motion_model import OdometryDelta
from robosoccer_perception.localization.particle_filter import ParticleFilter

field = FieldGeometry.load("s_field")
localizer = ParticleFilter(field, seed=0)
localizer.init_gaussian([-2.0, -1.0, 0.3])           # or localizer.init_uniform()

# each cycle: relative odometry, then whatever perception saw in the robot frame
estimate, info = localizer.step(OdometryDelta(dx=0.03, dy=0.0, dyaw=0.01), observation_frame)
print(estimate.pose, estimate.confidence, estimate.converged, estimate.symmetric_ambiguity)
```

`observation_frame` is an `ObservationFrame` (`observations/types.py`). In simulation it comes
from `OracleObservationGenerator`; `evaluation/benchmark.py::run_scenario` shows the full loop.

The vision pipeline, on any RGB image:

```python
from robosoccer_perception.inference.pipeline import VisionPipeline

pipeline = VisionPipeline.from_checkpoints(
    "outputs/detection/yolo11n/weights/best.pt", "outputs/segmentation/unet_small/best.pt")
result = pipeline.run(image_rgb)                  # (H, W, 3) uint8
print(result.ball, result.goal_posts, result.robots, result.timing_ms)
line_mask = result.segmentation.line_mask         # (H, W) bool
```

Localization from camera frames (Phase 3):

```python
from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.ground_projection import EstimatedCameraCalibration
from robosoccer_perception.pipeline.vision_localization import VisionLocalizationPipeline

pipeline = VisionLocalizationPipeline.from_checkpoints(
    "outputs/detection/yolo11n/weights/best.pt", "outputs/segmentation/unet_small/best.pt")
camera = StereoCameraConfig.load("zed_mini_sim").with_resolution(640, 360)   # the image size used
calibration = EstimatedCameraCalibration.from_camera_config(camera)
pipeline.init_gaussian([-2.0, -1.0, 0.3])           # or pipeline.init_uniform()

out = pipeline.step(image_rgb, timestamp, OdometryDelta(dx=0.03, dy=0.0, dyaw=0.01),
                    calibration, calibration.pose(pan=head_pan))    # the robot's own estimate
print(out.pose, out.estimate.converged, out.estimate.symmetric_ambiguity, out.timing_ms)
```

`step` takes the image, the odometry and the robot's belief about its camera. It has no access
to the simulator; see [docs/vision_localization.md](docs/vision_localization.md).

## Layout

```
configs/
  fields/{s,m,l}_field.yaml         field dimensions (the only place they are written down)
  cameras/zed_mini_sim.yaml         approximate ZED Mini profile and mount
  observations/{nominal,clean}.yaml oracle noise, dropout and false-positive rates
  localization/particle_filter.yaml every localizer parameter
  scenes/default.yaml               turf, lighting, ball, goal appearance
  datasets/{smoke,baseline,large}.yaml         what a dataset contains          (Phase 2)
  randomization/{none,mild,strong}.yaml        domain randomization presets     (Phase 2)
  detection/yolo11n{,_smoke}.yaml              detector training                (Phase 2)
  segmentation/unet_{small,smoke}.yaml         segmenter training               (Phase 2)
  perception/feature_extraction.yaml           geometric extraction parameters  (Phase 3)
  perception/measurement_noise.yaml            uncertainty model, calibration beliefs (Phase 3)
  evaluation/vision_localization.yaml          benchmark settings and sweep     (Phase 3)
  robots/{humanoid_prototype_1,placeholder}.yaml   the body that carries the camera   (Phase 4)
  mounts/{head_low,head_mid,head_high,head_tall,legacy_rig}.yaml   virtual camera mounts,
                                               all experimental                  (Phase 4)
  motion/*.yaml                                synthetic body motion profiles    (Phase 4)
  imu/{nominal,ideal,dropouts}.yaml            simulated IMU                     (Phase 4)
  evaluation/humanoid_fusion.yaml              Phase 4 benchmark and sweeps      (Phase 4)
assets/
  robots/humanoid_prototype_1/      imported model: skeleton, visual meshes, measured body
                                    geometry, manifest with source commit and hashes
src/robosoccer_perception/
  geometry/        field.py  landmarks.py  distance_map.py  transforms.py  camera.py
                   ground_projection.py (pixel -> ground, camera estimates, covariance)
                   quaternion.py  mount.py (body geometry, camera mount)  robot_description.py
  simulation/      mujoco_scene.py  stereo_camera.py  trajectories.py  ground_truth.py  odometry.py
                   scene_objects.py (robots, ball looks, backdrop)
                   humanoid.py (import and posing of humanoid_prototype_1)
                   body_motion.py (synthetic 6-DOF body motion)  imu.py (simulated IMU)
  estimation/      imu_types.py  orientation.py (tilt filter)  sync.py (camera / IMU time)
                   camera_pose.py (per-image camera pose)  calibration_monitor.py
                   (no simulator: this is robot-side code)
  observations/    types.py  oracle.py  noise.py
  localization/    particle_filter.py  motion_model.py  measurement_model.py  estimation.py
  evaluation/      scenarios.py  metrics.py  benchmark.py
  datasets/        schema.py  annotation.py  randomization.py  sampling.py  image_aug.py
                   generator.py  io.py
  detection/       yolo.py  metrics.py  evaluation.py
  segmentation/    model.py  data.py  losses.py  trainer.py  metrics.py  evaluation.py
                   segmenter.py  export.py
  inference/       types.py  pipeline.py  geometry.py  profiling.py
  perception/      line_features.py  line_fitting.py  intersections.py  goal_features.py
                   uncertainty.py  extractor.py  observation_builder.py  types.py  config.py
  pipeline/        vision_localization.py (image + odometry -> pose; no simulator)
  vision_benchmark/  camera_truth.py  sim_camera.py  scenarios.py  corruption.py  runner.py
                   feature_metrics.py  feature_evaluation.py  benchmark.py  setup.py
  humanoid_benchmark/  scenarios.py  runner.py (simulation side, robot side, arms)
                   benchmark.py (aggregation, calibration and timing sweeps)  setup.py
  visualization/   field_plot.py  diagnostics.py  camera_overlay.py  style.py
                   vision_overlay.py  vision_plots.py  vision_localization_plots.py
                   humanoid_plots.py
  viewer/          session.py  frame.py  recording.py  render.py  worker.py  headless.py
                   app.py  qt_env.py   (the interactive debugging viewer)
                   humanoid_session.py  catalog.py   (Phase 4 scenarios in the viewer)
scripts/           render_field.py  run_localization.py  benchmark.py
                   generate_dataset.py  inspect_dataset.py  train_detector.py  evaluate_detector.py
                   train_segmentation.py  evaluate_segmentation.py  run_vision.py  export_models.py
                   run_vision_localization.py  benchmark_vision_localization.py
                   evaluate_features.py  visualize_pipeline.py
                   import_humanoid_model.py  render_humanoid.py  benchmark_humanoid_fusion.py
tests/             one file per area; see docs/validation.md, docs/vision_validation.md,
                   docs/phase3_validation.md and docs/phase4_validation.md
docs/              the documents below
```

## Documentation

| Document | Read it for |
|---|---|
| [docs/architecture.md](docs/architecture.md) | how the pieces fit, design rules, extension points |
| [docs/coordinate_frames.md](docs/coordinate_frames.md) | frames, transform naming, MuJoCo versus ROS optical frames |
| [docs/field_geometry.md](docs/field_geometry.md) | field configs, outside-of-line convention, landmarks, distance map |
| [docs/camera_model.md](docs/camera_model.md) | the virtual ZED Mini, what is verified, what is not modelled |
| [docs/localization.md](docs/localization.md) | the particle filter explained for newcomers, symmetry, recovery, limits |
| [docs/validation.md](docs/validation.md) | Phase 1: test coverage, benchmark results, known failures, limitations |
| [docs/synthetic_data.md](docs/synthetic_data.md) | dataset generation, label schema, occlusion policy, randomization, splits |
| [docs/object_detection.md](docs/object_detection.md) | the YOLO baseline: model choice, licensing, training, evaluation, export |
| [docs/segmentation.md](docs/segmentation.md) | the U-Net baseline: model, class imbalance, line metrics, training, evaluation |
| [docs/vision_validation.md](docs/vision_validation.md) | Phase 2: tests, measured results, failure cases, what is not validated |
| [docs/phase3_interfaces.md](docs/phase3_interfaces.md) | what vision hands over and what localization expects (written before Phase 3; now annotated) |
| [docs/geometric_perception.md](docs/geometric_perception.md) | Phase 3: ground projection, line points, goal posts, junctions, circle, the uncertainty model |
| [docs/vision_localization.md](docs/vision_localization.md) | Phase 3: the camera-to-pose pipeline, the ground-truth boundary, benchmark arms and scenarios |
| [docs/phase3_validation.md](docs/phase3_validation.md) | Phase 3: oracle versus vision versus odometry, calibration sensitivity, latency, failures, limits |
| [docs/viewer.md](docs/viewer.md) | the interactive viewer: install, controls, what each panel shows, recordings, limits |
| [docs/humanoid_integration.md](docs/humanoid_integration.md) | Phase 4: the audit of `humanoid_prototype_1`, what was imported and what was left out, how the body is posed |
| [docs/camera_mount.md](docs/camera_mount.md) | Phase 4: the virtual head and the configurable camera mount, every assumption, what to change when the real head exists |
| [docs/humanoid_fusion.md](docs/humanoid_fusion.md) | Phase 4: body motion, the simulated IMU, the tilt filter, camera / IMU synchronisation, uncertainty-aware projection, the mount monitor |
| [docs/phase4_validation.md](docs/phase4_validation.md) | Phase 4: fixed against IMU-assisted localization, calibration and timing sensitivity, failures, limits, next steps |

## Things to know before trusting a number

- **The field is symmetric.** No localizer can tell a pose from its half-turn mirror image
  using the field alone. With no side information this filter reports both and says so.
- **The Phase 1 oracle is a model, not a detector.** Its noise levels are engineering guesses.
  Phase 3 keeps it as the reference arm that vision is compared against.
- **The camera profile is approximate** and the mount is a placeholder for a robot that is not
  built yet.
- **Several field values are assumptions** (goal height, post diameter, mark size, ball size).
  They are marked in the YAML files and listed in `docs/field_geometry.md`.
- **Vision results are synthetic-on-synthetic.** The detector and the segmenter were trained
  and tested on images from the same simulator. Their scores say the pipeline works; they say
  nothing yet about a real ZED Mini on a real field. The robots in particular are placeholders.
- **Localization from images works in the simulator, and only there.** With a correct camera
  estimate the vision arm matches the oracle (2 to 4 cm, about 0.5 deg, 23 frames per second
  at 640x360). Every image was rendered by this repository and the camera is an ideal pinhole
  on a platform that does not walk.
- **Camera calibration decides the accuracy.** On Phase 3's rigid platform a hidden 2 deg
  pitch error costs 34 cm and the filter still reports itself converged; 5 cm of height costs
  12 cm. Phase 4 is about exactly this weakness: see the last bullets.
- **A robot carried across the centre line re-localizes on the mirrored pose**, from images
  and from oracle observations alike. Only outside information can prevent it.
- **The humanoid in the simulator is a posed picture of the robot, not the robot.** It is
  `humanoid_prototype_1`'s real geometry and joint tree, but nothing is simulated: it cannot
  balance or fall, and the stepping is an animation that has no effect on the camera.
- **The robot has no head, so every camera position is an assumption.** The mount presets in
  `configs/mounts/` are experimental. The camera height is derived from the torso's measured
  top (0.679 m) plus a preset's offset; between 0.74 and 0.90 m the height made no measurable
  difference to localization in the simulator.
- **A fixed camera model does not survive a moving torso.** With synthetic walking motion
  (a 2 degree lean, 1 to 2 degrees of oscillation) Phase 3's pipeline averages about 1 m of
  position error and reports itself converged while lost. With the simulated IMU's roll and
  pitch it is at 6 to 16 cm, and with uncertainty-aware projection on top at 3 to 8 cm
  (the true-orientation upper bound is 2 to 4 cm), at 21.6 frames per second.
- **Those IMU results come from a simulated IMU on synthetic motion.** The filter was tuned on
  the same family of motion it was tested on. No real IMU sample and no measured gait has been
  used.
- **An IMU cannot see a wrong camera mount.** A 2 degree mount pitch error still costs 58 cm
  with the IMU alone. Uncertainty-aware projection brings that to 11 cm, and the mount monitor,
  which reads the error from the field lines, to 7 cm. The monitor cannot tell a pitch error
  from a height error.
