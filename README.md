# soccer-perception

Perception and localization research for the UTRA RoboSoccer humanoid.

The repository has two parts, built in two phases:

- **Phase 1, localization on oracle observations.** Lets the team develop and **measure**
  localization before any neural perception model exists and before the robot can walk.
- **Phase 2, vision.** Generates labelled synthetic camera images from the same simulator,
  trains a small object detector (ball, goal posts, robots) and a small segmentation network
  (field, field lines), evaluates both, and runs them together on camera frames.

The two are **not connected yet**: the localizer still runs on oracle observations, and the
vision models stop at boxes and masks. Joining them is Phase 3
([docs/phase3_interfaces.md](docs/phase3_interfaces.md)).

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

It is a standalone Python package (`robosoccer_perception`). It does not use ROS and does not
depend on `robosoccer-rl` or `soccer-software` at run time.

Not here: the ZED SDK, ROS 2 nodes or messages, SLAM, humanoid dynamics or locomotion, any use
of the vision outputs for localization, and any validation on real camera footage.

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
.venv/bin/python -m pytest                    # everything, about 1.5 minutes
.venv/bin/python -m pytest -m "not slow"      # skips full localization runs and training smoke tests
.venv/bin/python -m pytest -m "not vision"    # Phase 1 and the parts of Phase 2 that need no PyTorch
```

Tests that render images are skipped automatically when no OpenGL context is available, and
tests that need PyTorch or Ultralytics when the `vision` extra is not installed. No test uses
the network or needs downloaded weights.

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
src/robosoccer_perception/
  geometry/        field.py  landmarks.py  distance_map.py  transforms.py  camera.py
  simulation/      mujoco_scene.py  stereo_camera.py  trajectories.py  ground_truth.py  odometry.py
                   scene_objects.py (robots, ball looks, backdrop)
  observations/    types.py  oracle.py  noise.py
  localization/    particle_filter.py  motion_model.py  measurement_model.py  estimation.py
  evaluation/      scenarios.py  metrics.py  benchmark.py
  datasets/        schema.py  annotation.py  randomization.py  sampling.py  image_aug.py
                   generator.py  io.py
  detection/       yolo.py  metrics.py  evaluation.py
  segmentation/    model.py  data.py  losses.py  trainer.py  metrics.py  evaluation.py
                   segmenter.py  export.py
  inference/       types.py  pipeline.py  geometry.py  profiling.py
  visualization/   field_plot.py  diagnostics.py  camera_overlay.py  style.py
                   vision_overlay.py  vision_plots.py
scripts/           render_field.py  run_localization.py  benchmark.py
                   generate_dataset.py  inspect_dataset.py  train_detector.py  evaluate_detector.py
                   train_segmentation.py  evaluate_segmentation.py  run_vision.py  export_models.py
tests/             one file per area; see docs/validation.md and docs/vision_validation.md
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
| [docs/phase3_interfaces.md](docs/phase3_interfaces.md) | what vision hands over, and what is still missing before localization can use it |

## Things to know before trusting a number

- **The field is symmetric.** No localizer can tell a pose from its half-turn mirror image
  using the field alone. With no side information this filter reports both and says so.
- **Perception is an oracle.** Its noise levels are engineering guesses, not measurements of a
  real detector. Results show that the localizer works given such observations, not how well
  the robot will localize.
- **The camera profile is approximate** and the mount is a placeholder for a robot that is not
  built yet.
- **Several field values are assumptions** (goal height, post diameter, mark size, ball size).
  They are marked in the YAML files and listed in `docs/field_geometry.md`.
- **Vision results are synthetic-on-synthetic.** The detector and the segmenter were trained
  and tested on images from the same simulator. Their scores say the pipeline works; they say
  nothing yet about a real ZED Mini on a real field. The robots in particular are placeholders.
- **Vision and localization are not joined.** No localization number in this repository was
  produced from an image.
