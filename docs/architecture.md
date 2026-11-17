# Architecture

This repository lets the team develop and measure localization before any neural perception
exists and before the robot can walk. It does that with four pieces that only talk through
small, typed interfaces:

```
configs/fields/*.yaml
        │
        ▼
  FieldGeometry ──────────────┬──────────────┬───────────────┬──────────────┐
  (geometry/field.py)         │              │               │              │
                              ▼              ▼               ▼              ▼
                        MuJoCo scene    LandmarkMap     DistanceMap     field plots
                        (rendering)     (point          (line           (visualization)
                                        landmarks)      likelihood)

 true trajectory ──► SyntheticOdometry ──► OdometryDelta ─────┐
       │                                                       ▼
       ├──────────► OracleObservationGenerator ─► ObservationFrame ─► ParticleFilter ─► PoseEstimate
       │                                                                                   │
       └───────────────────────────► evaluation: errors, metrics, plots ◄──────────────────┘
```

## Packages

| Package | Contents | Needs |
|---|---|---|
| `geometry/` | `field.py` (the one authoritative field description), `landmarks.py`, `distance_map.py`, `transforms.py` (frames), `camera.py` (pinhole + stereo extrinsics) | NumPy |
| `simulation/` | `trajectories.py`, `ground_truth.py`, `odometry.py` (synthetic odometry) | NumPy |
| `simulation/` | `mujoco_scene.py`, `stereo_camera.py` (rendering) | MuJoCo |
| `observations/` | `types.py` (what perception hands to localization), `noise.py`, `oracle.py` | NumPy |
| `localization/` | `motion_model.py`, `measurement_model.py`, `particle_filter.py`, `estimation.py` | NumPy |
| `evaluation/` | `scenarios.py`, `metrics.py`, `benchmark.py` | NumPy |
| `visualization/` | `field_plot.py`, `diagnostics.py`, `camera_overlay.py`, `style.py` | Matplotlib |

MuJoCo is imported only by the rendering modules. Geometry, the oracle, the localizer and
the benchmarks run without it, which keeps the test suite fast and lets the localizer be reused
on the robot later.

## Phase 2: the vision side

A second, separate chain turns the same simulator into training data and models:

```
 MuJoCo scene (+ robots, backdrop) ──► colour image ─────────────┐
        │                                                        ▼
        └─► geom-id image ─► boxes + semantic mask ─► dataset ─► training ─► detector, segmenter
            (labels read from the renderer)                                        │
                                                                                   ▼
                                          any RGB image ─► VisionPipeline ─► VisionResult
                                                                             (boxes, mask, timings)
```

| Package | Contents | Needs |
|---|---|---|
| `simulation/scene_objects.py` | humanoid placeholders, ball looks, backdrop | MuJoCo |
| `datasets/` | `schema.py` (class ids, label policy), `annotation.py` (labels from geom ids), `randomization.py`, `sampling.py` (viewpoints and layouts), `image_aug.py`, `generator.py`, `io.py` | MuJoCo, OpenCV |
| `detection/` | `yolo.py` (the only module that imports Ultralytics), `metrics.py`, `evaluation.py` | PyTorch, Ultralytics (metrics: NumPy only) |
| `segmentation/` | `model.py`, `data.py`, `losses.py`, `trainer.py`, `metrics.py`, `evaluation.py`, `segmenter.py`, `export.py` | PyTorch, OpenCV |
| `inference/` | `types.py` (what vision outputs), `pipeline.py`, `geometry.py` (inspection utilities), `profiling.py` | NumPy; models are passed in |

The two chains share `geometry/` and `simulation/` and nothing else. The vision packages do not
import `observations/`, `localization/` or `evaluation/`, and those do not import the vision
packages, PyTorch or OpenCV (`tests/test_vision_pipeline.py` checks both directions). The link
between them, vision outputs to `ObservationFrame`, is Phase 3, below.

Design rules 1 to 5 below apply to Phase 2 as well, with two additions:

- **Labels come from rendering, not from intention.** A box exists because pixels show the
  object, never because a sampler meant the object to be visible.
- **Models see images only.** Ground truth is read after inference, to score it.

## Phase 3: from images to a pose

Phase 3 adds the piece between the two chains, and a benchmark that compares it with the
oracle on identical runs. Neither the vision pipeline nor the particle filter was changed.

```
                       the robot's side (no simulator)                        the simulator's side
 ┌───────────────────────────────────────────────────────────────────┐   ┌──────────────────────────┐
 │ RGB image ─► VisionPipeline ─► VisionResult                       │◄──│ MuJoCo render with the   │
 │   EstimatedCameraCalibration, EstimatedCameraPose                 │   │ TRUE camera (mount error,│
 │        └─► GeometricFeatureExtractor ─► FeatureSet                │   │ head shake): hidden      │
 │                 └─► build_observation_frame ─► ObservationFrame   │   │                          │
 │ OdometryDelta ─────────────────► ParticleFilter ─► PoseEstimate ──┼──►│ evaluation: pose errors, │
 │                                                    FeatureSet ────┼──►│ feature scores vs truth  │
 └───────────────────────────────────────────────────────────────────┘   └──────────────────────────┘
```

| Package | Contents | Needs |
|---|---|---|
| `geometry/ground_projection.py` | the robot's camera calibration and pose estimate, pixel to ground, validity, covariance | NumPy |
| `perception/` | `line_features.py`, `line_fitting.py`, `intersections.py`, `goal_features.py`, `uncertainty.py`, `extractor.py`, `observation_builder.py`, `types.py`, `config.py` | NumPy, OpenCV |
| `pipeline/` | `vision_localization.py`: `VisionLocalizationPipeline.step(image, timestamp, odometry_delta, camera_calibration, estimated_camera_pose)` | the above; models are passed in |
| `vision_benchmark/` | `camera_truth.py`, `sim_camera.py`, `scenarios.py`, `corruption.py`, `runner.py`, `feature_metrics.py`, `feature_evaluation.py`, `benchmark.py`, `setup.py` | MuJoCo |
| `viewer/` | `session.py`, `frame.py`, `recording.py`, `render.py`, `worker.py`, `headless.py`, `app.py`, `qt_env.py` (the interactive debugging viewer) | MuJoCo, OpenCV; PySide6 for the window |

`perception/` and `pipeline/` are the runtime. They import neither MuJoCo, nor `simulation/`,
nor the oracle, nor `vision_benchmark/`. The benchmark is a package of its own, not part of
`evaluation/`, because Phase 1's evaluator has to stay free of the vision stack.

Two more rules:

- **The robot's camera and the simulator's camera are different types.**
  `EstimatedCameraCalibration` / `EstimatedCameraPose` are what the pipeline is given;
  `SimulationCameraTruth` is what the image is rendered with. They cannot be passed in each
  other's place, and the benchmark makes them differ on purpose.
- **Arms are not mixed.** A run's vision arm gets observations from image predictions only;
  the oracle arm, the dead-reckoning arm and the "perfect vision" diagnostic are separate
  localizers on the same trajectory, reported under their own names.

Details: [geometric_perception.md](geometric_perception.md),
[vision_localization.md](vision_localization.md), [phase3_validation.md](phase3_validation.md).

### The viewer

`viewer/` is an observer on top of Phase 3 ([viewer.md](viewer.md)). `session.py` runs one
benchmark scenario a frame at a time with the benchmark's own building blocks and random
streams, calls `VisionLocalizationPipeline.step` exactly as the benchmark does, and packs
each step into a `ViewerFrame`; `render.py` draws the four panels from that frame; a worker
thread keeps the simulation and the networks away from the window. It adds no estimator
code: the particle filter, the models, the extraction and the benchmark are untouched, and a
session reproduces the benchmark's vision arm bit for bit.

## Design rules

1. **One field description.** `FieldGeometry` is built from a YAML file and nothing else holds
   field dimensions. The scene, the landmark map, the distance map, the oracle, the scenarios
   and the plots are all derived from it, so switching fields is a configuration change.
2. **Ground truth stops at a firewall.** The localizer receives `OdometryDelta` (relative
   motion) and `ObservationFrame` (anonymous detections in the robot frame). Neither type has a
   field for a global pose or a landmark identity. A global pose reaches the filter exactly
   once, through `init_gaussian`. `tests/test_benchmarks.py` records everything the benchmark
   passes to the filter and replays it without a simulator; the estimates are bit-identical.
   Phase 3 moves the firewall out to the image: `tests/test_ground_truth_isolation.py` does
   the same recording and replay for `VisionLocalizationPipeline.step`.
3. **No ROS.** Types are plain dataclasses and NumPy arrays. Frame names and conventions follow
   REP-103 and REP-105 so that wrapping them in ROS 2 messages later is mechanical.
4. **Deterministic.** Every random draw comes from a seeded `numpy.random.Generator`. A
   scenario and a seed fully determine a run.
5. **The simulator is kinematic.** The camera platform is placed, never balanced. No policy,
   no physics stepping, no dependency on the locomotion work.

## Relationship to the other repositories

Nothing is imported from or written to `robosoccer-rl` or `soccer-software`. What was taken
from them is convention only:

- MuJoCo 3.4.0, the version both pin.
- Ground truth as "pose of the robot root in the simulator world frame plus simulation time",
  as in soccer-software's `GroundTruthFrame` (`sim_world -> ground_truth/pelvis`). Here the
  world frame is the field frame and the root is the base footprint.
- REP-103 body axes and REP-105 frame roles (`odom -> base` owned by odometry; the localizer
  would own `map -> odom`).

The Unitree G1 model was **not** copied. soccer-software's G1 description has no camera, its
meshes are about 38 MB of binary STL files, and a legged model adds nothing while the camera is
moved kinematically. The platform is a mocap body with a placeholder shape; see
`simulation/mujoco_scene.py` for where a humanoid model would attach.

## Extension points

| To add | Touch |
|---|---|
| A field size | a new `configs/fields/*.yaml` |
| The calibrated ZED Mini | a new `configs/cameras/*.yaml` (and a distortion model in `geometry/camera.py`) |
| Another image feature (penalty marks, stereo range) | a module in `perception/`, a field in `FeatureSet`, a line in `build_observation_frame` |
| The measured camera mount | `configs/cameras/*.yaml` and the beliefs in `configs/perception/measurement_noise.yaml` |
| Another detector or segmenter | a class with `predict(image_rgb)` returning `Detection`s or a `Segmentation` |
| A dataset variant | a new `configs/datasets/*.yaml` or `configs/randomization/*.yaml` |
| Real odometry | produce `OdometryDelta`s from the walking engine / IMU |
| A humanoid model | replace the placeholder geoms of the `base` body in `build_spec`; for other robots, `scene_objects.add_humanoid` |
| A ROS 2 node | wrap `VisionLocalizationPipeline.step`; take `T_base_camera` from TF; publish `map -> odom` from `PoseEstimate` |
