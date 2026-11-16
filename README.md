# soccer-perception

Perception and localization research for the UTRA RoboSoccer humanoid.

This first MVP lets the team develop and **measure** localization before any neural perception
model exists and before the robot can walk. It contains:

- a configurable RoboCup field (S, M and L sizes of the 2026 Humanoid League rulebook);
- a MuJoCo scene of that field with a movable virtual ZED Mini stereo camera;
- an **oracle** that turns simulator ground truth into noisy, anonymous landmark and
  field-line observations, standing in for detectors that are not trained yet;
- noisy synthetic odometry;
- a 3-DOF (x, y, yaw) particle-filter localizer;
- deterministic benchmark scenarios with metrics, plots and JSON/CSV results.

It is a standalone Python package (`robosoccer_perception`). It does not use ROS and does not
depend on `robosoccer-rl` or `soccer-software` at run time.

Not in this MVP: neural networks or training of any kind, the ZED SDK, ROS 2 nodes or messages,
SLAM, humanoid dynamics or locomotion.

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
.venv/bin/python -m pytest                 # everything, about 1 minute
.venv/bin/python -m pytest -m "not slow"   # skips the full localization runs, about 10 s
```

Tests that render images are skipped automatically when no OpenGL context is available.

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

## Layout

```
configs/
  fields/{s,m,l}_field.yaml         field dimensions (the only place they are written down)
  cameras/zed_mini_sim.yaml         approximate ZED Mini profile and mount
  observations/{nominal,clean}.yaml oracle noise, dropout and false-positive rates
  localization/particle_filter.yaml every localizer parameter
  scenes/default.yaml               turf, lighting, ball, goal appearance
src/robosoccer_perception/
  geometry/        field.py  landmarks.py  distance_map.py  transforms.py  camera.py
  simulation/      mujoco_scene.py  stereo_camera.py  trajectories.py  ground_truth.py  odometry.py
  observations/    types.py  oracle.py  noise.py
  localization/    particle_filter.py  motion_model.py  measurement_model.py  estimation.py
  evaluation/      scenarios.py  metrics.py  benchmark.py
  visualization/   field_plot.py  diagnostics.py  camera_overlay.py  style.py
scripts/           render_field.py  run_localization.py  benchmark.py
tests/             one file per area; see docs/validation.md
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
| [docs/validation.md](docs/validation.md) | test coverage, benchmark results, known failures, limitations, next steps |

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
