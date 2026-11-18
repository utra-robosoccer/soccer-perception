# Pipeline viewer

An interactive window for watching the vision-localization pipeline work, one frame at a time:
what the camera saw, what the networks predicted, what was extracted from it, where the
particle filter thinks the robot is, and how wrong it is. It runs the benchmark scenarios
live in the simulator, or replays a recorded run.

Since Phase 4 it also shows **UTRA's humanoid_prototype_1** walking, swaying and being pushed
around the field, with a third-person picture of the robot in the camera panel, a choice of
virtual camera mounts, and the robot's estimated camera orientation next to the true one
([Phase 4 scenarios](#phase-4-the-humanoid-and-its-moving-camera)).

It is a debugging tool for the simulator. It does not talk to ROS, a real camera or the
physical robot. The humanoid in it is a posed, visual-only model: it is not simulated
dynamically and its stepping is an animation.

```
+----------------------------+----------------------------+-----------+
| CAMERA                     | SEGMENTATION               | overlay   |
| image, detector boxes,     | field / line classes, line | toggles   |
| labels and scores          | points, junctions, circle, |           |
|                            | goal-post feet             | output    |
+----------------------------+----------------------------+ buttons   |
| LOCALIZATION MAP           | DIAGNOSTICS                |           |
| particles, estimate, truth,| errors, confidence, ESS,   |           |
| observed features,         | latency; scenario, time,   |           |
| ambiguity and uncertainty  | simulation FPS             |           |
+----------------------------+----------------------------+-----------+
  scenario  seed  Run  < Step  Step >  Reset  speed  [ground truth]     slider
  robot  camera mount  body motion  camera pose                 (Phase 4 scenarios)
```

## Install

```bash
.venv/bin/python -m pip install -e ".[all,vision,viewer]"
```

The `viewer` extra adds `PySide6-Essentials` (about 250 MB installed). The models and the simulator come
from the `all` and `vision` extras, as for the benchmark, and the trained checkpoints must
exist (`outputs/detection/yolo11n/weights/best.pt`, `outputs/segmentation/unet_small/best.pt`;
see the README's "Vision" section).

On Ubuntu, Qt 6.5 and newer needs one system library that a default desktop does not have:

```bash
sudo apt install libxcb-cursor0
```

Without root, put a copy where the viewer looks for it:

```bash
cd /tmp && apt download libxcb-cursor0 && dpkg -x libxcb-cursor0_*.deb xcb-cursor
mkdir -p <repository>/.venv/lib/qt-extra
cp -a xcb-cursor/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0* <repository>/.venv/lib/qt-extra/
```

If the library is missing, the viewer prints these instructions and exits; it does not crash.
(`import cv2` also redirects Qt to OpenCV's bundled Qt 5 plugins; the viewer undoes that by
itself, in `viewer/qt_env.py`.)

## Run

From the repository root. With a ROS workspace sourced, prefix with `env -u PYTHONPATH`.

```bash
.venv/bin/python scripts/visualize_pipeline.py                         # h05_walk_rectangle: the
                                                                       # humanoid walking, paused
.venv/bin/python scripts/visualize_pipeline.py h02 --fusion fixed      # sway, Phase 3 camera model
.venv/bin/python scripts/visualize_pipeline.py h10 --fusion imu_selfcal --autoplay
.venv/bin/python scripts/visualize_pipeline.py h05 --mount head_high --motion walking_rough
.venv/bin/python scripts/visualize_pipeline.py h05 --robot placeholder --mount legacy_rig
.venv/bin/python scripts/visualize_pipeline.py v13a --seed 2 --autoplay     # a Phase 3 scenario
.venv/bin/python scripts/visualize_pipeline.py v12a --speed 0.5
.venv/bin/python scripts/visualize_pipeline.py --list                  # all 34 scenarios
.venv/bin/python scripts/visualize_pipeline.py --replay outputs/viewer/recordings/<file>.npz

# no window: run a scenario to its end, save a recording, a video and two pictures
.venv/bin/python scripts/visualize_pipeline.py v04 --headless --record --video --screenshots 2
```

A scenario may be given by any prefix of its id. Other options: `--no-ground-truth`,
`--max-steps N`, `--particles N` (how many are drawn), `--resolution 1280 720`, `--device`,
`--settings` (the benchmark settings file that names the checkpoints).

The window opens paused on the first frame of the run. The first inference after loading
takes one to two seconds (CUDA warm-up); after that a frame takes about 50 ms.

## Controls

| Control | Key | What it does |
|---|---|---|
| Scenario, Seed | | load a benchmark scenario; the run is the benchmark's run for that seed |
| Robot | | Phase 4: `humanoid_prototype_1` or the `placeholder` rig of Phases 1 to 3 |
| Camera mount | | Phase 4: a virtual mount preset (`head_low` ... `head_tall`, `legacy_rig`); all experimental |
| Body motion | | Phase 4: `standing`, `sway_pitch`, `sway_roll`, `walking`, `walking_rough`, `disturbed`, ... |
| Camera pose | | Phase 4: how the robot estimates its camera pose: fixed (Phase 3), IMU, IMU + uncertainty, IMU + mount self-calibration, or the naive newest-IMU-sample pairing |
| Run / Pause | Space | start or stop |
| Step > | Right | one frame forward: from the history if there is one, else simulate one |
| < Step | Left | one frame back (from the history; the filter is not run backwards) |
| Reset | R | restart the run from its first frame |
| Speed | | 0.25x, 0.5x, 1x, 2x, 4x of real time, or max |
| Ground truth | G | show or hide the true pose, its trail and the error curves |
| Slider | Home, End | scrub through every frame simulated so far |
| Screenshot | S | the four panels as one PNG |
| Save recording | | everything simulated so far, for replay |
| Open recording | | replay a saved run; "Back to live" returns to the simulation |
| Export video | | the run as an `.mp4`, with the overlays as currently set |
| Overlay checkboxes | | each drawn layer of each panel, individually |

There is one transport. Every frame of the run is kept (compressed) in a history, and the
slider moves a cursor through it. **Run** plays the history forward from the cursor and, when
it reaches the newest frame of a live run, continues simulating. So after pausing and
scrubbing back to look at something, Run replays what was already seen and then carries on.
When a run is complete, Run plays it again from the start.

## What the panels show

**Camera.** The image exactly as it is given to the pipeline, with the detector's boxes
(ball orange, goal post yellow, robot blue), class names and confidences.

**Segmentation.** The segmenter's classes over the image (field tinted cyan, lines magenta,
background dimmed) and what geometric extraction made of them:

| Mark | Meaning |
|---|---|
| orange dot | a line point sent to the localizer |
| square with L, T or X | a classified junction; grey with `?` when its shape could not be established (not sent to the localizer) |
| violet outline | the centre-circle fit, drawn where the fitted circle would appear; labelled "circle candidate (rejected)" when it failed the acceptance tests |
| yellow cross | the foot of a goal post with a trusted range |
| yellow ring | a goal post reduced to bearing only |
| thin white line | a fitted line segment (off by default) |

**Localization map.** The field from above, +x to the right.

| Mark | Meaning |
|---|---|
| filled orange triangle | the estimated pose and heading |
| orange ellipse and wedge | 2 sigma of position and of heading, as the filter reports them |
| open orange triangles with a percentage | other hypotheses the filter still holds, and their share of the belief |
| blue ring with a line | the true pose and heading (ground truth) |
| yellow dots | particles (600 of them by default, evenly chosen) |
| cyan dots and diamonds | the line points and landmarks of this frame, placed on the map **with the estimated pose**: when the estimate is right they lie on the painted lines |
| thin orange / blue lines | the paths of the estimate and of the truth so far |

The title bar states the filter's own verdict in words: `CONVERGED`, `NOT CONVERGED` or
`AMBIGUOUS: the mirrored pose fits equally well`, plus "robot moved (kidnapped)" and
"odometry lost" on the frames where that happens.

**Diagnostics.** Scenario, seed, simulated time, frame number, simulation rate (frames per
wall-clock second and as a multiple of real time) and the stage timings of this frame; then
four plots over the whole run with a cursor at the current frame:

1. position error against the filter's reported 1 sigma (dashed: the 0.20 m target);
2. yaw error against the reported 1 sigma (dashed: 5 deg);
3. the filter's confidence and its effective sample size as a share of the particles; yellow
   ticks along the top mark the frames flagged ambiguous;
4. latency: inference (detector + segmenter) and the whole pipeline step.

Plots 1 and 2 are where a calibration error shows: in `v09_camera_pitch_error` the error
curve sits far above the reported sigma. Dashed vertical lines mark kidnappings.

## Phase 4: the humanoid and its moving camera

Scenarios whose id starts with `h` (`--list`) put a moving body under the camera: standing,
forward / backward and lateral sway, walking, turning, abrupt disturbances, a mount that is
not where the robot believes, different head heights, lost IMU samples and timing errors. They
are the scenarios of `scripts/benchmark_humanoid_fusion.py`, and a viewer session reproduces
the benchmark arm it shows bit for bit.

The second control bar chooses the body, the mount, the motion and the camera-pose estimator.
"As the scenario" keeps the scenario's own; changing any of them restarts the run. For a
Phase 3 scenario (`v...`) the bar is greyed out.

What is added to the four panels:

**Camera.** The image comes from the virtual ZED Mini on the selected mount. In the top-right
corner is a third-person picture of the robot (toggle: "Robot view"). It shows the real
humanoid model stepping across the field and, drawn on it, the body frame's axes (red, green,
blue), the mount offset (yellow), the virtual head mast (grey), the camera housing and the
viewing direction (magenta). The markers and the mast are aids, not parts; the robot's own
camera never sees them.

**Diagnostics.** Four more lines of text and one more plot:

| Line | Content |
|---|---|
| camera (estimated) | the pitch, roll and height the robot projects with. No ground truth is involved |
| camera (true) | the same from the simulator, and the orientation error (shown with ground truth only) |
| body tilt | the estimated roll and pitch with their uncertainty, how they were obtained (`imu`, `nominal`, `nominal_fallback`), the synchronisation method and the distance in time to the nearest IMU sample; the true tilt beside it |
| mount | the mount transform in use, flagged EXPERIMENTAL, and with self-calibration the monitor's status, estimated bias and current correction |

The extra plot is the camera pitch below the horizon: estimated (orange) against true (blue).
With "Fixed camera" the orange line is flat while the blue one moves; with the IMU it follows.
In `h10_mount_pitch_error` the two stay 2 degrees apart until the self-calibration closes the
gap.

Things worth watching:

| Run | What it shows |
|---|---|
| `h02 --fusion fixed`, then `--fusion imu` | a +-4 degree sway destroys localization with a fixed camera model and is harmless with the IMU |
| `h05 --fusion imu` against `imu_uncertainty` | the same pose error, an honest (wider) reported uncertainty |
| `h10 --fusion imu` against `imu_selfcal` | a mount error the IMU cannot see, and the monitor finding it in the field lines |
| `h05 --robot placeholder --mount legacy_rig` | the Phase 1 to 3 rig under the same motion |
| `h05 --mount head_low` / `head_tall` | what the head's height does to the view |

The recordings hold all of this (including the robot picture), so a Phase 4 run replays
without the simulator. Recordings made before Phase 4 still open.

Measured on the RTX 2050 laptop at 640x360 with the humanoid: the third-person picture adds
about 6 ms of rendering per frame and the IMU filter under 2 ms; the window still runs at real
time (10 frames per second) with headroom.

## Recordings, videos, screenshots

Everything is written under `outputs/viewer/`, which Git ignores (`outputs/`, `*.npz`,
`*.mp4`):

| Folder | Content | Size |
|---|---|---|
| `recordings/<scenario>_seed<k>_<time>.npz` | every frame's image (JPEG), mask (PNG), detections, features, particles, estimate, timings and ground truth | about 57 kB per frame: 33 MB for the 58 s of `v04_rectangle` |
| `videos/<...>.mp4` | the four panels, 1926x1142, H.264 through `ffmpeg` when it is installed (else OpenCV's MPEG-4) | about 10 MB per minute |
| `screenshots/<...>_stepNNNN.png` | the four panels of one frame | 0.8 MB |

A recording replays without the simulator and without the models: the same panels, the same
toggles, the same scrubbing. Screenshots and videos are always drawn at 960x540 per panel,
whatever the size of the window.

## Ground truth

The viewer shows ground truth; the estimator never gets it. `viewer/session.py` runs each
step in three parts that share no variables:

- `_simulate` (the simulator) knows the true pose and returns a `_Sensors` packet: the image,
  the timestamp, the odometry increment and the commanded head angles;
- `_localize` (the robot) receives that packet and calls
  `VisionLocalizationPipeline.step(image, timestamp, odometry, calibration, camera_estimate)`,
  the same call the benchmark makes, which rejects any other type;
- `_observe` (the viewer) puts the pipeline's output and, for display only, the true pose
  into the frame the panels are drawn from.

The filter is initialised by the scenario's rule, as in the benchmark (a noisy guess, one
half of the field, or nothing). Tests check that a session's estimates are **bit for bit**
the benchmark's vision arm, that everything handed to `step` is sensor data, that a session
built with `with_truth=False` gives the same estimates, and that switching the ground-truth
display off in the window changes neither the estimates nor leaves any trace in the picture.

## How it is built

| Module (`src/robosoccer_perception/viewer/`) | Role | Needs |
|---|---|---|
| `session.py` | one scenario, one `ViewerFrame` per `step()`; pause, step, reset | MuJoCo |
| `frame.py` | the `ViewerFrame` packet, its packed form, the per-run `Timeline` | OpenCV |
| `recording.py` | recordings on disk, read lazily | NumPy |
| `render.py` | the four panels as RGB arrays | OpenCV |
| `worker.py` | the thread that runs the session | - |
| `headless.py` | recording, video and pictures without a window | OpenCV, `ffmpeg` optional |
| `app.py`, `qt_env.py` | the window | PySide6 |

- **One inference per frame.** A `ViewerFrame` is made by one `session.step()` and carries
  the image, the predictions, the features, the estimate, the timings and the truth of that
  step. All four panels are drawn from the frame under the cursor in one call, so they cannot
  show different timestamps.
- **The window does not wait for inference.** Simulation, networks and filter run in a worker
  thread. The window redraws at most 30 times a second from the newest frame it has; drawing
  the four panels takes about 12 ms.
- **Frames do not pile up.** The worker stops when four frames have not been taken yet, so a
  slow window slows the simulation instead of growing a queue.
- **The picture is the same everywhere.** The window, the screenshots and the videos are
  drawn by `render.py`; nothing is drawn with Qt.

Measured on the RTX 2050 laptop at 640x360: 10 frames per second at 1x (real time, with
headroom), about 17 frames per second at "max" (1.7x real time; a frame costs the pipeline's
45 to 50 ms, 7 ms of rendering the simulated image and 1 ms of packing).

## Tests

```bash
.venv/bin/python -m pytest tests/test_viewer.py tests/test_viewer_gui.py tests/test_viewer_models.py
```

| File | Tests | What it establishes |
|---|---|---|
| `test_viewer.py` | 41 | frame packing and recordings round-trip; every overlay toggle changes its own panel and no other; hiding ground truth leaves no trace; the map states ambiguity; video export; **a session reproduces the benchmark's vision arm bit for bit, and each frame's image is the one its estimate was computed from**; **the estimator is given sensor data only**; reset repeats the run exactly; the worker steps, runs, pauses, resets, paces itself to the speed, waits for a slow consumer and reports failures |
| `test_viewer_gui.py` | 17 | the real window without a screen: opens paused; Run, Pause, Step, back-step and Reset; **the four panels are the frame under the slider**; replay from the history, then live again; overlay and ground-truth toggles; screenshot, recording, replay of a recording, video; four benchmark scenarios; changing the seed; **the window keeps running while inference takes 150 ms**; the Qt start-up checks |
| `test_viewer_models.py` | 4 | with the trained YOLO11n and U-Net: three scenarios through the viewer match the benchmark, and the headless demonstration runs end to end |
| `test_viewer_humanoid.py` | 12 | Phase 4: frames carry the robot's belief (`FusionView`) apart from the truth; old recordings still open; **a humanoid session reproduces the benchmark arm bit for bit**; hiding ground truth does not change an estimate; body, mount, motion and estimator controls; a Phase 4 run records, exports and replays; the window's Phase 4 controls |

The window tests use Qt's `offscreen` platform and are skipped when PySide6 is not installed.
In addition the window was driven on the laptop's display with the trained models: three
scenarios, pause, step, scrub, replay, reset, speed changes, screenshot, recording, replay of
the recording and video export.

## Limits

- **Simulation only.** No ROS, no real camera, no physical robot. The humanoid is a posed
  visual model: it does not balance, and its stepping is an animation
  ([humanoid_integration.md](humanoid_integration.md)).
- **The oracle arm is not offered.** The viewer shows estimators a robot could run; the
  benchmark's orientation oracle uses ground truth and stays in the benchmark.
- **Benchmark scenarios only.** The robot follows the scenario's trajectory; it cannot be
  driven by hand, and nothing in the scene can be moved from the window.
- **One localizer.** Only one arm is shown at a time; to compare two camera-pose estimators,
  run the scenario twice (the runs are deterministic) or read the benchmark.
- **Live speed is bounded by inference.** About 1.7x real time on this laptop at 640x360, so
  "2x" and "4x" are reached only when replaying. At 1280x720 a step takes about 60 ms.
- **Stepping back is replay.** The history is shown again; the filter's state is not
  rewound. To change something and run again, use Reset.
- **No parameter editing.** Thresholds, noise models and the calibration are those of the
  configuration files; the viewer does not change them.
- **Recorded images are JPEG** (quality 90). Replayed pictures are slightly lossy; every
  number in a recording is exact. Particles are stored as the drawn subset.
- **The simulator's true camera is not drawn.** A mount error shows as observed lines that
  do not sit on the map's lines, not as a second camera.
- **Tested on one machine**: Ubuntu 24.04, X11, RTX 2050. Wayland and other platforms were
  not tried.
- The panels use OpenCV's built-in font (ASCII only), and video export pauses the run while
  it writes.
