# Validation

What has been checked, what the numbers are, what failed, and what has not been validated.
Regenerate everything here with `pytest` and `scripts/benchmark.py`.

This document covers **Phase 1** (localization on oracle observations) and is unchanged by
Phase 2 except for this note. The vision pipeline's tests and results are in
[vision_validation.md](vision_validation.md). The 172 tests and the benchmark below were
re-run after Phase 2 was added; see that document for the outcome.

## Automated tests

172 tests, all passing on the development machine (Ubuntu, Python 3.12.3, MuJoCo 3.4.0,
NumPy 2.5.5, `MUJOCO_GL=glfw`), in about one minute.

| File | Tests | What it establishes |
|---|---|---|
| `test_field_geometry.py` | 49 | config values equal the rulebook numbers; both edges of every line type under the outside-of-line convention; circle radius; marks; corner arcs only on M/L; goals; symmetry; landmark counts and positions; distance map accuracy, memory, regeneration on change; field switching; rejection of invalid configs |
| `test_transforms.py` | 21 | angle wrapping; 90/180 degree cases; inverses and round trips in SE(2) and SE(3); right-handedness; optical versus MuJoCo camera frame; pitch/pan directions; stereo geometry; camera-to-field projection |
| `test_stereo_camera.py` | 16 | disparity versus baseline; scene contents; M and L fields compile; teleport to arbitrary poses; MuJoCo ground truth equals the analytic chain; rig controls; scripted trajectories; **7 rendering tests**: stereo frames, rendered ball versus projection (3 positions, both eyes), asymmetric intrinsics, rendered lines at projected landmarks, overview versus paint mask |
| `test_observations.py` | 19 | observation types carry no identity or global pose; noise-free observations are exact; field of view, range limits, head pan; mirrored poses give identical observations; noise statistics; dropout, false positives, outliers, class confusion; occlusion; odometry noise, drift, lost updates, kidnap blindness; motion model |
| `test_particle_filter.py` | 29 | initialization modes; likelihood prefers the true pose and is identical at the mirrored pose; ambiguous correspondences; bounded outlier influence; systematic resampling; ESS; resampling only when needed; cluster-aware estimate; tracking; global localization; symmetry handling; kidnapped-robot recovery; missing inputs; determinism |
| `test_benchmarks.py` | 38 | all scenarios exist and are deterministic; metric definitions on synthetic traces; reproducibility; the ground-truth firewall (record and replay); a tracking run meets the targets; kidnap recovery; a confidently wrong filter is flagged; M and L runs; MuJoCo-driven run equals the kinematic run; result files |

The 7 rendering tests need an OpenGL context and are skipped without one. On the development
machine only MuJoCo's `glfw` backend works (it needs a display); `egl` and `osmesa` fail there,
as already noted in robosoccer-rl.

## Benchmark

Run with `scripts/benchmark.py` on 2026-11-16: 17 scenario variants, seeds 0-9 each (170
runs), 1000 particles, 10 Hz updates, the `nominal` oracle profile unless stated. Full output:
`outputs/benchmark/` (`results.json`, `summary.md`, `summary.csv`, `runs.csv`, `timeseries/`,
`plots/`).

Terms: **localized** = position error < 0.30 m and yaw error < 10 deg. **Conv.** = share of
runs that became localized and stayed so for 2 s. **Conv. (sym.)** = the same, accepting
either the true pose or its half-turn mirror image. RMSE and P95 are pooled over all seeds,
over the steps after convergence. **Wrong half** = share of runs that end on the mirrored pose.

| Scenario | Field | Conv. | Conv. (sym.) | Pos RMSE [m] | Pos P95 [m] | Yaw RMSE [deg] | Yaw P95 [deg] | Wrong half | Kidnap recovery | Update [ms] |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 stationary, known pose | S | 100 % | 100 % | 0.022 | 0.039 | 0.19 | 0.40 | 0 % | | 6.2 |
| s02 rotation in place | S | 100 % | 100 % | 0.023 | 0.040 | 0.38 | 0.64 | 0 % | | 5.4 |
| s03 straight line | S | 100 % | 100 % | 0.034 | 0.053 | 0.37 | 0.51 | 0 % | | 3.3 |
| s04 rectangle | S | 100 % | 100 % | 0.017 | 0.031 | 0.24 | 0.46 | 0 % | | 3.1 |
| s05 circle | S | 100 % | 100 % | 0.030 | 0.042 | 0.46 | 0.61 | 0 % | | 3.1 |
| s06 random start, global init | S | 30 % | 100 % | 0.048 | 0.105 | 0.65 | 1.28 | **70 %** | | 3.2 |
| s07 partial visibility (2.5 m range) | S | 100 % | 100 % | 0.017 | 0.032 | 0.46 | 0.91 | 0 % | | 2.0 |
| s08 noisy odometry, 15 % lost | S | 100 % | 100 % | 0.027 | 0.050 | 0.45 | 0.88 | 0 % | | 3.2 |
| s09 dropout and false positives | S | 100 % | 100 % | 0.023 | 0.038 | 0.32 | 0.57 | 0 % | | 2.5 |
| s10a kidnapped, same side | S | 100 % | 100 % | 0.039 | 0.068 | 0.49 | 0.74 | 0 % | 0.4 s, 10/10 | 3.1 |
| s10b kidnapped, across the field | S | 100 % | 100 % | 0.029 | 0.057 | 0.32 | 0.64 | **100 %** | mirror only, 0.4 s | 3.0 |
| s11a symmetry, no prior | S | 40 % | 100 % | 0.050 | 0.129 | 0.67 | 1.56 | **60 %** | | 3.2 |
| s11b symmetry, own-half prior | S | 100 % | 100 % | 0.056 | 0.146 | 0.84 | 1.95 | 0 % | | 2.9 |
| s11c symmetry, wrong prior | S | 0 % | 100 % | - | - | - | - | **100 %** | | 2.9 |
| s12 field switching | S | 100 % | 100 % | 0.022 | 0.038 | 0.32 | 0.60 | 0 % | | 3.0 |
| s12 field switching | M | 100 % | 100 % | 0.032 | 0.060 | 0.36 | 0.63 | 0 % | | 2.8 |
| s12 field switching | L | 100 % | 100 % | 0.079 | 0.166 | 0.55 | 1.10 | 0 % | | 2.3 |

### Against the engineering targets

The targets for clean, observable simulation are position RMSE < 0.15 m and yaw RMSE < 3 deg.
Every scenario in which the filter has valid side information meets them, under the `nominal`
(noisier than `clean`) profile: worst case 0.079 m and 0.99 deg (L field; own-half prior).
That includes the degraded-sensing scenarios s07, s08 and s09 and all three field sizes.

### Failures, all expected, all reported by the benchmark

- **s06, s11a (no side information).** The filter localizes up to the field's symmetry in
  10/10 runs (after 2.5 s and 0.8 s on average), keeps both mirror-image hypotheses, flags
  `symmetric_ambiguity` on 100 % of steps and never reports `converged`. The pose it reports is
  one of the two; it is the mirrored one in 7/10 and 6/10 runs. This is a property of the
  field, not a defect that tuning can remove. `ParticleFilter.resolve_symmetry` is the hook
  for outside information.
- **s10b (kidnapped across the field centre).** The filter re-localizes within 0.4 s, but onto
  the **mirrored** pose in 10/10 runs, because the same-side prior assumes a short move. It
  then reports `converged` while wrong (63 % of all steps in that scenario). The filter cannot
  detect this; the benchmark does.
- **s11c (initialized at the mirrored pose).** Confidently wrong for the whole run, in 10/10
  runs, with `converged` reported on 100 % of steps.

### Smaller findings

- **Premature `converged`.** With the own-half prior (s11b) the filter reports convergence
  while not yet localized on 3.0 % of steps, during the first seconds. Elsewhere this is below
  0.7 %. Treat `converged` as "the particles agree", not "the pose is right".
- **Kidnap recovery (s10a)** took 0.2 to 0.8 s over ten runs. That speed comes from sensor
  resetting with clean landmark classes; expect it to be slower with real detections.
- **L field** accuracy is about three times worse than S. Much of a 22 m field is beyond the
  8 m perception range from any given point, and range noise grows with the square of range.
- **Particle count is not scaled with field size.** Global initialization on M and L was not
  benchmarked; 1000 particles are thin for a 384 m^2 carpet.
- **Resampling** happened on 17 to 44 % of updates depending on the scenario; mean effective
  sample size stayed between 53 % and 68 % of N.

### Held-out seeds

Parameters were tuned while looking at seeds 0-4. Seeds 5-9 were first run afterwards:

| Scenario group | Pos RMSE seeds 0-4 [m] | Pos RMSE seeds 5-9 [m] |
|---|---|---|
| s01-s05 (worst of the five) | 0.024 | 0.042 |
| s07-s09 (worst of the three) | 0.027 | 0.027 |
| s10a | 0.038 | 0.039 |
| s11b | 0.066 | 0.044 |
| s12 L field | 0.084 | 0.073 |

No scenario changed its convergence, divergence or wrong-half outcome between the two groups
except through the symmetry coin flip in s06 and s11a.

### Run time

Predict + update + estimate takes 2 to 6 ms on average per cycle (95th percentile below 7 ms,
maximum 10.9 ms over 170 runs) on a laptop x86-64 CPU, single-threaded NumPy, 1000 particles.
The whole benchmark takes about 7 minutes. Latencies were measured while other processes were
running and vary by about a factor of two between runs.

### What tuning was needed, and why

Three failure modes appeared during development and were fixed in the filter, not hidden in
the metrics:

1. **Aliased recovery hypotheses.** With few landmarks in view, freshly injected particles at a
   look-alike pose could take over the estimate for a few seconds (seen under heavy false
   positives and on the L field). Fix: injected particles start with a tenth of a survivor's
   weight, and injection needs a sustained drop in agreement.
2. **Side prior lost during convergence.** With an own-half prior, a brief false convergence
   could move the side anchor and let the filter recover onto the mirrored pose. Fix: the
   anchor follows the estimate only after 2 s of continuous convergence.
3. **Confident coin flip.** With no prior, resampling noise picked one of the two mirrored
   poses and the filter then reported convergence. Fix: keep the particle set symmetric until
   outside information arrives.

## Acceptance checklist

| # | Criterion | Evidence |
|---|---|---|
| 1 | S field renders correctly | `test_overview_render_matches_field_geometry` (IoU > 0.95 against the analytic paint mask); `outputs/render/s_field_overview.png` |
| 2 | M and L load without code changes | `test_other_fields_compile_without_code_changes`, benchmark `s12_field_m/l` |
| 3 | Valid goals, posts, lines, centre circle | `test_scene_contents`, `test_goals`, `test_center_circle_radius` |
| 4 | Platform teleports to arbitrary poses | `test_teleport_to_arbitrary_poses_and_ground_truth` (25 random poses and rig states) |
| 5 | Scripted trajectories without RL | `test_scripted_trajectory_is_followed_without_any_policy` |
| 6 | Left/right stereo images | `test_stereo_frames`; `outputs/render/s_field_stereo_{left,right}.png` |
| 7 | Baseline and camera transforms verified | `test_stereo_relative_geometry`, `test_rendered_ball_matches_the_projection_model` |
| 8 | Ground-truth pose available to evaluation | `GroundTruth`, `MujocoScene.ground_truth`, `test_mujoco_ground_truth_gives_the_same_run` |
| 9 | Oracle observations | `test_noise_free_observations_are_exact_and_in_the_base_frame` |
| 10 | Configurable noise and dropout | `test_measurement_noise_grows_with_range`, `test_dropout_false_positives_and_outliers` |
| 11 | Noisy synthetic odometry | `test_noisy_odometry_drifts_and_is_reproducible`, `test_missed_odometry_updates` |
| 12 | Particle filter estimates x, y, yaw | benchmark s01-s05 |
| 13 | Initialization from a known approximate pose | `test_gaussian_initialisation`, `test_tracking_from_an_approximate_pose` |
| 14 | Global initialization | `test_global_localisation_converges_up_to_symmetry`, benchmark s06 |
| 15 | Ambiguous hypotheses represented | `test_two_symmetric_modes_are_not_averaged`, `test_global_initialisation_represents_both_symmetric_hypotheses` |
| 16 | Benchmark detects errors and failures | `test_the_benchmark_detects_a_confidently_wrong_filter`, s10b and s11c above |
| 17 | A scenario converges | `test_a_tracking_scenario_converges_accurately` |
| 18 | Kidnapped-robot recovery tested | `test_kidnapped_robot_recovers_without_ground_truth`, `test_kidnapped_scenario_recovers` |
| 19 | Particle-cloud visualization | `outputs/benchmark/plots/*_particles.png` |
| 20 | Machine-readable results | `outputs/benchmark/results.json`, `summary.csv`, `runs.csv`, `timeseries/*.csv` |
| 21 | Automated tests pass | 172 passed |
| 22 | README has exact commands | [README.md](../README.md) |
| 23 | Existing repositories unmodified | checked with `git --no-optional-locks status` and file modification times: no file in either repository changed during the work |
| 24 | No training, no ROS integration | none added; dependencies are NumPy, PyYAML, MuJoCo, Matplotlib, pytest |

## Limitations

### Simulator

- **Kinematic platform.** The camera is placed, not carried by a walking robot. There is no
  head shake, no body sway beyond the oracle's per-frame pitch jitter, no falls.
- **Placeholder robot.** The Unitree G1 is not in the scene; the platform is a capsule. The
  robot's own body never appears in or blocks its camera image.
- **Clean scene.** Procedural turf, two directional lights, no spectators, referees, other
  robots or field-side clutter. Lines are perfectly white and never worn or occluded by turf.
- **Timing.** One observation and one odometry step per 0.1 s, perfectly synchronized, no
  latency. The stereo camera's 30 Hz clock is not exercised by the benchmarks.

### Oracle perception

- **It is not perception.** No image is processed during localization benchmarks.
  Observations come from geometry plus a noise model.
- **The noise model is a guess.** Range noise `0.02 + 0.01 r^2` m, bearing noise 1 deg, 10 %
  dropout and the rest are plausible numbers, not measurements. Real detectors have errors
  that are correlated in time, depend on viewing angle and lighting, and fail in clusters.
- **Errors are independent between frames.** A real detector that misclassifies a junction
  tends to do so for many consecutive frames. The filter's robustness
  to independent outliers will overstate its robustness to persistent ones.
- **Line points are sampled uniformly along lines in metric space**, not uniformly in the
  image, and false line points are spread uniformly over the view sector.
- **False landmarks are uniformly distributed**; real false positives cluster on things that
  look like landmarks (a robot's white leg, a goal net, a line seen through a goal).
- **Occlusion** covers goal posts, the ball and explicitly supplied cylinders only.
- **Junctions have no orientation** and marks are points; a real extractor can provide more.

### Camera

- Intrinsics are an ideal pinhole chosen to match the datasheet field of view. No factory or
  measured calibration is used; distortion, stereo misalignment and rolling shutter are absent.
- The mount (0.90 m, 25 deg down) is a placeholder. Accuracy and visibility both depend
  strongly on camera height and pitch, so every number below will move with the real robot.

### Localizer

See "Limits" in [localization.md](localization.md). The ones that matter most:

- The field's half-turn symmetry is unresolvable without outside information.
- `converged` reflects particle concentration, not correctness.
- Parameters were tuned by looking at benchmark seeds 0-4. Seeds 5-9 were first run after
  tuning stopped; they are a small held-out check, not proof of generality.

### Not validated at all

- Anything on a real image or a real robot.
- Localization from rendered images (the stereo renders are verified geometrically but are not
  yet consumed by anything).
- Odometry from a walking humanoid, which slips and drifts differently from this model.
- Run time on the robot's computer (latency numbers are for a laptop CPU, pure NumPy).
- Fields with rulebook values that were assumed here (goal height, post diameter, mark size).

## Expected sim-to-real issues

1. **Ground-plane projection error.** Landmark ranges come from projecting pixels onto the
   ground with the camera pose. A humanoid's head pose while walking is uncertain by degrees,
   not the 0.25 deg assumed here; expect range errors several times larger and strongly
   correlated across all detections in a frame. Stereo depth can cross-check this for near
   features.
2. **Line detection quality.** Real masks include goal posts, robots, the ball and reflections.
   The likelihood's outlier floor helps, but structured false lines are harder than random ones.
3. **Landmark misclassification that persists** across frames.
4. **Odometry.** Leg odometry of a walking humanoid can be off by tens of percent and has
   heading drift on every step; kicks and falls are kidnappings.
5. **Symmetry in play.** After a fall or a manual placement the same-side assumption can be
   wrong. Game state (kick-off side, penalties) and teammates are the usual way out.
6. **Lighting and appearance.** A detector trained only on these renders will not transfer.

## Next phase

Recommended order. None of this is implemented.

1. **Dataset generator from the simulator.** The scene already yields images, camera poses and
   landmark geometry, so labels are free: project `LandmarkMap`, the goals and the ball into
   each image for boxes and keypoints, and rasterize `FieldGeometry` through the camera for
   line masks. Add domain randomization (turf colour, lighting, line wear, distractors, camera
   pose jitter) to `configs/scenes/`. Add other robots as scene objects.
2. **YOLO baseline for ball, goal posts and robots.** Train a small detector on generated plus
   real images (the team's and public RoboCup sets). Evaluate on real images only. Deliverable:
   per-class boxes with scores.
3. **Field and field-line segmentation.** A small segmentation network (or a classical
   green/white pipeline as a baseline) producing field and line masks. Measure against
   rasterized ground truth in simulation and hand-labelled real frames.
4. **Geometric feature extraction.** From the line mask: sub-sample line pixels; fit lines and
   the centre circle; classify junctions as L / T / X with orientation. From detections:
   post-base and mark pixels.
5. **Real-image measurement generation.** Back-project those pixels to the ground plane with
   `StereoCameraModel` and the calibrated mount, refine near ranges with stereo disparity, and
   fill `ObservationFrame`. Then fit `configs/observations/*.yaml` to the measured error of
   this pipeline, so the oracle imitates the real perception stack.
6. **Integration with this particle filter.** Run the benchmarks with the image-based pipeline
   in place of the oracle (the scene renders, the pipeline observes, the filter and the metrics
   are unchanged). Then add what the real data shows is needed: junction orientation in the
   likelihood, negative information, detector confidence, and a symmetry-breaking input from
   game state.
7. **ROS 2 wrapper**, after the above works offline: a node around `ParticleFilter.step`
   publishing `map -> odom`.
