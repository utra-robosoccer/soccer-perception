# Vision-localization validation (Phase 3)

What was checked, what was measured, what failed, and what has **not** been validated.
Phase 1's validation is in [validation.md](validation.md), Phase 2's in
[vision_validation.md](vision_validation.md).

Read this first: **every number in this document is synthetic.** The images were rendered by
this repository's simulator, and the detector and segmenter were trained on images from the
same simulator. The camera is an ideal pinhole on a kinematic platform over a perfectly flat
floor. The results show that the camera-to-pose pipeline is correct and how it degrades under
controlled errors. They are not evidence of how the robot will localize with a real ZED Mini
on a real field.

## Summary

Benchmark of 2026-11-17: 16 scenarios, 5 seeds each, 640x360 images, about 34,000 camera
frames, on an RTX 2050 laptop. Position / yaw RMSE over every step of every run:

| | Vision (YOLO11n + U-Net) | Oracle observations | Odometry only |
|---|---|---|---|
| Clean motion (`v01` to `v05`) | 2.0 to 4.0 cm / 0.31 to 0.64 deg | 2.0 to 2.9 cm / 0.26 to 0.48 deg | 0.37 to 0.94 m / 3.7 to 14.6 deg |
| Hard but calibrated (`v06`, `v07`, `v08`, `v11`, `v14`) | 1.7 to 4.4 cm / 0.24 to 0.66 deg | 1.8 to 4.2 cm / 0.24 to 0.61 deg | 0.53 to 2.2 m / 9.7 to 48 deg |
| Camera pitch wrong by 2 deg (`v09`) | **34 cm** / 2.4 deg | 1.8 cm / 0.27 deg | 0.79 m / 12.7 deg |
| Camera height wrong by 5 cm (`v10`) | 12 cm / 0.56 deg | 2.3 cm / 0.27 deg | 0.64 m / 8.9 deg |

What the numbers say:

1. **With a correct camera estimate, localizing from images is as accurate as localizing from
   oracle observations**: 2 to 4 cm and about 0.5 deg, in all ten calibrated scenarios, with
   every run of every seed converged and none diverged. That is at least 5 times inside the
   aspirational targets (0.20 m, 5 deg). Dead reckoning alone is 10 to 60 times worse.
2. **The trained networks are not the limiting factor in the simulator.** Replacing their
   predictions with rendered labels (`perfect_vision`) changes nothing measurable.
3. **Camera calibration is the limiting factor, and the dominant term is pitch.** A hidden
   2 deg pitch error costs 34 cm, which misses the target; the oracle arm, which does not
   project pixels, is unaffected. Position error grows by about 12 to 14 cm per degree of
   pitch error, about 2.5 cm per centimetre of height error and about 7 cm (plus 1.6 deg of
   yaw) per degree of roll error. To stay under 0.20 m the pitch has to be known to about
   1.5 deg, the height to about 7 cm and the roll to about 2 deg.
4. **With a calibration error the filter is confidently wrong.** In `v09`, on 24 to 40 %
   of the steps of each run it reports itself converged while its error exceeds 0.3 m. Its
   covariance describes particle spread, not a bias shared by every measurement.
5. **The field's 180-degree symmetry is not resolved by vision, and is not hidden.** From a
   global start the estimate is flagged ambiguous for the whole run and 1 of 5 vision runs
   (2 of 5 oracle runs) ends on the mirrored pose. A robot carried across the centre line
   re-localizes on the mirrored pose in 5 of 5 runs, for vision and oracle alike.
6. **It runs at 23 frames per second at 640x360** (44 ms: detector 8.5, segmenter 21.7,
   geometry 11.0, filter 2.4) and 17 at 1280x720, using 136 MB of GPU memory.

Nothing above has been tried on a real image.

## Automated tests

```bash
.venv/bin/python -m pytest                    # 408 tests, about 3 minutes
.venv/bin/python -m pytest -m "not slow"      # 379 tests
.venv/bin/python -m pytest -m "not vision and not render"   # 353 tests: no GPU, no OpenGL
```

The 276 tests of Phases 1 and 2 pass unmodified; no existing test or source file was edited.
Phase 3 adds 132:

| File | Tests | What it establishes |
|---|---|---|
| `test_ground_projection.py` | 31 | the estimated-camera chain equals Phase 1's camera model to 1e-12 for pan, pitch, roll and height; ray conventions; the optical axis meets the ground where trigonometry says; projection inverts the forward model; **every rejection reason** (above horizon, near horizon, too far, too close, image border, non-finite, camera below ground); **a pitch error moves ranges exactly as Phase 1's `apply_pitch_error` modelled**; a height error scales ranges; yaw and roll directions; intrinsics errors; IMU tilt; covariance against the closed form, its growth with range, its two parts, each parameter's contribution, finite near the horizon; only estimate types are accepted; **rendered images with an off-centre principal point back-project onto paint, and do not when `cx` is mirrored** |
| `test_feature_extraction.py` | 35 | configuration files equal the code defaults; centreline of a thick band, blob removal, input-image coordinates at 1280 px; minimum-distance thinning; segments split at gaps and merged when collinear; **a circle is found from a full circle and a 200 deg arc, refused from a 100 deg arc, and refused when laid tangent to straight lines**; L / T / X from where lines end; **an end without turf seen beyond it gives no class**; short stubs and conflicting classes give no class; X from circle and line; extracted features lie where the field says for six poses; uncertainty grows with range and knows the line's direction; a wrong camera estimate moves the points as predicted; empty and mismatched inputs; repeatability; **the foot of a leaning post is at the inner corner of its box**; bottom-cut posts give bearing only with the right bearing; hidden feet and implausible boxes get no range; weak and non-post boxes are rejected; post uncertainty follows range, confidence and occluders; **the frame produced is the same type as the oracle's and the Phase 1 filter consumes it unchanged**; bearing-only and unclassed features are mapped onto the schema, not added to it |
| `test_vision_localization.py` | 11 | **the pipeline tracks a walk from images and odometry alone** (final error under 10 cm from a 30 cm wrong start); vision corrects what dead reckoning cannot; reproducibility; **a wrong camera estimate biases the pose** (the pipeline cannot discover the true camera); a robot at the mirrored pose sees the same image; **from a global start both mirror hypotheses survive, the estimate is flagged ambiguous and never "converged"**, and `resolve_symmetry` settles it either way; an own-half prior breaks the symmetry and a wrong prior is believed; `step` rejects every other input type; `step`, `step_input` and `step_with_predictions` agree; lost odometry diffuses, the first frame does not; IMU tilt |
| `test_ground_truth_isolation.py` | 22 | every runtime module is free of simulator, oracle, dataset and benchmark imports; **the pipeline imports and localizes in an interpreter where `import mujoco` fails**; the signature of `step`; no boundary type can carry a global pose or an identity; ground-truth objects of nine kinds are rejected; true and estimated camera share no attribute name; the true camera follows commands plus hidden errors; **record and replay**: everything the benchmark gives the vision arm during a run with a hidden mount error is an image, a number, an `OdometryDelta` and the NOMINAL camera estimate, and replaying it into a fresh pipeline without a simulator reproduces the estimates bit for bit; the simulator refuses estimates |
| `test_vision_benchmark.py` | 29 | all 14 required scenarios exist and change what they claim; trajectories are reproducible, differ by seed, stay on the carpet; the damage model; feature scoring on constructed cases (true, wrong class, false, unclassed; posts; circle; recall) and zero error on exact features; **a mount error and the equal-and-opposite estimate error are the same mismatch and give the same localization error**; sweep arms; a run gives every arm the same world; reproducibility; a hidden mount error hurts image-fed arms and not the oracle; the benchmark and the sweep write their files |
| `test_vision_localization_models.py` | 4 | with the **trained** YOLO11n and U-Net: one frame from pixels to a pose update, stage by stage; feature accuracy on 40 random views; the vision arm localizes about as well as the oracle on a clean walk; the pipeline loads from checkpoints and steps at 1280x720. Skipped when the checkpoints are missing |

Tests that render need an OpenGL context and are skipped without one. Only the last file needs
the trained checkpoints and a GPU-capable PyTorch; the others use analytically painted views
of the field and a colour-threshold stand-in for the networks (`tests/phase3_helpers.py`), so
the geometry and the whole pipeline are tested deterministically in milliseconds per frame.

## How the benchmark was run

```bash
.venv/bin/python scripts/benchmark_vision_localization.py      # about 1 h 50 min on the laptop
.venv/bin/python scripts/evaluate_features.py --frames 400     # and the variants named below
```

- **Arms.** Every frame is rendered once and given to four independent localizers that share
  the trajectory, the odometry stream, the initial guess and the filter seed:
  `vision` (trained YOLO11n + U-Net on the image), `oracle` (Phase 1's simulated observations),
  `odometry` (no observations), `perfect_vision` (rendered labels through the same geometry;
  a diagnostic that separates network error from geometry error).
- **What the vision arm is given**: the image, the odometry increment, the camera intrinsics
  and the NOMINAL mount (0.90 m, 25 deg down, commanded pan). Never the true camera pose.
  The simulated camera differs from that estimate by a head jitter of 0.25 deg (1 sigma) in
  pitch and roll on every frame, and by the mount error of `v09` and `v10`.
- **Seeds.** Five per scenario. A seed changes the trajectory (start shift up to 0.2 m, turn
  up to 5 deg), the odometry noise, the initial guess, the filter's random stream, the head
  jitter and, in `v14`, the look of the scene.
- **Start.** A Gaussian guess of 0.3 m and 10 deg around the true pose, except in the two
  symmetry scenarios (global, and own half).
- **Localized** means a position error under 0.3 m and a yaw error under 10 deg. **Converged**
  means localized and still so 2 s later. **Tracking** statistics cover the steps after
  convergence; **all-step** statistics cover every step.
- The filter, its configuration and its measurement model are Phase 1's, unchanged.

## Oracle, vision and odometry

Position RMSE [m] / yaw RMSE [deg] over **every step** of the five runs:

| Scenario | Oracle | Vision | Perfect vision | Odometry only |
|---|---|---|---|---|
| `v01_stationary` | 0.029 / 0.26 | 0.040 / 0.64 | 0.040 / 0.59 | 0.432 / 3.71 |
| `v02_rotation` | 0.028 / 0.48 | 0.029 / 0.47 | 0.032 / 0.51 | 0.372 / 6.14 |
| `v03_straight_line` | 0.025 / 0.29 | 0.021 / 0.34 | 0.024 / 0.34 | 0.930 / 12.25 |
| `v04_rectangle` | 0.020 / 0.32 | 0.020 / 0.34 | 0.024 / 0.35 | 0.941 / 14.59 |
| `v05_curve` | 0.023 / 0.30 | 0.022 / 0.31 | 0.026 / 0.34 | 0.579 / 9.96 |
| `v06_partial_visibility` | 0.019 / 0.49 | 0.024 / 0.55 | 0.024 / 0.56 | 0.995 / 14.14 |
| `v07_noisy_odometry` | 0.042 / 0.61 | 0.044 / 0.66 | 0.046 / 0.65 | 2.207 / 47.54 |
| `v08_missing_false_features` | 0.024 / 0.47 | 0.018 / 0.32 | 0.017 / 0.22 | 0.963 / 16.37 |
| `v09_camera_pitch_error` | 0.018 / 0.27 | **0.342 / 2.41** | 0.355 / 2.52 | 0.793 / 12.71 |
| `v10_camera_height_error` | 0.023 / 0.27 | **0.121 / 0.56** | 0.115 / 0.56 | 0.635 / 8.92 |
| `v11_goal_post_occlusion` | 0.020 / 0.35 | 0.024 / 0.43 | 0.024 / 0.42 | 0.534 / 9.72 |
| `v12a_symmetry_no_prior` | 2.701 / 113.3 | 1.908 / 80.4 | 2.675 / 113.6 | 4.765 / 106.6 |
| `v12b_symmetry_own_half_prior` | 0.254 / 2.83 | 0.237 / 3.82 | 0.403 / 8.94 | 2.850 / 74.42 |
| `v13a_kidnapped_same_side` | 0.210 / 10.38 | 0.210 / 10.16 | 0.208 / 10.37 | 3.184 / 119.3 |
| `v13b_kidnapped_across_field` | 3.661 / 142.8 | 3.655 / 142.7 | 3.658 / 142.8 | 3.635 / 94.0 |
| `v14_strong_randomization` | 0.018 / 0.24 | 0.017 / 0.24 | 0.020 / 0.25 | 1.054 / 17.48 |

The all-step figures of `v12` and `v13` are dominated by what those scenarios do on purpose
(a global start, a kidnapping, mirrored endings); the next table and the sections on symmetry
and kidnapping are the ones to read for them. In `v08` the oracle column uses Phase 1's harsh
dropout and false-positive profile, so its arms are not damaged in the same way.

The vision arm in detail (tracking statistics are pooled over the steps after convergence):

| Scenario | Converged | Pos RMSE / MAE / P95 [m] | Yaw RMSE / MAE / P95 [deg] | Ended not localized | Ended mirrored |
|---|---|---|---|---|---|
| `v01_stationary` | 5 / 5 | 0.040 / 0.025 / 0.090 | 0.64 / 0.32 / 1.39 | 0 | 0 |
| `v02_rotation` | 5 / 5 | 0.029 / 0.017 / 0.059 | 0.47 / 0.28 / 0.93 | 0 | 0 |
| `v03_straight_line` | 5 / 5 | 0.021 / 0.018 / 0.041 | 0.34 / 0.28 / 0.61 | 0 | 0 |
| `v04_rectangle` | 5 / 5 | 0.020 / 0.016 / 0.035 | 0.34 / 0.25 / 0.64 | 0 | 0 |
| `v05_curve` | 5 / 5 | 0.022 / 0.017 / 0.037 | 0.31 / 0.23 / 0.59 | 0 | 0 |
| `v06_partial_visibility` | 5 / 5 | 0.020 / 0.014 / 0.037 | 0.49 / 0.30 / 0.75 | 0 | 0 |
| `v07_noisy_odometry` | 5 / 5 | 0.038 / 0.031 / 0.071 | 0.59 / 0.44 / 1.12 | 0 | 0 |
| `v08_missing_false_features` | 5 / 5 | 0.017 / 0.014 / 0.029 | 0.24 / 0.18 / 0.48 | 0 | 0 |
| `v09_camera_pitch_error` | 5 / 5 | 0.342 / 0.293 / 0.641 | 2.41 / 1.76 / 5.06 | 0 | 0 |
| `v10_camera_height_error` | 5 / 5 | 0.121 / 0.119 / 0.159 | 0.56 / 0.43 / 1.20 | 0 | 0 |
| `v11_goal_post_occlusion` | 5 / 5 | 0.024 / 0.015 / 0.053 | 0.43 / 0.28 / 0.99 | 0 | 0 |
| `v12a_symmetry_no_prior` | 4 / 5 | 0.058 / 0.029 / 0.142 | 0.87 / 0.45 / 1.89 | 1 | 1 |
| `v12b_symmetry_own_half_prior` | 5 / 5 | 0.063 / 0.033 / 0.173 | 1.07 / 0.52 / 2.30 | 0 | 0 |
| `v13a_kidnapped_same_side` | 5 / 5 | 0.033 / 0.019 / 0.072 | 0.43 / 0.28 / 0.76 | 0 | 0 |
| `v13b_kidnapped_across_field` | 5 / 5 | 0.049 / 0.031 / 0.115 | 0.67 / 0.41 / 1.59 | 5 | 5 |
| `v14_strong_randomization` | 5 / 5 | 0.017 / 0.014 / 0.030 | 0.24 / 0.18 / 0.46 | 0 | 0 |

Observations:

- **Vision equals the oracle wherever the camera estimate is right.** The differences between
  the two columns (a few millimetres either way) are smaller than the spread between seeds:
  per-seed position RMSE in `v04` is 1.5 to 2.5 cm for vision and 1.7 to 2.7 cm for the oracle.
- **The stationary scenario is the worst clean one** (4.0 cm, one seed at 7.8 cm against 1.1 to
  2.6 cm for the others). A robot that does not move sees the same features with the same
  errors on every frame, so nothing averages out; the oracle shows the same pattern (4.5 cm
  on its worst seed).
- **Looking 50 deg down (`v06`) costs nothing**, although the camera then sees 19 line points
  per frame instead of 46 and almost never a goal post. Near features are the accurate ones.
- **Perfect labels do not help.** `perfect_vision` is within 4 mm of `vision` in every
  calibrated scenario. In this simulator the networks' errors are below the geometry's.
- **`v09` is the failure of this phase** and is analysed under "Camera calibration" below.

## Geometric perception

### Random views

`scripts/evaluate_features.py`: 400 random robot poses on the field, nominal camera with the
0.25 deg head jitter, scored against the simulator.

| | Trained models | Rendered labels | Trained models, `strong` appearance |
|---|---|---|---|
| Line points per frame | 38.4 | 39.8 | 36.0 |
| Line-point error, mean / P95 | 2.1 / 11.2 cm | 2.2 / 11.5 cm | 4.7 / 14.5 cm |
| ... of points at 0-2 m / 2-4 m / beyond 4 m (mean) | 0.1 / 1.2 / 4.6 cm | 0.0 / 1.2 / 4.9 cm | 2.3 / 2.4 / 9.7 cm |
| Line points within 1 / 2 reported sigma | 99.3 / 99.96 % | 99.3 / 99.99 % | 96.4 / 97.1 % |
| Candidate pixels with a valid projection | 89.7 % | 91.4 % | 91.7 % |
| Goal posts: with range / bearing only / not a post | 344 / 56 / 0 | 347 / 56 / 0 | 291 / 98 / 1 |
| Post ground-point error, mean / median / P95 | 9.3 / 5.1 / 33 cm | 9.3 / 5.2 / 32 cm | 9.6 / 4.9 / 31 cm |
| Post bearing error, mean / P95 | 0.15 / 0.44 deg | 0.15 / 0.46 deg | 0.21 / 0.62 deg |
| Junctions reported, precision | 693, 99.7 % | 690, 99.6 % | 567, 98.1 % |
| Junction recall: point in view / arms in view | 70 % / 87 % | 70 % / 86 % | 57 % / 70 % |
| ... L / T / X reported (correct) | 307 (307) / 269 (267) / 117 (117) | 305 (305) / 270 (267) / 115 (115) | 240 (238) / 215 (210) / 112 (108) |
| Junction position error, mean | 3.4 cm | 3.4 cm | 3.2 cm |
| Centre circle found when at least 150 deg in view | 90 of 101 | 89 of 101 | 88 of 101 |
| False centre circles | 0 | 0 | 0 |
| Centre error, mean | 3.3 cm | 3.2 cm | 3.1 cm |
| Geometry time per frame | 10.3 ms | 10.4 ms | 10.1 ms |

- Line-point error is the distance from the reported point, moved to the field with the true
  pose, to the nearest paint; it is zero for a point that lands on paint, which is why the
  median is zero. The rejected pixels are those beyond the 6 m range limit (7 to 9 %) and
  those within 2 px of the image border (1.3 %).
- "Arms in view" recall counts only junctions whose every arm is visible for at least the
  length needed to decide its class; a corner whose second arm is outside the image cannot be
  told from a line end and is correctly left unclassed.
- The reported uncertainty is **conservative**: the mean reported sigma is 12 cm against a
  mean error of 2 cm. It is sized for a head whose pitch is known to 0.5 deg, and the
  simulated head is better than that.
- Under `strong` appearance 2.9 % of line points are more than two sigma from any paint
  (network false positives on bright scene elements), and recall falls. Localization does
  not notice (`v14`, 1.7 cm).

### During the benchmark runs

The same scoring over every frame of the benchmark (vision arm, five seeds pooled):

| Scenario | Line points / frame | Line error mean / P95 [m] | Posts: range / bearing only / not a post | Post ground error mean / P95 [m] | Junctions reported / precision | Junction recall (in view / arms in view) | Circle found / expected, wrong |
|---|---|---|---|---|---|---|---|
| `v01_stationary` | 54.2 | 0.037 / 0.164 | 2010 / 0 / 0 | 0.179 / 0.430 | 1945 / 100 % | 97 % / 97 % | 971 / 1005, 0 |
| `v02_rotation` | 48.8 | 0.022 / 0.105 | 1226 / 0 / 0 | 0.084 / 0.243 | 2680 / 100 % | 76 % / 90 % | 287 / 290, 2 |
| `v03_straight_line` | 48.1 | 0.023 / 0.114 | 2015 / 39 / 0 | 0.102 / 0.314 | 2631 / 100 % | 76 % / 87 % | 364 / 383, 3 |
| `v04_rectangle` | 45.6 | 0.021 / 0.105 | 3553 / 0 / 0 | 0.095 / 0.283 | 6119 / 100 % | 70 % / 85 % | 406 / 423, 7 |
| `v05_curve` | 47.2 | 0.022 / 0.108 | 1922 / 0 / 0 | 0.080 / 0.222 | 3904 / 100 % | 77 % / 86 % | 0 / 0, 9 |
| `v06_partial_visibility` | 19.3 | 0.002 / 0.015 | 149 / 0 / 1 | 0.027 / 0.059 | 1628 / 100 % | 43 % / 81 % | 263 / 263, 0 |
| `v07_noisy_odometry` | 45.5 | 0.022 / 0.111 | 3549 / 0 / 0 | 0.096 / 0.293 | 6074 / 100 % | 69 % / 84 % | 397 / 421, 10 |
| `v08_missing_false_features` | 40.2 | 0.049 / 0.236 | 1716 / 1249 / 1141 | 0.104 / 0.335 | 4528 / 95 % | 50 % / 60 % | 340 / 424, 11 |
| `v09_camera_pitch_error` | 46.7 | 0.240 / 0.743 | 2882 / 663 / 338 | 0.701 / 1.387 | 4563 / 66 % | 35 % / 42 % | 22 / 431, 137 |
| `v10_camera_height_error` | 45.2 | 0.095 / 0.257 | 3577 / 0 / 0 | 0.248 / 0.538 | 6544 / 99 % | 69 % / 83 % | 214 / 426, 6 |
| `v11_goal_post_occlusion` | 35.7 | 0.016 / 0.076 | 1503 / 838 / 4 | 0.111 / 0.418 | 2959 / 100 % | 64 % / 92 % | 157 / 164, 0 |
| `v12a_symmetry_no_prior` | 43.7 | 0.019 / 0.098 | 1408 / 0 / 0 | 0.086 / 0.316 | 3844 / 100 % | 78 % / 92 % | 304 / 312, 0 |
| `v12b_symmetry_own_half_prior` | 43.3 | 0.018 / 0.099 | 1410 / 0 / 0 | 0.083 / 0.285 | 3869 / 100 % | 79 % / 93 % | 311 / 316, 0 |
| `v13a_kidnapped_same_side` | 43.7 | 0.023 / 0.115 | 2289 / 38 / 0 | 0.106 / 0.346 | 4523 / 100 % | 76 % / 92 % | 672 / 697, 4 |
| `v13b_kidnapped_across_field` | 45.2 | 0.023 / 0.119 | 2395 / 3 / 0 | 0.092 / 0.308 | 4793 / 100 % | 79 % / 92 % | 617 / 639, 1 |
| `v14_strong_randomization` | 43.1 | 0.037 / 0.125 | 3367 / 155 / 17 | 0.102 / 0.325 | 5231 / 100 % | 60 % / 73 % | 410 / 428, 9 |

"100 %" junction precision is rounded: for example `v04` has 5 wrong classes or positions
among 6119, `v05` has 15 among 3904, all of them T junctions. "Wrong" circles are accepted
circles whose centre is more than 0.3 m from the centre mark.

- **A false centre circle is accepted once in 200 to 400 frames on some trajectories**
  (`v04`, `v05`, `v07`, `v14`: 7 to 10 in roughly 1900 to 2900 frames; none in the 400 random
  views). The three examined, in `v05` seed 1, were circles of the centre-circle radius laid
  through the corner region of the penalty and goal areas 4 to 5 m away, with 150 to
  165 deg of support: just over the acceptance threshold. Each one enters the filter as a
  centre-mark observation 0.3 m or more from where it should be. The filter's outlier model
  absorbed all of them (no effect on the error of those runs), but a false landmark is a
  false landmark, and the circle test should be tightened or range-limited before it is
  relied on. This is the one clean-scene false-feature mode that was found.
- **Posts beyond 6 m have 20 to 40 cm of range error** (`v01`: the stationary robot looks
  at a goal 6.4 to 6.8 m away; mean ground error 18 cm, P95 43 cm). That is one or two pixels at that
  distance, it is inside the reported sigma, and it is why far posts are down-weighted.

## Robustness

### Camera calibration

Calibration sweep on `v04_rectangle`, three seeds. The simulator's camera is untouched; the
robot is given a wrong **estimate**, one parameter at a time. A test confirms that this is
the same mismatch as a hidden mount error of the opposite sign
(`test_estimate_error_is_the_mirror_image_of_a_mount_error`).

| Pitch error [deg] | Pos RMSE [m] | Pos P95 [m] | Yaw RMSE [deg] | Ended not localized | Pos RMSE believing 2 deg [m] |
|---|---|---|---|---|---|
| -3 | 0.706 | 1.423 | 15.43 | 1 of 3 | 0.208 |
| -2 | 0.334 | 0.609 | 1.93 | 0 | 0.146 |
| -1 | 0.138 | 0.210 | 0.69 | 0 | 0.083 |
| -0.5 | 0.078 | 0.117 | 0.42 | 0 | 0.051 |
| 0 | 0.020 | 0.035 | 0.29 | 0 | 0.026 |
| +0.5 | 0.064 | 0.099 | 0.34 | 0 | 0.035 |
| +1 | 0.117 | 0.167 | 0.50 | 0 | 0.068 |
| +2 | 0.265 | 0.465 | 1.77 | 0 | 0.142 |
| +3 | 0.800 | 1.853 | 23.95 | 0 | 0.211 |

| Height error [m] | Pos RMSE [m] | Yaw RMSE [deg] | | Roll error [deg] | Pos RMSE [m] | Yaw RMSE [deg] |
|---|---|---|---|---|---|---|
| -0.10 | 0.270 | 1.22 | | -4 | 0.425 | 7.98 |
| -0.05 | 0.117 | 0.56 | | -2 | 0.144 | 2.99 |
| -0.02 | 0.048 | 0.29 | | -1 | 0.080 | 1.54 |
| 0 | 0.020 | 0.29 | | 0 | 0.020 | 0.29 |
| +0.02 | 0.060 | 0.31 | | +1 | 0.082 | 1.72 |
| +0.05 | 0.131 | 0.60 | | +2 | 0.146 | 3.25 |
| +0.10 | 0.266 | 1.35 | | +4 | 2.129 (1 of 3 lost) | 72.67 |

| Pan (yaw) error [deg] | Pos RMSE [m] | Yaw RMSE [deg] | | Focal-length error | Pos RMSE [m] | Yaw RMSE [deg] |
|---|---|---|---|---|---|---|
| -4 | 0.040 | 3.74 | | -5 % | 0.058 | 0.68 |
| -2 | 0.026 | 1.95 | | -2 % | 0.033 | 0.39 |
| -1 | 0.020 | 1.04 | | 0 | 0.020 | 0.29 |
| +1 | 0.023 | 0.85 | | +2 % | 0.024 | 0.28 |
| +2 | 0.029 | 1.77 | | +5 % | 0.045 | 0.50 |
| +4 | 0.046 | 3.53 | | | | |

Figure: `outputs/vision_localization/sweep/sensitivity.png`.

- **Pitch**: about 12 to 14 cm of position error per degree up to 2 deg, then a cliff. At
  3 deg the filter no longer tracks (0.7 to 0.8 m; at -3 deg one run in three ends lost). The cause is the
  geometry and nothing else: a pitch error moves a ground point by `(r^2 + h^2) / h` per
  radian, which is 9 cm per degree at 2 m and 33 cm per degree at 4 m, and it moves every
  point the same way, so the filter cannot average it out.
- **Height**: about 2.5 cm per centimetre. A height error scales every range by the same
  factor (5 cm is 5.6 %), which the filter partly resolves when it sees lines on both sides.
- **Roll**: 7 cm and 1.6 deg of heading per degree, and a cliff between 2 and 4 deg
  (asymmetric: at +4 deg one run in three is lost).
- **Pan**: a pan error becomes a heading error of nearly the same size and barely moves the
  position. It is indistinguishable from the robot being turned.
- **Focal length**: 5 % costs 5 to 6 cm.
- **To stay under 0.20 m and 5 deg** on this trajectory, by interpolation: pitch within about
  1.5 deg, height within about 7 cm, roll within about 2 deg, pan within about 4 deg, focal
  length within 5 %. These hold one at a time; combined errors were not swept.
- **Admitting the uncertainty helps.** Telling the pipeline that its pitch is uncertain to
  2 deg instead of 0.5 deg (last column) halves the error at 2 deg (0.14 m instead of 0.27 to
  0.33) and removes the cliff at 3 deg (0.21 m instead of 0.7 to 0.8), for 6 mm more error
  when the pitch is in fact right. The default was left at 0.5 deg so that the benchmark
  reports the unassisted behaviour; the number to use on the robot has to come from measuring
  its head.

The two calibration scenarios of the benchmark agree with the sweep. In
`v09_camera_pitch_error` (camera really 2 deg further down) the vision arm's line points are
24 cm from paint on average (39 cm beyond 4 m), 42 % of them more than two reported sigma
away, junction precision falls to 66 %, and the centre circle is found correctly in 22 of
431 frames and wrongly in 137. Localization: 0.34 m RMSE, 0.64 m at the 95th percentile,
every run "localized" at its end (final errors 14 to 16 cm) but far from it in between.
In `v10_camera_height_error` (camera really 5 cm higher) every feature is about 5.6 % too
near; localization is biased by a steady 12 cm.

Two properties of this failure matter for anything built on top:

1. **The filter does not know.** In `v09` it reports `converged` on 24 to 40 % of the steps
   while its error exceeds the localized gate. A consumer of the pose cannot use the reported
   covariance to detect a calibration bias.
2. **The known circle radius is a calibration alarm.** With the height wrong by 5 cm the
   centre circle is accepted in 48 of 101 random views instead of 90, and with the pitch
   wrong by 2 deg in 12 of 102, because the projected circle no longer has the radius the
   field says it has. That drop is measurable online and nothing uses it yet.

### Occlusion and truncation of goal posts

`v11_goal_post_occlusion`: the robot works in front of a goal with three robots standing at
the posts. Of 2341 post observations, 838 (36 %) were reduced to bearing only (foot hidden
by a robot, foot not on turf, cut by the image border) and 4 were not posts. Mean bearing
error 0.80 deg; ground error of the posts that kept a range 11 cm (P95 42 cm). Localization:
2.4 cm / 0.43 deg, the same as the oracle (2.0 cm / 0.35 deg).

### Missing and invented features

`v08_missing_false_features` damages the predictions before the geometry sees them: 10 % of
frames lost, half the detections missed, 0.5 invented posts per frame, mask patches erased
and false strokes painted. Of 2965 post observations 1141 (38 %) were invented ones; the
plausibility checks took the range away from 1249 observations but cannot remove a box that
looks like a post. Line points near the robot were on average 9 cm from paint (false
strokes), junction precision was 95 %, and 10 % of frames produced no observation at all.
Localization: 1.8 cm / 0.32 deg. The filter's outlier floor and tempering, unchanged from
Phase 1, are what absorb this.

### Appearance

`v14_strong_randomization`: a different turf, lighting, backdrop and ball for every seed, with
exposure, blur, noise and JPEG effects on every frame. 1.7 cm / 0.24 deg, the same as the
clean lap. Junction recall falls from 70 to 60 % and 17 of 3522 post observations are not
posts. This is appearance variation **inside the distribution the networks were trained on**;
it says nothing about real images.

### 180-degree symmetry

The field looks the same from a pose and from its half-turn mirror.

| | Vision | Oracle | Perfect vision |
|---|---|---|---|
| `v12a` global start, no prior: runs ending on the mirrored pose | 1 of 5 | 2 of 5 | 2 of 5 |
| `v12a` share of steps flagged `symmetric_ambiguity` | 100 % | 100 % | 100 % |
| `v12a` error of the runs that chose the true pose (tracking RMSE) | 5.8 cm / 0.87 deg | 2.8 cm / 0.58 deg | - |
| `v12b` own-half prior: runs ending on the mirrored pose | 0 of 5 | 0 of 5 | 0 of 5 |
| `v12b` time to converge | 2.1 s | 0.8 s | - |

Without outside information the choice between the two poses is a coin toss, as it must be;
five seeds cannot distinguish 1 in 5 from 2 in 5. What matters is that the estimate is
flagged ambiguous on every step and is never reported as converged. The own-half prior is
external game-state information; given a wrong prior the filter believes it (tested in
`test_vision_localization.py`).

### Kidnapping

| | Vision | Oracle |
|---|---|---|
| `v13a` moved within its side: recovered | 5 of 5, in 0.2 s | 5 of 5, in 0.5 s on average (worst 1.3 s) |
| `v13b` moved across the centre line: recovered the true pose | **0 of 5** | **0 of 5** |
| `v13b` settled on the mirrored pose | 5 of 5, in 0.3 s | 5 of 5, in 0.3 s |
| `v13b` share of steps reported converged while wrong | 63 % | 63 % |

`v13b` is an expected, unavoidable failure and the most dangerous one found: after being
carried across the field the robot is confidently wrong by a half turn, and nothing in the
images can reveal it. The oracle arm fails identically, so this is a property of the field
and the filter's recovery prior, not of vision. It needs outside information: a referee
signal, the team's side, a compass, or continuity across the pick-up.

## Performance

Measured inside `VisionLocalizationPipeline.step` on a laptop (NVIDIA RTX 2050 4 GB, 12 CPU
threads, PyTorch 2.15), single image, no batching, one process. Rendering is not included.

| Stage | 640x360: mean / median / P95 [ms] | 1280x720: mean [ms] |
|---|---|---|
| Detector (YOLO11n) | 8.5 / 8.5 / 9.3 | 11.1 |
| Segmenter (U-Net) | 21.7 / 21.3 / 23.4 | 32.5 |
| Geometric extraction (CPU, NumPy / OpenCV) | 11.0 / 11.5 / 15.0 | 11.5 |
| Particle filter (CPU) | 2.4 / 2.5 / 3.4 | 2.4 |
| **Whole step** | **43.9 / 44.6 / 49.1** | **57.7** |
| Frames per second at the mean | 22.8 | 17.3 |

- 640x360: 33,960 frames of the benchmark. The largest single step was 229 ms (the first
  inference after loading). 1280x720: one 581-frame run of `v04_rectangle`, where the
  accuracy was the same (1.6 cm / 0.34 deg).
- Peak GPU memory 136 MB allocated, 186 MB reserved. Process memory 1.9 GB, most of it
  PyTorch and the simulator.
- The networks take 70 % of the time and always run at their training size, so the larger
  image mostly costs resizing. The geometry works on a reduced mask and does not grow.
- These are desktop-GPU numbers. Nothing was measured on the robot's computer, and a 30 Hz
  camera would already be dropping frames here.

## Failure cases

| Failure | Where measured | Size | Status |
|---|---|---|---|
| Camera pitch estimate wrong | `v09`, sweep | 0.34 m at 2 deg; lost at 3 deg | open: needs calibration, or online pitch estimation |
| Filter confident while biased | `v09` | converged-but-wrong on 24 to 40 % of steps | open |
| Camera height estimate wrong | `v10`, sweep | 0.12 m at 5 cm | open: needs the real standing and walking height |
| Camera roll estimate wrong | sweep | 0.15 m and 3 deg at 2 deg; lost at 4 deg | open: needs the IMU in the loop |
| Mirrored pose after crossing the centre line | `v13b` | 5 of 5 runs, vision and oracle | inherent; needs outside information |
| Mirrored pose from a global start | `v12a` | 1 of 5 (vision), 2 of 5 (oracle) | inherent; flagged on every step |
| False centre circle at the penalty-area corner | `v04`, `v05`, `v07`, `v14` | 1 frame in 200 to 400 on those trajectories | open: tighten or range-limit the circle test |
| Wrong T junction | all | 0.1 to 0.4 % of reported junctions | accepted |
| Invented posts pass the plausibility checks | `v08` | 38 % of post observations when 0.5 per frame are injected | absorbed by the filter; no identity check exists |
| Far posts | `v01` | 18 cm on average, 43 cm at P95, at 6.4 to 6.8 m | inside the reported sigma |
| Stationary robot does not average its errors | `v01` | up to 7.8 cm on one seed | accepted |

No scenario produced a crash, a non-finite estimate or a run that the oracle arm survived and
the vision arm did not, except through a camera-calibration error.

## Against the brief's targets

| Target | Result |
|---|---|
| Position RMSE under 0.20 m in clean simulation | met: 2.0 to 4.0 cm (`v01` to `v05`), and 1.7 to 4.4 cm in the five harder calibrated scenarios |
| Yaw RMSE under 5 deg in clean simulation | met: 0.31 to 0.64 deg |
| Vision close to the oracle | met where the camera estimate is right; not met with a 2 deg pitch error (0.34 m against 0.02 m) |
| Better than odometry only | met by a factor of 10 or more in every scenario except `v13b`, where every arm ends on the wrong pose |
| Robust to moderate calibration error | **partly**: 5 cm of height and 1 deg of pitch stay under 0.20 m; 2 deg of pitch does not |
| Symmetry reported, not hidden | met: ambiguity flagged on every step; mirrored endings counted |
| Kidnapped recovery | met within a side (0.2 s); across the field it recovers the mirrored pose |
| No ground truth in the localizer | enforced by types and tests, including a record-and-replay of a benchmark run |

## What has not been validated

- **Real images.** Nothing here has seen a photograph. The detector and segmenter were trained
  and tested on the same renderer; their real-world accuracy is unknown, and it is the first
  thing that will change.
- **A real camera.** The model is an ideal pinhole: no lens distortion, no rolling shutter, no
  motion blur from a walking gait, no auto-exposure, no stereo rectification error. The ZED
  Mini was not used and its SDK is not a dependency.
- **A walking robot.** The camera rides a platform that glides over the floor with 0.25 deg of
  head jitter. A humanoid's head pitches and rolls by degrees at every step, and the sweep
  shows that 2 deg of unmodelled pitch is already too much. The IMU path
  (`imu_roll`, `imu_pitch`) exists and is tested for geometry, but no scenario exercises it
  with realistic gait motion or IMU error.
- **Non-flat ground**, a field of different dimensions, worn or curved lines, lines of another
  colour, a second field in view.
- **Latency between image and odometry.** Both are perfectly synchronised here.
- **Other robots as moving occluders** beyond the three static ones of `v11`.
- **Combined calibration errors**, and any online estimation of them.
- **Statistics.** Five seeds per scenario and three per sweep point show the size of an
  effect, not its confidence interval; rates such as "1 of 5" are anecdotes.
- **The robot's computer.** All timings are from one laptop GPU.

## Where the outputs are

Everything is under `outputs/vision_localization/` and is not committed.

| Path | Content |
|---|---|
| `benchmark/summary.md`, `summary.csv`, `runs.csv`, `results.json` | every arm of every scenario; every run; all metrics |
| `benchmark/features.csv` | geometric-perception scores per scenario and arm |
| `benchmark/timeseries/` | per-step errors of every run |
| `benchmark/plots/` | `overview_position_rmse.png`, `overview_yaw_rmse.png`, and per scenario: arms compared, trajectory, particles |
| `benchmark/frames/<scenario>/` | for three steps: the image, the predictions, the features on the image, the features on the ground with true paint, the particles |
| `sweep/sweep.json`, `sweep.csv`, `sensitivity.png` | the calibration sweep |
| `features/*.json` | the random-view evaluations |
| `runs/<scenario>_seed<k>/` | single demonstration runs (`v04_rectangle`, `v04_rectangle` at 1280x720, `v11_goal_post_occlusion`) |
