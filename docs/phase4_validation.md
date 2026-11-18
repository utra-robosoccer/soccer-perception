# Phase 4 validation: localization from a camera that moves

What was measured, what it shows and what it does not. Methods are in
[humanoid_fusion.md](humanoid_fusion.md); the model and the mount in
[humanoid_integration.md](humanoid_integration.md) and [camera_mount.md](camera_mount.md).

**Everything here is synthetic.** The images are rendered by this repository, the networks were
trained on images from the same renderer, the body motion is a set of sinusoids and pulses
that were never fitted to a robot, the IMU is simulated, and every camera mount is an
assumption because the robot has no head. The numbers say how the pieces behave together in
that world. They are not evidence about the real robot, a real ZED Mini or a real IMU.

## Summary

| Question | Answer from the benchmark |
|---|---|
| Does the actual robot model load in the perception simulator? | Yes: 21 link meshes (307 874 triangles), the full joint tree, 0.8 s to build, no measurable rendering cost. As a posed, visual-only body |
| What does body motion do to Phase 3's fixed camera model? | It breaks it. Over the eight motion scenarios the fixed camera averages 0.98 m position error; walking a rectangle it is lost in one run of three, turning in two of three, with an unsteady gait in all three |
| Does IMU-assisted orientation fix that? | Mostly. The same eight scenarios average 0.110 m with the IMU (C) and 0.055 m with the IMU and uncertainty-aware projection (D), against 0.027 m for the orientation oracle. No IMU arm lost localization in any motion scenario |
| How good is the roll / pitch estimate? | 0.2 to 0.5 degrees RMS standing and swaying, 0.4 to 0.8 walking, about 1 degree with an unsteady gait or 35 % of the IMU samples missing. The fixed model is wrong by 2 to 4 degrees in the same runs |
| Is the reported uncertainty honest? | The tilt's is (error / reported std between 0.4 and 1.2). The localizer's is not with C: its error is 2 to 5 times what it reports. With D the ratio is 0.5 to 1.5 in every motion scenario |
| Can a wrong mount be detected without ground truth? | A 2 degree pitch error: yes, from the field lines; the bounded correction recovers 1.7 to 2.3 degrees and brings the error from 0.58 m (C) or 0.11 m (D) to 0.07 m. The IMU alone cannot see it |
| What does it cost? | 2.2 ms per frame for the IMU chain; the whole pipeline takes 46 ms (21.6 frames per second) at 640 x 360, against 44 ms in Phase 3 |
| What is still weak? | Standing still nothing is gained. Pure lateral sway is slightly worse with C than with the fixed camera. A height error is absorbed as pitch, not identified. With the IMU arriving later than the image the orientation is mostly held, not extrapolated. Nothing has been checked on hardware |

## How it was measured

`scripts/benchmark_humanoid_fusion.py`: 18 scenarios, seeds 0 to 2, six arms, 640 x 360, the
trained YOLO11n and U-Net of Phase 2, the Phase 1 particle filter with its Phase 3 settings.
52 minutes for the benchmark and about 35 for the two sweeps on the RTX 2050 laptop. Results:
`outputs/humanoid_fusion/` (not tracked by Git; regenerate with the script).

Within a run every arm receives the **same** rendered images, IMU samples, odometry increments,
initial guess and filter seed. The networks run once per frame and all arms use those
predictions. The arms differ only in the camera pose they project with:

| Arm | Camera pose | Ground truth |
|---|---|---|
| A `fixed` | the believed mount on an upright, level body, every frame (Phase 3) | none |
| B `oracle_orientation` | the **true** camera orientation at the believed camera position | **yes**: an upper bound, built by the evaluator, labelled in every table |
| C `imu` | IMU roll / pitch interpolated to the image timestamp, believed mount | none |
| D `imu_uncertainty` | as C, with per-frame uncertainty and the shared-error model | none |
| `imu_latest` | as C, but the newest IMU estimate instead of interpolation | none |
| `imu_selfcal` | as D, with the mount monitor's bounded correction | none |

Unless a scenario says otherwise: `humanoid_prototype_1`, mount `head_mid` (camera 0.799 m
above the ground), the nominal IMU, an image that reaches the pipeline 30 ms after exposure,
0.1 degree of unreported mount flex per frame, a first guess 0.3 m and 10 degrees (1 sigma)
off, and the robot standing for 2 s before it walks.

Position and yaw RMSE cover the **whole run, including the first seconds** in which the filter
converges from its guess. That is deliberately harsh: an arm that loses the pose is not rescued
by a tracking window. It also means the numbers are not directly comparable with Phase 3's
"2 to 4 cm", which was measured over the tracking window on a rigid platform 0.90 m high; the
tracking-window values of arm D here are 3 to 8 cm (below). Camera, projection and consistency
statistics leave out the first 3 s. With three seeds, differences of a centimetre or two
between arms are within the spread and are not interpreted.

## Localization: fixed camera against IMU-assisted (deliverable E)

Position RMSE over the whole run, metres, three seeds pooled. Bold: the best arm that uses no
ground truth.

| Scenario | A fixed | B oracle | C imu | D imu + unc. | D + self-cal. | C, newest sample |
|---|---|---|---|---|---|---|
| `h01_standing` | 0.088 | 0.082 | 0.072 | 0.089 | 0.087 | **0.069** |
| `h02_sway_pitch` | 1.612 | 0.037 | 0.148 | 0.062 | **0.058** | 0.124 |
| `h03_sway_roll` | 0.068 | 0.020 | 0.082 | 0.059 | **0.048** | 0.083 |
| `h04_walk_straight` | 0.356 | 0.043 | 0.160 | 0.083 | **0.078** | 0.110 |
| `h05_walk_rectangle` | 0.737 | 0.018 | 0.069 | 0.034 | **0.030** | 0.068 |
| `h06_turning` | 1.093 | 0.021 | 0.060 | **0.031** | **0.031** | 0.060 |
| `h07_disturbances` | 0.193 | 0.040 | 0.083 | 0.075 | **0.074** | 0.083 |
| `h08_walk_disturbed` | 0.568 | 0.021 | 0.139 | 0.054 | **0.046** | 0.117 |
| `h09_walk_rough` | 3.245 | 0.018 | 0.141 | 0.045 | **0.037** | 0.131 |
| `h10_mount_pitch_error` | 2.501 | 0.018 | 0.581 | 0.112 | **0.069** | 0.443 |
| `h11_mount_roll_height_error` | 0.215 | 0.091 | 0.173 | 0.082 | **0.055** | 0.177 |
| `h12_head_low` | 0.675 | 0.019 | 0.083 | **0.041** | 0.043 | 0.085 |
| `h13_head_high` | 0.309 | 0.015 | 0.099 | 0.048 | **0.035** | 0.098 |
| `h14_head_tall` | 0.368 | 0.022 | 0.084 | 0.044 | **0.039** | 0.083 |
| `h15_imu_dropouts` | 0.706 | 0.021 | 0.166 | **0.051** | **0.051** | 0.157 |
| `h16_timing_offset` | 1.899 | 0.022 | 0.095 | 0.046 | **0.041** | 0.117 |
| `h17_imu_late` | 0.437 | 0.023 | 0.074 | **0.047** | 0.049 | 0.072 |
| `h18_placeholder_walk` | 0.321 | 0.023 | 0.100 | **0.045** | 0.047 | 0.105 |
| mean, `h02` to `h09` (motion, correct mount) | 0.984 | 0.027 | 0.110 | 0.055 | 0.050 | 0.097 |

Yaw RMSE, degrees:

| Scenario | A fixed | B oracle | C imu | D imu + unc. | D + self-cal. |
|---|---|---|---|---|---|
| `h01_standing` | 1.18 | 1.15 | 0.82 | 1.28 | 1.29 |
| `h02_sway_pitch` | 50.43 | 0.67 | 1.94 | 1.47 | 1.47 |
| `h03_sway_roll` | 0.99 | 0.24 | 0.42 | 0.52 | 0.49 |
| `h04_walk_straight` | 6.12 | 0.64 | 1.42 | 0.94 | 0.93 |
| `h05_walk_rectangle` | 19.47 | 0.30 | 0.76 | 0.65 | 0.62 |
| `h06_turning` | 39.54 | 0.41 | 0.73 | 0.52 | 0.51 |
| `h07_disturbances` | 8.86 | 0.64 | 0.78 | 1.21 | 1.22 |
| `h08_walk_disturbed` | 18.11 | 0.31 | 1.55 | 0.68 | 0.60 |
| `h09_walk_rough` | 105.79 | 0.30 | 1.54 | 0.77 | 0.65 |
| `h10_mount_pitch_error` | 86.72 | 0.27 | 14.10 | 1.14 | 0.69 |
| `h11_mount_roll_height_error` | 3.24 | 0.50 | 2.75 | 1.29 | 0.85 |
| `h15_imu_dropouts` | 15.29 | 0.27 | 3.09 | 0.68 | 0.55 |
| `h16_timing_offset` | 87.12 | 0.27 | 1.19 | 0.44 | 0.41 |
| `h17_imu_late` | 8.51 | 0.28 | 0.90 | 0.71 | 0.71 |

Over the tracking window (from the first time a run stays within 0.30 m and 10 degrees), arm
D measures 0.031 m / 0.52 deg walking the rectangle, 0.031 / 0.52 turning, 0.044 / 0.71 with
the unsteady gait, 0.061 / 0.90 under pitch sway and 0.075 / 0.86 standing.

**Divergence and incorrect convergence.** "Lost" is the share of runs that ended more than
0.30 m or 10 degrees off. "Converged but wrong" is the share of frames in which the filter
called itself converged while it was not localized.

| Scenario | A fixed: lost / converged but wrong | C imu | D imu + unc. | D + self-cal. |
|---|---|---|---|---|
| `h02_sway_pitch` | 100 % / 71 % | 0 % / 6 % | 0 % / 0 % | 0 % / 0 % |
| `h04_walk_straight` | 0 % / 6 % | 0 % / 7 % | 0 % / 0 % | 0 % / 0 % |
| `h05_walk_rectangle` | 33 % / 24 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| `h06_turning` | 67 % / 30 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| `h08_walk_disturbed` | 33 % / 21 % | 0 % / 3 % | 0 % / 0 % | 0 % / 0 % |
| `h09_walk_rough` | 100 % / 67 % | 0 % / 1 % | 0 % / 0 % | 0 % / 0 % |
| `h10_mount_pitch_error` | 100 % / 69 % | **33 % / 51 %** | 0 % / 0 % | 0 % / 0 % |
| `h11_mount_roll_height_error` | 0 % / 2 % | 0 % / 6 % | 0 % / 0 % | 0 % / 0 % |
| `h12_head_low` | 33 % / 17 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| `h15_imu_dropouts` | 33 % / 21 % | 0 % / 4 % | 0 % / 0 % | 0 % / 0 % |
| `h16_timing_offset` | 33 % / 33 % | 0 % / 3 % | 0 % / 0 % | 0 % / 0 % |
| `h18_placeholder_walk` | 33 % / 17 % | 0 % / 2 % | 0 % / 0 % | 0 % / 0 % |

In the scenarios not listed no arm was lost and the IMU arms were "converged but wrong" in at
most 1 % of the frames (the fixed arm in up to 21 %). Arms D and D + self-calibration were
never lost and never "converged but wrong" in any scenario. No IMU arm ended a run on the
mirrored pose; the fixed arm did in one run of `h16_timing_offset` and spent 18 % of the frames of `h09_walk_rough` there.

What this says:

* **The fixed camera model does not survive a moving torso.** The walking profile's 2 degree
  forward lean and 1 to 2 degrees of oscillation are exactly the kind of error Phase 3 found to
  cost 34 cm; here it costs 0.3 to 3 m and, worse, the filter says it has converged in a fifth
  to two thirds of the frames.
* **The IMU arm (C) removes most of it** but is left 5 to 16 cm off, because a tilt error of
  half a degree still moves far points by 10 cm or more and the localizer trusts them.
* **Uncertainty-aware projection (D) halves the remaining error** in almost every moving
  scenario and removes every case of "converged but wrong". It does this without a better
  orientation: C and D project with the same pose.
* **Standing still (`h01`) nothing is gained.** All arms, the oracle included, are at 7 to
  9 cm for the whole 20 s, most of which is the first seconds of convergence from the 0.3 m
  guess; 6 to 8 cm remain over the tracking window, more than Phase 3 measured on its platform.
  From one fixed viewpoint the perception error does not average out, and the camera is 10 cm
  lower than in Phase 3.
* **Lateral sway (`h03`) is the one motion where C is worse than the fixed camera** (0.082 m
  against 0.068 m). A roll error of 3 degrees costs the fixed camera little; the IMU arm
  trades it for the IMU's own pitch error (0.4 degrees, mostly accelerometer bias), and
  pitch is what the projection is sensitive to. D and the self-calibrating arm are better than
  both.
* **The placeholder rig (`h18`) behaves like the humanoid** under the same motion: the result
  is about the camera's motion, not the body that carries it.
* **A lost filter's error is a matter of luck.** For the fixed arm, `h05`, `h15`, `h16` and
  `h17` are the same condition (it ignores the IMU) with different random draws; its error
  there is 0.74, 0.71, 1.90 and 0.44 m. The size of a number in the "A fixed" column means
  little beyond "lost"; the IMU arms' numbers for the same four scenarios (D: 0.034, 0.051,
  0.046, 0.047 m) differ because the IMU conditions do.

## IMU roll and pitch estimation (deliverable D)

Arm C, three seeds, after the first 3 s. Angles in degrees.

| Scenario | True body tilt, RMS about its mean: roll / pitch (mean pitch) | Estimated body tilt, RMSE: roll / pitch | Camera pitch / roll error of the fixed model | Reported 1 sigma | Error / reported |
|---|---|---|---|---|---|
| `h01_standing` | 0.21 / 0.29 (0.0) | 0.23 / 0.19 | 0.31 / 0.26 | 0.42 | 0.50 |
| `h02_sway_pitch` | 0.21 / 2.82 (0.0) | 0.08 / 0.45 | 2.82 / 0.27 | 0.45 | 0.77 |
| `h03_sway_roll` | 2.82 / 0.28 (0.0) | 0.42 / 0.40 | 0.29 / 3.11 | 0.44 | 1.06 |
| `h04_walk_straight` | 1.68 / 1.25 (1.7) | 0.44 / 0.82 | 2.24 / 1.96 | 0.97 | 0.66 |
| `h05_walk_rectangle` | 1.79 / 1.18 (1.9) | 0.51 / 0.42 | 2.29 / 2.05 | 0.76 | 0.70 |
| `h06_turning` | 1.73 / 1.21 (1.8) | 0.38 / 0.44 | 2.29 / 2.01 | 0.91 | 0.52 |
| `h07_disturbances` | 1.21 / 1.22 (0.1) | 0.53 / 0.41 | 1.30 / 1.37 | 1.30 | 0.42 |
| `h08_walk_disturbed` | 2.03 / 1.55 (1.9) | 0.80 / 0.77 | 2.48 / 2.33 | 1.39 | 0.81 |
| `h09_walk_rough` | 3.18 / 2.29 (3.4) | 1.13 / 0.91 | 4.13 / 3.67 | 1.20 | 0.98 |
| `h15_imu_dropouts` | 1.76 / 1.17 (1.9) | 1.29 / 0.94 | 2.29 / 2.02 | 2.02 | 0.72 |
| `h16_timing_offset` (40 ms clock offset) | 1.76 / 1.16 (1.9) | 0.68 / 0.83 | 2.29 / 2.02 | 0.76 | 1.18 |
| `h17_imu_late` (IMU 80 ms late) | 1.75 / 1.17 (1.9) | 0.74 / 0.98 | 2.29 / 2.00 | 1.69 | 0.60 |

* Whenever the body moves, the estimate is two to five times closer to the truth than a level
  body. Standing, it is about as good as assuming level, which is all that can be expected with
  0.2 to 0.3 degrees of true motion and a 0.3 degree accelerometer-bias floor.
* The error is not zero-mean within a run: an accelerometer bias of 5 mg is a constant 0.3
  degree tilt. Across the scenarios the mean camera pitch error of arm C is between -0.2 and
  +0.6 degrees. To the camera this is indistinguishable from a small mount error, which is
  why the self-calibrating arm reduces it.
* The reported standard deviation is of the right size, slightly conservative in most cases
  (ratio below 1), and it rises where it should: with disturbances, with lost samples, with a
  late IMU. It does **not** rise for the hidden clock offset of `h16` (ratio 1.18): nothing in
  the data tells the filter that its timestamps are wrong.
* Lost samples (`h15`: 35 % of the samples, outages up to 0.39 s): the tilt error doubles to
  about 1 degree, 5 % of the frames fall back to the level body because the reported
  uncertainty exceeds the limit, and D holds 0.051 m.

## Consistency (confidence against error)

RMS position error over RMS reported standard deviation (1 is consistent, above 1 is
overconfident), and the share of frames with the error inside two reported standard deviations.

| Scenario | A fixed | B oracle | C imu | D imu + unc. | D + self-cal. |
|---|---|---|---|---|---|
| `h01_standing` | 2.21 (80 %) | 1.87 (81 %) | 1.96 (64 %) | 1.51 (79 %) | 1.42 (81 %) |
| `h02_sway_pitch` | 18.51 (3 %) | 0.74 (96 %) | 2.72 (44 %) | 1.10 (91 %) | 0.98 (92 %) |
| `h03_sway_roll` | 1.30 (88 %) | 0.43 (100 %) | 2.97 (33 %) | 0.92 (100 %) | 0.53 (100 %) |
| `h04_walk_straight` | 6.24 (9 %) | 0.79 (97 %) | 5.00 (57 %) | 1.45 (80 %) | 1.35 (82 %) |
| `h05_walk_rectangle` | 9.97 (5 %) | 0.31 (100 %) | 2.24 (57 %) | 0.56 (100 %) | 0.46 (100 %) |
| `h06_turning` | 11.49 (8 %) | 0.41 (100 %) | 2.27 (44 %) | 0.50 (100 %) | 0.48 (100 %) |
| `h07_disturbances` | 4.52 (87 %) | 0.78 (95 %) | 3.00 (13 %) | 1.23 (87 %) | 1.21 (86 %) |
| `h08_walk_disturbed` | 8.73 (8 %) | 0.47 (100 %) | 4.38 (37 %) | 0.95 (97 %) | 0.77 (97 %) |
| `h09_walk_rough` | 37.32 (4 %) | 0.41 (100 %) | 4.51 (13 %) | 0.79 (100 %) | 0.60 (100 %) |
| `h10_mount_pitch_error` | 30.37 (2 %) | 0.49 (100 %) | 11.06 (2 %) | 2.20 (46 %) | 1.06 (93 %) |
| `h11_mount_roll_height_error` | 4.39 (14 %) | 3.02 (0 %) | 5.61 (0 %) | 1.56 (78 %) | 0.87 (98 %) |

Arm C is overconfident everywhere: it projects with an orientation that is half a degree off
and reports the uncertainty of a perfectly known camera. Arm D is within 0.5 to 1.5 in every
scenario with a correct mount, at the price of being somewhat under-confident when walking
(0.5 to 0.6). A wrong mount still makes D overconfident (2.2): its uncertainty covers the tilt
estimate, not a calibration error it does not know about. Standing, every arm is overconfident
by 1.4 to 2.2, the oracle included: that is the perception's own bias from a fixed viewpoint,
not a camera-pose effect.

## Camera projection error

Mean distance by which ground points 1 to 4 m ahead are misplaced by the camera pose an arm
uses, metres. This isolates the camera-pose estimate from perception and the filter.

| Scenario | A fixed | B oracle | C / D imu | D + self-cal. | C, newest sample |
|---|---|---|---|---|---|
| `h02_sway_pitch` | 0.425 | 0.014 | 0.061 | 0.052 | 0.049 |
| `h05_walk_rectangle` | 0.415 | 0.022 | 0.065 | 0.059 | 0.070 |
| `h09_walk_rough` | 0.972 | 0.036 | 0.122 | 0.095 | 0.136 |
| `h10_mount_pitch_error` | 0.949 | 0.021 | 0.414 | 0.082 | 0.415 |
| `h11_mount_roll_height_error` | 0.327 | 0.102 | 0.127 | 0.077 | 0.130 |
| `h15_imu_dropouts` | 0.415 | 0.021 | 0.130 | 0.130 | 0.125 |
| `h16_timing_offset` | 0.416 | 0.021 | 0.127 | 0.123 | 0.075 |
| `h17_imu_late` | 0.416 | 0.021 | 0.151 | 0.143 | 0.157 |

The oracle's 2 cm is the walking bob and the mount flex, which it does not know either. In
`h11` the oracle is 10 cm off: it has the right orientation and the wrong height.

## Camera mounting error (deliverable F, part 1)

`h10`: the camera is really pitched 2 degrees further down than the calibration says.
`h11`: it is rolled 1.5 degrees and 3 cm higher.

| | `h10` position RMSE | `h10` mean camera pitch error | `h11` position RMSE |
|---|---|---|---|
| A fixed | 2.501 m, all runs lost | | 0.215 m |
| B oracle orientation | 0.018 m | 0 | 0.091 m |
| C imu | 0.581 m, one run of three lost, "converged" while wrong in 51 % of the frames | -2.15 deg | 0.173 m |
| D imu + uncertainty | 0.112 m | -2.15 deg | 0.082 m |
| D + self-calibration | 0.069 m | reduced to 0.68 deg RMS | 0.055 m |

* **The IMU does not help with a mount error, as it cannot.** Arm C's body tilt is as good as
  in `h05` (0.5 degrees); its camera pitch is wrong by the full 2 degrees in every frame.
* **D survives it** without knowing about it, because far points are discounted.
* **The monitor finds it.** Final pitch corrections in the three seeds: +2.26, +1.68 and +2.30
  degrees (true: +2.00) at the end of the 50 s lap. In `h11` the roll corrections
  are +2.03, +1.73 and +1.61 degrees (true: +1.50) with -0.4 to -0.8 degrees of pitch, which
  is the 3 cm height error being absorbed as pitch: the monitor cannot tell the two apart.
  This is also why the self-calibrating arm beats the orientation oracle in `h11`.
* **With a correct mount the monitor does little harm.** Over the 16 scenarios without a
  mount error its final corrections are within 0.7 degrees of pitch and 1.0 degree of roll
  (median 0.26 and 0.19), part of which is the IMU's accelerometer bias being corrected, part
  of which is noise. Its position error is within 0.2 cm of D's where it is not better.

Figures: `outputs/humanoid_fusion/benchmark/plots/run_h10_mount_pitch_error.png` shows the
IMU estimate following the shape of the true camera pitch 2 degrees below it.

## Calibration sweep (deliverable F, part 2)

The walking rectangle (`h05`), three seeds, with a deliberately wrong calibration on top of
whatever each arm estimates. The error is the robot's estimate minus the truth, so +2 degrees
of pitch means the robot believes its camera looks 2 degrees further down than it does. All
arms of a seed share one set of images. Position RMSE over the whole run in metres; in
brackets the share of runs that ended lost, where it is not zero. The orientation oracle is at
0.018 m. Files: `outputs/humanoid_fusion/sweeps/calibration_sweep.{json,csv}`,
`calibration_sensitivity.png`.

**Camera pitch**

| Pitch error | A fixed | C imu | D imu + unc. | D + self-cal. | Self-cal.'s final pitch correction |
|---|---|---|---|---|---|
| -3 deg | 2.923 (100 %) | 1.945 (100 %) | 0.174 | 0.103 | +3.00 deg (the bound) |
| -2 deg | 2.737 (100 %) | 0.388 | 0.126 | 0.075 | +2.12 deg |
| -1 deg | 1.195 (100 %) | 0.187 | 0.074 | 0.050 | +1.12 deg |
| 0 | 0.737 (33 %) | 0.069 | 0.034 | 0.030 | +0.12 deg |
| +1 deg | 0.178 | 0.143 | 0.068 | 0.042 | -0.88 deg |
| +2 deg | 0.192 | 0.351 | 0.127 | 0.066 | -1.88 deg |
| +3 deg | 0.570 | 2.312 (33 %) | 0.185 | 0.090 | -2.87 deg |

**Camera roll**

| Roll error | A fixed | C imu | D imu + unc. | D + self-cal. | Self-cal.'s final roll correction |
|---|---|---|---|---|---|
| -2 deg | 0.543 | 0.179 | 0.070 | 0.052 | +1.78 deg |
| -1 deg | 0.884 (33 %) | 0.103 | 0.041 | 0.033 | +0.78 deg |
| 0 | 0.737 (33 %) | 0.069 | 0.034 | 0.030 | -0.21 deg |
| +1 deg | 0.700 (33 %) | 0.112 | 0.051 | 0.039 | -1.21 deg |
| +2 deg | 2.454 (67 %) | 0.182 | 0.075 | 0.052 | -2.21 deg |

**Camera height**

| Height error | A fixed | C imu | D imu + unc. | D + self-cal. | Self-cal.'s final pitch correction |
|---|---|---|---|---|---|
| -5 cm | 0.179 | 0.159 | 0.082 | 0.050 | -1.04 deg |
| -2 cm | 0.222 | 0.083 | 0.040 | 0.030 | -0.34 deg |
| 0 | 0.737 (33 %) | 0.069 | 0.034 | 0.030 | +0.12 deg |
| +2 cm | 1.745 (33 %) | 0.089 | 0.049 | 0.037 | +0.58 deg |
| +5 cm | 0.914 (100 %) | 0.166 | 0.090 | 0.063 | +1.25 deg |

* **Pitch is still the sensitive parameter, and the IMU does not change that.** With a good
  body tilt, arm C loses 7 to 12 cm for the first degree and about 20 cm more for the second;
  at 3 degrees it is lost (all three runs at -3, one of three at +3). This is Phase 3's
  finding (34 cm for 2 degrees) again, on a moving body (35 to 39 cm).
* **Uncertainty-aware projection flattens the curve**: 7 cm at 1 degree, 13 cm at 2, 17 to
  19 cm at 3, and never lost.
* **The monitor removes the injected error**, to within 0.13 degrees of pitch and 0.22 of roll
  over the whole range, and stops at its 3 degree bound. The 5 to 10 cm that remain in the
  table are mostly the time the correction takes to get there (about 20 s in the analytic
  tests); the run is only 50 s long.
* **Roll costs about a third of what pitch costs**: C goes from 7 cm to 10 to 11 cm at
  1 degree and 18 cm at 2 degrees, D to 4 to 5 cm and 7 cm.
* **Height**: 2 cm cost C about 2 cm; 5 cm cost 9 to 10 cm, the 12 cm of Phase 3 within the
  spread. The monitor turns a height error into a pitch correction of about 0.2 degrees per
  centimetre and recovers most of the accuracy that way, with the wrong explanation.
* **The fixed arm's column is not a sensitivity curve.** It is best at +1 to +2 degrees
  because believing the camera looks further down happens to cancel the walking profile's
  2 degree forward lean. Two wrongs.

Sensitivity to the virtual head's height is in the next section and to timing in the one
after.

## Virtual head height

The rectangle lap with the four head presets (`h12`, `h05`, `h13`, `h14`), position RMSE in
metres. Figure: `outputs/humanoid_fusion/benchmark/plots/camera_height.png`.

| Mount | Camera height | A fixed | B oracle | C imu | D imu + unc. | D + self-cal. |
|---|---|---|---|---|---|---|
| `head_low` | 0.739 m | 0.675 | 0.019 | 0.083 | 0.041 | 0.043 |
| `head_mid` | 0.799 m | 0.737 | 0.018 | 0.069 | 0.034 | 0.030 |
| `head_high` | 0.859 m | 0.309 | 0.015 | 0.099 | 0.048 | 0.035 |
| `head_tall` | 0.899 m | 0.368 | 0.022 | 0.084 | 0.044 | 0.039 |

Between 0.74 and 0.90 m the height of the virtual head makes no difference that three seeds
can resolve: D is at 3 to 5 cm and the oracle at about 2 cm for all four. In particular the
lowest head, which is below the 0.78 to 1.02 m the networks were trained at, is not worse.
The fixed camera does better with the two taller heads (0.31 to 0.37 m against 0.68 to
0.74 m), consistent with range error per degree of pitch falling as the camera rises, but it
is unusable at every height. This says the *software* does not care much which of these
heights the head ends up at. It says nothing about fields of view blocked by the robot's own
arms or shoulders, which depend on a head that does not exist.

## Timing errors (deliverable F, part 3)

**Image and IMU timestamps that disagree.** The walking rectangle, three seeds. Each arm pairs
every image with the orientation of `offset` milliseconds later (positive) or earlier than the
image's true time, as it would with a clock offset it does not know about. Files:
`outputs/humanoid_fusion/sweeps/timing_sweep.json`, `timing_sensitivity.png`.

| Timestamp mismatch | C: camera pitch / roll error, RMS | C position RMSE | D position RMSE |
|---|---|---|---|
| -80 ms | 1.27 / 1.00 deg | 0.080 m | 0.037 m |
| -40 ms | 0.80 / 0.58 deg | 0.060 m | 0.035 m |
| -20 ms | 0.58 / 0.50 deg | 0.066 m | 0.034 m |
| 0 | 0.44 / 0.59 deg | 0.069 m | 0.034 m |
| +20 ms | 0.47 / 0.78 deg | 0.069 m | 0.034 m |
| +40 ms | 0.66 / 1.02 deg | 0.066 m | 0.034 m |
| +80 ms | 0.50 / 0.82 deg | 0.068 m | 0.032 m |
| newest IMU sample, no interpolation | 0.50 / 0.82 deg | 0.068 m | |

For reference: fixed camera 0.737 m, orientation oracle 0.018 m.

* **The orientation error grows with the mismatch as expected**: the walking profile turns the
  body at up to 25 to 30 degrees per second, and 40 ms of mismatch add 0.2 to 0.4 degrees RMS,
  80 ms up to 0.8.
* **Localization hardly notices** on this gait: C stays between 6 and 8 cm and D between 3.2
  and 3.7 cm across the whole range. A timing error produces an orientation error that swings
  with the stride and averages out over a few frames; a mount error does not. With faster or
  less regular body motion the orientation error per millisecond is larger in proportion.
* **Interpolating to the image timestamp was not shown to matter for localization here.** The
  newest-sample pairing is an orientation 26 ms newer than the image (30 ms camera latency
  less 4 ms IMU latency), and its pose error is the same as the interpolated arm's. It is the
  right thing to do and costs nothing, but this benchmark does not demonstrate a benefit.
  Across the 18 scenarios the two differ by less than the spread of three seeds, and under the
  slow pitch sway of `h02` the naive pairing is even slightly better (0.124 against 0.148 m):
  there the filter's estimate lags by a few tens of milliseconds because the pendulum's own
  acceleration leaks through the accelerometer feedback, and pairing with a newer sample
  happens to cancel part of that lag.
* **+80 ms equals the naive pairing.** No IMU sample that new has arrived when the image is
  processed, extrapolation is limited to 50 ms, so the newest estimate is held.
* **The roll error is smallest at -20 ms, not at zero**, for the same reason as in `h02`: the
  stride's lateral acceleration shifts the phase of the roll estimate slightly.

**The scenario versions** (the IMU stream itself is changed, and the arms do not know):

| Scenario | C tilt error roll / pitch | How the orientation was obtained | C position | D position |
|---|---|---|---|---|
| `h05` reference | 0.51 / 0.42 deg | interpolated in 99.8 % of the frames | 0.069 m | 0.034 m |
| `h16` IMU clock 40 ms off | 0.68 / 0.83 deg | interpolated, at the wrong instant | 0.095 m | 0.046 m |
| `h17` IMU arrives 80 ms late | 0.74 / 0.98 deg | extrapolated in 10 %, held in 90 %; nearest sample 52 ms away | 0.074 m | 0.047 m |
| `h15` 35 % of the samples lost | 1.29 / 0.94 deg | interpolated in 92 % of the frames, across gaps up to 0.39 s; 5 % of the frames fell back to the level body | 0.166 m | 0.051 m |

The synchronisation statistics (method used, largest bracket, offset to the nearest sample) are
stored per run in `results.json` and shown live in the viewer.

## Processing latency and frame rate

Primary arm, 640 x 360, RTX 2050, mean over the 18 scenarios; first five frames of each run
left out.

| Stage | Mean [ms] |
|---|---|
| IMU filter and camera pose (about 20 samples per frame) | 2.2 |
| detector (YOLO11n) | 8.3 |
| segmenter (U-Net) | 21.2 |
| geometric extraction | 11.5 |
| particle filter | 2.9 |
| **whole pipeline** | **46.3** (95th percentile 50.5) |

That is 21.6 frames per second, against 23 in Phase 3: the IMU chain adds 2.2 ms, of which
almost all is the filter's 110 microseconds per IMU sample in Python. Rendering the scene with
the humanoid takes 9.8 ms and is the simulator's cost, not the pipeline's. GPU memory: 136 MB
allocated at peak. The mount monitor adds 1.1 ms per frame to the arm that runs it (measured
separately on one frame with 57 line points); the shared-error model adds nothing measurable.

## Tests (deliverable H)

```bash
.venv/bin/python -m pytest          # 585 tests, about 5.5 minutes
```

585 tests: the 469 that existed before Phase 4 and 116 new ones. All 585 pass on the
development laptop (5 min 25 s), none skipped: the trained checkpoints, the imported humanoid,
PySide6 and an OpenGL context were all present.

| File | Tests | What it establishes |
|---|---|---|
| `test_mount_and_motion.py` | 25 | rotation conventions (positive pitch looks down; tilt independent of yaw); **the camera height is the body's geometry plus the preset's offset, and a mount without a height is refused**; every shipped mount is marked experimental; the mount chain equals Phase 3's camera model for a level body; a rig on a tilted body is still a `RigState`; body pitch and mount pitch are not interchangeable; robot descriptions; every motion profile: rigid, sway about the ankles, walking in all six degrees of freedom and only while walking, abrupt but continuous disturbances, determinism per seed |
| `test_imu_orientation.py` | 32 | **simulated IMU**: a level sensor at rest reads +g; rates and specific force follow the motion analytically; lever arm; sensor axes; **noise, bias and bias drift have the configured size**; variable rate, latency, dropouts, clock offset; a sample carries no orientation. **Tilt filter**: converges and learns the gyro bias; follows a fast tilt through the gyro; **does not mistake a step's acceleration for tilt**; accuracy per motion profile over six seeds, with an honest uncertainty; beats a level body whenever the body moves; **uncertainty grows across an outage**; refuses bad samples. **Synchronisation**: interpolation gives the orientation at the image timestamp; the newest sample is the wrong one for an older image; extrapolation, holding and stale requests say so. **Camera pose**: fixed mode is Phase 3's; IMU mode reproduces the true camera pitch of a swaying body; **an IMU cannot see a wrong mount** |
| `test_humanoid_model.py` | 13 | **the actual humanoid model loads**: 20 hinge joints, 21 meshes, over 300 000 triangles, documented axes, nothing collides, nothing actuated; the bundle records its source and what was left out; **stored body geometry equals what the model measures**; the importer moves no vertex; ankle four-bars stay closed; feet stay on the ground; **camera and body are where the mount chain says, read back from MuJoCo**; markers never reach the robot's own camera; the camera sees its own body when pointed at it; head presets change the rendered camera height; the placeholder still works; scenes without a robot are the Phase 3 scenes |
| `test_humanoid_fusion.py` | 20 | scenarios cover the required conditions; arms; **the shared-error model bounds what a frame claims about the camera** and leaves Phase 3 bit for bit; **the monitor finds a 2 degree pitch bias from field lines, stays quiet when calibrated, is bounded**, learns nothing before convergence; with the simulator: **localization under camera motion needs the IMU**; tilt reported with an honest uncertainty; deterministic, arms independent; a mount error is a bias the IMU arm keeps; the placeholder runs through the same pipeline; the report is written. **Ground-truth isolation**: sensor packets carry no pose; the robot is told the believed mount, never the true one; an oracle pose is refused by non-oracle arms; **recorded packets alone reproduce the estimates** |
| `test_ground_truth_isolation.py` | 33 (22 before) | the Phase 3 tripwire, now also scanning `estimation/` and the mount geometry; **the tilt filter, synchronisation and camera-pose estimator run with MuJoCo and the simulation package blocked**; estimators take sensor types only |
| `test_viewer_humanoid.py` | 12 | see [viewer.md](viewer.md) |
| `test_humanoid_fusion_models.py` | 2 | with the **trained** YOLO11n and U-Net: the pose holds under sway with the IMU and is lost without it; the viewer session matches the benchmark |

Two existing tests had to follow intended changes, and nothing else in the 469 was touched:
`test_viewer_gui.py::test_benchmark_scenarios_can_be_selected` asserted that the scenario list
has exactly 16 entries; it now checks that the 16 Phase 3 scenarios are all still there next
to the 18 new ones. `test_viewer.py::test_every_toggle_changes_its_own_panel` runs once per
overlay toggle; its frame was given a robot picture so that the new "Robot view" toggle has
something to draw.

**Phase 1 to 3 results with Phase 4 switched off.** Scenes built without a robot have no
additional body, camera or geom; feature extraction with `shared_camera_error: false` returns
the same arrays; `VisionLocalizationPipeline.step` has the same signature (pinned by a test);
the Phase 3 viewer session still reproduces the Phase 3 benchmark arm bit for bit. The Phase 1
and Phase 3 benchmarks were not re-run in full (about 2 hours); their test-sized versions pass
unchanged.

## Demonstration outputs (deliverable I)

All under `outputs/` (not tracked by Git).

| What | Where |
|---|---|
| Benchmark tables | `outputs/humanoid_fusion/benchmark/summary.md`, `summary.csv`, `results.json` |
| Every scenario, every arm | `outputs/humanoid_fusion/benchmark/plots/overview.png` |
| Head height | `outputs/humanoid_fusion/benchmark/plots/camera_height.png` |
| Camera pitch / roll and position error over time, one per scenario | `outputs/humanoid_fusion/benchmark/plots/run_<scenario>.png` |
| Calibration sweep | `outputs/humanoid_fusion/sweeps/calibration_sensitivity.png`, `calibration_sweep.json`, `calibration_sweep.csv` |
| Timing sweep | `outputs/humanoid_fusion/sweeps/timing_sensitivity.png`, `timing_sweep.json` |
| Benchmark log | `outputs/humanoid_fusion_benchmark.log` |
| The humanoid with each mount (third person, see-through, own camera) | `outputs/humanoid/mount_humanoid_prototype_1_<preset>.png`, `mounts_humanoid_prototype_1.png`, `mounts_humanoid_prototype_1.json`; a tilted body in `outputs/humanoid/tilted/`; the placeholder in `mount_placeholder_legacy_rig.png` |
| Viewer videos, trained networks | `outputs/viewer/videos/phase4_h05_walk_rectangle_imu_uncertainty.mp4`, `phase4_h10_mount_pitch_error_imu_selfcal.mp4`, `phase4_h02_sway_pitch_fixed.mp4`, `phase4_h02_sway_pitch_imu_uncertainty.mp4` |
| Viewer recordings (replay with `--replay`) | `outputs/viewer/recordings/phase4_h05_walk_rectangle_imu_uncertainty.npz`, `phase4_h10_mount_pitch_error_imu_selfcal.npz` |
| Four-panel screenshots | `outputs/viewer/screenshots/h05_walk_rectangle_seed0_step0{125,250,375}.png`, `h10_mount_pitch_error_seed0_step0{125,250,375}.png`, `h02_sway_pitch_seed0_step0100.png` |

The two sway videos are the same run with the same images: with the fixed camera model the
estimate leaves the field (1.69 m RMSE); with the IMU it stays on the robot (0.052 m).

## Limits

* **Synthetic motion.** The gait is sinusoids. A real gait has foot impacts (sharp vertical
  and pitch transients), irregular steps and structural flex; all of them are harder for a
  tilt filter than what was tested, and none has been measured on this robot.
* **Simulated IMU.** White noise, a constant bias and a random-walk drift. No vibration
  rectification, scale-factor or cross-axis error, temperature effect or saturation. The
  figures are generic.
* **Global shutter, no blur.** A ZED Mini has a rolling shutter; at 30 degrees per second of
  body rotation that skews the image by a fraction of a degree top to bottom, of the same
  order as the tilt error achieved here.
* **The standing height is assumed known.** The walking bob (8 to 13 mm) is not estimated and
  is the largest part of the oracle's remaining projection error. A real robot would get the
  camera height from leg kinematics.
* **Pitch and height are confounded** in the mount monitor, and it needs a converged localizer
  and lines at several ranges. It was tested on two mount errors and one field size.
* **A late IMU is held, not predicted.** With 80 ms of IMU latency against 30 ms of camera
  latency 90 % of the frames use an orientation about 50 ms old (`h17`). The uncertainty
  rises, and D holds 4.7 cm, but the right fix on a robot is to process the image when its IMU
  samples are there.
* **The filter's gains were chosen on this synthetic motion.** They were set on motion-only
  runs (body motion and IMU, no images, no localizer) with the same profiles and independently
  drawn noise, and the monitor's settings on analytically painted views. Nothing was tuned on
  the benchmark's localization results: one trial of two scenarios with seed 0 was looked at
  before the benchmark and no parameter was changed after it. But the motion family the
  filter was tuned on is the motion family it was tested on.
* **One field size, no other robots, no kidnapping, nominal appearance.** Phase 3's scenarios
  for those were not repeated on the moving body.
* **Three seeds.**
* **The humanoid is not simulated.** No balance, no contact, no falls; the detector has never
  seen this robot's shape as an opponent; and the head, its mass and its field of view are all
  still to come.

## Next steps

**Before the first real ZED Mini test**

1. Calibrate the camera: intrinsics and distortion from the ZED's factory calibration, checked
   against a target. Add a distortion model to `geometry/camera.py` or rectify first.
2. Record synchronized image and IMU data with the camera held and moved by hand over a real
   field section: stationary, slow pitch and roll sweeps, shaken. The ZED Mini has its own
   IMU; logging it next to the robot's gives a second reference.
3. Replay those logs through `CameraPoseEstimator` and compare the tilt with a reference (a
   level, a known target, or the field lines themselves). This replaces the simulated IMU
   figures with measured ones and is the first hard test of the filter's gains.
4. Measure the time offset between image stamps and IMU stamps (for example by correlating
   image-derived pitch rate with the gyro). The timing sweep above says what an error costs.
5. Run the trained networks on the recorded images before anything else is concluded: the
   sim-to-real gap of YOLO11n and the U-Net is still entirely unmeasured.

**When the head is designed**

6. Write its mount file from CAD and re-run `scripts/benchmark_humanoid_fusion.py` with it
   ([camera_mount.md](camera_mount.md)); check what the body occludes in the image.
7. Calibrate the mount on the robot and compare with what the monitor estimates.

**When the robot walks**

8. Log pelvis IMU data during walking and fit `configs/motion/` to it, or replay it directly.
   Replace the standing-height assumption with leg kinematics.
9. Add the scenarios Phase 4 did not repeat on a moving body: opponents, occlusion,
   kidnapping, appearance shift, the larger fields.

**ROS 2 integration** (nothing here touches `soccer-software`)

10. A localization node around `ArmEstimator`: `sensor_msgs/Imu` into `add_imu`, the image's
    header stamp into `estimate`, the mount from a static transform or `configs/mounts/`,
    `OdometryDelta` from the walking engine, `map -> odom` out. The node needs the camera pose
    at the image's stamp, which is what `OrientationHistory.at` provides.
11. The IMU filter costs 110 microseconds per sample in Python; at 200 Hz that is 2 % of a
    core. If the robot already runs an attitude estimator, its roll and pitch can replace
    `TiltEstimator` behind the same history, provided it also reports an uncertainty.
12. Publish the monitor's status and the field-line residual as diagnostics before trusting
    its correction on hardware.
