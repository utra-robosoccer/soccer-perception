# Camera motion, the simulated IMU and IMU-assisted camera pose

Phase 3 projected every image to the ground with one fixed camera pose. A humanoid's camera
moves with its torso, and Phase 3 found that a hidden 2 degree pitch error costs about 34 cm of
localization accuracy. This page describes what Phase 4 adds between the camera and the
unchanged Phase 3 pipeline. Results are in [phase4_validation.md](phase4_validation.md).

```
 path over the field ─► BodyMotion ─┬─► true camera pose ─► MuJoCo render ─► RGB image ──┐
        (synthetic, hidden)         ├─► SimulatedImu ─► ImuSample (gyro, accel, time) ───┤
                                    └─► footprint poses ─► SyntheticOdometry ────────────┤
 ───────────────────────────────── simulator above, robot below ─────────────────────────┤
   ImuSample ─► TiltEstimator ─► OrientationHistory ─► tilt at the image timestamp       │
   CameraMount (believed) + BodyGeometry ─► CameraPoseEstimator ─► EstimatedCameraPose   │
                                                         + per-frame uncertainty         ▼
   image ─► YOLO11n + U-Net ─► geometric features ─► ObservationFrame ─► particle filter ─► pose
                                          │
                                          └─► CalibrationMonitor ─► bounded mount correction
```

Everything about the body's motion is **synthetic** and the IMU is **simulated**. The robot
cannot walk yet and has no IMU in its CAD.

## 1. Body motion (`simulation/body_motion.py`)

A Phase 1 trajectory gives the path. `BodyMotion` turns it into the 6-DOF motion of the pelvis
as an analytic function of time, by adding:

| Component | What it does | Where the numbers are |
|---|---|---|
| Gait | lateral shift, roll and yaw at the stride frequency; bob, pitch and a speed ripple at the step frequency; a forward lean. Scaled by how fast the path moves | `gait_*` |
| Postural sway | slow pitch / roll of the whole body about the ankles (an inverted pendulum: the body also moves forward when it pitches forward) | `sway_*` |
| Wobble | band-limited random roll, pitch and height | `noise_*` |
| Disturbances | a tilt of 4 to 8 degrees reached in 0.12 s, recovering in 0.7 s, at random times, about the ankles | `disturbance_*` |

Profiles in `configs/motion/` (amplitudes are one-sided):

| Profile | Use | Main content |
|---|---|---|
| `rigid` | Phase 3 behaviour | nothing |
| `standing` | A. stationary standing | sway 0.35 / 0.25 deg, wobble 0.15 deg |
| `sway_pitch` | B. forward / backward sway | pitch +-4 deg at 0.5 Hz |
| `sway_roll` | C. lateral sway | roll +-4 deg at 0.45 Hz |
| `walking` | D. walking-like, E. turning | 2.2 steps/s, roll 2.5, pitch 1.5, yaw 2 deg, lateral 2 cm, bob 8 mm, lean 2 deg |
| `walking_rough` | an unsteady gait | about 1.7 times `walking` |
| `disturbed`, `walking_disturbed` | F. abrupt disturbances | standing or walking plus tilts |

Mount uncertainty (G) is not a motion: it is the hidden `MountError` of a scenario
([camera_mount.md](camera_mount.md)).

The amplitudes are plausible for a 70 cm humanoid and have **not** been fitted to measured
motion. A seed fixes every random phase and event.

The *footprint* (the ground point under the pelvis, turned to the pelvis' heading) is the
ground truth the localizer is scored against. It includes the gait's lateral shift and yaw, as
a real pelvis does; odometry is derived from it.

Not modelled: rolling shutter and motion blur (the camera is a global-shutter snapshot), foot
impacts, flexing of the structure beyond 0.1 degree of white jitter per frame.

## 2. Simulated IMU (`simulation/imu.py`)

The IMU is derived from the body's *motion*, never from its orientation:

1. **Ideal measurements.** At each sample instant the sensor's position is differentiated
   twice and its orientation once (central differences over +-1 ms). That gives the angular
   velocity and the specific force `R^T (a - g)`; a sensor at rest reads `[0, 0, +9.81]`.
   Lever-arm, centripetal and tangential terms are included because the sensor's own position
   is what is differentiated.
2. **Imperfections** (`configs/imu/nominal.yaml`):

| Effect | Nominal value |
|---|---|
| rate, with a wandering sampling interval | 200 Hz, 2 % jitter |
| gyro noise / turn-on bias / bias drift | 0.002 rad/s, 0.010 rad/s, 2e-4 rad/s/sqrt(s) |
| accelerometer noise / bias / bias drift | 0.03 m/s^2, 0.05 m/s^2 (5 mg), 1e-3 m/s^2/sqrt(s) |
| latency | 4 ms +- 1 ms |
| lost samples, outages | none (`dropouts.yaml`: 30 % and 0.1 to 0.4 s outages) |
| clock offset, axis misalignment | none (hidden parameters, set per scenario) |

These are typical figures for a consumer MEMS IMU after a bench calibration, not measurements
of the robot's sensor. The IMU's place is the model's placeholder site (the pelvis origin) and
is configurable through the body geometry.

What the robot receives is `ImuSample(timestamp, gyro, accel)`. It contains no orientation. The
true roll and pitch are kept in the stream for scoring and are not part of a sample.

## 3. Orientation filter (`estimation/orientation.py`)

A Mahony-style complementary filter on a quaternion. It reports roll, pitch and a standard
deviation; it does not observe yaw and does not claim to. It does not import MuJoCo.

* **Gyroscope**: the orientation is integrated from the bias-corrected rate.
* **Accelerometer**: the disagreement `e = (f x up_predicted) / g` is fed back with gain `kp`
  (and into the gyro bias with `ki`). `e` is not normalised by `|f|`, so it equals the body's
  horizontal acceleration over g, which averages to zero over a stride.
* **The accelerometer is not a perfect gravity sensor.** While the robot walks its reading
  swings by 5 to 10 degrees of apparent tilt at every step. The feedback gain is therefore
  lowered with the *recent level* of disagreement, averaged over more than a stride. Single
  gross outliers have their pull capped.
* **Start-up and outages**: the gain is boosted for the first seconds and after a long gap; the
  bias is not learned during that transient.
* **Uncertainty**: the filter's error follows a two-state linear system (tilt error, bias
  error). Its covariance is propagated with the same gains the filter uses; it grows across
  lost samples and with fast motion. The unobservable accelerometer bias is a constant floor
  (0.3 degrees for 5 mg).

Three design mistakes found by the tests are worth recording, because each looks reasonable:

| Tempting rule | What it did | What is used instead |
|---|---|---|
| lower the accelerometer weight sample by sample when `\|f\|` is not g | the weight favoured one half of each stride, so the stride's accelerations no longer cancelled: a tilt bias of 1 to 3 degrees | a weight from the level over a whole stride |
| shrink the correction as the disagreement grows (`w ~ 1/e^2`) | positive feedback under a fast ankle sway: the estimate drifted away | a cap on the correction (`w ~ 1/e`), applied once in both feedback paths |
| apply the whole gap as feedback time after an outage | the estimate snapped to the accelerometer's instantaneous direction | one ordinary step, more variance, a short boost |

Measured in the benchmark with the nominal IMU (three seeds, after 3 s; full table in
[phase4_validation.md](phase4_validation.md)):

| Motion | Estimated tilt, RMSE roll / pitch | True tilt about its mean, RMS roll / pitch |
|---|---|---|
| standing | 0.23 / 0.19 deg | 0.21 / 0.29 deg |
| pitch sway, +-4 deg | 0.08 / 0.45 deg | 0.21 / 2.82 deg |
| roll sway, +-4 deg | 0.42 / 0.40 deg | 2.82 / 0.28 deg |
| disturbances | 0.53 / 0.41 deg | 1.21 / 1.22 deg, 8 deg peaks |
| walking (rectangle) | 0.51 / 0.42 deg | 1.79 / 1.18 deg plus a 1.9 deg lean |
| unsteady walking | 1.13 / 0.91 deg | 3.18 / 2.29 deg plus a 3.4 deg lean |
| walking, 35 % of the samples lost | 1.29 / 0.94 deg | as walking |

The reported standard deviation is between 0.9 and 2.4 times the actual error in these cases
(mostly conservative). Standing still, the filter is no better than assuming a level body; its
value is in everything else. Under a slow sway about the ankles (0.5 Hz) the pendulum's own
acceleration leaks through the accelerometer feedback: 0.45 degrees of the +-4 degrees remain,
in the form of a small lag.

## 4. Camera / IMU synchronisation (`estimation/sync.py`)

An image reaches the pipeline tens of milliseconds after exposure; a body pitching at 30 deg/s
turns one degree in 33 ms. `OrientationHistory` keeps recent estimates by timestamp and answers
"what was the tilt at the image's timestamp":

| Case | Method | Uncertainty added |
|---|---|---|
| timestamp between two estimates | spherical interpolation of the quaternions | none (a clock-agreement term only) |
| IMU later than the image, by up to 50 ms | newest estimate advanced with its own rate | half the rate times the horizon |
| later than that | newest estimate held | rate times the gap |
| older than the history | oldest estimate | rate times the gap |
| nothing yet | none; the level body is used and reported as a fallback | the maximum |

Each answer carries its timing; `SyncStatistics` counts the methods used, the largest bracket
and the offsets over a run (the timestamp-mismatch diagnostics, shown in the viewer and stored
with every benchmark run). `OrientationHistory.latest` is the naive pairing (the newest
estimate, whatever the image's timestamp); it exists so the benchmark can measure the
difference.

## 5. Camera pose per image (`estimation/camera_pose.py`)

```
T_footprint_camera = T_footprint_body(standing height, roll, pitch)  @  T_body_rig  @  T_rig_optical
                     └────────── from the IMU ──────────┘    └ believed mount ┘
```

The result is an `EstimatedCameraPose`, the type `VisionLocalizationPipeline.step` already took
in Phase 3. **The pipeline's interface did not change.** It still receives an image, a
timestamp, an odometry increment, a calibration and an estimated camera pose, and still rejects
simulator types.

`mode: fixed` assumes an upright, level body in every frame (Phase 3). `mode: imu` uses the
filter. The body's height is the nominal standing height; the walking bob is not estimated.

## 6. Uncertainty-aware projection

Two things change when `use_uncertainty` is on (benchmark arm D).

**Per-frame uncertainty.** The tilt's standard deviation (filter plus synchronisation) is
combined in quadrature with the mount's calibration uncertainty and becomes that frame's pitch
and roll uncertainty; the height uncertainty grows by 1 cm while the body moves. Geometric
extraction propagates it to every ground point, as in Phase 3: far points widen most.

**The camera error is shared by the whole frame.** A pitch error `d` moves every ground point
together, point `i` by `J_i d`. Phase 3 added that as if it were independent per point, and a
localizer that multiplies `n` such likelihoods credits itself with information
`sum(J_i^2 / var_i)` about `d`, far more than the `1 / var(d)` the robot has. With
`shared_camera_error` the camera part's variance is multiplied by the number of independent
terms that share it,

```
var_i = pixel_i + n * camera_i ,   n = line_point_weight * min(points, max_line_points)
```

which bounds that sum by `1 / var(d)` exactly: a frame never claims to know the camera better
than the robot's estimate of it. `n` is capped at 12. A test checks the bound (it is exceeded
by a factor of more than 1.5 without the model). The particle filter is not modified; only the
standard deviations it is given change, and with the option off the extraction is bit for bit
that of Phase 3.

The price: far points are discounted more, and those whose standard deviation then exceeds
the extractor's 0.45 m limit are not reported at all (in one typical frame 24 of 57 line
points remain). Information is given up when the camera pose is in fact good: in the
benchmark D is about 2 cm worse than C standing still (within the spread of three seeds) and
about a factor of two better whenever the body moves.

## 7. Mount-calibration monitor (`estimation/calibration_monitor.py`)

The IMU cannot see a wrong mount calibration. The painted lines can: a pitch error displaces a
point at range `r` by about `(r^2 + h^2) / h` per radian, so near and far lines stop agreeing.

Per frame, from image-derived line points, the localizer's pose and the static map only:

1. match each line point to the nearest painted line; the gate widens with range;
2. linearise the point-to-line residuals in the pose (x, y, yaw) and the camera's pitch and
   roll;
3. eliminate the pose, which leaves that frame's information about pitch and roll;
4. accumulate over frames with a 2 degree prior and a memory of about 7 s.

Outputs: a field-line consistency residual, the estimated bias with a formal standard
deviation, a status (`collecting`, `ok`, `bias_suspected`, `correcting`) and a **bounded
correction**: at most 3 degrees, moved by 10 % of the remaining gap per frame, and only for a
component that is well determined after at least 40 usable frames. It can be switched to
warning only.

How to read it:

* A large bias is **underestimated until it is corrected** (about 0.8 degrees reported for a
  true 2 degrees with the correction off): far points are too far from their lines to be
  matched, and the localizer's pose has already bent to hide part of the error. With the
  correction on, each step improves the next estimate and the loop converges.
* The reported standard deviation is how repeatable the fit is, not how accurate.
* It cannot tell a pitch error from a camera-**height** or focal-length error; all stretch
  ranges. A height error is partly absorbed as pitch.
* It also absorbs the IMU's accelerometer bias, which looks like a constant body tilt. That is
  wanted.
* It collects nothing until the localizer has converged, so it cannot rescue a filter that is
  already lost.
* It uses no ground truth; a test replays recorded sensor packets into a fresh estimator with
  the simulator closed and gets identical estimates.

## 8. Where ground truth is, and is not

| Uses ground truth | Does not |
|---|---|
| rendering, the simulated IMU, odometry generation | the tilt filter, synchronisation, camera-pose estimator |
| the benchmark's scoring | feature extraction, the localizer |
| the viewer's `TruthView` (display only) | the mount monitor and its correction |
| the benchmark's `oracle_orientation` arm, by name | every other arm |

`tests/test_ground_truth_isolation.py` scans the estimators' imports, runs them with MuJoCo and
the simulation package blocked, and pins their inputs to sensor types.
`tests/test_humanoid_fusion.py` checks that the robot is told the believed mount and never the
true one, that an oracle pose is refused by every non-oracle arm, and that recorded sensor
packets alone reproduce a run.
