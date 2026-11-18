# Coordinate frames

All frames are right-handed. Distances are metres, angles radians (configuration files use
degrees where the key ends in `_deg`).

## Naming

`T_a_b` is the pose of frame `b` expressed in frame `a`. It maps coordinates from `b` to `a`:

```
p_a   = T_a_b @ p_b          (4x4 homogeneous)
T_a_c = T_a_b @ T_b_c        (adjacent names cancel)
T_b_a = invert_transform(T_a_b)
```

`R_a_b` is the rotation part: its columns are the axes of `b` written in `a`. The code uses
these names literally (`T_field_base`, `T_base_rig`, ...), so a product whose inner names do not
cancel is a bug you can see.

## Frames

| Frame | Origin | Axes | ROS counterpart |
|---|---|---|---|
| `field` | centre of the playing field, on the ground | +x along the length toward the reference goal, +y toward the reference touchline, +z up | `map` |
| `base` | on the ground under the robot | +x forward, +y left, +z up (REP-103) | `base_footprint` |
| `rig` | midpoint between the two optical centres | +x viewing direction, +y left, +z up | `zed_camera_center` |
| `left_optical`, `right_optical` | optical centre of each camera | +z forward, +x right, +y down (ROS optical) | `zed_left_camera_optical_frame`, ... |
| MuJoCo camera | optical centre | +x right, +y up, looks along **-z** | none |

The field frame never changes during a match, whichever goal a team attacks. "Reference goal"
and "reference touchline" are simply the ones at +x and +y.

A robot pose is `(x, y, yaw) = T_field_base`: the base origin in the field frame and the
heading of base +x, counter-clockwise from field +x. 2-D poses are NumPy arrays `[x, y, yaw]`.

## The chain

```
T_field_base      from the pose:    Trans(x, y, 0) * Rz(yaw)
T_base_rig        from the mount:   Trans(forward, lateral, height) * Rz(pan) * Ry(pitch)
T_rig_optical     fixed:            Trans(0, +-baseline/2, 0) * R_BODY_OPTICAL
T_field_camera  = T_field_base @ T_base_rig @ T_rig_optical
```

- **Pitch** is a right-handed rotation about base +y, so **positive pitch looks down**.
- **Pan** is a rotation about base +z, so positive pan looks left.
- The left camera is at rig `+y` (the robot's left), the right camera at `-y`.
- `T_left_right` is a pure translation of one baseline along the left camera's `+x`.

Inverses: `T_base_field = invert_transform(T_field_base)` and so on;
`se2_inverse` does the same for 2-D poses.

### With a body that tilts (Phase 4)

Phase 4 puts the robot's body between the base and the rig. The `base` frame is called the
**footprint** there: the point on the ground under the body origin, turned to the body's
heading.

```
T_field_footprint   from the pose:   Trans(x, y, 0) * Rz(yaw)
T_footprint_body    from the tilt:   Trans(0, 0, height) * Ry(pitch) * Rx(roll)
T_body_rig          from the mount:  Trans(mount position) * Rz(yaw) * Ry(pitch) * Rx(roll)
T_rig_optical       fixed, as above
T_field_camera    = T_field_footprint @ T_footprint_body @ T_body_rig @ T_rig_optical
```

| Frame | Origin | Axes | ROS counterpart |
|---|---|---|---|
| `footprint` (= `base`) | on the ground under the body origin | +x along the body's heading, +z up | `base_footprint` |
| `body` | the robot's own reference: the `pelvis` of humanoid_prototype_1, midway between the hip-pitch axes | +x forward, +y left, +z up (REP-103) | `base_link` / `pelvis` |
| IMU | the model's placeholder `imu` site at the pelvis origin | as the body | `imu_link` |

- Roll, pitch and yaw are intrinsic Z-Y-X angles: `R = Rz(yaw) Ry(pitch) Rx(roll)`. With +x
  forward and +z up, **positive pitch tips the nose down** and positive roll lifts the left
  side. The body's tilt, the mount's angles and `RigState` all use this.
- `T_footprint_body` has no yaw and no horizontal offset by definition: yaw and position
  belong to the footprint pose that the localizer estimates.
- The mount's angles are in the BODY frame; a mount pitch of 25 degrees on a body pitched 3
  degrees forward looks 28 degrees below the horizon, and the camera has also moved forward
  with the torso. Body tilt and mount tilt are not interchangeable (a test checks the
  difference).
- Quaternions are `(w, x, y, z)`, MuJoCo's order; `q_a_b` rotates coordinates from `b` to `a`
  like `R_a_b` (`geometry/quaternion.py`).
- An accelerometer reports specific force: at rest and level it reads `[0, 0, +9.81]`.
- Because `Trans * Rz * Ry * Rx` is a general rigid transform, a rig on a tilted body is still
  a `RigState`; the scene, the oracle and the labels did not have to change.

## Optical frame versus MuJoCo camera frame

They are not the same, and the difference is handled explicitly in two places:

```
R_BODY_OPTICAL   = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]     optical axes written in a body frame
R_OPTICAL_MUJOCO = diag(1, -1, -1)                          half turn about x
```

- Building the scene: the camera orientation given to MuJoCo is
  `R_rig_mujoco = R_rig_optical @ R_OPTICAL_MUJOCO`.
- Reading ground truth back: `R_field_optical = R_field_mujoco @ R_OPTICAL_MUJOCO.T`.

A second difference was found by test rather than by reading: MuJoCo's `principalpixel` has
the opposite horizontal sign to OpenCV's `cx` (and the same vertical sign as `cy`). See
`_add_pinhole_camera` in `simulation/mujoco_scene.py`.

## Image coordinates

OpenCV / ROS convention: `u` to the right, `v` down, integer coordinates at pixel centres. A
point `(X, Y, Z)` in an optical frame projects to `u = fx X / Z + cx`, `v = fy Y / Z + cy`. A
symmetric image has `cx = (width - 1) / 2`.

## Odometry

`OdometryDelta(dx, dy, dyaw)` is the pose of the base at the end of an interval expressed in
the base frame at its start: `T_base(t0)_base(t1)`. Composing it onto a pose is
`se2_compose(pose, delta)`.

## What the tests pin down

`tests/test_transforms.py` and `tests/test_stereo_camera.py` check, among others:

- 90 and 180 degree rotations with hand-computed results; right-handedness of `rot_x/y/z`;
- inverses and round trips in SE(2) and SE(3), and that the two agree;
- the optical axes, and that the MuJoCo frame differs from the optical frame;
- pitch and pan directions; camera height is the rig-centre height;
- stereo geometry: parallel axes, 63 mm between optical centres, left camera on the left;
- camera-to-field projection with hand-computed pixels;
- the same quantities read back from MuJoCo match the analytic chain to 1e-9;
- **rendered** images agree with the model: a ball appears within 1.5 px of its projection in
  both images with the predicted disparity, including with `fx != fy` and an off-centre
  principal point; rendered pixels at projected line junctions are white.

## Toward ROS 2 TF

The localizer estimates `T_field_base`. In a REP-105 tree with `odom -> base_footprint` owned by
odometry, a localization node publishes `map -> odom = T_field_base @ inverse(T_odom_base)`.
`T_base_rig` becomes a static transform (plus the head joints, once the robot has them).

With Phase 4's chain: `base_footprint -> base_link` is `T_footprint_body` (height, roll and
pitch; on the robot it comes from the IMU and leg kinematics), and `base_link -> camera` is
the mount, a static transform from the head's calibration. The localization node needs the
camera pose **at the image's header stamp**, not the latest transform: on ROS that is a
`lookupTransform` with the image's time, which is what `OrientationHistory.at` does here.
