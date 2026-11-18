# The virtual head and the camera mount

**The head of `humanoid_prototype_1` has not been designed.** Nothing on this page is a
mechanical specification, and no number here should be read as the final position of the ZED
Mini. What exists is a *configurable* rigid transform from the robot's body to the camera, with
a handful of experimental presets, so that perception can be studied now and the real transform
dropped in later.

## Frames

```
field <- footprint <- body (pelvis) <- rig <- {left, right}_optical
```

| Frame | Where | Changes |
|---|---|---|
| `footprint` | on the ground under the pelvis origin, +x along the body's heading, +z up. Its pose `[x, y, yaw]` is what the localizer estimates. Earlier documents call it `base` | as the robot moves |
| `body` | the model's `pelvis` frame: midway between the hip-pitch axes, +x forward, +y left, +z up. `T_footprint_body = Trans(0, 0, height) Ry(pitch) Rx(roll)` | with every step and sway |
| `rig` | midpoint of the two lenses, +x along the viewing direction | never (rigid mount) |
| `*_optical` | ROS optical frame: +z forward, +x right, +y down | never |

The mount is `T_body_rig`. Four things are kept apart in the code (`geometry/mount.py`):

1. **Body orientation**: roll and pitch of the pelvis. An IMU on the body can estimate it.
2. **Mount orientation**: the fixed rotation of the camera on the body. A property of the
   mechanics, known only as well as it was calibrated.
3. **Camera orientation**: the product of the two. Ground projection needs this.
4. **Mount calibration error**: true mount minus believed mount. The IMU cannot see it.

## Configuration

One file per mount in `configs/mounts/`:

```yaml
name: head_mid
experimental: true
parent_frame: pelvis
reference: torso_top            # body | torso_top | ground
xyz_m: [0.03, 0.0, 0.12]        # forward, left, up from the reference, metres
rpy_deg: [0.0, 25.0, 0.0]       # roll, pitch, yaw; POSITIVE PITCH LOOKS DOWN
head_extension: true            # draw a mast from the torso to the camera (a marker)
```

* `xyz_m` gives all three offsets (forward, lateral, vertical); `rpy_deg` gives roll, pitch and
  yaw. The rotation is `Rz(yaw) Ry(pitch) Rx(roll)` in the body frame: positive pitch looks
  down, positive yaw looks left, positive roll tips the right side of the image down. This is
  the convention of `RigState` in Phases 1 to 3.
* `reference` says where the offset is measured from:
  `torso_top` is the top-centre of the torso mesh, taken from the model
  (`assets/robots/<robot>/body_geometry.yaml`); `body` is the pelvis origin; `ground` is the
  floor under the pelvis in the stand pose.
* **The camera height is never typed in.** With `reference: torso_top` it is
  `standing height of the pelvis + torso top above the pelvis + xyz_m[2]`, all but the last
  measured from the model. A mount whose height is unset (`null`) is refused with an error.
  The old 0.90 m appears only in `legacy_rig`, which is labelled as the Phase 1 to 3
  placeholder.

### Presets

All are `experimental: true`. Heights are for the stand pose of `humanoid_prototype_1`
(pelvis 0.406 m, torso top 0.679 m above the ground).

| Preset | Offset above the torso | Camera height | Note |
|---|---|---|---|
| `head_low` | 6 cm | 0.739 m | lowest head considered; below the heights the networks were trained at |
| `head_mid` | 12 cm | 0.799 m | default for Phase 4 |
| `head_high` | 18 cm | 0.859 m | |
| `head_tall` | 22 cm | 0.899 m | the height Phases 1 to 3 assumed |
| `legacy_rig` | (0.90 m above the ground) | 0.900 m | not tied to any body; the Phase 1 to 3 rig |

Every preset is 3 cm ahead of the torso centre and pitched 25 degrees down, the Phase 3 pitch,
so that results differ by height alone. A new experimental height is a new five-line file; no
code changes. The virtual ZED Mini itself (`configs/cameras/zed_mini_sim.yaml`: 63 mm baseline,
90 degree field of view) is unchanged.

**Assumptions in these presets**, all of them to be replaced by measurements: the height, the
forward offset, the 25 degree pitch, zero roll and yaw, a perfectly rigid connection to the
pelvis, and the camera's intrinsics (still the simulated profile, not a factory calibration).

## What the robot believes and what is true

A scenario names the mount the robot **believes** it has. The simulator renders with the
**true** mount, which is the believed one plus a hidden `MountError` (height, pitch, roll, yaw)
and a little per-frame flex (0.1 degree) that nothing reports. The robot's side is built from
the believed mount only; a test checks that it never receives the true one.

## Seeing it

The viewer's camera panel has a third-person inset of the robot. It draws, on the body:

| Marker | Meaning |
|---|---|
| red / green / blue sticks at the pelvis | the body frame's x / y / z axes |
| thin yellow line | the configured mount offset, from the body origin to the camera |
| grey mast | the virtual head extension: from the torso top to the camera |
| dark box | the camera housing (ZED Mini outer dimensions) |
| magenta stick | the camera's viewing direction |

These markers are in their own render group and are never visible to the robot's own camera.
The mast is a visual aid: it has no mass, no inertia, no actuator and does not collide.
`scripts/render_humanoid.py` writes the same view for every preset to `outputs/humanoid/`.

The diagnostics panel prints the mount transform in use, the camera height, and the estimated
camera pitch and roll next to the true ones.

## When the real head exists

1. Measure the transform from the pelvis frame to the midpoint of the ZED Mini's lenses
   (CAD first, then a calibration against the field or a target).
2. Write it as `configs/mounts/<name>.yaml` with `reference: body`, the measured `xyz_m` and
   `rpy_deg`, and `experimental: false`.
3. Replace `configs/cameras/zed_mini_sim.yaml` with the camera's own calibration.
4. Put the measured uncertainty of that calibration into
   `configs/perception/measurement_noise.yaml` (`calibration:`), which today holds beliefs.
5. If the head has a meshes-and-joints model, re-import the robot
   (`scripts/import_humanoid_model.py`); `torso_top` presets then stop being useful and can go.

**If the head gets neck joints**, nothing in localization changes. The estimator composes
`T_footprint_body @ T_body_rig @ T_rig_optical`; with a neck, `T_body_rig` becomes
`T_body_neck @ T_neck_rig(pan, tilt)` evaluated from the measured joint angles at the image
timestamp, exactly where `CameraPoseEstimator._pose` composes the mount today. The pipeline's
interface already takes a per-image camera pose. No neck joints are invented here: none are
documented.

**When the IMU is placed** (the CAD has none yet; the model carries a placeholder site at the
pelvis origin), re-import the model: the importer reads the `imu` site into the robot's body
geometry (`imu_xyz_m`, `imu_rpy_deg`), and the simulated IMU and the filter both take the
IMU's pose from there.
