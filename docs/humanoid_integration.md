# The humanoid in the perception simulator

Phase 4 puts UTRA's `humanoid_prototype_1` on the simulated field in place of the capsule that
carried the camera in Phases 1 to 3. This page records what the model is, what was taken from
it, what was left out, and how the body is moved. The camera mount has its own page,
[camera_mount.md](camera_mount.md); motion, the IMU and the estimators are in
[humanoid_fusion.md](humanoid_fusion.md).

**In one paragraph.** The official model loads in this repository's MuJoCo scene as a
*kinematic, visual-only* body: all 21 link meshes triangle for triangle, the complete joint
tree, the IMU site. It is posed, never simulated. It cannot balance, fall or collide here, and
the stepping it shows is an animation, not a gait. The robot has no head; the camera sits on a
configurable virtual mount above the torso. The placeholder platform remains available, and
scenes built without a robot are the Phase 1 to 3 scenes, unchanged.

## Model audit

Source: `soccer-software`, `src/humanoid_description/robots/humanoid_prototype_1/`
(`mjcf/robot.xml`, `mjcf/scene.xml`, `urdf/robot.urdf`, `meshes/`), documented in that
repository's `docs/humanoid-prototype-1.md`. Audited at commit `bb4a3bb`.

| | Found |
|---|---|
| Bodies / joints | 22 bodies; a floating base and 20 hinge joints (16 driven, 4 passive in the two ankle four-bars) |
| Mass | 13.94 kg (the documentation's 13.9 kg) |
| Visual geometry | 21 meshes, 307 874 triangles |
| Collision geometry | 243 geoms (CoACD convex pieces, one box per foot) |
| Constraints | two `connect` equalities that close the ankle four-bars |
| Sensors | `framequat`, `gyro`, `accelerometer` on the site `imu` |
| IMU | a placeholder: the site `imu` at the pelvis origin, identity rotation. The CAD has no IMU |
| Head | none |

**Frames.** The floating base is the body `pelvis`. Its frame sits midway between the two
hip-pitch axes at hip height, +x forward, +y left, +z up (REP-103): the same axis convention as
this repository's `base` frame, so no axis swap is needed. Joint axes follow the
documentation: hip pitch about +y, hip roll about +x, hip yaw about +z, knee and ankle about
+y. At zero the legs are straight and the arms are in the CAD's T-pose.

**Geometry that matters for a camera.** Measured from the meshes by the importer, not typed in:

| Quantity | Value |
|---|---|
| Pelvis origin above the ground in the stand pose (`home` keyframe) | 0.4057 m (the official keyframe says 0.405699) |
| Top of the torso mesh above the pelvis origin | 0.2730 m |
| Top of the torso above the ground, stand pose | 0.679 m |
| Top of the torso above the ground, legs straight | 0.686 m |
| Width, arms lowered as in the stand pose | 0.55 m |

The documentation gives the robot's height as "0.72 m". The highest point of the model's own
geometry is 0.686 m with straight legs and 0.679 m in the stand pose. This repository uses the
measured numbers and does not use 0.72 m anywhere. Either way that number is not a camera
height: there is no head.

**The downloaded copies.** `~/Downloads/humanoid_prototype_1_model/humanoid_prototype_1` is
byte-identical to the repository model (`robot.xml`, `scene.xml`, `robot.urdf` and all 261 mesh
files have the same hashes). `~/Downloads/raw_cad_fusion_export/Assembly` is the Fusion 360
export that the repository ingests (`model/robots/humanoid_prototype_1/source/`): the 105
meshes are identical and `Assembly.xml` differs only in line endings (CRLF). The repository
model is the one that was imported. None of the three was modified.

## What was integrated

`scripts/import_humanoid_model.py` reads the official model and writes a bundle to
`assets/robots/humanoid_prototype_1/`:

| File | Content |
|---|---|
| `skeleton.xml` | bodies, inertials, hinge joints and sites of `robot.xml`, nothing else |
| `visual_meshes.npz` | the 21 visual meshes: the STL triangles with duplicated corners merged. No vertex is moved and no triangle dropped (1.9 MB instead of 14 MB) |
| `body_geometry.yaml` | the measured standing height, torso top and IMU pose |
| `manifest.json` | source commit, SHA-256 of `robot.xml` and of every mesh, triangle counts, the `home` stance, and the lists below |

| Taken over, unchanged | Left out |
|---|---|
| every body, joint axis, joint range, frame offset | the floating-base joint (the pelvis is placed directly) |
| every link's visual mesh | all 243 collision geoms and their meshes |
| inertials (kept so the tree is the official one; unused) | contacts and contact exclusions |
| the `imu` site and the linkage pin sites | the equality constraints of the ankle four-bars |
| the `home` stand pose | actuators and joint dynamics |
| | sensors (the IMU is simulated by this repository) |

The reasons for leaving things out are the ones the model's own documentation gives: actuator
parameters and joint ranges are provisional, and collision fidelity matters for walking and
getting up, not for what a camera sees. Nothing was "fixed".

To re-import after the official model changes:

```bash
python scripts/import_humanoid_model.py \
    --source <soccer-software>/src/humanoid_description/robots/humanoid_prototype_1
python scripts/import_humanoid_model.py --check     # stored geometry against a fresh measurement
```

The source is only read. `tests/test_humanoid_model.py` fails if the stored body geometry no
longer matches what the model measures, or if the stand pose in `configs/robots/` differs from
the model's `home` keyframe.

## How the body is moved

Nothing is simulated. `MujocoScene(robot=..., mount=...)` hangs the pelvis from a MuJoCo mocap
body; each frame the scene is given

* the footprint pose `[x, y, yaw]` (as in Phases 1 to 3),
* `T_footprint_body`: the pelvis in the footprint frame (height, roll, pitch),
* joint angles,

and only forward kinematics is evaluated.

**Ankle four-bars.** With the equality constraints gone, the horn and rod of each ankle would
come apart when the ankle pitches. `HumanoidBody` closes each linkage geometrically: for the
ankle's whole range it solves the two passive angles at which the rod's pin meets the horn's
pin and tabulates them. The pins stay within 0.2 mm (tested).

**Stepping.** While the body "walks", `HumanoidBody.joints_for` moves the legs with planar
two-link inverse kinematics so that the feet stay under the pelvis and near the ground (within
2 cm in the tests), lifts the swing foot, counter-rolls the hips and swings the arms. This is
**cosmetic**. It is phase-locked to the synthetic body motion so that the picture is coherent,
and it has no effect on the camera pose or the IMU, which come from the body motion alone. It
is not a gait and says nothing about how the robot will walk.

**The placeholder** (`configs/robots/placeholder.yaml`) is the capsule of Phases 1 to 3 with an
*assumed* body frame at hip height, so that it can tilt like a body. With the mount
`legacy_rig` its camera is exactly the Phase 1 to 3 rig, 0.90 m above the ground.

## Cost

Measured on the development laptop (RTX 2050), 640 x 360:

| | |
|---|---|
| Build the scene with the humanoid | 0.8 s (0.01 s with the placeholder) |
| Render one camera image | 6 to 7 ms, the same as without the humanoid within noise |
| Third-person picture (viewer only) | 5 to 6 ms |

The full 308 000-triangle model costs about half a millisecond per frame, so no simplified
model is used. A simplified-collision configuration was not needed because no collision is
evaluated at all.

## Limits

* **Visual, not physical.** The body cannot fall, slip, push or be pushed. Disturbances are
  scripted tilts ([humanoid_fusion.md](humanoid_fusion.md)).
* **Appearance is assumed.** The meshes carry no colour or material; every part is drawn in one
  dark grey. Cables, electronics and the battery are not in the CAD export.
* **The robot does not see itself in the default mounts.** From a head above the torso looking
  25 degrees down, the arms and feet are outside the image, so the body does not appear in its
  own camera (nothing hides it: with a mount that looks at the feet, it does appear, and a test
  checks that). A real head may see the arms when they are raised; that is not exercised.
* **The other robots on the field are still the placeholder humanoids** of Phase 2. The
  detector was trained on those, not on this robot's shape.
* **No head, no neck.** The camera is rigid to the body. See [camera_mount.md](camera_mount.md)
  for how a head joint would be added.
