# Geometric perception (Phase 3)

How boxes and masks from the Phase 2 networks become metric measurements in the robot's frame:
where each feature is, how sure we are, and when we refuse to say.

```
VisionResult ──► GeometricFeatureExtractor.extract(vision, calibration, camera_pose) ──► FeatureSet
(boxes, masks)        perception/extractor.py                                            line points, goal posts,
                                                                                         junctions, circle
```

Code: `geometry/ground_projection.py` (pure NumPy) and the `perception/` package (NumPy and
OpenCV). Nothing here imports the simulator; the inputs are image predictions, the robot's own
belief about its camera, and the static field description. Results are measured in
[phase3_validation.md](phase3_validation.md).

## What the robot is allowed to know

| Known | Where it comes from |
|---|---|
| The field map: line layout, centre-circle radius, goal-post diameter and height | `FieldGeometry`, the rulebook |
| Camera intrinsics and the nominal mount | `EstimatedCameraCalibration`, from `configs/cameras/*.yaml` |
| Head pan and pitch | the robot commands them; `EstimatedCameraCalibration.pose(pan, pitch)` |
| How far each of these is trusted | `CalibrationUncertainty` in `configs/perception/measurement_noise.yaml` |

Not known, and never passed in: the robot's pose on the field, the true camera pose, what is
in the image, which landmark is which.

The true camera of the simulator is a different type, `SimulationCameraTruth`
(`vision_benchmark/camera_truth.py`), with different field names. Every function that takes a
camera pose calls `require_estimate`, which rejects anything that is not an
`EstimatedCameraCalibration` / `EstimatedCameraPose`.

## Ground projection

Frames are those of [coordinate_frames.md](coordinate_frames.md): the base frame is on the
ground under the robot (+x forward, +y left); the optical frame has +z forward, +x right, +y
down; pixels are `(u, v)` with integer values at pixel centres.

A pixel has the viewing ray

```
d_c = [(u - cx) / fx, (v - cy) / fy, 1]            in the optical frame
d_b = R d_c,  origin o                              in the base frame   (R, o from T_base_camera)
```

It meets the ground `z = 0` at

```
s = -o_z / d_b,z          point = o_xy + s * d_b,xy          (only if d_b,z < 0)
```

`T_base_camera` is built with the same chain as the renderer's camera model
(`Trans(forward, lateral, height) Rz(pan) Ry(pitch) Rx(roll) Trans(0, baseline/2, 0) R_BODY_OPTICAL`),
so there is one convention, not two (`tests/test_ground_projection.py` checks the chains are
equal to 1e-12). MuJoCo's horizontal principal-point sign, found in Phase 1, is handled where
the scene is built; a rendering test back-projects white pixels of an image with deliberately
asymmetric intrinsics and checks that they land on paint, and that they do not when `cx` is
mirrored.

`project_to_ground` is vectorized and returns, per pixel, the point, a validity flag and the
reason for rejection:

| Flag | Meaning | Default limit |
|---|---|---|
| `ABOVE_HORIZON` | the ray does not point down | |
| `NEAR_HORIZON` | it does, but at less than `min_depression_deg` | 4 deg |
| `TOO_FAR` / `TOO_CLOSE` | outside `[min_range, max_range]` from the robot | 0.25 m to 6 m |
| `OUTSIDE_IMAGE` | within `image_margin_px` of the border, or outside | 2 px |
| `UNSTABLE` | non-finite pixel or result, or a camera height that is not positive | |

`ground_to_pixels` is the inverse, by the robot's own calibration. `EstimatedCameraPose` also
offers `perturbed(d_height, d_pitch, d_roll, d_yaw)` (used for uncertainty and for the
calibration experiments) and `with_base_tilt(roll, pitch)` for an IMU-measured torso tilt.

### Why pitch matters most

For a point straight ahead at ground range `r` from a camera at height `h`:

```
d r / d pitch = (r^2 + h^2) / h         one pixel along the view covers  (r^2 + h^2) / (f h)
```

With `h = 0.9 m` and `f = 320 px` (640-pixel-wide images):

| Range | 1 px along the view | 0.5 deg of pitch | 2 deg of pitch |
|---|---|---|---|
| 1 m | 0.6 cm | 1.8 cm | 7 cm |
| 2 m | 1.7 cm | 4.7 cm | 19 cm |
| 4 m | 5.8 cm | 16 cm | 65 cm |
| 6 m | 12.8 cm | 36 cm | 143 cm |

A height error scales every range by `h_believed / h_true`: 5 cm at 0.9 m is 5.6 %.

## Uncertainty model

Every feature gets its standard deviations from the same three ingredients
(`geometry/ground_projection.py::projection_covariance`, `perception/uncertainty.py`):

1. **Pixel part.** The feature's pixel is known to `pixel_std` (1 px for line centres, 1.5 px
   for junctions, 1.5 x 3 px for a post's foot: its row is the weaker coordinate). Mapped to
   the ground through the Jacobian of the projection. Independent from one feature to the next.
2. **Camera part.** The effect of the believed 1-sigma errors of height (2 cm), pitch, roll and
   yaw (0.5 deg each) and focal length (1 %) at that pixel. Common to every feature of a
   frame: a pitch error moves all of them together.
3. **Feature terms.** A floor (a 5 cm line has no sharper centre than about 2 cm), division by
   `confidence ^ 0.5` (confidence clipped at 0.25), an inflation of 1.5 for features that reuse
   the line pixels (junctions, the circle), and penalties for truncation and occlusion.

Jacobians are numerical (central differences), cross-checked against the closed form for
pitch. The two parts are kept apart in `ProjectionCovariance` (`pixel`, `camera`, `total`),
and reduced to what the localizer's schema takes:

- **line points**: one number, the 1-sigma error **across the line**. What the localizer
  scores is a point's distance to the nearest line, so only the error across the line counts.
  A line running straight away from the robot is almost unaffected by a pitch error; a line
  across the view takes all of it. Where a point lies on a fitted segment or on the circle,
  the covariance is projected onto that line's normal; elsewhere the direction-free RMS is
  used. Points with more than 0.45 m are dropped.
- **landmarks**: `(range_std, bearing_std)`, the covariance rotated into the viewing direction.

Approximations, on purpose:

- First order, Gaussian.
- **The camera part is added to each feature as if it were independent**, which it is not. A
  hundred projected line pixels are not a hundred measurements. Over-confidence is limited by
  (a) keeping at most 60 line points per frame, no two closer than 0.20 m; (b) the localizer's
  existing likelihood tempering (`line_point_weight: 0.3`) and outlier floors, unchanged from
  Phase 1; (c) inflating junctions and the circle, which are derived from the same pixels.
- No covariance between features is carried, and no graphical model is built.

The reported values are conservative in the clean simulator (more than 99 % of line-point
errors are within one reported sigma; see the validation document). That is deliberate: the
belief of 0.5 deg of pitch error is meant for a real head, not for the simulated one.

## Field-line points

`perception/line_features.py`, then the extractor:

1. Threshold the U-Net's line probability at 0.5; drop connected blobs smaller than 12 px.
   Masks wider than 640 px are reduced first, because the segmenter works at 640 px anyway.
2. **Centreline by distance transform**: a line pixel is kept when no neighbour is farther
   from the edge of the line band than it is. A near line is a band many pixels thick; this
   takes its middle. (Morphological thinning in NumPy costs tens of milliseconds; this costs
   about one.)
3. Keep one centreline pixel per 4 x 4 px cell, the most central one.
4. Project to the ground; reject by the table above.
5. For fitting: at most one point per 6 cm cell on the ground. For the localizer: a subset in
   which **no two points are closer than 0.20 m**, at most 60, preferring confident pixels.
6. Uncertainty across the line, as described above.

Fragmented lines, lines hidden behind a robot, and missing stretches need no special case:
points are independent, and a line that is not seen contributes nothing. Small false positives
are removed in step 1; larger ones become outliers, which the localizer's likelihood floor
bounds.

## Straight segments and the centre circle

`perception/line_fitting.py`. Fitting happens **on the ground**, where lines are straight,
meet at right angles, and the circle has a known radius, so every threshold is a length.

- **Segments**: sequential RANSAC on points within 4.5 m. Each point's tolerance is 3 cm
  combined with its own pixel noise (far points are judged more leniently; camera-pose error is
  not in the tolerance, because it moves a whole line and leaves it straight). The best line
  is refitted by total least squares, cut where its support has a gap of more than 0.30 m
  (plus the natural spacing of far pixels), and pieces of at least 0.5 m and 6 points are
  kept. Collinear pieces are merged.
- **Circle**: RANSAC for a circle of the **known radius** (two points give two candidate
  centres), refined by Gauss-Newton. It is accepted only if the supporting points span at
  least 150 deg, fill at least half of that span, number at least 14, **and** make up at least
  75 % of all points within 15 cm of the circle.

The last test exists because of a failure found during development: a circle of the
centre-circle radius can be laid tangent to two or three straight lines of the nested penalty
and goal areas and collect 150 deg of "support" where it touches them. Before the test 14 of
25 accepted circles were false; with it, tangent lines are recognised because they continue
past the touching point, just outside the tolerance, where a real circle has nothing.

A short arc is never turned into a centre: under 150 deg the candidate is reported with
`valid = False` and is not sent to the localizer.

## Junctions

`perception/intersections.py`. Two segments meeting within 20 deg of a right angle form a
junction at the crossing of their lines. Its class follows from where the crossing lies on
each segment:

```
the line continues >= 0.30 m on both sides     -> it goes THROUGH
the line starts or stops at the crossing       -> it ENDS

end + end -> L          through + end -> T          through + through -> X
```

The hard part is not finding crossings but **not inventing classes**. A line that leaves the
image, disappears behind a robot or runs out of the usable range also "ends", and would turn a
T into an L. So an end counts only when the ground 0.20 m and 0.35 m past it is **positively
seen to be turf** in the predicted mask (inside the image, away from its border, class
`field`). If that cannot be established, or an arm is too short to judge, the junction is kept
with `type = None` and a flag (`end_not_observed`, `end_not_on_turf`, `short_arm`) and is
**not** sent to the localizer. Candidates at one place that disagree about the class cancel.

X junctions: the field has exactly two, where the halfway line crosses the centre circle.
They are placed from an accepted circle and a segment through its centre that reaches past
the circle (`centre +- radius` along the line).

A junction is "a T, 2.1 m ahead": never a specific T of the field.

## Goal posts

`perception/goal_features.py`. A box does not say where a post stands. Two facts make it
usable.

**A post is vertical, so in the image it leans.** With the camera pitched down, images of
vertical lines converge on the vertical vanishing point, far below the image. A post away from
the image centre leans outward; its box is several times wider than the post, and the foot is
at the **inner bottom corner**, not at the middle of the bottom edge. This was found the hard
way: the first implementation used the bottom-centre and was off by up to 5 deg in bearing.
`foot_pixel` solves for the foot column and the post's apparent width from the box and the
vanishing point, which follows from the estimated camera orientation alone:

```
lean = (u_foot - u_vp) * (y1 - y0) / (v_vp - y1)          box width = lean + post width
```

The foot is projected to the ground and moved half a post diameter along the view (the lowest
visible point of a round post is its near side).

**Every point of a vertical post has the same bearing.** So when the foot cannot be trusted,
the bearing is still good.

| Situation | Detected by | Result |
|---|---|---|
| Box touches the bottom border | box vs image size | bearing only (`cut_bottom`) |
| Foot above the horizon | projection | bearing only (`foot_above_horizon`) |
| Apparent width far from a 10 cm post at that range | known diameter, ratio outside 0.45 to 2.2 | bearing only (`width_mismatch`) |
| Box much shorter than a 1.25 m post, whole post in view | known height, ratio below 0.6 | bearing only (`height_mismatch`) |
| Not turf or line just under the foot | predicted mask, 9 x 5 px patch | bearing only (`foot_not_on_turf`) |
| Foot farther than 8 m | projection | bearing only (`beyond_range`) |
| Box touches a side border | box vs image size | width assumed from the range; bearing std widened (`cut_side`) |
| A robot or ball box overlaps the foot | other detections | range std doubled (`occluder_near_foot`) |
| Confidence below 0.35, box wider than tall, foot beyond 14 m or closer than 0.3 m | | rejected |

A bearing-only post is passed on as a point on the bearing ray at a rough range (from the
post's apparent width when available) with a **range standard deviation of at least 1.5 m**,
so that only its direction constrains the localizer. No change to the observation schema is
needed.

Which of the four posts was seen is not decided here and cannot be.

## What is handed to the localizer

`perception/observation_builder.py::build_observation_frame` reduces a `FeatureSet` to the
Phase 1 `ObservationFrame`. `ObservationSelection` (the `observations:` section of
`configs/perception/feature_extraction.yaml`) chooses what is passed on.

| Feature | Becomes | Left out when |
|---|---|---|
| Line points | `LinePointsObservation(points, std)` | |
| Goal post with range | `LandmarkObservation(GOAL_POST, position, range_std, bearing_std)` | |
| Goal post, bearing only | the same, with `range_std >= 1.5 m` | `bearing_only_goal_posts: false` |
| Junction with a class | `L_INTERSECTION` / `T_INTERSECTION` / `X_INTERSECTION` | no class could be established |
| Accepted circle | `CENTER_MARK` at the fitted centre | the candidate failed a check |

Penalty marks are not extracted (a 15 cm cross is a handful of pixels beyond 2 m); the
`PENALTY_MARK` class is simply never reported. The ball and other robots are not localization
landmarks; robot and ball boxes are used only as possible occluders of a post's foot.

## Cost

About 10 ms per 640x360 frame on one CPU core (measured in the validation document): centreline
extraction about 2 ms, covariance about 3 ms, segment and circle fitting about 3 ms, the rest
junctions and posts. Bounded by construction: at most a few hundred candidate points, a fixed
number of RANSAC hypotheses, at most 60 line points out.

## Known limits

- **Flat ground, level torso.** The ground is `z = 0` of the base frame. A walking robot's
  torso tilts; `with_base_tilt` takes IMU roll and pitch, but nothing in the simulator
  exercises it beyond a unit test.
- **No lens distortion.** Real ZED Mini images must be rectified first.
- **A systematic pitch or height error is not detected.** It biases every range in the same
  direction, and the reported uncertainty does not grow. See the calibration sweep.
- **A false centre circle is still accepted occasionally**: once in 200 to 400 frames on some
  benchmark trajectories, none in 400 random views. The cases examined were circles laid
  through the corner region of the penalty and goal areas 4 to 5 m away with 150 to 165 deg
  of support, just over the threshold. With a 2 deg pitch error the rate is far higher
  (26 false against 12 correct in 400 random views).
- **Junction recall is limited by design**: an arm shorter than 0.30 m in view, or an end
  that cannot be verified, gives no class.
- **Goal posts are assumed vertical and of rulebook size.** A leaning or non-standard goal
  breaks the width and height checks (they fall back to bearing only).
- **Thresholds were set on the simulator.** They are lengths and pixel counts with physical
  meaning, but none has seen a real image.
