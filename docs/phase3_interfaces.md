# Interfaces for Phase 3

> **Status: Phase 3 is built.** This document was written at the end of Phase 2 to describe
> the hand-over, and is kept because its description of the two interfaces is still accurate.
> Every step in "The gap" below now exists; the last column of that table says where. For how
> it works see [geometric_perception.md](geometric_perception.md) and
> [vision_localization.md](vision_localization.md); for how well,
> [phase3_validation.md](phase3_validation.md).

Phase 2 ends at "image in, boxes and masks out". Phase 3 turns those into the relative
observations the particle filter already consumes. This document describes the hand-over: what
Phase 2 produces, what Phase 1 expects, and the steps in between.

```
                    Phase 2 (done)                         Phase 3 (to do)                 Phase 1 (done)
image ─► VisionPipeline.run ─► VisionResult ─► geometric feature extraction ─► ObservationFrame ─► ParticleFilter.step
         detector + segmenter   boxes, mask,     ground projection, line points,   landmarks +
                                probabilities    junctions, circle, posts          line points
```

Nothing in `datasets/`, `detection/`, `segmentation/` or `inference/` imports the localizer or
builds an `ObservationFrame`; `tests/test_vision_pipeline.py` enforces that.

## What Phase 2 hands over

`inference/pipeline.py::VisionPipeline.run(image_rgb) -> VisionResult`. The input is one
`(H, W, 3)` uint8 RGB image and nothing else: no pose, no simulator state.

```python
from robosoccer_perception.inference.pipeline import VisionPipeline

pipeline = VisionPipeline.from_checkpoints(
    "outputs/detection/yolo11n/weights/best.pt", "outputs/segmentation/unet_small/best.pt")
result = pipeline.run(image_rgb)

result.ball                        # Detection | None: the most confident ball
result.goal_posts, result.robots   # tuple[Detection, ...], most confident first
result.segmentation.mask           # (H, W) uint8: 0 background, 1 field, 2 field_line
result.segmentation.line_mask      # (H, W) bool
result.segmentation.line_confidence  # (H, W) float32 in [0, 1]
result.timing_ms                   # {"detector": ..., "segmenter": ..., "total": ...}
```

| Type (`inference/types.py`) | Fields |
|---|---|
| `Detection` | `class_id` (0 ball, 1 goal_post, 2 robot), `confidence`, `bbox_xyxy`; helpers `center`, `bottom_center`, `size`, `class_name` |
| `Segmentation` | `mask`, `probabilities` `(3, H, W)`; helpers `field_mask`, `line_mask`, `line_confidence`, `field_confidence` |
| `VisionResult` | `width`, `height`, `detections`, `segmentation`, `timing_ms`, `source`; `to_dict()`, `save()` |

Conventions that matter downstream:

- All coordinates are pixels **of the image that was passed in**, whatever size the models work
  at internally.
- Boxes are in pixel-edge coordinates (image spans `[0, W] x [0, H]`). `geometry/camera.py`
  uses pixel-centre coordinates. **Subtract 0.5** before calling `CameraIntrinsics.ray`.
- `VisionResult.save` writes `<name>.json`, `<name>_mask.png` (class ids) and
  `<name>_line_confidence.png` (0-255); the JSON carries `vision_result_version`.
- Either model can be missing (`segmentation is None`, or `detections == ()`).

The pipeline accepts any detector with `predict(image_rgb) -> list[Detection]` and any
segmenter with `predict(image_rgb) -> Segmentation`, so models can be replaced without touching
Phase 3.

## What Phase 1 expects

`observations/types.py::ObservationFrame`, in the robot's **base frame** (on the ground, +x
forward, +y left):

- `landmarks`: `LandmarkObservation(type, position (2,), range_std, bearing_std, confidence)`
  with `type` one of `GOAL_POST`, `L_INTERSECTION`, `T_INTERSECTION`, `X_INTERSECTION`,
  `PENALTY_MARK`, `CENTER_CIRCLE`. Anonymous: no landmark identity.
- `line_points`: `LinePointsObservation(points (M, 2), std (M,))`.

The filter uses the reported standard deviations to weight each measurement, and by default up
to 40 line points per update.

## The gap, step by step

| Step | Input | Output | Built in Phase 3 as |
|---|---|---|---|
| 1. Pick line pixels | `line_mask` | a bounded, spread-out set of centreline pixels | `perception/line_features.py::centreline_pixels` (distance-transform ridge; replaces the slower `inference.geometry.sample_line_pixels`) |
| 2. Project to the ground | pixels, intrinsics, `T_base_camera` | points in the base frame, with validity flags | `geometry/ground_projection.py::project_to_ground` |
| 3. Line-point uncertainty | range, camera height, pitch uncertainty | `std` per point, across the line | `projection_covariance`, `perception/uncertainty.py::line_point_std` |
| 4. Goal posts | `goal_post` boxes | foot on the ground or bearing only, `GOAL_POST` landmark | `perception/goal_features.py` |
| 5. Junctions L / T / X | projected line points | landmark observations | `perception/line_fitting.py`, `perception/intersections.py` |
| 6. Centre circle | projected line points | `CENTER_MARK` | `perception/line_fitting.py::fit_circle`. Penalty marks: **not built** |
| 7. Assemble `ObservationFrame` | all of the above | `ObservationFrame` | `perception/observation_builder.py` |
| 8. Close the loop | frames, odometry | pose estimates versus ground truth | `pipeline/vision_localization.py`, `vision_benchmark/` |

The Phase 2 utilities for steps 1 and 2 (`inference/geometry.py`) remain as inspection tools:
`scripts/run_vision.py --ground` draws the projected line pixels over the field with the TRUE
camera pose. The Phase 3 pipeline does not use them; it never has the true camera pose.

```python
from robosoccer_perception.inference import geometry

pixels = geometry.sample_line_pixels(result.segmentation.line_mask, max_points=200)   # (N, 2) u, v
T_base_camera = camera_model.T_base_camera("left", rig)       # from head kinematics, NOT from the simulator
points, valid = geometry.pixels_to_ground(pixels, camera_model.intrinsics, T_base_camera)
```

### Notes for whoever builds Phase 3

(Written before Phase 3. What became of each note: pitch sensitivity is measured in the
calibration sweep; uncertainty is derived as suggested; goal posts fall back to bearing only,
and their foot turned out NOT to be the box's bottom centre because posts lean in the image;
the ball and stereo are still unused; symmetry is preserved and reported.)

**The camera pose relative to the ground must come from the robot**, not from ground truth:
head joint angles and, once the robot walks, the IMU's roll and pitch. Ground projection is
very sensitive to pitch. At 0.9 m camera height a 1 degree pitch error moves a point 5 m away
by about 0.5 m. Phase 1's oracle models this as `pitch_jitter_std_deg`; real values will be
needed.

**Uncertainty grows fast with distance.** One pixel covers roughly
`(range^2 + height^2) / (fx * height)` metres of ground along the viewing direction: about
2 cm at 2 m, 9 cm at 5 m and 22 cm at 8 m for the 640-px-wide images (fx = 320, camera 0.9 m
up). Step 3 should derive `std` from this plus the pitch uncertainty, and
should probably drop line points beyond some range. The segmentation evaluation reports line
recall and precision by distance for the same reason.

**Goal posts.** `Detection.bottom_center` is where the post meets the turf *only if the foot is
visible*. If the box touches the bottom image border, or a robot stands in front of the foot,
the bottom of the box is not the foot and projecting it gives a wrong range. Check the box
against the image border, and consider using the bearing only (it stays valid) when the foot is
not trustworthy. The stereo pair can give range independently of the foot.

**The ball** is not a localization landmark, but the same projection gives its position: the
ball's centre is `ball.radius` above the ground, so intersect the ray through `center` with the
plane `z = radius`, not `z = 0`.

**Robots** are obstacles, not landmarks. They also explain missing line pixels: the
segmentation labels a robot's pixels as `background`, so lines stop at robots by design.

**Line pixels near the horizon** are unreliable twice over: thin in the image (low recall) and
enormous on the ground. `pixels_to_ground` marks rays at or above the horizon as invalid;
anything within a few degrees below it deserves the same treatment.

**Stereo.** Phase 2 uses the left image only. The right image and the disparity relation in
`geometry/camera.py` are available for range that does not depend on the ground-plane
assumption.

**Symmetry.** Nothing in the vision output distinguishes the two halves of the field: both
goals are white and the lines are mirror images. The 180-degree ambiguity documented in
[localization.md](localization.md) remains, and must not be resolved with simulator ground
truth.

**Evaluating Phase 3.** The dataset manifest stores, per frame, the true camera pose
(`T_field_left_optical`), the platform pose and rig state, and every object's true position.
That is what to score extracted observations against. Those fields are for scoring only; they
must not reach the models or the localizer.

**Timing budget.** Vision takes the time reported in
[vision_validation.md](vision_validation.md) per frame on the development laptop; the particle
filter takes 2 to 6 ms per update. Whatever Phase 3 adds sits between the two.

## What Phase 2 deliberately did not do

(All but the last two items are now done by Phase 3, outside the Phase 2 packages.)

- No conversion of detections or masks into `LandmarkObservation` / `LinePointsObservation`.
- No junction, circle or penalty-mark extraction.
- No data association and no changes to the particle filter.
- No ROS 2 nodes or messages and no ZED SDK. The types are plain dataclasses and NumPy arrays
  with REP-103 axes, so wrapping them later is mechanical.
- No use of the right camera image.
