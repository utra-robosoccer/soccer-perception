# Synthetic data

How labelled camera images are generated from the MuJoCo scene: what is in the scene, how
labels are derived, what is randomized, how splits are formed, and what the files look like.

Everything here is produced by `scripts/generate_dataset.py`. Generated datasets live under
`outputs/datasets/` and are not tracked by Git.

```bash
.venv/bin/python scripts/generate_dataset.py smoke       # 64 frames, about 10 s
.venv/bin/python scripts/generate_dataset.py baseline    # 2400 frames, about 6 min, 150 MB
.venv/bin/python scripts/inspect_dataset.py outputs/datasets/baseline --count 12
```

The second command checks the dataset and writes label previews to
`outputs/datasets/baseline/previews/`. **Look at them** before training on a dataset you
changed: each shows an image next to the same image with its mask and boxes drawn on.

## What is in the scene

Audit of the Phase 1 scene, and what Phase 2 added. Phase 1 behaviour is unchanged when the
new configuration keys are absent (`tests/test_scene_objects.py` checks this).

| Item | Phase 1 | Phase 2 |
|---|---|---|
| Field surface, border strip | carpet plane with a procedural turf texture | optional soft patches (`turf.blotch`) |
| Field markings | thin boxes, exact rulebook geometry | unchanged |
| Goals | two posts, crossbar, see-through net panels | nets moved to their own geom group so label rendering can look through them |
| Ball | 7 cm radius sphere, checker texture | selectable look (`checker`, `spots`, `flat`), colours, radius; can be hidden and turned |
| Other robots | none | **humanoid placeholders**, any number, posed per frame |
| Backdrop | flat grey floor and sky | optional walls and boxes outside the carpet |
| Camera | left/right pinhole cameras on a movable rig | optional camera **roll** (`RigState.roll`) |
| Object identifiers | geom names only | per-pixel geom ids: `MujocoScene.render_segmentation` |

### The humanoid placeholder and its limits

`simulation/scene_objects.py::add_humanoid` builds a robot from a box torso with a coloured
band, a head (sphere or box), and capsule arms and legs with six hinge joints (shoulders, hips,
knees). Height, build and colours are configurable; per frame it is placed, turned, given a
limb pose, and may lie on the ground (front, back or side).

It is a stand-in, and a detector trained on it should be read accordingly:

- No meshes, hands, cables, sensors, or realistic team markings. Real robots look different.
- One body plan. A wheeled robot, a goalkeeper on its knees, or a human referee is not covered.
- Limb poses are random angles, not a gait; a "walking" pose may float a foot above the turf.
- The robot the camera is mounted on never appears in its own image (no visible arms or feet).

Swapping in a real robot model means replacing `add_humanoid`; the labelling code needs no
change because it works from rendered geom ids, whatever the geoms are.

## Labels

### Method

MuJoCo renders, besides the colour image, an image holding the **id of the geom seen at each
pixel**. All labels are read from that image (`datasets/annotation.py`), so they describe what
the camera actually sees:

- **Object boxes** are the tight box around the pixels showing an object's geoms. Nothing is
  projected from 3-D centres and no size is assumed. Hidden parts and parts outside the image
  have no pixels, so occlusion and clipping at the image border are handled by construction.
- Each object is also rendered alone, giving its un-occluded silhouette. The ratio
  visible / un-occluded is the object's `visible_fraction`.
- The **semantic mask** maps the turf geom to `field`, line geoms to `field_line`, everything
  else to `background`.

Id images cannot be anti-aliased, while colour images are. The id image is therefore rendered
`supersample` (default 2) times finer than the camera image and reduced by coverage: box edges
are known to half a pixel, and a pixel is `field_line` when at least half of it is paint.

### Classes

| Detection id | Name | What gets a box |
|---|---|---|
| 0 | `ball` | the ball |
| 1 | `goal_post` | each upright post separately; the crossbar is not part of it |
| 2 | `robot` | each other robot, all of its parts in one box |

| Mask value | Name | Pixels |
|---|---|---|
| 0 | `background` | floor outside the carpet, sky, walls, clutter, and every object on the field |
| 1 | `field` | carpet (inside the lines and the border strip) |
| 2 | `field_line` | painted lines and marks |

The ids are defined once, in `datasets/schema.py`, and used by the generator, both trainers,
both evaluators and the inference types. A model whose class names differ is refused at load
time.

### Policies (`LabelPolicy`, recorded in every dataset)

| Rule | Default | Effect |
|---|---|---|
| Occluders | fixed | Pixels showing the ball, a robot, a post or a crossbar are `background`. A line hidden by a robot is not labelled as a line. |
| Goal nets | fixed | Nets are ignored when labelling: what is seen through a net is labelled as visible. |
| Shadows | fixed | Shadows do not change labels. |
| `line_coverage_threshold` | 0.5 | Lines thinner than half a pixel (far away) fade out of the mask. |
| `min_visible_fraction` | 0.15 | An object more than 85 % hidden gets no box (reason `occluded`). |
| `min_visible_pixels`, `min_box_side_px` | 6, 2 | Objects smaller than this get no box (reason `too_small`). |

Every object of a detection class appears in the manifest for every frame, with a `reason`:
`ok` (labelled), `occluded`, `too_small`, `out_of_view`, or `hidden` (not in this frame).
Unlabelled objects that are partly in the image are treated as **ignore regions** by the
evaluation: detecting one is not counted as a false positive.

### Known label limits

- A net dims what is behind it in the colour image; labels do not reflect that.
- A post seen in two pieces around an occluder gets one box spanning both pieces.
- Boxes are axis-aligned around visible pixels; with camera roll they are not the tightest
  possible box in a rotated sense.
- Truncation is a flag (`truncated`: the un-occluded silhouette touches the image border), not
  a fraction. `visible_fraction` is relative to the part of the object inside the image.
- Motion blur and defocus (image-space, kept mild) smear object edges by up to a few pixels;
  labels are not adjusted.

### Coordinate convention of boxes

Boxes in the manifest and in `inference.types.Detection` are `x0, y0, x1, y1` in **pixel-edge**
coordinates: the image spans `[0, W] x [0, H]`. The camera model in `geometry/camera.py` uses
pixel-centre coordinates (integer = centre). Subtract 0.5 to go from the first to the second.

## Viewpoints and layouts

Camera poses are those of a humanoid's head camera, not arbitrary: the camera stands on the
carpet at head height, pitched down, panned within neck range. The mount is centred on the
Phase 1 placeholder (0.90 m high, 25 degrees down), which is **not a hardware specification**;
when the real head is designed, change `configs/cameras/zed_mini_sim.yaml` and the `camera`
section of the randomization presets.

Each frame is drawn from one of twelve layout modes (`datasets/sampling.py`); the mode is
stored with the frame so that results can be broken down by it.

| Mode | Share | What it sets up |
|---|---|---|
| `random` | 30 % | camera anywhere on the carpet, any heading; ball and robots mostly in view |
| `ball_far` | 8 % | ball 4 to 10 m away |
| `ball_near_line` | 8 % | ball on or touching a painted line |
| `ball_occluded` | 8 % | a robot between the camera and the ball |
| `ball_image_edge` | 5 % | ball cut by the left, right or bottom image border |
| `goal_view` | 10 % | looking toward a goal from 1 to 10 m |
| `goal_post_edge` | 5 % | a goal post at the left or right image border |
| `robot_on_line` | 7 % | a robot standing on a line |
| `robot_close` | 5 % | a robot within a metre: large, cut off |
| `empty_turf` | 4 % | camera pitched down, no ball, no robots |
| `midfield` | 5 % | camera near the centre circle |
| `touchline` | 5 % | camera at a touchline, looking along it or into the field |

A mode only arranges the scene. Whether an object ends up visible is decided by rendering; the
label never comes from what a mode intended.

## Domain randomization

Three presets in `configs/randomization/`. Every sampled value is stored in the dataset.

| | `none` | `mild` | `strong` |
|---|---|---|---|
| Purpose | debug labels; the "no randomization" arm | one well-lit indoor field, varied | wider than any single venue |
| Camera height | 0.90 m | +/- 4 cm (0.78-1.02) | +/- 10 cm (0.60-1.30) |
| Camera pitch | 25 deg | +/- 5 deg (10-42) | +/- 9 deg (2-55) |
| Camera roll | 0 | +/- 1 deg (max 3) | +/- 3 deg (max 8) |
| Head pan | +/- 30 deg | +/- 35 deg | +/- 60 deg |
| Turf colour, stripes, grain, patches | default | narrow | wide |
| Line brightness and tint | default | 0.85-0.96 | 0.72-1.0 |
| Light direction, intensity, tint, shadows on/off | default | moderate | wide |
| Floor, sky, walls, clutter | default (none) | some | more, with white and green distractors |
| Goal colour, net opacity | default | narrow | wide |
| Ball look, colours, radius | checker, 7 cm | 3 looks, 6.6-7.4 cm | 3 looks, 6-11 cm |
| Robot size, build, colours | default | 0.55-0.95 m, greys | 0.45-1.4 m, some colourful |
| Exposure, gamma, white balance, saturation, vignetting | off | mild | strong |
| Sensor noise, defocus, motion blur, JPEG artefacts | off | mild | strong |

Scene properties are sampled once per scenario and baked into the MuJoCo model; camera and
image properties are sampled per frame.

**Photometric versus geometric.** The generator applies only photometric image operations
(`image_aug.apply_photometric`), which cannot move a label. Geometric operations (`hflip`,
`crop`, `resize`, `pad_to_multiple`) are used at training time and always take the mask and
boxes along; `tests/test_image_aug.py` checks that they stay aligned. (Those tests caught a
half-pixel shift between images and masks caused by OpenCV's default nearest-neighbour
resizing; masks now use the centre-aligned variant.)

Not randomized, by choice: field geometry, goal dimensions, line width, lens distortion.

## Splits and leakage

A dataset is made of **scenarios**. A scenario is one draw of the scene appearance (turf,
lighting, backdrop, ball, robot looks) plus a handful of frames from unrelated viewpoints.

- Splits are sets of whole scenarios. No scenario contributes to two splits, so a validation or
  test image never shares its turf texture, lighting or robot looks with a training image.
- Frames inside a scenario are independent draws of camera pose and layout, not consecutive
  frames of a trajectory, so there are no near-duplicates within a split either.
- Seeds are `[dataset seed, split, scenario]`. A scenario's content depends only on those
  three numbers: adding scenarios, or generating splits in another order, changes nothing that
  already exists.
- A split can use its own randomization preset (`randomization:` under the split), for example
  to train on `mild` and test on `strong`.

What this does not give you: evaluation on a different *kind* of world. Every split comes from
the same simulator, the same robot placeholder and the same renderer. See
[vision_validation.md](vision_validation.md).

## Files

```
outputs/datasets/<name>/
  dataset.json       header: versions, configuration, field, camera, seed, label policy,
                     class ids, randomization presets, counts, manifest checksum
  manifest.jsonl     one JSON line per frame
  scenarios.jsonl    one JSON line per scenario: seed key and scene overrides
  data.yaml          Ultralytics dataset description (relative paths; the folder can be moved)
  images/<split>/<scenario>_<frame>.jpg
  labels/<split>/<scenario>_<frame>.txt     YOLO: class cx cy w h, normalised to [0, 1]
  masks/<split>/<scenario>_<frame>.png      8-bit, pixel value = semantic class id
  previews/, statistics.json                written by scripts/inspect_dataset.py
```

One manifest line, abridged:

```json
{"frame_id": "train_0003_007", "split": "train", "scenario_id": "train_0003",
 "image": "images/train/train_0003_007.jpg", "label": "labels/train/train_0003_007.txt",
 "mask": "masks/train/train_0003_007.png", "image_sha1": "...", "width": 640, "height": 360,
 "randomization": "mild",
 "frame": {"mode": "ball_occluded", "base_pose_xy_yaw": [1.2, -0.4, 2.1],
           "rig": {"height": 0.93, "pitch": 0.41, "pan": -0.2, "roll": 0.01, "...": "..."},
           "ball_xy": [0.1, 0.9], "robots": [{"xy": [0.6, 0.3], "yaw": 1.0, "fallen": null}, null, null]},
 "T_field_left_optical": [["4x4 camera pose"]],
 "photometric": {"exposure_gain": 1.04, "gamma": 0.97, "noise_std": 1.8, "jpeg_quality": null},
 "objects": [{"instance": "ball", "class_id": 0, "labeled": true, "reason": "ok",
              "bbox_xyxy": [301.5, 170.0, 318.0, 186.5], "visible_fraction": 0.62,
              "truncated": false, "distance_m": 2.3}],
 "semantic_pixels": {"background": 91021, "field": 133112, "field_line": 6267}}
```

Read a dataset from Python with `datasets.io.SoccerDataset(root)`: `.frames(split)`,
`.image(record)`, `.mask(record)`, `.boxes(record)`. `validate_dataset(root)` checks every
frame.

## Configuration

`configs/datasets/baseline.yaml` lists every key with a comment. The presets:

| Preset | Frames | Size on disk | Use |
|---|---|---|---|
| `smoke` | 64 | 4 MB | tests, checking that things run |
| `baseline` | 2400 (1600 / 320 / 480) | 150 MB | the documented baseline |
| `large` | 13 000 | about 0.8 GB | when the baseline is data-limited; not generated by default |

Resolution is configurable (`resolution:` or `--resolution W H`); the field of view is kept.
Images are 640x360 by default, half the simulated ZED Mini's 1280x720: the detector is trained
at 640 px anyway, and disk and time drop by four.

Generation is deterministic on a given machine: the same configuration and seed give identical
files (`tests/test_dataset_generation.py::test_generation_is_deterministic`). Across machines
the GPU's rasterizer may differ in the last bit of some pixels, so compare `manifest_sha256`
only between datasets generated on the same machine.

Generation needs an OpenGL context (see the README). It takes about 0.15 s per frame at
640x360 on the development laptop.
