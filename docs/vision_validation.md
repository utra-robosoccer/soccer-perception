# Vision validation (Phase 2)

What was checked, what was measured, what went wrong along the way, and what has **not** been
validated. Phase 1's validation is in [validation.md](validation.md).

Read this first: **every number in this document is synthetic-on-synthetic.** The models were
trained on images rendered by this repository's simulator and tested on other images from the
same simulator. The results show that the pipeline is correct and that the models learn the
task as the simulator poses it. They are not evidence of how the models will do on a real
ZED Mini looking at a real field. No real image was used anywhere in Phase 2.

## Summary

| | Result | Where |
|---|---|---|
| Tests | 276 pass (172 from Phase 1, 104 new), 78 s | `pytest` |
| Phase 1 benchmark | all 170 runs reproduce the Phase 1 metrics exactly | below |
| Dataset | 2400 labelled frames, 640x360, generated in 5.8 min, 151 MB | `outputs/datasets/baseline/` |
| Detector (YOLO11n), test split | mAP50 0.982, mAP50-95 0.878, precision 98.0 %, recall 98.1 % | [object_detection.md](object_detection.md) |
| Segmenter (small U-Net), test split | mean IoU 98.0 %, line IoU 95.1 %, line F1 97.5 % (99.8 % within 2 px) | [segmentation.md](segmentation.md) |
| Both models, one 640x360 frame | 36 ms end to end (28 frames per second) | below |
| Both models, one 1280x720 frame | 43 ms end to end (23 frames per second) | below |
| Same models on a wider synthetic domain (`strong`) | mAP50 0.913, line IoU 82.2 % | below |

Machine: laptop with an NVIDIA GeForce RTX 2050 (4 GB), 12 CPU threads, 15 GB RAM, Ubuntu,
Python 3.12.3, MuJoCo 3.4.0, NumPy 2.5.5, PyTorch 2.15.0 (CUDA 13.2), Ultralytics 8.4.215,
OpenCV 5.0.0. Measured on 2026-11-16.

## Automated tests

```bash
.venv/bin/python -m pytest                    # 276 tests, about 1.5 minutes
.venv/bin/python -m pytest -m "not slow"      # 256 tests, about 30 s
```

| File | Tests | What it establishes |
|---|---|---|
| `test_scene_objects.py` | 17 | the default scene has no Phase 2 content; zero camera roll leaves poses unchanged; robots stand on the turf in any pose and lie on it when fallen; hidden objects are out of sight; ball looks compile; **rendering**: geom ids at every pixel, the ball's pixels where the camera model projects it, supersampled view consistent with the plain one, isolated renders restore the scene |
| `test_dataset_annotation.py` | 15 | coverage-based mask reduction; pixel-edge boxes; YOLO text round trip; **rendering**: ball box equals the projected ball (three distances), post box ends at the post's foot, **box clipped at the image border**, **fully occluded ball gets no label**, **partly occluded ball gets a box around the visible part only**, reasons for unlabelled objects; mask agrees with rendered colours (lines white, field green); **line-labelled pixels back-project onto paint**; pixels of robots and the ball standing on lines are `background` |
| `test_dataset_generation.py` | 23 | configuration loading and validation; one seed per scenario; presets load, sample deterministically, and `strong` varies more than `mild`; camera mounts stay inside physical limits and centred on the placeholder mount; every layout mode gives valid, deterministic, on-carpet layouts and puts things where it claims; **rendering**: the generated dataset validates frame by frame, the manifest records everything needed to reproduce it, **generation is deterministic** (identical bytes), **splits share no scenario, appearance, viewpoint or image**, a scenario does not depend on the rest of the dataset, existing directories are not overwritten, broken labels are caught, every preset renders |
| `test_image_aug.py` | 10 | photometric operations are deterministic, keep shape and do what they say; blur and JPEG do not move objects; **flip, crop, resize and pad transform masks and boxes with the image**, alone and chained |
| `test_detection.py` | 11 | IoU; greedy, one-to-one, class-aware matching; ignore regions; AP on hand-computed cases; perfect predictions score 1; confusion matrix and false-positive rate; stratified recall and false-positive categories on a constructed case; configurations load; a random-weight network loads and returns the output schema; the grey margin |
| `test_segmentation.py` | 11 | confusion matrix and IoU on hand-computed cases; ignored pixels; **accuracy stays high when every line is missed while line scores go to zero**; tolerant scores forgive a one-pixel shift and nothing more; model shapes and size; loss weights and ignored pixels; **image and mask stay aligned through augmentation on real frames**; checkpoint round trip and output schema at several image sizes; configurations load; a short training run lowers the loss and its checkpoint evaluates |
| `test_vision_pipeline.py` | 12 | output types and JSON schema; pipeline with either model missing; input checking; skeletonization; line-pixel sampling; ground projection inverts the camera model; latency statistics; **vision code does not import the localizer, the localizer does not import vision code or PyTorch, inference code cannot reach simulator state** |
| `test_vision_integration.py` | 5 | end to end on CPU in about 10 s: generate a dataset, train a detector for one epoch and a segmenter for two, load both, run the combined pipeline on dataset frames and on a 1280x720 camera frame, save predictions and overlays, run both evaluations, export to ONNX and run it |

Tests that render need an OpenGL context and are skipped without one; tests that need PyTorch
or Ultralytics are skipped when the `vision` extra is missing. **No test uses the network or
downloaded weights**: detector tests build YOLO11n from its architecture file with random
weights.

Full training is not part of the test suite. The slowest tests train for a few seconds on 12
frames.

## Phase 1 is unchanged

- The 172 Phase 1 tests pass unmodified. No Phase 1 test file was edited (`conftest.py` gained
  fixtures only).
- `scripts/benchmark.py` was re-run after all Phase 2 changes (17 scenario variants, seeds 0-9,
  170 runs) into a separate folder and compared with the stored Phase 1 output, run by run.
  **Every metric column of every run is identical**; only the wall-clock columns differ. The
  numbers in [validation.md](validation.md) stand as written. (The re-run's latencies are not
  quoted because a model was training on the same machine at the time.)
- Phase 1 source changes are limited to: an optional `roll` in `RigState` (skipped when zero),
  optional scene content in `mujoco_scene.py` (robots, backdrop, ball look, turf patches, light
  tint; all absent by default), goal nets moved to their own geom group (still drawn), and new
  methods (`render_segmentation`, `set_robots`). Field symmetry remains an open limitation and
  nothing in Phase 2 touches it.

## Labels

Label correctness was checked three ways:

1. **By construction.** Labels are read from the renderer's per-pixel geom ids, not computed
   from intended positions ([synthetic_data.md](synthetic_data.md#labels)).
2. **By tests** against independent geometry: the camera model (ball box versus projected ball),
   the field model (line pixels back-projected onto paint: over 99 % within a pixel's footprint,
   median error under 5 mm inside 4 m), and the rendered colours.
3. **By eye.** `scripts/inspect_dataset.py` previews were looked at for all three presets
   during development. That is how one generator fault was found and fixed: a far ball was
   occasionally placed off the carpet when the camera faced outward.

Baseline dataset: 1600 / 320 / 480 frames in 80 / 16 / 24 scenarios; 7529 boxes (1480 ball,
3003 goal_post, 3046 robot); 192 objects in view but unlabelled as too occluded and 13 as too
small; 6 % of the training frames have no box at all; pixel shares 39.6 % background, 57.5 % field, 2.8 %
line. Twelve frames sampled across the three splits were regenerated with the final code and
matched the stored images, labels and masks byte for byte.

## Model results

Details, breakdowns and failure cases are in [object_detection.md](object_detection.md) and
[segmentation.md](segmentation.md). The headline numbers, on the 480-frame test split:

**Detector** (YOLO11n, 40 epochs, 13 min of training), at confidence 0.25:

| Class | AP50 | AP50-95 | Precision | Recall |
|---|---|---|---|---|
| ball | 0.977 | 0.821 | 96.9 % | 96.6 % |
| goal_post | 0.980 | 0.897 | 98.7 % | 98.2 % |
| robot | 0.989 | 0.918 | 97.9 % | 98.8 % |

Ball recall by distance: 98.1 % under 2 m, 96.4 % at 2-5 m, 93.9 % beyond 5 m. By apparent
size: 87.5 % under 8 px, 95.7 % at 8-16 px, 98 % above. By occlusion: 100 % unoccluded, 80 %
at 50-95 % visible, 50 % below half visible. 0.065 false positives per image.

**Segmenter** (U-Net, 1.96 M parameters, 40 epochs, 35 min of training):

| Class | IoU | Precision | Recall |
|---|---|---|---|
| background | 99.5 % | 99.7 % | 99.8 % |
| field | 99.5 % | 99.8 % | 99.7 % |
| field_line | 95.1 % | 96.8 % | 98.2 % |

Line recall / precision by ground distance: 99.0 / 98.2 % under 2 m, 97.6 / 96.0 % at 2-4 m,
95.5 / 93.2 % at 4-6 m, 95.9 / 84.8 % at 6-9 m, 87.1 / 63.8 % beyond 9 m.

### Combined pipeline

`scripts/run_vision.py`, both models, one frame at a time, after warm-up:

| Input | Device | Detector | Segmenter | End to end | Frames |
|---|---|---|---|---|---|
| 640x360 dataset frames | GPU, FP32 | 8.7 ms | 26.9 ms | 35.6 ms mean, 37.2 ms p95 | 12 |
| 1280x720 rendered frames | GPU, FP32 | 11.4 ms | 31.9 ms | 43.2 ms mean, 45.9 ms p95 | 8 |
| 1280x720 rendered frames | GPU, FP16 (`--half`) | 11.4 ms | 27.7 ms | 39.2 ms mean | 8 |
| 1280x720 rendered frames | CPU | 26.6 ms | 133.1 ms | 159.7 ms mean | 4 |

GPU memory held by PyTorch for both models together: 136 MB peak allocated, 186 MB reserved,
plus the CUDA context. Process RAM: 1.6 to 1.8 GB. The models run one after the other, not in
parallel. These are few frames on a laptop that was otherwise idle; treat the numbers as
indicative, and note that the simulated camera is configured for 30 frames per second (a
ZED Mini can deliver 60 at HD720), more than this pipeline processes at that resolution.

As a geometric sanity check, predicted line pixels were thinned, sampled and projected onto
the ground with the true camera pose (`--ground` draws this). Over the 12 test frames above
(3184 sampled points): within 4 m of the camera more than 90 % of the points land on paint and
the 99th-percentile distance to paint is 9 cm; between 4 and 8 m the 90th percentile is 3 cm.
This says the masks are usable for Phase 3. It is a check on a dozen frames, not an
evaluation, and it uses the true camera pose, which the robot will not have.

### Outside the training distribution

Both models were trained on the `mild` preset only. The same checkpoints on 480-frame test sets
rendered with the other presets:

| Test set | Detector mAP50 | mAP50-95 | Recall | FP / image | Segmenter mean IoU | Line IoU |
|---|---|---|---|---|---|---|
| `none` | 0.979 | 0.896 | 98.0 % | 0.081 | 98.5 % | 96.1 % |
| `mild` (as trained) | 0.982 | 0.878 | 98.1 % | 0.065 | 98.0 % | 95.1 % |
| `strong` | 0.913 | 0.744 | 89.4 % | 0.279 | 82.1 % | 82.2 % |

The drop on `strong` is the most informative result of this phase. Inside the appearance range
they were trained on, both models are close to saturated; a step outside it costs 7 points of
mAP50 and 13 of line IoU, quadruples the false positives, and introduces class confusions and
field/background errors that do not occur in-distribution. A real field is further outside
that range than `strong` is. The in-distribution scores should therefore be read as an upper
bound, not as an expectation.

## Things found while validating

**The detector needs a grey margin around the image.** This project's evaluation and
Ultralytics' validator disagreed on goal posts (AP50 0.957 against 0.989). The metric code was
not the cause: fed the validator's predictions, it gave 0.984. The validator letterboxes
640x360 images to 384x672, leaving 16 px of grey at the sides; plain `predict()` leaves none,
and without it the network gives a full-height box to a post whose foot alone is in view.
`YoloDetector` now adds the margin: goal-post precision 92.7 % to 98.7 %, recall 93.6 % to
98.2 %. Any other runtime for this model must do the same.
[Details](object_detection.md#the-grey-margin-an-inference-detail-that-changes-results).

**Masks were half a pixel off after resizing.** OpenCV's default nearest-neighbour mode samples
at pixel corners while its image interpolation samples at centres. The alignment test on
augmented training pairs failed until masks were resized with the centre-aligned mode. Line
labels are two or three pixels wide, so this mattered.

**Average precision capped at 0.99.** The first AP implementation gave a perfect detector
100/101; a unit test with perfect predictions caught it.

**False positives were being misattributed.** The first analysis looked only at the semantic
class under a false box, which called every loose box on a real ball "background" (objects are
background in the mask). False positives are now first sorted into badly placed, other object
and nothing there.

**Running out of memory.** Training the detector and the segmenter at the same time, plus a
CPU inference job, exhausted the laptop's 15 GB of RAM and killed the session. Trainings were
re-run one at a time with two dataloader workers; that is now the configured default.

## Acceptance criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Phase 1 localization tests still pass | met | 172 tests; benchmark identical run by run |
| 2 | Images generated reproducibly | met | `test_generation_is_deterministic`; 12 baseline frames regenerated identically |
| 3 | Ball, goal-post and robot labels | met | `datasets/annotation.py`; 7529 boxes in the baseline set |
| 4 | Field / line masks | met | `masks/`; mask tests |
| 5 | Annotations align with rendered objects | met | projection, back-projection and colour tests; previews |
| 6 | Occlusion and clipping handled | met | clipping and occlusion tests; `visible_fraction`, `truncated`, `reason` per object |
| 7 | Randomization presets work | met | `none`, `mild`, `strong`; preset tests; three test sets generated |
| 8 | Splits avoid near-duplicate leakage | met | scenario-level splits; `test_splits_share_no_scenario_appearance_or_viewpoint` |
| 9 | Detector training pipeline works | met | 40-epoch run; smoke config; integration test |
| 10 | YOLO baseline runs inference | met | `YoloDetector`; `run_vision.py` |
| 11 | Detector evaluation gives useful metrics | met | mAP, per-class AP, confusion matrix, stratified recall, false-positive causes, latency, memory, examples |
| 12 | Segmentation model runs inference | met | `Segmenter`; `run_vision.py` |
| 13 | Segmentation evaluation has line-specific metrics | met | line IoU / precision / recall / F1, tolerant scores, by distance |
| 14 | Combined inference on simulated camera images | met | `outputs/vision/rendered_1280x720/` |
| 15 | Annotated visualizations | met | dataset previews, evaluation examples, pipeline overlays, ground plots |
| 16 | New unit tests pass | met | 104 new tests |
| 17 | Configuration and execution documented | met | README and the five Phase 2 documents |
| 18 | Other repositories untouched | met | nothing was written to `robosoccer-rl` or `soccer-software`; they were only listed and searched |
| 19 | No Phase 3 localization integration | met | import-boundary tests; no `ObservationFrame` is built from vision output |

## What has not been validated

- **Real images.** None used. No sim-to-real claim is made.
- **Real camera geometry.** Images are ideal pinhole renders: no lens distortion, no rolling
  shutter, no real exposure behaviour. The camera profile is approximate and the mount is a
  placeholder.
- **Real robots.** The `robot` class has only ever seen one placeholder body. Its 98.8 % recall
  says nothing about a real humanoid.
- **Real goals, balls and backgrounds.** Plain white tubes, three procedural ball patterns,
  walls and boxes. No spectators, referees, banners, cables or second field.
- **Motion.** Every frame is a still. No walking-induced blur or shake beyond a mild synthetic
  blur, no temporal consistency, no tracking.
- **Statistical spread.** One dataset seed, one training seed, one run per model. Differences
  of a point or less between configurations should not be read as real.
- **Hyper-parameters.** Defaults were used; nothing was tuned, and no comparison of model
  sizes, image sizes or YOLO11n against YOLO26n was run.
- **Training on `strong`.** Presets were compared only at test time.
- **Larger fields.** Datasets were generated for the S field only. The generator takes
  `--field m_field` / `l_field`, and scenes for those compile, but no model was trained or
  tested on them.
- **Deployment.** ONNX export is tested to load and run, not benchmarked. Nothing was run on a
  Jetson or on the robot's computer. No ROS 2 node exists.
- **The right camera.** Only left images are used.
- **Use for localization.** No pose was ever estimated from an image. See
  [phase3_interfaces.md](phase3_interfaces.md).

## Reproduce

Everything in this document, from an empty `outputs/` (about one hour, mostly the two trainings):

```bash
.venv/bin/python -m pip install -e ".[all,vision]"
.venv/bin/python scripts/train_detector.py --download yolo11n.pt

.venv/bin/python -m pytest

.venv/bin/python scripts/generate_dataset.py baseline
.venv/bin/python scripts/inspect_dataset.py outputs/datasets/baseline
.venv/bin/python scripts/generate_dataset.py baseline --splits test --randomization strong --name baseline_strong_test
.venv/bin/python scripts/generate_dataset.py baseline --splits test --randomization none --name baseline_none_test

.venv/bin/python scripts/train_detector.py yolo11n
.venv/bin/python scripts/train_segmentation.py unet_small        # after the detector has finished

.venv/bin/python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt
.venv/bin/python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt
.venv/bin/python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt --dataset outputs/datasets/baseline_strong_test --no-ultralytics-val
.venv/bin/python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt --dataset outputs/datasets/baseline_strong_test
.venv/bin/python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt --dataset outputs/datasets/baseline_none_test --no-ultralytics-val
.venv/bin/python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt --dataset outputs/datasets/baseline_none_test

.venv/bin/python scripts/run_vision.py --render 8 --ground --output outputs/vision/rendered_1280x720
.venv/bin/python scripts/run_vision.py --dataset outputs/datasets/baseline --count 12 --ground --output outputs/vision/dataset_test
```

The dataset is bit-reproducible on one machine. Trained weights are not: GPU training is
repeatable only approximately, so expect the last digit of the scores to differ.

Where the results of this run are (not tracked by Git):

| What | Path |
|---|---|
| Dataset, label previews | `outputs/datasets/baseline/`, `outputs/datasets/baseline/previews/` |
| Detector run, training curves | `outputs/detection/yolo11n/` (`weights/best.pt`, `results.csv`, `results.png`) |
| Detector evaluation | `outputs/detection/yolo11n/eval_baseline_test/` (`summary.md`, `metrics.json`, `confusion_matrix.png`, `recall_*.png`, `examples/`) |
| Segmenter run, training curves | `outputs/segmentation/unet_small/` (`best.pt`, `history.csv`, `history.png`) |
| Segmenter evaluation | `outputs/segmentation/unet_small/eval_baseline_test/` (`summary.md`, `metrics.json`, `line_by_distance.png`, `examples/`) |
| Cross-domain evaluations | `.../eval_baseline_strong_test_test/`, `.../eval_baseline_none_test_test/` under both run folders |
| Combined pipeline | `outputs/vision/rendered_1280x720/`, `outputs/vision/dataset_test/` (`*_overlay.jpg`, `*.json`, `*_mask.png`, `*_ground.png`, `summary.json`) |
| Phase 1 benchmark re-run | `outputs/benchmark_phase2_check/` |
