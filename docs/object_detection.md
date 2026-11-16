# Object detection

A lightweight detector for three classes: `ball`, `goal_post`, `robot`. This document covers the
model choice and its licence, how to train, evaluate, run and export it, what was measured, and
one inference detail that matters more than it looks (the grey margin).

All numbers here are **synthetic-on-synthetic**: trained and tested on renders from the same
simulator. They show that the pipeline works. They are not a prediction of real-camera
performance. See [vision_validation.md](vision_validation.md).

## Model

**YOLO11n**, through the Ultralytics package, fine-tuned from COCO-pretrained weights.

| | |
|---|---|
| Parameters | 2.6 million (5.3 MB checkpoint) |
| Input | 640 px on the long side; images are letterboxed |
| Why this one | smallest member of a well-supported family; trains in minutes on a 4 GB GPU; the task brief names it as the preferred baseline |
| Alternative | `yolo26n.pt` is supported by the installed Ultralytics and by this code (`--model yolo26n.pt`); it has not been trained or compared here |

Everything that touches Ultralytics is in `detection/yolo.py`. The rest of the code sees only
`inference.types.Detection`, so another detector can be dropped in by writing a class with a
`predict(image_rgb) -> list[Detection]` method.

### Licensing

Read this before distributing a model or building it into something.

- The **`ultralytics` package is AGPL-3.0**. So are the **pretrained YOLO11 / YOLO26 weights**,
  and Ultralytics' position is that models trained with the package or fine-tuned from those
  weights are covered too. AGPL-3.0 requires making the complete corresponding source available
  to anyone who uses the software, including over a network. Ultralytics sells an Enterprise
  licence that removes this requirement.
- For an open university robotics team publishing its code this is usually acceptable, but it
  is a decision for the team, not something this repository can settle. If AGPL is not
  acceptable, replace the detector (for example with a model under Apache-2.0 or MIT); the
  dataset, the evaluation code and the pipeline interface do not depend on Ultralytics.
- This repository does not redistribute Ultralytics code or weights. Checkpoints are downloaded
  by you into `weights/`, which Git ignores, as are all trained models (`*.pt`, `*.onnx`).
- PyTorch is BSD-3-Clause, OpenCV is Apache-2.0, ONNX Runtime is MIT.
- COCO, which the pretrained weights were trained on, has its own terms for the images; no COCO
  data is used or stored here.

### Network access and telemetry

Ultralytics can download weights, fonts and missing packages by itself and can report anonymous
usage statistics. `detection/yolo.py` sets `YOLO_OFFLINE=1` before importing it (unless you set
the variable yourself), which turns all of that off. The one deliberate download is:

```bash
.venv/bin/python scripts/train_detector.py --download yolo11n.pt     # 5.4 MB, into weights/
```

Note that importing Ultralytics creates or updates its settings file in
`~/.config/Ultralytics/`. Runs from this repository do not use the directories named there: the
output and weights paths are always passed explicitly.

## Train

```bash
.venv/bin/python scripts/generate_dataset.py baseline            # if not done yet
.venv/bin/python scripts/train_detector.py yolo11n               # configs/detection/yolo11n.yaml
```

Useful overrides: `--dataset DIR`, `--epochs N`, `--batch N`, `--imgsz N`, `--device cpu`,
`--model yolo26n.pt`, `--name RUN`, `--resume`.

`configs/detection/yolo11n.yaml` documents every setting. The defaults are sized for a 4 GB
laptop GPU: batch 16 at 640 px uses about 2.2 GB of GPU memory with mixed precision; two
dataloader workers keep RAM use moderate. If the GPU runs out of memory, halve `batch`. CPU
training works (`--device cpu`) but is slow; use it for the smoke preset only.

What a run leaves in `outputs/detection/<name>/`:

| File | Content |
|---|---|
| `weights/best.pt`, `weights/last.pt` | best validation checkpoint, and the last one (for `--resume`) |
| `results.csv` | losses and validation metrics per epoch |
| `args.yaml` | every training argument Ultralytics used |
| `train_summary.json` | configuration, hardware, duration, final metrics |
| `*.png`, `*.jpg` | Ultralytics' curves and sample batches |

Reproducibility: the seed is fixed and `deterministic: true` is set, which makes runs
repeatable on one machine as far as cuDNN allows. Expect the third decimal of mAP to move
between machines and library versions.

Training-time augmentation is Ultralytics' own (mosaic, scale, shift, HSV, left-right flip),
which transforms the boxes with the image. Up-down flips and rotation are off: the camera's
roll is already randomized in the renders.

Early stopping (`patience`) and checkpointing are handled by Ultralytics.

### Smoke test

```bash
.venv/bin/python scripts/generate_dataset.py smoke
.venv/bin/python scripts/train_detector.py yolo11n_smoke         # 2 epochs, about 10 s
```

This proves the loop runs. Its model detects nothing useful.

## Evaluate

```bash
.venv/bin/python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt
.venv/bin/python scripts/evaluate_detector.py outputs/detection/yolo11n/weights/best.pt --split val --conf 0.4
```

Writes `outputs/detection/<name>/eval_<dataset>_<split>/`:

| File | Content |
|---|---|
| `summary.md` | the tables below, for that run |
| `metrics.json` | everything, including latency, memory and hardware |
| `stratified_recall.csv` | recall per class and difficulty bin |
| `predictions.jsonl` | the detections at the operating threshold, per frame |
| `confusion_matrix.png`, `recall_<class>.png` | figures |
| `examples/correct`, `examples/false_positive`, `examples/false_negative` | annotated images: solid class-coloured boxes are correct, red dashed are false positives, white dashed are missed objects |
| `ultralytics_val/` | Ultralytics' own validation run, as a cross-check |

How the numbers are defined (`detection/metrics.py`, plain NumPy, tested without any model):

- A detection is correct when it has the class of a labelled object and IoU >= 0.5 with it.
  Matching is greedy by confidence, one detection per object.
- **AP** is computed over all confidences (COCO-style, 101 recall points). mAP50-95 averages
  over IoU 0.50 to 0.95.
- **Precision, recall, false positives and the confusion matrix** are taken at the operating
  confidence (`--conf`, default 0.25): they describe the detector as it would run.
- **Ignore regions.** Objects that are in the image but unlabelled (more than 85 % hidden, or
  tiny) are neither required nor punished: detecting one is not a false positive. Ultralytics'
  validator does not know about them and counts such detections as errors.
- **Recall by difficulty** uses the manifest's ground truth per object: distance, apparent size,
  visible fraction, clipping by the image border, robot posture and height, frame layout.
- **False positives** are split into *badly placed* (overlaps an object of its own class: a
  loose or duplicate box), *other object* (overlaps another class: a confusion) and *nothing
  there*; for the last group the semantic class under the box centre tells whether the backdrop,
  the turf or a line was mistaken for an object.

The detector only ever receives images. Labels and manifest are read after inference.

## Results of the baseline run

Setup: `configs/detection/yolo11n.yaml` on `outputs/datasets/baseline` (1600 training images,
`mild` randomization), 40 epochs, 13 minutes on an RTX 2050. Evaluated on the 480-image test
split, which shares no scene appearance with training. Operating point: confidence 0.25.

Training progress (validation split):

| Epoch | Train box loss | Train class loss | Val mAP50 | Val mAP50-95 |
|---|---|---|---|---|
| 1 | 1.012 | 1.635 | 0.919 | 0.673 |
| 5 | 0.841 | 0.625 | 0.923 | 0.695 |
| 10 | 0.737 | 0.530 | 0.945 | 0.784 |
| 20 | 0.623 | 0.440 | 0.963 | 0.830 |
| 30 | 0.549 | 0.385 | 0.965 | 0.855 |
| 40 | 0.414 | 0.293 | 0.969 | 0.864 |

Test split:

| Class | Objects | AP50 | AP50-95 | Precision | Recall | False positives per image |
|---|---|---|---|---|---|---|
| ball | 320 | 0.977 | 0.821 | 96.9 % | 96.6 % | 0.021 |
| goal_post | 598 | 0.980 | 0.897 | 98.7 % | 98.2 % | 0.017 |
| robot | 601 | 0.989 | 0.918 | 97.9 % | 98.8 % | 0.027 |
| **all** | 1519 | **0.982** | **0.878** | 98.0 % | 98.1 % | 0.065 |

5.4 % of test images contain at least one false positive. Cross-check: Ultralytics' own
validator reports mAP50 0.988 and mAP50-95 0.882 on the same split; scoring its predictions
with this project's metric code gives 0.983 and 0.879. The two implementations agree to within
their differences in curve interpolation and ignore regions.

### Ball

| Breakdown | Bin | Detected / total | Recall |
|---|---|---|---|
| Distance | near (< 2 m) | 101 / 103 | 98.1 % |
| | medium (2-5 m) | 162 / 168 | 96.4 % |
| | far (> 5 m) | 46 / 49 | 93.9 % |
| Apparent size | < 8 px | 14 / 16 | 87.5 % |
| | 8-16 px | 133 / 139 | 95.7 % |
| | 16-32 px | 106 / 108 | 98.1 % |
| | 32-96 px | 56 / 57 | 98.2 % |
| Occlusion | less than half visible | 7 / 14 | 50.0 % |
| | 50-95 % visible | 16 / 20 | 80.0 % |
| | unoccluded | 286 / 286 | 100 % |
| Image border | cut by the border | 26 / 26 | 100 % |

Recall falls with distance and size, as expected, but gently: at 640x360 a ball 7 m away is
still about 6 px across. **Occlusion is what loses balls**: every missed ball in the test split
was partly hidden. A ball half behind a robot's leg is detected about half the time.

### Goal posts

| Breakdown | Bin | Detected / total | Recall |
|---|---|---|---|
| Distance | near (< 3 m) | 163 / 163 | 100 % |
| | medium (3-6 m) | 226 / 231 | 97.8 % |
| | far (> 6 m) | 198 / 204 | 97.1 % |
| Apparent size | < 8 px | 2 / 11 | 18.2 % |
| | 8-16 px and larger | 585 / 587 | 99.7 % |
| Occlusion | less than half visible | 1 / 3 | 33.3 % |
| | 50-95 % visible | 28 / 29 | 96.6 % |
| | unoccluded | 558 / 566 | 98.6 % |
| Image border | cut by the border | 360 / 370 | 97.3 % |
| | fully inside | 227 / 228 | 99.6 % |

Almost two thirds of the labelled posts are cut by the image border (a post is taller than the
view at most distances), so the border case is the normal case. Slivers of a post under 8 px are
mostly missed. Background confusion is rare: of 8 false positives, 5 are loose boxes on real
posts, 1 overlaps a robot, and 2 sit where nothing is (one on the backdrop, one on turf). Note
that the `mild` backdrop is plain walls and boxes; real halls have white structures that look
far more like posts.

### Robots

| Breakdown | Bin | Detected / total | Recall |
|---|---|---|---|
| Distance | near (< 2 m) | 201 / 202 | 99.5 % |
| | medium (2-5 m) | 283 / 287 | 98.6 % |
| | far (> 5 m) | 110 / 112 | 98.2 % |
| Posture | upright | 543 / 546 | 99.5 % |
| | fallen | 51 / 55 | 92.7 % |
| Height | small (< 0.7 m) | 221 / 221 | 100 % |
| | medium (0.7-1.0 m) | 373 / 380 | 98.2 % |
| Occlusion | less than half visible | 2 / 5 | 40.0 % |
| | 50-95 % visible | 28 / 30 | 93.3 % |
| | unoccluded | 564 / 566 | 99.6 % |
| Image border | cut by the border | 107 / 109 | 98.2 % |

These are the easiest objects in the dataset, and that is a warning rather than good news: all
of them are the same placeholder body in different sizes and greys, so the detector has learnt
that body. 12 of 13 robot false positives are loose or duplicate boxes on real robots, typically
when two robots overlap.

### Speed and memory

Measured one image at a time, end to end (margin, letterbox, network, NMS), FP32, on the
development laptop (RTX 2050, 4 GB) with nothing else on the GPU:

| | |
|---|---|
| Latency, 640x360 input | 7.5 ms mean, 8.3 ms p95 (about 130 frames per second); 8.1 ms in a second run |
| Latency, 1280x720 input | 11.4 ms mean (the extra time is resizing the larger image) |
| Latency, 1280x720 input, CPU | 27 ms mean |
| GPU memory, PyTorch tensors | 49 MB peak allocated, 66 MB reserved (plus the CUDA context, a few hundred MB, shared with anything else using PyTorch) |
| Process RAM | 1.6 GB, mostly the PyTorch and CUDA libraries |

The 1280x720 and CPU figures come from `scripts/run_vision.py` on 8 and 4 frames
respectively, so treat them as indicative. No timing has been taken on a Jetson or any other
robot computer.

### Outside the training distribution

The same checkpoint, trained on `mild`, on test sets rendered with the other presets (same
layouts, different appearance; 480 frames each):

| Test set | mAP50 | mAP50-95 | Precision | Recall | False positives per image |
|---|---|---|---|---|---|
| `none` (default scene, no image effects) | 0.979 | 0.896 | 97.4 % | 98.0 % | 0.081 |
| `mild` (the training distribution) | 0.982 | 0.878 | 98.0 % | 98.1 % | 0.065 |
| `strong` (wider than training) | 0.913 | 0.744 | 90.7 % | 89.4 % | 0.279 |

On `strong`, per class: ball AP50 0.883 (recall 83.7 %), goal_post 0.913 (88.3 %), robot 0.943
(93.3 %). What breaks:

- **Unoccluded balls are missed** (246 of 280 found, against 286 of 286 on `mild`). `strong`
  balls are up to 11 cm in radius and may be coloured or plain; `mild` balls are 7 cm and
  mostly black and white.
- **Tall robots**: 259 of 290 robots of 1.0 m or more are found. The training set has none
  taller than 0.95 m.
- **Class confusions appear**: 27 robot and 23 goal-post false positives overlap an object of
  the other class (0 and 1 on `mild`), and 20 goal-post false positives sit where nothing is,
  16 of them on the backdrop, which in `strong` includes white walls and boxes.

This is still synthetic data, but it is a preview of the kind of drop to expect on a real
field, whose appearance is certainly outside the `mild` range. Training on `strong`, or on a
mix, is the obvious next experiment; it has not been run.

## The grey margin: an inference detail that changes results

`YoloDetector` surrounds each image with 16 px of grey before the network sees it, and shifts
the boxes back afterwards. This is not cosmetic.

During training (mosaic, random shifts) and in Ultralytics' validator (batch shapes rounded up
to the stride), an image never fills the network input edge to edge; there is grey around it.
The network learns that an object cut off at a visible image border ends there. Ultralytics'
plain `predict()` on a 640x360 image adds grey above and below but none at the sides. Given
such an input, the network guesses the hidden part of objects at the left and right borders: a
goal post of which only the foot is in view gets a box the full height of a post.

Measured on the test split, same weights, with and without the margin:

| | No margin | 16 px margin |
|---|---|---|
| goal_post precision | 92.7 % | 98.7 % |
| goal_post recall | 93.6 % | 98.2 % |
| goal_post recall, cut by the border | 90.0 % (333 / 370) | 97.3 % (360 / 370) |
| goal_post AP50-95 | 0.836 | 0.897 |
| mAP50 / mAP50-95 | 0.974 / 0.857 | 0.982 / 0.878 |
| False positives per image | 0.135 | 0.065 |

Ball and robot results barely change. Margins of 8, 16 and 32 px give the same result to within
one detection; 16 was chosen because it reproduces the validator's framing, not because it
scored best. The effect was found because this project's evaluation and Ultralytics' validator
disagreed about goal posts; the cause was the framing, not the metric.

Two consequences:

- **Anything that runs this detector outside `YoloDetector`** (an exported ONNX graph, a ROS
  node, a C++ runtime) must add the same margin, or post boxes at the image border will be
  wrong. This matters for localization: the bottom of a post box is where Phase 3 will read
  the post's position on the ground.
- It is a reminder that validation numbers describe the validator's pre-processing. Evaluate
  through the code path that will be deployed.

## Run it

```bash
.venv/bin/python scripts/run_vision.py --render 8 --no-segmenter      # simulated camera frames
.venv/bin/python scripts/run_vision.py --images some_image.png --no-segmenter
```

```python
from robosoccer_perception.detection.yolo import YoloDetector

detector = YoloDetector("outputs/detection/yolo11n/weights/best.pt", conf=0.25)
for d in detector.predict(image_rgb):            # (H, W, 3) uint8 RGB, any size
    print(d.class_name, d.confidence, d.bbox_xyxy)
```

Boxes are in the input image's pixels, pixel-edge convention (see
[synthetic_data.md](synthetic_data.md#coordinate-convention-of-boxes)).

## Export

```bash
.venv/bin/python scripts/export_models.py --detector outputs/detection/yolo11n/weights/best.pt
.venv/bin/python scripts/export_models.py --detector outputs/detection/yolo11n/weights/best.pt --format torchscript
```

ONNX and TorchScript need no extra packages. TensorRT, OpenVINO and others are supported by
Ultralytics but need their own toolchains, which are not installed here. The exported graph
contains the network only: letterboxing, **the grey margin**, and (depending on the format) NMS
are the caller's job. Exported models have not been benchmarked.

## Known failure cases and limits

- **Partly hidden balls.** Recall is 50 % when less than half the ball shows. The ball is small
  and usually near robots' feet, so this will be common in play.
- **Tiny slivers of posts** (under 8 px) are missed.
- **Fallen robots** are found less often than upright ones (93 % against 99.5 %).
- **Overlapping robots** produce loose or duplicate boxes.
- **Objects at the image border without the grey margin**: see above.
- **Everything real.** No real image has been through this model. The robots are placeholders;
  the ball is one of three procedural patterns; goals are plain white tubes; the backdrop is
  walls and boxes. Real fields add spectators, referees, advertising, cables, reflections,
  motion blur from walking, lens distortion and rolling shutter. Expect a large drop until the
  model has been fine-tuned on labelled ZED Mini footage.
- **One dataset, one seed, one run.** No repeat runs were made, so the spread of these numbers
  is unknown. Bins with a handful of objects (heavily occluded posts: 3) carry almost no
  information.
- **Image size.** Training and testing used 640x360. `YoloDetector` accepts 1280x720 frames and
  scales them to the same network size, so the result is the same; nothing has been trained at
  higher resolution, which would help distant balls.

## Next steps

1. Record and label a few hundred real ZED Mini frames; evaluate this model on them unchanged,
   then fine-tune. That number is the first one that means something for the robot.
2. Replace the placeholder robot with real robot models, or mix in real robot images.
3. Compare `mild` against `strong` randomization on the real frames, not on synthetic ones.
4. If far balls matter, train at a larger image size or add a higher-resolution detection head.
