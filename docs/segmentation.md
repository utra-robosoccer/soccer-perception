# Field and field-line segmentation

A small network that labels every pixel as `background`, `field` or `field_line`. This document
covers the model, how thin lines and class imbalance are handled, how to train, evaluate, run
and export it, and what was measured.

All numbers here are **synthetic-on-synthetic**: trained and tested on renders from the same
simulator. See [vision_validation.md](vision_validation.md) for what that does and does not
establish.

## Why segmentation, and why these classes

Localization needs to know where the painted lines are on the ground. A box detector cannot
describe a line; a pixel mask can, and the mask's line pixels project onto the ground plane
(Phase 3). The `field` class is there for two reasons: it tells line paint from other white
things (a line is surrounded by turf), and later it gives the field boundary.

Class ids are defined in `datasets/schema.py`: 0 `background`, 1 `field`, 2 `field_line`.
Pixels showing the ball, a robot or a goal are `background`: the network is trained not to
call a robot's white leg a line, and not to draw a line through a robot that stands on one.

## Model

`segmentation/model.py::SmallUNet`: a U-Net with 16 base channels and four down-sampling steps.

| | |
|---|---|
| Parameters | 1.96 million (7.9 MB checkpoint) |
| Input | RGB, any size whose sides are multiples of 16 (the wrapper pads) |
| Output | one logit per class per pixel, at input resolution |
| Why a U-Net | lines are a few pixels wide; the decoder needs full-resolution features, which skip connections provide |
| Why this small | three classes with strong colour and shape cues; a baseline does not need more, and it must run next to the detector on a 4 GB GPU |

It is written in plain PyTorch with no dependency on Ultralytics, so its licence situation is
simple (PyTorch is BSD-3-Clause; the model code is this repository's).

**Replacing it.** The model is a `torch.nn.Module` mapping `(N, 3, H, W)` to
`(N, 3, H, W)` logits. To try another architecture, add it to `MODELS` in
`segmentation/model.py` and name it under `model.architecture` in the configuration; training,
evaluation and inference need no change. To use a model from another framework entirely, give
the pipeline any object with `predict(image_rgb) -> Segmentation`.

## Thin lines and class imbalance

Lines are about 3 % of the pixels; field is 58 % and background 40 %. A network that never
predicts a line is 97 % "accurate". Three things address this:

- **Class-weighted cross-entropy.** Weights are computed from the training split as
  `(largest class share / class share) ** 0.5`, capped at 10. For the baseline dataset that is
  `[1.2, 1.0, 4.5]` for background, field, line. The square root is a compromise: plain inverse
  frequency (about 20 for lines) makes the network paint fat lines.
- **A Dice term on the line class**, weight 0.5, which rewards overlap of predicted and labelled
  line regions directly.
- **Model selection on line IoU**, not on accuracy or loss: `monitor: line_iou` decides which
  epoch is saved as `best.pt` and when early stopping triggers.

And in evaluation, overall accuracy is printed with a warning next to it; the headline numbers
are the line scores below.

## Metrics

`segmentation/metrics.py` (plain NumPy; tested with hand-computed cases).

| Metric | Meaning |
|---|---|
| Per-class IoU, mean IoU | intersection over union of predicted and labelled pixels of a class |
| Line precision / recall / F1 (exact) | pixel-exact: a predicted line pixel counts only where the label says line |
| Line precision / recall / F1 **within 2 px** | a predicted line pixel counts if a labelled line pixel is within 2 px of it, and a labelled one counts as found if a predicted one is within 2 px |
| Line recall and precision **by distance** | line pixels grouped by how far along the ground they are from the camera |
| Per-frame line IoU | mean, median and 10th percentile over frames, to see the bad frames an average hides |

Why two flavours of line scores: a line 3 px wide that is predicted one pixel off has an exact
IoU of only 0.5, although for localization it is nearly perfect. The tolerant score separates
"the line is found but its edge is a pixel off" from "the line is missing or invented". Report
both.

Why by distance: lines get thinner with distance until they drop below one pixel, and a pixel
far away covers a lot of ground. The breakdown shows up to what range the mask is worth
projecting.

## Train

```bash
.venv/bin/python scripts/generate_dataset.py baseline            # if not done yet
.venv/bin/python scripts/train_segmentation.py unet_small        # configs/segmentation/unet_small.yaml
```

Useful overrides: `--dataset DIR`, `--epochs N`, `--batch N`, `--device cpu`, `--name RUN`,
`--learning-rate X`, `--seed N`.

`configs/segmentation/unet_small.yaml` documents every setting. Training uses AdamW with a
one-cycle learning-rate schedule, mixed precision on the GPU, batch 8, and random
scale / crop / left-right flip as augmentation. The crop is 512x352, so the network trains on
image regions and is evaluated on whole images.

**Pixel alignment.** Every geometric augmentation goes through `datasets/image_aug.py`, which
transforms the mask together with the image; masks are resized with centre-aligned nearest
neighbour so that class ids are never blended and never shift against the image.
`tests/test_segmentation.py::test_segmentation_dataset_keeps_image_and_mask_aligned` checks
this on real dataset frames (and caught a half-pixel shift from OpenCV's default
nearest-neighbour mode during development).

What a run leaves in `outputs/segmentation/<name>/`:

| File | Content |
|---|---|
| `best.pt`, `last.pt` | best epoch by the monitored metric, and the last epoch |
| `history.csv`, `history.json`, `history.png` | loss and validation metrics per epoch |
| `train_summary.json` | configuration, hardware, duration, peak GPU memory, dataset checksum |

A checkpoint is a plain dictionary (weights, model configuration, normalisation, class names,
training image width, validation metrics) and loads with `torch.load(..., weights_only=True)`.

### Smoke test

```bash
.venv/bin/python scripts/generate_dataset.py smoke
.venv/bin/python scripts/train_segmentation.py unet_smoke        # 3 epochs, about 10 s, CPU is fine
```

This proves the loop runs. Its model is not usable.

## Evaluate

```bash
.venv/bin/python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt
.venv/bin/python scripts/evaluate_segmentation.py outputs/segmentation/unet_small/best.pt --split val --tolerance 1
```

Writes `outputs/segmentation/<name>/eval_<dataset>_<split>/`:

| File | Content |
|---|---|
| `summary.md` | the tables below, for that run |
| `metrics.json` | everything, including latency, memory and hardware |
| `per_frame.json` | line IoU and mean IoU of every frame |
| `confusion_matrix.png`, `line_by_distance.png` | figures |
| `examples/typical/` | evenly spaced frames: image, prediction, errors |
| `examples/worst_line_iou/` | the frames with the lowest line IoU |

In the error panel, correctly predicted line pixels are white, false line pixels red, missed
line pixels yellow, and field / background disagreements blue.

The segmenter only ever receives images. Masks, and the camera pose used for the distance
breakdown, are read after inference.

## Results of the baseline run

Setup: `configs/segmentation/unet_small.yaml` on `outputs/datasets/baseline` (1600 training
images, `mild` randomization), 40 epochs, 35 minutes on an RTX 2050, 1.6 GB peak GPU memory
during training. The best epoch by validation line IoU was 33. Evaluated on the 480-image test
split, which shares no scene appearance with training.

Training progress (validation split):

| Epoch | Train loss | Val loss | Val mean IoU | Val line IoU | Val line F1 | Val line F1 within 2 px |
|---|---|---|---|---|---|---|
| 1 | 0.884 | 0.603 | 0.896 | 0.753 | 0.859 | 0.964 |
| 2 | 0.368 | 0.141 | 0.950 | 0.909 | 0.952 | 0.971 |
| 5 | 0.096 | 0.052 | 0.968 | 0.931 | 0.964 | 0.981 |
| 10 | 0.066 | 0.045 | 0.970 | 0.931 | 0.964 | 0.983 |
| 20 | 0.052 | 0.032 | 0.975 | 0.937 | 0.967 | 0.984 |
| 33 (best) | 0.040 | 0.028 | 0.980 | 0.948 | 0.974 | 0.986 |
| 40 | 0.039 | 0.028 | 0.979 | 0.948 | 0.973 | 0.986 |

Most of the result is there after five epochs; the remaining thirty-five buy 1.7 points of
line IoU.

Test split:

| Class | IoU | Precision | Recall | F1 | Share of pixels |
|---|---|---|---|---|---|
| background | 99.5 % | 99.7 % | 99.8 % | 99.7 % | 39.1 % |
| field | 99.5 % | 99.8 % | 99.7 % | 99.7 % | 58.0 % |
| field_line | **95.1 %** | 96.8 % | 98.2 % | 97.5 % | 2.9 % |
| **mean** | **98.0 %** | | | | |

| Field-line scores | Precision | Recall | F1 |
|---|---|---|---|
| Exact | 96.8 % | 98.2 % | 97.5 % |
| Within 2 px | 99.6 % | 99.9 % | 99.8 % |

Pixel accuracy is 99.7 %, and would be about 97 % for a model that found no line at all, which
is why it is not the headline. Per frame, over the 453 test frames that contain lines: mean
line IoU 0.947, median 0.956, 10th percentile 0.919.

The gap between the exact and the tolerant scores says what the remaining error is: nearly
every wrong line pixel is within two pixels of a real line. The network finds the lines and
disagrees with the labels about their edges, mostly by drawing them slightly too wide
(precision is lower than recall).

### By distance

| Distance along the ground | Labelled line pixels | Recall | Precision |
|---|---|---|---|
| under 2 m | 2 000 437 | 99.0 % | 98.2 % |
| 2-4 m | 847 145 | 97.6 % | 96.0 % |
| 4-6 m | 246 812 | 95.5 % | 93.2 % |
| 6-9 m | 87 906 | 95.9 % | 84.8 % |
| over 9 m | 5 788 | 87.1 % | 63.8 % |

Recall holds up with distance; **precision does not**. Beyond 6 m lines are one or two pixels
wide, so a prediction one pixel too fat is 15 to 35 % false pixels. Those false pixels lie next
to real lines in the image, but one pixel at that range is 13 cm or more on the ground, so
they are not harmless once projected. This is the main input to Phase 3's decision on how far
out to use line points.

### Speed and memory

Measured one image at a time, end to end (normalisation, network, softmax, argmax, copy to the
CPU), on the development laptop (RTX 2050, 4 GB) with nothing else on the GPU:

| | |
|---|---|
| Latency, 640x360 input, FP32 | 24.9 ms mean, 25.8 ms p95 (40 frames per second) |
| Latency, 1280x720 input (resized to 640 wide inside), FP32 | 32 ms mean |
| Latency, 1280x720 input, FP16 autocast (`--half`) | 28 ms mean |
| Latency, 1280x720 input, CPU | 133 ms mean |
| GPU memory, PyTorch tensors | 126 MB peak allocated, 174 MB reserved (plus the CUDA context) |
| Process RAM | 1.2 GB |

The segmenter is three times slower than the detector although it has fewer parameters: a
U-Net does most of its work at full image resolution. If the frame rate matters, the first
things to try are a stride-2 stem (work at half resolution, up-sample the logits) and FP16;
neither has been tried here.

### Outside the training distribution

The same checkpoint, trained on `mild`, on test sets rendered with the other presets (same
layouts, different appearance):

| Test set | Mean IoU | Line IoU | Line F1 | Line F1 within 2 px |
|---|---|---|---|---|
| `none` (default scene, no image effects) | 98.5 % | 96.1 % | 98.0 % | 99.8 % |
| `mild` (the training distribution) | 98.0 % | 95.1 % | 97.5 % | 99.8 % |
| `strong` (wider than training) | 82.1 % | 82.2 % | 90.2 % | 95.4 % |

On `strong`, line recall beyond 6 m falls to 72 % and precision to 62 %, and field/background
confusion appears (mean IoU drops by 16 points): very dark, over-exposed or oddly coloured turf
is no longer recognised as field. This is still a synthetic test, but it is the honest preview
of what an unseen real field will do to a model trained on a narrow appearance range. Training
on `strong`, or on a mix, is the obvious next experiment; it has not been run.

## Known failure cases and limits

- **Far lines are drawn too wide** (see the distance table). Exact precision past 6 m is 85 %
  and past 9 m 64 %.
- **Through a goal net.** Labels ignore the nets, the image does not: seen through one or two
  net panels the turf is dimmed and the network calls part of it `background` and misses lines
  behind the net. The worst test frames are of this kind (camera inside or behind a goal).
- **Unusual turf appearance** (the `strong` results above).
- **Line ends at occluders.** Lines stop where a robot or the ball covers them, by design.
  Phase 3 must not read a line ending at a robot as a line ending on the field.
- **No notion of which line.** The mask says "paint", not "touchline" or "centre circle";
  that is Phase 3's job.
- **Everything real.** Real lines are worn, uneven in width, sometimes taped; real turf has
  specular highlights and shadows of people; real images have lens distortion, motion blur and
  rolling shutter. White objects next to the field (boards, shoes, robot parts) are the classic
  false line. None of this has been measured.
- **One dataset, one seed, one run.** The spread across training runs is unknown.
- **Resolution.** Trained and evaluated at 640x360. Thin far lines would benefit from native
  1280x720 training at four times the compute.

## Run it

```bash
.venv/bin/python scripts/run_vision.py --render 8 --no-detector --ground
.venv/bin/python scripts/run_vision.py --images some_image.png --no-detector
```

```python
from robosoccer_perception.segmentation.segmenter import Segmenter

segmenter = Segmenter("outputs/segmentation/unet_small/best.pt")
result = segmenter.predict(image_rgb)      # (H, W, 3) uint8 RGB, any size
result.mask                                # (H, W) uint8 class ids
result.line_mask, result.field_mask        # (H, W) bool
result.line_confidence                     # (H, W) float32 in [0, 1]
```

**Image size.** The checkpoint records the image width it was trained at (640). Inputs of
another width, such as the camera's native 1280x720, are resized to that width before the
network and the class probabilities are resized back, because a convolutional network only
knows what lines look like at the scale it saw. The mask returned is at the input's resolution,
but its detail is that of a 640-px-wide image. `Segmenter(..., native_scale=True)` skips the
resizing; expect worse results unless the model was trained at that scale.

## Export

```bash
.venv/bin/python scripts/export_models.py --segmenter outputs/segmentation/unet_small/best.pt
```

Writes `best.onnx` (dynamic batch and image size) and `best.json`, which states the
pre-processing the graph expects: RGB scaled to [0, 1], then `(x - 0.5) / 0.25` per channel,
sides multiples of 16. Resizing to the training width, softmax and argmax are the caller's
job. The export is tested to load and run in ONNX Runtime; it has not been benchmarked. It
uses PyTorch's older TorchScript-based exporter, which prints a deprecation warning; the newer
exporter needs the `onnxscript` package, which is not a dependency of this project.

## Next steps

1. Label lines in a few dozen real ZED Mini frames and measure this checkpoint on them.
2. Train on `strong` (or `mild` + `strong`) and compare on the three synthetic test sets and on
   the real frames.
3. Decide, with Phase 3, the range beyond which line pixels are discarded.
4. If speed matters, try a half-resolution stem and FP16.
