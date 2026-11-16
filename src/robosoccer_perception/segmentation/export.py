"""Export of a trained segmentation checkpoint to ONNX.

The exported graph is the bare network: ``image`` ``(N, 3, H, W)`` float32, already normalised
as the checkpoint's ``normalization`` says, to ``logits`` ``(N, classes, H, W)``. Resizing to
the training width, normalisation, softmax and argmax stay outside (see
`segmentation.segmenter.Segmenter` for the reference implementation).

PyTorch's TorchScript-based exporter is used (``dynamo=False``). It is deprecated and prints a
warning, but the newer exporter needs the ``onnxscript`` package, which this project does not
depend on.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from robosoccer_perception.segmentation.segmenter import load_checkpoint


def export_segmenter_onnx(checkpoint, output=None, height: int = 368, width: int = 640,
                          opset: int = 17) -> Path:
    """Write ``<output>`` (default: next to the checkpoint) and a ``.json`` sidecar with the
    pre-processing the graph expects. ``height`` and ``width`` must be multiples of the
    network's stride; the batch and spatial axes are exported as dynamic."""
    model, metadata = load_checkpoint(checkpoint, "cpu")
    stride = getattr(model, "stride", 16)
    if height % stride or width % stride:
        raise ValueError(f"height and width must be multiples of {stride}")
    output = Path(output) if output else Path(checkpoint).with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (torch.zeros(1, 3, height, width),), str(output), input_names=["image"],
        output_names=["logits"], opset_version=opset, dynamo=False,
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}})
    output.with_suffix(".json").write_text(json.dumps({
        "input": "image: float32 (N, 3, H, W), RGB / 255, then (x - mean) / std; H and W "
                 f"multiples of {stride}",
        "output": "logits: float32 (N, classes, H, W); argmax over axis 1 gives class ids",
        "normalization": metadata["normalization"], "classes": metadata["classes"],
        "train_width": metadata["train_width"], "stride": stride}, indent=2))
    return output
