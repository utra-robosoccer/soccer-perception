"""Loss for imbalanced segmentation: weighted cross-entropy plus a Dice term on the lines."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from robosoccer_perception.datasets import schema


class SegmentationLoss(nn.Module):
    """``cross_entropy(class_weights) + dice_weight * dice(dice_class)``.

    Cross-entropy with class weights keeps the rare line class from being ignored; the Dice
    term rewards overlap of the predicted and labelled line regions directly, which is what
    line IoU measures. Pixels labelled `schema.IGNORE_INDEX` contribute to neither.
    """

    def __init__(self, class_weights=None, dice_weight: float = 0.5,
                 dice_class: int = schema.FIELD_LINE, ignore_index: int = schema.IGNORE_INDEX):
        super().__init__()
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)
        self.dice_weight, self.dice_class, self.ignore_index = dice_weight, dice_class, ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()                      # the loss is computed in full precision
        loss = F.cross_entropy(logits, target, weight=self.class_weights,
                               ignore_index=self.ignore_index)
        if self.dice_weight > 0:
            valid = (target != self.ignore_index).float()
            probability = torch.softmax(logits, dim=1)[:, self.dice_class] * valid
            truth = (target == self.dice_class).float()
            intersection = (probability * truth).sum()
            dice = (2 * intersection + 1.0) / (probability.sum() + truth.sum() + 1.0)
            loss = loss + self.dice_weight * (1.0 - dice)
        return loss
