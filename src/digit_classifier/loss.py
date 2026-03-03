"""Loss functions for training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BCELossWithSmoothing(nn.Module):
    """Binary cross-entropy for multi-class (DeiT-III style).

    Uses BCEWithLogitsLoss with label smoothing. Accepts either hard labels
    (class indices) or soft targets (from mixup). Per DeiT-III: BCE is used
    for ImageNet-1k training and works well with Mixup/CutMix.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self._bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if labels.dim() == 1:
            target = torch.full(
                (labels.size(0), self.num_classes),
                self.smoothing / max(1, self.num_classes - 1),
                device=logits.device,
                dtype=logits.dtype,
            )
            target.scatter_(1, labels.unsqueeze(1).long(), 1.0 - self.smoothing)
        else:
            target = labels.float().to(logits.device)
        return self._bce(logits, target)
