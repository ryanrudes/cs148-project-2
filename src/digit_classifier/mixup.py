"""Thin wrapper around *timm*'s Mixup / CutMix implementation.

The wrapper is kept in its own module because it depends on the third-party
``timm`` library and has a distinct responsibility (batch-level augmentation
of both images *and* targets).
"""

from __future__ import annotations

from timm.data.mixup import Mixup
from torch import Tensor


def create_mixup_cutmix(
    num_classes: int,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    cutmix_minmax: tuple[float, float] | None = None,
    prob: float = 1.0,
    switch_prob: float = 0.5,
    mode: str = "batch",
    label_smoothing: float = 0.0,
) -> Mixup:
    """Create a *timm* Mixup / CutMix callable."""
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=cutmix_minmax,
        prob=prob,
        switch_prob=switch_prob,
        mode=mode,
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )


class MixupCutmixApply:
    """Convenience wrapper: passes through unchanged if ``mixup`` is ``None``."""

    def __init__(self, mixup: Mixup | None) -> None:
        self.mixup = mixup

    def __call__(self, images: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if self.mixup is None:
            return images, targets
        return self.mixup(images, targets)
