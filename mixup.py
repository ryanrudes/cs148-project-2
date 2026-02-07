# NOTE:
# - Mixup/CutMix is used on batches after DataLoader returns a batch, not per-sample.
# - Typical usage is inside the training loop:
#     images, targets = mixup_fn(images, targets)
# - This works best when targets are integer class indices (shape [B]).

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
) -> Mixup | None:
    """Create a timm Mixup/CutMix callable.

    Returns:
        timm.data.mixup.Mixup instance, or None if timm isn't available.
    """
    if Mixup is None:
        return None

    return Mixup(
        mixup_alpha=mixup_alpha,        # set 0.0 to disable mixup
        cutmix_alpha=cutmix_alpha,      # set 0.0 to disable cutmix
        cutmix_minmax=cutmix_minmax,    # overrides cutmix_alpha if not None
        prob=prob,
        switch_prob=switch_prob,
        mode=mode,                      # "batch" is the common default
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )


class MixupCutmixApply:
    """Convenience wrapper: does nothing if mixup is None."""

    def __init__(self, mixup: Mixup | None):
        self.mixup = mixup

    def __call__(self, images: Tensor, targets: Tensor):
        if self.mixup is None:
            return images, targets
        return self.mixup(images, targets)
