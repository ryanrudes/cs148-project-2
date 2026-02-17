"""Core dataset class for digit images stored as in-memory tensors."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


class DigitDataset(Dataset):
    """PyTorch dataset wrapping pre-loaded image tensors and integer labels.

    Parameters
    ----------
    images : Tensor
        Float tensor of shape ``(N, C, H, W)`` in [0, 1].
    labels : Tensor
        Long tensor of shape ``(N,)`` with digit labels 0–9.
    transform : callable, optional
        A callable with signature ``transform(image, label) -> image``.
        Both training and validation transforms receive the label; the
        validation wrapper simply ignores it
        (see :class:`~digit_classifier.augmentation.ApplyTransform`).
    """

    def __init__(
        self,
        images: Tensor,
        labels: Tensor,
        transform: Callable[..., Tensor] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image, label)

        return image, label
