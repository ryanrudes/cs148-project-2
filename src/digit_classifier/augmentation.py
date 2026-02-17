"""YOLO-style augmentation pipeline with digit-safe hyper-parameters.

Images entering this pipeline are tensors of shape ``(C, H, W)`` in [0, 1]
that have already been resized and colour-converted by the deterministic
preprocessor (see :func:`get_preprocessor`).

The horizontal flip is **label-conditional**: only digits 0 and 8 are flipped,
since all other digits change identity under reflection.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor
from torchvision import transforms as TV1
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from ultralytics.data.augment import classify_augmentations


# Digits that are symmetric under horizontal flip.
SYMMETRIC_DIGITS: frozenset[int] = frozenset({0, 8})


# ---------------------------------------------------------------------------
# Small helper transforms
# ---------------------------------------------------------------------------

class ApplyTransform:
    """Wrap a single-argument transform so it accepts ``(image, label)``.

    The :class:`DigitDataset` always passes both ``image`` and ``label`` to its
    transform.  Validation only needs to normalise the image, so this wrapper
    silently discards the label.
    """

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, img: Tensor, label: int) -> Tensor:  # noqa: ARG002
        return self.transform(img)


class _UltralyticsColorErasing:
    """Apply the Ultralytics colour-jitter + random-erasing subset (3-ch only)."""

    def __init__(self, pipeline: TV1.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, x: Tensor) -> Tensor:
        return self.pipeline(x) if x.shape[0] == 3 else x


class _RandomBGRSwap:
    """Optionally swap RGB ↔ BGR with probability *p* (CHW tensor)."""

    def __init__(self, p: float = 0.0) -> None:
        self.p = float(p)

    def __call__(self, x: Tensor) -> Tensor:
        if self.p > 0 and random.random() < self.p:
            return x.flip(0)
        return x


class _FinalNormalize:
    """Channel-count-aware normalisation (RGB **or** greyscale)."""

    def __init__(self, norm_rgb: T.Normalize, norm_gray: T.Normalize) -> None:
        self.norm_rgb = norm_rgb
        self.norm_gray = norm_gray

    def __call__(self, x: Tensor) -> Tensor:
        return self.norm_rgb(x) if x.shape[0] == 3 else self.norm_gray(x)


class _LabelConditionalHFlip:
    """Horizontal flip that fires only for symmetric digits (0, 8)."""

    def __init__(self, p: float = 0.5, symmetric: frozenset[int] = SYMMETRIC_DIGITS) -> None:
        self.p = p
        self.symmetric = symmetric

    def __call__(self, img: Tensor, label: int) -> Tensor:
        if label in self.symmetric and random.random() < self.p:
            return F.hflip(img)
        return img


# ---------------------------------------------------------------------------
# Deterministic preprocessor (shared by caching and external datasets)
# ---------------------------------------------------------------------------

def get_preprocessor(color: bool, size: int) -> T.Compose:
    """Return a deterministic transform: colour-convert → resize → centre-crop → float32.

    Aspect ratio is preserved by resizing the shortest edge to *size* and then
    centre-cropping to a square.
    """
    steps: list[Any] = [T.RGB() if color else T.Grayscale()]
    steps.extend([
        T.Resize(size),
        T.CenterCrop((size, size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    return T.Compose(steps)


# ---------------------------------------------------------------------------
# Ultralytics colour / erasing subset
# ---------------------------------------------------------------------------

def _build_ultralytics_color_erasing(
    *,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    erasing: float,
) -> TV1.Compose:
    """Extract *only* the colour-jitter and random-erasing parts of the
    Ultralytics ``classify_augmentations`` pipeline.

    We call the full factory with dummy spatial parameters, then strip every
    transform whose class-name matches a spatial / conversion step.
    """
    full = classify_augmentations(
        size=1,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        hflip=0.0,
        vflip=0.0,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        erasing=erasing,
    )
    drop_names = {
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "ToTensor",
        "Normalize",
    }
    kept = [t for t in getattr(full, "transforms", []) if type(t).__name__ not in drop_names]
    return TV1.Compose(kept)


# ---------------------------------------------------------------------------
# Full YOLO-style augmentation callable
# ---------------------------------------------------------------------------

class YOLOAugment:
    """Full YOLO-style augmentation pipeline for classification tensors.

    The callable signature is ``(image, label) -> image`` so it can be used
    directly as the ``transform`` of :class:`DigitDataset`.

    Parameters
    ----------
    hsv_h, hsv_s, hsv_v : float
        Colour-jitter gains for hue, saturation and value.
    degrees, translate, scale, shear, perspective : float
        Geometric augmentation parameters.
    fliplr, flipud : float
        Flip probabilities.  Horizontal flip is label-conditional.
    bgr : float
        RGB ↔ BGR channel-swap probability (default off).
    erasing : float
        Random-erasing probability.
    mean, std : tuple[float, ...] | None
        Per-channel normalisation statistics applied at the end.
    size : int | None
        If given, a ``RandomResizedCrop`` is prepended to the pipeline.
    """

    def __init__(
        self,
        *,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        bgr: float = 0.0,
        erasing: float = 0.4,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        size: int | None = None,
    ) -> None:
        self._conditional_hflip = _LabelConditionalHFlip(p=fliplr)

        ultra_pipeline = _build_ultralytics_color_erasing(
            hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, erasing=erasing,
        )

        if mean is None or std is None:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        norm_rgb = T.Normalize(mean=tuple(mean), std=tuple(std))
        norm_gray = T.Normalize(mean=(float(mean[0]),), std=(float(std[0]),))

        steps: list[Any] = []
        if size is not None:
            steps.append(T.RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))

        steps.extend([
            _UltralyticsColorErasing(ultra_pipeline),
            _RandomBGRSwap(p=bgr),
            T.RandomAffine(
                degrees=degrees,
                translate=(translate, translate) if translate > 0 else None,
                scale=(max(1 - scale, 0.01), 1 + scale) if scale > 0 else None,
                shear=(-shear, shear, -shear, shear) if shear > 0 else None,
            ),
            T.RandomPerspective(
                distortion_scale=perspective,
                p=0.5 if perspective > 0 else 0.0,
            ),
            T.RandomVerticalFlip(p=flipud),
            _FinalNormalize(norm_rgb, norm_gray),
        ])
        self._transform = T.Compose(steps)

    def __call__(self, img: Tensor, label: int | None = None) -> Tensor:
        img = self._transform(img)
        if label is not None:
            img = self._conditional_hflip(img, label)
        return img


def build_yolo_augmentor(
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    size: int | None = None,
    cfg: "AugmentConfig | None" = None,
) -> YOLOAugment:
    """Convenience factory with digit-safe defaults.

    If an :class:`~digit_classifier.config.AugmentConfig` is passed the values
    are taken from it; otherwise the class defaults are used.
    """
    if cfg is not None:
        return YOLOAugment(
            fliplr=cfg.fliplr, erasing=cfg.erasing, scale=cfg.scale,
            degrees=cfg.degrees, shear=cfg.shear, translate=cfg.translate,
            hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
            perspective=cfg.perspective, flipud=cfg.flipud, bgr=cfg.bgr,
            mean=mean, std=std, size=size,
        )
    return YOLOAugment(
        fliplr=0.5, erasing=0.1, scale=0.2, degrees=15.0, shear=4.0,
        translate=0.15, mean=mean, std=std, size=size,
    )
