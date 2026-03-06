"""Augmentation pipelines: YOLO-style, 3-Augment (DeiT-III), and AutoAugment SVHN.

Images entering these pipelines are tensors of shape ``(C, H, W)`` in [0, 1]
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
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from ultralytics.data.augment import classify_augmentations

from digit_classifier.config import AugmentConfig


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
    cfg: AugmentConfig | None = None,
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


# ---------------------------------------------------------------------------
# 3-Augment (DeiT-III): grayscale / solarize / gaussian blur → color jitter → hflip
#
# Matches the official DeiT implementation (facebookresearch/deit augment.py) and
# timm's auto_augment_policy_3a (timm/data/auto_augment.py). References:
#   - https://github.com/facebookresearch/deit/blob/main/augment.py
#   - https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/auto_augment.py
# ---------------------------------------------------------------------------

# PIL ImageOps.solarize uses threshold=128 by default; for float [0,1] that is 128/255
_SOLARIZE_THRESHOLD_DEIT = 128 / 255.0

# DeiT main.py --color-jitter default is 0.3 (brightness, contrast, saturation; no hue)
_COLOR_JITTER_DEIT = (0.3, 0.3, 0.3, 0.0)

# GaussianBlur: DeiT uses radius in [0.1, 2.0]; we use sigma in same range for torchvision
_BLUR_SIGMA_DEIT = (0.1, 2.0)


class _ThreeAugmentRandom:
    """Apply one of grayscale, solarize, or gaussian blur with equal probability.

    Matches DeiT's RandomChoice([gray_scale(p=1), Solarization(p=1), GaussianBlur(p=1)]).
    """

    def __init__(
        self,
        solarize_threshold: float = _SOLARIZE_THRESHOLD_DEIT,
        blur_kernel_size: int = 23,
        blur_sigma: tuple[float, float] = _BLUR_SIGMA_DEIT,
    ) -> None:
        self._grayscale = T.RandomGrayscale(p=1.0)
        self._solarize = T.RandomSolarize(threshold=solarize_threshold, p=1.0)
        self._blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)

    def __call__(self, x: Tensor) -> Tensor:
        r = random.random()
        if r < 1 / 3:
            return self._grayscale(x)
        if r < 2 / 3:
            return self._solarize(x)
        return self._blur(x)


class ThreeAugment:
    """3-Augment pipeline from DeiT-III: random aug → color jitter → label-conditional hflip.

    Exact match for the official DeiT augment.py and timm's 3a policy. Order per paper:
    one of (grayscale, solarize, gaussian blur) → color jitter → horizontal flip.

    When *size* is provided, prepends RandomResizedCrop to produce the target resolution
    (allows using a cache at one resolution while training at another).

    The callable signature is ``(image, label) -> image`` for :class:`DigitDataset`.
    """

    def __init__(
        self,
        *,
        fliplr: float = 0.5,
        color_jitter: tuple[float, float, float, float] = _COLOR_JITTER_DEIT,
        solarize_threshold: float = _SOLARIZE_THRESHOLD_DEIT,
        blur_sigma: tuple[float, float] = _BLUR_SIGMA_DEIT,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        size: int | None = None,
    ) -> None:
        self._resize = (
            T.RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            if size is not None
            else None
        )
        self._conditional_hflip = _LabelConditionalHFlip(p=fliplr)
        self._random_aug = _ThreeAugmentRandom(
            solarize_threshold=solarize_threshold,
            blur_sigma=blur_sigma,
        )
        self._color_jitter = T.ColorJitter(*color_jitter)
        if mean is None or std is None:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self._norm_rgb = T.Normalize(mean=tuple(mean), std=tuple(std))
        self._norm_gray = T.Normalize(mean=(float(mean[0]),), std=(float(std[0]),))

    def __call__(self, img: Tensor, label: int | None = None) -> Tensor:
        if self._resize is not None:
            img = self._resize(img)
        img = self._random_aug(img)
        img = self._color_jitter(img)
        if label is not None:
            img = self._conditional_hflip(img, label)
        norm = self._norm_rgb if img.shape[0] == 3 else self._norm_gray
        return norm(img)


# ---------------------------------------------------------------------------
# AutoAugment SVHN policy (torchvision)
# ---------------------------------------------------------------------------

class AutoAugmentTransform:
    """AutoAugment with SVHN policy, wrapped for (image, label) -> image.

    Expects float [0, 1] input; converts to uint8 for AutoAugment, then back.

    When *size* is provided, prepends RandomResizedCrop to produce the target resolution.
    """

    def __init__(
        self,
        *,
        fliplr: float = 0.5,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        size: int | None = None,
    ) -> None:
        self._resize = (
            T.RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            if size is not None
            else None
        )
        self._autoaugment = AutoAugment(policy=AutoAugmentPolicy.SVHN)
        self._conditional_hflip = _LabelConditionalHFlip(p=fliplr)
        if mean is None or std is None:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self._norm_rgb = T.Normalize(mean=tuple(mean), std=tuple(std))
        self._norm_gray = T.Normalize(mean=(float(mean[0]),), std=(float(std[0]),))

    def __call__(self, img: Tensor, label: int | None = None) -> Tensor:
        if self._resize is not None:
            img = self._resize(img)
        # AutoAugment expects uint8 [0, 255]
        img_uint8 = (img.clamp(0, 1) * 255).to(torch.uint8)
        img_uint8 = self._autoaugment(img_uint8)
        img = img_uint8.to(torch.float32) / 255.0
        if label is not None:
            img = self._conditional_hflip(img, label)
        norm = self._norm_rgb if img.shape[0] == 3 else self._norm_gray
        return norm(img)


# ---------------------------------------------------------------------------
# Augmentor factory
# ---------------------------------------------------------------------------

AUGMENT_SCHEMES: frozenset[str] = frozenset({"yolo", "three_augment", "autoaugment"})


def build_augmentor(
    scheme: str,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    size: int | None = None,
    cfg: AugmentConfig | None = None,
) -> YOLOAugment | ThreeAugment | AutoAugmentTransform:
    """Build an augmentor by scheme name.

    Parameters
    ----------
    scheme : str
        One of ``"yolo"``, ``"three_augment"``, ``"autoaugment"``.
    mean, std : tuple[float, ...] | None
        Normalisation statistics.
    size : int | None
        Target size (used by YOLO for RandomResizedCrop).
    cfg : AugmentConfig | None
        Augment config (used by YOLO for fliplr etc.).

    Returns
    -------
    YOLOAugment | ThreeAugment | AutoAugmentTransform
        Callable with signature ``(image, label) -> image``.
    """
    scheme = scheme.lower().strip()
    if scheme not in AUGMENT_SCHEMES:
        raise ValueError(f"Unknown augment_scheme: {scheme!r}. Choose from {sorted(AUGMENT_SCHEMES)}")

    if scheme == "yolo":
        return build_yolo_augmentor(mean=mean, std=std, size=size, cfg=cfg)
    if scheme == "three_augment":
        fliplr = cfg.fliplr if cfg is not None else 0.5
        return ThreeAugment(fliplr=fliplr, mean=mean, std=std, size=size)
    if scheme == "autoaugment":
        fliplr = cfg.fliplr if cfg is not None else 0.5
        return AutoAugmentTransform(fliplr=fliplr, mean=mean, std=std, size=size)
    raise AssertionError(f"Unhandled scheme: {scheme}")
