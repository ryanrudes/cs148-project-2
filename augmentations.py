"""
This is an implementation of the YOLO26 image augmentation pipeline.
Details are here:
https://docs.ultralytics.com/guides/yolo-data-augmentation/#using-a-configuration-file
Images entering this pipeline are already tensors of shape (C, H, W) in [0, 1]
(from ToTensor), resized/grayscaled as needed, and cached for quick loading.

The deterministic preprocessing operations like resize, grayscale, and normalization are not
included here, since they are applied before the random augmentations for efficiency and
cached locally for quick loading. This module implements both the random and
deterministic augmentations, but the preprocessed data is prepared and cached in
preprocess_data.py.
"""

from ultralytics.data.augment import classify_augmentations

from torchvision import transforms as TV1
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F

from torch import Tensor
from PIL import Image

import torch
import random


# Digits that are always symmetric under horizontal flip
SYMMETRIC_DIGITS = {0, 8}


class ApplyUltralyticsColorErasing:
    """Apply cached Ultralytics color+erasing pipeline only for 3-channel tensors."""

    def __init__(self, ultra_color_erasing):
        self.ultra_color_erasing = ultra_color_erasing

    def __call__(self, x: Tensor) -> Tensor:
        return self.ultra_color_erasing(x) if x.shape[0] == 3 else x


class RandomBGRSwap:
    """Optionally swap RGB<->BGR with probability p (expects CHW tensor)."""

    def __init__(self, p: float = 0.0):
        self.p = float(p)

    def __call__(self, x: Tensor) -> Tensor:
        if self.p > 0 and random.random() < self.p:
            return x.flip(0)
        return x


class FinalNormalize:
    """Normalize once at the end, supporting RGB or grayscale."""

    def __init__(self, norm_rgb, norm_gray):
        self.norm_rgb = norm_rgb
        self.norm_gray = norm_gray

    def __call__(self, x: Tensor) -> Tensor:
        return self.norm_rgb(x) if x.shape[0] == 3 else self.norm_gray(x)


def convert_to_rgb_if_needed(img: Image) -> Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def get_preprocessor(
    color: bool,
    size: int,
) -> T.Compose:
    if color:
        transforms = [T.RGB()]
    else:
        transforms = [T.Grayscale()]

    transforms.extend([
        T.Resize((size, size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    return T.Compose(transforms)


class LabelConditionalHFlip:
    """Random does a horizontal flip only when the label is a symmetric digit."""
    def __init__(self, p: float = 0.5, symmetric_labels: set[int] = SYMMETRIC_DIGITS):
        self.p = p
        self.symmetric_labels = symmetric_labels

    def __call__(self, img: Tensor, label: int) -> Tensor:
        if label in self.symmetric_labels and random.random() < self.p:
            return F.hflip(img)
        return img


def build_ultralytics_color_erasing(
    *,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    erasing: float,
    auto_augment: str | None = None,
    force_color_jitter: bool = False,
    interpolation: str = "BILINEAR",
) -> TV1.Compose:
    """Reuse Ultralytics classify_augmentations(), but drop its preprocessing steps.

    Ultralytics' classify_augmentations() returns:
      RandomResizedCrop -> (optional flips) -> (optional AA) -> ColorJitter -> ToTensor -> Normalize -> RandomErasing

    Our pipeline already does deterministic preprocessing elsewhere and caches it.
    So here we keep only the *color* and *random erasing* pieces.
    """
    t = classify_augmentations(
        size=1,  # required int by Ultralytics; we strip the crop anyway
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        hflip=0.0,
        vflip=0.0,
        auto_augment=auto_augment,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
        interpolation=interpolation,
    )

    drop = {
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "ToTensor",
        "Normalize",
    }
    kept = [tr for tr in getattr(t, "transforms", []) if tr.__class__.__name__ not in drop]
    return TV1.Compose(kept)


class YOLOAugment:
    """Full YOLO-style augmentation pipeline for classification tensors.

    Wraps all individual augmentations with YOLO26 default hyper-parameters.

    Args:
        hsv_h:       hue gain              (default 0.015)
        hsv_s:       saturation gain        (default 0.7)
        hsv_v:       brightness gain        (default 0.4)
        degrees:     max rotation degrees   (default 0.0)
        translate:   max translation frac   (default 0.1)
        scale:       scale range            (default 0.5  → [1-s, 1+s])
        shear:       max shear degrees      (default 0.0)
        perspective: perspective distortion (default 0.0)
        fliplr:      horiz-flip probability (default 0.5)
        flipud:      vert-flip probability  (default 0.0)
        bgr:         channel-swap prob      (default 0.0)
        erasing:     random-erase prob      (default 0.4; your preset overrides it)
        mean:       normalization mean(s) applied at the end (default 0.5 per channel)
        std:        normalization std(s) applied at the end  (default 0.5 per channel)
    """

    def __init__(
        self,
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
    ):
        self.conditional_hflip = LabelConditionalHFlip(p=fliplr)

        # Build once (not per-sample) to avoid overhead.
        # Only valid for 3-channel inputs; for 1-channel we skip this and rely on other augs.
        self.ultra_color_erasing = build_ultralytics_color_erasing(
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            erasing=erasing,
        )

        # Final normalization (applied once at the end). Defaults to mapping [0,1] -> [-1,1].
        if mean is None or std is None:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self._norm_rgb = T.Normalize(mean=tuple(mean), std=tuple(std))
        # Grayscale fallback uses the first channel's stats.
        self._norm_gray = T.Normalize(mean=(float(mean[0]),), std=(float(std[0]),))

        self.transform = T.Compose([
            # Color + erasing augmentations (Ultralytics classify_augmentations subset)
            # Apply only when the tensor has 3 channels.
            ApplyUltralyticsColorErasing(self.ultra_color_erasing),

            # Optional RGB<->BGR channel swap (kept for parity with YOLO hyp, default off)
            RandomBGRSwap(p=bgr),

            # Geometric augmentations
            T.RandomAffine(
                degrees=degrees,
                translate=(translate, translate) if translate > 0 else None,
                scale=(max(1 - scale, 0.01), 1 + scale) if scale > 0 else None,
                shear=(-shear, shear, -shear, shear) if shear > 0 else None,
            ),
            T.RandomPerspective(distortion_scale=perspective, p=0.5 if perspective > 0 else 0.0),
            # NOTE: hflip is handled by LabelConditionalHFlip, not here
            T.RandomVerticalFlip(p=flipud),

            # Normalize once at the end (supports RGB or grayscale)
            FinalNormalize(self._norm_rgb, self._norm_gray),
        ])

    def __call__(self, img: Tensor, label: int | None = None) -> Tensor:
        img = self.transform(img)
        if label is not None:
            img = self.conditional_hflip(img, label)
        return img


def get_yolo_augmentor(mean: tuple[float, ...] | None = None, std: tuple[float, ...] | None = None) -> YOLOAugment:
    # YOLO26-style augmentation pipeline with digit-safe hyperparameters
    augment = YOLOAugment(
        fliplr=0.5,       # horizontal flip (only applied to symmetric digits: 0, 8)
        erasing=0.1,      # mild erasing (maybe go back to 0.1)
        scale=0.2,        # gentler zoom range (0.8x–1.2x)
        degrees=15,       # small rotation is fine, just not large
        shear=4.0,        # small shear is fine, just not large
        translate=0.15,
        mean=mean,
        std=std,
    )

    return augment
