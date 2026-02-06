from torchvision import transforms as T
from torchvision.transforms import functional as F

import torch
import random
import math

COLOR = False
SIZE = 64

if COLOR:
    preprocess = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((SIZE, SIZE)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
else:
    preprocess = T.Compose([
        T.Grayscale(),
        T.Resize((SIZE, SIZE)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])


# ---------------------------------------------------------------------------
# YOLO-style data augmentation pipeline (adapted for classification tensors)
# ---------------------------------------------------------------------------
# Default values taken from Ultralytics YOLO26 / YOLOv8 configuration.
# The images entering this pipeline are already tensors (C, H, W) normalised
# to [-1, 1] via Normalize((0.5,…), (0.5,…)).
# ---------------------------------------------------------------------------


class Unnormalize:
    """Undo Normalize((0.5,…), (0.5,…)) → map [-1, 1] back to [0, 1]."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img * 0.5 + 0.5


class Renormalize:
    """Re-apply Normalize((0.5,…), (0.5,…)) → map [0, 1] back to [-1, 1]."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img.clamp(0, 1) - 0.5) / 0.5


class HSVAugmentation:
    """YOLO-style random HSV jitter.

    Adjusts hue, saturation, and brightness (value) independently.

    Args:
        h_gain: maximum hue shift as a fraction of 0.5 (default 0.015).
        s_gain: maximum saturation scale factor (default 0.7).
        v_gain: maximum brightness scale factor (default 0.4).
    """

    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Random factors in [-gain, +gain]
        h_delta = random.uniform(-self.h_gain, self.h_gain)  # hue shift (passed to adjust_hue)
        s_factor = 1.0 + random.uniform(-self.s_gain, self.s_gain)  # saturation multiplier
        v_factor = 1.0 + random.uniform(-self.v_gain, self.v_gain)  # brightness multiplier

        img = F.adjust_hue(img, h_delta)
        img = F.adjust_saturation(img, s_factor)
        img = F.adjust_brightness(img, v_factor)
        return img.clamp(0, 1)


class RandomBGRFlip:
    """Randomly swap RGB channels to BGR with a given probability.

    Args:
        p: probability of applying the channel swap (default 0.0, matching YOLO).
    """

    def __init__(self, p: float = 0.0):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return img.flip(0)  # reverse channel dim
        return img


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
        erasing:     random-erase prob      (default 0.4)
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
    ):
        self.transform = T.Compose([
            # Unnormalise so colour operations work on [0, 1]
            Unnormalize(),

            # Color augmentations
            HSVAugmentation(h_gain=hsv_h, s_gain=hsv_s, v_gain=hsv_v),
            RandomBGRFlip(p=bgr),

            # Geometric augmentations
            T.RandomAffine(
                degrees=degrees,
                translate=(translate, translate) if translate > 0 else None,
                scale=(max(1 - scale, 0.01), 1 + scale) if scale > 0 else None,
                shear=(-shear, shear, -shear, shear) if shear > 0 else None,
            ),
            T.RandomPerspective(distortion_scale=perspective, p=0.5 if perspective > 0 else 0.0),
            T.RandomHorizontalFlip(p=fliplr),
            T.RandomVerticalFlip(p=flipud),

            # Random erasure operates on tensors
            T.RandomErasing(p=erasing, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),

            # Renormalise back to [-1, 1]
            Renormalize(),
        ])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)


# YOLO26-style augmentation pipeline with digit-safe hyperparameters
augment = YOLOAugment(
    fliplr=0.0,       # no horizontal flip
    erasing=0.1,      # mild erasing
    scale=0.2,        # gentler zoom range (0.8x–1.2x)
    degrees=15,       # small rotation is fine, just not large (maybe go back to 5?)
    shear=4.0,        # small shear is fine, just not large
    translate=0.15,
)
