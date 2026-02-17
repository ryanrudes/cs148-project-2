"""Centralised configuration dataclasses for the training pipeline.

Every value that was previously hard-coded as a module-level constant in the
original codebase is exposed here as a typed, documented field with the same
default that the original code used.  CLI flags map 1-to-1 to these fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Settings for dataset loading, splitting, and external mixing."""

    dataset_name: str = "mnist_rgb_224"
    image_size: int = 224
    color: bool = True
    train_fraction: float = 0.9
    batch_size: int = 128
    split_seed: int = 42
    mix_external: bool = True
    primary_fraction: float = 0.95
    gdrive_url: str = (
        "https://drive.google.com/uc?id=1_gIar-Q89tWll-dnJUE077UujzAVMPxQ"
    )


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the ResNeXt model."""

    layers: tuple[int, ...] = (3, 4, 23, 3)
    num_classes: int = 10
    groups: int = 64
    width_per_group: int = 4
    drop_path_rate: float = 0.1


@dataclass
class AugmentConfig:
    """YOLO-style augmentation hyper-parameters (digit-safe defaults)."""

    fliplr: float = 0.5
    erasing: float = 0.1
    scale: float = 0.2
    degrees: float = 15.0
    shear: float = 4.0
    translate: float = 0.15
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    perspective: float = 0.0
    flipud: float = 0.0
    bgr: float = 0.0


@dataclass
class TrainingConfig:
    """Optimiser, scheduler, regularisation and logging settings."""

    epochs: int = 900
    warmup_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.05
    eta_min: float = 1e-8
    scheduler_t0: int = 50
    scheduler_t_mult: int = 2
    ema_decay: float = 0.995
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_minmax: tuple[float, float] = (0.02, 0.45)
    mixup_prob: float = 0.5
    mixup_mode: str = "elem"
    label_smoothing: float = 0.1
    mixup_off_last_n: int = 10
    grad_clip_norm: float = 1.0
    compile_model: bool = True
    wandb_enabled: bool = True
    wandb_project: str = "CS148-MNIST"


@dataclass
class Config:
    """Top-level container that groups every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
