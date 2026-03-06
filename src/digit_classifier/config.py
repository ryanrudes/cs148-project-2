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
    augment_scheme: str = "yolo"  # "yolo" | "three_augment" | "autoaugment"
    train_fraction: float = 0.9
    batch_size: int = 128
    repeat_aug: bool = False
    repeat_aug_repeats: int = 3
    split_seed: int = 42
    mix_external: bool = True
    external_only: bool = False  # train only on external data; skip loading primary dataset
    external_val_source: str = "MNIST Test"  # which external to use for validation when external_only
    test_dataset_path: str | None = None  # Pareidolia output dir (e.g. dataset_out) for test eval
    primary_fraction: float = 0.95
    num_workers: int = -1  # -1 = auto (cpu_count - 1, capped)
    calibrate_workers: bool = False  # benchmark to find best num_workers before training
    prefetch_factor: int = 2
    external_cache: bool = False
    external_cache_max_mb: float = 2048  # 0 = disabled; -1 = auto from available RAM; >0 = MB
    gdrive_url: str = (
        "https://drive.google.com/uc?id=1_gIar-Q89tWll-dnJUE077UujzAVMPxQ"
    )


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for ResNeXt or DeiT."""

    model_type: str = "deit"  # "resnext" or "deit"
    layers: tuple[int, ...] = (3, 4, 23, 3)
    num_classes: int = 10
    groups: int = 64
    width_per_group: int = 4
    drop_path_rate: float = 0.1
    use_flash_attention: bool = False
    deit_model: str = "base"
    layer_scale_init: float = 1e-4  # LayerScale init (DeiT-III uses 1e-4)


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
    eta_min: float = 1e-5  # DeiT-III / timm default (min_lr)
    scheduler_t0: int = 50
    scheduler_t_mult: int = 2
    ema_decay: float = 0.995
    ema_enabled: bool = True
    # toggle cosine warm-restarts; when False use a single cycle CosineAnnealingLR
    warm_restarts: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_minmax: tuple[float, float] | None = (0.02, 0.45)  # None = DeiT-III style (standard CutMix)
    mixup_prob: float = 0.5
    mixup_mode: str = "elem"
    label_smoothing: float = 0.1
    bce_loss: bool = False
    mixup_off_last_n: int = 10
    drop_path_increment: float = 0.0  # add this much to stochastic depth every N epochs (0 = disabled)
    drop_path_increment_every: int = 0  # increment interval in epochs (0 = disabled)
    grad_clip_norm: float = 1.0
    weight_decay_exclude: bool = True
    layer_decay: float = 0.0
    amp_enabled: bool = True
    compile_model: bool = True
    wandb_enabled: bool = True
    wandb_project: str = "CS148-MNIST"
    replace_best_checkpoint: bool = True  # overwrite best.pt / model-best artifact instead of accumulating
    checkpoint_enabled: bool = True  # when False, no checkpoints are saved to disk or wandb
    val_every_n_epochs: int = 1  # validate every N epochs (1 = every epoch)
    progress_bars: bool = False  # show Rich progress bars for each epoch's batches
    show_data_wait: bool = False  # show % of time GPU waits for data (requires progress_bars)
    wandb_watch: str = "gradients"  # "gradients" | "all" | "none"
    resume_path: str | None = None  # full resume (optimizer, scheduler, epoch); same resolution
    pretrain_path: str | None = None  # load weights only; allows different resolution for fine-tune


@dataclass
class Config:
    """Top-level container that groups every sub-config."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
