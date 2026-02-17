"""Training loop with all the bells and whistles.

Preserves every behavioural invariant from the original pipeline:

- Dynamic loss switching via :func:`select_train_criterion`.
- EMA model with ``use_buffers=True``.
- Warm-restart scheduler with pre-restart checkpoints.
- ``RatioBatchSampler`` when external data is present.
- Mixup / CutMix disabled for the final *N* epochs.
"""

from __future__ import annotations

import os
from multiprocessing import cpu_count, freeze_support

import numpy as np
import torch
import torch.nn as nn
import wandb
from rich.console import Console
from rich.pretty import pretty_repr
from rich.table import Table
from timm.loss import SoftTargetCrossEntropy
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, Metric

from digit_classifier.config import Config
from digit_classifier.external import DEFAULT_EXTERNAL_FRACTIONS
from digit_classifier.mixup import MixupCutmixApply, create_mixup_cutmix
from digit_classifier.model import ResNeXt
from digit_classifier.sampler import RatioBatchSampler
from digit_classifier.splitting import split_dataset

console = Console()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_device() -> tuple[torch.device, str]:
    """Return the best available device and its short name."""
    if torch.cuda.is_available():
        name = "cuda"
    elif torch.backends.mps.is_available():
        name = "mps"
    else:
        name = "cpu"
    return torch.device(name), name


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _build_metrics(num_classes: int, device: torch.device) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {
        "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
        "top_2_accuracy": Accuracy(task="multiclass", num_classes=num_classes, top_k=2),
        "top_3_accuracy": Accuracy(task="multiclass", num_classes=num_classes, top_k=3),
        "top_5_accuracy": Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
        "top_9_accuracy": Accuracy(task="multiclass", num_classes=num_classes, top_k=9),
        "f1_score": F1Score(task="multiclass", num_classes=num_classes),
    }
    for m in metrics.values():
        m.to(device)
    return metrics


def _update_metrics(metrics: dict[str, Metric], logits: Tensor, labels: Tensor) -> None:
    _, preds = torch.max(logits, 1)
    for metric in metrics.values():
        if hasattr(metric, "top_k") and metric.top_k > 1:
            metric.update(logits, labels)
        else:
            metric.update(preds, labels)


def _compute_and_reset(metrics: dict[str, Metric]) -> dict[str, float]:
    values = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return values


# ---------------------------------------------------------------------------
# Loss selection
# ---------------------------------------------------------------------------

def select_train_criterion(active_mixup: MixupCutmixApply | None) -> nn.Module:
    """Return the appropriate loss for the current mixup state.

    When mixup produces soft targets we need ``SoftTargetCrossEntropy``;
    when mixup is off, plain ``CrossEntropyLoss`` with integer labels.
    """
    if active_mixup is not None:
        return SoftTargetCrossEntropy()
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Warm-restart epoch computation
# ---------------------------------------------------------------------------

def compute_warm_restart_epochs(
    warmup_epochs: int,
    t0: int,
    t_mult: int,
    num_epochs: int,
) -> list[int]:
    """Return 0-based epoch indices where a cosine warm-restart begins."""
    restarts: list[int] = []
    cycle_len = t0
    cursor = warmup_epochs
    while cursor + cycle_len <= num_epochs:
        cursor += cycle_len
        restarts.append(cursor)
        cycle_len *= t_mult
    return restarts


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Metric],
    scaler: GradScaler,
    device: torch.device,
    mixup_fn: MixupCutmixApply | None = None,
    ema: AveragedModel | None = None,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch and return computed metrics + loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    pin = device.type == "cuda"

    for images, labels in loader:
        if images.dtype != torch.float32:
            images = images.float()
        images = images.to(device, non_blocking=pin)
        labels = labels.long().to(device, non_blocking=pin)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, labels)

        num_batches += 1
        running_loss += (loss.item() - running_loss) / num_batches

        # Recover hard labels for metrics when mixup produced soft targets.
        if labels.dim() == 2 and labels.dtype.is_floating_point:
            _update_metrics(metrics, logits, labels.argmax(dim=1))
        else:
            _update_metrics(metrics, logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update_parameters(model)

    result = _compute_and_reset(metrics)
    result["cross_entropy"] = running_loss
    return result


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metrics: dict[str, Metric],
    device: torch.device,
) -> dict[str, float]:
    """Run one validation pass and return computed metrics + loss."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    pin = device.type == "cuda"

    with torch.no_grad():
        for images, labels in loader:
            if images.dtype != torch.float32:
                images = images.float()
            images = images.to(device, non_blocking=pin)
            labels = labels.long().to(device, non_blocking=pin)

            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)

            num_batches += 1
            running_loss += (loss.item() - running_loss) / num_batches
            _update_metrics(metrics, logits, labels)

    result = _compute_and_reset(metrics)
    result["cross_entropy"] = running_loss
    return result


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

def _create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    primary_fraction: float,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    num_workers = min(8, max(1, cpu_count() - 1))
    pin_memory = device.type == "cuda"
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    common: dict = dict(
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
    )

    original_count = getattr(train_dataset, "num_original", None)
    if original_count is not None:
        sampler = RatioBatchSampler(
            original_count=int(original_count),
            total_count=len(train_dataset),
            batch_size=batch_size,
            primary_fraction=primary_fraction,
            drop_last=True,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **common,
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _log_epoch_table(
    epoch: int,
    train: dict[str, float],
    val_raw: dict[str, float],
    val_ema: dict[str, float],
    lr: float,
) -> None:
    table = Table(title=f"Epoch {epoch}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Train", justify="right")
    table.add_column("Val (raw)", justify="right")
    table.add_column("Val (EMA)", justify="right")

    all_keys = dict.fromkeys(list(train) + list(val_raw) + list(val_ema))
    for key in all_keys:
        table.add_row(
            key,
            f"{train.get(key, 0):.5f}",
            f"{val_raw.get(key, 0):.5f}",
            f"{val_ema.get(key, 0):.5f}",
        )
    table.add_row("lr", f"{lr:.2e}", "", "")
    console.print(table)


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def load_cached_dataset(cfg: Config) -> tuple[Tensor, Tensor, tuple | None, tuple | None]:
    """Load the preprocessed ``.npz`` file.

    The cache may store images as uint8 (disk-efficient) or float32.  Either
    way this function returns float32 tensors in [0, 1].
    """
    path = os.path.join("datasets", cfg.data.dataset_name + ".npz")
    data = np.load(path)
    images = torch.from_numpy(data["images"])
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()
    mean = tuple(data["mean"]) if "mean" in data else None
    std = tuple(data["std"]) if "std" in data else None
    return images, labels, mean, std


def train(cfg: Config) -> None:
    """Run the full training pipeline driven by *cfg*."""
    device, device_name = _detect_device()
    console.print(f"[bold]Device:[/bold] {device}")

    # --- CUDA-specific backend flags ---
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if getattr(torch.backends, "cudnn", None) is not None:
            conv_obj = getattr(torch.backends.cudnn, "conv", None)
            if conv_obj is not None and hasattr(conv_obj, "fp32_precision"):
                try:
                    conv_obj.fp32_precision = "tf32"
                except Exception:
                    pass
        cuda_be = getattr(torch.backends, "cuda", None)
        if cuda_be is not None:
            matmul_obj = getattr(cuda_be, "matmul", None)
            if matmul_obj is not None and hasattr(matmul_obj, "fp32_precision"):
                try:
                    matmul_obj.fp32_precision = "tf32"
                except Exception:
                    pass

    # --- Data ---
    images, labels, cached_mean, cached_std = load_cached_dataset(cfg)
    console.print(f"Loaded [cyan]{cfg.data.dataset_name}[/cyan]: {images.shape}, mean={cached_mean}, std={cached_std}")

    train_dataset, val_dataset, mean, std = split_dataset(
        images, labels, cached_mean, cached_std,
        train_fraction=cfg.data.train_fraction,
        mix_external=cfg.data.mix_external,
        external_fractions=DEFAULT_EXTERNAL_FRACTIONS if cfg.data.mix_external else None,
        color=cfg.data.color,
        size=cfg.data.image_size,
        seed=cfg.data.split_seed,
        augment_cfg=cfg.augment,
    )
    console.print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    train_loader, val_loader = _create_dataloaders(
        train_dataset, val_dataset, cfg.data.batch_size,
        cfg.data.primary_fraction, device,
    )

    # --- Mixup / CutMix ---
    tc = cfg.training
    mixup = MixupCutmixApply(create_mixup_cutmix(
        num_classes=cfg.model.num_classes,
        mixup_alpha=tc.mixup_alpha,
        cutmix_alpha=tc.cutmix_alpha,
        cutmix_minmax=tc.cutmix_minmax,
        prob=tc.mixup_prob,
        label_smoothing=tc.label_smoothing,
        mode=tc.mixup_mode,
    ))

    # --- Model ---
    model_name = "ResNeXt"  # Capture before torch.compile changes __class__
    model = ResNeXt(
        layers=list(cfg.model.layers),
        num_classes=cfg.model.num_classes,
        groups=cfg.model.groups,
        width_per_group=cfg.model.width_per_group,
        drop_path_rate=cfg.model.drop_path_rate,
    ).to(device)

    if tc.compile_model:
        model = torch.compile(model)

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tc.ema_decay), use_buffers=True)

    # --- Optimiser & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=tc.warmup_epochs)
    main_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=tc.scheduler_t0, T_mult=tc.scheduler_t_mult, eta_min=tc.eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[tc.warmup_epochs],
    )

    warm_restart_epochs = compute_warm_restart_epochs(
        tc.warmup_epochs, tc.scheduler_t0, tc.scheduler_t_mult, tc.epochs,
    )
    if warm_restart_epochs:
        console.print(f"[bold]Warm-restart epochs:[/bold] {warm_restart_epochs}")

    # --- Metrics ---
    metrics = _build_metrics(cfg.model.num_classes, device)
    val_criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # --- Wandb ---
    # Resolve the effective batch size for logging (batch_sampler → None for .batch_size)
    effective_batch_size = cfg.data.batch_size

    if tc.wandb_enabled:
        wandb.init(
            project=tc.wandb_project,
            config={
                "dataset": cfg.data.dataset_name,
                "model": model_name,
                "optimizer": optimizer.__class__.__name__,
                "lr": tc.lr,
                "batch_size": effective_batch_size,
                "epochs": tc.epochs,
                "device": device_name,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)
        checkpoint_dir = os.path.join("checkpoints", wandb.run.id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        wandb.config.update({"mean": mean, "std": std})

    # --- Training loop ---
    best_val_accuracy = 0.0

    for epoch in range(tc.epochs):
        # Disable mixup for the final N epochs.
        active_mixup = mixup if epoch < tc.epochs - tc.mixup_off_last_n else None
        if epoch == tc.epochs - tc.mixup_off_last_n:
            console.print(f"[yellow]Disabling mixup for final {tc.mixup_off_last_n} epochs[/yellow]")

        train_criterion = select_train_criterion(active_mixup)

        train_metrics = train_epoch(
            model, train_loader, train_criterion, optimizer, metrics, scaler,
            device, mixup_fn=active_mixup, ema=ema, grad_clip_norm=tc.grad_clip_norm,
        )
        val_metrics_raw = validate(model, val_loader, val_criterion, metrics, device)
        val_metrics_ema = validate(ema, val_loader, val_criterion, metrics, device)

        # --- Pre-restart checkpoint (before scheduler.step) ---
        if warm_restart_epochs and (epoch + 1) in warm_restart_epochs:
            ckpt_root = checkpoint_dir if "checkpoint_dir" in dir() else "checkpoints"
            os.makedirs(ckpt_root, exist_ok=True)
            pre_path = os.path.join(ckpt_root, f"pre_restart_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_metrics_ema.get("accuracy"),
            }, pre_path)
            console.print(f"[magenta]Saved pre-restart checkpoint:[/magenta] {pre_path}")

            if tc.wandb_enabled:
                try:
                    art = wandb.Artifact(f"pre-restart-epoch-{epoch + 1}", type="model",
                                         metadata={"epoch": epoch + 1, "val_accuracy": val_metrics_ema.get("accuracy")})
                    art.add_file(pre_path)
                    wandb.log_artifact(art)
                except Exception:
                    pass

        scheduler.step()

        # --- Logging ---
        current_lr = scheduler.get_last_lr()[0]
        _log_epoch_table(epoch + 1, train_metrics, val_metrics_raw, val_metrics_ema, current_lr)

        if tc.wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "lr": current_lr,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val_raw/{k}": v for k, v in val_metrics_raw.items()},
                **{f"val_ema/{k}": v for k, v in val_metrics_ema.items()},
            })

        # --- Best-model checkpoint ---
        val_accuracy = val_metrics_ema["accuracy"]
        if tc.wandb_enabled and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
            }, ckpt_path)

            art = wandb.Artifact("model-best", type="model",
                                 metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy})
            art.add_file(ckpt_path)
            wandb.log_artifact(art)
            console.print(f"[green]Saved best model (val_accuracy={val_accuracy:.4f}) at epoch {epoch + 1}[/green]")

    if tc.wandb_enabled:
        wandb.finish()
    console.print("[bold green]Training complete.[/bold green]")
