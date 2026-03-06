"""Training loop with all the bells and whistles.

Preserves every behavioural invariant from the original pipeline:

- Dynamic loss switching via :func:`select_train_criterion`.
- Optional EMA model with ``use_buffers=True`` (can be disabled via config).
- Warm-restart scheduler with pre-restart checkpoints (can be turned off via config).
- ``RatioBatchSampler`` when external data is present.
- Mixup / CutMix disabled for the final *N* epochs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import asdict
import sys
import threading
from contextlib import nullcontext
from multiprocessing import cpu_count, freeze_support

import numpy as np
import torch
import torch.nn as nn
import wandb
from rich.console import Console
from rich.pretty import pretty_repr
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from timm.loss import SoftTargetCrossEntropy
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from torchmetrics import Accuracy, F1Score, Metric

from digit_classifier.config import Config
from digit_classifier.external import (
    CachedExternalDataset,
    DEFAULT_EXTERNAL_FRACTIONS,
    ExternalOnDemandDataset,
)
from digit_classifier.loss import BCELossWithSmoothing
from digit_classifier.mixup import MixupCutmixApply, create_mixup_cutmix
from digit_classifier.model import ResNeXt
from digit_classifier.vit import build_deit3
from digit_classifier.sampler import RatioBatchSampler, RepeatAugRatioBatchSampler, RepeatAugSampler
from digit_classifier.splitting import split_dataset, split_dataset_external_only

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


def _setup_ddp() -> tuple[int, int, int]:
    """Initialize DDP when WORLD_SIZE > 1. Returns (rank, world_size, local_rank)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        import torch.distributed as dist
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = os.environ.get("DDP_BACKEND", "nccl")
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


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

def select_train_criterion(
    active_mixup: MixupCutmixApply | None,
    num_classes: int,
    bce_loss: bool = False,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """Return the appropriate loss for the current mixup state and config.

    When bce_loss is True (DeiT-III style): use BCEWithLogitsLoss with smoothing.
    Otherwise: SoftTargetCrossEntropy when mixup is active, CrossEntropyLoss when off.
    """
    if bce_loss:
        return BCELossWithSmoothing(num_classes=num_classes, smoothing=label_smoothing)
    if active_mixup is not None:
        return SoftTargetCrossEntropy()
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Parameter groups (weight decay exclusion, layer-wise LR decay)
# ---------------------------------------------------------------------------

def _get_layer_id_for_vit(name: str, num_blocks: int) -> int:
    """Assign layer ID for ViT/DeiT. Embedding=0, blocks=1..depth, norm+head=depth+1."""
    if name.startswith("patch_embed") or name.startswith("cls_token") or name.startswith("pos_embed"):
        return 0
    if name.startswith("blocks."):
        # blocks.0.xxx -> 1, blocks.1.xxx -> 2, ...
        block_idx = int(name.split(".")[1])
        return block_idx + 1
    return num_blocks + 1


def _get_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    weight_decay_exclude: bool,
    layer_decay: float,
) -> list[dict]:
    """Build param groups with optional weight decay exclusion and layer-wise LR decay."""
    if not weight_decay_exclude and layer_decay <= 0:
        return [{"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay}]

    # Count transformer blocks for layer decay
    num_blocks = max(
        (int(n.split(".")[1]) for n in model.state_dict() if n.startswith("blocks.")),
        default=-1,
    ) + 1
    num_layers = num_blocks + 2  # embedding + blocks + norm/head

    groups: dict[tuple[int, float], list[torch.nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Weight decay: exclude 1D params (bias), LayerNorm, LayerScale gamma
        if weight_decay_exclude and (
            param.ndim <= 1 or "norm" in name or "bias" in name or "gamma" in name
        ):
            wd = 0.0
        else:
            wd = weight_decay

        # Layer ID for LR scaling
        if num_blocks > 0:
            layer_id = _get_layer_id_for_vit(name, num_blocks)
        else:
            layer_id = 0

        key = (layer_id, wd)
        if key not in groups:
            groups[key] = []
        groups[key].append(param)

    param_groups = []
    for (layer_id, wd), params in sorted(groups.items()):
        if num_blocks > 0 and layer_decay > 0:
            lr_scale = layer_decay ** (num_layers - 1 - layer_id)
        else:
            lr_scale = 1.0
        param_groups.append({"params": params, "lr": lr * lr_scale, "weight_decay": wd})
    return param_groups


# ---------------------------------------------------------------------------
# Warm-restart epoch computation
# ---------------------------------------------------------------------------

def _config_to_wandb_dict(cfg: Config) -> dict:
    """Flatten all config into wandb-friendly dict with prefixed keys for sweep plots."""
    out: dict = {}
    for section, obj in [
        ("data", cfg.data),
        ("model", cfg.model),
        ("augment", cfg.augment),
        ("training", cfg.training),
    ]:
        d = asdict(obj)
        for k, v in d.items():
            if k == "gdrive_url":
                continue  # skip long URL
            key = f"{section}/{k}"
            if isinstance(v, tuple):
                v = list(v)
            out[key] = v
    return out


def _get_model_config_for_checkpoint(mc, image_size: int, patch_size: int | None = None) -> dict:
    """Build model_config dict with only params relevant to the model type."""
    base = {"num_classes": mc.num_classes}
    if mc.model_type == "resnext":
        base.update({
            "layers": list(mc.layers),
            "groups": mc.groups,
            "width_per_group": mc.width_per_group,
        })
    else:
        from digit_classifier.vit import _patch_size_for_image_size
        ps = patch_size if patch_size is not None else _patch_size_for_image_size(image_size, mc.deit_model)
        base.update({
            "deit_model": mc.deit_model,
            "image_size": image_size,
            "patch_size": ps,
        })
    return base


def _all_reduce_metrics(metrics_dict: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    """All-reduce metrics across DDP ranks (average)."""
    if world_size <= 1:
        return metrics_dict
    import torch.distributed as dist
    out = {}
    for k, v in metrics_dict.items():
        t = torch.tensor(v, dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = (t / world_size).item()
    return out


def _get_deit_model(model: nn.Module) -> nn.Module | None:
    """Get the underlying DeiT3 from a possibly torch.compile- or DDP-wrapped model."""
    m = getattr(model, "_orig_mod", model)
    m = getattr(m, "module", m)
    return m if hasattr(m, "set_drop_path_rate") else None


def _infer_model_type_from_state_dict(state_dict: dict) -> str:
    """Infer model type from state dict keys (for old checkpoints without model_type)."""
    keys = list(state_dict.keys())
    if any(k.startswith("blocks.") for k in keys):
        return "deit"
    return "resnext"


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    *,
    model_type: str | None = None,
    target_image_size: int | None = None,
) -> tuple[nn.Module, dict]:
    """Load checkpoint and build the appropriate model (ResNeXt or DeiT).

    For DeiT, when target_image_size differs from the checkpoint's image_size,
    position embeddings are interpolated to the new resolution.

    Returns (model, ckpt_dict). If model_type is None, uses checkpoint metadata
    or infers from state dict keys.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint must contain 'ema_state_dict' or 'model_state_dict'")

    mt = model_type or ckpt.get("model_type") or _infer_model_type_from_state_dict(state)
    config = ckpt.get("model_config", {})

    num_classes = config.get("num_classes", 10)
    stripped = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}

    if mt == "resnext":
        model = ResNeXt(
            layers=config.get("layers", [3, 4, 23, 3]),
            num_classes=num_classes,
            groups=config.get("groups", 64),
            width_per_group=config.get("width_per_group", 4),
        ).to(device)
    else:
        from digit_classifier.vit import build_deit3, resize_pos_embed

        ckpt_image_size = config.get("image_size", 224)
        effective_image_size = target_image_size if target_image_size is not None else ckpt_image_size

        ckpt_patch_size = config.get("patch_size")
        if ckpt_patch_size is None:
            ckpt_patch_size = 14 if config.get("deit_model", "base") == "huge" else 16
        if ckpt_image_size != effective_image_size and "pos_embed" in stripped:
            stripped["pos_embed"] = resize_pos_embed(
                stripped["pos_embed"],
                orig_size=ckpt_image_size,
                new_size=effective_image_size,
                patch_size=ckpt_patch_size,
            )

        model = build_deit3(
            size=config.get("deit_model", "base"),
            num_classes=num_classes,
            drop_path_rate=0.0,
            image_size=effective_image_size,
            patch_size=ckpt_patch_size,
        ).to(device)

    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in stripped.items() if k in model_keys}
    model.load_state_dict(filtered, strict=True)
    model.eval()
    return model, ckpt


def run_eval(
    checkpoint_path: str,
    test_dataset_path: str,
    *,
    dataset_name: str = "mnist_rgb_224",
    image_size: int | None = None,
    batch_size: int = 128,
    device: str = "auto",
    test_preload: bool = True,
) -> dict[str, float]:
    """Evaluate a checkpoint (EMA model) on the pareidolia test dataset."""
    if device == "auto":
        dev, _ = _detect_device()
    else:
        dev = torch.device(device)

    model, ckpt = build_model_from_checkpoint(checkpoint_path, dev)
    num_classes = ckpt.get("model_config", {}).get("num_classes", 10)
    if image_size is None:
        image_size = ckpt.get("model_config", {}).get("image_size", 224)

    # Load mean/std from cached dataset (must match training normalization)
    npz_path = os.path.join("datasets", dataset_name + ".npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        mean = tuple(data["mean"]) if "mean" in data else (0.5, 0.5, 0.5)
        std = tuple(data["std"]) if "std" in data else (0.5, 0.5, 0.5)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        console.print("[yellow]No cached dataset found; using mean=0.5, std=0.5. Pass --dataset to match training.[/yellow]")

    from digit_classifier.pareidolia_dataset import PareidoliaTestDataset
    test_dataset = PareidoliaTestDataset(
        root_dir=test_dataset_path,
        color=True,
        size=image_size,
        mean=mean,
        std=std,
        preload=test_preload,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    metrics = _build_metrics(num_classes, dev)
    criterion = nn.CrossEntropyLoss()
    results = validate(model, test_loader, criterion, metrics, dev, use_amp=(dev.type == "cuda"))

    console.print(f"[bold]Eval on {test_dataset_path}[/bold] ({len(test_dataset)} samples)")
    for k, v in results.items():
        console.print(f"  {k}: {v:.5f}")
    return results


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
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    progress_bars: bool = False,
    show_data_wait: bool = False,
    epoch: int = 0,
    total_epochs: int = 1,
    prefetched: tuple[Iterator, tuple[Tensor, Tensor]] | None = None,
    on_last_batch_start: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Run one training epoch and return computed metrics + loss."""
    import time

    model.train()
    running_loss = 0.0
    num_batches = 0
    pin = device.type == "cuda"
    if use_amp:
        amp_ctx = autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else autocast(device_type=device.type)
    else:
        amp_ctx = nullcontext()

    total_batches = len(loader)
    columns: list = [
        TextColumn("[bold blue]Train[/] Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    if show_data_wait:
        columns.append(TextColumn("•"))
        columns.append(TextColumn("wait {task.fields[wait_pct]:.0f}%"))
    columns.append(TextColumn("• loss {task.fields[loss]:.4f}"))

    progress_ctx = (
        Progress(*columns, console=console)
        if progress_bars
        else nullcontext()
    )

    with progress_ctx as progress:
        if progress_bars:
            task = progress.add_task(
                "",
                total=total_batches,
                epoch=f"{epoch + 1}/{total_epochs}",
                loss=0.0,
                wait_pct=0.0,
            )

        if prefetched is not None:
            loader_iter, (images, labels) = prefetched
        else:
            loader_iter = iter(loader)
            images, labels = None, None
        data_time = 0.0
        compute_time = 0.0

        for batch_idx in range(total_batches):
            if batch_idx == total_batches - 1 and on_last_batch_start is not None:
                on_last_batch_start()
            if images is None:
                t0 = time.perf_counter()
                try:
                    images, labels = next(loader_iter)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                data_time += t1 - t0
            else:
                t0 = t1 = time.perf_counter()

            if device.type == "cuda":
                torch.cuda.synchronize()
            if images.dtype != torch.float32:
                images = images.float()
            images = images.to(device, non_blocking=pin)
            labels = labels.long().to(device, non_blocking=pin)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            optimizer.zero_grad()

            with amp_ctx:
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

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            compute_time += t2 - t1

            if progress is not None:
                total = data_time + compute_time
                wait_pct = (data_time / total * 100) if total > 0 else 0.0
                progress.update(task, advance=1, loss=running_loss, wait_pct=wait_pct)

            images, labels = None, None  # Consumed; fetch next on next iteration

    result = _compute_and_reset(metrics)
    result["cross_entropy"] = running_loss
    return result


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metrics: dict[str, Metric],
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    progress_bars: bool = False,
    show_data_wait: bool = False,
    progress_label: str = "Val",
) -> dict[str, float]:
    """Run one validation pass and return computed metrics + loss."""
    import time

    model.eval()
    running_loss = 0.0
    num_batches = 0
    pin = device.type == "cuda"
    if use_amp:
        amp_ctx = autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else autocast(device_type=device.type)
    else:
        amp_ctx = nullcontext()

    total_batches = len(loader)
    val_columns: list = [
        TextColumn(f"[bold cyan]{progress_label}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    if show_data_wait:
        val_columns.append(TextColumn("•"))
        val_columns.append(TextColumn("wait {task.fields[wait_pct]:.0f}%"))

    progress_ctx = (
        Progress(*val_columns, console=console)
        if progress_bars
        else nullcontext()
    )

    with torch.no_grad(), progress_ctx as progress:
        if progress_bars:
            task = progress.add_task("", total=total_batches, wait_pct=0.0)

        loader_iter = iter(loader)
        data_time = 0.0
        compute_time = 0.0

        for batch_idx in range(total_batches):
            t0 = time.perf_counter()
            try:
                images, labels = next(loader_iter)
            except StopIteration:
                break
            t1 = time.perf_counter()
            data_time += t1 - t0

            if device.type == "cuda":
                torch.cuda.synchronize()
            if images.dtype != torch.float32:
                images = images.float()
            images = images.to(device, non_blocking=pin)
            labels = labels.long().to(device, non_blocking=pin)

            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)

            num_batches += 1
            running_loss += (loss.item() - running_loss) / num_batches
            _update_metrics(metrics, logits, labels)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            compute_time += t2 - t1

            if progress is not None:
                total = data_time + compute_time
                wait_pct = (data_time / total * 100) if total > 0 else 0.0
                progress.update(task, advance=1, wait_pct=wait_pct)

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
    repeat_aug: bool = False,
    repeat_aug_repeats: int = 3,
    test_dataset: Dataset | None = None,
    *,
    num_workers: int = -1,
    prefetch_factor: int = 2,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    if num_workers < 0:
        num_workers = min(16, max(1, cpu_count() - 1))
    pin_memory = device.type == "cuda"
    persistent = num_workers > 0
    common: dict = dict(
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor

    ddp = world_size > 1
    ddp_kw = {"num_replicas": world_size, "rank": rank} if ddp else {}

    original_count = getattr(train_dataset, "num_original", None)
    # Use ratio sampler only when we have both primary (num_original > 0) and external data
    has_mixed = (
        original_count is not None
        and original_count > 0
        and original_count < len(train_dataset)
    )

    if repeat_aug and has_mixed:
        batch_sampler = RepeatAugRatioBatchSampler(
            original_count=int(original_count),
            total_count=len(train_dataset),
            batch_size=batch_size,
            primary_fraction=primary_fraction,
            num_repeats=repeat_aug_repeats,
            drop_last=True,
            **ddp_kw,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **common)
    elif repeat_aug:
        sampler = RepeatAugSampler(
            train_dataset,
            shuffle=True,
            num_repeats=repeat_aug_repeats,
            **ddp_kw,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            **common,
        )
    else:
        if has_mixed:
            sampler = RatioBatchSampler(
                original_count=int(original_count),
                total_count=len(train_dataset),
                batch_size=batch_size,
                primary_fraction=primary_fraction,
                drop_last=True,
                **ddp_kw,
            )
            train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
        elif ddp:
            dist_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                sampler=dist_sampler,
                batch_size=batch_size,
                drop_last=True,
                **common,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **common,
            )

    if ddp:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, **common)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = None
    if test_dataset is not None:
        if ddp:
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, **common)
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _log_epoch_table(
    epoch: int,
    train: dict[str, float],
    val_raw: dict[str, float],
    val_ema: dict[str, float],
    lr: float,
    test_ema: dict[str, float] | None = None,
    use_ema: bool = True,
) -> None:
    has_test = test_ema is not None
    table = Table(title=f"Epoch {epoch}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Train", justify="right")
    if use_ema:
        table.add_column("Val (raw)", justify="right")
        table.add_column("Val (EMA)", justify="right")
    else:
        table.add_column("Val", justify="right")
    if has_test:
        table.add_column("Test" + (" (EMA)" if use_ema else ""), justify="right")

    all_keys = dict.fromkeys(
        list(train) + list(val_raw) + list(val_ema) + (list(test_ema) if test_ema else [])
    )
    for key in all_keys:
        if use_ema:
            row = [
                key,
                f"{train.get(key, 0):.5f}",
                f"{val_raw.get(key, 0):.5f}",
                f"{val_ema.get(key, 0):.5f}",
            ]
        else:
            row = [key, f"{train.get(key, 0):.5f}", f"{val_ema.get(key, 0):.5f}"]
        if has_test:
            row.append(f"{test_ema.get(key, 0):.5f}")
        table.add_row(*row)
    lr_row = ["lr", f"{lr:.2e}", "", ""] if use_ema else ["lr", f"{lr:.2e}", ""]
    if has_test:
        lr_row.append("")
    table.add_row(*lr_row)
    console.print(table)


# ---------------------------------------------------------------------------
# Calibration and auto-tuning
# ---------------------------------------------------------------------------

def _calibrate_num_workers_on_dataset(
    train_dataset: Dataset,
    batch_size: int,
    prefetch_factor: int,
    num_batches: int = 50,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    """Benchmark different num_workers on the real training dataset and return the best."""
    import time

    show_progress = world_size <= 1 or rank == 0

    total_samples = len(train_dataset)
    actual_batches = min(num_batches, max(1, total_samples // batch_size))
    if actual_batches < 2:
        if show_progress:
            console.print(
                "[yellow]Worker calibration skipped: dataset too small "
                f"({total_samples} samples, batch_size {batch_size})[/yellow]"
            )
        return min(16, max(1, cpu_count() - 1))

    candidates = [0, 4, 8, 16, 24, 32]
    max_workers = max(1, cpu_count() - 1)
    candidates = [n for n in candidates if n <= max_workers]
    if max_workers > 32 and max_workers not in candidates:
        candidates.append(min(max_workers, 64))

    best_nw = 0
    best_throughput = 0.0
    results: list[tuple[int, float]] = []

    if show_progress:
        console.print("[bold]Worker calibration (on real data):[/bold] starting…")
    progress_columns = (
        TextColumn("[bold blue]Worker calibration[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
    if show_progress:
        with Progress(*progress_columns, console=console) as progress:
            task = progress.add_task("calibrating", total=len(candidates))
            for nw in candidates:
                progress.update(task, description=f"Testing {nw} workers")
                loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=nw,
                    prefetch_factor=prefetch_factor if nw > 0 else None,
                    drop_last=True,
                    pin_memory=False,
                )
                start = time.perf_counter()
                count = 0
                for _ in loader:
                    count += 1
                    if count >= actual_batches:
                        break
                elapsed = time.perf_counter() - start
                throughput = count * batch_size / elapsed if elapsed > 0 else 0
                results.append((nw, throughput))
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_nw = nw
                progress.advance(task)
    else:
        for nw in candidates:
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=nw,
                prefetch_factor=prefetch_factor if nw > 0 else None,
                drop_last=True,
                pin_memory=False,
            )
            start = time.perf_counter()
            count = 0
            for _ in loader:
                count += 1
                if count >= actual_batches:
                    break
            elapsed = time.perf_counter() - start
            throughput = count * batch_size / elapsed if elapsed > 0 else 0
            results.append((nw, throughput))
            if throughput > best_throughput:
                best_throughput = throughput
                best_nw = nw

    if show_progress:
        console.print("[bold]Worker calibration (on real data):[/bold]")
        for nw, tp in results:
            mark = " ← best" if nw == best_nw else ""
            console.print(f"  [dim]{nw} workers:[/dim] {tp:.0f} samples/s{mark}")
    return best_nw


def _profile_compile_modes(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_warmup: int = 5,
    num_timed: int = 15,
) -> tuple[str, dict[str, float]]:
    """Profile torch.compile modes and return the one with highest throughput."""
    import time

    try:
        import torch.distributed as dist
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        is_rank0 = True

    modes = ["default", "reduce-overhead", "max-autotune"]
    batch_size = train_loader.batch_size if hasattr(train_loader, "batch_size") else getattr(
        train_loader.batch_sampler, "batch_size", 128
    )
    criterion = nn.CrossEntropyLoss()
    pin = device.type == "cuda"
    use_amp = device.type == "cuda"
    amp_ctx = autocast(device_type=device.type) if use_amp else nullcontext()
    scaler = GradScaler(enabled=use_amp)

    best_mode = "default"
    best_throughput = 0.0
    results: list[tuple[str, float]] = []

    if is_rank0:
        console.print("[bold]Compile mode profiling:[/bold] starting…")
    def _profile_one_mode(mode: str) -> float:
        compiled = torch.compile(model, mode=mode)
        compiled.train()
        opt = torch.optim.AdamW(compiled.parameters(), lr=1e-4)
        loader_iter = iter(train_loader)
        for _ in range(num_warmup):
            try:
                images, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                images, labels = next(loader_iter)
            if images.dtype != torch.float32:
                images = images.float()
            images = images.to(device, non_blocking=pin)
            labels = labels.long().to(device, non_blocking=pin)
            opt.zero_grad()
            with amp_ctx:
                logits = compiled(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        loader_iter = iter(train_loader)
        start = time.perf_counter()
        count = 0
        for _ in range(num_timed):
            try:
                images, labels = next(loader_iter)
            except StopIteration:
                break
            if images.dtype != torch.float32:
                images = images.float()
            images = images.to(device, non_blocking=pin)
            labels = labels.long().to(device, non_blocking=pin)
            opt.zero_grad()
            with amp_ctx:
                logits = compiled(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            count += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return count * batch_size / elapsed if elapsed > 0 else 0.0

    progress_columns = (
        TextColumn("[bold blue]Compile mode profiling[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
    if is_rank0:
        with Progress(*progress_columns, console=console) as progress:
            task = progress.add_task("profiling", total=len(modes))
            for mode in modes:
                progress.update(task, description=f"Profiling {mode}")
                throughput = _profile_one_mode(mode)
                results.append((mode, throughput))
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_mode = mode
                progress.advance(task)
    else:
        for mode in modes:
            throughput = _profile_one_mode(mode)
            results.append((mode, throughput))
            if throughput > best_throughput:
                best_throughput = throughput
                best_mode = mode

    if is_rank0:
        console.print("[bold]Compile mode profiling:[/bold]")
        for mode, tp in results:
            mark = " ← best" if mode == best_mode else ""
            console.print(f"  [dim]{mode}:[/dim] {tp:.0f} samples/s{mark}")
    results_dict = {f"compile_{mode}_samples_per_s": tp for mode, tp in results}
    results_dict["compile_best_mode"] = best_mode
    return best_mode, results_dict


def _wrap_external_with_cache(
    train_dataset: Dataset,
    external_cache_max_mb: float,
    num_workers: int,
) -> None:
    """Wrap ExternalOnDemandDataset entries in train_dataset with CachedExternalDataset (mutates)."""
    if not isinstance(train_dataset, ConcatDataset) or external_cache_max_mb <= 0:
        return
    max_bytes = int((external_cache_max_mb * 1024 * 1024) / max(1, num_workers))
    for i, ds in enumerate(train_dataset.datasets):
        if isinstance(ds, ExternalOnDemandDataset):
            train_dataset.datasets[i] = CachedExternalDataset(ds, max_bytes)


def _resolve_external_cache_max_mb(value: float) -> float:
    """Resolve external_cache_max_mb; -1 means auto from available RAM."""
    if value >= 0:
        return value
    try:
        import psutil
        available_bytes = psutil.virtual_memory().available
        # 15% of available RAM, cap at 4 GB
        auto_mb = min(4096, int(available_bytes * 0.15 / (1024 * 1024)))
        return max(256, auto_mb)  # at least 256 MB
    except ImportError:
        return 2048


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
    tc = cfg.training
    if tc.resume_path and tc.pretrain_path:
        raise ValueError("Cannot use both --resume and --pretrain; choose one.")

    rank, world_size, local_rank = _setup_ddp()
    device, device_name = _detect_device()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        device_name = f"cuda:{local_rank} (rank {rank}/{world_size})"
    if rank == 0:
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
    external_cache_max_mb = _resolve_external_cache_max_mb(cfg.data.external_cache_max_mb)
    if cfg.data.external_cache and cfg.data.external_cache_max_mb < 0 and rank == 0:
        console.print(f"External cache (auto): [cyan]{external_cache_max_mb:.0f} MB[/cyan]")

    # When calibrating, build dataset with cache off first so we can benchmark on real data
    use_cache_for_build = cfg.data.external_cache and not cfg.data.calibrate_workers
    if use_cache_for_build:
        num_workers = cfg.data.num_workers
        if num_workers < 0:
            num_workers = min(16, max(1, cpu_count() - 1))
    else:
        num_workers = 1  # placeholder when cache off (calibrating or no cache)

    if cfg.data.external_only:
        if not cfg.data.mix_external:
            raise ValueError("external_only requires mix_external=True")
        train_dataset, val_dataset, mean, std = split_dataset_external_only(
            external_val_source=cfg.data.external_val_source,
            external_val_split=cfg.data.external_val_split,
            external_val_fraction=cfg.data.external_val_fraction,
            color=cfg.data.color,
            size=cfg.data.image_size,
            seed=cfg.data.split_seed,
            external_cache=use_cache_for_build,
            external_cache_max_mb=external_cache_max_mb if use_cache_for_build else 0,
            num_workers=num_workers,
        )
        if rank == 0:
            console.print(f"External-only: {len(train_dataset):,} train / {len(val_dataset):,} val")
    else:
        images, labels, cached_mean, cached_std = load_cached_dataset(cfg)
        if rank == 0:
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
            augment_scheme=cfg.data.augment_scheme,
            external_cache=use_cache_for_build,
            external_cache_max_mb=external_cache_max_mb if use_cache_for_build else 0,
            num_workers=num_workers,
        )
    train_val_msg = f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples"

    if cfg.data.calibrate_workers:
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()  # Sync so rank 1 doesn't interleave output with rank 0's progress bar
        num_workers = _calibrate_num_workers_on_dataset(
            train_dataset,
            batch_size=cfg.data.batch_size,
            prefetch_factor=cfg.data.prefetch_factor,
            rank=rank,
            world_size=world_size,
        )
        if cfg.data.external_cache and external_cache_max_mb > 0:
            _wrap_external_with_cache(train_dataset, external_cache_max_mb, num_workers)
            if rank == 0:
                console.print(
                    f"  [dim]External cache: {external_cache_max_mb:.0f} MB total "
                f"({external_cache_max_mb / max(1, num_workers):.0f} MB per worker × {num_workers})[/dim]"
            )
    else:
        num_workers = cfg.data.num_workers
        if num_workers < 0:
            num_workers = min(16, max(1, cpu_count() - 1))
    if rank == 0:
        console.print(f"DataLoader workers: [cyan]{num_workers}[/cyan]")

    test_dataset = None
    if cfg.data.test_dataset_path:
        from digit_classifier.pareidolia_dataset import PareidoliaTestDataset
        test_dataset = PareidoliaTestDataset(
            root_dir=cfg.data.test_dataset_path,
            color=cfg.data.color,
            size=cfg.data.image_size,
            mean=mean,
            std=std,
            preload=cfg.data.test_preload,
        )
        if rank == 0:
            console.print(
                f"Test (pareidolia): {len(test_dataset)} samples (no augmentation)"
                + (f", skipped {test_dataset.skipped} missing" if test_dataset.skipped else "")
            )
        train_val_msg += f", Test: {len(test_dataset)} samples"
    if rank == 0:
        console.print(train_val_msg)

    train_loader, val_loader, test_loader = _create_dataloaders(
        train_dataset, val_dataset, cfg.data.batch_size,
        cfg.data.primary_fraction, device,
        repeat_aug=cfg.data.repeat_aug,
        repeat_aug_repeats=cfg.data.repeat_aug_repeats,
        test_dataset=test_dataset,
        num_workers=num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        rank=rank,
        world_size=world_size,
    )
    if world_size > 1:
        console.print(f"[dim]Rank {rank}/{world_size}: dataloaders ready[/dim]")

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
    mc = cfg.model
    use_flash_attention = mc.use_flash_attention
    if use_flash_attention is None:
        if mc.model_type == "deit" and device.type == "cuda":
            cuda_be = getattr(torch.backends, "cuda", None)
            if cuda_be is not None and hasattr(cuda_be, "enable_flash_sdp"):
                try:
                    cuda_be.enable_flash_sdp(True)
                    use_flash_attention = True
                except Exception:
                    use_flash_attention = False
            else:
                use_flash_attention = False
        else:
            use_flash_attention = False
    elif use_flash_attention and device.type == "cuda":
        cuda_be = getattr(torch.backends, "cuda", None)
        if cuda_be is not None and hasattr(cuda_be, "enable_flash_sdp"):
            try:
                cuda_be.enable_flash_sdp(True)
            except Exception:
                pass
    pretrain_patch_size: int | None = None
    if tc.pretrain_path and mc.model_type == "deit":
        pretrain_ckpt = torch.load(tc.pretrain_path, map_location=device, weights_only=False)
        pc = pretrain_ckpt.get("model_config", {})
        pretrain_patch_size = pc.get("patch_size")
        if pretrain_patch_size is None:
            pretrain_patch_size = 14 if pc.get("deit_model", "base") == "huge" else 16

    if mc.model_type == "resnext":
        model_name = "ResNeXt"
        model = ResNeXt(
            layers=list(mc.layers),
            num_classes=mc.num_classes,
            groups=mc.groups,
            width_per_group=mc.width_per_group,
            drop_path_rate=mc.drop_path_rate,
        ).to(device)
    else:
        model_name = f"deit3_{mc.deit_model}_patch16_224"
        model = build_deit3(
            size=mc.deit_model,
            num_classes=mc.num_classes,
            drop_path_rate=mc.drop_path_rate,
            image_size=cfg.data.image_size,
            use_flash_attention=use_flash_attention,
            init_values=mc.layer_scale_init,
            patch_size=pretrain_patch_size,
        ).to(device)
    if world_size > 1:
        console.print(f"[dim]Rank {rank}/{world_size}: model built[/dim]")
    if rank == 0 and mc.model_type == "deit":
        deit = _get_deit_model(model)
        if deit is not None:
            console.print(f"[dim]DeiT image_size={cfg.data.image_size}, patch_size={deit.config.patch_size}[/dim]")

    start_epoch = 0
    resume_ckpt: dict | None = None
    if tc.pretrain_path:
        resume_ckpt = torch.load(tc.pretrain_path, map_location=device, weights_only=False)
        state = resume_ckpt.get("ema_state_dict", resume_ckpt.get("model_state_dict"))
        if state is None:
            raise KeyError(f"Checkpoint {tc.pretrain_path} must contain 'ema_state_dict' or 'model_state_dict'")
        stripped = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
        ckpt_config = resume_ckpt.get("model_config", {})
        ckpt_image_size = ckpt_config.get("image_size", 224)
        if mc.model_type == "deit" and ckpt_image_size != cfg.data.image_size and "pos_embed" in stripped:
            from digit_classifier.vit import resize_pos_embed
            ckpt_patch_size = ckpt_config.get("patch_size")
            if ckpt_patch_size is None:
                ckpt_patch_size = 14 if ckpt_config.get("deit_model", "base") == "huge" else 16
            stripped["pos_embed"] = resize_pos_embed(
                stripped["pos_embed"],
                orig_size=ckpt_image_size,
                new_size=cfg.data.image_size,
                patch_size=ckpt_patch_size,
            )
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in stripped.items() if k in model_keys}
        model.load_state_dict(filtered, strict=True)
        if rank == 0:
            console.print(f"[bold]Loaded pretrained weights from[/bold] {tc.pretrain_path} (image_size {ckpt_image_size} -> {cfg.data.image_size})")
    elif tc.resume_path:
        resume_ckpt = torch.load(tc.resume_path, map_location=device, weights_only=False)
        ckpt_config = resume_ckpt.get("model_config", {})
        ckpt_image_size = ckpt_config.get("image_size", 224)
        if ckpt_image_size != cfg.data.image_size:
            raise ValueError(
                f"Resume requires same resolution: checkpoint has image_size={ckpt_image_size}, "
                f"config has {cfg.data.image_size}. Use --pretrain for fine-tuning at different resolution."
            )
        state = resume_ckpt.get("ema_state_dict", resume_ckpt.get("model_state_dict"))
        if state is None:
            raise KeyError(f"Checkpoint {tc.resume_path} must contain 'ema_state_dict' or 'model_state_dict'")
        stripped = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in stripped.items() if k in model_keys}
        model.load_state_dict(filtered, strict=True)
        start_epoch = resume_ckpt.get("epoch", 0)
        if rank == 0:
            console.print(f"[bold]Resumed from[/bold] {tc.resume_path} (epoch {start_epoch})")

    if world_size > 1:
        import torch.distributed as dist
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            torch.cuda.synchronize()
        dist.barrier()
        console.print(f"[dim]Rank {rank}/{world_size}: barrier passed, wrapping DDP…[/dim]")
        sys.stdout.flush()
        model = DDP(model, device_ids=[local_rank])
        console.print(f"[dim]Rank {rank}/{world_size}: DDP wrapped[/dim]")

    param_groups = _get_param_groups(
        model.module if hasattr(model, "module") else model,
        lr=tc.lr * world_size,
        weight_decay=tc.weight_decay,
        weight_decay_exclude=tc.weight_decay_exclude,
        layer_decay=tc.layer_decay,
    )

    compile_profiling: dict[str, float] | None = None
    if tc.compile_model:
        if tc.compile_mode is not None:
            model = torch.compile(model, mode=tc.compile_mode)
        elif device.type == "cuda" and world_size == 1:
            best_mode, compile_profiling = _profile_compile_modes(
                model, train_loader, device, num_warmup=5, num_timed=15
            )
            model = torch.compile(model, mode=best_mode)
            if rank == 0:
                console.print(f"[dim]Using compile mode: {best_mode}[/dim]")
        else:
            model = torch.compile(model)
    if world_size > 1 and tc.compile_model:
        console.print(f"[dim]Rank {rank}/{world_size}: torch.compile done[/dim]")

    if tc.ema_enabled:
        ema_base = model.module if hasattr(model, "module") else model
        ema = AveragedModel(ema_base, multi_avg_fn=get_ema_multi_avg_fn(tc.ema_decay), use_buffers=True)
        if resume_ckpt is not None and "ema_state_dict" in resume_ckpt:
            try:
                ema.load_state_dict(resume_ckpt["ema_state_dict"])
                if rank == 0:
                    console.print("[dim]Restored EMA state[/dim]")
            except Exception as e:
                if rank == 0:
                    console.print(f"[yellow]Could not restore EMA: {e}[/yellow]")
    else:
        ema = None

    # --- Optimiser & scheduler ---
    optimizer = torch.optim.AdamW(param_groups)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=tc.warmup_epochs)

    if tc.warm_restarts:
        main_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=tc.scheduler_t0, T_mult=tc.scheduler_t_mult, eta_min=tc.eta_min,
        )
        warm_restart_epochs = compute_warm_restart_epochs(
            tc.warmup_epochs, tc.scheduler_t0, tc.scheduler_t_mult, tc.epochs,
        )
    else:
        # simple cosine schedule for the remainder of training (single cycle)
        cycles = max(1, tc.epochs - tc.warmup_epochs)
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cycles, eta_min=tc.eta_min,
        )
        warm_restart_epochs = []

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[tc.warmup_epochs],
    )
    if world_size > 1:
        console.print(f"[dim]Rank {rank}/{world_size}: optimizer/scheduler ready[/dim]")
    if warm_restart_epochs and rank == 0:
        console.print(f"[bold]Warm-restart epochs:[/bold] {warm_restart_epochs}")

    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                if rank == 0:
                    console.print("[dim]Restored optimizer state[/dim]")
            except Exception as e:
                if rank == 0:
                    console.print(f"[yellow]Could not restore optimizer: {e}[/yellow]")
        if "scheduler_state_dict" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
                if rank == 0:
                    console.print("[dim]Restored scheduler state[/dim]")
            except Exception as e:
                if rank == 0:
                    console.print(f"[yellow]Could not restore scheduler: {e}[/yellow]")

    # --- AMP dtype ---
    amp_dtype: torch.dtype | None = None
    if device.type == "cuda" and tc.amp_enabled:
        if tc.amp_dtype == "bfloat16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                console.print("[yellow]bfloat16 not supported on this GPU; falling back to float16[/yellow]")
                amp_dtype = torch.float16
        elif tc.amp_dtype == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float16

    # --- Metrics ---
    metrics = _build_metrics(cfg.model.num_classes, device)
    if tc.bce_loss:
        val_criterion = BCELossWithSmoothing(
            num_classes=cfg.model.num_classes,
            smoothing=tc.label_smoothing,
        )
    else:
        val_criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda" and tc.amp_enabled and amp_dtype != torch.bfloat16))

    # --- Wandb ---
    # Resolve the effective batch size for logging (batch_sampler → None for .batch_size)
    effective_batch_size = cfg.data.batch_size

    wandb_run_id: str | None = None
    if tc.resume_path:
        resume_dir = os.path.basename(os.path.dirname(os.path.abspath(tc.resume_path)))
        if resume_dir != "local":
            wandb_run_id = resume_dir

    if tc.wandb_enabled and rank == 0:
        wandb_config = _config_to_wandb_dict(cfg)
        # Runtime / resolved values (override or add to config)
        wandb_config.update({
            "model": model_name,
            "optimizer": optimizer.__class__.__name__,
            "batch_size": effective_batch_size,
            "device": device_name,
            "num_workers": num_workers,
            "world_size": world_size,
        })
        if mc.model_type == "deit":
            deit = _get_deit_model(model)
            if deit is not None:
                wandb_config["patch_size"] = deit.config.patch_size
        wandb_config["mean"] = mean
        wandb_config["std"] = std
        if compile_profiling is not None:
            wandb_config.update(compile_profiling)
        if wandb_run_id is not None:
            wandb.init(project=tc.wandb_project, id=wandb_run_id, resume="must", config=wandb_config)
            checkpoint_dir = os.path.join("checkpoints", wandb_run_id)
        else:
            wandb.init(project=tc.wandb_project, config=wandb_config)
            if tc.checkpoint_enabled:
                checkpoint_dir = os.path.join("checkpoints", wandb.run.id)
                os.makedirs(checkpoint_dir, exist_ok=True)
            else:
                checkpoint_dir = "checkpoints/local"
        if tc.wandb_watch and tc.wandb_watch != "none":
            wandb.watch(model, log=tc.wandb_watch, log_freq=100)
    else:
        if tc.checkpoint_enabled and rank == 0:
            checkpoint_dir = os.path.join("checkpoints", wandb_run_id) if wandb_run_id else "checkpoints/local"
            os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    best_val_accuracy = resume_ckpt.get("val_accuracy", 0.0) if resume_ckpt else 0.0
    best_test_accuracy = resume_ckpt.get("test_accuracy", 0.0) if resume_ckpt else 0.0
    pending_async: list[threading.Thread] = []  # wandb.log and checkpoint saves run async

    if world_size > 1:
        console.print(f"[dim]Rank {rank}/{world_size} reached epoch loop[/dim]")
    elif rank == 0:
        console.print("[dim]Starting epoch loop (first batch may take a few minutes with large datasets)…[/dim]")

    prefetched: tuple[Iterator, tuple[Tensor, Tensor]] | None = None
    for epoch in range(start_epoch, tc.epochs):
        # Repeated augmentation: set epoch for reproducible shuffle
        loader_sampler = getattr(train_loader, "sampler", None) or getattr(
            train_loader, "batch_sampler", None
        )
        if loader_sampler is not None and hasattr(loader_sampler, "set_epoch"):
            loader_sampler.set_epoch(epoch)

        # Scheduled stochastic depth increase
        deit = _get_deit_model(model)
        if deit is not None and tc.drop_path_increment > 0 and tc.drop_path_increment_every > 0:
            step = epoch // tc.drop_path_increment_every
            effective_max = cfg.model.drop_path_rate + step * tc.drop_path_increment
            deit.set_drop_path_rate(effective_max)

        # Disable mixup for the final N epochs.
        active_mixup = mixup if epoch < tc.epochs - tc.mixup_off_last_n else None
        if epoch == tc.epochs - tc.mixup_off_last_n and rank == 0:
            console.print(f"[yellow]Disabling mixup for final {tc.mixup_off_last_n} epochs[/yellow]")

        train_criterion = select_train_criterion(
            active_mixup,
            num_classes=cfg.model.num_classes,
            bce_loss=tc.bce_loss,
            label_smoothing=tc.label_smoothing,
        )

        # Prefetch next epoch's first batch during last batch's compute (overlaps regardless of validation)
        prefetch_out: dict = {}
        if epoch + 1 < tc.epochs and loader_sampler is not None and hasattr(loader_sampler, "set_epoch"):

            def _on_last_batch_start() -> None:
                def _prefetch() -> None:
                    loader_sampler.set_epoch(epoch + 1)
                    it = iter(train_loader)
                    try:
                        batch = next(it)
                        prefetch_out["result"] = (it, batch)
                    except StopIteration:
                        pass

                t = threading.Thread(target=_prefetch)
                t.start()
                prefetch_out["thread"] = t

            on_last_batch = _on_last_batch_start
        else:
            on_last_batch = None

        train_metrics = train_epoch(
            model, train_loader, train_criterion, optimizer, metrics, scaler,
            device, mixup_fn=active_mixup, ema=ema, grad_clip_norm=tc.grad_clip_norm,
            use_amp=tc.amp_enabled,
            amp_dtype=amp_dtype,
            progress_bars=(tc.progress_bars or tc.show_data_wait) and rank == 0,
            show_data_wait=tc.show_data_wait and rank == 0,
            epoch=epoch,
            total_epochs=tc.epochs,
            prefetched=prefetched,
            on_last_batch_start=on_last_batch,
        )
        train_metrics = _all_reduce_metrics(train_metrics, device, world_size)

        if "thread" in prefetch_out:
            prefetch_out["thread"].join()
            prefetched = prefetch_out.get("result")
        else:
            prefetched = None

        do_validate = epoch == 0 or (epoch + 1) % tc.val_every_n_epochs == 0 or epoch == tc.epochs - 1
        if do_validate:
            val_metrics_raw = validate(
                model, val_loader, val_criterion, metrics, device,
                use_amp=tc.amp_enabled,
                amp_dtype=amp_dtype,
                progress_bars=(tc.progress_bars or tc.show_data_wait) and rank == 0,
                show_data_wait=tc.show_data_wait and rank == 0,
            )
            val_metrics_raw = _all_reduce_metrics(val_metrics_raw, device, world_size)
            if ema is not None:
                val_metrics_ema = validate(
                    ema, val_loader, val_criterion, metrics, device,
                    use_amp=tc.amp_enabled,
                    amp_dtype=amp_dtype,
                    progress_bars=(tc.progress_bars or tc.show_data_wait) and rank == 0,
                    show_data_wait=tc.show_data_wait and rank == 0,
                )
                val_metrics_ema = _all_reduce_metrics(val_metrics_ema, device, world_size)
            else:
                val_metrics_ema = dict(val_metrics_raw)
        else:
            val_metrics_raw = {}
            val_metrics_ema = {}

        test_metrics_ema: dict[str, float] | None = None
        if test_loader is not None and do_validate:
            eval_model = ema if ema is not None else model
            test_metrics_ema = validate(
                eval_model, test_loader, val_criterion, metrics, device,
                use_amp=tc.amp_enabled,
                amp_dtype=amp_dtype,
                progress_bars=(tc.progress_bars or tc.show_data_wait) and rank == 0,
                show_data_wait=tc.show_data_wait and rank == 0,
                progress_label="Test",
            )
            test_metrics_ema = _all_reduce_metrics(test_metrics_ema, device, world_size)

        # --- Pre-restart checkpoint (before scheduler.step) ---
        if tc.checkpoint_enabled and rank == 0 and warm_restart_epochs and (epoch + 1) in warm_restart_epochs:
            ckpt_root = checkpoint_dir if "checkpoint_dir" in dir() else "checkpoints"
            os.makedirs(ckpt_root, exist_ok=True)
            pre_path = os.path.join(ckpt_root, f"pre_restart_epoch_{epoch + 1}.pt")
            model_state = (model.module if hasattr(model, "module") else model).state_dict()
            save_dict: dict[str, object] = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_accuracy": val_metrics_ema.get("accuracy"),
                "model_type": mc.model_type,
                "model_config": _get_model_config_for_checkpoint(mc, cfg.data.image_size),
            }
            if ema is not None:
                save_dict["ema_state_dict"] = ema.state_dict()

            def _save_pre_restart():
                torch.save(save_dict, pre_path)
                if tc.wandb_enabled:
                    try:
                        art = wandb.Artifact(f"pre-restart-epoch-{epoch + 1}", type="model",
                                             metadata={"epoch": epoch + 1, "val_accuracy": val_metrics_ema.get("accuracy")})
                        art.add_file(pre_path)
                        wandb.log_artifact(art)
                    except Exception:
                        pass

            t = threading.Thread(target=_save_pre_restart)
            t.start()
            pending_async.append(t)
            console.print(f"[magenta]Saved pre-restart checkpoint:[/magenta] {pre_path}")

        scheduler.step()

        # --- Logging ---
        current_lr = scheduler.get_last_lr()[0]
        if rank == 0:
            _log_epoch_table(
                epoch + 1, train_metrics, val_metrics_raw, val_metrics_ema, current_lr,
                test_ema=test_metrics_ema,
                use_ema=ema is not None,
            )

        if tc.wandb_enabled and rank == 0:
            log_dict = {
                "epoch": epoch + 1,
                "lr": current_lr,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            if ema is not None:
                log_dict.update({f"val_raw/{k}": v for k, v in val_metrics_raw.items()})
                log_dict.update({f"val_ema/{k}": v for k, v in val_metrics_ema.items()})
            else:
                log_dict.update({f"val/{k}": v for k, v in val_metrics_ema.items()})
            if test_metrics_ema is not None:
                prefix = "test_ema" if ema is not None else "test"
                log_dict.update({f"{prefix}/{k}": v for k, v in test_metrics_ema.items()})
            def _log():
                wandb.log(log_dict)
            t = threading.Thread(target=_log)
            t.start()
            pending_async.append(t)

        # --- Best-model checkpoint (only when we ran validation) ---
        if do_validate:
            val_accuracy = val_metrics_ema["accuracy"]
        else:
            val_accuracy = 0.0
        if do_validate and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if tc.checkpoint_enabled and rank == 0:
                ckpt_path = os.path.join(checkpoint_dir, "best.pt")
                model_state = (model.module if hasattr(model, "module") else model).state_dict()
                save_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_accuracy": val_accuracy,
                    "model_type": mc.model_type,
                    "model_config": _get_model_config_for_checkpoint(mc, cfg.data.image_size),
                }
                if ema is not None:
                    save_dict["ema_state_dict"] = ema.state_dict()

                def _save_best():
                    torch.save(save_dict, ckpt_path)
                    if tc.wandb_enabled:
                        art_name = f"model-best-{wandb.run.id}"
                        if tc.replace_best_checkpoint:
                            try:
                                api = wandb.Api()
                                prev = api.artifact(
                                    f"{wandb.run.entity}/{wandb.run.project}/{art_name}:best",
                                    type="model",
                                )
                                prev.delete(delete_aliases=True)
                            except Exception:
                                pass  # no previous artifact or not found
                        art = wandb.Artifact(art_name, type="model",
                                             metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy})
                        art.add_file(ckpt_path)
                        wandb.log_artifact(art, aliases=["best"])

                t = threading.Thread(target=_save_best)
                t.start()
                pending_async.append(t)
                console.print(f"[green]Saved best model (val_accuracy={val_accuracy:.4f}) at epoch {epoch + 1}[/green]")

        # --- Best-test checkpoint (when we ran test and test improved) ---
        test_accuracy = test_metrics_ema["accuracy"] if test_metrics_ema is not None else 0.0
        if do_validate and test_metrics_ema is not None and test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            if tc.checkpoint_enabled and rank == 0:
                ckpt_path_test = os.path.join(checkpoint_dir, "best_test.pt")
                model_state_test = (model.module if hasattr(model, "module") else model).state_dict()
                val_acc_for_test = val_metrics_ema["accuracy"]
                save_dict_test = {
                    "epoch": epoch + 1,
                    "model_state_dict": model_state_test,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_accuracy": val_acc_for_test,
                    "test_accuracy": test_accuracy,
                    "model_type": mc.model_type,
                    "model_config": _get_model_config_for_checkpoint(mc, cfg.data.image_size),
                }
                if ema is not None:
                    save_dict_test["ema_state_dict"] = ema.state_dict()

                def _save_best_test():
                    torch.save(save_dict_test, ckpt_path_test)
                    if tc.wandb_enabled:
                        art_name_test = f"model-best-test-{wandb.run.id}"
                        if tc.replace_best_checkpoint:
                            try:
                                api = wandb.Api()
                                prev = api.artifact(
                                    f"{wandb.run.entity}/{wandb.run.project}/{art_name_test}:best-test",
                                    type="model",
                                )
                                prev.delete(delete_aliases=True)
                            except Exception:
                                pass
                        art_test = wandb.Artifact(art_name_test, type="model",
                                                  metadata={"epoch": epoch + 1, "val_accuracy": val_acc_for_test, "test_accuracy": test_accuracy})
                        art_test.add_file(ckpt_path_test)
                        wandb.log_artifact(art_test, aliases=["best-test"])

                t = threading.Thread(target=_save_best_test)
                t.start()
                pending_async.append(t)
                console.print(f"[green]Saved best test model (test_accuracy={test_accuracy:.4f}) at epoch {epoch + 1}[/green]")

        # --- Latest checkpoint (every epoch, always overwrite) ---
        if tc.checkpoint_latest and tc.checkpoint_enabled and rank == 0:
            ckpt_path_latest = os.path.join(checkpoint_dir, "latest.pt")
            model_state_latest = (model.module if hasattr(model, "module") else model).state_dict()
            val_acc_latest = val_metrics_ema.get("accuracy") if do_validate else None
            save_dict_latest = {
                "epoch": epoch + 1,
                "model_state_dict": model_state_latest,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_accuracy": val_acc_latest,
                "model_type": mc.model_type,
                "model_config": _get_model_config_for_checkpoint(mc, cfg.data.image_size),
            }
            if ema is not None:
                save_dict_latest["ema_state_dict"] = ema.state_dict()

            def _save_latest():
                torch.save(save_dict_latest, ckpt_path_latest)

            t = threading.Thread(target=_save_latest)
            t.start()
            pending_async.append(t)

    if rank == 0:
        for t in pending_async:
            t.join()
    if tc.wandb_enabled and rank == 0:
        wandb.finish()
    if rank == 0:
        console.print("[bold green]Training complete.[/bold green]")
