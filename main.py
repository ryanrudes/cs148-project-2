from multiprocessing import freeze_support, cpu_count
from rich.pretty import pprint

from torchmetrics import Metric, Accuracy, F1Score
from timm.loss import SoftTargetCrossEntropy

import torchvision.transforms.v2 as T
import torch.nn as nn
import numpy as np
import torch
import wandb
import os

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch import Tensor

from external_datasets import ExternalDataset, load_external_dataset
from mixup import create_mixup_cutmix, MixupCutmixApply
from augmentations import get_yolo_augmentor
from preprocess_data import compute_mean_std
from dataset import DummyMNIST
from resnext import ResNeXt


DATASET_NAME = "mnist_rgb_224"
SIZE = 224
COLOR = True
WANDB = True

# External dataset mixing toggle
MIX_EXTERNAL = True  # Set to True to mix in samples from another dataset (e.g., SVHN)
EXTERNAL_DATASET = "SVHN"  # Currently supported: "SVHN"

# The fractions in this dictionary are relative to the size of the original training set
# So we would add in fraction * len(original_training_set) examples from each external dataset
EXTERNAL_DATA_FRACTIONS = {
    ExternalDataset.SVHN_TRAIN: 0.2,
    ExternalDataset.SVHN_TEST: 0.2,
    ExternalDataset.SVHN_EXTRA: 0.2,
}


cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
device_name = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
DEVICE = torch.device(device_name)
print("Using device:", DEVICE)


def load_data():
    dataset_path = os.path.join("datasets", DATASET_NAME + ".npz")
    data = np.load(dataset_path)
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"]).long()

    # If mean/std exist in the preprocessed dataset, return them so we can normalize consistently
    mean = tuple(data["mean"]) if "mean" in data else None
    std = tuple(data["std"]) if "std" in data else None

    return images, labels, mean, std


def split_dataset(images, labels, mean=None, std=None, train_fraction: float = 0.9):
    num_samples = len(images)
    num_train = int(train_fraction * num_samples)

    # Deterministic index split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_samples, generator=generator)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    val_images = images[val_idx]
    val_labels = labels[val_idx]

    num_original_train = len(train_images)

    if MIX_EXTERNAL:
        for ext_dataset, ext_fraction in EXTERNAL_DATA_FRACTIONS.items():
            print("Loading external dataset:", ext_dataset.value)

            num_to_add = int(ext_fraction * num_original_train)

            ext_images, ext_labels = load_external_dataset(ext_dataset, COLOR, SIZE, max_samples=num_to_add, rnd=generator)
            if num_to_add > 0:
                sample_idxs = torch.randperm(len(ext_images), generator=generator)[:num_to_add]
            chosen_images = ext_images[sample_idxs]
            chosen_labels = ext_labels[sample_idxs]

            # Concatenate into training set (augmentor will normalize per-sample during __getitem__)
            train_images = torch.cat([train_images, chosen_images], dim=0)
            train_labels = torch.cat([train_labels, chosen_labels], dim=0)

            print(f"Mixed in {num_to_add} samples from {ext_dataset.value} into training set (new train size {len(train_images)})")

    # Recompute mean and std after mixing in external datasets
    mean, std = compute_mean_std(train_images, color=COLOR)
    print("Computed mean/std after mixing external datasets:", mean, std)

    # Build normalization for validation using dataset mean/std (fallback to 0.5 if not available)
    #num_channels = train_images.shape[1]
    #if mean is None or std is None:
    #    default_mean = tuple([0.5] * num_channels)
    #    default_std = tuple([0.5] * num_channels)
    #else:
    #    default_mean = tuple(mean)
    #    default_std = tuple(std)

    val_norm = T.Normalize(mean=mean, std=std)

    train_dataset = DummyMNIST(
        images=train_images,
        labels=train_labels,
        transform=get_yolo_augmentor(mean=mean, std=std),
    )

    val_dataset = DummyMNIST(
        images=val_images,
        labels=val_labels,
        transform=lambda img, label: val_norm(img),
    )

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    # Apparently on MPS there's usually little benefit from multi-worker loading
    num_workers = cpu_count() if DEVICE.type == "cuda" else 0

    # This gives you faster CPU to GPU transfer, but it only helps on CUDA
    # not on MPS
    pin_memory = DEVICE.type == "cuda"

    # This avoids re-spawning workers each epoch if using multiple workers
    persistent_workers = num_workers > 0

    # Prefetch 2 batches for each worker
    # On CUDA this can speed things up by overlapping data loading with GPU computation
    prefetch_factor = 2 if num_workers > 0 else None

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)

    return train_loader, val_loader


def update_metrics(metrics: dict[str, Metric], logits: Tensor, labels: Tensor):
    _, preds = torch.max(logits, 1)

    for metric in metrics.values():
        if hasattr(metric, 'top_k') and metric.top_k > 1:
            # For top-k accuracy, we need to pass the raw logits instead of the predicted class indices
            metric.update(logits, labels)
        else:
            metric.update(preds, labels)


def reset_metrics(metrics: dict[str, Metric]):
    for metric in metrics.values():
        metric.reset()


def compute_metrics(metrics: dict[str, Metric]) -> dict[str, float]:
    return {name: metric.compute().item() for name, metric in metrics.items()}


def train_epoch(model, train_loader, criterion, optimizer, metrics, scaler, mixup_fn=None):
    model.train()
    mean_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(DEVICE).float()
        labels = labels.to(DEVICE).long()

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        with autocast(device_type=DEVICE.type):
            logits = model(images)  # [B, num_classes]
            # IMPORTANT: do NOT squeeze logits; it breaks when batch_size==1
            loss = criterion(logits, labels)

        num_batches += 1
        mean_loss += (loss.item() - mean_loss) / num_batches

        # For metric updates, when labels are soft we should use the hard labels
        # (timm Mixup returns soft targets; we recover hard labels via argmax)
        if labels.dim() == 2 and labels.dtype.is_floating_point:
            hard_labels = labels.argmax(dim=1)
            update_metrics(metrics, logits, hard_labels)
        else:
            update_metrics(metrics, logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    metric_values = compute_metrics(metrics)
    metric_values["cross_entropy"] = mean_loss
    reset_metrics(metrics)
    return metric_values


def validate(model, val_loader, criterion, metrics):
    model.eval()
    mean_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            with autocast(device_type=DEVICE.type):
                logits = model(images)
                loss = criterion(logits, labels)

            num_batches += 1
            mean_loss += (loss.item() - mean_loss) / num_batches

            update_metrics(metrics, logits, labels)

    metric_values = compute_metrics(metrics)
    metric_values["cross_entropy"] = mean_loss
    reset_metrics(metrics)

    return metric_values


def train():
    images, labels, mean, std = load_data()
    print("Loaded data from cache:", images.shape, labels.shape,
          "mean:", mean, "std:", std)

    train_dataset, val_dataset = split_dataset(images, labels, mean, std)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

    mixup = MixupCutmixApply(create_mixup_cutmix(num_classes=10, mixup_alpha=0.2, cutmix_alpha=1.0))

    """
    import cv2
    
    for images, labels in train_loader:
        images = images.to(DEVICE).float()
        labels = labels.to(DEVICE).long()

        images, labels = mixup(images, labels)

        for i in range(images.shape[0]):
            print(images[i].min(), images[i].max(), labels[i])
            cv2.imshow("Sample Image", images[i].permute(1, 2, 0).cpu().numpy())
            cv2.waitKey(0)

    exit()
    """

    model = ResNeXt(
        # layers=[3, 4, 6, 3],
        # layers=[2, 3, 4, 2],
        layers=[1, 2, 3, 1],
        # layers=[3, 4, 23, 3],
        num_classes=10,
        groups=32,
        width_per_group=4,
        drop_path_rate=0.1,
    )
    model = model.to(DEVICE)

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99), use_buffers=True)

    num_epochs = 300

    # For training, Mixup/CutMix augmentations make the targets soft one-hot mixtures, so
    # there is a special loss for that
    train_criterion = SoftTargetCrossEntropy()
    # For validation, the labels are still hard integers, so regular cross-entropy is fine
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Define the evaluation metrics
    metrics = {
        "accuracy": Accuracy(task="multiclass", num_classes=10),
        "top_2_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=2),
        "top_3_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=3),
        "top_4_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=4),
        "top_5_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=5),
        "top_6_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=6),
        "top_7_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=7),
        "top_8_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=8),
        "top_9_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=9),
        "f1_score": F1Score(task="multiclass", num_classes=10),
    }

    # Put all the metrics on the same device as the model
    for metric in metrics.values():
        metric.to(DEVICE)

    if WANDB:
        wandb.init(
            project="CS148-MNIST",
            config={
                "dataset": DATASET_NAME,
                "model": model.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "lr": optimizer.defaults["lr"],
                "batch_size": train_loader.batch_size,
                "epochs": 10000,
                "device": device_name,
            },
        )

        wandb.watch(model, log="gradients", log_freq=100)

        checkpoint_dir = os.path.join("checkpoints", wandb.run.id)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # GradScaler is CUDA-only; disable it automatically on MPS/CPU.
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            metrics,
            scaler,
            mixup_fn=mixup,
        )
        ema.update_parameters(model)

        # Validate with both raw and EMA model
        val_metrics_raw = validate(model, val_loader, val_criterion, metrics)
        val_metrics_ema = validate(ema, val_loader, val_criterion, metrics)
        scheduler.step()

        if WANDB:
            # Log both sets of validation metrics
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr()[0],
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val_raw/{k}": v for k, v in val_metrics_raw.items()},
                    **{f"val_ema/{k}": v for k, v in val_metrics_ema.items()},
                }
            )

        pprint(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val_raw": val_metrics_raw,
                "val_ema": val_metrics_ema,
            }
        )

        # Save checkpoint if EMA val accuracy improved
        val_accuracy = val_metrics_ema["accuracy"]
        if WANDB and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            # Log to wandb as an artifact
            artifact = wandb.Artifact(
                name=f"model-best",
                type="model",
                metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy},
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

            print(f"Saved best model (val_accuracy={val_accuracy:.4f}) at epoch {epoch + 1}")

    if WANDB:
        wandb.finish()


if __name__ == "__main__":
    freeze_support()
    train()
