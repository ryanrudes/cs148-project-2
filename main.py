from multiprocessing import freeze_support

from torch.utils.data import Dataset, DataLoader
from torchmetrics import Metric, Accuracy, F1Score

from torch import Tensor

import torch.nn as nn
import numpy as np
import torch
import wandb
import os

from rich.pretty import pprint

from dataset import DummyMNIST
from augmentations import augment
from net import CNN, ResidualCNN, CNNWithResidualConnections, ClassificationHead
from torchvision.models import resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
from torchvision.models import ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights
from resnext import ResNeXt

DATASET_NAME = "mnist_rgb_64"
COLOR = True

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
device_name = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
DEVICE = torch.device(device_name)
print("Using device:", DEVICE)


def load_data():
    dataset_path = os.path.join("datasets", DATASET_NAME + ".npz")
    preprocessed_data = np.load(dataset_path)
    images = torch.from_numpy(preprocessed_data["images"])
    labels = preprocessed_data["labels"]
    return images, labels


def split_dataset(images, labels, train_fraction: float = 0.9):
    num_samples = len(images)
    num_train = int(train_fraction * num_samples)

    # Deterministic index split
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(42))
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_dataset = DummyMNIST(
        images=images[train_idx],
        labels=labels[train_idx],
        transform=augment,
    )

    val_dataset = DummyMNIST(
        images=images[val_idx],
        labels=labels[val_idx],
        transform=None,
    )

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,              # parallel data loading
        persistent_workers=True,    # avoid re-spawning workers each epoch
        pin_memory=(DEVICE.type == "cuda"),  # faster CPU→GPU transfer (CUDA only, doesn't help on MPS)
        prefetch_factor=2,          # prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=(DEVICE.type == "cuda"),
        prefetch_factor=2,
    )

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


def train_epoch(model, train_loader, criterion, optimizer, metrics):
    model.train()
    mean_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(DEVICE).float()
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits.squeeze(), labels)

        num_batches += 1
        mean_loss += (loss.item() - mean_loss) / num_batches

        update_metrics(metrics, logits, labels)

        loss.backward()
        optimizer.step()

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

            logits = model(images)
            loss = criterion(logits.squeeze(), labels)

            num_batches += 1
            mean_loss += (loss.item() - mean_loss) / num_batches

            _, preds = torch.max(logits, 1)
            update_metrics(metrics, logits, labels)

    metric_values = compute_metrics(metrics)
    metric_values["cross_entropy"] = mean_loss
    reset_metrics(metrics)

    return metric_values


def train():
    images, labels = load_data()
    train_dataset, val_dataset = split_dataset(images, labels)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

    """
    model = ResidualCNN(
        #batch_norm=True,
        #dropout=True,
        dropout=0.2,
        num_classes=10,
        input_channels=3 if COLOR else 1,
    )
    """
    # model = resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    # model.fc = ClassificationHead(in_features=model.fc.in_features, num_classes=10)
    model = ResNeXt(
        # layers=[3, 4, 6, 3],
        # layers=[2, 3, 4, 2],
        layers=[1, 2, 3, 1],
        num_classes=10,
        groups=32,
        width_per_group=4,
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Define the evaluation metrics
    metrics = {
        "accuracy": Accuracy(task="multiclass", num_classes=10),
        "top_2_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=2),
        "top_3_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=3),
        "top_5_accuracy": Accuracy(task="multiclass", num_classes=10, top_k=5),
        "f1_score": F1Score(task="multiclass", num_classes=10),
    }

    # Put all the metrics on the same device as the model
    for metric in metrics.values():
        metric.to(DEVICE)

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

    wandb.watch(model, log="all", log_freq=100)

    best_val_accuracy = 0.0
    checkpoint_dir = os.path.join("checkpoints", wandb.run.id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(10000):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, metrics)
        val_metrics = validate(model, val_loader, criterion, metrics)

        wandb.log(
            {
                "epoch": epoch + 1,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
        )

        pprint({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics
        })

        # Save checkpoint if val accuracy improved
        val_accuracy = val_metrics["accuracy"]
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
            }, checkpoint_path)

            # Log to wandb as an artifact
            artifact = wandb.Artifact(
                name=f"model-best",
                type="model",
                metadata={"epoch": epoch + 1, "val_accuracy": val_accuracy},
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

            print(f"Saved best model (val_accuracy={val_accuracy:.4f}) at epoch {epoch + 1}")

    wandb.finish()


if __name__ == "__main__":
    freeze_support()
    train()
