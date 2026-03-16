from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digit_classifier.clip import CLIPConfig
    from transformers import CLIPProcessor

from pathlib import Path

from torch.utils.data import Dataset

import torch
import numpy as np
import logging

log = logging.getLogger(__name__)

def load_mnist_in_the_wild(
    cfg: CLIPConfig,
    cache_dir: str = "datasets",
):
    log.info(f"Loading MNIST in the Wild dataset")
    dataset_name = f"mnist_itw_rgb_{cfg.image_size}"
    path = Path(cache_dir) / f"{dataset_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m digit_classifier download && "
            f"python -m digit_classifier preprocess --name {dataset_name} --color --size {cfg.image_size}"
        )
    data = np.load(path)
    images = torch.from_numpy(data["images"])
    if images.dtype != torch.uint8:
        raise TypeError(f"Images must be uint8, got {images.dtype}")
    labels = torch.from_numpy(data["labels"]).long()
    mean = tuple(data["mean"]) if "mean" in data else None
    std = tuple(data["std"]) if "std" in data else None
    return images, labels, mean, std

class MNISTInTheWild(Dataset):
    def __init__(self, cfg: CLIPConfig, processor: CLIPProcessor, cache_dir: str = "datasets"):
        self.images, self.labels, self.mean, self.std = load_mnist_in_the_wild(cfg, cache_dir)
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]