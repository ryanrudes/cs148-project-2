"""Load pareidolia test set (generate-pareidolia output) for evaluation.

Test images are loaded with resize + normalize only — no augmentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from digit_classifier.augmentation import get_preprocessor


class PareidoliaTestDataset(Dataset):
    """Dataset of pareidolia images from metadata.jsonl + images/.

    Loads images on demand. Applies only resize + normalize (no augmentation).
    With preload=True (default), loads all images into memory at init for fast iteration.
    """

    def __init__(
        self,
        root_dir: str | Path,
        color: bool = True,
        size: int = 224,
        mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        std: tuple[float, ...] = (0.5, 0.5, 0.5),
        preload: bool = True,
    ) -> None:
        self.root = Path(root_dir)
        self.color = color
        self.size = size
        self.mean = mean
        self.std = std

        metadata_path = self.root / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Pareidolia metadata not found: {metadata_path}")

        self.samples: list[tuple[str, int]] = []
        skipped = 0
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                img_path = row.get("image_path")
                digit = int(row.get("digit", 0))
                if img_path is None:
                    continue
                full_path = self.root / img_path
                if not full_path.exists():
                    skipped += 1
                    continue
                self.samples.append((str(img_path), digit))
        self.skipped = skipped

        self.preprocessor = get_preprocessor(color, size)
        self.normalize = T.Normalize(mean=mean, std=std)

        self._preloaded: list[tuple[Tensor, int]] | None = None
        if preload and self.samples:
            self._preloaded = []
            for img_path, label in self.samples:
                full_path = self.root / img_path
                img = Image.open(full_path).convert("RGB" if self.color else "L")
                x = self.preprocessor(img)
                if isinstance(x, Tensor):
                    pass
                else:
                    import numpy as np
                    x = torch.from_numpy(x) if hasattr(x, "__array__") else torch.tensor(x)
                if x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
                    x = x.permute(2, 0, 1)
                x = x.to(torch.float32)
                x = self.normalize(x)
                self._preloaded.append((x, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        if self._preloaded is not None:
            return self._preloaded[idx]
        img_path, label = self.samples[idx]
        full_path = self.root / img_path
        img = Image.open(full_path).convert("RGB" if self.color else "L")
        x = self.preprocessor(img)
        if isinstance(x, Tensor):
            pass
        else:
            import numpy as np
            x = torch.from_numpy(x) if hasattr(x, "__array__") else torch.tensor(x)
        if x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        x = x.to(torch.float32)
        x = self.normalize(x)
        return x, label
