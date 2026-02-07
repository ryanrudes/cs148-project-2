import torch
import os

from enum import Enum
from tqdm import tqdm

from augmentations import get_preprocessor
from torchvision import datasets


DOWNLOAD_ROOT = os.path.join(os.path.dirname(__file__), "data")


class ExternalDataset(Enum):
    SVHN_TRAIN = "SVHN Train"
    SVHN_TEST = "SVHN Test"
    SVHN_EXTRA = "SVHN Extra"


def dataset_factory(dataset: ExternalDataset) -> datasets.VisionDataset:
    match dataset:
        case ExternalDataset.SVHN_TRAIN:
            return datasets.SVHN(root=DOWNLOAD_ROOT, split="train", download=True)
        case ExternalDataset.SVHN_TEST:
            return datasets.SVHN(root=DOWNLOAD_ROOT, split="test", download=True)
        case ExternalDataset.SVHN_EXTRA:
            return datasets.SVHN(root=DOWNLOAD_ROOT, split="extra", download=True)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset}")


def load_external_dataset(
    dataset: ExternalDataset,
    color: bool,
    size: int,
    max_samples: int | None = None,
    rnd: torch.Generator | None = None,
):
    dataset_obj = dataset_factory(dataset)
    total = len(dataset_obj)
    idxs = torch.randperm(total, generator=rnd)

    if max_samples is not None:
        idxs = idxs[:max_samples]

    preprocessor = get_preprocessor(color, size)

    images = []
    labels = []

    for i in tqdm(idxs, desc=f"Preprocessing {dataset.value}"):
        image, label = dataset_obj[i]
        assert isinstance(label, int) and 0 <= label <= 9, f"Unexpected label: {label}"

        image = preprocessor(image)

        images.append(image)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels
