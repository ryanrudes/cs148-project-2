"""On-demand loading and deduplication of external digit datasets.

Supported sources include SVHN, MNIST, EMNIST, USPS, QMNIST and Semeion —
all fetched via ``torchvision.datasets`` on first use.
"""

from __future__ import annotations

import hashlib
import os
import ssl
from enum import Enum

import numpy as np
import torch
from torch import Tensor

# Python 3.14 tightened SSL validation, which breaks some dataset hosts
# (e.g. USPS at csie.ntu.edu.tw has a certificate missing the Subject Key
# Identifier extension).  We fall back to unverified context globally for
# torchvision downloads since these are well-known public datasets.
ssl._create_default_https_context = ssl._create_unverified_context

# Disable tqdm globally so torchvision's internal download bars don't
# conflict with Rich's Jupyter rendering.
from tqdm import tqdm
from unittest.mock import patch as _patch
tqdm.__init_original__ = tqdm.__init__
_orig_init = tqdm.__init__
def _silent_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _orig_init(self, *args, **kwargs)
tqdm.__init__ = _silent_init

from torch.utils.data import Dataset
from torchvision import datasets

from digit_classifier.augmentation import get_preprocessor

DOWNLOAD_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")


# ---------------------------------------------------------------------------
# Enum of every supported external source
# ---------------------------------------------------------------------------

class ExternalDataset(Enum):
    SVHN_TRAIN = "SVHN Train"
    SVHN_TEST = "SVHN Test"
    SVHN_EXTRA = "SVHN Extra"
    MNIST_TRAIN = "MNIST Train"
    MNIST_TEST = "MNIST Test"
    SEMEION = "Semeion"
    EMNIST_TRAIN = "EMNIST Train"
    EMNIST_TEST = "EMNIST Test"
    QMNIST_TRAIN = "QMNIST Train"
    QMNIST_TEST = "QMNIST Test"


# The default fractions dict uses ``-1`` for every source, meaning "use all
# samples".  The :class:`RatioBatchSampler` controls the per-batch mixing
# ratio instead.
DEFAULT_EXTERNAL_FRACTIONS: dict[ExternalDataset, int] = {
    ds: -1 for ds in ExternalDataset
}


def _dataset_factory(dataset: ExternalDataset) -> datasets.VisionDataset:
    """Instantiate the torchvision dataset object (downloads on first call)."""
    root = DOWNLOAD_ROOT
    match dataset:
        case ExternalDataset.SVHN_TRAIN:
            return datasets.SVHN(root=root, split="train", download=True)
        case ExternalDataset.SVHN_TEST:
            return datasets.SVHN(root=root, split="test", download=True)
        case ExternalDataset.SVHN_EXTRA:
            return datasets.SVHN(root=root, split="extra", download=True)
        case ExternalDataset.MNIST_TRAIN:
            return datasets.MNIST(root=root, train=True, download=True)
        case ExternalDataset.MNIST_TEST:
            return datasets.MNIST(root=root, train=False, download=True)
        case ExternalDataset.SEMEION:
            return datasets.SEMEION(root=root, download=True)
        case ExternalDataset.EMNIST_TRAIN:
            return datasets.EMNIST(root=root, split="digits", train=True, download=True)
        case ExternalDataset.EMNIST_TEST:
            return datasets.EMNIST(root=root, split="digits", train=False, download=True)
        case ExternalDataset.QMNIST_TRAIN:
            return datasets.QMNIST(root=root, what="train", compat=True, download=True)
        case ExternalDataset.QMNIST_TEST:
            return datasets.QMNIST(root=root, what="test", compat=True, download=True)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset}")


# ---------------------------------------------------------------------------
# Lazy per-sample dataset
# ---------------------------------------------------------------------------

class ExternalOnDemandDataset(Dataset):
    """Lazily loads and preprocesses external samples one at a time.

    This avoids materialising entire external datasets in memory.  Each call
    to ``__getitem__`` fetches the raw image from the underlying torchvision
    dataset, applies the deterministic preprocessor, and returns a
    ``(tensor, label)`` pair.

    The ``preprocessor`` attribute may be monkey-patched after construction
    (e.g. to append normalisation) — see :func:`split_dataset`.
    """

    def __init__(
        self,
        dataset: ExternalDataset,
        color: bool,
        size: int,
        max_samples: int | None = None,
        rnd: torch.Generator | None = None,
    ) -> None:
        self.dataset = dataset
        self.dataset_obj = _dataset_factory(dataset)
        total = len(self.dataset_obj)

        idxs = torch.randperm(total, generator=rnd) if rnd is not None else torch.randperm(total)
        if max_samples is not None:
            idxs = idxs[:max_samples]

        self.indices: list[int] = idxs.tolist()
        self.preprocessor = get_preprocessor(color, size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        real_idx = self.indices[idx]
        image, label = self.dataset_obj[real_idx]
        image = self.preprocessor(image)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if not isinstance(image, Tensor):
            raise TypeError(f"Preprocessor returned unexpected type: {type(image)}")

        # HWC → CHW if needed
        if image.ndim == 3 and image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)

        image = image.to(torch.float32)
        return image, int(label)


# ---------------------------------------------------------------------------
# NIST-like deduplication
# ---------------------------------------------------------------------------

_NIST_LIKE: frozenset[ExternalDataset] = frozenset({
    ExternalDataset.MNIST_TRAIN,
    ExternalDataset.MNIST_TEST,
    ExternalDataset.EMNIST_TRAIN,
    ExternalDataset.EMNIST_TEST,
    ExternalDataset.QMNIST_TRAIN,
    ExternalDataset.QMNIST_TEST,
})


def is_nist_like_dataset(dataset: ExternalDataset) -> bool:
    """Return ``True`` for NIST-style datasets prone to inter-dataset overlap."""
    return dataset in _NIST_LIKE


def _tensor_canonical_bytes(img: Tensor) -> bytes:
    """Convert a CHW float tensor to canonical uint8 bytes for fingerprinting."""
    if not isinstance(img, Tensor):
        img = torch.as_tensor(img)
    if img.dtype.is_floating_point:
        img_u8 = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        img_u8 = img.to(torch.uint8)
    return bytes(img_u8.cpu().numpy().tobytes())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def compute_external_manifest_hash(
    dataset_names: list[str],
    color: bool,
    size: int,
    seed: int,
    train_fraction: float,
) -> str:
    """Compute a short hex hash that uniquely identifies an external data config.

    The hash changes whenever the set of external sources, image format, or
    split parameters change — any of which would produce different cached data.
    """
    parts = [
        ",".join(sorted(dataset_names)),
        f"color={color}",
        f"size={size}",
        f"seed={seed}",
        f"frac={train_fraction}",
    ]
    blob = "|".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def deduplicate_nist_like_datasets(
    external_datasets: list[ExternalOnDemandDataset],
) -> int:
    """Remove duplicate images across NIST-like external datasets.

    Deduplication uses SHA-1 fingerprints of canonical uint8 pixel data.
    Only NIST-like datasets are checked; SVHN and Semeion are left untouched.
    The internal (original) dataset is not involved — it comes from a
    completely different source with no overlap.

    Datasets are processed in order: the first occurrence of an image is kept,
    later duplicates are removed.

    The function mutates each dataset's ``indices`` list **in place** and
    returns the total number of removed samples.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

    con = Console()
    seen: set[str] = set()
    removed = 0

    nist_datasets = [ext for ext in external_datasets if is_nist_like_dataset(getattr(ext, "dataset", None))]
    if not nist_datasets:
        return 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=con,
    ) as progress:
        for ext in nist_datasets:
            ext_task = progress.add_task(
                f"[magenta]Deduplicating {ext.dataset.value}",
                total=len(ext.indices),
            )
            new_indices: list[int] = []
            for real_idx in ext.indices:
                img_raw, _ = ext.dataset_obj[real_idx]
                img_proc = ext.preprocessor(img_raw)
                fp = hashlib.sha1(_tensor_canonical_bytes(img_proc)).hexdigest()
                if fp in seen:
                    removed += 1
                else:
                    seen.add(fp)
                    new_indices.append(real_idx)
                progress.advance(ext_task)

            ext.indices = new_indices

    return removed
