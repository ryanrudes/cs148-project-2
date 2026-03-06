"""On-demand loading and deduplication of external digit datasets.

Supported sources include SVHN, MNIST, EMNIST, USPS, QMNIST and Semeion —
all fetched via ``torchvision.datasets`` on first use.
"""

from __future__ import annotations

import hashlib
import os
import ssl
from collections import OrderedDict
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


def get_external_only_fractions(val_source: ExternalDataset) -> dict[ExternalDataset, int]:
    """External fractions for external-only mode: all sources except val_source."""
    return {ds: -1 for ds in ExternalDataset if ds != val_source}


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
# Memory-bounded LRU cache for external datasets
# ---------------------------------------------------------------------------

def _tensor_bytes(t: Tensor) -> int:
    """Return approximate memory size of a tensor in bytes."""
    return t.numel() * t.element_size()


def _get_available_memory_bytes() -> int | None:
    """Return available system memory in bytes, or None if unavailable.

    Uses psutil when installed; otherwise returns None (no guard).
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None


class CachedExternalDataset(Dataset):
    """LRU cache wrapper for ExternalOnDemandDataset with a memory budget.

    Caches loaded samples up to *max_bytes* per worker. Evicts least-recently-used
    entries when the budget is exceeded. When psutil is installed, also evicts
    aggressively if available system memory falls below *reserve_bytes* to avoid
    OOM when other processes consume RAM.
    """

    def __init__(
        self,
        underlying: ExternalOnDemandDataset,
        max_bytes: int,
        *,
        reserve_bytes: int = 256 * 1024 * 1024,  # 256 MB headroom for other processes
    ) -> None:
        self._underlying = underlying
        self._max_bytes = max_bytes
        self._reserve_bytes = reserve_bytes
        self._cache: OrderedDict[int, tuple[Tensor, int]] = OrderedDict()
        self._bytes_used = 0

    def __len__(self) -> int:
        return len(self._underlying)

    def _evict_until_fits(self, entry_size: int) -> None:
        """Evict LRU entries until we fit within budget and (if checkable) available RAM."""
        while self._cache:
            available = _get_available_memory_bytes()
            over_budget = self._bytes_used + entry_size > self._max_bytes
            low_available = available is not None and available < entry_size + self._reserve_bytes

            if not over_budget and not low_available:
                break

            evict_idx = next(iter(self._cache))
            evict_img, _ = self._cache.pop(evict_idx)
            self._bytes_used -= _tensor_bytes(evict_img)

            # When available memory is low, evict until we're at 50% of budget
            # to leave headroom for other processes
            if low_available and self._bytes_used <= self._max_bytes // 2:
                break

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        if self._max_bytes <= 0:
            return self._underlying[idx]

        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        img, label = self._underlying[idx]
        entry_size = _tensor_bytes(img) + 8  # label negligible

        self._evict_until_fits(entry_size)

        # Only cache if within budget and (when checkable) enough system RAM headroom
        within_budget = self._bytes_used + entry_size <= self._max_bytes
        available = _get_available_memory_bytes()
        has_headroom = available is None or available >= entry_size + self._reserve_bytes
        if within_budget and has_headroom:
            self._cache[idx] = (img, label)
            self._cache.move_to_end(idx)
            self._bytes_used += entry_size

        return img, label


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
    external_only: bool = False,
    val_source: str | None = None,
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
        f"ext_only={external_only}",
    ]
    if val_source is not None:
        parts.append(f"val={val_source}")
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
