"""Train / validation splitting with external dataset mixing and statistics.

This module implements the full data-preparation pipeline that runs *after*
the raw images have been preprocessed and cached as an ``.npz`` file:

1. Deterministic 90 / 10 split (seed 42).
2. Load external datasets lazily (on-demand per sample).
3. Deduplicate NIST-like overlaps across external sources, caching the
   surviving index lists so dedup only runs once.
4. Compute per-channel mean / std from the **original** training images only.
5. Wrap external preprocessors so they normalise with the same statistics.
6. Return a ``ConcatDataset`` with a ``num_original`` attribute that the
   :class:`~digit_classifier.sampler.RatioBatchSampler` reads.
"""

from __future__ import annotations

import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import v2 as T

from digit_classifier.augmentation import ApplyTransform, build_augmentor
from digit_classifier.config import AugmentConfig
from digit_classifier.dataset import DigitDataset


class _NormalisedPreprocessor:
    """Picklable wrapper that applies a preprocessor then normalises.

    This replaces the closure that was previously used to wrap external
    dataset preprocessors — closures can't be pickled, which breaks
    multi-worker DataLoaders.
    """

    def __init__(self, preprocessor: Any, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.preprocessor = preprocessor
        self.mean = mean
        self.std = std

    def __call__(self, img: Any) -> Tensor:
        t = self.preprocessor(img)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        t = t.to(torch.float32)
        return T.Normalize(mean=self.mean, std=self.std)(t)
from digit_classifier.external import (
    CachedExternalDataset,
    ExternalDataset,
    ExternalOnDemandDataset,
    compute_external_manifest_hash,
    deduplicate_nist_like_datasets,
    get_external_only_fractions,
)

console = Console()

DEDUP_CACHE_DIR = "datasets"


# ---------------------------------------------------------------------------
# Internal helpers for channel statistics
# ---------------------------------------------------------------------------

def _accumulate_stats_from_tensor(tensor: Tensor) -> tuple[Tensor, Tensor, int]:
    """Sum and sum-of-squares over a ``(N, C, H, W)`` tensor."""
    s1 = tensor.sum(dim=(0, 2, 3))
    s2 = (tensor * tensor).sum(dim=(0, 2, 3))
    n = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
    return s1, s2, n


# ---------------------------------------------------------------------------
# Dedup index cache (tiny — just a JSON of surviving indices per source)
# ---------------------------------------------------------------------------

def _dedup_cache_path(manifest_hash: str) -> Path:
    return Path(DEDUP_CACHE_DIR) / f"dedup_indices_{manifest_hash}.json"


def _try_load_dedup_cache(manifest_hash: str) -> dict[str, list[int]] | None:
    """Load cached dedup indices if available. Returns {source_name: [indices]}."""
    path = _dedup_cache_path(manifest_hash)
    if not path.exists():
        return None
    console.print(f"[dim]Loading cached dedup indices from [cyan]{path}[/cyan][/dim]")
    with open(path) as f:
        return json.load(f)


def _save_dedup_cache(
    manifest_hash: str,
    datasets_list: list[ExternalOnDemandDataset],
) -> None:
    """Save the surviving indices after dedup (a few KB)."""
    path = _dedup_cache_path(manifest_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {ds.dataset.value: ds.indices for ds in datasets_list}
    with open(path, "w") as f:
        json.dump(cache, f)
    console.print(f"  [green]Saved dedup indices to {path}[/green]")


# ---------------------------------------------------------------------------
# External-only validation dataset wrapper
# ---------------------------------------------------------------------------

class _ExternalValDataset(Dataset):
    """Wraps ExternalOnDemandDataset and applies a transform to each image."""

    def __init__(self, underlying: ExternalOnDemandDataset, transform: Any) -> None:
        self.underlying = underlying
        self.transform = transform

    def __len__(self) -> int:
        return len(self.underlying)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img, label = self.underlying[idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_dataset_external_only(
    *,
    external_val_source: str = "MNIST Test",
    color: bool = True,
    size: int = 224,
    seed: int = 42,
    external_cache: bool = False,
    external_cache_max_mb: float = 2048,
    num_workers: int = 1,
) -> tuple[ConcatDataset | Dataset, Dataset, tuple[float, ...], tuple[float, ...]]:
    """Build train/val from external sources only (no primary dataset).

    Returns ``(train_dataset, val_dataset, mean, std)``. Uses default mean/std
    (0.5 per channel). Sets ``num_original=0`` so DataLoader uses simple shuffle.
    """
    try:
        val_source = ExternalDataset(external_val_source)
    except ValueError:
        valid = [e.value for e in ExternalDataset]
        raise ValueError(f"external_val_source must be one of {valid}") from None

    external_fractions = get_external_only_fractions(val_source)
    dataset_names = [ds.value for ds in external_fractions]
    manifest_hash = compute_external_manifest_hash(
        dataset_names, color, size, seed, 1.0,
        external_only=True, val_source=val_source.value,
    )

    num_channels = 3 if color else 1
    mean_t = (0.5,) * num_channels
    std_t = (0.5,) * num_channels

    generator = torch.Generator().manual_seed(seed)
    dedup_cache = _try_load_dedup_cache(manifest_hash)

    # Load training externals (all except val source)
    external_datasets_list: list[ExternalOnDemandDataset] = []
    total_external = len(external_fractions)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading external datasets (train)", total=total_external)
        for ext_dataset, ext_fraction in external_fractions.items():
            progress.update(task, description=f"[cyan]Loading {ext_dataset.value}")
            max_samples = None if ext_fraction == -1 else int(ext_fraction)
            if max_samples is not None and max_samples == 0 and ext_fraction > 0:
                max_samples = 1
            try:
                ext_ds = ExternalOnDemandDataset(
                    ext_dataset, color, size, max_samples=max_samples, rnd=generator,
                )
            except Exception as exc:
                console.print(f"  [yellow]⚠ Skipping {ext_dataset.value}: {exc}[/yellow]")
                progress.advance(task)
                continue
            if len(ext_ds) == 0:
                progress.advance(task)
                continue
            if dedup_cache is not None and ext_ds.dataset.value in dedup_cache:
                ext_ds.indices = dedup_cache[ext_ds.dataset.value]
            external_datasets_list.append(ext_ds)
            console.print(f"  [dim]✓ {ext_dataset.value}: {len(ext_ds):,} samples[/dim]")
            progress.advance(task)

    if not external_datasets_list:
        raise ValueError("No external training datasets loaded; check external_val_source and sources")

    total_ext = sum(len(ds) for ds in external_datasets_list)
    console.print(f"[bold]External train:[/bold] {len(external_datasets_list)} sources, {total_ext:,} samples")

    if dedup_cache is None:
        console.print("[dim]Deduplicating NIST-like datasets …[/dim]")
        removed = deduplicate_nist_like_datasets(external_datasets_list)
        if removed > 0:
            console.print(f"  [yellow]Removed {removed:,} duplicate samples[/yellow]")
        _save_dedup_cache(manifest_hash, external_datasets_list)
    else:
        console.print(f"  [dim]Using cached dedup indices[/dim]")

    if external_cache and external_cache_max_mb > 0:
        max_mb_per_worker = external_cache_max_mb / max(1, num_workers)
        max_bytes_per_worker = int(max_mb_per_worker * 1024 * 1024)
        external_datasets_list = [
            CachedExternalDataset(ds, max_bytes_per_worker)
            for ds in external_datasets_list
        ]

    # Wrap preprocessors with normalisation
    for ext in external_datasets_list:
        target = ext._underlying if isinstance(ext, CachedExternalDataset) else ext
        target.preprocessor = _NormalisedPreprocessor(target.preprocessor, mean_t, std_t)

    train_dataset: ConcatDataset | Dataset = ConcatDataset(
        external_datasets_list,  # type: ignore[list-item]
    )
    setattr(train_dataset, "num_original", 0)

    # Load validation external (preprocessor already does resize, crop, normalize)
    console.print(f"[dim]Loading validation: {val_source.value}[/dim]")
    val_ext = ExternalOnDemandDataset(val_source, color, size, max_samples=None, rnd=generator)
    val_ext.preprocessor = _NormalisedPreprocessor(val_ext.preprocessor, mean_t, std_t)
    val_dataset = _ExternalValDataset(val_ext, T.Identity())
    console.print(f"  [dim]✓ {val_source.value}: {len(val_dataset):,} samples[/dim]")

    console.print(f"[bold]Mean:[/bold] ({', '.join(f'{v:.4f}' for v in mean_t)})")
    console.print(f"[bold]Std:[/bold]  ({', '.join(f'{v:.4f}' for v in std_t)})")
    console.print(f"[bold green]Dataset ready (external-only):[/bold green] {len(train_dataset):,} train / {len(val_dataset):,} val")
    return train_dataset, val_dataset, mean_t, std_t


def split_dataset(
    images: Tensor,
    labels: Tensor,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    *,
    train_fraction: float = 0.9,
    mix_external: bool = True,
    external_fractions: dict[ExternalDataset, float | int] | None = None,
    color: bool = True,
    size: int = 224,
    seed: int = 42,
    augment_cfg: AugmentConfig | None = None,
    augment_scheme: str = "yolo",
    external_cache: bool = False,
    external_cache_max_mb: float = 2048,
    num_workers: int = 1,
) -> tuple[ConcatDataset | DigitDataset, DigitDataset, tuple[float, ...], tuple[float, ...]]:
    """Split *images* / *labels* into train and validation sets.

    Returns ``(train_dataset, val_dataset, mean, std)``.
    """
    num_samples = images.shape[0]
    num_train = int(train_fraction * num_samples)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    val_images = images[val_idx]
    val_labels = labels[val_idx]

    num_original_train = len(train_images)
    console.print(f"Split: [green]{num_original_train}[/green] train / [blue]{len(val_images)}[/blue] val")

    # ------------------------------------------------------------------
    # External datasets (lazy loading + dedup index caching)
    # ------------------------------------------------------------------
    external_datasets_list: list[ExternalOnDemandDataset] = []

    if mix_external and external_fractions is not None:
        dataset_names = [ds.value for ds in external_fractions]
        manifest_hash = compute_external_manifest_hash(
            dataset_names, color, size, seed, train_fraction,
        )

        # Check for cached dedup indices
        dedup_cache = _try_load_dedup_cache(manifest_hash)

        total_external = len(external_fractions)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading external datasets", total=total_external)

            for ext_dataset, ext_fraction in external_fractions.items():
                progress.update(task, description=f"[cyan]Loading {ext_dataset.value}")

                if ext_fraction == -1:
                    max_samples = None
                else:
                    max_samples = int(round(ext_fraction * num_original_train))
                    if max_samples == 0 and ext_fraction > 0:
                        max_samples = 1

                try:
                    ext_ds = ExternalOnDemandDataset(
                        ext_dataset, color, size, max_samples=max_samples, rnd=generator,
                    )
                except Exception as exc:
                    console.print(f"  [yellow]⚠ Skipping {ext_dataset.value}: {exc}[/yellow]")
                    progress.advance(task)
                    continue

                if len(ext_ds) == 0:
                    progress.advance(task)
                    continue

                # Apply cached dedup indices if available
                if dedup_cache is not None and ext_ds.dataset.value in dedup_cache:
                    ext_ds.indices = dedup_cache[ext_ds.dataset.value]

                external_datasets_list.append(ext_ds)
                console.print(f"  [dim]✓ {ext_dataset.value}: {len(ext_ds):,} samples[/dim]")
                progress.advance(task)

        if external_datasets_list:
            total_ext = sum(len(ds) for ds in external_datasets_list)
            console.print(f"[bold]External datasets loaded:[/bold] {len(external_datasets_list)} sources, {total_ext:,} total samples")

            # Run dedup only if we don't have cached indices
            if dedup_cache is None:
                console.print("[dim]Deduplicating NIST-like datasets …[/dim]")
                removed = deduplicate_nist_like_datasets(external_datasets_list)
                if removed > 0:
                    console.print(f"  [yellow]Removed {removed:,} duplicate samples[/yellow]")
                else:
                    console.print(f"  [dim]No duplicates found[/dim]")

                _save_dedup_cache(manifest_hash, external_datasets_list)
            else:
                console.print(f"  [dim]Using cached dedup indices (skipped deduplication)[/dim]")

            # Wrap with memory-bounded cache after dedup (dedup mutates .indices)
            # Divide by num_workers so total across all workers stays within user limit
            if external_cache and external_cache_max_mb > 0:
                max_mb_per_worker = external_cache_max_mb / max(1, num_workers)
                max_bytes_per_worker = int(max_mb_per_worker * 1024 * 1024)
                external_datasets_list = [
                    CachedExternalDataset(ds, max_bytes_per_worker)
                    for ds in external_datasets_list
                ]
                console.print(
                    f"  [dim]External cache: max {external_cache_max_mb:.0f} MB total "
                    f"({max_mb_per_worker:.0f} MB per worker × {num_workers})[/dim]"
                )

    has_externals = len(external_datasets_list) > 0

    # ------------------------------------------------------------------
    # Compute mean / std
    # ------------------------------------------------------------------
    use_cached = (not mix_external or not has_externals) and mean is not None and std is not None

    if use_cached:
        mean_list = list(mean)
        std_list = list(std)
        console.print(f"[dim]Using cached mean/std[/dim]")
    else:
        console.print("[dim]Computing mean/std from original training images …[/dim]")
        num_channels = train_images.shape[1]
        s1 = torch.zeros(num_channels)
        s2 = torch.zeros(num_channels)
        n = 0

        if train_images.numel() > 0:
            batch_s1, batch_s2, batch_n = _accumulate_stats_from_tensor(train_images)
            s1 += batch_s1
            s2 += batch_s2
            n += batch_n

        if n == 0:
            raise ValueError("No pixels found when computing mean/std")

        mean_list = (s1 / n).tolist()
        std_list = torch.sqrt(s2 / n - torch.tensor(mean_list) ** 2).tolist()

    mean_t = tuple(mean_list)
    std_t = tuple(std_list)

    console.print(f"  [bold]Mean:[/bold] ({', '.join(f'{v:.4f}' for v in mean_t)})")
    console.print(f"  [bold]Std:[/bold]  ({', '.join(f'{v:.4f}' for v in std_t)})")

    # ------------------------------------------------------------------
    # Build training dataset
    # ------------------------------------------------------------------
    train_orig_dataset = DigitDataset(
        images=train_images,
        labels=train_labels,
        transform=build_augmentor(
            augment_scheme,
            mean=mean_t,
            std=std_t,
            size=size,
            cfg=augment_cfg,
        ),
    )

    if has_externals:
        # Wrap each external preprocessor so it returns normalised tensors
        # using the *original* dataset's statistics.
        for ext in external_datasets_list:
            target = ext._underlying if isinstance(ext, CachedExternalDataset) else ext
            target.preprocessor = _NormalisedPreprocessor(target.preprocessor, mean_t, std_t)

        train_dataset: ConcatDataset | DigitDataset = ConcatDataset(
            [train_orig_dataset] + external_datasets_list,  # type: ignore[list-item]
        )
    else:
        train_dataset = train_orig_dataset

    # Tag with the original count so RatioBatchSampler knows where to split.
    setattr(train_dataset, "num_original", num_original_train)

    # ------------------------------------------------------------------
    # Validation dataset (resize to target size, then normalise)
    # ------------------------------------------------------------------
    val_transform = T.Compose([
        T.Resize(size),
        T.CenterCrop((size, size)),
        T.Normalize(mean=mean_t, std=std_t),
    ])
    val_dataset = DigitDataset(
        images=val_images,
        labels=val_labels,
        transform=ApplyTransform(val_transform),
    )

    console.print(f"[bold green]Dataset ready:[/bold green] {len(train_dataset):,} train / {len(val_dataset):,} val")
    return train_dataset, val_dataset, mean_t, std_t
