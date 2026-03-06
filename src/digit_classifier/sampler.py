"""Custom samplers for training.

- :class:`RatioBatchSampler`: Controls original / external mixing ratio per batch.
- :class:`RepeatAugSampler`: Repeated augmentation (DeiT-III / timm style).
- :class:`RepeatAugRatioBatchSampler`: Combines repeated augmentation with ratio control.
"""

from __future__ import annotations

import math
import random
from typing import Iterator

import torch
from torch.utils.data import Sampler


class RatioBatchSampler:
    """Yield batches with a target fraction of original-dataset indices.

    Parameters
    ----------
    original_count : int
        Number of original training images (indices ``0 .. original_count-1``).
    total_count : int
        Total dataset length (original + external).
    batch_size : int
        Desired batch size.
    primary_fraction : float
        Target fraction of each batch drawn from the original pool (default 0.95).
    drop_last : bool
        If ``True``, discard the final incomplete batch.
    seed : int | None
        RNG seed for reproducibility.
    num_replicas : int
        For DDP: number of processes (default 1).
    rank : int
        For DDP: this process's rank (default 0).
    """

    def __init__(
        self,
        original_count: int,
        total_count: int,
        batch_size: int,
        primary_fraction: float = 0.95,
        drop_last: bool = True,
        seed: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.original_count = int(original_count)
        self.external_count = int(total_count - original_count)
        self.total_count = int(total_count)
        self.batch_size = int(batch_size)
        self.primary_fraction = float(primary_fraction)
        self.k_primary = max(1, int(round(self.batch_size * self.primary_fraction)))
        self.k_secondary = self.batch_size - self.k_primary
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rnd = random.Random((self.seed or 0) + self.epoch * 1000)

        orig_idx = [i for i in range(self.original_count) if i % self.num_replicas == self.rank]
        ext_idx = [
            i for i in range(self.original_count, self.original_count + self.external_count)
            if (i - self.original_count) % self.num_replicas == self.rank
        ]

        rnd.shuffle(orig_idx)
        rnd.shuffle(ext_idx)

        p_orig = 0
        p_ext = 0

        while True:
            # Check whether enough original indices remain for a full primary chunk.
            if p_orig + self.k_primary > len(orig_idx):
                if self.drop_last:
                    break
                rnd.shuffle(orig_idx)
                p_orig = 0

            primary_block = orig_idx[p_orig : p_orig + self.k_primary]
            p_orig += self.k_primary

            secondary_block: list[int] = []
            if self.k_secondary > 0:
                if self.external_count == 0:
                    # Fallback: draw extra originals when no externals exist.
                    extra_needed = self.k_secondary
                    if p_orig + extra_needed > len(orig_idx):
                        if self.drop_last:
                            break
                        rnd.shuffle(orig_idx)
                        p_orig = 0
                    secondary_block = orig_idx[p_orig : p_orig + extra_needed]
                    p_orig += extra_needed
                else:
                    # Draw from the external pool, cycling if exhausted.
                    if p_ext + self.k_secondary > len(ext_idx):
                        remaining = len(ext_idx) - p_ext
                        secondary_block.extend(ext_idx[p_ext:])
                        rnd.shuffle(ext_idx)
                        p_ext = 0
                        need = self.k_secondary - remaining
                        secondary_block.extend(ext_idx[p_ext : p_ext + need])
                        p_ext += need
                    else:
                        secondary_block = ext_idx[p_ext : p_ext + self.k_secondary]
                        p_ext += self.k_secondary

            batch = primary_block + secondary_block
            if len(batch) != self.batch_size:
                if self.drop_last:
                    break
                while len(batch) < self.batch_size:
                    batch.append(orig_idx[p_orig % len(orig_idx)])
                    p_orig += 1

            yield batch

    def __len__(self) -> int:
        """Number of batches per epoch. Stops when primary is exhausted."""
        if self.drop_last:
            if self.external_count == 0:
                return self.total_count // self.batch_size
            return self.original_count // self.k_primary
        if self.external_count == 0:
            return -(-self.total_count // self.batch_size)
        # Approximate when not drop_last (rare)
        return -(-self.original_count // self.k_primary)

    def set_epoch(self, epoch: int) -> None:
        """For DDP: vary shuffle per epoch."""
        self.epoch = epoch


class RepeatAugSampler(Sampler[int]):
    """Repeated augmentation sampler (DeiT-III / timm RASampler style).

    Repeats each sample index ``num_repeats`` times so that different augmented
    versions of the same image are seen in different batches. Supports both
    single-GPU (num_replicas=1, rank=0) and distributed training.

    Based on https://github.com/facebookresearch/deit/blob/main/samplers.py
    and timm's RepeatAugSampler.
    """

    def __init__(
        self,
        dataset: object,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        num_repeats: int = 3,
        selected_round: int = 0,
        selected_ratio: float = 0.0,
    ) -> None:
        if num_replicas is None or rank is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    num_replicas = num_replicas or dist.get_world_size()
                    rank = rank if rank is not None else dist.get_rank()
                else:
                    num_replicas = num_replicas or 1
                    rank = rank if rank is not None else 0
            except Exception:
                num_replicas = num_replicas or 1
                rank = rank if rank is not None else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) * num_repeats / num_replicas))
        self.total_size = self.num_samples * num_replicas

        selected_ratio = selected_ratio or num_replicas
        if selected_round:
            self.num_selected_samples = int(
                math.floor(len(dataset) // selected_round * selected_round / selected_ratio)
            )
        else:
            # Use full repeated set when no rounding (single-GPU / small datasets)
            self.num_selected_samples = self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        indices = torch.repeat_interleave(indices, repeats=int(self.num_repeats), dim=0)
        indices = indices.tolist()

        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices[: self.num_selected_samples])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class RepeatAugRatioBatchSampler:
    """Combines repeated augmentation with RatioBatchSampler's primary/external ratio.

    Repeats each sample index ``num_repeats`` times (for different augmentations),
    then forms batches with ``primary_fraction`` from the original pool and the
    remainder from the external pool. Use when both repeat_aug and external data
    are enabled. Supports DDP via num_replicas and rank.
    """

    def __init__(
        self,
        original_count: int,
        total_count: int,
        batch_size: int,
        primary_fraction: float = 0.95,
        num_repeats: int = 3,
        drop_last: bool = True,
        seed: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.original_count = int(original_count)
        self.external_count = int(total_count - original_count)
        self.total_count = int(total_count)
        self.batch_size = int(batch_size)
        self.primary_fraction = float(primary_fraction)
        self.k_primary = max(1, int(round(self.batch_size * self.primary_fraction)))
        self.k_secondary = self.batch_size - self.k_primary
        self.num_repeats = int(num_repeats)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.epoch = 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed((self.seed or 0) + self.epoch * 1000)

        orig_full = torch.arange(self.original_count)
        orig_full = orig_full[orig_full % self.num_replicas == self.rank]
        orig_idx = torch.repeat_interleave(orig_full, repeats=self.num_repeats, dim=0)
        orig_idx = orig_idx[torch.randperm(len(orig_idx), generator=g)].tolist()

        ext_full = torch.arange(self.original_count, self.original_count + self.external_count)
        ext_full = ext_full[(ext_full - self.original_count) % self.num_replicas == self.rank]
        ext_idx = torch.repeat_interleave(ext_full, repeats=self.num_repeats, dim=0)
        ext_idx = ext_idx[torch.randperm(len(ext_idx), generator=g)].tolist()

        p_orig = 0
        p_ext = 0

        while True:
            if p_orig + self.k_primary > len(orig_idx):
                if self.drop_last:
                    break
                g.manual_seed((self.seed or 0) + self.epoch * 1000 + 1)
                orig_idx = torch.arange(self.original_count).repeat_interleave(self.num_repeats)
                orig_idx = orig_idx[torch.randperm(len(orig_idx), generator=g)].tolist()
                p_orig = 0

            primary_block = orig_idx[p_orig : p_orig + self.k_primary]
            p_orig += self.k_primary

            secondary_block: list[int] = []
            if self.k_secondary > 0:
                if self.external_count == 0:
                    extra_needed = self.k_secondary
                    if p_orig + extra_needed > len(orig_idx):
                        if self.drop_last:
                            break
                        g.manual_seed((self.seed or 0) + self.epoch * 1000 + 2)
                        orig_idx = torch.arange(self.original_count).repeat_interleave(self.num_repeats)
                        orig_idx = orig_idx[torch.randperm(len(orig_idx), generator=g)].tolist()
                        p_orig = 0
                    secondary_block = orig_idx[p_orig : p_orig + extra_needed]
                    p_orig += extra_needed
                else:
                    if p_ext + self.k_secondary > len(ext_idx):
                        remaining = len(ext_idx) - p_ext
                        secondary_block.extend(ext_idx[p_ext:])
                        g.manual_seed((self.seed or 0) + self.epoch * 1000 + 3)
                        ext_idx = torch.arange(
                            self.original_count,
                            self.original_count + self.external_count,
                        ).repeat_interleave(self.num_repeats)
                        ext_idx = ext_idx[torch.randperm(len(ext_idx), generator=g)].tolist()
                        p_ext = 0
                        need = self.k_secondary - remaining
                        secondary_block.extend(ext_idx[p_ext : p_ext + need])
                        p_ext += need
                    else:
                        secondary_block = ext_idx[p_ext : p_ext + self.k_secondary]
                        p_ext += self.k_secondary

            batch = primary_block + secondary_block
            if len(batch) != self.batch_size:
                if self.drop_last:
                    break
                while len(batch) < self.batch_size:
                    batch.append(orig_idx[p_orig % len(orig_idx)])
                    p_orig += 1

            yield batch

    def __len__(self) -> int:
        """Number of batches per epoch. Stops when primary (repeated) is exhausted."""
        # Per-replica primary length when using DDP (each rank gets 1/num_replicas of data)
        orig_per_replica = (self.original_count + self.num_replicas - 1) // self.num_replicas
        orig_len = orig_per_replica * self.num_repeats
        if self.drop_last:
            if self.external_count == 0:
                return orig_len // self.batch_size
            return orig_len // self.k_primary
        if self.external_count == 0:
            return -(-orig_len // self.batch_size)
        return -(-orig_len // self.k_primary)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
