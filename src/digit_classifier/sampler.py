"""Custom batch sampler for controlling original / external mixing ratio.

The :class:`RatioBatchSampler` composes each batch so that approximately
``primary_fraction`` of the indices come from the first ``original_count``
items (the "original" training images) and the remainder come from the
external pool.  The external pool is cycled so that the *entire* external
dataset is seen across training.
"""

from __future__ import annotations

import random
from typing import Iterator


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
    """

    def __init__(
        self,
        original_count: int,
        total_count: int,
        batch_size: int,
        primary_fraction: float = 0.95,
        drop_last: bool = True,
        seed: int | None = None,
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

    def __iter__(self) -> Iterator[list[int]]:
        rnd = random.Random(self.seed)

        orig_idx = list(range(self.original_count))
        ext_idx = list(range(self.original_count, self.original_count + self.external_count))

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
        if self.drop_last:
            return self.total_count // self.batch_size
        return -(-self.total_count // self.batch_size)
