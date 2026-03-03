"""Tests for repeated augmentation + external data integration.

Ensures that when both repeat_aug and external data are enabled, the training
pipeline uses RepeatAugRatioBatchSampler and correctly mixes original/external
samples with the target primary_fraction while applying repeated augmentation.
"""

import torch
from torch.utils.data import ConcatDataset, TensorDataset

from digit_classifier.sampler import RepeatAugRatioBatchSampler
from digit_classifier import training


# ---------------------------------------------------------------------------
# Direct RepeatAugRatioBatchSampler tests
# ---------------------------------------------------------------------------


def test_repeat_aug_ratio_batch_sampler_respects_primary_fraction():
    """Each batch should have exactly k_primary from original, k_secondary from external."""
    original_count = 100
    external_count = 50
    total_count = original_count + external_count
    batch_size = 32
    primary_fraction = 0.95
    num_repeats = 3

    sampler = RepeatAugRatioBatchSampler(
        original_count=original_count,
        total_count=total_count,
        batch_size=batch_size,
        primary_fraction=primary_fraction,
        num_repeats=num_repeats,
        drop_last=True,
        seed=42,
    )

    k_primary = int(round(batch_size * primary_fraction))
    k_secondary = batch_size - k_primary

    batches = list(sampler)
    assert len(batches) > 0

    for batch in batches:
        assert len(batch) == batch_size
        primary_indices = [i for i in batch if i < original_count]
        secondary_indices = [i for i in batch if i >= original_count]
        assert len(primary_indices) == k_primary
        assert len(secondary_indices) == k_secondary


def test_repeat_aug_ratio_batch_sampler_uses_repeated_indices():
    """Repeated augmentation means the same index can appear across different batches."""
    original_count = 20
    external_count = 10
    total_count = original_count + external_count
    batch_size = 8
    num_repeats = 3

    sampler = RepeatAugRatioBatchSampler(
        original_count=original_count,
        total_count=total_count,
        batch_size=batch_size,
        primary_fraction=0.75,
        num_repeats=num_repeats,
        drop_last=True,
        seed=123,
    )

    batches = list(sampler)
    all_indices = [idx for batch in batches for idx in batch]

    # With repeats, we expect to see indices repeated across the epoch
    # (each of 30 unique samples appears 3 times = 90 total; we draw from these)
    from collections import Counter
    counts = Counter(all_indices)
    # At least some indices should appear more than once (repeated aug)
    assert any(c >= 2 for c in counts.values()), "Expected repeated indices from repeated augmentation"


def test_repeat_aug_ratio_batch_sampler_set_epoch_changes_order():
    """Different epochs should produce different batch order (reproducible per epoch)."""
    sampler = RepeatAugRatioBatchSampler(
        original_count=30,
        total_count=50,
        batch_size=10,
        primary_fraction=0.8,
        num_repeats=3,
        drop_last=True,
        seed=0,
    )

    sampler.set_epoch(0)
    batches_epoch_0 = list(sampler)

    sampler.set_epoch(1)
    batches_epoch_1 = list(sampler)

    # Order should differ between epochs
    assert batches_epoch_0 != batches_epoch_1


def test_repeat_aug_ratio_batch_sampler_same_epoch_same_order():
    """Same epoch + seed should produce identical order."""
    sampler1 = RepeatAugRatioBatchSampler(
        original_count=30,
        total_count=50,
        batch_size=10,
        primary_fraction=0.8,
        num_repeats=3,
        drop_last=True,
        seed=99,
    )
    sampler1.set_epoch(5)
    batches1 = list(sampler1)

    sampler2 = RepeatAugRatioBatchSampler(
        original_count=30,
        total_count=50,
        batch_size=10,
        primary_fraction=0.8,
        num_repeats=3,
        drop_last=True,
        seed=99,
    )
    sampler2.set_epoch(5)
    batches2 = list(sampler2)

    assert batches1 == batches2


# ---------------------------------------------------------------------------
# Integration: _create_dataloaders uses RepeatAugRatioBatchSampler
# ---------------------------------------------------------------------------


def test_create_dataloaders_uses_repeat_aug_ratio_when_both_enabled():
    """When repeat_aug=True and dataset has external data, use RepeatAugRatioBatchSampler."""
    # Build a dataset with original + external (num_original < len)
    orig_ds = TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(0, 10, (8,)))
    ext_ds = TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,)))
    train_dataset = ConcatDataset([orig_ds, ext_ds])
    setattr(train_dataset, "num_original", 8)

    val_dataset = TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,)))

    train_loader, val_loader = training._create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=4,
        primary_fraction=0.75,
        device=torch.device("cpu"),
        repeat_aug=True,
        repeat_aug_repeats=3,
    )

    # Should use batch_sampler (RepeatAugRatioBatchSampler)
    assert train_loader.batch_sampler is not None
    assert isinstance(train_loader.batch_sampler, RepeatAugRatioBatchSampler)

    # Verify batches have correct composition
    batches = list(train_loader.batch_sampler)
    k_primary = int(round(4 * 0.75))  # 3
    k_secondary = 1
    for batch in batches:
        assert len(batch) == 4
        n_primary = sum(1 for i in batch if i < 8)
        n_secondary = sum(1 for i in batch if i >= 8)
        assert n_primary == k_primary
        assert n_secondary == k_secondary


# ---------------------------------------------------------------------------
# Full training run with repeat_aug + external data
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    """Minimal model for fast training tests."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.shape[0], -1))


def test_train_runs_with_repeat_aug_and_external_data(monkeypatch):
    """Full training loop should complete when repeat_aug and mix_external are both enabled."""
    monkeypatch.setattr(
        training,
        "load_cached_dataset",
        lambda cfg: (
            torch.randn(16, 1, 1, 1),
            torch.randint(0, cfg.model.num_classes, (16,)),
            None,
            None,
        ),
    )

    def fake_split(images, labels, *args, mix_external=None, **kwargs):
        # Simulate original (8) + external (8) = 16 total
        orig = TensorDataset(images[:8], labels[:8])
        ext = TensorDataset(images[8:], labels[8:])
        train_ds = ConcatDataset([orig, ext])
        setattr(train_ds, "num_original", 8)
        val_ds = TensorDataset(images[:4], labels[:4])
        return train_ds, val_ds, None, None

    monkeypatch.setattr(training, "split_dataset", fake_split)
    monkeypatch.setattr(
        training,
        "build_deit3",
        lambda size, num_classes=10, drop_path_rate=None, image_size=224, use_flash_attention=False, init_values=None: _TinyModel(num_classes),
    )

    class FakeWandb:
        run = type("R", (), {"id": "fakeid"})
        config = type("C", (), {"update": lambda *a, **k: None})

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def watch(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def log_artifact(*args, **kwargs):
            return None

        @staticmethod
        def finish(*args, **kwargs):
            return None

    monkeypatch.setattr(training, "wandb", FakeWandb)

    # Track which sampler type was used
    sampler_used = {}

    original_create = training._create_dataloaders

    def capturing_create(*args, **kwargs):
        train_loader, val_loader = original_create(*args, **kwargs)
        bs = getattr(train_loader, "batch_sampler", None)
        s = getattr(train_loader, "sampler", None)
        if bs is not None:
            sampler_used["type"] = type(bs).__name__
        elif s is not None:
            sampler_used["type"] = type(s).__name__
        else:
            sampler_used["type"] = "shuffle"
        return train_loader, val_loader

    monkeypatch.setattr(training, "_create_dataloaders", capturing_create)

    cfg = training.Config()
    cfg.data.repeat_aug = True
    cfg.data.repeat_aug_repeats = 3
    cfg.data.mix_external = True
    cfg.training.epochs = 1
    cfg.training.batch_size = 4
    cfg.training.ema_enabled = False
    cfg.training.wandb_enabled = False
    cfg.training.compile_model = False

    # Must not raise
    training.train(cfg)

    assert sampler_used.get("type") == "RepeatAugRatioBatchSampler"
