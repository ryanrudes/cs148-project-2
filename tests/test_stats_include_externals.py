"""Test that mean/std is computed from original training images only."""

import torch

import digit_classifier.splitting as splitting_mod
from digit_classifier.external import ExternalDataset


class _FakeExternalOnDemand:
    """Stand-in for ``ExternalOnDemandDataset`` with constant-value images."""

    def __init__(self, dataset, color, size, max_samples=None, rnd=None):
        self.dataset = dataset
        self._imgs = [torch.full((1, 8, 8), 0.8), torch.full((1, 8, 8), 0.8)]
        self.indices = [0, 1]
        self.preprocessor = lambda x: x

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        return self._imgs[idx], 0


def test_stats_from_original_only(monkeypatch):
    """Even when external data is mixed in, mean/std comes from the original images."""
    images = torch.full((4, 1, 8, 8), 0.2)
    labels = torch.zeros(4, dtype=torch.long)

    monkeypatch.setattr(splitting_mod, "ExternalOnDemandDataset", _FakeExternalOnDemand)

    _, _, mean_t, _ = splitting_mod.split_dataset(
        images, labels,
        train_fraction=1.0,
        mix_external=True,
        external_fractions={ExternalDataset.SVHN_TRAIN: 1.0},
        color=True,
        size=8,
        seed=0,
    )

    # Mean should be 0.2 (from original images only), NOT influenced by external 0.8 values
    assert len(mean_t) == 1
    assert abs(mean_t[0] - 0.2) < 1e-6
