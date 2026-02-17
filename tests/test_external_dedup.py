"""Tests for NIST-like deduplication in :mod:`digit_classifier.external`."""

import torch

from digit_classifier.external import (
    ExternalDataset,
    deduplicate_nist_like_datasets,
    is_nist_like_dataset,
)


class _FakeDatasetObj:
    def __init__(self, imgs, labels):
        self._imgs = imgs
        self._labels = labels

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, idx):
        return self._imgs[idx], self._labels[idx]


class _FakeExt:
    """Minimal stand-in for :class:`ExternalOnDemandDataset`."""

    def __init__(self, dataset_enum, imgs, labels, preprocessor=lambda x: x):
        self.dataset = dataset_enum
        self.dataset_obj = _FakeDatasetObj(imgs, labels)
        self.indices = list(range(len(imgs)))
        self.preprocessor = preprocessor


def test_dedup_across_externals():
    """Duplicate images across two NIST-like datasets should be removed."""
    X = torch.rand(1, 8, 8)
    Y = torch.rand(1, 8, 8)

    ext1 = _FakeExt(ExternalDataset.MNIST_TRAIN, [X, Y], [1, 2])
    ext2 = _FakeExt(ExternalDataset.EMNIST_TRAIN, [X, torch.rand(1, 8, 8)], [3, 4])

    removed = deduplicate_nist_like_datasets([ext1, ext2])
    assert removed == 1
    assert ext1.indices == [0, 1]
    assert ext2.indices == [1]


def test_dedup_within_single_dataset():
    """Duplicate images within the same dataset should be removed."""
    X = torch.rand(1, 8, 8)

    ext = _FakeExt(ExternalDataset.MNIST_TRAIN, [X, torch.rand(1, 8, 8), X], [1, 2, 3])

    removed = deduplicate_nist_like_datasets([ext])
    assert removed == 1
    assert ext.indices == [0, 1]


def test_non_nist_like_untouched():
    X = torch.rand(1, 8, 8)
    ext = _FakeExt(ExternalDataset.SVHN_TRAIN, [X], [1])

    removed = deduplicate_nist_like_datasets([ext])
    assert removed == 0
    assert ext.indices == [0]


def test_semeion_not_nist_like():
    assert not is_nist_like_dataset(ExternalDataset.SEMEION)

    ext = _FakeExt(ExternalDataset.SEMEION, [torch.rand(1, 8, 8)], [1])
    removed = deduplicate_nist_like_datasets([ext])
    assert removed == 0
