"""Tests for :class:`digit_classifier.sampler.RatioBatchSampler`."""

import itertools

from digit_classifier.sampler import RatioBatchSampler


def test_basic_ratio():
    sampler = RatioBatchSampler(
        original_count=8, total_count=12, batch_size=4,
        primary_fraction=0.75, drop_last=True, seed=123,
    )
    batches = list(itertools.islice(iter(sampler), 10))
    assert all(len(b) == 4 for b in batches)
    for b in batches:
        primaries = sum(1 for idx in b if idx < 8)
        assert 2 <= primaries <= 4


def test_no_external():
    sampler = RatioBatchSampler(
        original_count=10, total_count=10, batch_size=5,
        primary_fraction=0.8, drop_last=True, seed=1,
    )
    batches = list(sampler)
    assert all(all(idx < 10 for idx in b) for b in batches)


def test_len_and_drop_last():
    sampler = RatioBatchSampler(
        original_count=5, total_count=13, batch_size=4,
        primary_fraction=0.5, drop_last=True,
    )
    assert len(sampler) == 13 // 4

    sampler2 = RatioBatchSampler(
        original_count=5, total_count=13, batch_size=4,
        primary_fraction=0.5, drop_last=False,
    )
    assert len(sampler2) == -(-13 // 4)
