"""Tests for :func:`digit_classifier.training.compute_warm_restart_epochs`."""

from digit_classifier.training import compute_warm_restart_epochs


def test_basic():
    restarts = compute_warm_restart_epochs(10, 50, 2, 300)
    assert restarts == [60, 160]


def test_no_restarts():
    restarts = compute_warm_restart_epochs(5, 100, 2, 50)
    assert restarts == []
