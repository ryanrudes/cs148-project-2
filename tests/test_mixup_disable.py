"""Tests for dynamic loss switching when mixup is toggled on/off."""

import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader, Dataset

from digit_classifier.mixup import MixupCutmixApply, create_mixup_cutmix
from digit_classifier.training import select_train_criterion, train_epoch, _detect_device


class _TinyDataset(Dataset):
    def __init__(self, n: int = 8, channels: int = 1, size: int = 8, num_classes: int = 10):
        self.x = torch.randn(n, channels, size, size)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _make_model(channels: int = 1, size: int = 8, num_classes: int = 10) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(channels * size * size, num_classes))


def test_mixup_active_uses_soft_target_loss():
    mixup_fn = MixupCutmixApply(create_mixup_cutmix(
        num_classes=10, mixup_alpha=0.2, cutmix_alpha=1.0,
        prob=0.5, label_smoothing=0.0, mode="elem",
    ))
    crit = select_train_criterion(mixup_fn)
    assert isinstance(crit, SoftTargetCrossEntropy)


def test_mixup_disabled_uses_ce_loss():
    crit = select_train_criterion(None)
    assert isinstance(crit, nn.CrossEntropyLoss)


def test_train_epoch_runs_with_both_modes():
    device, _ = _detect_device()
    ds = _TinyDataset(n=8, channels=1, size=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = _make_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(enabled=False)

    mixup_fn = MixupCutmixApply(create_mixup_cutmix(
        num_classes=10, mixup_alpha=0.2, cutmix_alpha=1.0,
        prob=0.5, mode="elem",
    ))

    # With mixup
    crit = select_train_criterion(mixup_fn)
    train_epoch(model, loader, crit, optimizer, {}, scaler, device, mixup_fn=mixup_fn)

    # Without mixup
    crit2 = select_train_criterion(None)
    train_epoch(model, loader, crit2, optimizer, {}, scaler, device, mixup_fn=None)
