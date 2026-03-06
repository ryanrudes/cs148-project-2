import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import pytest

from digit_classifier import training
from digit_classifier.config import Config


class TinyModel(nn.Module):
    """Super‑lightweight model used in tests so training is almost instant."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # linear on a single scalar; input tensors will be flattened by the
        # custom ``forward`` below so there is no need for convolutions.
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.fc(x.view(b, -1))


@pytest.fixture(autouse=True)
def stub_dataset_and_model(monkeypatch):
    """Replace expensive pieces of ``train()`` with trivial stand‑ins.

    * ``load_cached_dataset`` returns a few random samples.
    * ``split_dataset`` hands back simple TensorDatasets so the rest of the
      code can build DataLoaders normally.
    * ``deit3_base_patch16_224`` is replaced with ``TinyModel`` to avoid
      pulling in the real DeiT implementation.
    * ``wandb`` calls are no‑ops to avoid network activity during tests.
    """

    monkeypatch.setattr(
        training,
        "load_cached_dataset",
        lambda cfg: (
            # use a single scalar per sample so TinyModel's linear layer
            # (in_features=1) is compatible with the flattened input
            torch.randn(8, 1, 1, 1),
            torch.randint(0, cfg.model.num_classes, (8,)),
            None,
            None,
        ),
    )

    def fake_split(images, labels, *args, **kwargs):
        ds = TensorDataset(images, labels)
        return ds, ds, None, None

    monkeypatch.setattr(training, "split_dataset", fake_split)
    monkeypatch.setattr(training, "build_deit3", lambda size, num_classes=10, drop_path_rate=None, image_size=224, use_flash_attention=False, init_values=None, patch_size=None, **kwargs: TinyModel(num_classes))

    # make wandb a no-op; training expects ``wandb.run.id`` later so
    # include a dummy ``run`` object with a constant identifier.
    class FakeWandb:
        run = type("R", (), {"id": "fakeid"})
        # ``config`` is an object that provides an ``update`` method in the
        # real library; tests just need a no-op stub.
        class _Cfg:
            @staticmethod
            def update(*args, **kwargs):
                return None

        config = _Cfg()

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

    yield


def test_train_respects_ema_flag(monkeypatch, tmp_path):
    """``train()`` should not create or update an EMA model when disabled."""

    # record what is passed to train_epoch for later assertions
    ema_arg = {}

    def fake_train_epoch(*args, ema=None, **kwargs):
        ema_arg["ema"] = ema
        # return dummy metrics so training loop progresses normally
        return {"cross_entropy": 0.0, "accuracy": 0.0}

    monkeypatch.setattr(training, "train_epoch", fake_train_epoch)

    # return perfect accuracy so that a "best model" checkpoint is written
    monkeypatch.setattr(training, "validate", lambda *args, **kwargs: {"cross_entropy": 0.0, "accuracy": 1.0})

    # intercept torch.save so we can inspect the checkpoint contents
    saved: dict[str, object] = {}

    def fake_save(obj, path):
        saved["obj"] = obj

    monkeypatch.setattr(torch, "save", fake_save)

    cfg = Config()
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.ema_enabled = False
    cfg.training.wandb_enabled = False

    # run training with the flag disabled
    training.train(cfg)

    assert "ema" in ema_arg and ema_arg["ema"] is None
    # no exception raised and EMA reference was None when disabled


def test_warm_restarts_flag(monkeypatch):
    """When ``warm_restarts`` is disabled we should not instantiate the
    warm-restart scheduler and instead fall back to a simple cosine cycle.

    This test hooks the two scheduler constructors so we can count which one
    the training code chooses.
    """

    calls = {"warm": 0, "cosine": 0}

    # preserve originals so we can delegate after counting
    orig_wr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    orig_cos = torch.optim.lr_scheduler.CosineAnnealingLR

    def fake_wr(opt, *args, **kwargs):
        calls["warm"] += 1
        return orig_wr(opt, *args, **kwargs)

    def fake_cosine(opt, *args, **kwargs):
        calls["cosine"] += 1
        return orig_cos(opt, *args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingWarmRestarts", fake_wr)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", fake_cosine)

    cfg = Config()
    cfg.training.epochs = 5
    cfg.training.batch_size = 2
    cfg.training.warmup_epochs = 1
    cfg.training.warm_restarts = False
    cfg.training.ema_enabled = False
    cfg.training.wandb_enabled = False
    cfg.training.compile_model = False

    training.train(cfg)

    assert calls["warm"] == 0
    assert calls["cosine"] == 1


def test_train_creates_ema_when_enabled(monkeypatch):
    """Verify that enabling EMA results in a non-``None`` object being
    passed through the pipeline and saved."""

    ema_arg = {}

    def fake_train_epoch(*args, ema=None, **kwargs):
        ema_arg["ema"] = ema
        return {"cross_entropy": 0.0, "accuracy": 0.0}

    monkeypatch.setattr(training, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(training, "validate", lambda *args, **kwargs: {"cross_entropy": 0.0, "accuracy": 0.0})

    cfg = Config()
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.ema_enabled = True
    cfg.training.wandb_enabled = False

    # no need to stub torch.save here; we simply care that ``ema`` is not None
    training.train(cfg)

    assert "ema" in ema_arg and ema_arg["ema"] is not None
