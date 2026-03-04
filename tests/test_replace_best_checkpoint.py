"""Tests for replace-best-checkpoint behavior.

Verifies that when replace_best_checkpoint is enabled, we overwrite best.pt
on disk and delete the previous wandb artifact before logging the new one.
"""

import os

import torch
from torch.utils.data import TensorDataset

from digit_classifier import training
from digit_classifier.config import Config


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.shape[0], -1))


def test_replace_best_checkpoint_overwrites_on_disk(monkeypatch, tmp_path):
    """With replace_best_checkpoint=True, best.pt should be overwritten (only one file)."""
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

    def fake_split(images, labels, *args, **kwargs):
        return TensorDataset(images, labels), TensorDataset(images[:4], labels[:4]), None, None

    monkeypatch.setattr(training, "split_dataset", fake_split)
    monkeypatch.setattr(
        training,
        "build_deit3",
        lambda size, num_classes=10, drop_path_rate=None, image_size=224, use_flash_attention=False, init_values=None: _TinyModel(num_classes),
    )

    # Return improving val accuracy so we save multiple "bests"
    val_accs = [0.1, 0.5, 0.9]
    call_idx = [0]

    def fake_validate(*args, **kwargs):
        idx = min(call_idx[0], len(val_accs) - 1)
        call_idx[0] += 1
        return {"accuracy": val_accs[idx], "cross_entropy": 0.0, "f1_score": 0.0}

    monkeypatch.setattr(training, "validate", fake_validate)

    class FakeWandb:
        run = type("R", (), {"id": "test-replace", "entity": "test", "project": "test"})
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

    # Use tmp_path for checkpoints
    monkeypatch.chdir(tmp_path)
    os.makedirs("checkpoints/test-replace", exist_ok=True)

    cfg = Config()
    cfg.training.epochs = 3
    cfg.training.batch_size = 4
    cfg.training.ema_enabled = False
    cfg.training.wandb_enabled = False
    cfg.training.compile_model = False
    cfg.training.replace_best_checkpoint = True

    training.train(cfg)

    assert os.path.exists("checkpoints/local/best.pt")
    ckpt = torch.load("checkpoints/local/best.pt", map_location="cpu", weights_only=False)
    # Should have the best (epoch 3 with acc 0.9)
    assert ckpt["val_accuracy"] == 0.9
    assert ckpt["epoch"] == 3


def test_replace_best_checkpoint_deletes_previous_wandb_artifact(monkeypatch):
    """With replace_best_checkpoint=True and wandb enabled, delete should be called before logging."""
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

    def fake_split(images, labels, *args, **kwargs):
        return TensorDataset(images, labels), TensorDataset(images[:4], labels[:4]), None, None

    monkeypatch.setattr(training, "split_dataset", fake_split)
    monkeypatch.setattr(
        training,
        "build_deit3",
        lambda size, num_classes=10, drop_path_rate=None, image_size=224, use_flash_attention=False, init_values=None: _TinyModel(num_classes),
    )

    # Return improving val accuracy: first save at epoch 1, second at epoch 2
    val_accs = [0.1, 0.6, 0.95]
    call_idx = [0]

    def fake_validate(*args, **kwargs):
        idx = min(call_idx[0], len(val_accs) - 1)
        call_idx[0] += 1
        return {"accuracy": val_accs[idx], "cross_entropy": 0.0, "f1_score": 0.0}

    monkeypatch.setattr(training, "validate", fake_validate)

    log_artifact_calls = []
    delete_calls = []
    api_artifact_calls = []

    class FakeArtifact:
        def __init__(self, name, type="model", metadata=None, **kwargs):
            self.name = name
            self.type = type
            self.metadata = metadata or {}
            self._files = []

        def add_file(self, path):
            self._files.append(path)

        def delete(self, delete_aliases=False):
            delete_calls.append({"name": self.name})

    class FakeWandb:
        Artifact = FakeArtifact
        run = type("R", (), {"id": "test-wandb-replace", "entity": "test", "project": "test"})
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
        def log_artifact(artifact, aliases=None):
            log_artifact_calls.append({"name": artifact.name, "aliases": aliases})

        @staticmethod
        def finish(*args, **kwargs):
            return None

        @staticmethod
        def Api():
            class FakeApi:
                def artifact(self, path, type=None):
                    # Track that we attempted to fetch previous artifact (for delete)
                    api_artifact_calls.append(path)
                    raise Exception("artifact not found")

            return FakeApi()

    monkeypatch.setattr(training, "wandb", FakeWandb)

    cfg = Config()
    cfg.training.epochs = 3
    cfg.training.batch_size = 4
    cfg.training.ema_enabled = False
    cfg.training.wandb_enabled = True
    cfg.training.compile_model = False
    cfg.training.replace_best_checkpoint = True

    training.train(cfg)

    # We should have logged 3 times (epochs 1, 2, 3 each improve)
    assert len(log_artifact_calls) == 3
    # All should be model-best-{run_id}
    assert all("model-best-" in c["name"] for c in log_artifact_calls)
    # With replace_best_checkpoint, we attempt to fetch/delete before each save (1st fails, 2nd/3rd would delete)
    assert len(api_artifact_calls) == 3, "Should attempt to fetch previous artifact before each save"
    # Verify we logged with "best" alias
    assert all(c.get("aliases") == ["best"] for c in log_artifact_calls)


def test_accumulate_best_checkpoint_does_not_delete(monkeypatch):
    """With replace_best_checkpoint=False, we should not call delete (just log new versions)."""
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

    def fake_split(images, labels, *args, **kwargs):
        return TensorDataset(images, labels), TensorDataset(images[:4], labels[:4]), None, None

    monkeypatch.setattr(training, "split_dataset", fake_split)
    monkeypatch.setattr(
        training,
        "build_deit3",
        lambda size, num_classes=10, drop_path_rate=None, image_size=224, use_flash_attention=False, init_values=None: _TinyModel(num_classes),
    )

    val_accs = [0.1, 0.6, 0.95]
    call_idx = [0]

    def fake_validate(*args, **kwargs):
        idx = min(call_idx[0], len(val_accs) - 1)
        call_idx[0] += 1
        return {"accuracy": val_accs[idx], "cross_entropy": 0.0, "f1_score": 0.0}

    monkeypatch.setattr(training, "validate", fake_validate)

    api_artifact_called = [False]

    class FakeArtifact:
        def __init__(self, name, type="model", metadata=None, **kwargs):
            self.name = name
            self.metadata = metadata or {}

        def add_file(self, path):
            pass

    class FakeWandb:
        Artifact = FakeArtifact
        run = type("R", (), {"id": "test-accumulate", "entity": "test", "project": "test"})
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
        def log_artifact(artifact, aliases=None):
            pass

        @staticmethod
        def finish(*args, **kwargs):
            return None

        @staticmethod
        def Api():
            class FakeApi:
                def artifact(self, path, type=None):
                    api_artifact_called[0] = True
                    raise Exception("not found")

            return FakeApi()

    monkeypatch.setattr(training, "wandb", FakeWandb)

    cfg = Config()
    cfg.training.epochs = 3
    cfg.training.batch_size = 4
    cfg.training.ema_enabled = False
    cfg.training.wandb_enabled = True
    cfg.training.compile_model = False
    cfg.training.replace_best_checkpoint = False  # accumulate mode

    training.train(cfg)

    # We should NOT have tried to fetch/delete previous artifact
    assert not api_artifact_called[0]
