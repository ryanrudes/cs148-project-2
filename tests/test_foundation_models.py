from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import digit_classifier.evolve_prompt as evolve_prompt
import digit_classifier.foundation_models as foundation_models
from digit_classifier.__main__ import _build_parser


def test_parser_clip_zero_shot_and_dino_args():
    parser = _build_parser()

    clip_args = parser.parse_args(
        [
            "clip",
            "--zero-shot",
            "--repo",
            "openai/clip-vit-large-patch14",
            "--epochs",
            "3",
            "--save-checkpoints",
        ]
    )
    assert clip_args.command == "clip"
    assert clip_args.foundation_model_family == "clip"
    assert clip_args.zero_shot is True
    assert clip_args.epochs == 3
    assert clip_args.save_checkpoints is True
    assert clip_args.sweep_project == "mnist-in-the-wild-clip"

    dino_args = parser.parse_args(
        [
            "dino",
            "--repo",
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "--sweep-action",
            "create",
            "--sweep-count",
            "7",
        ]
    )
    assert dino_args.command == "dino"
    assert dino_args.foundation_model_family == "dino"
    assert not hasattr(dino_args, "zero_shot")
    assert dino_args.sweep_action == "create"
    assert dino_args.sweep_count == 7
    assert dino_args.sweep_project == "mnist-in-the-wild-dino"

    with pytest.raises(SystemExit):
        parser.parse_args(["dino", "--zero-shot"])


def test_foundation_model_lookup_and_metadata():
    clip_repos = foundation_models.list_foundation_model_repos(
        foundation_models.FoundationModelFamily.CLIP
    )
    dino_repos = foundation_models.list_foundation_model_repos(
        foundation_models.FoundationModelFamily.DINO
    )

    assert len(clip_repos) == 4
    assert len(dino_repos) == 10

    convnext_variants = []
    for architecture in foundation_models.FoundationModelArchitecture:
        metadata = architecture.value
        assert foundation_models.get_foundation_model(metadata.repo, metadata.family) is architecture

        cfg = foundation_models.FoundationModelConfig(
            model=architecture,
            family=metadata.family,
        )
        assert cfg.family == metadata.family
        assert cfg.patch_size == metadata.patch_size
        assert cfg.repo == metadata.repo

        if "CONVNEXT" in architecture.name:
            convnext_variants.append(architecture)

    assert convnext_variants
    assert all(architecture.value.patch_size is None for architecture in convnext_variants)

    with pytest.raises(ValueError):
        foundation_models.FoundationModelConfig(
            model=foundation_models.FoundationModelArchitecture.DINO_V3_VIT_B16,
            family=foundation_models.FoundationModelFamily.DINO,
            zero_shot=True,
        )


class _FakeProcessor:
    def __call__(self, images=None, return_tensors=None, **kwargs):
        batch_size = len(images)
        pixel_values = torch.arange(
            batch_size * 3 * 2 * 2,
            dtype=torch.float32,
        ).reshape(batch_size, 3, 2, 2)
        return {"pixel_values": pixel_values}


class _FakeClipModel:
    def __init__(self):
        self.vision_calls = 0
        self.projection_calls = 0

    def vision_model(self, pixel_values):
        self.vision_calls += 1
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(pooler_output=torch.full((batch_size, 4), 2.0))

    def visual_projection(self, image_latents):
        self.projection_calls += 1
        return image_latents + 1.0


class _FakeDinoModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, pixel_values):
        self.calls += 1
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(pooler_output=torch.full((batch_size, 5), 4.0))


def test_compute_features_dispatches_by_family(monkeypatch, tmp_path):
    images = torch.zeros((3, 3, 2, 2), dtype=torch.uint8)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        foundation_models,
        "load_mnist_in_the_wild",
        lambda cfg: (images, labels, None, None),
    )
    monkeypatch.setattr(
        foundation_models,
        "track",
        lambda iterable, description=None: iterable,
    )

    clip_cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        feature_batch_size=2,
    )
    clip_model = _FakeClipModel()
    clip_processor = _FakeProcessor()

    clip_latents, clip_features, clip_normalized, clip_labels = (
        foundation_models.compute_foundation_model_features(
            clip_model,
            clip_processor,
            clip_cfg,
            torch.device("cpu"),
        )
    )

    assert clip_model.vision_calls == 2
    assert clip_model.projection_calls == 2
    assert torch.allclose(clip_latents, torch.full((3, 4), 2.0))
    assert torch.allclose(clip_features, torch.full((3, 4), 3.0))
    assert torch.equal(clip_labels, labels)
    assert (tmp_path / "cache" / "foundation_model_features_clip_openai_clip-vit-base-patch32.npz").exists()
    assert torch.allclose(
        clip_normalized,
        torch.full((3, 4), 0.5),
        atol=1e-6,
    )

    dino_cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.DINO_V3_CONVNEXT_BASE,
        family=foundation_models.FoundationModelFamily.DINO,
        feature_batch_size=2,
    )
    dino_model = _FakeDinoModel()
    dino_processor = _FakeProcessor()

    dino_latents, dino_features, dino_normalized, dino_labels = (
        foundation_models.compute_foundation_model_features(
            dino_model,
            dino_processor,
            dino_cfg,
            torch.device("cpu"),
        )
    )

    assert dino_model.calls == 2
    assert torch.allclose(dino_latents, torch.full((3, 5), 4.0))
    assert torch.allclose(dino_features, dino_latents)
    assert torch.equal(dino_labels, labels)
    assert (
        tmp_path
        / "cache"
        / "foundation_model_features_dino_facebook_dinov3-convnext-base-pretrain-lvd1689m.npz"
    ).exists()
    assert torch.allclose(
        dino_normalized,
        torch.full((3, 5), 1 / (5 ** 0.5)),
        atol=1e-6,
    )


class _DummyLoadedModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def parameters(self):
        return []


def test_run_foundation_model_dispatch(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(foundation_models, "seed_everything", lambda seed: None)
    monkeypatch.setattr(
        foundation_models,
        "load_foundation_model",
        lambda model: (_DummyLoadedModel(), object()),
    )
    monkeypatch.setattr(foundation_models, "freeze", lambda model: calls.append("freeze"))
    monkeypatch.setattr(
        foundation_models,
        "run_clip_zero_shot",
        lambda model, processor, device, cfg: calls.append("zero_shot"),
    )
    monkeypatch.setattr(
        foundation_models,
        "run_foundation_model_fine_tuning",
        lambda model, processor, device, cfg, parent_run=None: calls.append("fine_tune"),
    )

    clip_cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        zero_shot=True,
    )
    foundation_models.run_foundation_model(clip_cfg)
    assert calls == ["freeze", "zero_shot"]

    calls.clear()
    dino_cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.DINO_V3_VIT_B16,
        family=foundation_models.FoundationModelFamily.DINO,
    )
    foundation_models.run_foundation_model(dino_cfg)
    assert calls == ["freeze", "fine_tune"]


def test_evolve_prompt_main_builds_clip_config(monkeypatch):
    captured = {}

    monkeypatch.setattr(evolve_prompt, "validate_evolution_config", lambda cfg: None)
    monkeypatch.setattr(
        evolve_prompt,
        "evolve_prompts",
        lambda cfg, clip_cfg: captured.setdefault("clip_cfg", clip_cfg),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve_prompt.py", "--clip-repo", "openai/clip-vit-base-patch16"],
    )

    evolve_prompt.main()

    clip_cfg = captured["clip_cfg"]
    assert isinstance(clip_cfg, foundation_models.FoundationModelConfig)
    assert clip_cfg.family == foundation_models.FoundationModelFamily.CLIP
    assert clip_cfg.repo == "openai/clip-vit-base-patch16"
