from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import digit_classifier.evolve_prompt as evolve_prompt
import digit_classifier.foundation_models as foundation_models
import digit_classifier.__main__ as cli
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
            "facebook/dino-vitb16",
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


def test_parser_clip_prompt_ablation_args_and_validation(tmp_path):
    parser = _build_parser()

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a photo of the digit {digit}\n", encoding="utf-8")

    args = parser.parse_args(
        [
            "clip",
            "--zero-shot",
            "--ablate-prompts",
            "--prompt-preset",
            "photo_digit",
            "--prompt-preset",
            "current",
            "--prompt-file",
            str(prompt_file),
            "--ablation-datasets",
            "both",
            "--test-dataset",
            "dataset_out",
            "--save-prompt-ablation",
            "ablation.json",
        ]
    )
    cli._validate_args(parser, args)
    assert args.ablate_prompts is True
    assert args.prompt_preset == ["photo_digit", "current"]
    assert args.prompt_file == str(prompt_file)
    assert args.ablation_datasets == "both"
    assert args.save_prompt_ablation == "ablation.json"

    with pytest.raises(SystemExit):
        parser.parse_args(["dino", "--ablate-prompts"])

    args = parser.parse_args(["clip", "--ablate-prompts"])
    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "clip",
            "--zero-shot",
            "--ablate-prompts",
            "--ablation-datasets",
            "both",
        ]
    )
    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)


def test_parser_finetune_shortcut_args():
    parser = _build_parser()

    args = parser.parse_args(
        [
            "finetune",
            "--repo",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "--project",
            "mnist-shortcut",
            "--device",
            "cuda",
        ]
    )
    assert args.command == "finetune"
    assert args.repo == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert args.project == "mnist-shortcut"
    assert args.device == "cuda"

    with pytest.raises(SystemExit):
        parser.parse_args(["finetune", "--repo", "facebook/dinov3-vitb16-pretrain-lvd1689m"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "finetune",
                "--repo",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "--project",
                "mnist-shortcut",
                "--epochs",
                "250",
            ]
        )


def test_parser_latent_visualize_compare_mode_requires_compare_args():
    parser = _build_parser()

    args = parser.parse_args(["latent-visualize", "--mode", "clip_over_dino"])
    assert args.command == "latent-visualize"
    assert args.mode == "clip_over_dino"

    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "latent-visualize",
            "--mode",
            "dino_over_clip",
            "--clip-repo",
            "openai/clip-vit-base-patch32",
            "--dino-repo",
            "facebook/dino-vitb16",
            "--clip-checkpoint",
            "clip-head.pt",
            "--dino-checkpoint",
            "dino-head.pt",
        ]
    )
    cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "latent-visualize",
            "--mode",
            "clip_over_dino",
            "--clip-repo",
            "openai/clip-vit-base-patch32",
            "--dino-repo",
            "facebook/dino-vitb16",
            "--clip-oof-bundle",
            "clip-oof.npz",
            "--dino-oof-bundle",
            "dino-oof.npz",
        ]
    )
    cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "latent-visualize",
            "--enable-comparison-modes",
        ]
    )
    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "latent-visualize",
            "--enable-comparison-modes",
            "--clip-repo",
            "openai/clip-vit-base-patch32",
            "--dino-repo",
            "facebook/dino-vitb16",
            "--clip-checkpoint",
            "clip-head.pt",
            "--dino-oof-bundle",
            "dino-oof.npz",
        ]
    )
    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)


def test_foundation_model_lookup_and_metadata():
    clip_repos = foundation_models.list_foundation_model_repos(
        foundation_models.FoundationModelFamily.CLIP
    )
    dino_repos = foundation_models.list_foundation_model_repos(
        foundation_models.FoundationModelFamily.DINO
    )

    assert len(clip_repos) == 4
    assert len(dino_repos) == 22

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
    assert foundation_models.FoundationModelArchitecture.DINO_V1_VIT_B8.value.patch_size == 8
    assert (
        foundation_models.FoundationModelArchitecture.DINO_V1_VIT_S16.value.repo
        == "facebook/dino-vits16"
    )
    assert foundation_models.FoundationModelArchitecture.DINO_V2_GIANT.value.patch_size == 14
    assert (
        foundation_models.FoundationModelArchitecture.DINO_V2_REG_GIANT.value.repo
        == "facebook/dinov2-with-registers-giant"
    )

    with pytest.raises(ValueError):
        foundation_models.FoundationModelConfig(
            model=foundation_models.FoundationModelArchitecture.DINO_V3_VIT_B16,
            family=foundation_models.FoundationModelFamily.DINO,
            zero_shot=True,
        )


def test_zero_shot_prompt_definition_loading_and_validation(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n# comment\n"
        "a photo of the digit {digit}\n"
        "a real-world scene of the number {digit}\n",
        encoding="utf-8",
    )

    cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        zero_shot=True,
        ablate_prompts=True,
        prompt_presets=("photo_number", "photo_number"),
        prompt_file_path=str(prompt_file),
    )
    definitions = foundation_models.resolve_zero_shot_prompt_definitions(cfg)

    assert [definition.prompt_id for definition in definitions] == [
        "photo_number",
        "file_1",
        "file_2",
    ]
    assert definitions[0].template == foundation_models.CLIP_ZERO_SHOT_PROMPT_PRESETS["photo_number"]
    assert definitions[1].source == "file"
    assert definitions[2].template == "a real-world scene of the number {digit}"

    with pytest.raises(ValueError):
        foundation_models.validate_zero_shot_prompt_template(
            "a photo of digit five",
            source="test",
        )
    with pytest.raises(ValueError):
        foundation_models.validate_zero_shot_prompt_template(
            "digit {digit} and number {digit}",
            source="test",
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


class _FakeZeroShotModel:
    def __init__(self):
        self.logit_scale = torch.tensor(0.0)


def test_load_classifier_from_checkpoint_reconstructs_saved_head(tmp_path):
    torch.manual_seed(0)

    cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        head_type="deep_mlp",
        layer_norm=True,
    )
    classifier = foundation_models.build_classifier(cfg, input_dim=5).eval()
    inputs = torch.randn(4, 5)
    expected = classifier(inputs)

    checkpoint_path = tmp_path / "clip_head.pt"
    torch.save({"classifier_state_dict": classifier.state_dict()}, checkpoint_path)

    loaded = foundation_models.load_classifier_from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )
    actual = loaded(inputs)

    assert torch.allclose(actual, expected)


def test_run_clip_zero_shot_prompt_ablation_builtin_only_on_mnist(monkeypatch):
    cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        zero_shot=True,
        ablate_prompts=True,
    )

    normalized_image_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    prompt_ids = list(foundation_models.CLIP_ZERO_SHOT_PROMPT_PRESETS)
    encoded = torch.zeros((len(prompt_ids), 10, 2), dtype=torch.float32)
    for idx, prompt_id in enumerate(prompt_ids):
        if prompt_id == "current":
            encoded[idx, 0] = torch.tensor([1.0, 0.0])
            encoded[idx, 1] = torch.tensor([0.0, 1.0])
        elif prompt_id == "photo_digit":
            encoded[idx, 0] = torch.tensor([0.8, 0.2])
            encoded[idx, 1] = torch.tensor([0.2, 0.8])
        elif prompt_id == "photo_number":
            encoded[idx, 0] = torch.tensor([0.6, 0.4])
            encoded[idx, 1] = torch.tensor([0.4, 0.6])
        elif prompt_id == "objects_form":
            encoded[idx, 0] = torch.tensor([0.0, 1.0])
            encoded[idx, 1] = torch.tensor([1.0, 0.0])
        else:
            encoded[idx, 0] = torch.tensor([0.5, 0.5])
            encoded[idx, 1] = torch.tensor([0.5, 0.5])

    monkeypatch.setattr(
        foundation_models,
        "_load_zero_shot_dataset_features",
        lambda model, processor, cfg, device, dataset_key: (normalized_image_features, labels),
    )
    monkeypatch.setattr(
        foundation_models,
        "encode_zero_shot_prompt_definitions",
        lambda model, processor, device, prompt_definitions, text_batch_size: encoded,
    )
    monkeypatch.setattr(foundation_models.console, "print", lambda *args, **kwargs: None)

    result = foundation_models.run_clip_zero_shot_prompt_ablation(
        _FakeZeroShotModel(),
        object(),
        torch.device("cpu"),
        cfg,
    )

    assert result["datasets_evaluated"] == ["mnist_in_the_wild"]
    assert len(result["prompt_definitions"]) == len(foundation_models.CLIP_ZERO_SHOT_PROMPT_PRESETS)
    mnist_rows = result["per_dataset_results"]["mnist_in_the_wild"]
    assert len(mnist_rows) == len(foundation_models.CLIP_ZERO_SHOT_PROMPT_PRESETS)
    assert mnist_rows[0]["prompt_id"] == "current"
    assert mnist_rows[0]["is_best"] is True


def test_run_clip_zero_shot_prompt_ablation_builtins_plus_file_and_json(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "# comment\n"
        "a photo of the digit {digit}\n"
        "a real-world scene where objects form the digit {digit}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "prompt_ablation.json"
    cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        zero_shot=True,
        ablate_prompts=True,
        prompt_presets=("current",),
        prompt_file_path=str(prompt_file),
        ablation_datasets="both",
        test_dataset_path="dataset_out",
        save_prompt_ablation_path=str(output_path),
    )

    mnist_features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    pareidolia_features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset_features = {
        "mnist_in_the_wild": (mnist_features, labels),
        "pareidolia": (pareidolia_features, labels),
    }

    def fake_load_features(model, processor, cfg, device, dataset_key):
        return dataset_features[dataset_key]

    def fake_encode(model, processor, device, prompt_definitions, text_batch_size):
        encoded = torch.zeros((len(prompt_definitions), 10, 2), dtype=torch.float32)
        mapping = {
            "current": (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
            "file_1": (torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])),
            "file_2": (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])),
        }
        for idx, prompt_definition in enumerate(prompt_definitions):
            class_zero, class_one = mapping[prompt_definition.prompt_id]
            encoded[idx, 0] = class_zero
            encoded[idx, 1] = class_one
        return encoded

    monkeypatch.setattr(foundation_models, "_load_zero_shot_dataset_features", fake_load_features)
    monkeypatch.setattr(foundation_models, "encode_zero_shot_prompt_definitions", fake_encode)
    monkeypatch.setattr(foundation_models.console, "print", lambda *args, **kwargs: None)

    result = foundation_models.run_clip_zero_shot_prompt_ablation(
        _FakeZeroShotModel(),
        object(),
        torch.device("cpu"),
        cfg,
    )

    assert result["datasets_evaluated"] == ["mnist_in_the_wild", "pareidolia"]
    assert sorted(result["per_dataset_results"]) == ["mnist_in_the_wild", "pareidolia"]
    assert [definition.prompt_id for definition in result["prompt_definitions"]] == [
        "current",
        "file_1",
        "file_2",
    ]
    assert len(result["overall_ranking"]) == 3
    assert result["overall_ranking"][0]["prompt_id"] == "current"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["model_repo"] == cfg.repo
    assert payload["datasets_evaluated"] == ["mnist_in_the_wild", "pareidolia"]
    assert [entry["prompt_id"] for entry in payload["prompt_definitions"]] == [
        "current",
        "file_1",
        "file_2",
    ]
    assert sorted(payload["per_dataset_results"]) == ["mnist_in_the_wild", "pareidolia"]
    assert len(payload["overall_ranking"]) == 3


def test_repeated_split_plan_and_oof_bundle_round_trip(tmp_path):
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    dataset_hash = "dataset-hash"
    plan = foundation_models.build_repeated_stratified_split_plan(
        labels,
        n_folds=2,
        cv_repeats=3,
        seed=7,
        dataset_hash=dataset_hash,
    )
    assert plan["n_folds"] == 2
    assert plan["cv_repeats"] == 3
    assert len(plan["splits"]) == 6

    split_plan_path = tmp_path / "shared_splits.json"
    foundation_models.save_split_plan(plan, split_plan_path)
    loaded_plan = foundation_models.load_split_plan(split_plan_path)
    foundation_models.validate_split_plan(
        loaded_plan,
        n_samples=len(labels),
        dataset_hash=dataset_hash,
    )
    assert loaded_plan["plan_id"] == plan["plan_id"]

    cfg = foundation_models.FoundationModelConfig(
        model=foundation_models.FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
        family=foundation_models.FoundationModelFamily.CLIP,
        n_folds=2,
        cv_repeats=3,
    )
    mean_probs = np.tile(
        np.linspace(0.01, 0.1, 10, dtype=np.float32),
        (len(labels), 1),
    )
    mean_probs /= mean_probs.sum(axis=1, keepdims=True)
    counts = np.full(len(labels), 3, dtype=np.int64)

    oof_bundle_path = tmp_path / "clip_oof.npz"
    foundation_models.save_oof_prediction_bundle(
        oof_bundle_path,
        cfg=cfg,
        dataset_hash=dataset_hash,
        plan_id=plan["plan_id"],
        labels=labels,
        mean_probs=mean_probs,
        counts=counts,
    )
    bundle = foundation_models.load_oof_prediction_bundle(oof_bundle_path)
    foundation_models.validate_oof_prediction_bundle(
        bundle,
        expected_family=foundation_models.FoundationModelFamily.CLIP,
        expected_repo=cfg.repo,
        expected_dataset_hash=dataset_hash,
        expected_labels=labels,
        expected_plan_id=plan["plan_id"],
    )
    assert bundle["counts"].tolist() == counts.tolist()
    assert bundle["predictions"].shape == (len(labels),)


def test_plot_latent_space_exposes_multiple_embedding_views():
    from digit_classifier.latent_visualization import plot_latent_space

    fig = plot_latent_space(
        np.random.randn(6, 3),
        np.arange(6) % 3,
        reducer="tsne",
        reducer_kwargs={"perplexity": 2},
        open_browser=False,
        point_metadata={
            "clip_prediction": np.arange(6) % 3,
            "dino_prediction": (np.arange(6) + 1) % 3,
            "clip_true_label_probability": np.linspace(0.2, 0.8, 6),
            "dino_true_label_probability": np.linspace(0.1, 0.7, 6),
        },
        embedding_views={
            "clip": np.random.randn(6, 3),
            "dino": np.random.randn(6, 4),
        },
        initial_embedding_view="clip",
        embedding_view_labels={"clip": "CLIP", "dino": "DINO"},
        queryable_embedding_view="clip",
    )

    assert fig.layout.meta["initial_embedding_view"] == "clip"
    assert fig.layout.meta["queryable_embedding_view"] == "clip"
    assert sorted(fig.layout.meta["embedding_view_coords"].keys()) == ["clip", "dino"]
    assert "clip_true_label_probability" in fig.layout.meta["customdata_fields"]
    assert "dino_true_label_probability" in fig.layout.meta["customdata_fields"]


def test_plot_latent_space_side_panel_html_includes_density_heatmap_controls(tmp_path):
    from digit_classifier.latent_visualization import plot_latent_space

    output_path = tmp_path / "latent_space.html"
    result = plot_latent_space(
        np.random.randn(6, 3),
        np.arange(6) % 3,
        images=np.zeros((6, 8, 8), dtype=np.uint8),
        reducer="tsne",
        reducer_kwargs={"perplexity": 2},
        output_path=output_path,
        open_browser=False,
        point_metadata={
            "clip_prediction": np.arange(6) % 3,
            "dino_prediction": (np.arange(6) + 1) % 3,
            "clip_true_label_probability": np.linspace(0.2, 0.8, 6),
            "dino_true_label_probability": np.linspace(0.1, 0.7, 6),
        },
    )

    assert result == output_path
    html = output_path.read_text(encoding="utf-8")
    assert 'id="heatmap-toggle-btn"' in html
    assert 'id="point-color-mode-select"' in html
    assert 'id="heatmap-metric-select"' in html
    assert 'id="scatter-toggle-btn"' in html
    assert 'id="scatter-size-input"' in html
    assert 'id="home-scale-toggle-btn"' in html
    assert 'id="heatmap-resolution-input"' in html
    assert 'id="heatmap-opacity-input"' in html
    assert "function buildHeatmapTrace(indices)" in html
    assert "function buildScatterMarkerConfig(indices, opacity, size)" in html
    assert "function syncScatterSize()" in html
    assert "function buildInterpolatedLocalAdvantageGrid(sums, counts)" in html
    assert "function clampHeatmapOpacity(value)" in html
    assert "function buildRangeRelayout(indices)" in html
    assert "function getHomeReferenceIndices(fallbackIndices)" in html
    assert "function updateScatterVisibility()" in html
    assert "option value=\"true_label_gap\"" in html
    assert "option value=\"local_advantage\"" in html
    assert "CLIP - DINO p(true)" in html
    assert "CLIP advantage" in html
    assert "title: {text: 'Density'}" in html
    assert "resolveCurrentHomeRelayout()" in html


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


def test_handle_finetune_shortcut_resolves_family_and_runs_create_then_agent(monkeypatch):
    created_cfgs = []
    agent_cfgs = []

    def fake_create(cfg):
        created_cfgs.append(cfg)
        return "sweep-123"

    def fake_agent(cfg):
        agent_cfgs.append(cfg)

    monkeypatch.setattr(foundation_models, "create_foundation_model_sweep", fake_create)
    monkeypatch.setattr(foundation_models, "run_foundation_model_sweep_agent", fake_agent)

    dino_args = cli.argparse.Namespace(
        repo="facebook/dino-vitb16",
        project="dino-project",
        device="cpu",
    )
    cli._handle_finetune(dino_args)

    assert len(created_cfgs) == 1
    assert len(agent_cfgs) == 1
    created = created_cfgs[0]
    agent = agent_cfgs[0]
    assert created.family == foundation_models.FoundationModelFamily.DINO
    assert created.repo == "facebook/dino-vitb16"
    assert created.sweep_project == "dino-project"
    assert created.sweep_method == "bayes"
    assert created.n_folds == 5
    assert created.epochs == 250
    assert created.device == "cpu"
    assert created.early_stopping_patience == 10
    assert created.early_stopping_min_delta == 0.0
    assert created.sweep_count == 75
    assert created.sweep_id == "sweep-123"
    assert agent is created

    created_cfgs.clear()
    agent_cfgs.clear()

    clip_args = cli.argparse.Namespace(
        repo="openai/clip-vit-base-patch16",
        project="clip-project",
        device="mps",
    )
    cli._handle_finetune(clip_args)

    created = created_cfgs[0]
    assert created.family == foundation_models.FoundationModelFamily.CLIP
    assert created.repo == "openai/clip-vit-base-patch16"
    assert created.sweep_project == "clip-project"
    assert created.device == "mps"


def test_handle_latent_visualize_uses_default_clip_repo(monkeypatch):
    calls = []

    def fake_run_latent_visualization(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(foundation_models, "run_latent_visualization", fake_run_latent_visualization)

    args = cli.argparse.Namespace(
        clip_repo=None,
        device="cpu",
        mode="regular",
        enable_comparison_modes=False,
        dino_repo=None,
        clip_checkpoint=None,
        dino_checkpoint=None,
        clip_oof_bundle=None,
        dino_oof_bundle=None,
        feature_batch_size=16,
        batch_size=32,
    )
    cli._handle_latent_visualize(args)

    assert calls == [
        {
            "clip_repo": foundation_models.DEFAULT_MODELS[
                foundation_models.FoundationModelFamily.CLIP
            ].value.repo,
            "device": "cpu",
            "mode": "regular",
            "comparison_enabled": False,
            "dino_repo": None,
            "clip_checkpoint_path": None,
            "dino_checkpoint_path": None,
            "clip_oof_bundle_path": None,
            "dino_oof_bundle_path": None,
            "feature_batch_size": 16,
            "classifier_batch_size": 32,
        }
    ]


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
