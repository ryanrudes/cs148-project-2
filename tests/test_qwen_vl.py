from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import digit_classifier.__main__ as cli
import digit_classifier.qwen_evolve as qwen_evolve
import digit_classifier.qwen_vl as qwen_vl
from digit_classifier.__main__ import _build_parser


def test_parser_qwen_args_and_validation():
    parser = _build_parser()

    args = parser.parse_args(
        [
            "qwen",
            "--prompt",
            "Focus on the overall digit shape.",
            "--dataset",
            "mnist",
        ]
    )
    cli._validate_args(parser, args)
    assert args.command == "qwen"
    assert args.repo == qwen_vl.DEFAULT_QWEN_VL_REPO
    assert args.system_prompt is None

    args = parser.parse_args(
        [
            "qwen",
            "--prompt",
            "Focus on the overall digit shape.",
            "--dataset",
            "pareidolia",
            "--test-dataset",
            "dataset_out",
        ]
    )
    cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "qwen",
            "--prompt",
            "Focus on the overall digit shape.",
            "--dataset",
            "pareidolia",
        ]
    )
    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)

    args = parser.parse_args(
        [
            "qwen",
            "--prompt",
            "Focus on the overall digit shape.",
            "--dataset",
            "mnist",
            "--test-dataset",
            "dataset_out",
        ]
    )
    with pytest.raises(SystemExit):
        cli._validate_args(parser, args)


def test_parser_qwen_evolve_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "qwen-evolve",
            "--repo",
            qwen_vl.DEFAULT_QWEN_VL_REPO,
            "--device",
            "cpu",
            "--population-size",
            "10",
            "--generations",
            "3",
            "--elite-size",
            "2",
            "--children-per-generation",
            "4",
            "--batch-size",
            "8",
            "--system-prompt",
            "You are a custom digit classifier.",
            "--evolution-samples-per-class",
            "12",
            "--holdout-samples-per-class",
            "9",
        ]
    )
    cli._validate_args(parser, args)
    assert args.command == "qwen-evolve"
    assert args.population_size == 10
    assert args.generations == 3
    assert args.elite_size == 2
    assert args.children_per_generation == 4
    assert args.batch_size == 8
    assert args.system_prompt == "You are a custom digit classifier."
    assert args.evolution_samples_per_class == 12
    assert args.holdout_samples_per_class == 9


def test_parser_tokenize_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "tokenize",
            "digit seven",
            "--repo",
            qwen_vl.DEFAULT_QWEN_VL_REPO,
            "--add-special-tokens",
        ]
    )
    cli._validate_args(parser, args)
    assert args.command == "tokenize"
    assert args.text == "digit seven"
    assert args.add_special_tokens is True


def test_handle_qwen_dispatches_to_zero_shot(monkeypatch):
    captured = {}

    def fake_run_qwen_zero_shot(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(qwen_vl, "run_qwen_zero_shot", fake_run_qwen_zero_shot)

    args = cli.argparse.Namespace(
        repo=qwen_vl.DEFAULT_QWEN_VL_REPO,
        prompt="Focus on the overall shape.",
        dataset="mnist",
        device="cpu",
        batch_size=8,
        system_prompt="You are a custom digit classifier.",
        test_dataset=None,
    )
    cli._handle_qwen(args)

    assert captured == {
        "repo": qwen_vl.DEFAULT_QWEN_VL_REPO,
        "prompt": "Focus on the overall shape.",
        "dataset": "mnist",
        "device": "cpu",
        "batch_size": 8,
        "system_prompt": "You are a custom digit classifier.",
        "test_dataset_path": None,
    }


def test_handle_qwen_evolve_dispatches_batch_size(monkeypatch):
    captured = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured["cfg_kwargs"] = kwargs

    def fake_run_qwen_prompt_evolution(cfg):
        captured["cfg_type"] = type(cfg).__name__

    monkeypatch.setattr("digit_classifier.qwen_evolve.QwenEvolutionConfig", _FakeConfig)
    monkeypatch.setattr("digit_classifier.qwen_evolve.run_qwen_prompt_evolution", fake_run_qwen_prompt_evolution)

    args = cli.argparse.Namespace(
        repo=qwen_vl.DEFAULT_QWEN_VL_REPO,
        device="cuda",
        llm_model="Qwen/Qwen2.5-3B-Instruct",
        population_size=10,
        generations=3,
        elite_size=2,
        children_per_generation=4,
        batch_size=16,
        system_prompt="You are a custom digit classifier.",
        evolution_samples_per_class=12,
        holdout_samples_per_class=8,
        random_seed=0,
        cache_dir="cache/test",
    )
    cli._handle_qwen_evolve(args)

    assert captured["cfg_kwargs"]["batch_size"] == 16
    assert captured["cfg_kwargs"]["system_prompt"] == "You are a custom digit classifier."
    assert captured["cfg_kwargs"]["evolution_samples_per_class"] == 12
    assert captured["cfg_kwargs"]["holdout_samples_per_class"] == 8


def test_handle_tokenize_dispatches(monkeypatch):
    captured = {}

    def fake_run_qwen_tokenize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(qwen_vl, "run_qwen_tokenize", fake_run_qwen_tokenize)

    args = cli.argparse.Namespace(
        repo=qwen_vl.DEFAULT_QWEN_VL_REPO,
        text="digit 7",
        add_special_tokens=False,
    )
    cli._handle_tokenize(args)

    assert captured == {
        "repo": qwen_vl.DEFAULT_QWEN_VL_REPO,
        "text": "digit 7",
        "add_special_tokens": False,
    }


def test_parse_qwen_digit_response_is_strict():
    assert qwen_vl.parse_qwen_digit_response("5") == 5
    assert qwen_vl.parse_qwen_digit_response(" 7\n") == 7
    assert qwen_vl.parse_qwen_digit_response("10") is None
    assert qwen_vl.parse_qwen_digit_response("digit 3") is None
    assert qwen_vl.parse_qwen_digit_response("") is None


def test_compute_qwen_eval_metrics_counts_invalids():
    metrics = qwen_vl.compute_qwen_eval_metrics(
        [0, None, 2, 1],
        [0, 1, 2, 3],
    )
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.parse_rate == pytest.approx(0.75)
    assert metrics.invalid_count == 1
    assert metrics.num_samples == 4


def test_build_qwen_conversation_with_custom_system_prompt():
    conversation = qwen_vl.build_qwen_conversation_with_system_prompt(
        instruction_body="Focus on the overall shape.",
        system_prompt="You are a custom digit classifier.",
    )

    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"][0]["text"] == "You are a custom digit classifier."
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"][1]["text"] == "Focus on the overall shape."


def test_inspect_qwen_tokenization_uses_loaded_tokenizer(monkeypatch):
    class _FakeTokenizer:
        vocab_size = 100
        special_tokens_map = {"eos_token": "</s>"}

        def __len__(self):
            return 105

        def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
            assert text == "digit 7"
            assert add_special_tokens is True
            return {"input_ids": [11, 22]}

        def convert_ids_to_tokens(self, input_ids):
            assert input_ids == [11, 22]
            return ["digit", "7"]

        def decode(self, input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return "digit 7"

    monkeypatch.setattr(qwen_vl, "load_qwen_tokenizer", lambda repo: _FakeTokenizer())

    result = qwen_vl.inspect_qwen_tokenization(
        repo=qwen_vl.DEFAULT_QWEN_VL_REPO,
        text="digit 7",
        add_special_tokens=True,
    )

    assert result.tokenizer_class == "_FakeTokenizer"
    assert result.vocab_size == 100
    assert result.tokenizer_length == 105
    assert result.input_ids == [11, 22]
    assert result.tokens == ["digit", "7"]
    assert result.decoded == "digit 7"


def test_rgb_pil_from_numpy_image_supports_chw_and_hwc():
    chw = np.zeros((3, 4, 5), dtype=np.uint8)
    chw[0] = 255
    hwc = np.moveaxis(chw, 0, -1)

    chw_image = qwen_vl._rgb_pil_from_numpy_image(chw)
    hwc_image = qwen_vl._rgb_pil_from_numpy_image(hwc)

    assert chw_image.mode == "RGB"
    assert hwc_image.mode == "RGB"
    assert chw_image.size == (5, 4)
    assert hwc_image.size == (5, 4)


def test_load_qwen_dataset_prefers_336_then_falls_back(tmp_path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    images_224 = np.zeros((2, 3, 2, 2), dtype=np.uint8)
    labels = np.array([1, 2], dtype=np.int64)
    np.savez(datasets_dir / "mnist_itw_rgb_224.npz", images=images_224, labels=labels)

    bundle = qwen_vl.load_qwen_dataset("mnist", datasets_dir=datasets_dir)
    assert bundle.source_path.endswith("mnist_itw_rgb_224.npz")
    assert bundle.num_samples == 2

    images_336 = np.ones((3, 3, 2, 2), dtype=np.uint8)
    np.savez(datasets_dir / "mnist_itw_rgb_336.npz", images=images_336, labels=np.array([0, 1, 2]))

    preferred_bundle = qwen_vl.load_qwen_dataset("mnist", datasets_dir=datasets_dir)
    assert preferred_bundle.source_path.endswith("mnist_itw_rgb_336.npz")
    assert preferred_bundle.num_samples == 3


def test_load_qwen_pareidolia_dataset_reports_skips(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)

    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps({"image_path": "images/sample.png", "digit": 4}) + "\n"
        + json.dumps({"image_path": "images/missing.png", "digit": 7}) + "\n",
        encoding="utf-8",
    )

    bundle = qwen_vl.load_qwen_dataset("pareidolia", test_dataset_path=tmp_path)
    assert bundle.dataset_key == "pareidolia"
    assert bundle.num_samples == 1
    assert bundle.skipped_count == 1


def test_predict_qwen_digits_updates_progress(monkeypatch):
    bundle = qwen_vl.QwenDatasetBundle(
        dataset_key="mnist_in_the_wild",
        display_name="MNIST-in-the-Wild",
        labels=np.array([0, 1, 2], dtype=np.int64),
        image_loader=lambda idx: Image.new("RGB", (2, 2), color=(idx, idx, idx)),
        source_path="datasets/mnist_itw_rgb_336.npz",
    )
    recorded_updates: list[dict[str, object]] = []

    class _FakeProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def refresh(self):
            return None

        def add_task(self, description, *, total, batch, total_batches):
            recorded_updates.append(
                {
                    "kind": "add_task",
                    "description": description,
                    "total": total,
                    "batch": batch,
                    "total_batches": total_batches,
                }
            )
            return 1

        def update(self, task_id, **kwargs):
            payload = {"kind": "update", "task_id": task_id}
            payload.update(kwargs)
            recorded_updates.append(payload)

    monkeypatch.setattr(qwen_vl, "_create_qwen_eval_progress", lambda disable=False: _FakeProgress())
    monkeypatch.setattr(
        qwen_vl,
        "generate_qwen_responses",
        lambda *args, **kwargs: ["0"] * len(kwargs.get("images", args[3])),
    )

    predictions, raw = qwen_vl.predict_qwen_digits(
        model=object(),
        processor=object(),
        device=torch.device("cpu"),
        dataset_bundle=bundle,
        instruction_body="Focus on the overall shape.",
        batch_size=2,
        show_progress=True,
    )

    assert predictions == [0, 0, 0]
    assert raw == ["0", "0", "0"]
    assert recorded_updates[0] == {
        "kind": "add_task",
        "description": "Evaluating MNIST-in-the-Wild",
        "total": 3,
        "batch": 0,
        "total_batches": 2,
    }
    advances = [update["advance"] for update in recorded_updates if update["kind"] == "update" and "advance" in update]
    assert advances == [2, 1]


def test_build_qwen_split_indices_is_reproducible_and_balanced():
    labels = np.repeat(np.arange(10), 100)
    first = qwen_evolve.build_qwen_split_indices(labels, random_seed=7)
    second = qwen_evolve.build_qwen_split_indices(labels, random_seed=7)

    assert first == second
    assert len(first["evolution_indices"]) == 500
    assert len(first["holdout_indices"]) == 500

    evolution_labels = labels[first["evolution_indices"]]
    holdout_labels = labels[first["holdout_indices"]]
    assert all(int((evolution_labels == digit).sum()) == 50 for digit in range(10))
    assert all(int((holdout_labels == digit).sum()) == 50 for digit in range(10))


def test_build_qwen_split_indices_supports_custom_split_sizes():
    labels = np.repeat(np.arange(10), 40)
    split = qwen_evolve.build_qwen_split_indices(
        labels,
        random_seed=11,
        evolution_samples_per_class=7,
        holdout_samples_per_class=5,
    )

    assert len(split["evolution_indices"]) == 70
    assert len(split["holdout_indices"]) == 50
    evolution_labels = labels[split["evolution_indices"]]
    holdout_labels = labels[split["holdout_indices"]]
    assert all(int((evolution_labels == digit).sum()) == 7 for digit in range(10))
    assert all(int((holdout_labels == digit).sum()) == 5 for digit in range(10))


def test_validate_qwen_candidate_prompt_rejects_output_override():
    with pytest.raises(ValueError):
        qwen_evolve.validate_qwen_candidate_prompt(
            "Respond with exactly one digit from 0-9 after you inspect the shape."
        )

    normalized = qwen_evolve.validate_qwen_candidate_prompt(
        "Focus on the global shape and overall geometry of the arrangement."
    )
    assert normalized == "Focus on the global shape and overall geometry of the arrangement."


def test_rank_qwen_prompt_scores_uses_accuracy_then_parse_rate_then_text():
    scores = [
        qwen_evolve.QwenPromptScore("b prompt", accuracy=0.9, parse_rate=0.8, invalid_count=2, num_samples=10),
        qwen_evolve.QwenPromptScore("a prompt", accuracy=0.9, parse_rate=0.8, invalid_count=2, num_samples=10),
        qwen_evolve.QwenPromptScore("c prompt", accuracy=0.9, parse_rate=0.9, invalid_count=1, num_samples=10),
    ]
    ranked = qwen_evolve.rank_qwen_prompt_scores(scores)
    assert [score.prompt for score in ranked] == ["c prompt", "a prompt", "b prompt"]


def test_evaluate_qwen_prompt_population_uses_prediction_cache(monkeypatch):
    bundle = qwen_vl.QwenDatasetBundle(
        dataset_key="mnist_in_the_wild",
        display_name="MNIST-in-the-Wild",
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
        image_loader=lambda idx: Image.new("RGB", (2, 2)),
        source_path="datasets/mnist_itw_rgb_336.npz",
    )
    calls = {"count": 0}

    def fake_evaluate(*args, **kwargs):
        calls["count"] += 1
        labels = bundle.labels[np.asarray(kwargs["indices"], dtype=np.int64)]
        predictions = [int(label) for label in labels]
        metrics = qwen_vl.compute_qwen_eval_metrics(predictions, labels)
        return metrics, predictions, [str(pred) for pred in predictions]

    monkeypatch.setattr(qwen_evolve, "evaluate_qwen_zero_shot_prompt", fake_evaluate)

    cache: dict[tuple[str, str], list[int | None]] = {}
    prompts = ["prompt one for global shape", "prompt two for silhouette focus"]
    indices = [0, 1, 2, 3]

    first = qwen_evolve.evaluate_qwen_prompt_population(
        prompts,
        model=object(),
        processor=object(),
        device=torch.device("cpu"),
        dataset_bundle=bundle,
        split_name="evolution",
        indices=indices,
        prediction_cache=cache,
        system_prompt=qwen_vl.QWEN_SYSTEM_PROMPT,
        batch_size=2,
        max_new_tokens=8,
    )
    second = qwen_evolve.evaluate_qwen_prompt_population(
        prompts,
        model=object(),
        processor=object(),
        device=torch.device("cpu"),
        dataset_bundle=bundle,
        split_name="evolution",
        indices=indices,
        prediction_cache=cache,
        system_prompt=qwen_vl.QWEN_SYSTEM_PROMPT,
        batch_size=2,
        max_new_tokens=8,
    )

    assert calls["count"] == 2
    assert first.cache_hits == 0
    assert first.cache_misses == 2
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert [score.accuracy for score in first.scores] == [1.0, 1.0]
    assert [score.accuracy for score in second.scores] == [1.0, 1.0]


def test_evaluate_qwen_prompt_population_updates_progress(monkeypatch):
    bundle = qwen_vl.QwenDatasetBundle(
        dataset_key="mnist_in_the_wild",
        display_name="MNIST-in-the-Wild",
        labels=np.array([0, 1, 2, 3], dtype=np.int64),
        image_loader=lambda idx: Image.new("RGB", (2, 2)),
        source_path="datasets/mnist_itw_rgb_336.npz",
    )
    recorded_updates: list[dict[str, object]] = []

    class _FakeProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def refresh(self):
            return None

        def add_task(self, description, *, total, cache_hits, cache_misses):
            recorded_updates.append(
                {
                    "kind": "add_task",
                    "description": description,
                    "total": total,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                }
            )
            return 1

        def update(self, task_id, **kwargs):
            payload = {"kind": "update", "task_id": task_id}
            payload.update(kwargs)
            recorded_updates.append(payload)

    def fake_evaluate(*args, **kwargs):
        labels = bundle.labels[np.asarray(kwargs["indices"], dtype=np.int64)]
        predictions = [int(label) for label in labels]
        metrics = qwen_vl.compute_qwen_eval_metrics(predictions, labels)
        return metrics, predictions, [str(pred) for pred in predictions]

    monkeypatch.setattr(qwen_evolve, "_create_qwen_population_progress", lambda disable=False: _FakeProgress())
    monkeypatch.setattr(qwen_evolve, "evaluate_qwen_zero_shot_prompt", fake_evaluate)

    result = qwen_evolve.evaluate_qwen_prompt_population(
        ["prompt one for global shape", "prompt two for silhouette focus"],
        model=object(),
        processor=object(),
        device=torch.device("cpu"),
        dataset_bundle=bundle,
        split_name="evolution",
        indices=[0, 1, 2, 3],
        prediction_cache={},
        system_prompt=qwen_vl.QWEN_SYSTEM_PROMPT,
        batch_size=2,
        max_new_tokens=8,
        show_progress=True,
    )

    assert result.cache_hits == 0
    assert result.cache_misses == 2
    assert result.num_prompts == 2
    assert recorded_updates[0] == {
        "kind": "add_task",
        "description": "Scoring evolution prompts",
        "total": 2,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    advances = [update["advance"] for update in recorded_updates if update["kind"] == "update" and "advance" in update]
    assert advances == [1, 1]


def test_run_qwen_prompt_evolution_writes_artifacts(monkeypatch, tmp_path):
    labels = np.repeat(np.arange(10), 100)
    bundle = qwen_vl.QwenDatasetBundle(
        dataset_key="mnist_in_the_wild",
        display_name="MNIST-in-the-Wild",
        labels=labels,
        image_loader=lambda idx: Image.new("RGB", (2, 2)),
        source_path="datasets/mnist_itw_rgb_336.npz",
    )

    class _FakePromptLLM:
        def __init__(self, *args, **kwargs):
            pass

        def generate_items(self, system_prompt, user_prompt, extractor, *, shuffle_rng=None):
            items = extractor(
                "Focus on the global outline and dominant geometry.\n"
                "Concentrate on the silhouette and major structural cues."
            )
            if shuffle_rng is not None:
                shuffle_rng.shuffle(items)
            return items

    def fake_load_model(repo, device="auto"):
        return object(), object(), torch.device("cpu")

    def fake_load_dataset(dataset, **kwargs):
        assert dataset == "mnist"
        return bundle

    def fake_evaluate(
        model,
        processor,
        device,
        dataset_bundle,
        prompt,
        *,
        system_prompt,
        indices,
        batch_size,
        max_new_tokens,
    ):
        assert system_prompt == qwen_vl.QWEN_SYSTEM_PROMPT
        label_subset = dataset_bundle.labels[np.asarray(indices, dtype=np.int64)]
        if "global" in prompt.lower():
            predictions = [int(label) for label in label_subset]
        elif "silhouette" in prompt.lower():
            predictions = [None for _ in label_subset]
        else:
            predictions = [0 for _ in label_subset]
        metrics = qwen_vl.compute_qwen_eval_metrics(predictions, label_subset)
        raw = ["" if pred is None else str(pred) for pred in predictions]
        return metrics, predictions, raw

    monkeypatch.setattr(qwen_evolve, "PromptLLM", _FakePromptLLM)
    monkeypatch.setattr(qwen_evolve, "load_qwen_vl_model", fake_load_model)
    monkeypatch.setattr(qwen_evolve, "load_qwen_dataset", fake_load_dataset)
    monkeypatch.setattr(qwen_evolve, "evaluate_qwen_zero_shot_prompt", fake_evaluate)
    monkeypatch.setattr(qwen_evolve.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(qwen_evolve.console, "rule", lambda *args, **kwargs: None)

    cfg = qwen_evolve.QwenEvolutionConfig(
        population_size=4,
        generations=2,
        elite_size=2,
        children_per_generation=2,
        cache_dir=str(tmp_path / "cache"),
        random_seed=3,
    )

    result = qwen_evolve.run_qwen_prompt_evolution(cfg)

    assert result["best_prompt"] is not None
    assert (tmp_path / "cache" / "generation_000.json").exists()
    assert (tmp_path / "cache" / "generation_001.json").exists()
    summary = json.loads((tmp_path / "cache" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["repo"] == qwen_vl.DEFAULT_QWEN_VL_REPO
    assert summary["best_prompt"] == result["best_prompt"]
    assert len(summary["generations"]) == 2
    assert summary["final_holdout_ranking"][0]["prompt"] == result["best_prompt"]
