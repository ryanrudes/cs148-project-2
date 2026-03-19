from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torchvision.transforms.functional import to_pil_image

from digit_classifier.prompt_evolution_common import get_best_device

try:
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on local transformers build
    _QWEN_IMPORT_ERROR = exc
    AutoProcessor = None
    AutoTokenizer = None
    Qwen2_5_VLForConditionalGeneration = None
else:
    _QWEN_IMPORT_ERROR = None


log = logging.getLogger(__name__)
console = Console()

SUPPORTED_QWEN_VL_REPOS: tuple[str, ...] = ("Qwen/Qwen2.5-VL-3B-Instruct",)
DEFAULT_QWEN_VL_REPO = SUPPORTED_QWEN_VL_REPOS[0]
QWEN_SYSTEM_PROMPT = (
    "You are an expert visual digit classifier. Look at the provided image and determine which "
    "single decimal digit it depicts. Reply with exactly one digit character from 0 to 9 and nothing else."
)
DEFAULT_QWEN_BATCH_SIZE = 4
DEFAULT_QWEN_MAX_NEW_TOKENS = 8
QWEN_TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "*.model",
    "chat_template.json",
)


@dataclass(frozen=True)
class QwenDatasetBundle:
    dataset_key: str
    display_name: str
    labels: np.ndarray
    image_loader: Callable[[int], Image.Image]
    source_path: str
    skipped_count: int = 0

    @property
    def num_samples(self) -> int:
        return int(len(self.labels))


@dataclass(frozen=True)
class QwenEvalMetrics:
    accuracy: float
    parse_rate: float
    invalid_count: int
    num_samples: int


@dataclass(frozen=True)
class QwenTokenizationResult:
    repo: str
    tokenizer_class: str
    vocab_size: int
    tokenizer_length: int
    special_tokens_map: dict[str, Any]
    text: str
    add_special_tokens: bool
    input_ids: list[int]
    tokens: list[str]
    decoded: str


def list_qwen_vl_repos() -> list[str]:
    return list(SUPPORTED_QWEN_VL_REPOS)


def validate_qwen_vl_repo(repo: str) -> str:
    if repo not in SUPPORTED_QWEN_VL_REPOS:
        raise ValueError(
            f"Unsupported Qwen VLM repository: {repo}. Supported repos: {', '.join(SUPPORTED_QWEN_VL_REPOS)}"
        )
    return repo


def resolve_qwen_device(device: str) -> str:
    if device == "auto":
        return get_best_device(use_mps=True)
    return device


def _download_qwen_tokenizer_assets(repo: str) -> str:
    validate_qwen_vl_repo(repo)
    try:
        return snapshot_download(
            repo,
            local_files_only=True,
            allow_patterns=list(QWEN_TOKENIZER_ALLOW_PATTERNS),
        )
    except LocalEntryNotFoundError:
        return snapshot_download(
            repo,
            allow_patterns=list(QWEN_TOKENIZER_ALLOW_PATTERNS),
        )


def load_qwen_tokenizer(repo: str = DEFAULT_QWEN_VL_REPO) -> Any:
    if AutoTokenizer is None:  # pragma: no cover - environment dependent
        raise ImportError("Tokenizer inspection requires transformers AutoTokenizer support") from _QWEN_IMPORT_ERROR

    snapshot_path = _download_qwen_tokenizer_assets(repo)
    return AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)


def _get_qwen_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_qwen_vl_model(
    repo: str = DEFAULT_QWEN_VL_REPO,
    *,
    device: str = "auto",
) -> tuple[Any, Any, torch.device]:
    if _QWEN_IMPORT_ERROR is not None:  # pragma: no cover - environment dependent
        raise ImportError(
            "Qwen2.5-VL support requires a transformers build with Qwen2.5-VL classes"
        ) from _QWEN_IMPORT_ERROR

    validate_qwen_vl_repo(repo)
    resolved_device = resolve_qwen_device(device)
    torch_device = torch.device(resolved_device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=_get_qwen_torch_dtype(resolved_device),
    )
    processor = AutoProcessor.from_pretrained(repo)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model.to(torch_device)
    model.eval()
    return model, processor, torch_device


def parse_qwen_digit_response(text: str) -> int | None:
    normalized = text.strip()
    if len(normalized) != 1 or not normalized.isdigit():
        return None
    return int(normalized)


def compute_qwen_eval_metrics(
    predictions: Sequence[int | None],
    labels: Sequence[int] | np.ndarray | torch.Tensor,
) -> QwenEvalMetrics:
    label_array = np.asarray(labels, dtype=np.int64)
    if len(predictions) != len(label_array):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match label count ({len(label_array)})"
        )

    valid_predictions = np.array([pred for pred in predictions if pred is not None], dtype=np.int64)
    invalid_count = sum(pred is None for pred in predictions)
    correct = 0
    valid_index = 0
    for idx, pred in enumerate(predictions):
        if pred is None:
            continue
        if valid_predictions[valid_index] == label_array[idx]:
            correct += 1
        valid_index += 1

    num_samples = int(len(label_array))
    return QwenEvalMetrics(
        accuracy=(correct / num_samples) if num_samples else 0.0,
        parse_rate=((num_samples - invalid_count) / num_samples) if num_samples else 0.0,
        invalid_count=invalid_count,
        num_samples=num_samples,
    )


def _rgb_pil_from_numpy_image(image: np.ndarray) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {image.shape}")

    # torchvision expects numpy images as HWC but torch tensors as CHW.
    if image.shape[-1] in (1, 3, 4):
        return to_pil_image(image).convert("RGB")
    if image.shape[0] in (1, 3, 4):
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        return to_pil_image(tensor).convert("RGB")
    raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")


def _load_mnist_in_the_wild_qwen_bundle(
    *,
    datasets_dir: str | Path = "datasets",
) -> QwenDatasetBundle:
    datasets_path = Path(datasets_dir)
    preferred_path = datasets_path / "mnist_itw_rgb_336.npz"
    fallback_path = datasets_path / "mnist_itw_rgb_224.npz"
    if preferred_path.exists():
        dataset_path = preferred_path
    elif fallback_path.exists():
        dataset_path = fallback_path
    else:
        raise FileNotFoundError(
            "MNIST-in-the-Wild dataset not found. Expected datasets/mnist_itw_rgb_336.npz "
            "or datasets/mnist_itw_rgb_224.npz."
        )

    data = np.load(dataset_path, mmap_mode="r")
    images = data["images"]
    labels = np.asarray(data["labels"], dtype=np.int64)

    def image_loader(index: int) -> Image.Image:
        return _rgb_pil_from_numpy_image(images[index])

    return QwenDatasetBundle(
        dataset_key="mnist_in_the_wild",
        display_name="MNIST-in-the-Wild",
        labels=labels,
        image_loader=image_loader,
        source_path=str(dataset_path),
        skipped_count=0,
    )


def _load_pareidolia_qwen_bundle(
    test_dataset_path: str | Path,
) -> QwenDatasetBundle:
    from digit_classifier.pareidolia_dataset import PareidoliaTestDataset

    dataset = PareidoliaTestDataset(root_dir=test_dataset_path, preload=False)
    if not dataset.samples:
        raise ValueError(f"No pareidolia samples found under {test_dataset_path}")

    image_paths = [dataset.root / relative_path for relative_path, _ in dataset.samples]
    labels = np.array([label for _, label in dataset.samples], dtype=np.int64)

    def image_loader(index: int) -> Image.Image:
        with Image.open(image_paths[index]) as image:
            return image.convert("RGB")

    return QwenDatasetBundle(
        dataset_key="pareidolia",
        display_name="Pareidolia",
        labels=labels,
        image_loader=image_loader,
        source_path=str(Path(test_dataset_path)),
        skipped_count=dataset.skipped,
    )


def load_qwen_dataset(
    dataset: str,
    *,
    test_dataset_path: str | Path | None = None,
    datasets_dir: str | Path = "datasets",
) -> QwenDatasetBundle:
    if dataset == "mnist":
        return _load_mnist_in_the_wild_qwen_bundle(datasets_dir=datasets_dir)
    if dataset == "pareidolia":
        if test_dataset_path is None:
            raise ValueError("Pareidolia evaluation requires a test dataset path")
        return _load_pareidolia_qwen_bundle(test_dataset_path)
    raise ValueError(f"Unsupported Qwen dataset: {dataset}")


def build_qwen_conversation(instruction_body: str) -> list[dict[str, Any]]:
    normalized_instruction = instruction_body.strip()
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": normalized_instruction},
            ],
        },
    ]


def _prepare_qwen_inputs(
    processor: Any,
    images: Sequence[Image.Image],
    instruction_body: str,
) -> dict[str, Any]:
    chat_text = processor.apply_chat_template(
        build_qwen_conversation(instruction_body),
        tokenize=False,
        add_generation_prompt=True,
    )
    batch_text = [chat_text] * len(images)
    inputs = processor(
        text=batch_text,
        images=list(images),
        padding=True,
        return_tensors="pt",
    )
    return dict(inputs)


def _create_qwen_eval_progress(*, disable: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("batch {task.fields[batch]}/{task.fields[total_batches]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=disable,
    )


def generate_qwen_responses(
    model: Any,
    processor: Any,
    device: torch.device,
    images: Sequence[Image.Image],
    instruction_body: str,
    *,
    max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS,
) -> list[str]:
    inputs = _prepare_qwen_inputs(processor, images, instruction_body)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_length:]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def predict_qwen_digits(
    model: Any,
    processor: Any,
    device: torch.device,
    dataset_bundle: QwenDatasetBundle,
    instruction_body: str,
    *,
    indices: Sequence[int] | None = None,
    batch_size: int = DEFAULT_QWEN_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS,
    show_progress: bool = False,
) -> tuple[list[int | None], list[str]]:
    target_indices = list(indices) if indices is not None else list(range(dataset_bundle.num_samples))
    predictions: list[int | None] = []
    raw_responses: list[str] = []
    effective_batch_size = max(1, batch_size)
    if not target_indices:
        return predictions, raw_responses

    total_batches = math.ceil(len(target_indices) / effective_batch_size)
    progress_cm = _create_qwen_eval_progress(disable=not show_progress)
    with progress_cm as progress:
        task_id = progress.add_task(
            f"Evaluating {dataset_bundle.display_name}",
            total=len(target_indices),
            batch=0,
            total_batches=total_batches,
        )
        progress.refresh()
        for batch_number, start in enumerate(range(0, len(target_indices), effective_batch_size), start=1):
            progress.update(
                task_id,
                description=f"Evaluating {dataset_bundle.display_name}",
                batch=batch_number,
                total_batches=total_batches,
            )
            progress.refresh()
            batch_indices = target_indices[start : start + effective_batch_size]
            batch_images = [dataset_bundle.image_loader(index) for index in batch_indices]
            responses = generate_qwen_responses(
                model,
                processor,
                device,
                batch_images,
                instruction_body,
                max_new_tokens=max_new_tokens,
            )
            raw_responses.extend(responses)
            predictions.extend(parse_qwen_digit_response(response) for response in responses)
            progress.update(
                task_id,
                advance=len(batch_indices),
                batch=batch_number,
                total_batches=total_batches,
            )
            progress.refresh()

    return predictions, raw_responses


def evaluate_qwen_zero_shot_prompt(
    model: Any,
    processor: Any,
    device: torch.device,
    dataset_bundle: QwenDatasetBundle,
    instruction_body: str,
    *,
    indices: Sequence[int] | None = None,
    batch_size: int = DEFAULT_QWEN_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS,
    show_progress: bool = False,
) -> tuple[QwenEvalMetrics, list[int | None], list[str]]:
    predictions, raw_responses = predict_qwen_digits(
        model,
        processor,
        device,
        dataset_bundle,
        instruction_body,
        indices=indices,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
    )
    if indices is None:
        labels = dataset_bundle.labels
    else:
        labels = dataset_bundle.labels[np.asarray(indices, dtype=np.int64)]
    metrics = compute_qwen_eval_metrics(predictions, labels)
    return metrics, predictions, raw_responses


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_qwen_eval_summary(
    *,
    repo: str,
    dataset_bundle: QwenDatasetBundle,
    prompt: str,
    metrics: QwenEvalMetrics,
) -> None:
    console.print(
        f"[bold]Qwen Zero-Shot Evaluation[/bold]\n"
        f"repo: {repo}\n"
        f"dataset: {dataset_bundle.display_name}\n"
        f"source: {dataset_bundle.source_path}\n"
        f"prompt: {prompt}\n"
        f"num_samples: {metrics.num_samples}\n"
        f"accuracy: {_format_percentage(metrics.accuracy)}\n"
        f"parse_rate: {_format_percentage(metrics.parse_rate)}\n"
        f"invalid_count: {metrics.invalid_count}\n"
        f"skipped_count: {dataset_bundle.skipped_count}"
    )


def inspect_qwen_tokenization(
    *,
    repo: str,
    text: str,
    add_special_tokens: bool = False,
) -> QwenTokenizationResult:
    tokenizer = load_qwen_tokenizer(repo)
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )
    input_ids = list(encoded["input_ids"])
    tokens = list(tokenizer.convert_ids_to_tokens(input_ids))
    decoded = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return QwenTokenizationResult(
        repo=repo,
        tokenizer_class=tokenizer.__class__.__name__,
        vocab_size=int(tokenizer.vocab_size),
        tokenizer_length=int(len(tokenizer)),
        special_tokens_map=dict(tokenizer.special_tokens_map),
        text=text,
        add_special_tokens=add_special_tokens,
        input_ids=input_ids,
        tokens=tokens,
        decoded=decoded,
    )


def print_qwen_tokenization(result: QwenTokenizationResult) -> None:
    console.print(
        f"[bold]Tokenizer Inspection[/bold]\n"
        f"repo: {result.repo}\n"
        f"tokenizer_class: {result.tokenizer_class}\n"
        f"vocab_size: {result.vocab_size}\n"
        f"tokenizer_length: {result.tokenizer_length}\n"
        f"add_special_tokens: {result.add_special_tokens}\n"
        f"text: {result.text}\n"
        f"decoded: {result.decoded}\n"
        f"special_tokens_map: {result.special_tokens_map}"
    )

    table = Table(title="Tokens", show_lines=False)
    table.add_column("Index", justify="right")
    table.add_column("Token ID", justify="right")
    table.add_column("Token")
    for idx, (token_id, token) in enumerate(zip(result.input_ids, result.tokens), start=1):
        table.add_row(str(idx), str(token_id), repr(token))
    console.print(table)


def run_qwen_tokenize(
    *,
    repo: str,
    text: str,
    add_special_tokens: bool = False,
) -> QwenTokenizationResult:
    result = inspect_qwen_tokenization(
        repo=repo,
        text=text,
        add_special_tokens=add_special_tokens,
    )
    print_qwen_tokenization(result)
    return result


def run_qwen_zero_shot(
    *,
    repo: str,
    prompt: str,
    dataset: str,
    device: str = "auto",
    test_dataset_path: str | Path | None = None,
    datasets_dir: str | Path = "datasets",
    batch_size: int = DEFAULT_QWEN_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    model, processor, torch_device = load_qwen_vl_model(repo, device=device)
    dataset_bundle = load_qwen_dataset(
        dataset,
        test_dataset_path=test_dataset_path,
        datasets_dir=datasets_dir,
    )
    metrics, _, _ = evaluate_qwen_zero_shot_prompt(
        model,
        processor,
        torch_device,
        dataset_bundle,
        prompt,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        show_progress=True,
    )
    print_qwen_eval_summary(
        repo=repo,
        dataset_bundle=dataset_bundle,
        prompt=prompt,
        metrics=metrics,
    )
    return {
        "repo": repo,
        "dataset": dataset_bundle.dataset_key,
        "display_name": dataset_bundle.display_name,
        "source_path": dataset_bundle.source_path,
        "prompt": prompt,
        "accuracy": metrics.accuracy,
        "parse_rate": metrics.parse_rate,
        "invalid_count": metrics.invalid_count,
        "num_samples": metrics.num_samples,
        "skipped_count": dataset_bundle.skipped_count,
    }
