from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm.rich import tqdm as rich_tqdm
from tqdm.std import TqdmExperimentalWarning
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
from transformers.utils import logging as hf_logging

from digit_classifier.mnist_in_the_wild import load_mnist_in_the_wild

try:
    import wandb
except Exception as exc:  # pragma: no cover - depends on local wandb install state
    _WANDB_IMPORT_ERROR = exc

    class _WandbImportStub:
        def __getattr__(self, name):
            raise ImportError("wandb is unavailable in this environment") from _WANDB_IMPORT_ERROR

    wandb = _WandbImportStub()

warnings.filterwarnings(
    "ignore",
    message="rich is experimental/alpha",
    category=TqdmExperimentalWarning,
)


class HFRichTqdm(rich_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)


def download(repo_id: str, *args, **kwargs):
    try:
        return snapshot_download(repo_id, *args, **kwargs, local_files_only=True)
    except LocalEntryNotFoundError:
        log.info(f"Downloaded model assets from {repo_id}")
        return snapshot_download(repo_id, *args, **kwargs, tqdm_class=HFRichTqdm)


@contextmanager
def quiet_transformers_loading():
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous_verbosity)
        hf_logging.enable_progress_bar()


logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("foundation_models")
log.setLevel(logging.INFO)
console = Console()


class FoundationModelFamily(Enum):
    CLIP = "clip"
    DINO = "dino"


class LatentVisualizationMode(Enum):
    REGULAR = "regular"
    CLIP_OVER_DINO = "clip_over_dino"
    DINO_OVER_CLIP = "dino_over_clip"

    @property
    def display_name(self) -> str:
        if self is LatentVisualizationMode.CLIP_OVER_DINO:
            return "CLIP > DINO"
        if self is LatentVisualizationMode.DINO_OVER_CLIP:
            return "DINO > CLIP"
        return "Regular"


class FoundationModelSize(Enum):
    TINY = "tiny"
    SMALL = "small"
    SMALL_PLUS = "small_plus"
    BASE = "base"
    LARGE = "large"
    GIANT = "giant"
    HUGE_PLUS = "huge_plus"
    GIANT_7B = "giant_7b"


@dataclass(frozen=True)
class FoundationModelArchitectureMetadata:
    family: FoundationModelFamily
    model_size: FoundationModelSize
    image_size: int
    patch_size: int | None
    repo: str
    supports_zero_shot: bool


class FoundationModelArchitecture(Enum):
    CLIP_VIT_BASE_PATCH32 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.CLIP,
        FoundationModelSize.BASE,
        224,
        32,
        "openai/clip-vit-base-patch32",
        True,
    )
    CLIP_VIT_BASE_PATCH16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.CLIP,
        FoundationModelSize.BASE,
        224,
        16,
        "openai/clip-vit-base-patch16",
        True,
    )
    CLIP_VIT_LARGE_PATCH14 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.CLIP,
        FoundationModelSize.LARGE,
        224,
        14,
        "openai/clip-vit-large-patch14",
        True,
    )
    CLIP_VIT_LARGE_PATCH14_336 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.CLIP,
        FoundationModelSize.LARGE,
        336,
        14,
        "openai/clip-vit-large-patch14-336",
        True,
    )
    DINO_V1_VIT_S8 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL,
        224,
        8,
        "facebook/dino-vits8",
        False,
    )
    DINO_V1_VIT_S16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL,
        224,
        16,
        "facebook/dino-vits16",
        False,
    )
    DINO_V1_VIT_B8 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.BASE,
        224,
        8,
        "facebook/dino-vitb8",
        False,
    )
    DINO_V1_VIT_B16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.BASE,
        224,
        16,
        "facebook/dino-vitb16",
        False,
    )
    DINO_V2_SMALL = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL,
        224,
        14,
        "facebook/dinov2-small",
        False,
    )
    DINO_V2_BASE = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.BASE,
        224,
        14,
        "facebook/dinov2-base",
        False,
    )
    DINO_V2_LARGE = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.LARGE,
        224,
        14,
        "facebook/dinov2-large",
        False,
    )
    DINO_V2_GIANT = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.GIANT,
        224,
        14,
        "facebook/dinov2-giant",
        False,
    )
    DINO_V2_REG_SMALL = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL,
        224,
        14,
        "facebook/dinov2-with-registers-small",
        False,
    )
    DINO_V2_REG_BASE = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.BASE,
        224,
        14,
        "facebook/dinov2-with-registers-base",
        False,
    )
    DINO_V2_REG_LARGE = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.LARGE,
        224,
        14,
        "facebook/dinov2-with-registers-large",
        False,
    )
    DINO_V2_REG_GIANT = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.GIANT,
        224,
        14,
        "facebook/dinov2-with-registers-giant",
        False,
    )
    DINO_V3_VIT_S16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL,
        224,
        16,
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        False,
    )
    DINO_V3_VIT_S16_PLUS = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL_PLUS,
        224,
        16,
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        False,
    )
    DINO_V3_VIT_B16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.BASE,
        224,
        16,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        False,
    )
    DINO_V3_VIT_L16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.LARGE,
        224,
        16,
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        False,
    )
    DINO_V3_VIT_H16_PLUS = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.HUGE_PLUS,
        224,
        16,
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        False,
    )
    DINO_V3_VIT_7B16 = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.GIANT_7B,
        224,
        16,
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        False,
    )
    DINO_V3_CONVNEXT_TINY = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.TINY,
        224,
        None,
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        False,
    )
    DINO_V3_CONVNEXT_SMALL = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.SMALL,
        224,
        None,
        "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        False,
    )
    DINO_V3_CONVNEXT_BASE = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.BASE,
        224,
        None,
        "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        False,
    )
    DINO_V3_CONVNEXT_LARGE = FoundationModelArchitectureMetadata(
        FoundationModelFamily.DINO,
        FoundationModelSize.LARGE,
        224,
        None,
        "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        False,
    )


DEFAULT_MODELS: dict[FoundationModelFamily, FoundationModelArchitecture] = {
    FoundationModelFamily.CLIP: FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32,
    FoundationModelFamily.DINO: FoundationModelArchitecture.DINO_V3_VIT_B16,
}

CLIP_ZERO_SHOT_PROMPT_PRESETS: dict[str, str] = {
    "current": "an image of natural objects arranged to form the digit {digit}",
    "photo_digit": "a photo of the digit {digit}",
    "photo_number": "a photo of the number {digit}",
    "objects_form": "a real-world scene where objects form the digit {digit}",
    "objects_arranged": "a photograph of objects arranged like the digit {digit}",
}


@dataclass(frozen=True)
class ZeroShotPromptDefinition:
    prompt_id: str
    template: str
    source: str


@dataclass
class FoundationModelConfig:
    """Settings for CLIP and DINO experiments."""

    model: FoundationModelArchitecture = FoundationModelArchitecture.CLIP_VIT_BASE_PATCH32
    family: FoundationModelFamily | None = None
    zero_shot: bool = False
    device: str = "auto"
    linear_probe: bool = False
    head_type: str = "mlp"
    epochs: int = 10
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    batch_size: int = 128
    feature_batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    deep_mlp: bool = False
    dropout: float = 0.2
    layer_norm: bool = False
    seed: int = 42
    n_folds: int = 5
    cv_repeats: int = 1
    log_fold_runs: bool = True
    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"
    split_plan_path: str | None = None
    save_split_plan_path: str | None = None
    save_oof_bundle_path: str | None = None
    test_dataset_path: str | None = None
    ablate_prompts: bool = False
    prompt_presets: tuple[str, ...] = ()
    prompt_file_path: str | None = None
    ablation_datasets: str | None = None
    save_prompt_ablation_path: str | None = None
    use_wandb: bool = True
    sweep_project: str = "mnist-in-the-wild-clip"
    sweep_id: str | None = None
    sweep_count: int | None = None
    sweep_method: str = "random"

    def __post_init__(self) -> None:
        model_family = self.model.value.family
        if self.family is None:
            self.family = model_family
        elif self.family != model_family:
            raise ValueError(
                f"Config family {self.family.value} does not match model family {model_family.value}"
            )
        if self.zero_shot and not self.supports_zero_shot:
            raise ValueError(f"Zero-shot is not supported for {self.family.value} models")

    @property
    def model_size(self) -> FoundationModelSize:
        return self.model.value.model_size

    @property
    def image_size(self) -> int:
        return self.model.value.image_size

    @property
    def patch_size(self) -> int | None:
        return self.model.value.patch_size

    @property
    def repo(self) -> str:
        return self.model.value.repo

    @property
    def supports_zero_shot(self) -> bool:
        return self.model.value.supports_zero_shot


def list_foundation_model_repos(family: FoundationModelFamily | str) -> list[str]:
    family = FoundationModelFamily(family)
    return [
        architecture.value.repo
        for architecture in FoundationModelArchitecture
        if architecture.value.family == family
    ]


def get_foundation_model(
    repo: str,
    family: FoundationModelFamily | str,
) -> FoundationModelArchitecture:
    family = FoundationModelFamily(family)
    for architecture in FoundationModelArchitecture:
        metadata = architecture.value
        if metadata.repo == repo and metadata.family == family:
            return architecture
    raise ValueError(f"Invalid {family.value.upper()} model repository: {repo}")


def load_foundation_model(model: FoundationModelArchitecture) -> tuple[Any, Any]:
    repo = model.value.repo
    family = model.value.family
    local_dir = download(repo)

    with quiet_transformers_loading():
        if family == FoundationModelFamily.CLIP:
            log.info(f"Loading CLIP model from {repo}")
            loaded_model = CLIPModel.from_pretrained(local_dir)
            log.info(f"Loading CLIP processor from {repo}")
            processor = CLIPProcessor.from_pretrained(local_dir, use_fast=False)
        else:
            log.info(f"Loading DINO model from {repo}")
            loaded_model = AutoModel.from_pretrained(local_dir)
            log.info(f"Loading DINO image processor from {repo}")
            processor = AutoImageProcessor.from_pretrained(local_dir)

    return loaded_model, processor


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)


def preview_images(images, labels):
    import cv2

    for image, label in zip(images, labels):
        cv2.imshow(str(label), np.asarray(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _sanitize_repo(repo: str) -> str:
    return repo.replace("/", "_")


def compute_dataset_hash(
    images: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> str:
    images_np = np.asarray(images.detach().cpu().numpy() if torch.is_tensor(images) else images)
    labels_np = np.asarray(labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels)
    hasher = hashlib.sha256()
    # Use sample count + label order so the fingerprint remains stable across
    # model-specific image resolutions while still detecting sample-order mismatches.
    hasher.update(np.asarray([int(images_np.shape[0])], dtype=np.int64).tobytes())
    hasher.update(np.asarray(labels_np.shape, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(labels_np.astype(np.int64, copy=False)).tobytes())
    return hasher.hexdigest()


def _compute_split_plan_id(
    dataset_hash: str,
    n_samples: int,
    splits: list[dict[str, Any]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(dataset_hash.encode("utf-8"))
    hasher.update(np.asarray([n_samples, len(splits)], dtype=np.int64).tobytes())
    for split in splits:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        val_idx = np.asarray(split["val_idx"], dtype=np.int64)
        hasher.update(np.asarray([repeat, fold], dtype=np.int64).tobytes())
        hasher.update(train_idx.tobytes())
        hasher.update(b"|")
        hasher.update(val_idx.tobytes())
        hasher.update(b";")
    return hasher.hexdigest()


def validate_split_plan(
    plan: dict[str, Any],
    *,
    n_samples: int,
    dataset_hash: str | None = None,
) -> None:
    required_fields = {
        "version",
        "dataset_hash",
        "n_samples",
        "n_folds",
        "cv_repeats",
        "splits",
        "plan_id",
    }
    missing = required_fields.difference(plan)
    if missing:
        raise ValueError(f"Split plan is missing required fields: {sorted(missing)}")

    plan_n_samples = int(plan["n_samples"])
    if plan_n_samples != n_samples:
        raise ValueError(
            f"Split plan n_samples ({plan_n_samples}) does not match dataset ({n_samples})"
        )
    if dataset_hash is not None and str(plan["dataset_hash"]) != dataset_hash:
        raise ValueError("Split plan dataset hash does not match the current dataset")

    n_folds = int(plan["n_folds"])
    cv_repeats = int(plan["cv_repeats"])
    if n_folds < 2:
        raise ValueError(f"Split plan n_folds must be at least 2, got {n_folds}")
    if cv_repeats < 1:
        raise ValueError(f"Split plan cv_repeats must be at least 1, got {cv_repeats}")

    splits = plan["splits"]
    if not isinstance(splits, list) or not splits:
        raise ValueError("Split plan must contain a non-empty list of splits")
    if len(splits) != n_folds * cv_repeats:
        raise ValueError(
            f"Split plan expected {n_folds * cv_repeats} splits, found {len(splits)}"
        )

    expected_plan_id = _compute_split_plan_id(
        str(plan["dataset_hash"]),
        plan_n_samples,
        splits,
    )
    if str(plan["plan_id"]) != expected_plan_id:
        raise ValueError("Split plan plan_id does not match its contents")

    overall_val_counts = np.zeros(plan_n_samples, dtype=np.int64)
    per_repeat_val_counts: dict[int, np.ndarray] = {
        repeat: np.zeros(plan_n_samples, dtype=np.int64)
        for repeat in range(1, cv_repeats + 1)
    }
    seen_pairs: set[tuple[int, int]] = set()

    full_index_set = np.arange(plan_n_samples, dtype=np.int64)
    for split in splits:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        if not (1 <= repeat <= cv_repeats):
            raise ValueError(f"Split plan repeat index {repeat} is out of range")
        if not (1 <= fold <= n_folds):
            raise ValueError(f"Split plan fold index {fold} is out of range")
        if (repeat, fold) in seen_pairs:
            raise ValueError(f"Split plan contains duplicate split (repeat={repeat}, fold={fold})")
        seen_pairs.add((repeat, fold))

        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        val_idx = np.asarray(split["val_idx"], dtype=np.int64)
        if train_idx.ndim != 1 or val_idx.ndim != 1:
            raise ValueError("Split plan train_idx and val_idx must be 1D")
        if len(train_idx) + len(val_idx) != plan_n_samples:
            raise ValueError("Split plan train_idx and val_idx do not cover the full dataset")
        if len(np.unique(train_idx)) != len(train_idx) or len(np.unique(val_idx)) != len(val_idx):
            raise ValueError("Split plan train_idx/val_idx contain duplicates")
        if np.any(train_idx < 0) or np.any(train_idx >= plan_n_samples):
            raise ValueError("Split plan train_idx contains out-of-range indices")
        if np.any(val_idx < 0) or np.any(val_idx >= plan_n_samples):
            raise ValueError("Split plan val_idx contains out-of-range indices")

        membership = np.zeros(plan_n_samples, dtype=np.int8)
        membership[train_idx] += 1
        membership[val_idx] += 1
        if not np.array_equal(membership, np.ones(plan_n_samples, dtype=np.int8)):
            raise ValueError("Split plan train_idx and val_idx must partition the dataset")
        if not np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), full_index_set):
            raise ValueError("Split plan train_idx and val_idx must cover every sample exactly once")

        overall_val_counts[val_idx] += 1
        per_repeat_val_counts[repeat][val_idx] += 1

    expected_overall = np.full(plan_n_samples, cv_repeats, dtype=np.int64)
    if not np.array_equal(overall_val_counts, expected_overall):
        raise ValueError("Split plan does not give every sample one validation slot per repeat")
    for repeat, repeat_counts in per_repeat_val_counts.items():
        if not np.array_equal(repeat_counts, np.ones(plan_n_samples, dtype=np.int64)):
            raise ValueError(
                f"Split plan repeat {repeat} does not validate every sample exactly once"
            )


def build_repeated_stratified_split_plan(
    labels: np.ndarray | torch.Tensor,
    *,
    n_folds: int,
    cv_repeats: int,
    seed: int,
    dataset_hash: str,
) -> dict[str, Any]:
    labels_np = np.asarray(labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if cv_repeats < 1:
        raise ValueError(f"cv_repeats must be at least 1, got {cv_repeats}")

    dummy_inputs = np.zeros((len(labels_np), 1), dtype=np.uint8)
    splits: list[dict[str, Any]] = []
    for repeat in range(1, cv_repeats + 1):
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed + repeat - 1,
        )
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(dummy_inputs, labels_np),
            start=1,
        ):
            splits.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                }
            )

    plan = {
        "version": 1,
        "dataset_hash": dataset_hash,
        "n_samples": int(len(labels_np)),
        "n_folds": int(n_folds),
        "cv_repeats": int(cv_repeats),
        "seed": int(seed),
        "splits": splits,
    }
    plan["plan_id"] = _compute_split_plan_id(dataset_hash, int(len(labels_np)), splits)
    validate_split_plan(plan, n_samples=len(labels_np), dataset_hash=dataset_hash)
    return plan


def save_split_plan(plan: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_split_plan(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    plan = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"Split plan at {input_path} is not a valid object")
    return plan


def save_oof_prediction_bundle(
    path: str | Path,
    *,
    cfg: FoundationModelConfig,
    dataset_hash: str,
    plan_id: str,
    labels: np.ndarray | torch.Tensor,
    mean_probs: np.ndarray,
    counts: np.ndarray,
) -> Path:
    labels_np = np.asarray(labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels)
    labels_np = labels_np.astype(np.int64, copy=False)
    mean_probs = np.asarray(mean_probs, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)

    if mean_probs.ndim != 2:
        raise ValueError(f"mean_probs must be 2D, got shape {mean_probs.shape}")
    if mean_probs.shape[0] != len(labels_np):
        raise ValueError("mean_probs length must match labels length")
    if counts.shape != (len(labels_np),):
        raise ValueError("counts length must match labels length")
    if np.any(counts <= 0):
        raise ValueError("OOF bundle counts must be positive for every sample")

    predictions = mean_probs.argmax(axis=-1).astype(np.int64)
    confidence = mean_probs.max(axis=-1).astype(np.float32)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.asarray(1, dtype=np.int64),
        family=np.asarray(cfg.family.value),
        repo=np.asarray(cfg.repo),
        dataset_hash=np.asarray(dataset_hash),
        plan_id=np.asarray(plan_id),
        n_folds=np.asarray(cfg.n_folds, dtype=np.int64),
        cv_repeats=np.asarray(cfg.cv_repeats, dtype=np.int64),
        labels=labels_np,
        counts=counts,
        mean_probs=mean_probs,
        predictions=predictions,
        confidence=confidence,
    )
    return output_path


def load_oof_prediction_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        required_fields = {
            "version",
            "family",
            "repo",
            "dataset_hash",
            "plan_id",
            "n_folds",
            "cv_repeats",
            "labels",
            "counts",
            "mean_probs",
            "predictions",
            "confidence",
        }
        missing = required_fields.difference(bundle.files)
        if missing:
            raise ValueError(
                f"OOF bundle at {bundle_path} is missing required fields: {sorted(missing)}"
            )

        result = {
            "version": int(bundle["version"].item()),
            "family": str(bundle["family"].item()),
            "repo": str(bundle["repo"].item()),
            "dataset_hash": str(bundle["dataset_hash"].item()),
            "plan_id": str(bundle["plan_id"].item()),
            "n_folds": int(bundle["n_folds"].item()),
            "cv_repeats": int(bundle["cv_repeats"].item()),
            "labels": bundle["labels"].astype(np.int64, copy=False),
            "counts": bundle["counts"].astype(np.int64, copy=False),
            "mean_probs": bundle["mean_probs"].astype(np.float32, copy=False),
            "predictions": bundle["predictions"].astype(np.int64, copy=False),
            "confidence": bundle["confidence"].astype(np.float32, copy=False),
        }

    labels = result["labels"]
    counts = result["counts"]
    mean_probs = result["mean_probs"]
    predictions = result["predictions"]
    confidence = result["confidence"]
    if mean_probs.ndim != 2 or mean_probs.shape[0] != len(labels):
        raise ValueError("OOF bundle mean_probs shape does not match labels")
    if counts.shape != (len(labels),):
        raise ValueError("OOF bundle counts shape does not match labels")
    if predictions.shape != (len(labels),):
        raise ValueError("OOF bundle predictions shape does not match labels")
    if confidence.shape != (len(labels),):
        raise ValueError("OOF bundle confidence shape does not match labels")
    if np.any(counts <= 0):
        raise ValueError("OOF bundle contains samples without any OOF predictions")
    return result


def validate_oof_prediction_bundle(
    bundle: dict[str, Any],
    *,
    expected_family: FoundationModelFamily | str | None = None,
    expected_repo: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_labels: np.ndarray | None = None,
    expected_plan_id: str | None = None,
) -> None:
    if expected_family is not None and bundle["family"] != FoundationModelFamily(expected_family).value:
        raise ValueError(
            f"OOF bundle family {bundle['family']!r} does not match expected family "
            f"{FoundationModelFamily(expected_family).value!r}"
        )
    if expected_repo is not None and bundle["repo"] != expected_repo:
        raise ValueError(
            f"OOF bundle repo {bundle['repo']!r} does not match expected repo {expected_repo!r}"
        )
    if expected_dataset_hash is not None and bundle["dataset_hash"] != expected_dataset_hash:
        raise ValueError("OOF bundle dataset hash does not match the current dataset")
    if expected_plan_id is not None and bundle["plan_id"] != expected_plan_id:
        raise ValueError("OOF bundles were generated from different split plans")
    if expected_labels is not None and not np.array_equal(bundle["labels"], expected_labels):
        raise ValueError("OOF bundle labels do not match the current dataset labels")


def _extract_image_features(
    model: Any,
    pixel_values: torch.Tensor,
    family: FoundationModelFamily,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if family == FoundationModelFamily.CLIP:
        image_latents = model.vision_model(pixel_values=pixel_values).pooler_output
        image_features = model.visual_projection(image_latents)
        normalized_image_features = F.normalize(image_features, dim=-1)
        return image_latents, image_features, normalized_image_features

    outputs = model(pixel_values=pixel_values)
    image_latents = outputs.pooler_output
    image_features = image_latents
    normalized_image_features = F.normalize(image_features, dim=-1)
    return image_latents, image_features, normalized_image_features


def compute_foundation_model_features(
    model: Any,
    processor: Any,
    cfg: FoundationModelConfig,
    device: torch.device,
):
    cache_path = (
        Path("cache")
        / f"foundation_model_features_{cfg.family.value}_{_sanitize_repo(cfg.repo)}.npz"
    )

    if cache_path.exists():
        log.info(f"Loading cached image features from {cache_path}")
        data = np.load(cache_path)
        image_latents = torch.from_numpy(data["image_latents"]).to(device)
        image_features = torch.from_numpy(data["image_features"]).to(device)
        normalized_image_features = torch.from_numpy(data["normalized_image_features"]).to(device)
        labels = torch.from_numpy(data["labels"]).to(device).long()
        return image_latents, image_features, normalized_image_features, labels

    log.info("Loading MNIST in the Wild dataset")
    images, labels, _, _ = load_mnist_in_the_wild(cfg)
    log.info("Preprocessing images")
    pil_images = [to_pil_image(image) for image in images]

    feature_batch_size = max(1, cfg.feature_batch_size)
    image_latents_chunks = []
    image_features_chunks = []
    normalized_image_features_chunks = []

    log.info(f"Computing image features in batches of {feature_batch_size}")
    with torch.inference_mode():
        for start in track(
            range(0, len(pil_images), feature_batch_size),
            description=f"Encoding {cfg.family.value.upper()} image features",
        ):
            end = min(start + feature_batch_size, len(pil_images))
            batch_images = pil_images[start:end]

            image_inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device)

            batch_image_latents, batch_image_features, batch_normalized_image_features = (
                _extract_image_features(model, pixel_values, cfg.family)
            )

            image_latents_chunks.append(batch_image_latents.cpu())
            image_features_chunks.append(batch_image_features.cpu())
            normalized_image_features_chunks.append(batch_normalized_image_features.cpu())

            del image_inputs
            del pixel_values
            del batch_image_latents
            del batch_image_features
            del batch_normalized_image_features
            if device.type == "cuda":
                torch.cuda.empty_cache()

    image_latents = torch.cat(image_latents_chunks, dim=0)
    image_features = torch.cat(image_features_chunks, dim=0)
    normalized_image_features = torch.cat(normalized_image_features_chunks, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.long)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving cached image features to {cache_path}")
    np.savez(
        cache_path,
        image_latents=image_latents.numpy(),
        image_features=image_features.numpy(),
        normalized_image_features=normalized_image_features.numpy(),
        labels=labels.numpy(),
    )

    return (
        image_latents.to(device),
        image_features.to(device),
        normalized_image_features.to(device),
        labels.to(device),
    )


def _load_pareidolia_eval_samples(
    test_dataset_path: str | Path,
) -> tuple[list[Path], torch.Tensor]:
    from digit_classifier.pareidolia_dataset import PareidoliaTestDataset

    dataset = PareidoliaTestDataset(root_dir=test_dataset_path, preload=False)
    if not dataset.samples:
        raise ValueError(f"No pareidolia samples found under {test_dataset_path}")

    image_paths = [dataset.root / relative_path for relative_path, _ in dataset.samples]
    labels = torch.tensor([label for _, label in dataset.samples], dtype=torch.long)
    return image_paths, labels


def _load_rgb_pil_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def compute_foundation_model_test_features(
    model: Any,
    processor: Any,
    cfg: FoundationModelConfig,
    device: torch.device,
    *,
    test_dataset_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_paths, labels = _load_pareidolia_eval_samples(test_dataset_path)
    normalized_image_features_chunks = []
    feature_batch_size = max(1, cfg.feature_batch_size)

    log.info(f"Loading pareidolia test dataset from {test_dataset_path}")
    log.info(f"Computing test image features in batches of {feature_batch_size}")
    with torch.inference_mode():
        for start in track(
            range(0, len(image_paths), feature_batch_size),
            description=f"Encoding {cfg.family.value.upper()} test image features",
        ):
            end = min(start + feature_batch_size, len(image_paths))
            batch_paths = image_paths[start:end]
            batch_images = [_load_rgb_pil_image(path) for path in batch_paths]

            image_inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device)
            _, _, batch_normalized_image_features = _extract_image_features(
                model,
                pixel_values,
                cfg.family,
            )
            normalized_image_features_chunks.append(batch_normalized_image_features.cpu())

            del image_inputs
            del pixel_values
            del batch_images
            del batch_normalized_image_features
            if device.type == "cuda":
                torch.cuda.empty_cache()

    normalized_image_features = torch.cat(normalized_image_features_chunks, dim=0)
    return normalized_image_features.to(device), labels.to(device)


def validate_zero_shot_prompt_template(template: str, *, source: str) -> str:
    normalized = template.strip()
    if not normalized:
        raise ValueError(f"Zero-shot prompt template from {source} is empty")
    if normalized.count("{digit}") != 1:
        raise ValueError(
            f"Zero-shot prompt template from {source} must contain '{{digit}}' exactly once"
        )
    return normalized


def resolve_zero_shot_prompt_definitions(cfg: FoundationModelConfig) -> list[ZeroShotPromptDefinition]:
    definitions: list[ZeroShotPromptDefinition] = []
    seen_preset_ids: set[str] = set()

    selected_presets = list(cfg.prompt_presets)
    if cfg.ablate_prompts and not selected_presets and not cfg.prompt_file_path:
        selected_presets = list(CLIP_ZERO_SHOT_PROMPT_PRESETS)
    if not cfg.ablate_prompts and not selected_presets:
        selected_presets = ["current"]

    for preset_id in selected_presets:
        if preset_id in seen_preset_ids:
            continue
        try:
            template = CLIP_ZERO_SHOT_PROMPT_PRESETS[preset_id]
        except KeyError as exc:
            raise ValueError(f"Unknown CLIP zero-shot prompt preset: {preset_id}") from exc
        definitions.append(
            ZeroShotPromptDefinition(
                prompt_id=preset_id,
                template=validate_zero_shot_prompt_template(
                    template,
                    source=f"preset {preset_id}",
                ),
                source="preset",
            )
        )
        seen_preset_ids.add(preset_id)

    if cfg.prompt_file_path:
        prompt_file_path = Path(cfg.prompt_file_path)
        lines = prompt_file_path.read_text(encoding="utf-8").splitlines()
        file_prompt_index = 1
        for line_number, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            definitions.append(
                ZeroShotPromptDefinition(
                    prompt_id=f"file_{file_prompt_index}",
                    template=validate_zero_shot_prompt_template(
                        stripped,
                        source=f"{prompt_file_path}:{line_number}",
                    ),
                    source="file",
                )
            )
            file_prompt_index += 1

    if not definitions:
        raise ValueError("No zero-shot prompt templates were selected")
    return definitions


def _resolve_zero_shot_ablation_dataset_keys(cfg: FoundationModelConfig) -> list[str]:
    selected = cfg.ablation_datasets
    if selected is None:
        return ["mnist_in_the_wild", "pareidolia"] if cfg.test_dataset_path else ["mnist_in_the_wild"]
    if selected == "mnist":
        return ["mnist_in_the_wild"]
    if selected == "pareidolia":
        return ["pareidolia"]
    if selected == "both":
        return ["mnist_in_the_wild", "pareidolia"]
    raise ValueError(f"Unsupported ablation dataset selection: {selected}")


def _load_zero_shot_dataset_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    cfg: FoundationModelConfig,
    device: torch.device,
    dataset_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset_key == "mnist_in_the_wild":
        _, _, normalized_image_features, labels = compute_foundation_model_features(
            model,
            processor,
            cfg,
            device,
        )
        return normalized_image_features, labels
    if dataset_key == "pareidolia":
        if not cfg.test_dataset_path:
            raise ValueError("Pareidolia zero-shot evaluation requires test_dataset_path")
        return compute_foundation_model_test_features(
            model,
            processor,
            cfg,
            device,
            test_dataset_path=cfg.test_dataset_path,
        )
    raise ValueError(f"Unsupported zero-shot dataset key: {dataset_key}")


def encode_zero_shot_prompt_definitions(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    prompt_definitions: list[ZeroShotPromptDefinition],
    *,
    text_batch_size: int,
) -> torch.Tensor:
    rendered_prompts = [
        prompt_definition.template.format(digit=digit)
        for prompt_definition in prompt_definitions
        for digit in range(10)
    ]
    encoded_batches = []
    effective_batch_size = max(1, text_batch_size)

    with torch.inference_mode():
        for start in range(0, len(rendered_prompts), effective_batch_size):
            end = min(start + effective_batch_size, len(rendered_prompts))
            batch_prompts = rendered_prompts[start:end]
            text_inputs = processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            text_latents = model.text_model(**text_inputs)
            text_features = model.text_projection(text_latents.pooler_output)
            encoded_batches.append(F.normalize(text_features, dim=-1).cpu())

    flat_text_features = torch.cat(encoded_batches, dim=0)
    return flat_text_features.view(len(prompt_definitions), 10, -1).to(device)


def rank_prompt_metric_rows(
    rows: list[dict[str, Any]],
    *,
    accuracy_key: str,
    nll_key: str,
) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda item: (-float(item[accuracy_key]), float(item[nll_key]), str(item["prompt_id"])),
        ),
        start=1,
    ):
        ranked_rows.append(
            {
                **row,
                "rank": rank,
                "is_best": rank == 1,
            }
        )
    return ranked_rows


def evaluate_zero_shot_prompt_definitions(
    normalized_image_features: torch.Tensor,
    labels: torch.Tensor,
    prompt_definitions: list[ZeroShotPromptDefinition],
    normalized_text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> list[dict[str, Any]]:
    batch_logits = logit_scale * torch.einsum(
        "nd,pcd->pnc",
        normalized_image_features,
        normalized_text_features,
    )
    probabilities = F.softmax(batch_logits, dim=-1).cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)

    rows = []
    for index, prompt_definition in enumerate(prompt_definitions):
        rows.append(
            {
                "prompt_id": prompt_definition.prompt_id,
                "template": prompt_definition.template,
                "source": prompt_definition.source,
                **_summarize_probabilities(probabilities[index], labels_np),
            }
        )
    return rank_prompt_metric_rows(rows, accuracy_key="accuracy", nll_key="mean_nll")


def summarize_zero_shot_ablation_overall(
    prompt_definitions: list[ZeroShotPromptDefinition],
    per_dataset_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    dataset_by_prompt_id = {
        dataset_name: {
            row["prompt_id"]: row
            for row in dataset_rows
        }
        for dataset_name, dataset_rows in per_dataset_results.items()
    }
    rows = []
    for prompt_definition in prompt_definitions:
        prompt_rows = [
            dataset_rows[prompt_definition.prompt_id]
            for dataset_rows in dataset_by_prompt_id.values()
        ]
        rows.append(
            {
                "prompt_id": prompt_definition.prompt_id,
                "template": prompt_definition.template,
                "source": prompt_definition.source,
                "macro_avg_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in prompt_rows], dtype=np.float64)
                ),
                "macro_avg_mean_nll": float(
                    np.mean([float(row["mean_nll"]) for row in prompt_rows], dtype=np.float64)
                ),
            }
        )
    return rank_prompt_metric_rows(
        rows,
        accuracy_key="macro_avg_accuracy",
        nll_key="macro_avg_mean_nll",
    )


def print_zero_shot_ablation_dataset_table(
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> None:
    table = Table(title=f"CLIP Zero-Shot Prompt Ablation: {dataset_name}")
    table.add_column("Rank", justify="right")
    table.add_column("Best")
    table.add_column("Prompt ID")
    table.add_column("Template", overflow="fold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Mean NLL", justify="right")
    for row in rows:
        table.add_row(
            str(row["rank"]),
            "*" if row["is_best"] else "",
            str(row["prompt_id"]),
            str(row["template"]),
            f"{row['accuracy']:.5f}",
            f"{row['mean_nll']:.5f}",
        )
    console.print(table)


def print_zero_shot_ablation_overall_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="CLIP Zero-Shot Prompt Ablation: Overall Ranking")
    table.add_column("Rank", justify="right")
    table.add_column("Best")
    table.add_column("Prompt ID")
    table.add_column("Template", overflow="fold")
    table.add_column("Macro Avg Acc", justify="right")
    table.add_column("Macro Avg NLL", justify="right")
    for row in rows:
        table.add_row(
            str(row["rank"]),
            "*" if row["is_best"] else "",
            str(row["prompt_id"]),
            str(row["template"]),
            f"{row['macro_avg_accuracy']:.5f}",
            f"{row['macro_avg_mean_nll']:.5f}",
        )
    console.print(table)


def save_zero_shot_prompt_ablation_results(
    output_path: str | Path,
    *,
    cfg: FoundationModelConfig,
    datasets_evaluated: list[str],
    prompt_definitions: list[ZeroShotPromptDefinition],
    per_dataset_results: dict[str, list[dict[str, Any]]],
    overall_ranking: list[dict[str, Any]],
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_repo": cfg.repo,
        "datasets_evaluated": datasets_evaluated,
        "prompt_definitions": [
            {
                "prompt_id": prompt_definition.prompt_id,
                "template": prompt_definition.template,
                "source": prompt_definition.source,
            }
            for prompt_definition in prompt_definitions
        ],
        "per_dataset_results": per_dataset_results,
        "overall_ranking": overall_ranking,
    }
    resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved_output_path


def run_clip_zero_shot_prompt_ablation(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    cfg: FoundationModelConfig,
) -> dict[str, Any]:
    prompt_definitions = resolve_zero_shot_prompt_definitions(cfg)
    datasets_evaluated = _resolve_zero_shot_ablation_dataset_keys(cfg)
    normalized_text_features = encode_zero_shot_prompt_definitions(
        model,
        processor,
        device,
        prompt_definitions,
        text_batch_size=cfg.batch_size,
    )
    logit_scale = model.logit_scale.exp()

    per_dataset_results: dict[str, list[dict[str, Any]]] = {}
    for dataset_key in datasets_evaluated:
        normalized_image_features, labels = _load_zero_shot_dataset_features(
            model,
            processor,
            cfg,
            device,
            dataset_key,
        )
        per_dataset_results[dataset_key] = evaluate_zero_shot_prompt_definitions(
            normalized_image_features,
            labels,
            prompt_definitions,
            normalized_text_features,
            logit_scale,
        )
        print_zero_shot_ablation_dataset_table(dataset_key, per_dataset_results[dataset_key])

    overall_ranking = summarize_zero_shot_ablation_overall(
        prompt_definitions,
        per_dataset_results,
    )
    print_zero_shot_ablation_overall_table(overall_ranking)

    result = {
        "model_repo": cfg.repo,
        "datasets_evaluated": datasets_evaluated,
        "prompt_definitions": prompt_definitions,
        "per_dataset_results": per_dataset_results,
        "overall_ranking": overall_ranking,
    }
    if cfg.save_prompt_ablation_path:
        saved_path = save_zero_shot_prompt_ablation_results(
            cfg.save_prompt_ablation_path,
            cfg=cfg,
            datasets_evaluated=datasets_evaluated,
            prompt_definitions=prompt_definitions,
            per_dataset_results=per_dataset_results,
            overall_ranking=overall_ranking,
        )
        result["saved_prompt_ablation"] = str(saved_path)
        log.info(f"Saved prompt ablation results to {saved_path}")
    return result


def run_clip_zero_shot(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    cfg: FoundationModelConfig,
):
    if not cfg.supports_zero_shot:
        raise ValueError(f"Zero-shot is not supported for {cfg.family.value} models")
    if cfg.ablate_prompts:
        return run_clip_zero_shot_prompt_ablation(model, processor, device, cfg)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total number of parameters in the model: {total_params}")

    prompt_definition = ZeroShotPromptDefinition(
        prompt_id="current",
        template=CLIP_ZERO_SHOT_PROMPT_PRESETS["current"],
        source="preset",
    )
    normalized_text_features = encode_zero_shot_prompt_definitions(
        model,
        processor,
        device,
        [prompt_definition],
        text_batch_size=10,
    )[0]
    logit_scale = model.logit_scale.exp()

    with torch.inference_mode():
        if cfg.test_dataset_path:
            normalized_image_features, labels = compute_foundation_model_test_features(
                model,
                processor,
                cfg,
                device,
                test_dataset_path=cfg.test_dataset_path,
            )
        else:
            _, _, normalized_image_features, labels = compute_foundation_model_features(
                model, processor, cfg, device
            )
        logits = logit_scale * (normalized_image_features @ normalized_text_features.T)
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
        results = _summarize_probabilities(probabilities, labels_np)
        target_name = cfg.test_dataset_path or "MNIST in the Wild"
        log.info(f"Zero-shot eval on {target_name} ({len(labels_np)} samples)")
        for key, value in results.items():
            if key == "num_samples":
                log.info(f"{key}: {int(value)}")
            else:
                log.info(f"{key}: {value:.5f}")

        if cfg.test_dataset_path:
            return results

        from digit_classifier.latent_visualization import plot_latent_space

        embeddings = normalized_image_features.detach().cpu().numpy()
        images, _, _, _ = load_mnist_in_the_wild(cfg)

        result = plot_latent_space(
            embeddings,
            labels_np,
            images=images.numpy(),
            reducer="umap",
            reducer_kwargs=dict(n_neighbors=15, min_dist=0.1),
            model=model,
            processor=processor,
            device=str(device),
        )
        if hasattr(result, "show"):
            result.show()
        return results


class FoundationModelFineTuningDataset(Dataset):
    def __init__(self, image_latents: torch.Tensor, labels: torch.Tensor):
        self.image_latents = image_latents
        self.labels = labels

    def __len__(self):
        return len(self.image_latents)

    def __getitem__(self, idx):
        return self.image_latents[idx], self.labels[idx]


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_classifier(cfg: FoundationModelConfig, input_dim: int) -> nn.Module:
    head_type = cfg.head_type
    if cfg.linear_probe:
        head_type = "linear"
    elif cfg.deep_mlp:
        head_type = "deep_mlp"

    if head_type == "linear":
        layers = [nn.Linear(input_dim, 10)]
    elif head_type == "deep_mlp":
        layers = [
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 10),
        ]
    elif head_type == "medium_mlp":
        layers = [
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 10),
        ]
    elif head_type == "deep_wide_mlp":
        layers = [
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 10),
        ]
    elif head_type == "deep_extra_wide_mlp":
        layers = [
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 10),
        ]
    elif head_type == "mlp":
        layers = [
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 10),
        ]
    else:
        raise ValueError(f"Invalid head_type: {head_type}")

    if cfg.layer_norm:
        layers.insert(0, nn.LayerNorm(input_dim))

    return nn.Sequential(*layers)


def build_classifier_from_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    """Rebuild a saved classifier head using only its state dict structure."""
    linear_weights = sorted(
        (
            int(key.split(".", maxsplit=1)[0]),
            tensor,
        )
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and tensor.ndim == 2
    )
    if not linear_weights:
        raise ValueError("Checkpoint does not contain any linear layers")

    first_linear_weight = linear_weights[0][1]
    input_dim = int(first_linear_weight.shape[1])

    has_layer_norm = False
    first_weight = state_dict.get("0.weight")
    first_bias = state_dict.get("0.bias")
    if (
        first_weight is not None
        and first_bias is not None
        and first_weight.ndim == 1
        and first_bias.ndim == 1
    ):
        has_layer_norm = True
        if int(first_weight.shape[0]) != input_dim:
            raise ValueError("LayerNorm input dimension does not match classifier input dimension")

    hidden_dims = [int(weight.shape[0]) for _, weight in linear_weights[:-1]]
    output_dim = int(linear_weights[-1][1].shape[0])

    layers: list[nn.Module] = []
    if has_layer_norm:
        layers.append(nn.LayerNorm(input_dim))

    if len(linear_weights) == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    elif len(linear_weights) == 2:
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dims[0], output_dim),
            ]
        )
    elif len(linear_weights) == 3:
        layers.extend(
            [
                nn.Linear(input_dim, hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dims[1], output_dim),
            ]
        )
    else:
        raise ValueError(
            f"Unsupported classifier architecture with {len(linear_weights)} linear layers"
        )

    return nn.Sequential(*layers)


def load_classifier_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a valid classifier checkpoint")
    state_dict = checkpoint.get("classifier_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a classifier_state_dict"
        )

    classifier = build_classifier_from_state_dict(state_dict)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def _get_classifier_input_dim(classifier: nn.Module) -> int:
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    raise ValueError("Classifier does not contain any linear layers")


def _get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise ValueError("Classifier does not have any parameters") from exc


def load_classifier_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> nn.Module:
    classifier = build_classifier_from_state_dict(state_dict)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def predict_classifier_probabilities(
    classifier: nn.Module,
    features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    probabilities = []
    classifier.eval()
    module_device = _get_module_device(classifier)
    with torch.inference_mode():
        for start in range(0, len(features), max(1, batch_size)):
            end = min(start + max(1, batch_size), len(features))
            batch_features = features[start:end].to(module_device)
            logits = classifier(batch_features)
            probabilities.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(probabilities, dim=0).numpy()


def predict_classifier_labels(
    classifier: nn.Module,
    features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    probabilities = predict_classifier_probabilities(classifier, features, batch_size)
    return probabilities.argmax(axis=-1).astype(np.int64, copy=False)


def _top_k_accuracy(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
) -> float:
    top_k = min(k, probabilities.shape[1])
    top_k_predictions = np.argsort(probabilities, axis=1)[:, -top_k:]
    return float(np.any(top_k_predictions == labels[:, None], axis=1).mean())


def _summarize_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=-1).astype(np.int64, copy=False)
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    true_probabilities = np.clip(true_probabilities, 1e-12, 1.0)
    return {
        "accuracy": float((predictions == labels).mean()),
        "top_2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
        "top_3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
        "top_5_accuracy": _top_k_accuracy(probabilities, labels, k=5),
        "top_9_accuracy": _top_k_accuracy(probabilities, labels, k=9),
        "mean_nll": float((-np.log(true_probabilities)).mean()),
        "num_samples": float(len(labels)),
    }


def run_foundation_model_eval(
    cfg: FoundationModelConfig,
    *,
    checkpoint_path: str | Path,
    test_dataset_path: str | Path,
) -> dict[str, float]:
    seed_everything(cfg.seed)

    device = get_device(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a valid classifier checkpoint")

    checkpoint_family = checkpoint.get("family")
    if checkpoint_family is not None and checkpoint_family != cfg.family.value:
        raise ValueError(
            f"Checkpoint family {checkpoint_family!r} does not match requested family {cfg.family.value!r}"
        )

    checkpoint_repo = checkpoint.get("repo")
    if checkpoint_repo is not None and checkpoint_repo != cfg.repo:
        raise ValueError(
            f"Checkpoint repo {checkpoint_repo!r} does not match requested repo {cfg.repo!r}"
        )

    state_dict = checkpoint.get("classifier_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a classifier_state_dict"
        )

    classifier = load_classifier_from_state_dict(state_dict, device)
    input_dim = _get_classifier_input_dim(classifier)

    model, processor = load_foundation_model(cfg.model)
    model = model.to(device)
    model.eval()
    freeze(model)

    normalized_image_features, labels = compute_foundation_model_test_features(
        model,
        processor,
        cfg,
        device,
        test_dataset_path=test_dataset_path,
    )
    if normalized_image_features.shape[-1] != input_dim:
        raise ValueError(
            f"Classifier expects feature dimension {input_dim}, but {cfg.repo} produces "
            f"{normalized_image_features.shape[-1]}. Check that --repo matches the checkpoint."
        )

    features = normalized_image_features.detach().cpu()
    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    probabilities = predict_classifier_probabilities(classifier, features, cfg.batch_size)
    results = _summarize_probabilities(probabilities, labels_np)

    log.info(f"Eval on {test_dataset_path} ({len(labels_np)} samples)")
    for key, value in results.items():
        if key == "num_samples":
            log.info(f"{key}: {int(value)}")
        else:
            log.info(f"{key}: {value:.5f}")
    return results


def build_latent_visualization_mask(
    mode: LatentVisualizationMode,
    labels: np.ndarray,
    clip_predictions: np.ndarray,
    dino_predictions: np.ndarray,
) -> np.ndarray:
    if mode is LatentVisualizationMode.REGULAR:
        return np.ones(len(labels), dtype=bool)
    if mode is LatentVisualizationMode.CLIP_OVER_DINO:
        return (clip_predictions == labels) & (dino_predictions != labels)
    if mode is LatentVisualizationMode.DINO_OVER_CLIP:
        return (dino_predictions == labels) & (clip_predictions != labels)
    raise ValueError(f"Unsupported latent visualization mode: {mode}")


def _load_frozen_foundation_model(
    architecture: FoundationModelArchitecture,
    device: torch.device,
) -> tuple[Any, Any]:
    model, processor = load_foundation_model(architecture)
    model = model.to(device)
    model.eval()
    freeze(model)
    return model, processor


def run_latent_visualization(
    *,
    clip_repo: str,
    device: str = "auto",
    mode: str = LatentVisualizationMode.REGULAR.value,
    comparison_enabled: bool = False,
    dino_repo: str | None = None,
    clip_checkpoint_path: str | None = None,
    dino_checkpoint_path: str | None = None,
    clip_oof_bundle_path: str | None = None,
    dino_oof_bundle_path: str | None = None,
    feature_batch_size: int = 32,
    classifier_batch_size: int = 128,
):
    from digit_classifier.latent_visualization import plot_latent_space

    mode_enum = LatentVisualizationMode(mode)
    comparison_enabled = comparison_enabled or mode_enum is not LatentVisualizationMode.REGULAR
    resolved_device = get_device(device)

    clip_architecture = get_foundation_model(clip_repo, FoundationModelFamily.CLIP)
    clip_cfg = FoundationModelConfig(
        model=clip_architecture,
        family=FoundationModelFamily.CLIP,
        device=device,
        batch_size=classifier_batch_size,
        feature_batch_size=feature_batch_size,
        use_wandb=False,
    )
    clip_model, clip_processor = _load_frozen_foundation_model(clip_architecture, resolved_device)
    _, _, clip_features, clip_labels = compute_foundation_model_features(
        clip_model,
        clip_processor,
        clip_cfg,
        resolved_device,
    )
    images, dataset_labels, _, _ = load_mnist_in_the_wild(clip_cfg)
    images_np = images.numpy()
    labels_np = clip_labels.detach().cpu().numpy()

    if not np.array_equal(np.asarray(dataset_labels), labels_np):
        raise ValueError("Dataset labels do not match the CLIP feature labels")
    dataset_hash = compute_dataset_hash(images_np, labels_np)

    point_metadata = None
    title = "Latent Space (2D)"

    clip_embeddings = clip_features.detach().cpu().numpy()
    embedding_views: dict[str, np.ndarray] = {"clip": clip_embeddings}

    if comparison_enabled:
        bundle_mode = bool(clip_oof_bundle_path or dino_oof_bundle_path)
        if not dino_repo:
            raise ValueError(
                "Comparison modes require dino_repo and either matching checkpoints or matching OOF bundles"
            )

        dino_architecture = get_foundation_model(dino_repo, FoundationModelFamily.DINO)
        dino_cfg = FoundationModelConfig(
            model=dino_architecture,
            family=FoundationModelFamily.DINO,
            device=device,
            batch_size=classifier_batch_size,
            feature_batch_size=feature_batch_size,
            use_wandb=False,
        )
        dino_model, dino_processor = _load_frozen_foundation_model(dino_architecture, resolved_device)
        _, _, dino_features, dino_labels = compute_foundation_model_features(
            dino_model,
            dino_processor,
            dino_cfg,
            resolved_device,
        )
        if not torch.equal(clip_labels.cpu(), dino_labels.cpu()):
            raise ValueError("CLIP and DINO features were computed over different label orderings")

        embedding_views["dino"] = dino_features.detach().cpu().numpy()

        if bundle_mode:
            if not clip_oof_bundle_path or not dino_oof_bundle_path:
                raise ValueError(
                    "Comparison OOF bundle mode requires both clip_oof_bundle_path and dino_oof_bundle_path"
                )
            clip_bundle = load_oof_prediction_bundle(clip_oof_bundle_path)
            dino_bundle = load_oof_prediction_bundle(dino_oof_bundle_path)
            validate_oof_prediction_bundle(
                clip_bundle,
                expected_family=FoundationModelFamily.CLIP,
                expected_repo=clip_repo,
                expected_dataset_hash=dataset_hash,
                expected_labels=labels_np,
            )
            validate_oof_prediction_bundle(
                dino_bundle,
                expected_family=FoundationModelFamily.DINO,
                expected_repo=dino_repo,
                expected_dataset_hash=dataset_hash,
                expected_labels=labels_np,
                expected_plan_id=clip_bundle["plan_id"],
            )
            sample_idx = np.arange(len(labels_np), dtype=np.int64)
            clip_predictions = clip_bundle["predictions"]
            dino_predictions = dino_bundle["predictions"]
            clip_true_label_probabilities = clip_bundle["mean_probs"][sample_idx, labels_np]
            dino_true_label_probabilities = dino_bundle["mean_probs"][sample_idx, labels_np]
        else:
            if not clip_checkpoint_path or not dino_checkpoint_path:
                raise ValueError(
                    "Comparison checkpoint mode requires clip_checkpoint_path and dino_checkpoint_path"
                )

            clip_classifier = load_classifier_from_checkpoint(
                clip_checkpoint_path,
                resolved_device,
            )
            clip_probabilities = predict_classifier_probabilities(
                clip_classifier,
                clip_features,
                classifier_batch_size,
            )
            clip_predictions = clip_probabilities.argmax(axis=-1).astype(np.int64, copy=False)
            clip_true_label_probabilities = clip_probabilities[
                np.arange(len(labels_np), dtype=np.int64),
                labels_np,
            ]

            dino_classifier = load_classifier_from_checkpoint(
                dino_checkpoint_path,
                resolved_device,
            )
            dino_probabilities = predict_classifier_probabilities(
                dino_classifier,
                dino_features,
                classifier_batch_size,
            )
            dino_predictions = dino_probabilities.argmax(axis=-1).astype(np.int64, copy=False)
            dino_true_label_probabilities = dino_probabilities[
                np.arange(len(labels_np), dtype=np.int64),
                labels_np,
            ]

        point_metadata = {
            "clip_prediction": clip_predictions,
            "dino_prediction": dino_predictions,
            "clip_true_label_probability": clip_true_label_probabilities,
            "dino_true_label_probability": dino_true_label_probabilities,
        }

    result = plot_latent_space(
        clip_embeddings,
        labels_np,
        images=images_np,
        reducer="umap",
        reducer_kwargs=dict(n_neighbors=15, min_dist=0.1),
        model=clip_model,
        processor=clip_processor,
        device=str(resolved_device),
        point_metadata=point_metadata,
        title=title,
        embedding_views=embedding_views,
        initial_embedding_view="clip",
        embedding_view_labels={"clip": "CLIP", "dino": "DINO"},
        queryable_embedding_view="clip",
        initial_filter_mode=mode_enum.value,
    )

    if hasattr(result, "show"):
        result.show()
    return result


def train_foundation_model_fold(
    fold: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    cfg: FoundationModelConfig,
    wandb_config: dict,
    wandb_group: str,
    parent_run=None,
):
    train_dataset = FoundationModelFineTuningDataset(train_features, train_labels)
    val_dataset = FoundationModelFineTuningDataset(val_features, val_labels)

    train_generator = torch.Generator()
    train_generator.manual_seed(cfg.seed + fold)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    input_dim = train_features.shape[-1]
    classifier = build_classifier(cfg, input_dim).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    run = None
    if cfg.use_wandb and cfg.log_fold_runs and parent_run is None:
        run_name_prefix = wandb_group
        run = wandb.init(
            project=cfg.sweep_project,
            config={**wandb_config, "fold": fold, "n_folds": cfg.n_folds},
            group=wandb_group,
            job_type="cv-fold",
            name=f"{run_name_prefix}-fold-{fold}",
            reinit=True,
        )

        run.define_metric("global_step")
        run.define_metric("batch/*", step_metric="global_step")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")

    best_val_accuracy = float("-inf")
    best_epoch = 0
    best_checkpoint = None
    global_step = 0
    non_finite_detected = False
    interrupted = False
    epochs_since_improvement = 0

    try:
        for epoch in range(cfg.epochs):
            mean_train_loss = 0.0
            num_train_batches = 0
            num_train_correct = 0
            num_train_total = 0

            classifier.train()
            for batch in train_dataloader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    log.warning(
                        f"Fold {fold}, Epoch {epoch + 1}: non-finite training loss detected "
                        f"(loss={loss.item()}, head_type={cfg.head_type}, layer_norm={cfg.layer_norm}, "
                        f"lr={cfg.lr}, batch_size={cfg.batch_size})"
                    )
                    non_finite_detected = True
                    break
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()

                num_train_batches += 1
                mean_train_loss += (loss.item() - mean_train_loss) / num_train_batches
                new_correct = (outputs.argmax(dim=-1) == labels).sum().item()
                num_train_correct += new_correct
                num_train_total += labels.shape[0]

                global_step += 1
                batch_metrics = {
                    "global_step": global_step,
                    "batch/loss": loss.item(),
                    "batch/acc": new_correct / labels.shape[0],
                }
                if run is not None:
                    run.log(batch_metrics)

            if non_finite_detected:
                break

            train_accuracy = num_train_correct / num_train_total

            mean_val_loss = 0.0
            num_val_batches = 0
            num_val_correct = 0
            num_val_total = 0
            classifier.eval()
            with torch.inference_mode():
                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = classifier(images)
                    loss = criterion(outputs, labels)
                    if not torch.isfinite(loss):
                        log.warning(
                            f"Fold {fold}, Epoch {epoch + 1}: non-finite validation loss detected "
                            f"(loss={loss.item()}, head_type={cfg.head_type}, layer_norm={cfg.layer_norm}, "
                            f"lr={cfg.lr}, batch_size={cfg.batch_size})"
                        )
                        non_finite_detected = True
                        break
                    num_val_batches += 1
                    mean_val_loss += (loss.item() - mean_val_loss) / num_val_batches
                    batch_correct = (outputs.argmax(dim=-1) == labels).sum().item()
                    num_val_correct += batch_correct
                    num_val_total += labels.shape[0]

            if non_finite_detected:
                best_val_accuracy = float("-inf")
                best_epoch = epoch + 1
                if run is not None:
                    run.summary["failed_non_finite"] = 1
                    run.summary["failure_epoch"] = epoch + 1
                break

            val_accuracy = num_val_correct / num_val_total
            log.info(
                f"Fold {fold}, Epoch {epoch + 1}, Loss: {mean_train_loss:.4f}, "
                f"Val Loss: {mean_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )

            improved = val_accuracy > (best_val_accuracy + cfg.early_stopping_min_delta)
            if improved:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                epochs_since_improvement = 0
                log.info(f"Fold {fold}: new best validation accuracy: {best_val_accuracy:.4f}")
                best_checkpoint = {
                    "fold": fold,
                    "epoch": best_epoch,
                    "classifier_state_dict": copy.deepcopy(classifier.state_dict()),
                    "best_val_acc": best_val_accuracy,
                }
            else:
                epochs_since_improvement += 1

            epoch_metrics = {
                "epoch": epoch + 1,
                "train/loss": mean_train_loss,
                "train/acc": train_accuracy,
                "val/loss": mean_val_loss,
                "val/acc": val_accuracy,
                "val/best_acc": best_val_accuracy,
            }
            if run is not None:
                run.log(epoch_metrics)

            if (
                cfg.early_stopping_patience is not None
                and epochs_since_improvement >= cfg.early_stopping_patience
            ):
                log.info(
                    f"Fold {fold}: early stopping at epoch {epoch + 1} "
                    f"(best val acc {best_val_accuracy:.4f} at epoch {best_epoch})"
                )
                if run is not None:
                    run.summary["early_stopped"] = 1
                    run.summary["early_stop_epoch"] = epoch + 1
                break
    except KeyboardInterrupt:
        interrupted = True
        log.warning(f"Fold {fold}: interrupted by user")

    if run is not None:
        run.summary["best_val_acc"] = best_val_accuracy
        run.summary["best_epoch"] = best_epoch
        run.summary["fold"] = fold
        if interrupted:
            run.summary["interrupted"] = 1
        run.finish(exit_code=130 if interrupted else 0)

    return {
        "fold": fold,
        "best_val_acc": best_val_accuracy,
        "best_epoch": best_epoch,
        "checkpoint": best_checkpoint,
        "failed_non_finite": non_finite_detected,
        "interrupted": interrupted,
    }


def _resolve_split_plan(
    cfg: FoundationModelConfig,
    *,
    labels: np.ndarray,
    dataset_hash: str,
) -> dict[str, Any]:
    if cfg.split_plan_path:
        plan = load_split_plan(cfg.split_plan_path)
        validate_split_plan(plan, n_samples=len(labels), dataset_hash=dataset_hash)
        cfg.n_folds = int(plan["n_folds"])
        cfg.cv_repeats = int(plan["cv_repeats"])
    else:
        plan = build_repeated_stratified_split_plan(
            labels,
            n_folds=cfg.n_folds,
            cv_repeats=cfg.cv_repeats,
            seed=cfg.seed,
            dataset_hash=dataset_hash,
        )

    if cfg.save_split_plan_path:
        saved_plan_path = save_split_plan(plan, cfg.save_split_plan_path)
        log.info(f"Saved split plan to {saved_plan_path}")
    return plan


def run_foundation_model_fine_tuning(
    model: Any,
    processor: Any,
    device: torch.device,
    cfg: FoundationModelConfig,
    parent_run=None,
):
    seed_everything(cfg.seed)

    _, _, normalized_image_features, labels = compute_foundation_model_features(
        model, processor, cfg, device
    )
    features = normalized_image_features.detach().cpu()
    labels = labels.detach().cpu().long()
    dataset_images, dataset_labels, _, _ = load_mnist_in_the_wild(cfg)
    dataset_hash = compute_dataset_hash(dataset_images, dataset_labels)
    if not np.array_equal(np.asarray(dataset_labels), labels.numpy()):
        raise ValueError("Dataset labels do not match the cached foundation-model labels")

    if cfg.n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {cfg.n_folds}")
    if cfg.cv_repeats < 1:
        raise ValueError(f"cv_repeats must be at least 1, got {cfg.cv_repeats}")

    split_plan = _resolve_split_plan(
        cfg,
        labels=labels.numpy(),
        dataset_hash=dataset_hash,
    )

    wandb_config = dict(
        family=cfg.family.value,
        model=cfg.model.value.repo,
        patch_size=cfg.model.value.patch_size,
        image_size=cfg.model.value.image_size,
        model_size=cfg.model.value.model_size.value,
        zero_shot=cfg.zero_shot,
        head_type=cfg.head_type,
        linear_probe=cfg.linear_probe,
        epochs=cfg.epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_min_delta=cfg.early_stopping_min_delta,
        batch_size=cfg.batch_size,
        feature_batch_size=cfg.feature_batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        val_fraction=cfg.val_fraction,
        deep_mlp=cfg.deep_mlp,
        layer_norm=cfg.layer_norm,
        dropout=cfg.dropout,
        seed=cfg.seed,
        n_folds=cfg.n_folds,
        cv_repeats=cfg.cv_repeats,
        log_fold_runs=cfg.log_fold_runs,
        save_checkpoints=cfg.save_checkpoints,
        checkpoint_dir=cfg.checkpoint_dir,
        split_plan_path=cfg.split_plan_path,
        save_split_plan_path=cfg.save_split_plan_path,
        save_oof_bundle_path=cfg.save_oof_bundle_path,
        use_wandb=cfg.use_wandb,
    )

    if parent_run is not None:
        wandb_group = f"sweep-{parent_run.id}"
    else:
        wandb_group = (
            f"cv-{cfg.family.value}-{_sanitize_repo(cfg.model.value.repo)}-"
            f"linear{int(cfg.linear_probe)}-deep{int(cfg.deep_mlp)}-"
            f"ln{int(cfg.layer_norm)}-seed{cfg.seed}-r{cfg.cv_repeats}"
        )

    fold_results = []
    best_overall = None
    oof_probability_sums: np.ndarray | None = None
    oof_counts: np.ndarray | None = None
    if cfg.save_oof_bundle_path and parent_run is None:
        oof_probability_sums = np.zeros((len(labels), 10), dtype=np.float64)
        oof_counts = np.zeros(len(labels), dtype=np.int64)

    for split_number, split in enumerate(split_plan["splits"], start=1):
        repeat = int(split["repeat"])
        fold_in_repeat = int(split["fold"])
        train_idx = torch.as_tensor(split["train_idx"], dtype=torch.long)
        val_idx = torch.as_tensor(split["val_idx"], dtype=torch.long)

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        result = train_foundation_model_fold(
            fold=split_number,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            device=device,
            cfg=cfg,
            wandb_config=wandb_config,
            wandb_group=wandb_group,
            parent_run=parent_run,
        )
        result["repeat"] = repeat
        result["fold_in_repeat"] = fold_in_repeat
        if result["checkpoint"] is not None:
            result["checkpoint"].update(
                {
                    "repeat": repeat,
                    "fold_in_repeat": fold_in_repeat,
                    "seed": cfg.seed,
                    "n_folds": cfg.n_folds,
                    "cv_repeats": cfg.cv_repeats,
                    "family": cfg.family.value,
                    "repo": cfg.repo,
                    "dataset_hash": dataset_hash,
                    "split_plan_id": split_plan["plan_id"],
                    "train_idx": train_idx.cpu(),
                    "val_idx": val_idx.cpu(),
                }
            )
        fold_results.append(result)

        if (
            oof_probability_sums is not None
            and oof_counts is not None
            and result["checkpoint"] is not None
        ):
            classifier = load_classifier_from_state_dict(
                result["checkpoint"]["classifier_state_dict"],
                device,
            )
            val_probabilities = predict_classifier_probabilities(
                classifier,
                val_features,
                cfg.batch_size,
            )
            val_idx_np = val_idx.numpy()
            oof_probability_sums[val_idx_np] += val_probabilities
            oof_counts[val_idx_np] += 1

        if result["interrupted"]:
            log.warning("Cross-validation interrupted by user")
            break

        if best_overall is None or result["best_val_acc"] > best_overall["best_val_acc"]:
            best_overall = result

    if not fold_results:
        raise RuntimeError("No fold results were produced")

    fold_accuracies = [
        result["best_val_acc"] if np.isfinite(result["best_val_acc"]) else 0.0
        for result in fold_results
    ]
    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = float(np.std(fold_accuracies))

    log.info(f"Cross-validation accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    checkpoint_path: Path | None = None
    if cfg.save_checkpoints and best_overall is not None and parent_run is None:
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / (
            f"{cfg.family.value}_{_sanitize_repo(cfg.model.value.repo)}_"
            f"seed{cfg.seed}_best_repeat{best_overall.get('repeat', 1)}_"
            f"fold{best_overall.get('fold_in_repeat', best_overall['fold'])}.pt"
        )
        torch.save(best_overall["checkpoint"], checkpoint_path)
        log.info(f"Saved best checkpoint to {checkpoint_path}")

    oof_bundle_path: Path | None = None
    if cfg.save_oof_bundle_path and parent_run is None:
        if oof_probability_sums is None or oof_counts is None:
            raise RuntimeError("OOF bundle accumulation was not initialized")
        expected_counts = np.full(len(labels), cfg.cv_repeats, dtype=np.int64)
        if not np.array_equal(oof_counts, expected_counts):
            raise RuntimeError(
                "OOF bundle could not be completed because some samples do not have "
                "exactly cv_repeats held-out predictions"
            )
        mean_oof_probabilities = oof_probability_sums / oof_counts[:, None]
        oof_bundle_path = save_oof_prediction_bundle(
            cfg.save_oof_bundle_path,
            cfg=cfg,
            dataset_hash=dataset_hash,
            plan_id=split_plan["plan_id"],
            labels=labels.numpy(),
            mean_probs=mean_oof_probabilities,
            counts=oof_counts,
        )
        log.info(f"Saved OOF prediction bundle to {oof_bundle_path}")

    if cfg.use_wandb:
        fold_table = wandb.Table(
            columns=["split", "repeat", "fold", "best_val_acc", "best_epoch"],
            data=[
                [
                    result["fold"],
                    result.get("repeat", 1),
                    result.get("fold_in_repeat", result["fold"]),
                    result["best_val_acc"],
                    result["best_epoch"],
                ]
                for result in fold_results
            ],
        )

        if parent_run is not None:
            parent_run.summary["cv/mean_best_val_acc"] = mean_accuracy
            parent_run.summary["cv/std_best_val_acc"] = std_accuracy
            parent_run.summary["cv/best_fold"] = best_overall["fold"] if best_overall else None
            parent_run.summary["cv/best_repeat"] = (
                best_overall.get("repeat", 1) if best_overall else None
            )
            parent_run.summary["cv/best_fold_in_repeat"] = (
                best_overall.get("fold_in_repeat", best_overall["fold"])
                if best_overall
                else None
            )
            parent_run.summary["cv/best_fold_val_acc"] = (
                best_overall["best_val_acc"] if best_overall else None
            )
            parent_run.summary["cv/num_failed_folds"] = sum(
                int(result["failed_non_finite"]) for result in fold_results
            )
            parent_run.summary["cv/interrupted"] = int(
                any(result["interrupted"] for result in fold_results)
            )
            parent_run.log({"cv/folds": fold_table})
            for result in fold_results:
                summary_key = (
                    f"cv/repeat_{result.get('repeat', 1)}_"
                    f"fold_{result.get('fold_in_repeat', result['fold'])}"
                )
                parent_run.summary[f"{summary_key}_best_val_acc"] = result[
                    "best_val_acc"
                ]
                parent_run.summary[f"{summary_key}_best_epoch"] = result[
                    "best_epoch"
                ]
        else:
            aggregate_run = wandb.init(
                project=cfg.sweep_project,
                config={**wandb_config, "aggregate": True},
                group=wandb_group,
                job_type="cv-summary",
                name="cv-summary",
                reinit=True,
            )

            aggregate_run.summary["cv/mean_best_val_acc"] = mean_accuracy
            aggregate_run.summary["cv/std_best_val_acc"] = std_accuracy
            aggregate_run.summary["cv/best_fold"] = best_overall["fold"] if best_overall else None
            aggregate_run.summary["cv/best_repeat"] = (
                best_overall.get("repeat", 1) if best_overall else None
            )
            aggregate_run.summary["cv/best_fold_in_repeat"] = (
                best_overall.get("fold_in_repeat", best_overall["fold"])
                if best_overall
                else None
            )
            aggregate_run.summary["cv/best_fold_val_acc"] = (
                best_overall["best_val_acc"] if best_overall else None
            )
            aggregate_run.summary["cv/num_failed_folds"] = sum(
                int(result["failed_non_finite"]) for result in fold_results
            )
            aggregate_run.summary["cv/interrupted"] = int(
                any(result["interrupted"] for result in fold_results)
            )
            aggregate_run.log({"cv/folds": fold_table})
            aggregate_run.finish()

    return {
        "split_plan": split_plan,
        "fold_results": fold_results,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "best_overall": best_overall,
        "saved_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "saved_oof_bundle": str(oof_bundle_path) if oof_bundle_path is not None else None,
    }


def build_foundation_model_sweep_config(cfg: FoundationModelConfig) -> dict:
    """Build a valid W&B sweep config dict for foundation-model hyperparameter search."""
    if cfg.sweep_method == "grid":
        parameters = {
            "lr": {"values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]},
            "weight_decay": {"values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]},
            "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},
            "head_type": {"values": ["linear", "mlp", "deep_mlp"]},
            "layer_norm": {"values": [True, False]},
            "batch_size": {"values": [64, 128, 256]},
        }
    else:
        parameters = {
            "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
            "head_type": {"values": ["linear", "mlp", "deep_mlp"]},
            "layer_norm": {"values": [True, False]},
            "batch_size": {"values": [64, 128, 256]},
        }
    return {
        "method": cfg.sweep_method,
        "metric": {"name": "cv/mean_best_val_acc", "goal": "maximize"},
        "parameters": parameters,
    }


def run_foundation_model_sweep_trial(base_cfg: FoundationModelConfig):
    parent_run = None
    interrupted = False
    try:
        parent_run = wandb.init(project=base_cfg.sweep_project, job_type="sweep-trial")
        cfg = copy.deepcopy(base_cfg)
        wconfig = wandb.config

        sweep_overrides = {
            "lr": float,
            "weight_decay": float,
            "dropout": float,
            "head_type": str,
            "layer_norm": None,
            "batch_size": int,
        }

        for key, cast in sweep_overrides.items():
            value = getattr(wconfig, key, None)
            if value is None:
                continue
            setattr(cfg, key, value if cast is None else cast(value))

        cfg.linear_probe = cfg.head_type == "linear"
        cfg.deep_mlp = cfg.head_type == "deep_mlp"
        cfg.save_checkpoints = False
        cfg.save_split_plan_path = None
        cfg.save_oof_bundle_path = None
        cfg.use_wandb = True
        cfg.log_fold_runs = False

        parent_run.config.update(
            {
                "linear_probe": cfg.linear_probe,
                "deep_mlp": cfg.deep_mlp,
            },
            allow_val_change=True,
        )

        seed_everything(cfg.seed)
        model, processor = load_foundation_model(cfg.model)
        device = get_device(cfg.device)
        model = model.to(device)
        model.eval()
        freeze(model)

        run_foundation_model_fine_tuning(model, processor, device, cfg, parent_run=parent_run)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("Sweep trial interrupted by user")
    finally:
        if parent_run is not None:
            if interrupted:
                parent_run.summary["interrupted"] = 1
            parent_run.finish(exit_code=130 if interrupted else 0)

    if interrupted:
        raise KeyboardInterrupt


def create_foundation_model_sweep(cfg: FoundationModelConfig) -> str:
    sweep_config = build_foundation_model_sweep_config(cfg)
    sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.sweep_project)
    log.info(f"Created sweep: {sweep_id}")
    print(f"Sweep ID: {sweep_id}")
    return sweep_id


def run_foundation_model_sweep_agent(cfg: FoundationModelConfig) -> None:
    if not cfg.sweep_id:
        raise ValueError("sweep_id is required when sweep_action is 'agent'")

    def trial_fn():
        run_foundation_model_sweep_trial(cfg)

    try:
        wandb.agent(
            sweep_id=cfg.sweep_id,
            function=trial_fn,
            count=cfg.sweep_count,
            project=cfg.sweep_project,
        )
    except KeyboardInterrupt:
        log.warning("Sweep agent interrupted by user")


def run_foundation_model(cfg: FoundationModelConfig):
    seed_everything(cfg.seed)

    cfg.linear_probe = cfg.head_type == "linear"
    cfg.deep_mlp = cfg.head_type == "deep_mlp"

    model, processor = load_foundation_model(cfg.model)
    device = get_device(cfg.device)
    model = model.to(device)
    model.eval()

    freeze(model)

    if cfg.family == FoundationModelFamily.CLIP and cfg.zero_shot:
        run_clip_zero_shot(model, processor, device, cfg)
    else:
        run_foundation_model_fine_tuning(model, processor, device, cfg)
