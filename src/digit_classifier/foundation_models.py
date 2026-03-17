from __future__ import annotations

import copy
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
from rich.logging import RichHandler
from rich.progress import track
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


class FoundationModelFamily(Enum):
    CLIP = "clip"
    DINO = "dino"


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
    log_fold_runs: bool = True
    save_checkpoints: bool = False
    checkpoint_dir: str = "checkpoints"
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


def run_clip_zero_shot(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    cfg: FoundationModelConfig,
):
    if not cfg.supports_zero_shot:
        raise ValueError(f"Zero-shot is not supported for {cfg.family.value} models")

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total number of parameters in the model: {total_params}")

    logit_scale = model.logit_scale.exp()
    prompt_template = "an image of natural objects arranged to form the digit {digit}"

    with torch.inference_mode():
        _, _, normalized_image_features, labels = compute_foundation_model_features(
            model, processor, cfg, device
        )
        prompts = [prompt_template.format(digit=i) for i in range(10)]

        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_latents = model.text_model(**text_inputs)
        text_features = model.text_projection(text_latents.pooler_output)
        normalized_text_features = F.normalize(text_features, dim=-1)

        logits = logit_scale * (normalized_image_features @ normalized_text_features.T)
        preds = logits.argmax(dim=-1)

        num_correct = (preds == labels).sum().item()
        accuracy = num_correct / len(labels)
        log.info(f"Accuracy: {accuracy:.4f}")

        from digit_classifier.latent_visualization import plot_latent_space

        embeddings = normalized_image_features.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
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

    if cfg.n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {cfg.n_folds}")

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
        log_fold_runs=cfg.log_fold_runs,
        save_checkpoints=cfg.save_checkpoints,
        checkpoint_dir=cfg.checkpoint_dir,
        use_wandb=cfg.use_wandb,
    )

    if parent_run is not None:
        wandb_group = f"sweep-{parent_run.id}"
    else:
        wandb_group = (
            f"cv-{cfg.family.value}-{_sanitize_repo(cfg.model.value.repo)}-"
            f"linear{int(cfg.linear_probe)}-deep{int(cfg.deep_mlp)}-"
            f"ln{int(cfg.layer_norm)}-seed{cfg.seed}"
        )

    splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    split_indices = list(splitter.split(features.numpy(), labels.numpy()))

    fold_results = []
    best_overall = None

    for fold, (train_idx, val_idx) in enumerate(split_indices, start=1):
        train_idx = torch.as_tensor(train_idx, dtype=torch.long)
        val_idx = torch.as_tensor(val_idx, dtype=torch.long)

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        result = train_foundation_model_fold(
            fold=fold,
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
        fold_results.append(result)

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
            f"seed{cfg.seed}_best_fold{best_overall['fold']}.pt"
        )
        torch.save(best_overall["checkpoint"], checkpoint_path)
        log.info(f"Saved best checkpoint to {checkpoint_path}")

    if cfg.use_wandb:
        fold_table = wandb.Table(
            columns=["fold", "best_val_acc", "best_epoch"],
            data=[
                [result["fold"], result["best_val_acc"], result["best_epoch"]]
                for result in fold_results
            ],
        )

        if parent_run is not None:
            parent_run.summary["cv/mean_best_val_acc"] = mean_accuracy
            parent_run.summary["cv/std_best_val_acc"] = std_accuracy
            parent_run.summary["cv/best_fold"] = best_overall["fold"] if best_overall else None
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
                parent_run.summary[f"cv/fold_{result['fold']}_best_val_acc"] = result[
                    "best_val_acc"
                ]
                parent_run.summary[f"cv/fold_{result['fold']}_best_epoch"] = result[
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
        "fold_results": fold_results,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "best_overall": best_overall,
        "saved_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
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
