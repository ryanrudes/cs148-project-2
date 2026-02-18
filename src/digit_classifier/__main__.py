"""Command-line interface for the digit classification pipeline.

Usage::

    python -m digit_classifier download
    python -m digit_classifier preprocess --color --size 224 --name mnist_rgb_224
    python -m digit_classifier train [--epochs 900] [--lr 1e-3] ...
    python -m digit_classifier infer --checkpoint best.pt
    python -m digit_classifier export-pipeline --checkpoint checkpoints/<run_id>/best.pt --output pipeline-cnn.pt
    python -m digit_classifier visualize [--num-batches 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support

from digit_classifier.config import AugmentConfig, Config, DataConfig, ModelConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _handle_download(args: argparse.Namespace) -> None:
    from digit_classifier.preprocessing import download_dataset
    download_dataset(url=args.url, output_zip=args.output, force=args.force)


def _handle_preprocess(args: argparse.Namespace) -> None:
    from digit_classifier.preprocessing import preprocess_and_cache
    preprocess_and_cache(
        dataset_name=args.name,
        color=args.color,
        size=args.size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        force=args.force,
    )


def _handle_train(args: argparse.Namespace) -> None:
    cfg = Config(
        data=DataConfig(
            dataset_name=args.dataset,
            image_size=args.size,
            color=args.color,
            train_fraction=args.train_fraction,
            batch_size=args.batch_size,
            split_seed=args.seed,
            mix_external=args.mix_external,
            primary_fraction=args.primary_fraction,
        ),
        model=ModelConfig(
            layers=tuple(args.layers),
            num_classes=args.num_classes,
            groups=args.groups,
            width_per_group=args.width_per_group,
            drop_path_rate=args.drop_path_rate,
        ),
        augment=AugmentConfig(),
        training=TrainingConfig(
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eta_min=args.eta_min,
            scheduler_t0=args.scheduler_t0,
            scheduler_t_mult=args.scheduler_t_mult,
            ema_decay=args.ema_decay,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            mixup_mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            mixup_off_last_n=args.mixup_off_last_n,
            grad_clip_norm=args.grad_clip_norm,
            compile_model=not args.no_compile,
            wandb_enabled=not args.no_wandb,
            wandb_project=args.wandb_project,
        ),
    )
    from digit_classifier.training import train
    train(cfg)


def _handle_infer(args: argparse.Namespace) -> None:
    from digit_classifier.inference import run_inference
    run_inference(
        checkpoint_path=args.checkpoint,
        layers=tuple(args.layers),
        num_classes=args.num_classes,
        groups=args.groups,
        width_per_group=args.width_per_group,
        input_size=args.size,
        camera_index=args.camera,
        smoothing_alpha=args.smoothing,
        device=args.device,
    )


def _handle_export_pipeline(args: argparse.Namespace) -> None:
    """Compile model + transforms into a TorchScript pipeline and save or push to HF Hub.

    The handler mirrors the `infer` model construction so the exported pipeline
    uses the same architecture and weights as the provided checkpoint.
    """
    import os
    import torch

    from digit_classifier.model import ResNeXt
    from digit_classifier.pipeline import DigitClassifierPipeline
    # reuse the small helper from inference to strip torch.compile prefixes
    from digit_classifier.inference import _strip_compile_prefix

    # load model to CPU for scripting
    dev = torch.device("cpu")
    model = ResNeXt(
        layers=list(args.layers),
        num_classes=args.num_classes,
        groups=args.groups,
        width_per_group=args.width_per_group,
    ).to(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
    model.eval()

    # determine mean/std from checkpoint if available (fallback to 0.5/0.5)
    if "mean" in ckpt and "std" in ckpt:
        mean = ckpt["mean"]
        std = ckpt["std"]
    else:
        mean = [0.5] * args.input_channels
        std = [0.5] * args.input_channels

    pipeline = DigitClassifierPipeline(
        model=model,
        input_size=args.size,
        input_channels=args.input_channels,
        mean=mean,
        std=std,
        device="cpu",
    )

    # Save locally
    pipeline.save_pipeline_local(args.output)
    print(f"Saved compiled pipeline to: {args.output}")

    # Optionally push to HuggingFace Hub
    if args.push_to_hf:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF token required for --push-to-hf (set HF_TOKEN or pass --hf-token)")
        if not args.hf_repo:
            raise RuntimeError("--hf-repo is required when --push-to-hf is used")
        pipeline.push_to_hub(token=token, repo_id=args.hf_repo, filename=args.hf_filename)


def _handle_push_cache(args: argparse.Namespace) -> None:
    from digit_classifier.hub import push_cache
    push_cache(repo_id=args.repo, cache_dir=args.cache_dir, private=not args.public)


def _handle_pull_cache(args: argparse.Namespace) -> None:
    from digit_classifier.hub import pull_cache
    pull_cache(repo_id=args.repo, cache_dir=args.cache_dir)


def _handle_visualize(args: argparse.Namespace) -> None:
    cfg = Config(
        data=DataConfig(dataset_name=args.dataset, image_size=args.size, color=args.color),
        model=ModelConfig(num_classes=args.num_classes),
        training=TrainingConfig(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        ),
    )
    from digit_classifier.visualize import visualize_batches
    visualize_batches(cfg, num_batches=args.num_batches)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digit_classifier",
        description="Digit classification training pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = sub.add_parser("download", help="Download raw dataset from Google Drive")
    dl.add_argument("--url", default=DataConfig.gdrive_url, help="Google Drive URL")
    dl.add_argument("--output", default="data/dataset.zip", help="Output zip path")
    dl.add_argument("--force", action="store_true", help="Re-download even if exists")

    # --- preprocess ---
    pp = sub.add_parser("preprocess", help="Preprocess raw images into cached .npz")
    pp.add_argument("--name", required=True, help="Dataset identifier (e.g. mnist_rgb_224)")
    pp.add_argument("--color", action="store_true", default=True, help="Use RGB (default)")
    pp.add_argument("--grayscale", action="store_true", help="Use grayscale instead of RGB")
    pp.add_argument("--size", type=int, default=224, help="Image size (default: 224)")
    pp.add_argument("--data-dir", default="data/dataset", help="Raw image directory")
    pp.add_argument("--output-dir", default="datasets", help="Output directory")
    pp.add_argument("--force", action="store_true", help="Overwrite existing cache")

    # --- train ---
    tr = sub.add_parser("train", help="Train the model")
    # Data
    tr.add_argument("--dataset", default="mnist_rgb_224")
    tr.add_argument("--size", type=int, default=224)
    tr.add_argument("--color", action="store_true", default=True)
    tr.add_argument("--train-fraction", type=float, default=0.9)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--no-external", dest="mix_external", action="store_false", default=True)
    tr.add_argument("--primary-fraction", type=float, default=0.95)
    # Model
    tr.add_argument("--layers", type=int, nargs="+", default=[3, 4, 23, 3])
    tr.add_argument("--num-classes", type=int, default=10)
    tr.add_argument("--groups", type=int, default=64)
    tr.add_argument("--width-per-group", type=int, default=4)
    tr.add_argument("--drop-path-rate", type=float, default=0.1)
    # Training
    tr.add_argument("--epochs", type=int, default=900)
    tr.add_argument("--warmup-epochs", type=int, default=20)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=0.05)
    tr.add_argument("--eta-min", type=float, default=1e-8)
    tr.add_argument("--scheduler-t0", type=int, default=50)
    tr.add_argument("--scheduler-t-mult", type=int, default=2)
    tr.add_argument("--ema-decay", type=float, default=0.995)
    tr.add_argument("--mixup-alpha", type=float, default=0.2)
    tr.add_argument("--cutmix-alpha", type=float, default=1.0)
    tr.add_argument("--mixup-prob", type=float, default=0.5)
    tr.add_argument("--mixup-mode", default="elem")
    tr.add_argument("--label-smoothing", type=float, default=0.1)
    tr.add_argument("--mixup-off-last-n", type=int, default=10)
    tr.add_argument("--grad-clip-norm", type=float, default=1.0)
    tr.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    tr.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    tr.add_argument("--wandb-project", default="CS148-MNIST")

    # --- infer ---
    inf = sub.add_parser("infer", help="Run webcam inference")
    inf.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    inf.add_argument("--layers", type=int, nargs="+", default=[3, 4, 23, 3])
    inf.add_argument("--num-classes", type=int, default=10)
    inf.add_argument("--groups", type=int, default=64)
    inf.add_argument("--width-per-group", type=int, default=4)
    inf.add_argument("--size", type=int, default=224)
    inf.add_argument("--camera", type=int, default=0)
    inf.add_argument("--smoothing", type=float, default=0.2)
    inf.add_argument("--device", default="auto")

    # --- export-pipeline ---
    ep = sub.add_parser(
        "export-pipeline",
        help="Compile model + transforms and save as a TorchScript pipeline",
    )
    ep.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ep.add_argument("--output", default="pipeline-cnn.pt", help="Local output path for compiled pipeline")
    ep.add_argument("--layers", type=int, nargs="+", default=[3, 4, 23, 3])
    ep.add_argument("--num-classes", type=int, default=10)
    ep.add_argument("--groups", type=int, default=64)
    ep.add_argument("--width-per-group", type=int, default=4)
    ep.add_argument("--size", type=int, default=224, help="Model input size (height=width)")
    ep.add_argument("--input-channels", type=int, choices=[1, 3], default=3)
    ep.add_argument("--push-to-hf", action="store_true", help="Upload compiled pipeline to HuggingFace Hub (requires HF_TOKEN or --hf-token and --hf-repo)")
    ep.add_argument("--hf-repo", help="HuggingFace repo id (e.g. username/repo)")
    ep.add_argument("--hf-filename", default="pipeline-cnn.pt", help="Filename to use on the Hub")
    ep.add_argument("--hf-token", help="HuggingFace token (optional; falls back to HF_TOKEN env var)")

    # --- push-cache ---
    pc = sub.add_parser("push-cache", help="Push dataset caches to HuggingFace Hub")
    pc.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. user/dataset-name)")
    pc.add_argument("--cache-dir", default="datasets", help="Local cache directory")
    pc.add_argument("--public", action="store_true", help="Make the repo public (default: private)")

    # --- pull-cache ---
    pl = sub.add_parser("pull-cache", help="Pull dataset caches from HuggingFace Hub")
    pl.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. user/dataset-name)")
    pl.add_argument("--cache-dir", default="datasets", help="Local cache directory")

    # --- visualize ---
    viz = sub.add_parser("visualize", help="Visualise augmented training batches")
    viz.add_argument("--dataset", default="mnist_rgb_224")
    viz.add_argument("--size", type=int, default=224)
    viz.add_argument("--color", action="store_true", default=True)
    viz.add_argument("--num-classes", type=int, default=10)
    viz.add_argument("--num-batches", type=int, default=1)
    viz.add_argument("--mixup-alpha", type=float, default=0.2)
    viz.add_argument("--cutmix-alpha", type=float, default=1.0)

    return parser


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    freeze_support()
    parser = _build_parser()
    args = parser.parse_args()

    # Handle --grayscale flag for preprocess
    if args.command == "preprocess" and args.grayscale:
        args.color = False

    handlers = {
        "download": _handle_download,
        "preprocess": _handle_preprocess,
        "train": _handle_train,
        "infer": _handle_infer,
        "export-pipeline": _handle_export_pipeline,
        "push-cache": _handle_push_cache,
        "pull-cache": _handle_pull_cache,
        "visualize": _handle_visualize,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
