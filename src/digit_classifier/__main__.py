"""Command-line interface for the digit classification pipeline.

Usage::

    python -m digit_classifier download
    python -m digit_classifier preprocess --color --size 224 --name mnist_rgb_224
    python -m digit_classifier train [--epochs 900] [--lr 1e-3] ...
    python -m digit_classifier infer --checkpoint best.pt
    python -m digit_classifier export-pipeline --checkpoint checkpoints/<run_id>/best.pt --output pipeline-cnn.pt
    python -m digit_classifier generate-pareidolia [--out dataset_out] [--per-digit 50] [--batch]
    python -m digit_classifier push-test-dataset --repo user/pareidolia-test [--dataset-dir dataset_out]
    python -m digit_classifier pull-test-dataset --repo user/pareidolia-test [--dataset-dir dataset_out]
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
            repeat_aug=args.repeat_aug,
            repeat_aug_repeats=args.repeat_aug_repeats,
            split_seed=args.seed,
            mix_external=args.mix_external,
            external_only=getattr(args, "external_only", False),
            external_val_source=getattr(args, "external_val_source", "MNIST Test"),
            external_val_split=getattr(args, "external_val_split", False),
            external_val_fraction=getattr(args, "external_val_fraction", 0.1),
            primary_fraction=args.primary_fraction,
            test_dataset_path=args.test_dataset,
            test_preload=not getattr(args, "no_preload_test", False),
            augment_scheme=args.augment_scheme,
            num_workers=args.num_workers,
            calibrate_workers=getattr(args, "calibrate_workers", False),
            prefetch_factor=args.prefetch_factor,
            external_cache=args.external_cache,
            external_cache_max_mb=args.external_cache_max_mb,
        ),
        model=ModelConfig(
            model_type=args.model_type,
            layers=tuple(args.layers),
            num_classes=args.num_classes,
            groups=args.groups,
            width_per_group=args.width_per_group,
            drop_path_rate=args.drop_path_rate,
            use_flash_attention={"auto": None, "on": True, "off": False}[args.use_flash_attention],
            deit_model=args.deit_model,
            layer_scale_init=args.layer_scale_init,
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
            ema_enabled=not args.no_ema,
            warm_restarts=not args.no_warm_restarts,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            cutmix_minmax=None if args.no_cutmix_minmax else (0.02, 0.45),
            mixup_prob=args.mixup_prob,
            mixup_mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            bce_loss=args.bce_loss,
            mixup_off_last_n=args.mixup_off_last_n,
            drop_path_increment=args.drop_path_increment,
            drop_path_increment_every=args.drop_path_increment_every,
            grad_clip_norm=args.grad_clip_norm,
            weight_decay_exclude=args.weight_decay_exclude,
            layer_decay=args.layer_decay,
            amp_enabled=args.amp_enabled,
            amp_dtype=args.amp_dtype,
            compile_model=not args.no_compile,
            compile_mode=getattr(args, "compile_mode", None),
            wandb_enabled=not args.no_wandb,
            wandb_project=args.wandb_project,
            replace_best_checkpoint=args.replace_best_checkpoint,
            checkpoint_enabled=not args.no_checkpoint,
            checkpoint_latest=getattr(args, "checkpoint_latest", False),
            val_every_n_epochs=args.val_every_n_epochs,
            progress_bars=getattr(args, "progress_bars", False),
            show_data_wait=getattr(args, "show_data_wait", False),
            wandb_watch=args.wandb_watch,
            resume_path=getattr(args, "resume", None),
            pretrain_path=getattr(args, "pretrain", None),
        ),
    )
    from digit_classifier.training import train
    train(cfg)


def _handle_infer(args: argparse.Namespace) -> None:
    from digit_classifier.inference import run_inference
    run_inference(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        input_size=args.size,
        input_channels=args.input_channels,
        camera_index=args.camera,
        smoothing_alpha=args.smoothing,
        device=args.device,
        mean=args.mean,
        std=args.std,
    )


def _handle_export_pipeline(args: argparse.Namespace) -> None:
    """Compile model + transforms into a TorchScript pipeline and save or push to HF Hub.

    The handler uses build_model_from_checkpoint so the exported pipeline
    uses the same architecture (ResNeXt or DeiT) and weights as the checkpoint.
    """
    import os
    import torch

    from digit_classifier.training import build_model_from_checkpoint
    from digit_classifier.pipeline import DigitClassifierPipeline

    dev = torch.device("cpu")
    model, ckpt = build_model_from_checkpoint(
        args.checkpoint, dev, model_type=getattr(args, "model_type", None)
    )

    # ViT/DeiT uploads must use pipeline-vit.pt (ee148a-project requirement)
    is_vit = hasattr(model, "patch_embed")
    if is_vit and args.hf_filename == "pipeline-cnn.pt":
        args.hf_filename = "pipeline-vit.pt"
    if is_vit and args.output == "pipeline-cnn.pt":
        args.output = "pipeline-vit.pt"

    input_size = args.size if args.size is not None else ckpt.get("model_config", {}).get("image_size", 224)

    # Determine mean/std: checkpoint > CLI args > fallback 0.5/0.5
    if "mean" in ckpt and "std" in ckpt:
        mean = list(ckpt["mean"])
        std = list(ckpt["std"])
    elif args.mean is not None and args.std is not None:
        mean = list(args.mean)
        std = list(args.std)
        if len(mean) != args.input_channels or len(std) != args.input_channels:
            raise ValueError(
                f"--mean and --std must have {args.input_channels} values each "
                f"(got {len(mean)} and {len(std)})"
            )
    else:
        mean = [0.5] * args.input_channels
        std = [0.5] * args.input_channels
        print(
            "Warning: Checkpoint has no mean/std; using 0.5/0.5. "
            "Pass --mean and --std to match your training normalization."
        )

    pipeline = DigitClassifierPipeline(
        model=model,
        input_size=input_size,
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


def _handle_push_test_dataset(args: argparse.Namespace) -> None:
    from digit_classifier.hub import push_test_dataset
    push_test_dataset(
        repo_id=args.repo,
        dataset_dir=args.dataset_dir,
        private=not args.public,
    )


def _handle_pull_test_dataset(args: argparse.Namespace) -> None:
    from digit_classifier.hub import pull_test_dataset
    pull_test_dataset(repo_id=args.repo, dataset_dir=args.dataset_dir)


def _handle_eval(args: argparse.Namespace) -> None:
    from digit_classifier.training import run_eval
    run_eval(
        checkpoint_path=args.checkpoint,
        test_dataset_path=args.test_dataset,
        dataset_name=args.dataset,
        image_size=args.size if args.size is not None else None,
        batch_size=args.batch_size,
        device=args.device,
        test_preload=not getattr(args, "no_preload_test", False),
    )


def _handle_generate_pareidolia(args: argparse.Namespace) -> None:
    try:
        from digit_classifier.pareidolia_generate import run, list_resolution_options
    except ImportError as e:
        raise SystemExit(
            "Pareidolia generation requires optional dependencies. "
            "Install with: pip install digit-classifier[pareidolia]\n"
            f"Original error: {e}"
        ) from e
    if args.list_resolutions:
        list_resolution_options()
        return
    run(
        out=args.out,
        digits=args.digits,
        per_digit=args.per_digit,
        provider=args.provider,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        image_model=args.image_model,
        size=args.size,
        quality=args.quality,
        output_format=args.format,
        sleep=args.sleep,
        seed=args.seed,
        gemini_model=args.gemini_model,
        gemini_imagen_model=args.gemini_imagen_model,
        gemini_llm_model=args.gemini_llm_model,
        use_batch=args.batch,
        batch_poll_seconds=args.batch_poll_seconds,
        skip_confirm=args.yes,
        max_retries=args.max_retries,
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        image_size=args.image_size,
        temperature=args.temperature,
    )


def _handle_visualize(args: argparse.Namespace) -> None:
    cfg = Config(
        data=DataConfig(
            dataset_name=args.dataset,
            image_size=args.size,
            color=args.color,
            batch_size=args.batch_size,
            repeat_aug=args.repeat_aug,
            repeat_aug_repeats=args.repeat_aug_repeats,
            primary_fraction=args.primary_fraction,
            mix_external=args.mix_external,
            train_fraction=args.train_fraction,
            split_seed=args.seed,
            augment_scheme=args.augment_scheme,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            external_cache=args.external_cache,
            external_cache_max_mb=args.external_cache_max_mb,
        ),
        model=ModelConfig(num_classes=args.num_classes),
        training=TrainingConfig(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            cutmix_minmax=None if args.no_cutmix_minmax else (0.02, 0.45),
            mixup_prob=args.mixup_prob,
            mixup_mode=args.mixup_mode,
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
    tr.add_argument("--repeat-aug", action="store_true",
                    help="Enable repeated augmentation (DeiT-III / timm RASampler)")
    tr.add_argument("--repeat-aug-repeats", type=int, default=3,
                    help="Number of repeats per sample when using repeated augmentation")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--no-external", dest="mix_external", action="store_false", default=True)
    tr.add_argument("--external-only", action="store_true",
                    help="Train only on external data; skip loading primary dataset")
    tr.add_argument("--external-val-source", type=str, default="MNIST Test",
                    help="External source for validation when --external-only (ignored if --external-val-split)")
    tr.add_argument("--external-val-split", action="store_true",
                    help="When --external-only: val = random subset of union; else val = one held-out source")
    tr.add_argument("--external-val-fraction", type=float, default=0.1,
                    help="Validation fraction when --external-val-split (default: 0.1)")
    tr.add_argument("--primary-fraction", type=float, default=0.95)
    tr.add_argument("--test-dataset", type=str, default=None,
                    help="Pareidolia output dir (e.g. dataset_out) for test evaluation; no augmentation")
    tr.add_argument("--no-preload-test", action="store_true",
                    help="Load pareidolia test images on demand instead of preloading (for huge test sets)")
    tr.add_argument("--augment-scheme", default="yolo",
                    choices=["yolo", "three_augment", "autoaugment"],
                    help="Augmentation pipeline: yolo (default), three_augment (DeiT-III), autoaugment (SVHN)")
    tr.add_argument("--num-workers", type=int, default=-1,
                    help="DataLoader workers (-1 = auto: cpu_count-1, capped at 16)")
    tr.add_argument("--prefetch-factor", type=int, default=2,
                    help="DataLoader prefetch factor per worker")
    tr.add_argument("--external-cache", action="store_true",
                    help="Enable LRU cache for external datasets (use with --external-cache-max-mb)")
    tr.add_argument("--external-cache-max-mb", type=float, default=2048,
                    help="Max MB for external sample cache (0=disabled, -1=auto from RAM). Use with --external-cache.")
    tr.add_argument("--calibrate-workers", action="store_true",
                    help="Benchmark num_workers before training and use the fastest")
    # Model
    tr.add_argument("--model", dest="model_type", type=str, default="deit",
                    choices=["resnext", "deit"],
                    help="Model architecture: resnext or deit (default: deit)")
    tr.add_argument("--layers", type=int, nargs="+", default=[3, 4, 23, 3],
                    help="ResNeXt layer config (e.g. 3 4 23 3); ignored for deit")
    tr.add_argument("--num-classes", type=int, default=10)
    tr.add_argument("--groups", type=int, default=64)
    tr.add_argument("--width-per-group", type=int, default=4)
    tr.add_argument("--drop-path-rate", type=float, default=0.1)
    tr.add_argument("--deit-model", choices=["tiny", "small", "base", "large", "huge"], default="base",
                    help="DeiT-III model size: tiny, small, base, large (304M), huge (632M, patch14)")
    tr.add_argument("--flash-attention", dest="use_flash_attention", choices=["auto", "on", "off"],
                    default="auto",
                    help="Use SDPA (Flash Attention when available): auto (default for DeiT+CUDA), on, off")
    tr.add_argument("--layer-scale-init", type=float, default=1e-4,
                    help="LayerScale initialization value (DeiT-III uses 1e-4)")
    # Training
    tr.add_argument("--epochs", type=int, default=900)
    tr.add_argument("--warmup-epochs", type=int, default=20)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=0.05)
    tr.add_argument("--eta-min", type=float, default=1e-5,
                    help="Min LR for cosine scheduler (DeiT-III uses 1e-5)")
    tr.add_argument("--scheduler-t0", type=int, default=50)
    tr.add_argument("--scheduler-t-mult", type=int, default=2)
    tr.add_argument("--ema-decay", type=float, default=0.995)
    tr.add_argument("--mixup-alpha", type=float, default=0.2)
    tr.add_argument("--cutmix-alpha", type=float, default=1.0)
    tr.add_argument("--no-cutmix-minmax", action="store_true",
                    help="Use standard CutMix (DeiT-III style) instead of minmax bbox sampling")
    tr.add_argument("--mixup-prob", type=float, default=0.5)
    tr.add_argument("--mixup-mode", default="elem")
    tr.add_argument("--label-smoothing", type=float, default=0.1)
    tr.add_argument("--bce-loss", dest="bce_loss", action="store_true", default=False,
                    help="Use binary cross-entropy (DeiT-III style) instead of cross-entropy")
    tr.add_argument("--mixup-off-last-n", type=int, default=10)
    tr.add_argument("--drop-path-increment", type=float, default=0.0,
                    help="Add this much to stochastic depth every N epochs (0 = disabled)")
    tr.add_argument("--drop-path-increment-every", type=int, default=0,
                    help="Increment stochastic depth every this many epochs (0 = disabled)")
    tr.add_argument("--grad-clip-norm", type=float, default=1.0)
    tr.add_argument("--no-weight-decay-exclusion", dest="weight_decay_exclude", action="store_false", default=True,
                    help="Apply weight decay to all params (default: exclude LayerNorm, bias, LayerScale)")
    tr.add_argument("--layer-decay", type=float, default=0.0,
                    help="Layer-wise LR decay (e.g. 0.75). 0 disables. Earlier layers get lower LR.")
    tr.add_argument("--no-amp", dest="amp_enabled", action="store_false", default=True,
                    help="Disable mixed precision (autocast); use float32 for stability")
    tr.add_argument("--amp-dtype", dest="amp_dtype", choices=["auto", "float16", "bfloat16"],
                    default="auto",
                    help="AMP dtype: auto (bf16 if supported else fp16), float16, bfloat16")
    tr.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    tr.add_argument("--compile-mode", dest="compile_mode", choices=["default", "reduce-overhead", "max-autotune"],
                    default=None,
                    help="torch.compile mode (default: profile and choose best). Use to skip profiling.")
    tr.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    tr.add_argument("--no-checkpoint", action="store_true",
                    help="Disable saving checkpoints to disk and wandb")
    tr.add_argument("--val-every-n", dest="val_every_n_epochs", type=int, default=1,
                    help="Run validation every N epochs (1 = every epoch)")
    tr.add_argument("--progress-bars", action="store_true",
                    help="Show Rich progress bars for each epoch's train/val batches")
    tr.add_argument("--show-data-wait", action="store_true",
                    help="Show %% of time GPU waits for data in progress bar (enables progress bars)")
    tr.add_argument("--wandb-watch", default="gradients",
                    choices=["gradients", "all", "none"],
                    help="wandb.watch mode: gradients, all, or none")
    tr.add_argument("--replace-best-checkpoint", action="store_true", default=True,
                    help="Overwrite best checkpoint on disk and wandb (default)")
    tr.add_argument("--accumulate-best-checkpoints", dest="replace_best_checkpoint", action="store_false",
                    help="Keep separate checkpoint versions instead of overwriting")
    tr.add_argument("--checkpoint-latest", action="store_true",
                    help="Save latest.pt after every epoch (always overwrite); enables resume from last epoch")
    tr.add_argument("--no-ema", action="store_true", help="Disable EMA")
    tr.add_argument("--no-warm-restarts", action="store_true",
                        help="Disable cosine warm-restart scheduler")
    tr.add_argument("--wandb-project", default="CS148-MNIST")
    tr.add_argument("--resume", type=str, default=None,
                    help="Resume training from checkpoint (same resolution; restores optimizer/scheduler)")
    tr.add_argument("--pretrain", type=str, default=None,
                    help="Fine-tune from checkpoint (load weights only; allows different resolution). "
                         "See docs/FINETUNING.md for recommended --drop-path-rate 0 and --layer-decay 0.65.")

    # --- infer ---
    inf = sub.add_parser("infer", help="Run webcam inference")
    inf.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    inf.add_argument("--layers", type=int, nargs="+", default=[3, 4, 23, 3])
    inf.add_argument("--num-classes", type=int, default=10)
    inf.add_argument("--groups", type=int, default=64)
    inf.add_argument("--width-per-group", type=int, default=4)
    inf.add_argument("--size", type=int, default=None,
                     help="Input size (default: from checkpoint, else 224)")
    inf.add_argument("--input-channels", type=int, choices=[1, 3], default=3)
    inf.add_argument("--camera", type=int, default=0)
    inf.add_argument("--smoothing", type=float, default=0.2)
    inf.add_argument("--device", default="auto")
    inf.add_argument("--mean", type=float, nargs="+", help="Per-channel mean (e.g. 0.57 0.52 0.48). Required if checkpoint lacks mean/std.")
    inf.add_argument("--std", type=float, nargs="+", help="Per-channel std (e.g. 0.23 0.23 0.23). Required if checkpoint lacks mean/std.")

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
    ep.add_argument("--size", type=int, default=None,
                    help="Model input size (default: from checkpoint, else 224)")
    ep.add_argument("--input-channels", type=int, choices=[1, 3], default=3)
    ep.add_argument("--push-to-hf", action="store_true", help="Upload compiled pipeline to HuggingFace Hub (requires HF_TOKEN or --hf-token and --hf-repo)")
    ep.add_argument("--hf-repo", help="HuggingFace repo id (e.g. username/repo)")
    ep.add_argument("--hf-filename", default="pipeline-cnn.pt",
                    help="Filename on the Hub (default: pipeline-vit.pt for ViT/DeiT, pipeline-cnn.pt for ResNeXt)")
    ep.add_argument("--hf-token", help="HuggingFace token (optional; falls back to HF_TOKEN env var)")
    ep.add_argument("--mean", type=float, nargs="+", help="Per-channel mean for normalization (e.g. 0.13 0.13 0.13 for RGB). Required if checkpoint lacks mean/std.")
    ep.add_argument("--std", type=float, nargs="+", help="Per-channel std for normalization (e.g. 0.31 0.31 0.31 for RGB). Required if checkpoint lacks mean/std.")

    # --- push-cache ---
    pc = sub.add_parser("push-cache", help="Push dataset caches to HuggingFace Hub")
    pc.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. user/dataset-name)")
    pc.add_argument("--cache-dir", default="datasets", help="Local cache directory")
    pc.add_argument("--public", action="store_true", help="Make the repo public (default: private)")

    # --- pull-cache ---
    pl = sub.add_parser("pull-cache", help="Pull dataset caches from HuggingFace Hub")
    pl.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. user/dataset-name)")
    pl.add_argument("--cache-dir", default="datasets", help="Local cache directory")

    # --- push-test-dataset ---
    pts = sub.add_parser("push-test-dataset", help="Push pareidolia test dataset to HuggingFace Hub")
    pts.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. user/pareidolia-test)")
    pts.add_argument("--dataset-dir", default="dataset_out", help="Local pareidolia output directory")
    pts.add_argument("--public", action="store_true", help="Make the repo public (default: private)")

    # --- pull-test-dataset ---
    pld = sub.add_parser("pull-test-dataset", help="Pull pareidolia test dataset from HuggingFace Hub")
    pld.add_argument("--repo", required=True, help="HuggingFace repo id (e.g. user/pareidolia-test)")
    pld.add_argument("--dataset-dir", default="dataset_out", help="Local directory to download into")

    # --- eval ---
    ev = sub.add_parser("eval", help="Evaluate checkpoint (EMA) on pareidolia test dataset")
    ev.add_argument("--checkpoint", required=True, help="Path to checkpoint (e.g. checkpoints/resnext.pt)")
    ev.add_argument("--test-dataset", required=True, help="Pareidolia test dir (e.g. dataset_out)")
    ev.add_argument("--dataset", default="mnist_rgb_224", help="Cached dataset name for mean/std (default: mnist_rgb_224)")
    ev.add_argument("--size", type=int, default=None,
                    help="Image size (default: from checkpoint, else 224)")
    ev.add_argument("--batch-size", type=int, default=128)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--no-preload-test", action="store_true",
                    help="Load test images on demand instead of preloading")

    # --- generate-pareidolia ---
    gp = sub.add_parser(
        "generate-pareidolia",
        help="Generate AI dataset where digits 0-9 are implied by OOD real-world objects (pareidolia)",
    )
    gp.add_argument("--out", type=str, default="dataset_out", help="Output directory")
    gp.add_argument("--digits", type=str, default="0-9",
                    help='Digits to generate, e.g. "0-9" or "2,7,9"')
    gp.add_argument("--per-digit", type=int, default=50, help="Images per digit")
    gp.add_argument("--provider", type=str, default="openai",
                    choices=["openai", "gemini", "gemini-imagen"],
                    help="Image provider: openai, gemini (native), gemini-imagen")
    gp.add_argument("--llm-provider", type=str, default=None,
                    choices=["openai", "gemini", "gemini-imagen"],
                    help="LLM for prompts (default: same as --provider)")
    gp.add_argument("--llm-model", type=str, default=None,
                    help="LLM model for prompts (default: env LLM_MODEL or model per provider)")
    gp.add_argument("--gemini-llm-model", type=str, default=None,
                    help="Gemini LLM for prompts, e.g. gemini-3.1-flash-lite-preview (default: env GEMINI_LLM_MODEL)")
    gp.add_argument("--image-model", type=str, default=None,
                    help="OpenAI image model (default: env IMAGE_MODEL or gpt-image-1)")
    gp.add_argument("--gemini-model", type=str, default=None,
                    help="Gemini native image model, e.g. gemini-3.1-flash-image-preview (default: env GEMINI_IMAGE_MODEL)")
    gp.add_argument("--gemini-imagen-model", type=str, default=None,
                    help="Gemini Imagen model (default: env GEMINI_IMAGEN_MODEL)")
    gp.add_argument("--size", type=str, default="1024x1024",
                    help='OpenAI image size. Options: auto, 1024x1024, 1536x1024, 1024x1536')
    gp.add_argument("--image-size", type=str, default=None,
                    help='Gemini/Imagen resolution: 1K, 2K, 4K (default: 1K)')
    gp.add_argument("--list-resolutions", action="store_true",
                    help="Print available resolution options and exit")
    gp.add_argument("--quality", type=str, default="auto",
                    help='OpenAI GPT image: "auto", "high", "medium", "low"')
    gp.add_argument("--format", type=str, default="png", choices=["png", "jpeg", "webp"])
    gp.add_argument("--sleep", type=float, default=0.2,
                    help="Seconds to sleep between samples (sync mode). Use 2+ for Gemini free tier.")
    gp.add_argument("--max-retries", type=int, default=5,
                    help="Max retries on 429 rate limit (Gemini sync mode; default: 5)")
    gp.add_argument("--thinking-budget", type=int, default=None,
                    help="Gemini thinking budget in tokens (2.5 models). 0=off, -1=dynamic. E.g. 1024, 8192.")
    gp.add_argument("--thinking-level", type=str, default=None,
                    choices=["low", "high"],
                    help="Gemini thinking level (3.x models): low or high")
    gp.add_argument("--temperature", type=float, default=None,
                    help="LLM temperature for prompt generation (0–2). Higher = more variety. Try 0.9–1.0 for creative prompts.")
    gp.add_argument("--seed", type=int, default=None,
                    help="Optional RNG seed for repeatable digit ordering")
    gp.add_argument("--batch", action="store_true",
                    help="Use Batch API (OpenAI or Gemini; 50%% cost reduction, ~24h window)")
    gp.add_argument("--batch-poll-seconds", type=float, default=10.0,
                    help="Seconds between batch status polls (default: 10)")
    gp.add_argument("--yes", "-y", action="store_true",
                    help="Skip cost confirmation prompt")

    # --- visualize ---
    viz = sub.add_parser("visualize", help="Visualise augmented training batches")
    viz.add_argument("--dataset", default="mnist_rgb_224")
    viz.add_argument("--augment-scheme", default="yolo",
                     choices=["yolo", "three_augment", "autoaugment"],
                     help="Augmentation pipeline to visualise")
    viz.add_argument("--size", type=int, default=224)
    viz.add_argument("--color", action="store_true", default=True)
    viz.add_argument("--num-classes", type=int, default=10)
    viz.add_argument("--num-batches", type=int, default=1)
    viz.add_argument("--batch-size", type=int, default=128)
    viz.add_argument("--repeat-aug", action="store_true",
                     help="Enable repeated augmentation (same as train --repeat-aug)")
    viz.add_argument("--repeat-aug-repeats", type=int, default=3,
                     help="Repeats per sample when using repeated augmentation")
    viz.add_argument("--primary-fraction", type=float, default=0.95,
                     help="Fraction of batch from primary dataset when mixing external")
    viz.add_argument("--no-external", dest="mix_external", action="store_false", default=True,
                     help="Disable external dataset mixing")
    viz.add_argument("--train-fraction", type=float, default=0.9)
    viz.add_argument("--seed", type=int, default=42)
    viz.add_argument("--mixup-alpha", type=float, default=0.2)
    viz.add_argument("--cutmix-alpha", type=float, default=1.0)
    viz.add_argument("--no-cutmix-minmax", action="store_true",
                     help="Use standard CutMix (DeiT-III style) instead of minmax bbox sampling")
    viz.add_argument("--mixup-prob", type=float, default=0.5)
    viz.add_argument("--mixup-mode", default="elem")
    viz.add_argument("--num-workers", type=int, default=-1,
                     help="DataLoader workers (-1 = auto)")
    viz.add_argument("--prefetch-factor", type=int, default=2)
    viz.add_argument("--external-cache", action="store_true",
                     help="Enable LRU cache for external datasets")
    viz.add_argument("--external-cache-max-mb", type=float, default=2048)

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
        "push-test-dataset": _handle_push_test_dataset,
        "pull-test-dataset": _handle_pull_test_dataset,
        "eval": _handle_eval,
        "generate-pareidolia": _handle_generate_pareidolia,
        "visualize": _handle_visualize,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
