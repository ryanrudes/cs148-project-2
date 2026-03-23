# Digit Classifier

https://github.com/user-attachments/assets/e00540d3-96e7-4360-9dc5-eb5e3a0b5e29


https://github.com/user-attachments/assets/53767d51-1e46-442c-ae86-29178b18e6b4


https://github.com/user-attachments/assets/7913d9f3-2399-483e-9ce6-53ac047e823b


https://github.com/user-attachments/assets/b402ce11-aa25-47d0-b2e9-c745f2e88d0c



A PyTorch training pipeline for digit classification using **ResNeXt** with
YOLO-style augmentation, external dataset mixing, and comprehensive experiment
tracking.

For wandb logs, see [here](https://wandb.ai/ryanrudes-caltech-California%20Institute%20of%20Technology%20-%20C/mnist-in-the-wild-clip/workspace?nw=nwuserryanrudescaltech) and [here](https://wandb.ai/ryanrudes-caltech-California%20Institute%20of%20Technology%20-%20C/mnist-in-the-wild-dino?nw=nwuserryanrudescaltech).

## Features

- **ResNeXt-101** with stochastic depth (drop-path) and grouped convolutions
- **YOLO-style augmentation** pipeline with digit-safe hyper-parameters (label-conditional horizontal flip for symmetric digits 0 and 8)
- **External dataset mixing** — SVHN, MNIST, EMNIST, USPS, QMNIST and Semeion are lazily loaded and mixed into training via a `RatioBatchSampler` (95 % original / 5 % external per batch)
- **NIST-like deduplication** — SHA-1 fingerprints remove exact pixel duplicates across external datasets (cached as a tiny JSON so dedup only runs once)
- **Mixup / CutMix** (element-wise mode) with dynamic loss switching — `SoftTargetCrossEntropy` while active, plain `CrossEntropyLoss` when disabled for the final *N* epochs
- **EMA model** with `use_buffers=True`
- **Warm-restart scheduler** — linear warmup followed by `CosineAnnealingWarmRestarts` with pre-restart checkpoints
- **AMP** with `GradScaler` (CUDA-only) and gradient norm clipping
- **Weights & Biases** integration for experiment tracking and artifact storage
- **Rich** console output — tables, progress bars and structured logging
- **HuggingFace Hub** integration for pushing/pulling dataset caches and pareidolia test datasets
- **Webcam inference** with real-time probability visualisation
- **Pareidolia test generation** — AI-generated OOD images (digits implied by real-world objects) via OpenAI or Gemini

## Quick start (local)

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"

# 1. Download the raw dataset from Google Drive
python -m digit_classifier download

# 2. Preprocess and cache as .npz
python -m digit_classifier preprocess --name mnist_rgb_224 --color --size 224

# 3. Train (runs external dataset download + dedup on first run)
python -m digit_classifier train

# 4. Run webcam inference (uses EMA model; pass --mean/--std if checkpoint lacks them)
python -m digit_classifier infer --checkpoint checkpoints/<run_id>/best.pt \
  --mean 0.57 0.52 0.48 --std 0.23 0.23 0.23

# 5. Export compiled pipeline (TorchScript)

This compiles the model _and_ preprocessing into a single TorchScript file
that accepts raw image tensors and can be uploaded to the Hub. The export
uses the **EMA** model from the checkpoint.

**Normalization (mean/std):** Checkpoints do not currently store mean/std.
You must pass `--mean` and `--std` so the exported pipeline matches your
training normalization. Get these from the training log (e.g. `mean=[0.13, 0.13, 0.13]`)
or from your cached dataset. If omitted, 0.5/0.5 is used (likely incorrect).

```bash
# RGB (3 channels) — use the mean/std printed during train
python -m digit_classifier export-pipeline \
  --checkpoint checkpoints/<run_id>/best.pt \
  --output pipeline-cnn.pt \
  --mean 0.13 0.13 0.13 \
  --std 0.31 0.31 0.31

# Grayscale (1 channel)
python -m digit_classifier export-pipeline \
  --checkpoint checkpoints/<run_id>/best.pt \
  --output pipeline-cnn.pt \
  --input-channels 1 \
  --mean 0.13 \
  --std 0.31

# Upload to HuggingFace Hub
python -m digit_classifier export-pipeline \
  --checkpoint checkpoints/<run_id>/best.pt \
  --output pipeline-cnn.pt \
  --mean 0.13 0.13 0.13 --std 0.31 0.31 0.31 \
  --push-to-hf --hf-repo <username>/<repo>
```

## Cloud training

After running locally at least once (so caches exist), push them to
HuggingFace Hub and pull on any cloud VM:

### Push caches (run locally)

```bash
# Authenticate with HuggingFace (one-time)
huggingface-cli login

# Push internal dataset cache + dedup indices to a private HF repo
python -m digit_classifier push-cache --repo <your-username>/digit-classification-cache
```

This uploads:
- `datasets/mnist_rgb_224.npz` — preprocessed internal dataset (~1.5 GB)
- `datasets/dedup_indices_*.json` — dedup index cache (a few KB)

### Pull caches and train (run on cloud VM)

```bash
# Install
pip install -e ".[dev]"

# Authenticate with HuggingFace
huggingface-cli login
# Or set the token directly:
# export HF_TOKEN=hf_...

# Pull caches from HuggingFace Hub
python -m digit_classifier pull-cache --repo <your-username>/digit-classification-cache

# Train — external datasets (SVHN, MNIST, etc.) download automatically
# from torchvision on first access; dedup is skipped (cached indices)
python -m digit_classifier train
```

### Multi-GPU training (DDP)

Use `torchrun` to train on multiple GPUs. Batch size is per-GPU; learning rate is scaled linearly by world size.

Use the `-m` flag so torchrun runs `python -m digit_classifier` (the standard way for pip-installed packages):

```bash
# 2 GPUs on one node
torchrun --nproc_per_node=2 -m digit_classifier train --epochs 900

# 4 GPUs
torchrun --nproc_per_node=4 -m digit_classifier train
```

## Pareidolia test dataset

Generate OOD test images where digits 0–9 are implied by real-world objects
(pareidolia), then push/pull to HuggingFace for use with `--test-dataset`:

```bash
# Install pareidolia extras (OpenAI, Gemini)
pip install -e ".[pareidolia]"

# Generate images (Gemini example)
python -m digit_classifier generate-pareidolia \
  --provider gemini \
  --gemini-llm-model gemini-3.1-pro-preview \
  --image-size 4K \
  --temperature 0.9 \
  --per-digit 50

# Push to HuggingFace
python -m digit_classifier push-test-dataset --repo <username>/pareidolia-test

# Pull on another machine
python -m digit_classifier pull-test-dataset --repo <username>/pareidolia-test

# Train with test eval
python -m digit_classifier train --test-dataset dataset_out
```

## CLI reference

| Command | Description |
|---|---|
| `download` | Fetch the raw JPEG archive from Google Drive |
| `preprocess` | Resize, colour-convert, compute mean/std and cache as `.npz` |
| `train` | Run the full training pipeline (see [docs/FINETUNING.md](docs/FINETUNING.md) for `--pretrain` best practices) |
| `infer` | Real-time webcam digit recognition (uses EMA; pass `--mean`/`--std` if checkpoint lacks them) |
| `export-pipeline` | Compile model + preprocessing into a TorchScript pipeline (uses EMA; pass `--mean`/`--std` if checkpoint lacks them) |
| `visualize` | Debug-view augmented + mixed-up training batches |
| `push-cache` | Push dataset caches to a HuggingFace Hub repo |
| `pull-cache` | Pull dataset caches from a HuggingFace Hub repo |
| `generate-pareidolia` | Generate AI pareidolia test images (digits implied by real-world objects); requires `[pareidolia]` extra |
| `push-test-dataset` | Push pareidolia test dataset to HuggingFace Hub |
| `pull-test-dataset` | Pull pareidolia test dataset from HuggingFace Hub |

Every training hyper-parameter is exposed as a CLI flag with the current
defaults.  Run `python -m digit_classifier train --help` for the full list.

## Project layout

```
src/digit_classifier/
  __init__.py          Package version
  __main__.py          CLI entry-point (argparse subcommands)
  config.py            Dataclass-based configuration
  model.py             ResNeXt + DropPath
  dataset.py           DigitDataset (tensor-backed)
  splitting.py         Train/val split + external mixing + stats
  external.py          ExternalOnDemandDataset + deduplication
  sampler.py           RatioBatchSampler
  augmentation.py      YOLO-style augmentation pipeline
  mixup.py             timm Mixup/CutMix wrapper
  preprocessing.py     Download + preprocess + cache
  training.py          Training loop
  inference.py         Webcam inference
  visualize.py         Debug batch visualisation
  hub.py               HuggingFace Hub push/pull
  pareidolia_generate.py  AI pareidolia image generation (LLM + image API)
  pareidolia_dataset.py   PareidoliaTestDataset loader
tests/                 pytest test suite
scripts/               fix_metadata_image_paths, generate_pareidolia_variety, etc.
```

## Running tests

```bash
pytest
```

## Training notes

Configuration for longer training runs and DeiT tiny experiments. DropPath rates
by model size: tiny/small = 0.0, base = 0.1, large = 0.4, huge = 0.5.

For longer training, use `--epochs 800`, `--weight-decay 0.05`,
`--drop-path-increment 0.05`, `--drop-path-increment-every 200`.

```bash
python -m digit_classifier train \
  --deit-model base \
  --warmup-epochs 5 \
  --epochs 400 \
  --batch-size 256 \
  --lr 3e-4 \
  --eta-min 1e-5 \
  --grad-clip-norm 1.0 \
  --weight-decay 0.02 \
  --drop-path-rate 0.0 \
  --layer-decay 0.0 \
  --mixup-prob 0.5 \
  --mixup-alpha 0.8 \
  --cutmix-alpha 1.0 \
  --bce-loss \
  --label-smoothing 0.0 \
  --repeat-aug \
  --repeat-aug-repeats 3 \
  --layer-scale-init 1e-6 \
  --mixup-off-last-n 0 \
  --test-dataset dataset_out \
  --flash-attention \
  --no-ema \
  --no-amp \
  --no-warm-restarts \
  --no-compile
```
