# Digit Classifier

A PyTorch training pipeline for digit classification using **ResNeXt** with
YOLO-style augmentation, external dataset mixing, and comprehensive experiment
tracking.

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
- **HuggingFace Hub** integration for pushing/pulling dataset caches
- **Webcam inference** with real-time probability visualisation

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

## CLI reference

| Command | Description |
|---|---|
| `download` | Fetch the raw JPEG archive from Google Drive |
| `preprocess` | Resize, colour-convert, compute mean/std and cache as `.npz` |
| `train` | Run the full training pipeline |
| `infer` | Real-time webcam digit recognition (uses EMA; pass `--mean`/`--std` if checkpoint lacks them) |
| `export-pipeline` | Compile model + preprocessing into a TorchScript pipeline (uses EMA; pass `--mean`/`--std` if checkpoint lacks them) |
| `visualize` | Debug-view augmented + mixed-up training batches |
| `push-cache` | Push dataset caches to a HuggingFace Hub repo |
| `pull-cache` | Pull dataset caches from a HuggingFace Hub repo |

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
tests/                 pytest test suite
```

## Running tests

```bash
pytest
```
