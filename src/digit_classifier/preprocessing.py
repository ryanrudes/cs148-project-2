"""Download, preprocess and cache the raw digit dataset.

Two operations live here:

1. **Download** — fetch the zipped JPEG archive from Google Drive.
2. **Preprocess** — apply deterministic transforms (resize, colour-convert)
   and store the result as a compressed ``.npz`` file together with the
   per-channel mean and standard deviation.
"""

from __future__ import annotations

import os
import zipfile
from collections import Counter
from pathlib import Path

import gdown
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from digit_classifier.augmentation import get_preprocessor

console = Console()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(
    url: str = "https://drive.google.com/uc?id=1_gIar-Q89tWll-dnJUE077UujzAVMPxQ",
    output_zip: str = "data/dataset.zip",
    force: bool = False,
) -> str:
    """Download and extract the raw JPEG dataset from Google Drive.

    Returns the path to the extracted directory.
    """
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    if not os.path.exists(output_zip) or force:
        console.print(f"[bold]Downloading dataset from Google Drive …[/bold]")
        gdown.download(url, output_zip, quiet=False)

    data_dir = output_zip.replace(".zip", "")
    with zipfile.ZipFile(output_zip, "r") as zf:
        zf.extractall(data_dir)
    console.print(f"Extracted to [cyan]{data_dir}[/cyan]")
    return data_dir


# ---------------------------------------------------------------------------
# Load raw images
# ---------------------------------------------------------------------------

def load_raw_images(data_dir: str) -> tuple[list[Image.Image], list[int]]:
    """Load JPEG images and parse labels from filenames.

    Expected filename pattern: ``*_label<digit>.jpg``.
    """
    images: list[Image.Image] = []
    labels: list[int] = []

    filenames = sorted(f for f in os.listdir(data_dir) if f.endswith(".jpg"))
    for fname in filenames:
        path = os.path.join(data_dir, fname)
        img = Image.open(path)
        label = int(path.split("_")[-1].replace(".jpg", "").replace("label", ""))
        images.append(img)
        labels.append(label)

    return images, labels


def print_label_distribution(labels: list[int]) -> None:
    """Print a summary of the label distribution using Rich."""
    counts = Counter(labels)
    console.print(f"[bold]Total images:[/bold] {len(labels)}")
    for digit in range(10):
        console.print(f"  Digit {digit}: {counts.get(digit, 0)} samples")


# ---------------------------------------------------------------------------
# Preprocess + compute stats
# ---------------------------------------------------------------------------

def preprocess_images(
    images: list[Image.Image],
    color: bool,
    size: int,
) -> list[np.ndarray]:
    """Apply the deterministic preprocessor and return a list of numpy arrays."""
    preprocessor = get_preprocessor(color, size)
    results: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Preprocessing {len(images):,} images → {size}×{size} {'RGB' if color else 'gray'}",
            total=len(images),
        )
        for img in images:
            tensor = preprocessor(img)
            results.append(tensor.numpy())
            progress.advance(task)

    return results


def compute_mean_std(
    images: list[np.ndarray] | torch.Tensor,
    color: bool,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Welford-style per-channel mean / std computation."""
    num_channels = 3 if color else 1
    s1 = torch.zeros(num_channels)
    s2 = torch.zeros(num_channels)
    n = 0

    total = len(images)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[blue]Computing channel statistics", total=total)
        for x in images:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            s1 += x.sum((1, 2))
            s2 += (x * x).sum((1, 2))
            n += x.shape[1] * x.shape[2]
            progress.advance(task)

    mean_t = s1 / n
    std_t = torch.sqrt(s2 / n - mean_t**2)
    return tuple(mean_t.tolist()), tuple(std_t.tolist())


# ---------------------------------------------------------------------------
# End-to-end caching
# ---------------------------------------------------------------------------

def preprocess_and_cache(
    dataset_name: str,
    color: bool,
    size: int,
    data_dir: str = "data/dataset",
    output_dir: str = "datasets",
    force: bool = False,
) -> Path:
    """Run the full preprocessing pipeline and save to *output_dir*.

    Returns the path to the saved ``.npz`` file.
    """
    output_path = Path(output_dir) / f"{dataset_name}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        console.print(f"[yellow]Dataset '{dataset_name}' already exists at {output_path}.[/yellow]")
        return output_path

    images, labels = load_raw_images(data_dir)
    print_label_distribution(labels)

    preprocessed = preprocess_images(images, color, size)
    mean, std = compute_mean_std(preprocessed, color)
    console.print(f"[bold]Mean:[/bold] {mean}")
    console.print(f"[bold]Std:[/bold]  {std}")

    # Store as uint8 to save disk space (~4x smaller than float32).
    # The float32 conversion (x / 255.0) happens at load time.
    arr = (np.array(preprocessed) * 255).round().clip(0, 255).astype(np.uint8)
    console.print(f"Preprocessed shape: {arr.shape} (uint8, {arr.nbytes / 1e6:.0f} MB)")

    np.savez(
        str(output_path),
        images=arr,
        labels=labels,
        mean=mean,
        std=std,
    )
    console.print(f"Saved to [green]{output_path}[/green]")
    return output_path
