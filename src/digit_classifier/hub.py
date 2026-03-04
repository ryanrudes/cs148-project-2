"""Push and pull dataset caches to/from HuggingFace Hub.

This allows you to build the internal ``.npz`` cache and run deduplication
locally once, push the results to a (private) HuggingFace dataset repo, and
pull them on a cloud machine so training can start with minimal setup.

What gets pushed:
- ``datasets/mnist_rgb_224.npz`` — the preprocessed internal dataset
- ``datasets/dedup_indices_*.json`` — deduplication index cache (a few KB)

External datasets (SVHN, MNIST, etc.) are **not** pushed — they download
automatically from torchvision on first access on the cloud machine.

Test dataset (pareidolia):
- ``push_test_dataset`` / ``pull_test_dataset`` — push/pull the generate-pareidolia
  output (metadata.jsonl + images/) to a HuggingFace dataset repo for use with
  ``--test-dataset`` during training.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from rich.console import Console

console = Console()

DEFAULT_CACHE_DIR = "datasets"


def push_cache(
    repo_id: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    private: bool = True,
) -> None:
    """Upload dataset caches to a HuggingFace Hub dataset repo.

    Pushes all ``.npz`` and ``.json`` files from *cache_dir*.
    Creates the repo if it does not already exist.
    """
    cache_path = Path(cache_dir)
    files = sorted(list(cache_path.glob("*.npz")) + list(cache_path.glob("*.json")))

    if not files:
        console.print(f"[yellow]No cache files found in {cache_dir}[/yellow]")
        return

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    console.print(f"[bold]Repo:[/bold] [cyan]https://huggingface.co/datasets/{repo_id}[/cyan]")

    for f in files:
        size_mb = f.stat().st_size / 1e6
        console.print(f"  Uploading [bold]{f.name}[/bold] ({size_mb:.1f} MB) …")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )

    console.print(f"[bold green]Pushed {len(files)} file(s) to {repo_id}[/bold green]")


def pull_cache(
    repo_id: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> None:
    """Download dataset caches from a HuggingFace Hub dataset repo into *cache_dir*."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Pulling from:[/bold] [cyan]https://huggingface.co/datasets/{repo_id}[/cyan]")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_path),
    )

    npz_count = len(list(cache_path.glob("*.npz")))
    json_count = len(list(cache_path.glob("*.json")))
    console.print(f"[bold green]Pulled {npz_count} .npz + {json_count} .json file(s) to {cache_dir}[/bold green]")


# ---------------------------------------------------------------------------
# Test dataset (pareidolia)
# ---------------------------------------------------------------------------

def push_test_dataset(
    repo_id: str,
    dataset_dir: str = "dataset_out",
    private: bool = True,
) -> None:
    """Upload the pareidolia test dataset to a HuggingFace Hub dataset repo.

    Pushes metadata.jsonl and images/ (all PNGs under images/{digit}/).
    Use with ``--test-dataset`` during training after pulling.
    """
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "metadata.jsonl"
    images_dir = dataset_path / "images"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Pareidolia metadata not found: {metadata_path}\n"
            "Run generate-pareidolia first, or specify the correct --dataset-dir."
        )
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Pareidolia images directory not found: {images_dir}\n"
            "Run generate-pareidolia first, or specify the correct --dataset-dir."
        )

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    console.print(f"[bold]Pushing test dataset to:[/bold] [cyan]https://huggingface.co/datasets/{repo_id}[/cyan]")

    api.upload_folder(
        folder_path=str(dataset_path),
        repo_id=repo_id,
        repo_type="dataset",
    )

    img_count = sum(1 for _ in images_dir.rglob("*.png"))
    console.print(f"[bold green]Pushed metadata.jsonl + {img_count} image(s) to {repo_id}[/bold green]")


def pull_test_dataset(
    repo_id: str,
    dataset_dir: str = "dataset_out",
) -> None:
    """Download the pareidolia test dataset from a HuggingFace Hub dataset repo.

    Downloads into *dataset_dir* (metadata.jsonl + images/). Use this path
    as ``--test-dataset`` when training.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Pulling test dataset from:[/bold] [cyan]https://huggingface.co/datasets/{repo_id}[/cyan]")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_path),
    )

    metadata_path = dataset_path / "metadata.jsonl"
    img_count = sum(1 for _ in dataset_path.glob("images/**/*.png")) if (dataset_path / "images").exists() else 0
    console.print(f"[bold green]Pulled test dataset to {dataset_dir}[/bold green]")
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
        console.print(f"  [dim]metadata.jsonl: {line_count} samples, {img_count} images[/dim]")
