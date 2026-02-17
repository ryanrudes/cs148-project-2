"""Push and pull dataset caches to/from HuggingFace Hub.

This allows you to build the internal ``.npz`` cache and run deduplication
locally once, push the results to a (private) HuggingFace dataset repo, and
pull them on a cloud machine so training can start with minimal setup.

What gets pushed:
- ``datasets/mnist_rgb_224.npz`` — the preprocessed internal dataset
- ``datasets/dedup_indices_*.json`` — deduplication index cache (a few KB)

External datasets (SVHN, MNIST, etc.) are **not** pushed — they download
automatically from torchvision on first access on the cloud machine.
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
