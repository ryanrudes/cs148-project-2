"""Debug visualisation of augmented training batches.

This is the cleaned-up version of the ``cv2.imshow`` / ``exit()`` debug block
that was left in the middle of ``train()`` in the original codebase.  It is
now a proper CLI sub-command.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from rich.console import Console

from digit_classifier.config import Config
from digit_classifier.external import DEFAULT_EXTERNAL_FRACTIONS
from digit_classifier.mixup import MixupCutmixApply, create_mixup_cutmix
from digit_classifier.splitting import split_dataset
from digit_classifier.training import _create_dataloaders, load_cached_dataset

console = Console()


def visualize_batches(cfg: Config, num_batches: int = 1) -> None:
    """Show *num_batches* augmented training batches in a CV2 window.

    Press any key to advance through individual samples within each batch.
    Press **q** to quit early.
    """
    images, labels, cached_mean, cached_std = load_cached_dataset(cfg)
    train_dataset, _, mean, std = split_dataset(
        images, labels, cached_mean, cached_std,
        train_fraction=cfg.data.train_fraction,
        mix_external=cfg.data.mix_external,
        external_fractions=DEFAULT_EXTERNAL_FRACTIONS if cfg.data.mix_external else None,
        color=cfg.data.color,
        size=cfg.data.image_size,
        seed=cfg.data.split_seed,
        augment_cfg=cfg.augment,
    )

    device = torch.device("cpu")
    train_loader, _ = _create_dataloaders(
        train_dataset, train_dataset, cfg.data.batch_size,
        cfg.data.primary_fraction, device,
    )

    tc = cfg.training
    mixup = MixupCutmixApply(create_mixup_cutmix(
        num_classes=cfg.model.num_classes,
        mixup_alpha=tc.mixup_alpha,
        cutmix_alpha=tc.cutmix_alpha,
        cutmix_minmax=tc.cutmix_minmax,
        prob=tc.mixup_prob,
        label_smoothing=tc.label_smoothing,
        mode=tc.mixup_mode,
    ))

    mean_arr = np.array(mean)
    std_arr = np.array(std)

    console.print("[bold]Visualising augmented batches — press any key to advance, q to quit.[/bold]")

    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        batch_images = batch_images.float()
        batch_labels = batch_labels.long()
        batch_images, batch_labels = mixup(batch_images, batch_labels)

        for i in range(batch_images.shape[0]):
            img = batch_images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * std_arr + mean_arr).clip(0, 1)
            img = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if batch_labels.dim() == 2:
                label_str = f"soft: {batch_labels[i].cpu().numpy().round(2)}"
            else:
                label_str = f"label: {int(batch_labels[i])}"

            cv2.setWindowTitle("Augmented Sample", f"Batch {batch_idx} / Sample {i} — {label_str}")
            cv2.imshow("Augmented Sample", img_bgr)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    console.print("[bold green]Done.[/bold green]")
