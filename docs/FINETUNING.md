# Fine-Tuning Best Practices

This document summarizes recommended settings when fine-tuning from a pretrained checkpoint (`--pretrain`).

## What We Already Do Correctly

- **Position embedding resize**: When fine-tuning at a different resolution than the pretrained checkpoint, position embeddings are interpolated automatically.
- **Head mismatch**: When `num_classes` differs (e.g. pretrain on 10 classes, fine-tune on 100), the classification head is skipped and re-initialized; backbone weights are loaded.
- **Weight decay exclusion**: LayerNorm, bias, and LayerScale params are excluded from weight decay (default).
- **Cosine schedule**: Cosine annealing with warmup is used.
- **EMA**: Optional EMA model tracks a smoothed version of weights.

## Recommended Overrides for Fine-Tuning

Based on [DeiT official fine-tuning](https://github.com/facebookresearch/deit/issues/45) and modern ViT recipes (BEiT, timm):

| Setting | Pretrain default | Fine-tune recommendation | Rationale |
|---------|------------------|--------------------------|-----------|
| `--drop-path-rate` | 0.1 | **0** | DeiT: "remove stochastic depth" during fine-tuning. Pretrained features benefit from deterministic forward passes. |
| `--layer-decay` | 0.0 | **0.65–0.75** | Earlier layers get lower LR; head gets full LR. Common in BEiT/timm fine-tuning. |
| `--lr` | 1e-3 | 1e-4 to 1e-3 | For AdamW, 1e-3 is often fine. Use 1e-4 if unstable. DeiT uses SGD 0.01 for their CIFAR/Cars recipe. |
| `--weight-decay` | 0.05 | 0.05 | Keep ImageNet-style weight decay when using AdamW. |
| `--epochs` | 900 | 50–300 | Fine-tuning typically needs fewer epochs than pretraining. |

## Example Fine-Tuning Command

```bash
python -m digit_classifier train \
  --pretrain checkpoints/<run_id>/best.pt \
  --drop-path-rate 0 \
  --layer-decay 0.65 \
  --epochs 100 \
  --lr 1e-4
```

For resolution change (e.g. 224 → 384):

```bash
python -m digit_classifier train \
  --pretrain checkpoints/<run_id>/best.pt \
  --size 384 \
  --drop-path-rate 0 \
  --layer-decay 0.65
```

## Resume vs Pretrain

- **`--resume`**: Full resume (optimizer, scheduler, epoch). Requires same resolution and architecture. Use when continuing a run.
- **`--pretrain`**: Load weights only. Allows different resolution and `num_classes`. Use for transfer learning.
