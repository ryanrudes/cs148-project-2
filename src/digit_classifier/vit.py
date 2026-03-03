from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


# NOTE: This file implements the DeiT-III *architecture* as used in timm's deit3_* models:
# - Standard ViT encoder
# - no distillation token
# - "no_embed_class" positional embedding style (pos embed applies to patch tokens only; cls is concatenated after)
# - Pre-norm blocks
# - LayerScale on both attention and MLP residual branches
# - Stochastic depth (DropPath)


@dataclass
class DeiTConfig:
    # Input / task
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_classes: int = 10

    # Model size (DeiT-III small defaults)
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0

    # Projections / norms
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6

    # Dropouts
    proj_drop: float = 0.0          # MLP + output proj dropout
    attn_drop: float = 0.0          # attention dropout
    drop_path_rate: float = 0.1     # stochastic depth (linearly increased over depth)

    # LayerScale
    init_values: float = 1e-4


class DropPath(nn.Module):
    """Stochastic Depth per sample (timm-style)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask, broadcast across remaining dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Use float mask to avoid bool->float casts in hot paths
        random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """LayerScale: learnable per-channel scale applied to residual branch output."""

    def __init__(self, dim: int, init_values: Optional[float] = None):
        super().__init__()
        if init_values is None:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.gamma is None:
            return x
        # broadcast over token dimension
        return x * self.gamma


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding using Conv2d(k=stride=patch_size). Output: (B, N, D)."""

    def __init__(self, config: DeiTConfig):
        super().__init__()
        assert config.image_size % config.patch_size == 0, "image_size must be divisible by patch_size"
        self.img_size = config.image_size
        self.patch_size = config.patch_size
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float):
        super().__init__()
        assert dim % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    """Pre-norm Transformer block with LayerScale + DropPath on each residual branch."""

    def __init__(self, config: DeiTConfig, drop_path: float):
        super().__init__()
        dim = config.hidden_size

        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = Attention(
            dim=dim,
            num_heads=config.num_heads,
            qkv_bias=config.qkv_bias,
            attn_drop=config.attn_drop,
            proj_drop=config.proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=config.init_values)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = MLP(dim=dim, mlp_ratio=config.mlp_ratio, drop=config.proj_drop)
        self.ls2 = LayerScale(dim, init_values=config.init_values)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class DeiT3(nn.Module):
    """DeiT-III architecture (ViT + LayerScale; no distillation token)."""

    def __init__(self, config: DeiTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config)
        num_patches = self.patch_embed.num_patches

        # DeiT-III uses the no_embed_class style: pos_embed covers patch tokens only.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.proj_drop)

        # Stochastic depth decay rule (linearly increasing drop_path over depth)
        dpr = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.ModuleList([Block(config, drop_path=dpr[i]) for i in range(config.depth)])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # Follow common ViT init conventions
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
        else:
            # fallback if running on a very old torch
            self.pos_embed.data.normal_(mean=0.0, std=0.02)
            self.cls_token.data.normal_(mean=0.0, std=0.02)
            self.head.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        # Patchify
        x = self.patch_embed(x)  # (B, N, C)

        # no_embed_class positional embedding behavior: add pos to patches first, then concat cls.
        x = x + self.pos_embed

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)  # (B, 1+N, C)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        feat = self.forward_features(x)
        return self.head(feat)


# Convenience builders matching common DeiT-III sizes

def deit3_tiny_patch16_224(
    num_classes: int = 10,
    drop_path_rate: float | None = None,
    image_size: int = 224,
) -> DeiT3:
    cfg = DeiTConfig(
        hidden_size=192, depth=12, num_heads=3, num_classes=num_classes,
        image_size=image_size,
    )
    if drop_path_rate is not None:
        cfg.drop_path_rate = drop_path_rate
    return DeiT3(cfg)


def deit3_small_patch16_224(
    num_classes: int = 10,
    drop_path_rate: float | None = None,
    image_size: int = 224,
) -> DeiT3:
    cfg = DeiTConfig(
        hidden_size=384, depth=12, num_heads=6, num_classes=num_classes,
        image_size=image_size,
    )
    if drop_path_rate is not None:
        cfg.drop_path_rate = drop_path_rate
    return DeiT3(cfg)


def deit3_base_patch16_224(
    num_classes: int = 10,
    drop_path_rate: float | None = None,
    image_size: int = 224,
) -> DeiT3:
    cfg = DeiTConfig(
        hidden_size=768, depth=12, num_heads=12, num_classes=num_classes,
        image_size=image_size,
    )
    if drop_path_rate is not None:
        cfg.drop_path_rate = drop_path_rate
    return DeiT3(cfg)