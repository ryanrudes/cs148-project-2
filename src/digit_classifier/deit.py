import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utils
# ----------------------------

def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    # Minimal truncated normal init (PyTorch has nn.init.trunc_normal_ in newer versions).
    # This wrapper uses the builtin if available.
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    # Fallback implementation (approx) if needed.
    # Note: for most projects, using nn.init.trunc_normal_ is fine.
    with torch.no_grad():
        # Use normal then clamp (approximate).
        tensor.normal_(mean, std)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """
    Stochastic Depth per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with broadcastable shape: (B, 1, 1) for token tensors.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ----------------------------
# Core layers
# ----------------------------

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using a Conv2d (kernel=stride=patch_size).
    Output: (B, N, D)
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    """
    Pre-norm Transformer block with:
      x = x + DropPath( LayerScale1 * Attn(LN(x)) )
      x = x + DropPath( LayerScale2 * MLP(LN(x)) )
    DeiT-III uses LayerScale with init 1e-4.  [oai_citation:2‡arXiv](https://arxiv.org/pdf/2204.07118?utm_source=chatgpt.com)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 1e-4,   # LayerScale init (DeiT III uses 1e-4 for all models).  [oai_citation:3‡arXiv](https://arxiv.org/pdf/2204.07118?utm_source=chatgpt.com)
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1 * self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.ls2 * self.mlp(self.norm2(x)))
        return x


# ----------------------------
# DeiT-III (architecture)
# ----------------------------

@dataclass
class DeiT3Config:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000

    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0

    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    init_values: float = 1e-4  # LayerScale init.  [oai_citation:4‡arXiv](https://arxiv.org/pdf/2204.07118?utm_source=chatgpt.com)


class DeiT3VisionTransformer(nn.Module):
    """
    DeiT-III architecture = ViT + LayerScale (and usual ViT niceties).
    No distillation token.  [oai_citation:5‡arXiv](https://arxiv.org/pdf/2204.07118?utm_source=chatgpt.com)
    """
    def __init__(self, cfg: DeiT3Config):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size, patch_size=cfg.patch_size, in_chans=cfg.in_chans, embed_dim=cfg.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

        # stochastic depth decay rule
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()

        self.blocks = nn.ModuleList([
            Block(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                init_values=cfg.init_values,
            )
            for i in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # Standard ViT init for Linear/Conv
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # fan_out init is fine; ViT papers often use trunc normal; conv here is patch embed.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Classifier head often zero-inited bias; weight already initialized above.
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls, x), dim=1)          # (B, N+1, D)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits


# ----------------------------
# Canonical DeiT-III sizes (architecture params)
# These match the usual ViT family sizes used by DeiT/deit3 in timm.  [oai_citation:6‡GitHub](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/deit.py?utm_source=chatgpt.com)
# ----------------------------

def deit3_tiny_patch16_224(num_classes: int = 10) -> DeiT3VisionTransformer:
    cfg = DeiT3Config(embed_dim=192, depth=12, num_heads=3, num_classes=num_classes, patch_size=16, img_size=224)

    return DeiT3VisionTransformer(cfg)

def deit3_small_patch16_224(num_classes: int = 10) -> DeiT3VisionTransformer:
    cfg = DeiT3Config(embed_dim=384, depth=12, num_heads=6, num_classes=num_classes, patch_size=16, img_size=224)
    return DeiT3VisionTransformer(cfg)

def deit3_base_patch16_224(num_classes: int = 10) -> DeiT3VisionTransformer:
    cfg = DeiT3Config(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes, patch_size=16, img_size=224)
    return DeiT3VisionTransformer(cfg)