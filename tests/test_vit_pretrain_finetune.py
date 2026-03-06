"""Tests for DeiT pretrain/fine-tune: resize_pos_embed and checkpoint loading."""

import torch

from digit_classifier.vit import build_deit3, resize_pos_embed
from digit_classifier.training import build_model_from_checkpoint


# ---------------------------------------------------------------------------
# resize_pos_embed
# ---------------------------------------------------------------------------


def test_resize_pos_embed_same_size_returns_unchanged():
    """When orig_size == new_size, return input unchanged."""
    pos = torch.randn(1, 196, 768)  # 224/16 = 14, 14*14=196
    out = resize_pos_embed(pos, orig_size=224, new_size=224)
    assert out.shape == pos.shape
    assert torch.allclose(out, pos)


def test_resize_pos_embed_224_to_384():
    """Interpolate from 224 (14x14) to 384 (24x24)."""
    pos = torch.randn(1, 196, 768)  # 14*14=196
    out = resize_pos_embed(pos, orig_size=224, new_size=384)
    assert out.shape == (1, 576, 768)  # 24*24=576


def test_resize_pos_embed_128_to_224():
    """Interpolate from 128 (8x8) to 224 (14x14)."""
    pos = torch.randn(1, 64, 384)  # 8*8=64, small model
    out = resize_pos_embed(pos, orig_size=128, new_size=224)
    assert out.shape == (1, 196, 384)  # 14*14=196


def test_resize_pos_embed_patch14():
    """Interpolate with patch_size=14 (DeiT huge)."""
    # 224/14 = 16, 16*16=256
    pos = torch.randn(1, 256, 1280)
    out = resize_pos_embed(pos, orig_size=224, new_size=384, patch_size=14)
    # 384/14 = 27.42... -> 27? No, 384/14 is not integer. 378/14=27. Let me use 392/14=28.
    # Actually 384/14 = 27.43, so we need a size divisible by 14. Use 392.
    out = resize_pos_embed(pos, orig_size=224, new_size=392, patch_size=14)
    assert out.shape == (1, 28 * 28, 1280)  # 784


# ---------------------------------------------------------------------------
# build_model_from_checkpoint with target_image_size
# ---------------------------------------------------------------------------


def test_build_model_from_checkpoint_resolution_change(tmp_path):
    """Loading a 224 checkpoint with target_image_size=384 interpolates pos_embed."""
    from digit_classifier.vit import build_deit3

    # Build and save a tiny DeiT at 224
    model = build_deit3(size="tiny", num_classes=10, image_size=224)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_type": "deit",
        "model_config": {
            "deit_model": "tiny",
            "num_classes": 10,
            "image_size": 224,
        },
    }
    path = tmp_path / "deit_224.pt"
    torch.save(ckpt, path)

    # Load with target_image_size=384
    loaded, _ = build_model_from_checkpoint(
        str(path), torch.device("cpu"), target_image_size=384
    )
    assert loaded.pos_embed.shape[1] == 24 * 24  # 576 patches for 384/16

    # Forward pass at 384
    x = torch.randn(2, 3, 384, 384)
    out = loaded(x)
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# Pretrain flow (integration)
# ---------------------------------------------------------------------------


def test_pretrain_at_different_resolution(tmp_path):
    """Pretrain at 128, then fine-tune at 224 loads with interpolated pos_embed."""
    from digit_classifier.config import Config
    from digit_classifier.training import train

    # Create a minimal dataset cache (unique name to avoid polluting real cache)
    import numpy as np
    import os
    os.makedirs("datasets", exist_ok=True)
    dataset_name = "test_pretrain_finetune_224"
    np.savez(
        f"datasets/{dataset_name}.npz",
        images=np.random.randint(0, 256, (20, 3, 224, 224), dtype=np.uint8),
        labels=np.random.randint(0, 10, 20),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    # Build and save a checkpoint at 128
    model_128 = build_deit3(size="tiny", num_classes=10, image_size=128)
    ckpt_128 = {
        "model_state_dict": model_128.state_dict(),
        "ema_state_dict": model_128.state_dict(),
        "model_type": "deit",
        "model_config": {"deit_model": "tiny", "num_classes": 10, "image_size": 128},
    }
    pretrain_path = tmp_path / "pretrain_128.pt"
    torch.save(ckpt_128, pretrain_path)

    # Run 1 epoch of fine-tune at 224 from that checkpoint
    cfg = Config()
    cfg.data.dataset_name = dataset_name
    cfg.data.image_size = 224
    cfg.data.mix_external = False
    cfg.training.epochs = 1
    cfg.training.pretrain_path = str(pretrain_path)
    cfg.training.wandb_enabled = False
    cfg.training.compile_model = False
    cfg.training.ema_enabled = False
    cfg.model.model_type = "deit"
    cfg.model.deit_model = "tiny"

    train(cfg)  # Should not raise; pos_embed is interpolated 128->224
