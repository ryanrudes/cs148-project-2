import pytest

try:
    from torchvision import v2 as _tv_v2  # ensure v2 is available for these tests
except Exception:  # pragma: no cover - environment without torchvision v2
    pytest.skip("torchvision v2 required for pipeline tests", allow_module_level=True)

import torch
import torch.nn as nn
from PIL import Image

from digit_classifier.pipeline import DigitClassifierPipeline


class DummyModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return logits that always predict class `3`
        b = x.shape[0]
        out = torch.zeros(b, self.num_classes, dtype=torch.float32)
        out[:, 3] = 1.0
        return out


def test_pipeline_forward_accepts_uint8_single_and_batch():
    model = DummyModel()
    pipeline = DigitClassifierPipeline(
        model=model,
        input_size=32,
        input_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        device="cpu",
    )

    # single image (C,H,W) uint8
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    pred = pipeline(img)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (1,)
    assert int(pred[0]) == 3

    # batched images (B,C,H,W)
    batch = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    preds = pipeline(batch)
    assert preds.shape == (2,)
    assert preds.tolist() == [3, 3]


def test_pipeline_run_and_script_save(tmp_path):
    model = DummyModel()
    pipeline = DigitClassifierPipeline(
        model=model,
        input_size=28,
        input_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        device="cpu",
    )

    # run() accepts PIL images
    pil = Image.new("RGB", (28, 28), color=(123, 123, 123))
    out = pipeline.run([pil])
    assert isinstance(out, list) and out[0] == 3

    # save -> load TorchScript and call with a raw uint8 tensor
    out_file = tmp_path / "pipeline.pt"
    pipeline.save_pipeline_local(str(out_file))

    scripted = torch.jit.load(str(out_file))
    inp = torch.randint(0, 256, (1, 3, 28, 28), dtype=torch.uint8)
    scripted_preds = scripted(inp)
    # scripted returns a tensor of indices
    assert isinstance(scripted_preds, torch.Tensor)
    assert scripted_preds.shape == (1,)
    assert int(scripted_preds[0]) == 3
