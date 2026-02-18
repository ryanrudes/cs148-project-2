"""Real-time webcam digit recognition with probability visualisation.

Loads a checkpoint, streams frames from the webcam, preprocesses each frame
with aspect-preserving resize + centre-crop, runs inference, and displays a
composite window with the camera feed and a probability-bar panel.

Key controls
------------
- **q** — quit
- **s** — save snapshot
- **v** — toggle between full-camera and model-input view
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from rich.console import Console

from digit_classifier.model import ResNeXt

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove ``_orig_mod.`` (torch.compile) and ``module.`` (AveragedModel) prefixes."""
    result = {}
    for k, v in state_dict.items():
        key = k.replace("_orig_mod.", "").replace("module.", "")
        result[key] = v
    return result


def _make_model_input(img_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio then centre-crop to *size* × *size*."""
    h, w = img_rgb.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h))
    sx = (new_w - size) // 2
    sy = (new_h - size) // 2
    return resized[sy : sy + size, sx : sx + size]


def _draw_prob_panel(probs: np.ndarray, width: int, height: int) -> np.ndarray:
    """Render a vertical probability-bar chart as a BGR image."""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 30
    margin = 10
    bar_h = (height - margin * 2) // 10
    for i in range(10):
        p = float(probs[i])
        top = margin + i * bar_h
        bottom = top + int(bar_h * 0.8)
        bar_w = int((width - margin * 3 - 80) * p)
        color = (int(60 * (1 - p)), int(200 * p + 50 * (1 - p)), int(60 * (1 - p)))
        cv2.rectangle(panel, (margin, top), (margin + bar_w, bottom), color, -1)
        cv2.rectangle(panel, (margin, top), (width - margin - 80, bottom), (50, 50, 50), 1)
        label = f"{i}: {p * 100:5.1f}%"
        cv2.putText(panel, label, (width - margin - 75, bottom - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def _overlay_prediction(frame: np.ndarray, pred: int, conf: float, mode: str) -> None:
    """Draw a semi-transparent prediction banner on *frame* (mutates in place)."""
    h, w = frame.shape[:2]
    text = f"Pred: {pred} ({conf * 100:4.1f}%)  View: {mode.upper()}"
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (180, 255, 180), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    checkpoint_path: str,
    *,
    layers: list[int] | tuple[int, ...] = (3, 4, 23, 3),
    num_classes: int = 10,
    groups: int = 64,
    width_per_group: int = 4,
    input_size: int = 224,
    camera_index: int = 0,
    smoothing_alpha: float = 0.2,
    bar_panel_width: int = 320,
    device: str = "auto",
) -> None:
    """Launch a real-time webcam inference window."""
    # --- Device ---
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)
    console.print(f"[bold]Inference device:[/bold] {dev}")

    # --- Model ---
    model = ResNeXt(
        layers=list(layers), num_classes=num_classes,
        groups=groups, width_per_group=width_per_group,
    ).to(dev)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
    model.eval()
    console.print(f"Loaded checkpoint: [cyan]{checkpoint_path}[/cyan]")

    # --- Mean / std from checkpoint or defaults ---
    if "mean" in ckpt:
        mean = np.array(ckpt["mean"], dtype=np.float32)
        std = np.array(ckpt["std"], dtype=np.float32)
    else:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    # --- Camera loop ---
    cap = cv2.VideoCapture(camera_index)
    running_probs = np.zeros(num_classes, dtype=np.float32)
    display_mode = "full"

    console.print("[bold]Controls:[/bold] q=quit  s=snapshot  v=toggle view")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        model_input_rgb = _make_model_input(img_rgb, input_size)

        tensor = torch.from_numpy(model_input_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor - mean_t) / std_t
        tensor = tensor.to(dev)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        running_probs = smoothing_alpha * running_probs + (1 - smoothing_alpha) * probs
        pred = int(probs.argmax())
        conf = float(probs[pred])

        h, w = frame_bgr.shape[:2]
        if display_mode == "full":
            left = frame_bgr
        else:
            square = cv2.cvtColor(model_input_rgb, cv2.COLOR_RGB2BGR)
            ds = min(h, w)
            square_resized = cv2.resize(square, (ds, ds))
            left = np.zeros((h, w, 3), dtype=np.uint8)
            sy, sx = (h - ds) // 2, (w - ds) // 2
            left[sy : sy + ds, sx : sx + ds] = square_resized

        panel = _draw_prob_panel(running_probs, bar_panel_width, h)
        composite = np.zeros((h, w + bar_panel_width, 3), dtype=np.uint8)
        composite[:, :w] = left
        composite[:, w:] = panel
        _overlay_prediction(composite[:, :w], pred, conf, display_mode)

        cv2.imshow("Digit Recognition", composite)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"snapshot_pred_{pred}_{int(conf * 100)}.png"
            cv2.imwrite(fname, composite)
            console.print(f"Saved snapshot: [green]{fname}[/green]")
        elif key == ord("v"):
            display_mode = "input" if display_mode == "full" else "full"
            console.print(f"View: [cyan]{display_mode}[/cyan]")

    cap.release()
    cv2.destroyAllWindows()
