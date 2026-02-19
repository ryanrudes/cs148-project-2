import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms


class DigitClassifierPipeline(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_size: int = 224,
        input_channels: int = 3,
        mean: tuple[float, ...] | list[float] | None = None,
        std: tuple[float, ...] | list[float] | None = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        # Determine normalization statistics
        if mean is None:
            mean = (0.5,) * input_channels
        if std is None:
            std = (0.5,) * input_channels

        # For TorchScript compatibility we implement preprocessing with
        # tensor operations (resize / center-crop / dtype / normalize).
        # Mean/std are stored as buffers so they are available inside the
        # scripted module without constructing tensors at runtime.
        self.input_size = input_size
        self.input_channels = input_channels
        self.input_height = input_size
        self.input_width = input_size
        self.mean = tuple(mean)
        self.std = tuple(std)

        # register mean/std as buffers for use in scripted preprocessing
        self.register_buffer("mean_tensor", torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer("std_tensor", torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1))

    def _preprocess_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """Preprocess a single image tensor (C,H,W) using PyTorch ops.

        Behaviour mirrors: Resize(min-side -> input_size) then center-crop,
        convert to float (if needed) and normalize using stored mean/std.
        """
        # ensure float in range [0,1]
        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 255.0
        else:
            img = img.to(torch.float32)

        # handle channel mismatch (simple, deterministic rules)
        c = img.shape[0]
        if c != self.input_channels:
            if c == 1 and self.input_channels == 3:
                img = img.repeat(3, 1, 1)
            elif c == 3 and self.input_channels == 1:
                img = img.mean(dim=0, keepdim=True)
            else:
                # truncate or repeat channels as best-effort
                if c > self.input_channels:
                    img = img[: self.input_channels]
                else:
                    reps = (self.input_channels + c - 1) // c
                    img = img.repeat(reps, 1, 1)[: self.input_channels]

        _, h, w = img.shape
        min_side = h if h < w else w
        scale = float(self.input_size) / float(min_side)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        # resize (bilinear)
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        img = img.squeeze(0)

        # center-crop
        top = (new_h - self.input_size) // 2
        left = (new_w - self.input_size) // 2
        img = img[:, top : top + self.input_size, left : left + self.input_size]

        # normalize using registered buffers
        img = (img - self.mean_tensor) / self.std_tensor
        return img

    @torch.jit.export
    def preprocess_layers(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess a single image tensor (C, H, W) from ToTensor output.

        Compatible with verify_pipeline.py: takes float [0,1] CHW, returns
        resized, cropped, normalized tensor (C, input_size, input_size).
        """
        return self._preprocess_tensor(tensor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inference on image tensors.

        Accepts preprocessed tensors `Tensor(B, C, H, W)` from preprocess_layers
        (verify_pipeline.py flow), or raw tensors `Tensor(C,H,W)` / `Tensor(B,C,H,W)`
        (uint8 or float [0,1]); raw input is preprocessed automatically.

        Returns class indices `Tensor(B,)`.
        """
        x = images
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # If already preprocessed (correct spatial size, float32 from preprocess_layers), run model directly
        _, c, h, w = x.shape
        if (
            h == self.input_size
            and w == self.input_size
            and x.dtype == torch.float32
        ):
            batch = x.to(self.device)
        else:
            # Raw input: preprocess each image
            processed: list[torch.Tensor] = []
            for i in range(x.shape[0]):
                processed.append(self._preprocess_tensor(x[i]))
            batch = torch.stack(processed, dim=0).to(self.device)

        logits = self.model(batch)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    @torch.jit.ignore
    def save_pipeline_local(self, path: str):
        """
        Compiles the ENTIRE pipeline (transforms + model + post)
        and saves it to a file.
        """
        self.cpu()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
        self.to(self.device)

    @torch.jit.ignore
    def push_to_hub(
        self,
        token: str,
        repo_id: str = 'ee148a-project',
        filename: str = "pipeline-cnn.pt",
    ):
        """
        Saves the pipeline to a local file and pushes it to the Hugging Face Hub.

        Args:
            token (str): HF token.
            repo_id (str): The ID of your repo,
                           e.g., "{username}/ee148a-project"
            filename (str): The name the file will have on the Hub,
                            e.g. 'pipeline-cnn.pt'
        """
        # 1. Save locally first
        local_path = f"temp_{filename}"
        self.save_pipeline_local(local_path)

        # 2. Initialize API
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)

        # 3. Upload the file
        print(f"Uploading {filename} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload compiled pipeline: {filename}"
        )

        # 4. Cleanup local temp file
        import os
        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"Success! Upload available at https://huggingface.co/{repo_id}/blob/main/{filename}")
        return True

    @torch.jit.ignore
    def run(self, pil_images: list):
        """Run pipeline on PIL images."""
        if self.input_channels == 3:
            convert_to = "RGB"
        elif self.input_channels == 1:
            convert_to = "L"

        tensor_list = [transforms.ToTensor()(img.convert(convert_to)) for img in pil_images]
        batch = torch.stack(tensor_list)
        predictions = self.forward(batch).tolist()

        return predictions


def save_and_export(
    pipeline: DigitClassifierPipeline,
    hf_info: dict,
):
    try:
        success = pipeline.push_to_hub(
            token=hf_info['token'],
            repo_id=f"{hf_info['username']}/{hf_info['repo_name']}",
            filename=hf_info['filename']
        )
        if success:
            import json
            with open('submission.json', 'w') as f:
                json.dump(hf_info, f, indent=4)
            print("Saved json to submission.json")
            return hf_info
    except Exception as e:
        print(f"Exception: {e}")