"""Frozen local DINOv3 ViT encoder with a trainable linear patch segmentation head."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTModel


class FrozenDINOv3Segmentation(nn.Module):
    def __init__(self, encoder, mean, std):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.patch_size = encoder.config.patch_size
        self.special_tokens = 1 + encoder.config.num_register_tokens
        if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError("expected three RGB means and positive standard deviations")
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.head = nn.Conv2d(encoder.config.hidden_size, 1, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("image must be float N x 3 x H x W")
        h, w = image.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch size")
        if not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1):
            raise ValueError("image must contain finite RGB values in [0,1]")
        with torch.no_grad():
            features = self.encoder(pixel_values=(image - self.mean) / self.std).last_hidden_state
        tokens = features[:, self.special_tokens:, :]
        gh, gw = h // self.patch_size, w // self.patch_size
        if tokens.shape[1] != gh * gw:
            raise ValueError("encoder patch token count does not match input grid")
        spatial = tokens.transpose(1, 2).reshape(image.shape[0], -1, gh, gw)
        return F.interpolate(self.head(spatial), size=(h, w), mode="bilinear", align_corners=False)


def build_local_dinov3(snapshot):
    snapshot = Path(snapshot)
    cfg = json.loads((snapshot / "preprocessor_config.json").read_text())
    encoder = DINOv3ViTModel.from_pretrained(str(snapshot), local_files_only=True)
    return FrozenDINOv3Segmentation(encoder, cfg["image_mean"], cfg["image_std"])
