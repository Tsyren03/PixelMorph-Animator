"""Image loading, tensor conversion, and export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def default_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def load_image_tensor(path: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    transform = default_transform(image_size)
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """CHW tensor in [-1, 1] -> HWC uint8 RGB."""
    img = tensor.squeeze(0).detach().cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    arr = tensor_to_numpy_rgb(tensor)
    Image.fromarray(arr).save(path)
