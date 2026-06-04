"""Training loop for Pixel Shuffler image translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import lpips
import torch
from torchvision.models import VGG19_Weights, vgg19

from .io_utils import load_image_tensor, save_tensor_image, tensor_to_numpy_rgb
from .losses import calc_gram_matrix, get_vgg_features, style_loss
from .model import DeformationUNet, apply_deformation, total_variation_loss


@dataclass
class TrainingConfig:
    iterations: int = 1000
    learning_rate: float = 3e-3
    content_weight: float = 15.0
    style_weight: float = 800.0
    tv_weight: float = 5.0
    image_size: int = 256
    preview_every: int = 25
    snapshot_every: int = 50


@dataclass
class TrainingResult:
    final_image: torch.Tensor
    deformation_field: torch.Tensor
    content: torch.Tensor
    style: torch.Tensor
    snapshots: list[tuple[int, torch.Tensor]]


ProgressCallback = Callable[[int, int, dict[str, float], torch.Tensor], None]
CancelCallback = Callable[[], bool]


class PixelShufflerTrainer:
    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        content_path: str | Path,
        style_path: str | Path,
        output_dir: str | Path = "output",
        on_progress: Optional[ProgressCallback] = None,
        should_cancel: Optional[CancelCallback] = None,
    ) -> TrainingResult:
        cfg = self.config
        device = self.device
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        content = load_image_tensor(content_path, cfg.image_size, device)
        style = load_image_tensor(style_path, cfg.image_size, device)

        net = DeformationUNet().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        lpips_fn = lpips.LPIPS(net="alex").to(device)

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False

        style_layers = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1"}
        style_features = get_vgg_features(style, vgg, style_layers)
        style_grams = {name: calc_gram_matrix(style_features[name]) for name in style_features}

        snapshots: list[tuple[int, torch.Tensor]] = []
        deformation_field = torch.zeros(1, 2, cfg.image_size, cfg.image_size, device=device)
        deformed = style

        for step in range(cfg.iterations):
            if should_cancel and should_cancel():
                break

            optimizer.zero_grad()
            deformation_field = net(content, style)
            deformed = apply_deformation(style, deformation_field)

            loss_content = lpips_fn(deformed, content).mean()
            loss_style = style_loss(deformed, style_grams, vgg, style_layers)
            loss_tv = total_variation_loss(deformation_field)

            total = (
                loss_content * cfg.content_weight
                + loss_style * cfg.style_weight
                + loss_tv * cfg.tv_weight
            )
            total.backward()
            optimizer.step()

            losses = {
                "content": float(loss_content.item()),
                "style": float(loss_style.item()),
                "tv": float(loss_tv.item()),
                "total": float(total.item()),
            }

            if on_progress and (step == 0 or (step + 1) % cfg.preview_every == 0):
                on_progress(step + 1, cfg.iterations, losses, deformed.detach())

            if (step + 1) % cfg.snapshot_every == 0:
                snap = deformed.detach().clone()
                snapshots.append((step + 1, snap))
                save_tensor_image(snap, out / f"step_{step + 1:04d}.png")

        save_tensor_image(deformed, out / "final_stylized.png")

        return TrainingResult(
            final_image=deformed.detach(),
            deformation_field=deformation_field.detach(),
            content=content,
            style=style,
            snapshots=snapshots,
        )
