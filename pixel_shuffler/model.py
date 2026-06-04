"""Neural deformation field (U-Net) and pixel warping for Pixel Shuffler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeformationUNet(nn.Module):
    """Predicts a 2-channel deformation field from content + style images."""

    def __init__(self) -> None:
        super().__init__()
        self.down1 = DoubleConv(6, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up2 = DoubleConv(128 + 256, 128)
        self.up1 = DoubleConv(64 + 128, 64)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = torch.cat([content, style], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        bottleneck = self.down3(self.pool(d2))
        u2 = self.up2(torch.cat([self.up(bottleneck), d2], dim=1))
        u1 = self.up1(torch.cat([self.up(u2), d1], dim=1))
        return torch.tanh(self.out_conv(u1)) * 0.5


def apply_deformation(
    image: torch.Tensor,
    deformation_field: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Warp `image` by scaling the predicted field with `alpha` in [0, 1]."""
    n, _c, h, w = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=image.device),
        torch.linspace(-1, 1, w, device=image.device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
    delta = deformation_field.permute(0, 2, 3, 1) * alpha
    grid = base_grid + delta
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )


def total_variation_loss(deformation_field: torch.Tensor) -> torch.Tensor:
    tv_h = torch.mean(torch.abs(deformation_field[:, :, 1:, :] - deformation_field[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(deformation_field[:, :, :, 1:] - deformation_field[:, :, :, :-1]))
    return tv_h + tv_w
