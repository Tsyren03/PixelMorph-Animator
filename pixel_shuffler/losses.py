"""Perceptual and style losses used during optimization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_vgg_features(image: torch.Tensor, model: torch.nn.Module, layer_ids: dict[str, str]) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_ids:
            features[name] = x
    return features


def calc_gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.size()
    flat = tensor.view(b * c, h * w)
    return torch.mm(flat, flat.t()) / (c * h * w)


def style_loss(
    deformed: torch.Tensor,
    style_grams: dict[str, torch.Tensor],
    vgg: torch.nn.Module,
    style_layer_ids: dict[str, str],
) -> torch.Tensor:
    features = get_vgg_features(deformed, vgg, style_layer_ids)
    loss = torch.tensor(0.0, device=deformed.device)
    for layer_id in style_layer_ids:
        loss = loss + F.mse_loss(calc_gram_matrix(features[layer_id]), style_grams[layer_id])
    return loss
