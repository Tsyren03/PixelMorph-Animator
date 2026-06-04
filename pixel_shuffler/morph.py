"""Morph animation by interpolating the learned deformation field."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .io_utils import tensor_to_numpy_rgb
from .model import apply_deformation


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def build_deformation_morph(
    style: torch.Tensor,
    deformation_field: torch.Tensor,
    num_frames: int = 60,
    easing: bool = True,
) -> list[np.ndarray]:
    """
    Animate from original style to fully warped result by scaling the field.
    This visualizes pixel rearrangement as in the Pixel Shuffler paper.
    """
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        alpha = smoothstep(t) if easing else t
        with torch.no_grad():
            warped = apply_deformation(style, deformation_field, alpha=alpha)
        frames.append(tensor_to_numpy_rgb(warped))
    return frames


def build_snapshot_morph(
  snapshots: Iterable[tuple[int, torch.Tensor]],
  style: torch.Tensor,
  hold_first: int = 8,
) -> list[np.ndarray]:
    """Animate through optimization snapshots, starting from the style image."""
    frames: list[np.ndarray] = [tensor_to_numpy_rgb(style)] * hold_first
    for _step, tensor in snapshots:
        frames.append(tensor_to_numpy_rgb(tensor))
    return frames


def build_blend_morph(
    content: torch.Tensor,
    result: torch.Tensor,
    num_frames: int = 60,
) -> list[np.ndarray]:
    """Optional crossfade from content structure to final stylized output."""
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        t = smoothstep(i / max(num_frames - 1, 1))
        with torch.no_grad():
            blended = (1.0 - t) * content + t * result
        frames.append(tensor_to_numpy_rgb(blended))
    return frames
