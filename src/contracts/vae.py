from __future__ import annotations

import torch

from .common import TensorSpec, assert_tensor

LATENTS = TensorSpec(
    name="vae.latents",
    shape=(-1, 4, 64, 64),
    dtype=torch.float32,
    notes="Latents to decode. Shape (B, 4, 64, 64).",
)

IMAGES = TensorSpec(
    name="vae.images",
    shape=(-1, 3, 512, 512),
    dtype=torch.float32,
    notes="Decoded images. Shape (B, 3, 512, 512). Range depends on postprocess.",
)


def validate_inputs(latents: torch.Tensor) -> None:
    assert_tensor(latents, LATENTS)


def validate_outputs(images: torch.Tensor) -> None:
    assert_tensor(images, IMAGES)
