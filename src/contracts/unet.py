from __future__ import annotations

import torch

from .common import TensorSpec, assert_tensor

# For 512x512 SD1.x: latent spatial = 64x64, channels=4

LATENTS_XT = TensorSpec(
    name="unet.latents_xt",
    shape=(-1, 4, 64, 64),
    dtype=torch.float32,
    notes="Noisy latents x_t at timestep t. Shape (B, 4, 64, 64).",
)

TIMESTEP = TensorSpec(
    name="unet.timestep",
    shape=(-1,),
    dtype=torch.int64,
    notes="Per-sample timestep index. Shape (B,).",
)

ENCODER_HIDDEN_STATES = TensorSpec(
    name="unet.encoder_hidden_states",
    shape=(-1, 77, 768),
    dtype=torch.float32,
    notes="CLIP embeddings. Shape (B, 77, 768).",
)

NOISE_PRED = TensorSpec(
    name="unet.noise_pred",
    shape=(-1, 4, 64, 64),
    dtype=torch.float32,
    notes="Predicted noise epsilon. Shape (B, 4, 64, 64).",
)


def validate_inputs(latents_xt: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> None:
    assert_tensor(latents_xt, LATENTS_XT)
    assert_tensor(timestep, TIMESTEP)
    assert_tensor(encoder_hidden_states, ENCODER_HIDDEN_STATES)


def validate_outputs(noise_pred: torch.Tensor) -> None:
    assert_tensor(noise_pred, NOISE_PRED)
