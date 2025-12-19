from __future__ import annotations

import torch

from .common import TensorSpec, assert_tensor

TIMESTEP = TensorSpec(
    name="scheduler.timestep",
    shape=(-1,),
    dtype=torch.int64,
    notes="DDIM timestep indices per batch element. Shape (B,).",
)

LATENTS_XT = TensorSpec(
    name="scheduler.latents_xt",
    shape=(-1, 4, 64, 64),
    dtype=torch.float32,
    notes="Latents x_t at timestep t. Shape (B, 4, 64, 64).",
)

NOISE_PRED = TensorSpec(
    name="scheduler.noise_pred",
    shape=(-1, 4, 64, 64),
    dtype=torch.float32,
    notes="UNet epsilon prediction. Shape (B, 4, 64, 64).",
)

LATENTS_XT_MINUS_1 = TensorSpec(
    name="scheduler.latents_xt_minus_1",
    shape=(-1, 4, 64, 64),
    dtype=torch.float32,
    notes="Updated latents x_{t-1}. Shape (B, 4, 64, 64).",
)


def validate_step_inputs(latents_xt: torch.Tensor, timestep: torch.Tensor, noise_pred: torch.Tensor) -> None:
    assert_tensor(latents_xt, LATENTS_XT)
    assert_tensor(timestep, TIMESTEP)
    assert_tensor(noise_pred, NOISE_PRED)


def validate_step_outputs(latents_xt_minus_1: torch.Tensor) -> None:
    assert_tensor(latents_xt_minus_1, LATENTS_XT_MINUS_1)
