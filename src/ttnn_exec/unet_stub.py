"""
unet_stub.py

TTNN UNet execution scaffold for SD v1.4.

This file:
- Enforces UNet contracts
- Defines the TTNN-facing execution interface
- Can run in CPU-stub mode for early integration
- Will later be backed by real TTNN ops

NO kernel logic lives here yet.
"""

from __future__ import annotations

import torch

from src.contracts import unet as unet_contract


class TTNNUNet:
    """
    Contract-locked UNet interface.

    Future:
    - load TTNN graph
    - execute on N300
    """

    def __init__(self):
        self.initialized = False

    def initialize(self):
        """
        Placeholder for TTNN graph compilation / loading.
        """
        self.initialized = True

    def forward(
        self,
        latents_xt: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute UNet forward pass.

        Contract:
        - latents_xt: (B,4,64,64) float32
        - timestep:   (B,) int64
        - encoder_hidden_states: (B,77,768) float32
        Returns:
        - noise_pred: (B,4,64,64) float32
        """

        if not self.initialized:
            raise RuntimeError("TTNNUNet not initialized")

        # Enforce input contracts
        unet_contract.validate_inputs(
            latents_xt=latents_xt,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        # -------------------------------------------------
        # STUB EXECUTION (CPU placeholder)
        # -------------------------------------------------
        # Replace this with TTNN execution later
        noise_pred = torch.zeros_like(latents_xt)

        # Enforce output contract
        unet_contract.validate_outputs(noise_pred)

        return noise_pred
