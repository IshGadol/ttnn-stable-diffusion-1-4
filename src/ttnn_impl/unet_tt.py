from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TTNNUNetConfig:
    """
    TTNN configuration for SD 1.4 UNet.

    Mirrors SD14UNetConfig but without requiring diffusers.
    """

    height: int = 512
    width: int = 512
    latent_channels: int = 4  # SD latent channels
    downscale_factor: int = 8  # 512/8 = 64 latent resolution


class TTNNUNet:
    """
    Skeleton TTNN implementation of the SD 1.4 UNet.

    For now:
    - Accepts latents, timesteps, and context.
    - Returns a dummy noise prediction tensor of shape (B, 4, 64, 64).

    Real implementation will replace this with TTNN graph execution.
    """

    def __init__(
        self,
        config: Optional[TTNNUNetConfig] = None,
        device: str = "ttnn",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or TTNNUNetConfig()
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dummy UNet forward.

        Returns:
            noise_pred: zeros with correct latent shape.
        """
        B = latents.shape[0]
        C = self.config.latent_channels
        H = self.config.height // self.config.downscale_factor
        W = self.config.width // self.config.downscale_factor

        return torch.zeros((B, C, H, W), dtype=self.dtype)
