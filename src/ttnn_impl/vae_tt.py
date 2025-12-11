from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class TTNNVAEDecoderConfig:
    """
    TTNN configuration for SD 1.4 VAE decoder.
    """

    out_channels: int = 3
    height: int = 512
    width: int = 512
    latent_channels: int = 4
    downscale_factor: int = 8  # 512->64 latent resolution


class TTNNVAEDecoder:
    """
    Skeleton TTNN implementation of the SD 1.4 VAE decoder.

    For now:
    - Accepts latents (B,4,64,64)
    - Returns dummy upsampled image (B,3,512,512)
    """

    def __init__(
        self,
        config: Optional[TTNNVAEDecoderConfig] = None,
        device: str = "ttnn",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or TTNNVAEDecoderConfig()
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        B = latents.shape[0]

        H = self.config.height
        W = self.config.width
        C = self.config.out_channels

        # Dummy image: just zero-tensor upsampled to correct resolution
        zeros = torch.zeros((B, C, H, W), dtype=self.dtype)
        return zeros
