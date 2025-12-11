from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import UNet2DConditionModel


@dataclass
class SD14UNetConfig:
    """
    Configuration for the Stable Diffusion 1.4 UNet.

    This is intentionally minimal and focused on:
    - model_id: HF repo containing the UNet weights.
    """

    model_id: str = "CompVis/stable-diffusion-v1-4"


class SD14UNet:
    """
    Thin wrapper around the SD 1.4 UNet (UNet2DConditionModel).

    Expected shapes:
        latents:  (B, 4, 64, 64)
        timesteps: (B,) or scalar
        context: (B, L, D)  (e.g. L=77, D=768)

    Returns:
        noise_pred: (B, 4, 64, 64)
    """

    def __init__(
        self,
        config: Optional[SD14UNetConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or SD14UNetConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self._unet: Optional[UNet2DConditionModel] = None

    def _load_unet(self) -> UNet2DConditionModel:
        if self._unet is not None:
            return self._unet

        unet = UNet2DConditionModel.from_pretrained(
            self.config.model_id,
            subfolder="unet",
        )
        unet = unet.to(self.device, dtype=self.dtype)
        self._unet = unet
        return unet

    @torch.inference_mode()
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a UNet denoising step.

        Args:
            latents: (B, 4, 64, 64)
            timesteps: scalar or (B,) tensor
            context: (B, L, D) text conditioning

        Returns:
            noise_pred: (B, 4, 64, 64)
        """
        unet = self._load_unet()

        latents = latents.to(self.device, dtype=self.dtype)
        context = context.to(self.device, dtype=self.dtype)
        timesteps = timesteps.to(self.device)

        out = unet(
            sample=latents,
            timestep=timesteps,
            encoder_hidden_states=context,
        )
        noise_pred = out.sample  # (B, 4, 64, 64)
        return noise_pred
