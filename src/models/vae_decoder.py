from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import AutoencoderKL


@dataclass
class SD14VAEDecoderConfig:
    """
    Configuration for the Stable Diffusion 1.4 VAE decoder.
    """

    model_id: str = "CompVis/stable-diffusion-v1-4"


class SD14VAEDecoder:
    """
    Thin wrapper around the SD 1.4 VAE decoder.

    Expected shapes:
        latents: (B, 4, 64, 64)

    Returns:
        images: (B, 3, H, W)  (nominally ~512x512, may be slightly different then cropped)
    """

    def __init__(
        self,
        config: Optional[SD14VAEDecoderConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or SD14VAEDecoderConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self._vae: Optional[AutoencoderKL] = None

    def _load_vae(self) -> AutoencoderKL:
        if self._vae is not None:
            return self._vae

        vae = AutoencoderKL.from_pretrained(
            self.config.model_id,
            subfolder="vae",
        )
        vae = vae.to(self.device, dtype=self.dtype)
        self._vae = vae
        return vae

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents into image space.

        Args:
            latents: (B, 4, 64, 64)

        Returns:
            images: (B, 3, H, W), in range [-1, 1]
        """
        vae = self._load_vae()
        latents = latents.to(self.device, dtype=self.dtype)
        images = vae.decode(latents).sample  # (B, 3, H, W)
        return images
