from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn.functional as F

from ttnn_impl.text_encoder_tt import TTNNTextEncoder, TTNNTextEncoderConfig
from ttnn_impl.unet_tt import TTNNUNet, TTNNUNetConfig
from ttnn_impl.vae_tt import TTNNVAEDecoder, TTNNVAEDecoderConfig


@dataclass
class TTNNStableDiffusionConfig:
    """
    TTNN Stable Diffusion 1.4 configuration.

    Mirrors CPUStableDiffusionConfig but uses TTNN modules.
    """

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512


class TTNNStableDiffusionPipeline:
    """
    Skeleton TTNN implementation of Stable Diffusion 1.4.

    This mirrors the CPUStableDiffusionPipelineWrapper structure:
    - Text encoder
    - UNet
    - Simple diffusion loop
    - VAE decoder

    For now, returns dummy images because TTNN kernels are not implemented yet.
    """

    def __init__(
        self,
        config: TTNNStableDiffusionConfig | None = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or TTNNStableDiffusionConfig()
        self.device_str = device or "ttnn"
        self.device = self.device_str
        self.dtype = dtype

        # TTNN model components
        self.text_encoder = TTNNTextEncoder(
            TTNNTextEncoderConfig(),
            device=self.device,
            dtype=self.dtype,
        )
        self.unet = TTNNUNet(
            TTNNUNetConfig(
                height=self.config.height,
                width=self.config.width,
                latent_channels=4,
                downscale_factor=8,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        self.vae_decoder = TTNNVAEDecoder(
            TTNNVAEDecoderConfig(
                out_channels=3,
                height=self.config.height,
                width=self.config.width,
                latent_channels=4,
                downscale_factor=8,
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def _encode_prompts(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Dummy TTNN text encoder: returns zeros with correct shape.
        """
        return self.text_encoder.encode(prompts)

    def _prepare_latents(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """
        TTNN initial latents: zeros for now.

        Shape: (B, 4, 64, 64)
        """
        H = self.config.height // 8
        W = self.config.width // 8
        C = 4

        return torch.zeros((batch_size, C, H, W), dtype=self.dtype)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ):
        """
        Run the TTNN diffusion loop (dummy for now).

        Returns:
            PIL.Image or numpy array placeholder.
        """
        B = 1

        # Classifier-free guidance: cond + uncond
        prompts = [prompt]
        uncond_prompts = [""]

        cond_context = self._encode_prompts(prompts)          # (1, 77, 768)
        uncond_context = self._encode_prompts(uncond_prompts)  # (1, 77, 768)
        context = torch.cat([uncond_context, cond_context], dim=0)

        # Dummy latents
        latents = self._prepare_latents(batch_size=B)

        # Diffusion loop: UNet returns zeros, scheduler omitted
        for _ in range(self.config.num_inference_steps):
            latent_model_input = torch.cat([latents] * 2, dim=0)
            noise_pred = self.unet(
                latents=latent_model_input,
                timesteps=torch.tensor([0]),
                context=context,
            )
            # Split → guidance → step
            noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + self.config.guidance_scale * (noise_text - noise_uncond)
            # For now, latents remain zeros

        # Decode dummy latents
        images = self.vae_decoder.decode(latents)  # (1, 3, 512, 512)

        # Convert from [-1, 1] to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)

        # Convert to numpy (placeholder)
        images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        return images_np[0]
