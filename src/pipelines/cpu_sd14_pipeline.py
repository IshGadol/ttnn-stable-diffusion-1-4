from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
from diffusers import PNDMScheduler
from diffusers.utils import numpy_to_pil

from models.clip_text_encoder import SD14TextEncoder, SD14TextEncoderConfig
from models.unet_sd14 import SD14UNet, SD14UNetConfig
from models.vae_decoder import SD14VAEDecoder, SD14VAEDecoderConfig


@dataclass
class CPUStableDiffusionConfig:
    """
    Configuration for the CPU (or CUDA) Stable Diffusion 1.4 reference pipeline.
    This is our correctness baseline for later TTNN ports.
    """

    model_id: str = "CompVis/stable-diffusion-v1-4"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512


class CPUStableDiffusionPipelineWrapper:
    """
    Modular SD 1.4 pipeline used as the CPU baseline.

    This implementation:
    - uses SD14TextEncoder, SD14UNet, and SD14VAEDecoder wrappers
    - uses a PNDMScheduler from diffusers
    - exposes a simple __call__ API mirroring what the TTNN path will use
    """

    def __init__(
        self,
        config: CPUStableDiffusionConfig | None = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or CPUStableDiffusionConfig()
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.dtype = dtype

        # Core model components
        self.text_encoder = SD14TextEncoder(
            SD14TextEncoderConfig(model_id=self.config.model_id),
            device=self.device,
            dtype=self.dtype,
        )
        self.unet = SD14UNet(
            SD14UNetConfig(model_id=self.config.model_id),
            device=self.device,
            dtype=self.dtype,
        )
        self.vae_decoder = SD14VAEDecoder(
            SD14VAEDecoderConfig(model_id=self.config.model_id),
            device=self.device,
            dtype=self.dtype,
        )

        # Scheduler (reference CPU scheduler)
        self.scheduler = PNDMScheduler.from_pretrained(
            self.config.model_id,
            subfolder="scheduler",
        )

    def _encode_prompts(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Encode prompts to CLIP text embeddings.

        Returns:
            context: (B, L, D)
        """
        return self.text_encoder.encode(prompts)

    def _prepare_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Prepare initial Gaussian latents for diffusion.

        Shape: (B, 4, 64, 64) for 512x512 images.
        """
        height = self.config.height
        width = self.config.width
        latent_channels = 4
        downscale_factor = 8

        latents = torch.randn(
            (
                batch_size,
                latent_channels,
                height // downscale_factor,
                width // downscale_factor,
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ):
        """
        Generate a single 512x512 image from a text prompt.

        Returns:
            PIL.Image.Image
        """
        batch_size = 1

        # Set up generator for reproducibility
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Classifier-free guidance: cond + uncond
        prompts = [prompt]
        uncond_prompts = [""]

        cond_context = self._encode_prompts(prompts)          # (1, L, D)
        uncond_context = self._encode_prompts(uncond_prompts)  # (1, L, D)
        context = torch.cat([uncond_context, cond_context], dim=0)  # (2, L, D)

        # Timesteps
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Initial latents
        latents = self._prepare_latents(batch_size=batch_size, generator=generator)

        # Denoising loop
        for t in timesteps:
            # Expand latents for classifier-free guidance: [uncond, cond]
            latent_model_input = torch.cat([latents] * 2, dim=0)

            # UNet forward
            noise_pred = self.unet(
                latents=latent_model_input,
                timesteps=t,
                context=context,
            )  # (2, 4, 64, 64)

            # Split uncond/cond and apply guidance
            noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + self.config.guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            step_result = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                generator=generator,
            )
            latents = step_result.prev_sample

        # Decode latents to images
        # SD 1.4 uses a scaling factor ~0.18215 in its pipeline; we follow that.
        latents = latents / 0.18215
        images = self.vae_decoder.decode(latents)  # (B, 3, H, W) in [-1, 1]

        # Convert from [-1, 1] to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)

        # To CPU numpy
        images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        pil_images = numpy_to_pil(images_np)
        image = pil_images[0]

        if output_path is not None:
            image.save(output_path)

        return image
