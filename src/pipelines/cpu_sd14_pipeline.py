from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline


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
    Thin wrapper around diffusers.StableDiffusionPipeline used as the CPU baseline.

    Notes:
    - Device defaults to 'cuda' if available, otherwise 'cpu'.
    - We keep all arguments explicit so the TTNN path can mirror this API.
    """

    def __init__(
        self,
        config: CPUStableDiffusionConfig | None = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.config = config or CPUStableDiffusionConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._pipe: Optional[StableDiffusionPipeline] = None

    def load(self) -> StableDiffusionPipeline:
        """
        Lazily load the diffusers pipeline and move it to the desired device.
        """
        if self._pipe is not None:
            return self._pipe

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,  # keep it simple; not part of bounty scope
        )
        pipe = pipe.to(self.device)
        self._pipe = pipe
        return self._pipe

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
        pipe = self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = pipe(
            prompt,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        )
        image = result.images[0]

        if output_path is not None:
            image.save(output_path)

        return image
