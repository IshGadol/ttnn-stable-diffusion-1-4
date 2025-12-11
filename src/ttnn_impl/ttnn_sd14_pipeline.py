from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch


@dataclass
class TTNNStableDiffusionConfig:
    """
    Configuration for the TTNN Stable Diffusion 1.4 pipeline.

    This intentionally mirrors CPUStableDiffusionConfig so that:
    - swapping CPU/TTNN implementations in higher-level code is trivial
    - tests can assert consistency on shared parameters
    """

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512

    # TTNN / device-specific knobs will be added later, e.g.:
    # nchip: int = 1
    # tile_shape: tuple[int, int] = ()


class TTNNStableDiffusionPipeline:
    """
    Skeleton TTNN pipeline for Stable Diffusion 1.4.

    For now this is a placeholder that:
    - exposes the same high-level API as CPUStableDiffusionPipelineWrapper
    - returns dummy tensors with the expected shapes
    - does NOT import or require any TTNN libraries yet

    This lets us:
    - write tests against the API surface and shapes
    - wire CLI / higher-level code without needing working hardware
    """

    def __init__(
        self,
        config: TTNNStableDiffusionConfig | None = None,
        device: str = "ttnn",  # logical device identifier, not a torch device
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or TTNNStableDiffusionConfig()
        self.device = device
        self.dtype = dtype

    def load(self) -> None:
        """
        Placeholder for TTNN graph compilation / weight loading.

        In a future implementation, this will:
        - load SD 1.4 weights
        - build TTNN graphs for UNet, VAE, and text encoder
        - set up any device / buffer state
        """
        # No-op for now.
        return None

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a forward pass of the TTNN SD 1.4 pipeline.

        For now:
        - returns a dict with a dummy latent tensor of shape (1, 4, 64, 64)
          which corresponds to a 512x512 latent-space resolution.
        - higher-level code or tests can assert against this shape.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional seed for later reproducibility.

        Returns:
            A dict containing at least:
                - "latents": torch.Tensor of shape (1, 4, 64, 64)
        """
        # placeholder deterministic behaviour if seed is set
        if seed is not None:
            torch.manual_seed(seed)

        # 512x512 SD latents typically use a 1/8 spatial downscale: 512 / 8 = 64
        latents = torch.zeros(
            (1, 4, 64, 64),
            dtype=self.dtype,
        )

        return {
            "prompt": prompt,
            "latents": latents,
            "config": {
                "height": self.config.height,
                "width": self.config.width,
                "num_inference_steps": self.config.num_inference_steps,
                "guidance_scale": self.config.guidance_scale,
            },
            "device": self.device,
        }
