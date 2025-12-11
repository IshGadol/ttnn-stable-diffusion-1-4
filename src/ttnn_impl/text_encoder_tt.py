from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TTNNTextEncoderConfig:
    """
    TTNN configuration for SD 1.4 text encoder.

    Mirrors SD14TextEncoderConfig but without any HF/transformers dependency.
    """

    max_length: int = 77
    embedding_dim: int = 768  # matches SD 1.4 CLIP text encoder


class TTNNTextEncoder:
    """
    Skeleton TTNN implementation of the SD 1.4 text encoder.

    For now:
    - Accepts a list of strings as prompts.
    - Returns a dummy tensor of correct shape: (B, L, D).

    Later:
    - Tokenization + CLIP encoder graph will be replaced by TTNN ops.
    """

    def __init__(
        self,
        config: Optional[TTNNTextEncoderConfig] = None,
        device: str = "ttnn",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or TTNNTextEncoderConfig()
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def encode(self, prompts: List[str]) -> torch.Tensor:
        """
        Dummy TTNN text encoder output.

        Args:
            prompts: list of strings (length B)

        Returns:
            (B, L, D) tensor of zeros.
        """
        B = len(prompts)
        L = self.config.max_length
        D = self.config.embedding_dim

        # Dummy CLIP context
        context = torch.zeros((B, L, D), dtype=self.dtype)
        return context
