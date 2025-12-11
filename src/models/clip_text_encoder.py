from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class SD14TextEncoderConfig:
    """
    Configuration for the Stable Diffusion 1.4 text encoder.

    This is intentionally minimal and focused on:
    - model_id: Hugging Face repo for SD 1.4 (text encoder + tokenizer)
    - max_length: token sequence length (77 for SD 1.x)
    """

    model_id: str = "CompVis/stable-diffusion-v1-4"
    max_length: int = 77


class SD14TextEncoder:
    """
    Thin wrapper around the SD 1.4 CLIP text encoder.

    Responsibilities:
    - Handle tokenization of one or more prompts.
    - Run the CLIP text encoder to produce context embeddings.
    - Expose a simple API for later TTNN mirroring.

    Expected output shape:
    - context: (B, L, D) where:
        B = batch size
        L = sequence length (77)
        D = embedding dim (768 for SD 1.4 text encoder)
    """

    def __init__(
        self,
        config: Optional[SD14TextEncoderConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or SD14TextEncoderConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self._tokenizer: Optional[CLIPTokenizer] = None
        self._encoder: Optional[CLIPTextModel] = None

    def _load_tokenizer(self) -> CLIPTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer",
        )
        self._tokenizer = tokenizer
        return tokenizer

    def _load_encoder(self) -> CLIPTextModel:
        if self._encoder is not None:
            return self._encoder

        encoder = CLIPTextModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
        )
        encoder = encoder.to(self.device, dtype=self.dtype)
        self._encoder = encoder
        return encoder

    def tokenize(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of prompts.

        Returns:
            input_ids: (B, L) LongTensor
            attention_mask: (B, L) LongTensor
        """
        tokenizer = self._load_tokenizer()
        encoded = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return input_ids, attention_mask

    @torch.inference_mode()
    def encode(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Encode a batch of text prompts into CLIP embeddings.

        Args:
            prompts: list of prompt strings, length B.

        Returns:
            context embeddings of shape (B, L, D).
        """
        encoder = self._load_encoder()
        input_ids, attention_mask = self.tokenize(prompts)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Last hidden state: (B, L, D)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
