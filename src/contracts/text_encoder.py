from __future__ import annotations

import torch

from .common import TensorSpec, assert_tensor

# SD1.x CLIP text encoder (OpenAI CLIP ViT-L/14 text model)
# max_length is 77, hidden_size is 768

INPUT_IDS = TensorSpec(
    name="text_encoder.input_ids",
    shape=(-1, 77),
    dtype=torch.int64,
    notes="Tokenized prompt ids. Shape (B, 77).",
)

ENCODER_HIDDEN_STATES = TensorSpec(
    name="text_encoder.encoder_hidden_states",
    shape=(-1, 77, 768),
    dtype=torch.float32,
    notes="CLIP text embeddings. Shape (B, 77, 768).",
)


def validate_inputs(input_ids: torch.Tensor) -> None:
    assert_tensor(input_ids, INPUT_IDS)


def validate_outputs(encoder_hidden_states: torch.Tensor) -> None:
    assert_tensor(encoder_hidden_states, ENCODER_HIDDEN_STATES)
