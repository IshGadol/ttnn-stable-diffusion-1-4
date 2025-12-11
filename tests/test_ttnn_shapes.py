from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttnn_impl.text_encoder_tt import (  # type: ignore[import]
    TTNNTextEncoderConfig,
    TTNNTextEncoder,
)
from ttnn_impl.unet_tt import (  # type: ignore[import]
    TTNNUNetConfig,
    TTNNUNet,
)
from ttnn_impl.vae_tt import (  # type: ignore[import]
    TTNNVAEDecoderConfig,
    TTNNVAEDecoder,
)


class TestTTNNShapes(unittest.TestCase):
    def test_ttnn_text_encoder_shape(self) -> None:
        cfg = TTNNTextEncoderConfig(max_length=77, embedding_dim=768)
        enc = TTNNTextEncoder(config=cfg, dtype=torch.float32)

        prompts = ["a castle on a hill", "a cat on a skateboard"]
        context = enc.encode(prompts)

        self.assertEqual(context.shape, (2, 77, 768))

    def test_ttnn_unet_shape(self) -> None:
        cfg = TTNNUNetConfig(height=512, width=512, in_channels=4, out_channels=4)
        unet = TTNNUNet(config=cfg, dtype=torch.float32)

        B = 2
        latents = torch.randn(B, 4, 64, 64)
        timesteps = torch.randint(low=0, high=1000, size=(B,), dtype=torch.long)
        context = torch.randn(B, 77, 768)

        out = unet(latents=latents, timesteps=timesteps, context=context)

        self.assertEqual(out.shape, (B, 4, 64, 64))

    def test_ttnn_vae_shape(self) -> None:
        cfg = TTNNVAEDecoderConfig(out_channels=3, height=512, width=512, latent_channels=4, downscale_factor=8)
        vae = TTNNVAEDecoder(config=cfg, dtype=torch.float32)

        B = 2
        latents = torch.randn(B, 4, 64, 64)
        images = vae.decode(latents)

        self.assertEqual(images.shape, (B, 3, 512, 512))


if __name__ == "__main__":
    unittest.main()
