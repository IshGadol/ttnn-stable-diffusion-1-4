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

from pipelines.cpu_sd14_pipeline import (  # type: ignore[import]
    CPUStableDiffusionConfig,
    CPUStableDiffusionPipelineWrapper,
)
from ttnn_impl.ttnn_sd14_pipeline import (  # type: ignore[import]
    TTNNStableDiffusionConfig,
    TTNNStableDiffusionPipeline,
)


class TestAPISurface(unittest.TestCase):
    def test_cpu_config_defaults(self) -> None:
        cfg = CPUStableDiffusionConfig()
        self.assertEqual(cfg.height, 512)
        self.assertEqual(cfg.width, 512)
        self.assertEqual(cfg.num_inference_steps, 30)
        self.assertAlmostEqual(cfg.guidance_scale, 7.5)

    def test_ttnn_config_defaults(self) -> None:
        cfg = TTNNStableDiffusionConfig()
        self.assertEqual(cfg.height, 512)
        self.assertEqual(cfg.width, 512)
        self.assertEqual(cfg.num_inference_steps, 30)
        self.assertAlmostEqual(cfg.guidance_scale, 7.5)

    def test_ttnn_dummy_latent_shape(self) -> None:
        cfg = TTNNStableDiffusionConfig()
        pipe = TTNNStableDiffusionPipeline(config=cfg)
        out = pipe(prompt="a test prompt", seed=123)

        self.assertIn("latents", out)
        latents = out["latents"]
        self.assertIsInstance(latents, torch.Tensor)
        self.assertEqual(tuple(latents.shape), (1, 4, 64, 64))


if __name__ == "__main__":
    unittest.main()
