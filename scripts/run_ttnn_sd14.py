#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ttnn_impl.ttnn_sd14_pipeline import (  # type: ignore[import]
    TTNNStableDiffusionConfig,
    TTNNStableDiffusionPipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TTNN Stable Diffusion 1.4 (skeleton implementation)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to feed into the TTNN SD 1.4 pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion steps (mirrors CPU config). Default: 30",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale. Default: 7.5",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Target image height in pixels. Default: 512",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Target image width in pixels. Default: 512",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="ttnn",
        help="Logical TTNN device identifier. Default: 'ttnn'",
    )
    parser.add_argument(
        "--save-latents",
        type=str,
        default=None,
        help="Optional path to save dummy latents as a .pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TTNNStableDiffusionConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )

    pipe = TTNNStableDiffusionPipeline(
        config=cfg,
        device=args.device,
        dtype=torch.float32,
    )

    print(
        f"[ttnn-sd14] device={pipe.device}, steps={cfg.num_inference_steps}, "
        f"guidance={cfg.guidance_scale}, size={cfg.height}x{cfg.width}"
    )
    print(f"[ttnn-sd14] Prompt: {args.prompt!r}")
    if args.seed is not None:
        print(f"[ttnn-sd14] Seed: {args.seed}")

    out = pipe(prompt=args.prompt, seed=args.seed)

    latents = out["latents"]
    print(f"[ttnn-sd14] Latents shape: {tuple(latents.shape)}, dtype={latents.dtype}")

    if args.save_latents is not None:
        out_path = Path(args.save_latents)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents, out_path)
        print(f"[ttnn-sd14] Latents saved to: {out_path}")


if __name__ == "__main__":
    main()
