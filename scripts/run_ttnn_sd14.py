#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.ttnn_sd14_pipeline import (  # type: ignore[import]
    TTNNStableDiffusionConfig,
    TTNNStableDiffusionPipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TTNN Stable Diffusion 1.4 (skeleton implementation).",
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
        help="Number of diffusion steps. Default: 30",
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
        "--output",
        type=str,
        default=None,
        help="Optional output image path (PNG). If omitted, no file is saved.",
    )
    return parser.parse_args()


def save_image_from_numpy(arr: np.ndarray, output_path: Path) -> None:
    """
    Convert a [H, W, 3] float image in [0, 1] to uint8 PNG.
    """
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255).astype("uint8")
    img = Image.fromarray(arr_uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


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
        device="ttnn",
    )

    print(
        f"[ttnn-sd14] device={pipe.device}, steps={cfg.num_inference_steps}, "
        f"guidance={cfg.guidance_scale}, size={cfg.height}x{cfg.width}"
    )
    print(f"[ttnn-sd14] Prompt: {args.prompt!r}")
    if args.seed is not None:
        print(f"[ttnn-sd14] Seed: {args.seed}")

    image_np = pipe(
        prompt=args.prompt,
        seed=args.seed,
        output_path=None,
    )

    print(f"[ttnn-sd14] Output image shape: {image_np.shape}")

    if args.output is not None:
        out_path = Path(args.output)
        save_image_from_numpy(image_np, out_path)
        print(f"[ttnn-sd14] Saved dummy TTNN image to: {out_path}")


if __name__ == "__main__":
    main()
