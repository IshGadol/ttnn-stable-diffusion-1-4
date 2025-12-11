#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.cpu_sd14_pipeline import (  # type: ignore[import]
    CPUStableDiffusionConfig,
    CPUStableDiffusionPipelineWrapper,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion 1.4 (CPU/CUDA diffusers baseline)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to generate from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/cpu_sd14_out.png",
        help="Output image path (PNG). Default: outputs/cpu_sd14_out.png",
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
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. 'cpu' or 'cuda'. Default: auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CPUStableDiffusionConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    pipe = CPUStableDiffusionPipelineWrapper(
        config=config,
        device=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cpu-sd14] Using device={pipe.device}, steps={config.num_inference_steps}, guidance={config.guidance_scale}")
    print(f"[cpu-sd14] Prompt: {args.prompt!r}")
    if args.seed is not None:
        print(f"[cpu-sd14] Seed: {args.seed}")

    try:
        image = pipe(
            prompt=args.prompt,
            seed=args.seed,
            output_path=str(out_path),
        )
    except Exception as exc:
        print("[cpu-sd14] ERROR during generation:")
        print(f"  {type(exc).__name__}: {exc}")
        print(
            "\nHints:\n"
            "- Ensure you have accepted the license for 'CompVis/stable-diffusion-v1-4' on Hugging Face.\n"
            "- Run `huggingface-cli login` or set the HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var.\n"
            "- Verify that your GPU drivers and CUDA stack are compatible with the installed torch build.\n"
        )
        sys.exit(1)

    print(f"[cpu-sd14] Image generated and saved to: {out_path}")
    # 'image' is a PIL.Image; we just return success here.


if __name__ == "__main__":
    main()
