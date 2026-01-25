#!/usr/bin/env python3
"""
UNet boundary runner for SD1.4 CPU baseline.

Captures boundary tensors around the UNet call:
- latent_model_input (B=2,4,64,64)  [uncond, cond]
- timestep (int)
- context / encoder_hidden_states (2, L, D)
- noise_pred (2,4,64,64)

Writes deterministic filenames into reports/unet_boundary/.

This is intended to support later TTNN bring-up:
- "feed captured inputs to TTNN UNet"
- compare TTNN output to captured noise_pred under tolerance policy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.cpu_sd14_pipeline import CPUStableDiffusionConfig, CPUStableDiffusionPipelineWrapper  # type: ignore[import]


def _stable_id(prompt: str, seed: Optional[int], steps: int, guidance: float, h: int, w: int) -> str:
    # Deterministic, filesystem-safe id (no hashing dependency required)
    # Keep prompt short-ish but stable.
    p = "".join(ch if ch.isalnum() else "_" for ch in prompt.strip())[:48]
    s = "none" if seed is None else str(seed)
    g = f"{guidance:.4f}".rstrip("0").rstrip(".")
    return f"sd14_unet_boundary__p_{p}__seed_{s}__steps_{steps}__g_{g}__{h}x{w}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda (default: auto-detect in pipeline)")
    ap.add_argument("--capture-step-index", type=int, default=0, help="Which diffusion step to save (0-based).")
    ap.add_argument("--outdir", type=str, default="reports/unet_boundary")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Explicitly disable parallelism noise for deterministic-ish behavior
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = CPUStableDiffusionConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    pipe = CPUStableDiffusionPipelineWrapper(config=cfg, device=args.device)

    capture: Dict[str, Any] = {"captures": []}

    run_id = _stable_id(args.prompt, args.seed, args.steps, args.guidance_scale, args.height, args.width)
    outdir = Path(args.outdir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "device": str(pipe.device),
        "dtype": str(pipe.dtype),
        "capture_step_index": args.capture_step_index,
    }

    # Run pipeline once; boundary_capture accumulates per-step tensors.
    _ = pipe(prompt=args.prompt, seed=args.seed, output_path=None, boundary_capture=capture)

    captures = capture.get("captures", [])
    if not captures:
        raise RuntimeError("No boundary captures were recorded. Pipeline hook may not be active.")

    idx = args.capture_step_index
    if idx < 0 or idx >= len(captures):
        raise ValueError(f"capture_step_index {idx} out of range (0..{len(captures)-1})")

    item = captures[idx]
    timestep = int(item["timestep"])
    latent_model_input = item["latent_model_input"]
    context = item["context"]
    noise_pred = item["noise_pred"]

    # Save tensors deterministically
    torch.save(latent_model_input, outdir / f"step_{idx:03d}_t{timestep}_latent_model_input.pt")
    torch.save(context, outdir / f"step_{idx:03d}_t{timestep}_context.pt")
    torch.save(noise_pred, outdir / f"step_{idx:03d}_t{timestep}_noise_pred.pt")

    # Write a small JSON manifest for easy inspection
    tensor_info = {
        "timestep": timestep,
        "latent_model_input": {"shape": list(latent_model_input.shape), "dtype": str(latent_model_input.dtype)},
        "context": {"shape": list(context.shape), "dtype": str(context.dtype)},
        "noise_pred": {"shape": list(noise_pred.shape), "dtype": str(noise_pred.dtype)},
    }
    meta["tensor_info"] = tensor_info

    (outdir / "boundary_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[unet-boundary] OK")
    print(f"[unet-boundary] outdir: {outdir}")
    print(f"[unet-boundary] saved step index: {idx} (timestep={timestep})")
    print(f"[unet-boundary] latent_model_input: {tuple(latent_model_input.shape)} {latent_model_input.dtype}")
    print(f"[unet-boundary] context: {tuple(context.shape)} {context.dtype}")
    print(f"[unet-boundary] noise_pred: {tuple(noise_pred.shape)} {noise_pred.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
