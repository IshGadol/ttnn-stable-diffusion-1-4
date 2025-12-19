#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.ttnn_exec.unet_stub import TTNNUNet
from src.ttnn_exec.output_writer import OutputLayout, write_final_latent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--item", required=True)
    args = ap.parse_args()

    # Load CPU baseline final latent as our "input latent"
    baseline = (
        Path("reports")
        / "parity_baseline"
        / args.date
        / args.suite
        / "runs"
        / args.item
        / "final_latent_x0.pt"
    )
    if not baseline.exists():
        raise SystemExit(f"Missing baseline latent: {baseline}")

    latents_xt = torch.load(baseline, map_location="cpu").to(torch.float32)

    # Minimal dummy timestep + dummy encoder states (contract-valid)
    B = latents_xt.shape[0]
    timestep = torch.zeros((B,), dtype=torch.int64)
    encoder_hidden_states = torch.zeros((B, 77, 768), dtype=torch.float32)

    u = TTNNUNet()
    u.initialize()

    # Stub UNet returns zeros-like tensor shaped like latents_xt
    noise_pred = u.forward(latents_xt, timestep, encoder_hidden_states)

    # For now, pretend "final_latent_x0" is the noise_pred (intentionally wrong)
    layout = OutputLayout(date=args.date, suite_id=args.suite, item_id=args.item)
    out_path = write_final_latent(layout, noise_pred)

    print(f"Wrote TTNN stub output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
