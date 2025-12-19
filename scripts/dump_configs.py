#!/usr/bin/env python3
"""
dump_configs.py

Dumps authoritative config snapshots for SD v1.x pipeline components:
- UNet
- VAE
- Text Encoder
- Scheduler

Writes:
  configs/sd14_config_dump.json

Safe to run without TTNN. Requires diffusers/transformers installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "configs" / "sd14_config_dump.json"


def _to_jsonable(obj):
    # Prefer real config dicts when available (diffusers / transformers configs)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, torch.dtype):
        return str(obj)
    return str(obj)


def main() -> int:
    # NOTE: Using SD v1-5 as default model_id. If you later want strict v1.4,
    # we will pin the exact model/revision consistently across the suite + manifests.
    model_id = "CompVis/stable-diffusion-v1-4"

    print(f"Loading pipeline: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


    dump = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "torch": torch.__version__,
        "components": {
            "unet": _to_jsonable(getattr(pipe.unet, "config", {})),
            "vae": _to_jsonable(getattr(pipe.vae, "config", {})),
            "text_encoder": _to_jsonable(getattr(pipe.text_encoder, "config", {})),
            "scheduler": _to_jsonable(getattr(pipe.scheduler, "config", {})),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(dump, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
