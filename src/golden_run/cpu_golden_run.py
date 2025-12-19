#!/usr/bin/env python3
"""
cpu_golden_run.py

CPU-only golden baseline harness for SD v1.4 (CompVis/stable-diffusion-v1-4).
- Uses DDIM scheduler
- Reads configs/prompt_suite_ddim_v1.yaml
- Saves:
  - final.png
  - final_latent_x0.pt
  - checkpoint latents at 0/25/50/75/100%
- Writes manifest.yaml with environment + run parameters

This is Phase 4 parity spine. No TTNN required.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO_ROOT / "configs" / "prompt_suite_ddim_v1.yaml"
REPORTS_ROOT = REPO_ROOT / "reports" / "parity_baseline"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_tensor(path: Path, t: torch.Tensor) -> None:
    # Save CPU float32 tensors for stable comparisons
    t = t.detach().to("cpu").to(torch.float32).contiguous()
    torch.save(t, path)


def save_png(path: Path, img: Image.Image) -> None:
    img.save(path)


@dataclass
class SuiteDefaults:
    steps: int
    guidance_scale: float
    height: int
    width: int
    batch_size: int = 1


@dataclass
class SuiteItem:
    id: str
    prompt: str
    seed: int
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None


def load_suite(path: Path) -> tuple[Dict[str, Any], SuiteDefaults, List[SuiteItem]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    defaults = data.get("defaults", {})
    d = SuiteDefaults(
        steps=int(defaults.get("steps", 50)),
        guidance_scale=float(defaults.get("guidance_scale", 7.5)),
        height=int(defaults.get("height", 512)),
        width=int(defaults.get("width", 512)),
        batch_size=int(defaults.get("batch_size", 1)),
    )

    items: List[SuiteItem] = []
    for it in data.get("items", []):
        items.append(
            SuiteItem(
                id=str(it["id"]),
                prompt=str(it["prompt"]),
                seed=int(it["seed"]),
                negative_prompt=str(it["negative_prompt"]) if "negative_prompt" in it else None,
                steps=int(it["steps"]) if "steps" in it else None,
                guidance_scale=float(it["guidance_scale"]) if "guidance_scale" in it else None,
                height=int(it["height"]) if "height" in it else None,
                width=int(it["width"]) if "width" in it else None,
            )
        )

    return data, d, items


def build_pipe(model_id: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # new name; avoids torch_dtype deprecation
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cpu")

    # Force DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Determinism-friendly settings
    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_one(
    pipe: StableDiffusionPipeline,
    out_dir: Path,
    item: SuiteItem,
    defaults: SuiteDefaults,
    *,
    suite_id: str,
    model_id: str,
    run_date: str,
    checkpoint_fracs: List[float],
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    steps = item.steps if item.steps is not None else defaults.steps
    cfg = item.guidance_scale if item.guidance_scale is not None else defaults.guidance_scale
    h = item.height if item.height is not None else defaults.height
    w = item.width if item.width is not None else defaults.width

    # Deterministic RNG on CPU
    gen = torch.Generator(device="cpu").manual_seed(item.seed)

    # Collect latents at selected steps
    checkpoints: Dict[str, torch.Tensor] = {}
    step_indices = sorted({max(0, min(steps - 1, int(round(frac * (steps - 1))))) for frac in checkpoint_fracs})

    def cb(pipeline, step: int, timestep: int, callback_kwargs: Dict[str, Any]):
        # diffusers >= 0.36 passes tensors via callback_kwargs
        if step in step_indices:
            latents = callback_kwargs["latents"]
            checkpoints[f"{step:03d}"] = (
                latents.detach()
                .to("cpu")
                .to(torch.float32)
                .clone()
            )
        return callback_kwargs

    result = pipe(
        prompt=item.prompt,
        negative_prompt=item.negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        height=h,
        width=w,
        generator=gen,
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pil",
    )

    # Save final image
    final_png = out_dir / "final.png"
    save_png(final_png, result.images[0])

    # Save final latents (x0) by re-running one more call with output_type="latent"
    # This keeps the baseline explicit even if pipeline output changes later.
    result_latent = pipe(
        prompt=item.prompt,
        negative_prompt=item.negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        height=h,
        width=w,
        generator=torch.Generator(device="cpu").manual_seed(item.seed),
        output_type="latent",
    )
    x0 = result_latent.images  # latent tensor
    final_x0 = out_dir / "final_latent_x0.pt"
    save_tensor(final_x0, x0)

    # Save checkpoints as requested: map step index -> percent label
    for frac in checkpoint_fracs:
        s = max(0, min(steps - 1, int(round(frac * (steps - 1)))))
        key = f"{s:03d}"
        if key not in checkpoints:
            continue
        pct = int(round(frac * 100))
        save_tensor(out_dir / f"latents_step_{pct:03d}.pt", checkpoints[key])

    return {
        "item_id": item.id,
        "prompt": item.prompt,
        "negative_prompt": item.negative_prompt,
        "seed": item.seed,
        "steps": steps,
        "guidance_scale": cfg,
        "height": h,
        "width": w,
        "artifacts": {
            "final_png": str(final_png.relative_to(REPO_ROOT)),
            "final_latent_x0": str(final_x0.relative_to(REPO_ROOT)),
            "checkpoint_fracs": checkpoint_fracs,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default=str(DEFAULT_SUITE), help="Path to prompt suite YAML")
    ap.add_argument("--only", type=str, default="", help="Run only this item_id (optional)")
    ap.add_argument("--date", type=str, default="", help="Override date folder (YYYY-MM-DD). Default: today (UTC)")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    suite_raw, defaults, items = load_suite(suite_path)

    suite_id = str(suite_raw.get("suite_id", "sd14_ddim_golden"))
    model_id = "CompVis/stable-diffusion-v1-4"  # strict

    run_date = args.date.strip() or datetime.now(timezone.utc).date().isoformat()
    base_dir = REPORTS_ROOT / run_date / suite_id
    runs_dir = base_dir / "runs"
    ensure_dir(runs_dir)

    # Checkpoint fractions required by Phase 4
    checkpoint_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]

    # Build pipeline
    pipe = build_pipe(model_id)

    selected = items
    if args.only.strip():
        selected = [it for it in items if it.id == args.only.strip()]
        if not selected:
            print(f"ERROR: item_id '{args.only.strip()}' not found in suite.", file=sys.stderr)
            return 2

    run_entries: List[Dict[str, Any]] = []
    for it in selected:
        out_dir = runs_dir / it.id
        print(f"Running item: {it.id}")
        entry = run_one(
            pipe,
            out_dir,
            it,
            defaults,
            suite_id=suite_id,
            model_id=model_id,
            run_date=run_date,
            checkpoint_fracs=checkpoint_fracs,
        )
        run_entries.append(entry)

    manifest = {
        "generated_utc": utc_now_iso(),
        "suite_file": str(suite_path.relative_to(REPO_ROOT)),
        "suite_sha256": sha256_file(suite_path),
        "suite_id": suite_id,
        "model_id": model_id,
        "scheduler": "DDIMScheduler",
        "defaults": {
            "steps": defaults.steps,
            "guidance_scale": defaults.guidance_scale,
            "height": defaults.height,
            "width": defaults.width,
            "batch_size": defaults.batch_size,
        },
        "env": {
            "python": str(sys.version).replace("\n", " "),
            "torch": str(torch.__version__),
            "hf_home": os.environ.get("HF_HOME", ""),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
        },
        "runs": run_entries,
    }

    manifest_path = base_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
