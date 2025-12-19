#!/usr/bin/env python3
"""
compare_latents.py

Parity comparator for SD v1.4 baseline tensors.
Compares two torch tensors saved via torch.save() and reports error metrics.

Usage examples:
  # Self-check (should be exact match)
  python src/parity/compare_latents.py \
    --a reports/parity_baseline/2025-12-19/sd14_ddim_golden_v1/runs/astronaut/final_latent_x0.pt \
    --b reports/parity_baseline/2025-12-19/sd14_ddim_golden_v1/runs/astronaut/final_latent_x0.pt

  # Later: compare TTNN output tensor vs CPU baseline
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    # Some pipelines may store dicts; allow {"tensor": ...}
    if isinstance(obj, dict):
        for k in ("tensor", "latents", "x0"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
    raise TypeError(f"Unsupported tensor payload in {path}: {type(obj)}")


def metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "a_shape": tuple(a.shape),
            "b_shape": tuple(b.shape),
        }

    # Ensure float32 for stable numeric comparisons
    a32 = a.detach().to("cpu").to(torch.float32).contiguous()
    b32 = b.detach().to("cpu").to(torch.float32).contiguous()

    diff = a32 - b32
    absdiff = diff.abs()

    max_abs = float(absdiff.max().item()) if absdiff.numel() else 0.0
    mean_abs = float(absdiff.mean().item()) if absdiff.numel() else 0.0
    rmse = float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0

    # cosine similarity (nan-safe)
    af = a32.flatten()
    bf = b32.flatten()

    a_norm = af.norm().item()
    b_norm = bf.norm().item()

    if a_norm == 0.0 and b_norm == 0.0:
        cos = 1.0
    elif a_norm == 0.0 or b_norm == 0.0:
        cos = 0.0
    else:
        cos = float(torch.dot(af, bf).item() / (a_norm * b_norm))

    return {
        "shape_match": True,
        "shape": tuple(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rmse": rmse,
        "cosine": cos,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to baseline tensor (.pt)")
    ap.add_argument("--b", required=True, help="Path to candidate tensor (.pt)")
    ap.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance gate")
    ap.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance gate (max_abs / max(|a|))")
    args = ap.parse_args()

    pa = Path(args.a)
    pb = Path(args.b)

    a = load_tensor(pa)
    b = load_tensor(pb)

    m = metrics(a, b)
    if not m.get("shape_match", False):
        print("SHAPE MISMATCH")
        print(m)
        return 2

    # Relative error denominator: max(|a|)
    amax = float(a.detach().to("cpu").abs().max().item()) if a.numel() else 0.0
    rel = (m["max_abs_err"] / amax) if amax != 0.0 else float("inf") if m["max_abs_err"] != 0.0 else 0.0
    m["max_abs_rel"] = rel

    print(f"A: {pa}")
    print(f"B: {pb}")
    print(f"shape: {m['shape']}")
    print(f"dtype_a: {m['dtype_a']}  dtype_b: {m['dtype_b']}")
    print(f"max_abs_err: {m['max_abs_err']:.8g}")
    print(f"mean_abs_err: {m['mean_abs_err']:.8g}")
    print(f"rmse: {m['rmse']:.8g}")
    print(f"cosine: {m['cosine']:.10f}")
    print(f"max_abs_rel: {m['max_abs_rel']:.8g}")
    print(f"gate atol={args.atol} rtol={args.rtol}")

    ok_atol = m["max_abs_err"] <= args.atol
    ok_rtol = m["max_abs_rel"] <= args.rtol
    if ok_atol and ok_rtol:
        print("PARITY: PASS")
        return 0
    print("PARITY: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
