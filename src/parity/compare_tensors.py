from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


@dataclass
class ParityMetrics:
    max_abs: float
    mean_abs: float
    rmse: float
    rel_l2: float


@dataclass
class ParityReport:
    ok: bool
    regime: str
    thresholds: Dict[str, float]
    metrics: ParityMetrics
    reference_path: str
    candidate_path: str
    details: Dict[str, Any]


def _load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "tensor" in obj:
        obj = obj["tensor"]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor at {path}, got {type(obj)}")
    return obj


def _compute_metrics(a: torch.Tensor, b: torch.Tensor, eps: float) -> ParityMetrics:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: ref={tuple(a.shape)} cand={tuple(b.shape)}")

    # Promote to float32 for stable metrics
    a32 = a.detach().to(dtype=torch.float32)
    b32 = b.detach().to(dtype=torch.float32)

    diff = (a32 - b32).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(torch.mean((a32 - b32) ** 2)).item()) if diff.numel() else 0.0

    a_l2 = float(torch.linalg.vector_norm(a32).item()) if a32.numel() else 0.0
    d_l2 = float(torch.linalg.vector_norm(a32 - b32).item()) if a32.numel() else 0.0
    rel_l2 = float(d_l2 / (a_l2 + eps)) if (a_l2 + eps) != 0 else float("inf")

    return ParityMetrics(max_abs=max_abs, mean_abs=mean_abs, rmse=rmse, rel_l2=rel_l2)


def load_policy(policy_path: Path, regime: str) -> Dict[str, Any]:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    regimes = policy.get("regimes", {})
    if regime not in regimes:
        raise KeyError(f"Regime '{regime}' not found in {policy_path}. Available: {list(regimes.keys())}")
    eps = float(policy.get("eps", {}).get("rel_l2", 1.0e-12))
    thresholds = regimes[regime]
    return {"thresholds": thresholds, "eps": eps}


def compare_tensors(
    reference_path: Path,
    candidate_path: Path,
    policy_path: Path,
    regime: str,
) -> ParityReport:
    ref = _load_tensor(reference_path)
    cand = _load_tensor(candidate_path)

    pol = load_policy(policy_path, regime)
    thr = pol["thresholds"]
    eps = float(pol["eps"])

    metrics = _compute_metrics(ref, cand, eps=eps)

    ok = (
        metrics.max_abs <= float(thr["max_abs"])
        and metrics.mean_abs <= float(thr["mean_abs"])
        and metrics.rmse <= float(thr["rmse"])
        and metrics.rel_l2 <= float(thr["rel_l2"])
    )

    details: Dict[str, Any] = {
        "ref_shape": list(ref.shape),
        "cand_shape": list(cand.shape),
        "ref_dtype": str(ref.dtype),
        "cand_dtype": str(cand.dtype),
        "numel": int(ref.numel()),
    }

    return ParityReport(
        ok=ok,
        regime=regime,
        thresholds={k: float(v) for k, v in thr.items()},
        metrics=metrics,
        reference_path=str(reference_path),
        candidate_path=str(candidate_path),
        details=details,
    )


def report_to_json(report: ParityReport) -> str:
    return json.dumps(asdict(report), indent=2)
