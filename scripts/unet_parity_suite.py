#!/usr/bin/env python3
"""
unet_parity_suite.py

Policy-driven parity suite runner for UNet boundary tensors.

Baseline layout (reference):
  reports/unet_boundary/<run_id>/step_XXX_tYYY_noise_pred.pt

Candidate layout (TTNN, future):
  reports/ttnn_unet_outputs/<run_id>/step_XXX_tYYY_noise_pred.pt

For each run_id under baseline, it compares the noise_pred tensor at the chosen step.
Writes:
- per-item JSON report (via scripts/compare_tensors.py)
- a suite_summary.json aggregating pass/fail counts and paths
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_CLI = REPO_ROOT / "scripts" / "compare_tensors.py"

BASE_ROOT = REPO_ROOT / "reports" / "unet_boundary"
CAND_ROOT_DEFAULT = REPO_ROOT / "reports" / "ttnn_unet_outputs"
REPORT_DIR_DEFAULT = REPO_ROOT / "reports" / "parity_unet"


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _find_noise_pred_file(run_dir: Path, step_index: int) -> Path:
    # Expect exactly one file matching step_{idx:03d}_t*_noise_pred.pt
    pat = f"step_{step_index:03d}_t*_noise_pred.pt"
    matches = sorted(run_dir.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No baseline noise_pred found for pattern {pat} in {run_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple baseline noise_pred matches for {pat} in {run_dir}: {matches}")
    return matches[0]


def run_compare(ref: Path, cand: Path, policy: Path, regime: str, report_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(COMPARE_CLI),
        "--reference", str(ref),
        "--candidate", str(cand),
        "--policy", str(policy),
        "--regime", regime,
        "--report-dir", str(report_dir),
    ]
    p = subprocess.run(cmd)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="configs/parity_thresholds.yaml")
    ap.add_argument("--regime", type=str, default="early_ttnn", help="stub|early_ttnn|final")
    ap.add_argument("--step-index", type=int, default=0, help="Which diffusion step index to compare (0-based)")
    ap.add_argument("--baseline-root", type=str, default=str(BASE_ROOT))
    ap.add_argument("--candidate-root", type=str, default=str(CAND_ROOT_DEFAULT))
    ap.add_argument("--report-dir", type=str, default=str(REPORT_DIR_DEFAULT))
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N runs (for quick checks)")
    args = ap.parse_args()

    base_root = Path(args.baseline_root)
    cand_root = Path(args.candidate_root)
    policy = Path(args.policy)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not base_root.exists():
        print(f"ERROR: baseline root not found: {base_root}")
        return 2

    run_dirs = sorted([p for p in base_root.iterdir() if p.is_dir()])
    if args.limit and args.limit > 0:
        run_dirs = run_dirs[: args.limit]

    if not run_dirs:
        print(f"ERROR: no run dirs under baseline root: {base_root}")
        return 2

    summary: Dict[str, Any] = {
        "ok": True,
        "regime": args.regime,
        "policy": str(policy),
        "baseline_root": str(base_root),
        "candidate_root": str(cand_root),
        "step_index": args.step_index,
        "items": [],
        "failures": 0,
        "total": 0,
        "generated_at": _now_ts(),
    }

    failures = 0
    total = 0

    for run_dir in run_dirs:
        run_id = run_dir.name
        total += 1

        try:
            ref = _find_noise_pred_file(run_dir, args.step_index)
        except Exception as e:
            failures += 1
            summary["items"].append(
                {"run_id": run_id, "ok": False, "error": f"baseline_missing: {e}", "ref": None, "cand": None}
            )
            print(f"[{run_id}] FAIL baseline_missing: {e}")
            continue

        cand_dir = cand_root / run_id
        if not cand_dir.exists():
            failures += 1
            summary["items"].append(
                {"run_id": run_id, "ok": False, "error": "candidate_missing_dir", "ref": str(ref), "cand": None}
            )
            print(f"[{run_id}] FAIL candidate_missing_dir: {cand_dir}")
            continue

        try:
            cand = _find_noise_pred_file(cand_dir, args.step_index)
        except Exception as e:
            failures += 1
            summary["items"].append(
                {"run_id": run_id, "ok": False, "error": f"candidate_missing: {e}", "ref": str(ref), "cand": None}
            )
            print(f"[{run_id}] FAIL candidate_missing: {e}")
            continue

        print(f"\n=== {run_id} ===")
        rc = run_compare(ref, cand, policy, args.regime, report_dir)
        ok = (rc == 0)
        if not ok:
            failures += 1

        summary["items"].append({"run_id": run_id, "ok": ok, "ref": str(ref), "cand": str(cand), "rc": rc})

    summary["failures"] = failures
    summary["total"] = total
    summary["ok"] = (failures == 0)

    out_path = report_dir / f"suite_summary_{args.regime}_step{args.step_index:03d}_{_now_ts()}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==============================")
    if failures == 0:
        print("UNET PARITY SUITE: PASS")
        print(f"suite_summary_json: {out_path}")
        return 0
    print(f"UNET PARITY SUITE: FAIL ({failures} failing item(s))")
    print(f"suite_summary_json: {out_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
