#!/usr/bin/env python3
"""
parity_check_suite.py

Runs parity comparisons across the prompt suite.

Modes:
1) baseline-self: compares each run's final_latent_x0.pt to itself (sanity PASS)
2) baseline-vs-ttnn: compares baseline final_latent_x0.pt to a TTNN-produced tensor
   placed in a parallel directory structure.

Expected layout:
baseline:
  reports/parity_baseline/<date>/<suite_id>/runs/<item_id>/final_latent_x0.pt

ttnn candidate (future):
  reports/ttnn_outputs/<date>/<suite_id>/runs/<item_id>/final_latent_x0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE = REPO_ROOT / "src" / "parity" / "compare_latents.py"

BASE_ROOT = REPO_ROOT / "reports" / "parity_baseline"
TTNN_ROOT = REPO_ROOT / "reports" / "ttnn_outputs"


def run_compare(a: Path, b: Path, atol: float, rtol: float) -> int:
    cmd = [
        sys.executable,
        str(COMPARE),
        "--a", str(a),
        "--b", str(b),
        "--atol", str(atol),
        "--rtol", str(rtol),
    ]
    p = subprocess.run(cmd)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD folder under reports/")
    ap.add_argument("--suite", required=True, help="suite_id folder (e.g. sd14_ddim_golden_v1)")
    ap.add_argument("--mode", choices=["baseline-self", "baseline-vs-ttnn"], default="baseline-self")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    baseline_runs = BASE_ROOT / args.date / args.suite / "runs"
    if not baseline_runs.exists():
        print(f"ERROR: baseline runs path not found: {baseline_runs}")
        return 2

    run_dirs = sorted([p for p in baseline_runs.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"ERROR: no run dirs found under: {baseline_runs}")
        return 2

    failures = 0
    for d in run_dirs:
        item_id = d.name
        a = d / "final_latent_x0.pt"
        if args.mode == "baseline-self":
            b = a
        else:
            b = TTNN_ROOT / args.date / args.suite / "runs" / item_id / "final_latent_x0.pt"

        if not a.exists():
            print(f"[{item_id}] MISSING baseline tensor: {a}")
            failures += 1
            continue
        if not b.exists():
            print(f"[{item_id}] MISSING candidate tensor: {b}")
            failures += 1
            continue

        print(f"\n=== {item_id} ===")
        rc = run_compare(a, b, args.atol, args.rtol)
        if rc != 0:
            failures += 1

    print("\n==============================")
    if failures == 0:
        print("SUITE PARITY: PASS")
        return 0
    print(f"SUITE PARITY: FAIL ({failures} failing item(s))")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
