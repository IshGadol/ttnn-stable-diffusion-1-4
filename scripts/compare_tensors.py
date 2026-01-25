#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parity.compare_tensors import compare_tensors, report_to_json  # type: ignore[import]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare two torch tensors with parity policy + JSON report output.")
    ap.add_argument("--reference", type=str, required=True, help="Path to reference .pt tensor")
    ap.add_argument("--candidate", type=str, required=True, help="Path to candidate .pt tensor")
    ap.add_argument("--policy", type=str, default="configs/parity_thresholds.yaml", help="Parity policy YAML")
    ap.add_argument("--regime", type=str, default="early_ttnn", help="Policy regime: stub|early_ttnn|final")
    ap.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    ap.add_argument("--report-dir", type=str, default="reports/parity", help="Directory to write JSON report")
    ap.add_argument("--report-name", type=str, default=None, help="Optional explicit report filename (json).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    ref = Path(args.reference)
    cand = Path(args.candidate)
    pol = Path(args.policy)

    report = compare_tensors(ref, cand, pol, args.regime)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    name = args.report_name
    if name is None:
        # deterministic-ish: based on filenames + regime + timestamp
        name = f"parity_{args.regime}_{ref.stem}__vs__{cand.stem}_{_now_ts()}.json"

    out_path = report_dir / name
    out_path.write_text(report_to_json(report), encoding="utf-8")

    # Human summary
    print(f"[parity] {'PASS' if report.ok else 'FAIL'} regime={report.regime}")
    print(f"[parity] ref: {ref}")
    print(f"[parity] cand: {cand}")
    print(f"[parity] metrics: max_abs={report.metrics.max_abs:.6g} mean_abs={report.metrics.mean_abs:.6g} rmse={report.metrics.rmse:.6g} rel_l2={report.metrics.rel_l2:.6g}")
    print(f"[parity] report_json: {out_path}")

    if args.json:
        print(report_to_json(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
