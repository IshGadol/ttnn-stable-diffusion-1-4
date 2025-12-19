#!/usr/bin/env bash
set -euo pipefail

DATE="${1:-}"
SUITE="${2:-}"

if [[ -z "$DATE" || -z "$SUITE" ]]; then
  echo "usage: $0 <date> <suite_id>"
  echo "example: $0 2025-12-19 sd14_ddim_golden_v1"
  exit 2
fi

items=(astronaut fantasy_castle lighthouse mechanical_watch portrait)

for item in "${items[@]}"; do
  python scripts/run_ttnn_stub_one.py --date "$DATE" --suite "$SUITE" --item "$item"
done

python scripts/parity_check_suite.py --date "$DATE" --suite "$SUITE" --mode baseline-vs-ttnn
