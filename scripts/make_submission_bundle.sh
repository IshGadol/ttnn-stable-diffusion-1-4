#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ts="$(date +%Y%m%d_%H%M%S)"
OUTDIR="reports/submission_bundle/${ts}"
BASENAME="ttnn_sd14_bundle_${ts}"
STAGE="${OUTDIR}/${BASENAME}"

mkdir -p "$OUTDIR"
mkdir -p "$STAGE"

echo "== submission bundle =="
echo "root: $ROOT"
echo "outdir: $OUTDIR"

# 1) Sanity gate
echo
echo "== repo sanity gate =="
bash scripts/repo_sanity_check.sh

# 2) Copy source-controlled scaffolding
echo
echo "== staging files =="
mkdir -p "${STAGE}/configs" "${STAGE}/scripts" "${STAGE}/src" "${STAGE}/tests"

cp -a README.md STATUS.md requirements.txt Makefile "${STAGE}/" 2>/dev/null || true
cp -a configs/. "${STAGE}/configs/"
cp -a scripts/. "${STAGE}/scripts/"
cp -a src/. "${STAGE}/src/"
cp -a tests/. "${STAGE}/tests/"

# 3) Remove local-only files from staged tree
echo
echo "== cleaning stage =="
rm -rf "${STAGE}/scripts/__pycache__" "${STAGE}/src/**/__pycache__" "${STAGE}/tests/__pycache__" 2>/dev/null || true
find "${STAGE}" -type f -name "*.pyc" -delete 2>/dev/null || true

# 4) Optional: include only SMALL boundary manifests (no tensors)
# This is useful evidence without huge blobs.
echo
echo "== optional: include boundary manifests only =="
mkdir -p "${STAGE}/evidence/unet_boundary_manifests"
if [ -d "reports/unet_boundary" ]; then
  find reports/unet_boundary -maxdepth 2 -type f -name "boundary_manifest.json" -print0 \
    | while IFS= read -r -d '' f; do
        run_id="$(basename "$(dirname "$f")")"
        cp "$f" "${STAGE}/evidence/unet_boundary_manifests/${run_id}_boundary_manifest.json"
      done
fi

# 5) Create tarball
echo
echo "== create tar.gz =="
tar -C "$OUTDIR" -czf "${OUTDIR}/${BASENAME}.tar.gz" "${BASENAME}"

echo "BUNDLE OK:"
echo "  ${OUTDIR}/${BASENAME}.tar.gz"
