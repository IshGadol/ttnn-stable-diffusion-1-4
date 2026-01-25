#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fail() {
  echo "REPO SANITY: FAIL: $*" >&2
  exit 1
}

warn() {
  echo "REPO SANITY: WARN: $*" >&2
}

cd "$ROOT"

echo "== repo sanity check =="
echo "root: $ROOT"

# 1) Forbidden tracked dirs that should never be committed
# Note: This script checks presence; gitignore should already prevent tracking.
if [ -d ".venv" ]; then
  warn ".venv/ exists (fine locally). Ensure it is ignored and not tracked."
fi

# 2) Caches
cache_hits="$(find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) \
  -not -path './.git/*' -not -path './.venv/*' 2>/dev/null | head -n 20 || true)"
if [ -n "$cache_hits" ]; then
  warn "cache directories found (first 20):"
  echo "$cache_hits"
  warn "These should be ignored; consider cleaning before packaging."
fi

# 3) Reports / outputs
if [ -d "reports" ]; then
  warn "reports/ exists (expected locally). Ensure it is ignored and not tracked."
fi
if [ -d "outputs" ]; then
  warn "outputs/ exists. Ensure it is ignored and not tracked."
fi

# 4) Large files threshold (default 50MB)
MAX_MB="${MAX_MB:-50}"
MAX_BYTES=$((MAX_MB * 1024 * 1024))

echo "Checking for files > ${MAX_MB}MB (excluding .git, .venv, reports, outputs)..."

large_files="$(find . -type f \
  -not -path './.git/*' \
  -not -path './.venv/*' \
  -not -path './reports/*' \
  -not -path './.hf_cache/*' \
  -not -path './outputs/*' \
  -size +"${MAX_BYTES}"c \
  2>/dev/null | head -n 20 || true)"

if [ -n "$large_files" ]; then
  fail "Large files detected (first 20 shown). Remove or gitignore them:\n$large_files"
fi

# 5) Tracked forbidden patterns (hard fail)
tracked_bad="$(git ls-files -z | tr '\0' '\n' | grep -E '(^\.venv/|^reports/|^outputs/|__pycache__/|\.pytest_cache/|\.mypy_cache/|\.ruff_cache/)' || true)"
if [ -n "$tracked_bad" ]; then
  fail "Forbidden paths are tracked by git:\n$tracked_bad"
fi

# 6) Optional: warn about HF cache path perms that can break runs
# (This does NOT fail; it only warns.)
if [ -d "/mnt/md0/models/hf_cache" ]; then
  # If hub dir exists and is not writable, warn
  if [ -d "/mnt/md0/models/hf_cache/hub" ] && [ ! -w "/mnt/md0/models/hf_cache/hub" ]; then
    warn "/mnt/md0/models/hf_cache/hub is not writable by current user; HF downloads may fail unless you override HF_HOME."
  fi
fi

echo "REPO SANITY: PASS"
