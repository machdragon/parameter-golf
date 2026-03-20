#!/usr/bin/env bash
# Write logs/${RUN_ID}.meta.txt with UTC time, git revision, branch, and GPU list.
#
# Usage (from repo root):
#   RUN_ID=my_run ./scripts/make_run_meta.sh
#
# On RunPod, run before or after training in the same environment as the job.
set -euo pipefail

RUN_ID="${RUN_ID:?Set RUN_ID (e.g. RUN_ID=prod_8xh100)}"
OUT="${1:-logs/${RUN_ID}.meta.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -L 2>/dev/null | nl -v 0 -s ', ' || true
  else
    echo "gpus=nvidia-smi not available"
  fi
} >"$OUT"

echo "Wrote $OUT"
