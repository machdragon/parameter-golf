#!/usr/bin/env bash
# Bundle training artifacts for transfer (RunPod, local, or after `modal volume get`).
#
# Usage:
#   ./scripts/bundle_run_results.sh RUN_ID [output.tar.gz] [root_dir]
#
# Default root_dir is the current directory (repo root on RunPod: /workspace/parameter-golf).
# After downloading Modal outputs: ./scripts/bundle_run_results.sh my-run ./my-run.tgz ./modal_runs/my-run
set -euo pipefail

RUN_ID="${1:?Usage: bundle_run_results.sh RUN_ID [output.tar.gz] [root_dir]}"
OUT="${2:-logs/${RUN_ID}_bundle.tar.gz}"
ROOT="${3:-.}"

cd "$ROOT"

if [[ ! -f "logs/${RUN_ID}.txt" ]]; then
  echo "Warning: logs/${RUN_ID}.txt not found under $ROOT (cwd=$(pwd))" >&2
fi

mapfile -t FILES < <(
  [[ -f "logs/${RUN_ID}.txt" ]] && echo "logs/${RUN_ID}.txt"
  [[ -f "logs/${RUN_ID}.meta.txt" ]] && echo "logs/${RUN_ID}.meta.txt"
  [[ -f "logs/${RUN_ID}.metrics.json" ]] && echo "logs/${RUN_ID}.metrics.json"
  [[ -f "final_model.int8.ptz" ]] && echo "final_model.int8.ptz"
  [[ -f "final_model.pt" ]] && echo "final_model.pt"
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No known artifacts found for RUN_ID=$RUN_ID under $ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
tar czvf "$OUT" "${FILES[@]}"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
