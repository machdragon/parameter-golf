#!/usr/bin/env bash
# After training on RunPod: meta + metrics JSON + tarball for runpodctl send.
#
# Usage (from repo root):
#   ./scripts/runpod_finish.sh <RUN_ID>
#
# Then on the pod:
#   runpodctl send "logs/${RUN_ID}_bundle.tar.gz"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:?Usage: runpod_finish.sh RUN_ID}"

export RUN_ID

echo "== writing logs/${RUN_ID}.meta.txt =="
./scripts/make_run_meta.sh

echo "== writing logs/${RUN_ID}.metrics.json =="
python3 scripts/extract_run_metrics.py "logs/${RUN_ID}.txt" -o "logs/${RUN_ID}.metrics.json"

echo "== bundling logs/${RUN_ID}_bundle.tar.gz =="
./scripts/bundle_run_results.sh "$RUN_ID" "logs/${RUN_ID}_bundle.tar.gz"

echo ""
echo "Next on pod:"
echo "  runpodctl send $(pwd)/logs/${RUN_ID}_bundle.tar.gz"
echo "On laptop:"
echo "  runpodctl receive <ONE_TIME_CODE>"
