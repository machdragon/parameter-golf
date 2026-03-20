#!/usr/bin/env bash
# Quick checks before a long RunPod job (template image + dataset).
# Run from repo root: ./scripts/runpod_preflight.sh
# Optional: ./scripts/runpod_preflight.sh 8  (expect 8 GPUs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPECT_GPUS="${1:-0}"

echo "== parameter-golf RunPod preflight (repo: $REPO_ROOT) =="

echo "-- Python / torch --"
python3 - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
except Exception as e:
    print("torch import error:", e)
PY

echo "-- nvidia-smi --"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
  N="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$EXPECT_GPUS" -gt 0 ]]; then
    if [[ "$N" -lt "$EXPECT_GPUS" ]]; then
      echo "WARNING: expected $EXPECT_GPUS GPUs, saw $N" >&2
    else
      echo "OK: $N GPU(s) visible (expected >= $EXPECT_GPUS)"
    fi
  fi
else
  echo "nvidia-smi not in PATH"
fi

echo "-- data (sp1024 default paths) --"
DATA_DIR="${RUNPOD_DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
TOK="${RUNPOD_TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
if [[ -d "$DATA_DIR" ]]; then
  NBIN=$(find "$DATA_DIR" -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')
  echo "OK: dataset dir $DATA_DIR ($NBIN train shard(s) at top level)"
else
  echo "MISSING: $DATA_DIR — run: python3 data/cached_challenge_fineweb.py --variant sp1024"
fi
if [[ -f "$TOK" ]]; then
  echo "OK: tokenizer $TOK"
else
  echo "MISSING: $TOK — run cached_challenge_fineweb.py"
fi

echo "-- disk --"
df -h "$REPO_ROOT" 2>/dev/null | tail -1 || true

echo "== preflight done =="
