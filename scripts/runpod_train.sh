#!/usr/bin/env bash
# RunPod-optimized training launcher (Parameter Golf template / CUDA image).
# Sets standard env vars, sanity-checks data, runs torchrun from repo root.
#
# Usage (from repo root on the pod, e.g. /workspace/parameter-golf):
#   ./scripts/runpod_train.sh 1x --run-id smoke_$(date +%Y%m%d_%H%M)
#   ./scripts/runpod_train.sh 8x --run-id prod_8xh100_001
#
# Optional env overrides before calling:
#   RUNPOD_DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
#   RUNPOD_TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
#   VOCAB_SIZE=1024
#
# Custom entrypoint (record submission script):
#   ./scripts/runpod_train.sh 8x --run-id my_record --train-gpt records/.../train_gpt.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPUS=0
RUN_ID=""
TRAIN_GPT="train_gpt.py"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Usage: $0 <1x|8x> --run-id <id> [--train-gpt path]"
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    1x) GPUS=1; shift ;;
    8x) GPUS=8; shift ;;
    --run-id)
      RUN_ID="${2:?}"
      shift 2
      ;;
    --train-gpt)
      TRAIN_GPT="${2:?}"
      shift 2
      ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ "$GPUS" -eq 0 ]]; then
  echo "error: specify topology: 1x or 8x" >&2
  usage
fi
if [[ -z "$RUN_ID" ]]; then
  echo "error: --run-id is required" >&2
  usage
fi

DATA_DIR="${RUNPOD_DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
TOK_FILE="${RUNPOD_TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# train_gpt expects DATA_PATH with trailing slash (matches README examples)
DATA_PATH="${DATA_DIR%/}/"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "error: dataset directory missing: $DATA_DIR" >&2
  echo "  On RunPod: python3 data/cached_challenge_fineweb.py --variant sp1024 [--train-shards N]" >&2
  exit 1
fi
if [[ ! -f "$TOK_FILE" ]]; then
  echo "error: tokenizer missing: $TOK_FILE" >&2
  echo "  On RunPod: python3 data/cached_challenge_fineweb.py --variant sp1024" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_GPT" ]]; then
  echo "error: train script not found: $TRAIN_GPT" >&2
  exit 1
fi

if command -v nvidia-smi &>/dev/null; then
  NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${NGPU:-0}" -lt "$GPUS" ]]; then
    echo "warning: torchrun --nproc_per_node=$GPUS but only $NGPU GPU(s) visible (nvidia-smi -L)" >&2
  fi
fi

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH="$TOK_FILE"
export VOCAB_SIZE

echo "runpod_train: RUN_ID=$RUN_ID nproc=$GPUS train_gpt=$TRAIN_GPT"
echo "  DATA_PATH=$DATA_PATH"
echo "  TOKENIZER_PATH=$TOKENIZER_PATH VOCAB_SIZE=$VOCAB_SIZE"
echo "  cwd=$REPO_ROOT"
echo "---"

exec torchrun --standalone --nproc_per_node="$GPUS" "$TRAIN_GPT"
