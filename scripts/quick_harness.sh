#!/usr/bin/env bash
# Quick baseline/candidate runs (ported from parameter-golf-old).
# Requires train_gpt.py support for USE_COMPILE, SDP_*, and quick_metric log line.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <baseline|candidate> [-- <extra train_gpt.py args>]" >&2
  exit 1
fi

profile="$1"
shift || true

if [[ "${1:-}" == "--" ]]; then
  shift
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/quick_harness"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
case "${profile}" in
  baseline)
    default_run_id="quick_baseline"
    ;;
  candidate)
    default_run_id="quick_candidate"
    ;;
  *)
    echo "unknown profile: ${profile} (expected baseline or candidate)" >&2
    exit 1
    ;;
esac

log_path="${LOG_DIR}/${profile}_${timestamp}.log"
latest_log_path="${LOG_DIR}/${profile}.latest.log"

export RUN_ID="${RUN_ID:-${default_run_id}}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024_1train}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Quick harness defaults (same as parameter-golf-old; override via env).
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export ITERATIONS="${ITERATIONS:-20}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export USE_COMPILE="${USE_COMPILE:-0}"
export SDP_CUDNN="${SDP_CUDNN:-0}"
export SDP_FLASH="${SDP_FLASH:-1}"
export SDP_MEM_EFFICIENT="${SDP_MEM_EFFICIENT:-0}"
export SDP_MATH="${SDP_MATH:-0}"

# Post-train eval: TTT LoRA dominates wall time on short runs; quick gate uses `quick_metric` only.
export SKIP_TTT_EVAL="${SKIP_TTT_EVAL:-1}"
# Set SKIP_POST_TRAIN_EVAL=1 for fastest smoke (skips int8 roundtrip val too — not for lane_5 promotion).
export SKIP_POST_TRAIN_EVAL="${SKIP_POST_TRAIN_EVAL:-0}"

echo "profile=${profile}"
echo "run_id=${RUN_ID}"
echo "log=${log_path}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "quick_defaults: WARMUP_STEPS=${WARMUP_STEPS} ITERATIONS=${ITERATIONS} VAL_LOSS_EVERY=${VAL_LOSS_EVERY}"
echo "quick_backend_defaults: USE_COMPILE=${USE_COMPILE} SDP_CUDNN=${SDP_CUDNN} SDP_FLASH=${SDP_FLASH} SDP_MEM_EFFICIENT=${SDP_MEM_EFFICIENT} SDP_MATH=${SDP_MATH}"
echo "eval_tail: SKIP_TTT_EVAL=${SKIP_TTT_EVAL} SKIP_POST_TRAIN_EVAL=${SKIP_POST_TRAIN_EVAL}"

cd "${ROOT_DIR}"

if command -v torchrun >/dev/null 2>&1; then
  launcher=(torchrun)
elif [[ -x "${ROOT_DIR}/.venv/bin/torchrun" ]]; then
  launcher=("${ROOT_DIR}/.venv/bin/torchrun")
else
  launcher=(python3 -m torch.distributed.run)
fi

"${launcher[@]}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py "$@" | tee "${log_path}"
cp -f "${log_path}" "${latest_log_path}"

python3 tools/quick_harness_report.py snapshot \
  --profile "${profile}" \
  --log "${log_path}" \
  --out-dir "${LOG_DIR}"

if [[ "${profile}" == "candidate" ]]; then
  baseline_json="${LOG_DIR}/baseline.json"
  candidate_json="${LOG_DIR}/candidate.json"
  if [[ ! -f "${baseline_json}" ]]; then
    echo "missing baseline snapshot: ${baseline_json}. Run baseline first." >&2
    exit 1
  fi
  python3 tools/quick_harness_report.py compare \
    --baseline "${baseline_json}" \
    --candidate "${candidate_json}" \
    --runtime-factor 1.10
fi
