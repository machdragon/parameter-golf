#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_lane5_quick_gate.sh

Optional environment variables:
  SEED=1337
  ITERATIONS=20
  RUNTIME_FACTOR=1.10
  VAL_BATCH_SIZE=<pass-through to train_gpt.py validation batch>
  DATA_PATH=./data/datasets/fineweb10B_sp1024_1train
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model

Behavior:
  - Runs quick_harness baseline once
  - Runs 3 lane_5 candidates: MDL_only, sparsity_only, full
  - Stores non-overwriting candidate snapshots/logs
  - Compares each candidate vs baseline
  - Writes lane5 quick-gate summary JSON under logs/quick_harness
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/quick_harness"
HARNESS_SCRIPT="${ROOT_DIR}/scripts/quick_harness.sh"
REPORT_SCRIPT="${ROOT_DIR}/tools/quick_harness_report.py"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${HARNESS_SCRIPT}" ]]; then
  echo "error: missing executable harness script: ${HARNESS_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${REPORT_SCRIPT}" ]]; then
  echo "error: missing report script: ${REPORT_SCRIPT}" >&2
  exit 1
fi

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024_1train}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"
export ITERATIONS="${ITERATIONS:-20}"
export RUNTIME_FACTOR="${RUNTIME_FACTOR:-1.10}"
export SKIP_TTT_EVAL=1
export SKIP_POST_TRAIN_EVAL=0

if [[ ! -f "${ROOT_DIR}/${TOKENIZER_PATH#./}" && ! -f "${TOKENIZER_PATH}" ]]; then
  echo "error: TOKENIZER_PATH not found: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if ! compgen -G "${ROOT_DIR}/${DATA_PATH#./}/fineweb_train_*.bin" > /dev/null && ! compgen -G "${DATA_PATH}/fineweb_train_*.bin" > /dev/null; then
  echo "error: DATA_PATH has no fineweb_train_*.bin shards: ${DATA_PATH}" >&2
  exit 1
fi
if ! compgen -G "${ROOT_DIR}/${DATA_PATH#./}/fineweb_val_*.bin" > /dev/null && ! compgen -G "${DATA_PATH}/fineweb_val_*.bin" > /dev/null; then
  echo "error: DATA_PATH has no fineweb_val_*.bin shards: ${DATA_PATH}" >&2
  exit 1
fi

echo "lane5_quick_gate config:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "  SEED=${SEED}"
echo "  ITERATIONS=${ITERATIONS}"
echo "  RUNTIME_FACTOR=${RUNTIME_FACTOR}"
echo "  SKIP_TTT_EVAL=${SKIP_TTT_EVAL}"
echo "  SKIP_POST_TRAIN_EVAL=${SKIP_POST_TRAIN_EVAL}"

cd "${ROOT_DIR}"

echo
echo "== baseline =="
unset COMPRESSION_OBJECTIVE_WEIGHT MDL_PENALTY_LAMBDA STRUCTURAL_SPARSITY_LAMBDA
"${HARNESS_SCRIPT}" baseline

baseline_json="${LOG_DIR}/baseline.json"
if [[ ! -f "${baseline_json}" ]]; then
  echo "error: baseline snapshot missing after baseline run: ${baseline_json}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
results_jsonl="$(mktemp)"
trap 'rm -f "${results_jsonl}"' EXIT

declare -a ablations=("MDL_only" "sparsity_only" "full")
for ablation in "${ablations[@]}"; do
  case "${ablation}" in
    MDL_only)
      export COMPRESSION_OBJECTIVE_WEIGHT=0
      export MDL_PENALTY_LAMBDA=1e-6
      export STRUCTURAL_SPARSITY_LAMBDA=0
      ;;
    sparsity_only)
      export COMPRESSION_OBJECTIVE_WEIGHT=0
      export MDL_PENALTY_LAMBDA=0
      export STRUCTURAL_SPARSITY_LAMBDA=1e-6
      ;;
    full)
      export COMPRESSION_OBJECTIVE_WEIGHT=0.01
      export MDL_PENALTY_LAMBDA=1e-6
      export STRUCTURAL_SPARSITY_LAMBDA=1e-6
      ;;
    *)
      echo "error: unknown ablation: ${ablation}" >&2
      exit 1
      ;;
  esac

  echo
  echo "== candidate (${ablation}) =="
  echo "  COMPRESSION_OBJECTIVE_WEIGHT=${COMPRESSION_OBJECTIVE_WEIGHT}"
  echo "  MDL_PENALTY_LAMBDA=${MDL_PENALTY_LAMBDA}"
  echo "  STRUCTURAL_SPARSITY_LAMBDA=${STRUCTURAL_SPARSITY_LAMBDA}"

  set +e
  "${HARNESS_SCRIPT}" candidate
  candidate_rc=$?
  set -e

  candidate_json="${LOG_DIR}/candidate.json"
  candidate_latest_log="${LOG_DIR}/candidate.latest.log"
  if [[ ! -f "${candidate_json}" || ! -f "${candidate_latest_log}" ]]; then
    echo "error: candidate artifacts missing for ${ablation}" >&2
    exit 1
  fi

  candidate_named_json="${LOG_DIR}/candidate_${ablation}.json"
  candidate_named_log="${LOG_DIR}/candidate_${ablation}.latest.log"
  cp -f "${candidate_json}" "${candidate_named_json}"
  cp -f "${candidate_latest_log}" "${candidate_named_log}"

  set +e
  compare_output="$(
    python3 "${REPORT_SCRIPT}" compare \
      --baseline "${baseline_json}" \
      --candidate "${candidate_named_json}" \
      --runtime-factor "${RUNTIME_FACTOR}" 2>&1
  )"
  compare_rc=$?
  set -e
  echo "${compare_output}"

  python3 - "${baseline_json}" "${candidate_named_json}" "${ablation}" "${candidate_rc}" "${compare_rc}" "${candidate_named_log}" "${candidate_named_json}" <<'PY' >> "${results_jsonl}"
import json
import pathlib
import sys

baseline_path = pathlib.Path(sys.argv[1]).resolve()
candidate_path = pathlib.Path(sys.argv[2]).resolve()
ablation = sys.argv[3]
candidate_rc = int(sys.argv[4])
compare_rc = int(sys.argv[5])
log_path = pathlib.Path(sys.argv[6]).resolve()
snapshot_path = pathlib.Path(sys.argv[7]).resolve()

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

bpb_improved = float(candidate["val_bpb"]) < float(baseline["val_bpb"])

record = {
    "ablation": ablation,
    "candidate_rc": candidate_rc,
    "compare_rc": compare_rc,
    "log_path": str(log_path),
    "snapshot_path": str(snapshot_path),
    "val_bpb": float(candidate["val_bpb"]),
    "train_time_ms": float(candidate["train_time_ms"]),
    "delta_vs_baseline": float(candidate["val_bpb"]) - float(baseline["val_bpb"]),
    "bpb_improved": bpb_improved,
    "runtime_ok": None,
    "pass": None,
}

print(json.dumps(record, sort_keys=True))
PY
done

summary_json="${LOG_DIR}/lane5_quick_gate_${timestamp}.json"
python3 - "${baseline_json}" "${results_jsonl}" "${summary_json}" "${RUNTIME_FACTOR}" "${SEED}" "${ITERATIONS}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

baseline_path = pathlib.Path(sys.argv[1]).resolve()
results_path = pathlib.Path(sys.argv[2]).resolve()
summary_path = pathlib.Path(sys.argv[3]).resolve()
runtime_factor = float(sys.argv[4])
seed = int(sys.argv[5])
iterations = int(sys.argv[6])

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
results = []
for line in results_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    runtime_ceiling = float(baseline["train_time_ms"]) * runtime_factor
    rec["runtime_ok"] = float(rec["train_time_ms"]) <= runtime_ceiling
    rec["pass"] = bool(rec["bpb_improved"] and rec["runtime_ok"])
    rec["runtime_ceiling_ms"] = runtime_ceiling
    results.append(rec)

passes = [r for r in results if r["pass"]]
if passes:
    best = min(passes, key=lambda r: float(r["val_bpb"]))
else:
    best = min(results, key=lambda r: float(r["val_bpb"])) if results else None

summary = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": "lane5_quick_gate",
    "runtime_factor": runtime_factor,
    "seed": seed,
    "iterations": iterations,
    "baseline": {
        "snapshot_path": str(baseline_path),
        "val_bpb": float(baseline["val_bpb"]),
        "train_time_ms": float(baseline["train_time_ms"]),
    },
    "results": results,
    "best_candidate": best,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"wrote {summary_path}")
PY

echo
echo "lane_5 quick gate summary: ${summary_json}"
