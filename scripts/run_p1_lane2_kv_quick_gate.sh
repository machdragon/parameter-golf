#!/usr/bin/env bash
# P1 lane_2 minimal runnable slice: NUM_KV_HEADS=2 vs default (4), fixed NUM_HEADS=8.
# Full matrix "recurrence+MQA" is not available until recurrence loops exist in train_gpt.py.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_p1_lane2_kv_quick_gate.sh

Env: same as quick_harness (DATA_PATH, TOKENIZER_PATH, SEED, ITERATIONS, SKIP_*).
Writes:
  logs/quick_harness/candidate_p1_lane2_kv2.json (+ .latest.log)
  logs/quick_harness/p1_lane2_kv_quick_gate_<timestamp>.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/quick_harness"
HARNESS="${ROOT_DIR}/scripts/quick_harness.sh"
REPORT="${ROOT_DIR}/tools/quick_harness_report.py"
mkdir -p "${LOG_DIR}"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024_1train}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"
export ITERATIONS="${ITERATIONS:-20}"
export RUNTIME_FACTOR="${RUNTIME_FACTOR:-1.10}"
export SKIP_TTT_EVAL=1
export SKIP_POST_TRAIN_EVAL="${SKIP_POST_TRAIN_EVAL:-0}"

cd "${ROOT_DIR}"

if [[ ! -x "${HARNESS}" ]]; then
  echo "error: missing ${HARNESS}" >&2
  exit 1
fi

echo "== P1 lane_2 KV ablation: baseline NUM_KV_HEADS=4 (default) =="
unset NUM_KV_HEADS
"${HARNESS}" baseline

baseline_json="${LOG_DIR}/baseline.json"
[[ -f "${baseline_json}" ]] || { echo "error: missing baseline.json" >&2; exit 1; }

echo
echo "== P1 lane_2 candidate: NUM_KV_HEADS=2 (MQA-style), NUM_HEADS=8 =="
export NUM_KV_HEADS=2
export NUM_HEADS=8
"${HARNESS}" candidate

candidate_json="${LOG_DIR}/candidate.json"
cand_log="${LOG_DIR}/candidate.latest.log"
[[ -f "${candidate_json}" && -f "${cand_log}" ]] || { echo "error: missing candidate artifacts" >&2; exit 1; }

tag="p1_lane2_kv2"
cp -f "${candidate_json}" "${LOG_DIR}/candidate_${tag}.json"
cp -f "${cand_log}" "${LOG_DIR}/candidate_${tag}.latest.log"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary="${LOG_DIR}/p1_lane2_kv_quick_gate_${timestamp}.json"

python3 - "${baseline_json}" "${LOG_DIR}/candidate_${tag}.json" "${summary}" "${RUNTIME_FACTOR}" "${SEED}" "${ITERATIONS}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

baseline_path = pathlib.Path(sys.argv[1]).resolve()
cand_path = pathlib.Path(sys.argv[2]).resolve()
out_path = pathlib.Path(sys.argv[3]).resolve()
runtime_factor = float(sys.argv[4])
seed = int(sys.argv[5])
iterations = int(sys.argv[6])

b = json.loads(baseline_path.read_text(encoding="utf-8"))
c = json.loads(cand_path.read_text(encoding="utf-8"))
ceiling = float(b["train_time_ms"]) * runtime_factor
bpb_ok = float(c["val_bpb"]) < float(b["val_bpb"])
rt_ok = float(c["train_time_ms"]) <= ceiling
passed = bpb_ok and rt_ok

summary = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": "p1_lane2_kv_quick_gate",
    "note": "NUM_KV_HEADS=2 vs default 4; recurrence not in trainer",
    "runtime_factor": runtime_factor,
    "seed": seed,
    "iterations": iterations,
    "baseline": {
        "snapshot_path": str(baseline_path),
        "num_kv_heads_config": "default_4",
        "val_bpb": float(b["val_bpb"]),
        "train_time_ms": float(b["train_time_ms"]),
    },
    "candidate": {
        "snapshot_path": str(cand_path),
        "num_kv_heads": 2,
        "num_heads": 8,
        "val_bpb": float(c["val_bpb"]),
        "train_time_ms": float(c["train_time_ms"]),
        "delta_val_bpb": float(c["val_bpb"]) - float(b["val_bpb"]),
        "delta_train_time_ms": float(c["train_time_ms"]) - float(b["train_time_ms"]),
        "bpb_improved": bpb_ok,
        "runtime_ok": rt_ok,
        "runtime_ceiling_ms": ceiling,
        "quick_gate_pass": passed,
    },
}
out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"wrote {out_path}")
PY

python3 "${REPORT}" compare --baseline "${baseline_json}" --candidate "${LOG_DIR}/candidate_${tag}.json" --runtime-factor "${RUNTIME_FACTOR}" || true

echo
echo "summary: ${summary}"
