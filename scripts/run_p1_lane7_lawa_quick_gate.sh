#!/usr/bin/env bash
# P1 lane_7 LAWA: EMA shadow (default) vs baseline. Uses stronger EMA decay for 20-step quick harness.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_p1_lane7_lawa_quick_gate.sh

Env (optional overrides):
  LAWA_MODE=ema|checkpoint   (default: ema)
  LAWA_EMA_DECAY=0.97      (default below; higher beta needs more steps)
  LAWA_INTERVAL / LAWA_WINDOW for checkpoint mode (e.g. 4 and 5 for 20 steps)

Writes:
  logs/quick_harness/candidate_p1_lane7_lawa.json
  logs/quick_harness/p1_lane7_lawa_quick_gate_<timestamp>.json
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

echo "== P1 lane_7 baseline (LAWA off) =="
unset LAWA_ENABLED LAWA_MODE LAWA_EMA_DECAY LAWA_INTERVAL LAWA_WINDOW
"${HARNESS}" baseline

baseline_json="${LOG_DIR}/baseline.json"
[[ -f "${baseline_json}" ]] || { echo "error: missing baseline.json" >&2; exit 1; }

echo
echo "== P1 lane_7 candidate (LAWA EMA) =="
export LAWA_ENABLED=1
export LAWA_MODE="${LAWA_MODE:-ema}"
export LAWA_EMA_DECAY="${LAWA_EMA_DECAY:-0.97}"
# checkpoint example for 20 steps: LAWA_MODE=checkpoint LAWA_INTERVAL=4 LAWA_WINDOW=5
set +e
"${HARNESS}" candidate
set -e

candidate_json="${LOG_DIR}/candidate.json"
cand_log="${LOG_DIR}/candidate.latest.log"
[[ -f "${candidate_json}" && -f "${cand_log}" ]] || { echo "error: missing candidate artifacts" >&2; exit 1; }

tag="p1_lane7_lawa"
cp -f "${candidate_json}" "${LOG_DIR}/candidate_${tag}.json"
cp -f "${cand_log}" "${LOG_DIR}/candidate_${tag}.latest.log"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary="${LOG_DIR}/p1_lane7_lawa_quick_gate_${timestamp}.json"

python3 - "${baseline_json}" "${LOG_DIR}/candidate_${tag}.json" "${summary}" "${RUNTIME_FACTOR}" "${SEED}" "${ITERATIONS}" "${LAWA_MODE:-ema}" "${LAWA_EMA_DECAY:-0.97}" <<'PY'
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
lawa_mode = sys.argv[7]
lawa_decay = float(sys.argv[8])

b = json.loads(baseline_path.read_text(encoding="utf-8"))
c = json.loads(cand_path.read_text(encoding="utf-8"))
ceiling = float(b["train_time_ms"]) * runtime_factor
bpb_ok = float(c["val_bpb"]) < float(b["val_bpb"])
rt_ok = float(c["train_time_ms"]) <= ceiling
passed = bpb_ok and rt_ok

summary = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mode": "p1_lane7_lawa_quick_gate",
    "runtime_factor": runtime_factor,
    "seed": seed,
    "iterations": iterations,
    "lawa_mode": lawa_mode,
    "lawa_ema_decay": lawa_decay,
    "baseline": {
        "snapshot_path": str(baseline_path),
        "val_bpb": float(b["val_bpb"]),
        "train_time_ms": float(b["train_time_ms"]),
    },
    "candidate": {
        "snapshot_path": str(cand_path),
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
