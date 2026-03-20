#!/usr/bin/env bash
# Phase 3: collect final_int8_zlib_roundtrip_exact + int8+zlib bytes vs 16 MiB from harness logs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/quick_harness"
REPORT="${ROOT_DIR}/tools/quick_harness_report.py"
OUT="${LANE5_PROMOTION_OUT:-${LOG_DIR}/lane5_promotion_metrics.json}"

if [[ ! -f "${REPORT}" ]]; then
  echo "error: missing ${REPORT}" >&2
  exit 1
fi

args=(python3 "${REPORT}" promotion --out "${OUT}" --limit-bytes $((16 * 1024 * 1024)))
args+=(--entry "baseline=${LOG_DIR}/baseline.latest.log")

shopt -s nullglob
cand=( "${LOG_DIR}"/candidate_*.latest.log )
shopt -u nullglob
if [[ ${#cand[@]} -eq 0 ]]; then
  echo "error: no candidate_*.latest.log under ${LOG_DIR}" >&2
  exit 1
fi
for f in "${cand[@]}"; do
  bn=$(basename "${f}" .latest.log)
  args+=(--entry "${bn}=${f}")
done

"${args[@]}"
echo "lane5 promotion metrics: ${OUT}"
