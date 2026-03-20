#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_open_pr_matrix.sh [--phase0] [--phase1] [--all] [--dry-run] [--fetch-open-scripts]
                               [--run-root <dir>] [--world-size <n>]

Defaults:
  - runs both phase0 and phase1 when no phase flag is passed
  - WORLD_SIZE defaults to 8 (override with --world-size or WORLD_SIZE_OVERRIDE)
  - RUN_ROOT defaults to runs/open_pr_matrix_<timestamp>

Options:
  --phase0               Run merged baseline matrix only
  --phase1               Run open-PR replay matrix only
  --all                  Run both phase0 and phase1
  --fetch-open-scripts   Download open PR train scripts into research/open_pr_replays
  --run-root <dir>       Output root for run folders
  --world-size <n>       torchrun nproc_per_node value
  --dry-run              Print generated commands only
  -h, --help             Show this message

Environment passthrough:
  DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MAX_WALLCLOCK_SECONDS
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%dT%H%M%S)"
run_root_default="${repo_root}/runs/open_pr_matrix_${timestamp}"
run_root="${run_root_default}"
world_size="${WORLD_SIZE_OVERRIDE:-8}"
torchrun_bin="${TORCHRUN_BIN:-}"
dry_run=0
fetch_open_scripts=0
phase0=0
phase1=0
phase_flag_seen=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase0)
      phase0=1
      phase_flag_seen=1
      ;;
    --phase1)
      phase1=1
      phase_flag_seen=1
      ;;
    --all)
      phase0=1
      phase1=1
      phase_flag_seen=1
      ;;
    --fetch-open-scripts)
      fetch_open_scripts=1
      ;;
    --run-root)
      shift
      [[ $# -gt 0 ]] || { echo "error: --run-root requires a value" >&2; exit 1; }
      run_root="$1"
      ;;
    --world-size)
      shift
      [[ $# -gt 0 ]] || { echo "error: --world-size requires a value" >&2; exit 1; }
      world_size="$1"
      ;;
    --dry-run)
      dry_run=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ "$phase_flag_seen" -eq 0 ]]; then
  phase0=1
  phase1=1
fi

if ! [[ "$world_size" =~ ^[0-9]+$ ]] || [[ "$world_size" -lt 1 ]]; then
  echo "error: world size must be a positive integer, got: ${world_size}" >&2
  exit 1
fi

if [[ -z "${torchrun_bin}" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    torchrun_bin="$(command -v torchrun)"
  elif [[ -x "${repo_root}/.venv/bin/torchrun" ]]; then
    torchrun_bin="${repo_root}/.venv/bin/torchrun"
  fi
fi

if [[ -z "${torchrun_bin}" || ! -x "${torchrun_bin}" ]]; then
  echo "error: torchrun not found; set TORCHRUN_BIN or install torchrun (repo .venv is auto-detected)." >&2
  exit 1
fi

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

# Normalize to absolute paths so runs work from per-run working directories.
if [[ "${DATA_PATH}" = /* ]]; then
  data_path_abs="${DATA_PATH}"
else
  data_path_abs="${repo_root}/${DATA_PATH#./}"
fi
if [[ "${TOKENIZER_PATH}" = /* ]]; then
  tokenizer_path_abs="${TOKENIZER_PATH}"
else
  tokenizer_path_abs="${repo_root}/${TOKENIZER_PATH#./}"
fi

if [[ ! -f "${tokenizer_path_abs}" ]]; then
  echo "error: TOKENIZER_PATH not found: ${tokenizer_path_abs}" >&2
  exit 1
fi
if ! compgen -G "${data_path_abs}/fineweb_train_*.bin" > /dev/null; then
  echo "error: DATA_PATH has no fineweb_train_*.bin shards: ${data_path_abs}" >&2
  exit 1
fi
if ! compgen -G "${data_path_abs}/fineweb_val_*.bin" > /dev/null; then
  echo "error: DATA_PATH has no fineweb_val_*.bin shards: ${data_path_abs}" >&2
  exit 1
fi

export DATA_PATH="${data_path_abs}"
export TOKENIZER_PATH="${tokenizer_path_abs}"

open_replays_dir="${repo_root}/research/open_pr_replays"
mkdir -p "${open_replays_dir}"

fetch_open_script() {
  local out="$1"
  local url="$2"
  echo "fetch: ${url}"
  curl -L -sS -o "${out}" "${url}"
  chmod +x "${out}" || true
}

if [[ "${fetch_open_scripts}" -eq 1 ]]; then
  fetch_open_script "${open_replays_dir}/pr150_train_gpt.py" "https://github.com/openai/parameter-golf/raw/5930991ebde94544128e3f27e50f13f91b015969/records%2Ftrack_10min_16mb%2F2026-03-20_Int6QAT_BigramHash_MLP1344%2Ftrain_gpt.py"
  fetch_open_script "${open_replays_dir}/pr156_train_gpt.py" "https://github.com/openai/parameter-golf/raw/effbd870163a0c5f02bc13f16cf875b698bb81ce/records%2Ftrack_10min_16mb%2F2026-03-20_Int6STE_NorMuon_SWA_SlidingWindow%2Ftrain_gpt.py"
  fetch_open_script "${open_replays_dir}/pr160_train_gpt.py" "https://github.com/openai/parameter-golf/raw/9c4146af4b677f92bffa83282b4d3584a91b2d3b/records%2Ftrack_10min_16mb%2F2026-03-19_ChaseNorton%2Ftrain_gpt.py"
  fetch_open_script "${open_replays_dir}/pr162_train_gpt.py" "https://github.com/openai/parameter-golf/raw/14cdf6f7a4c84d5016dbd6adc6df6d5208b83261/records%2Ftrack_10min_16mb%2F2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA%2Ftrain_gpt.py"
  fetch_open_script "${open_replays_dir}/pr164_train_gpt.py" "https://github.com/openai/parameter-golf/raw/7c44af135b0b314f78758460c3132cadcd480ddd/records%2Ftrack_10min_16mb%2F2026-03-20_OrthoInit_Muon_Int6_MLP3x_SmearBigram_1.1524%2Ftrain_gpt.py"
fi

declare -a phase0_runs=(
  "mx_p0_pr60_base_s1337|phase0|merged|PR60|records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py|SEED=1337"
  "mx_p0_pr60_base_s42|phase0|merged|PR60|records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py|SEED=42"
  "mx_p0_pr60_base_s7|phase0|merged|PR60|records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py|SEED=7"
  "mx_p0_pr52_ctrl_s1337|phase0|merged|PR52|records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py|SEED=1337"
  "mx_p0_pr52_ctrl_s42|phase0|merged|PR52|records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py|SEED=42"
)

declare -a phase1_runs=(
  "mx_p1_open150_qat_on_s1337|phase1|open|PR150|research/open_pr_replays/pr150_train_gpt.py|SEED=1337 USE_INT6=1 USE_ZSTD=1 ZSTD_LEVEL=22 EVAL_STRIDE=64"
  "mx_p1_open150_qat_off_s1337|phase1|open|PR150|research/open_pr_replays/pr150_train_gpt.py|SEED=1337 USE_INT6=0 USE_ZSTD=1 ZSTD_LEVEL=22 EVAL_STRIDE=64"
  "mx_p1_open156_normuon_s1337|phase1|open|PR156|research/open_pr_replays/pr156_train_gpt.py|SEED=1337 SWA_ENABLED=0 EVAL_STRIDE=64"
  "mx_p1_open156_normuon_swa_s1337|phase1|open|PR156|research/open_pr_replays/pr156_train_gpt.py|SEED=1337 SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=200 EVAL_STRIDE=64"
  "mx_p1_open160_qgv3_lzma_s1337|phase1|open|PR160|research/open_pr_replays/pr160_train_gpt.py|SEED=1337 COMPRESSOR=lzma QUANT_BITS=6 TOK_EMBED_BITS=8 FP16_EMBED=0 EVAL_STRIDE=256"
  "mx_p1_open160_zlib_ref_s1337|phase1|open|PR160|research/open_pr_replays/pr160_train_gpt.py|SEED=1337 COMPRESSOR=zlib QUANT_BITS=6 TOK_EMBED_BITS=8 FP16_EMBED=0 EVAL_STRIDE=256"
  "mx_p1_open162_bigram_smear_s1337|phase1|open|PR162|research/open_pr_replays/pr162_train_gpt.py|SEED=1337 BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 SWA_ENABLED=0 EVAL_STRIDE=64"
  "mx_p1_open162_bigram_off_s1337|phase1|open|PR162|research/open_pr_replays/pr162_train_gpt.py|SEED=1337 BIGRAM_VOCAB_SIZE=0 SWA_ENABLED=0 EVAL_STRIDE=64"
  "mx_p1_open164_v2_s1337|phase1|open|PR164|research/open_pr_replays/pr164_train_gpt.py|SEED=1337 EVAL_STRIDE=256 BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128"
  "mx_p1_open164_v2_nobigram_s1337|phase1|open|PR164|research/open_pr_replays/pr164_train_gpt.py|SEED=1337 EVAL_STRIDE=256 BIGRAM_VOCAB_SIZE=0"
)

declare -a run_specs=()
if [[ "${phase0}" -eq 1 ]]; then
  run_specs+=("${phase0_runs[@]}")
fi
if [[ "${phase1}" -eq 1 ]]; then
  run_specs+=("${phase1_runs[@]}")
fi

mkdir -p "${run_root}"
manifest_path="${run_root}/manifest.tsv"
{
  echo -e "run_id\tphase\ttrack\tsource_pr\tscript_rel\toverrides"
  for spec in "${run_specs[@]}"; do
    IFS='|' read -r run_id phase_name track source_pr script_rel overrides <<< "${spec}"
    echo -e "${run_id}\t${phase_name}\t${track}\t${source_pr}\t${script_rel}\t${overrides}"
  done
} > "${manifest_path}"

echo "open_pr_matrix config:"
echo "  repo_root=${repo_root}"
echo "  run_root=${run_root}"
echo "  phase0=${phase0} phase1=${phase1}"
echo "  world_size=${world_size}"
echo "  dry_run=${dry_run}"
echo "  fetch_open_scripts=${fetch_open_scripts}"
echo "  torchrun_bin=${torchrun_bin}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "  manifest=${manifest_path}"

run_one() {
  local spec="$1"
  IFS='|' read -r run_id phase_name track source_pr script_rel overrides <<< "${spec}"
  local script_path="${repo_root}/${script_rel}"
  local run_dir="${run_root}/${run_id}"

  if [[ ! -f "${script_path}" ]]; then
    echo "error: missing train script for ${run_id}: ${script_path}" >&2
    echo "hint: use --fetch-open-scripts for phase1 runs" >&2
    exit 1
  fi

  mkdir -p "${run_dir}"
  local -a base_pairs=(
    "RUN_ID=${run_id}"
    "DATA_PATH=${DATA_PATH}"
    "TOKENIZER_PATH=${TOKENIZER_PATH}"
    "VOCAB_SIZE=${VOCAB_SIZE}"
    "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
    "SOURCE_PR=${source_pr}"
    "MATRIX_PHASE=${phase_name}"
    "MATRIX_TRACK=${track}"
  )
  local -a extra_pairs=()
  if [[ -n "${overrides}" ]]; then
    # shellcheck disable=SC2206
    extra_pairs=(${overrides})
  fi

  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${run_dir}"
    printf 'env '
    printf '%q ' "${base_pairs[@]}"
    if [[ "${#extra_pairs[@]}" -gt 0 ]]; then
      printf '%q ' "${extra_pairs[@]}"
    fi
    printf '%q --standalone --nproc_per_node=%q %q\n' "${torchrun_bin}" "${world_size}" "${script_path}"
  } > "${run_dir}/command.sh"
  chmod +x "${run_dir}/command.sh"

  {
    printf '%s\n' "${base_pairs[@]}"
    printf 'WORLD_SIZE=%s\n' "${world_size}"
    printf 'TRAIN_SCRIPT=%s\n' "${script_path}"
    printf 'TRAIN_SCRIPT_REL=%s\n' "${script_rel}"
    if [[ "${#extra_pairs[@]}" -gt 0 ]]; then
      printf '%s\n' "${extra_pairs[@]}"
    fi
  } | sort > "${run_dir}/env.txt"

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "[dry-run] ${run_id}"
    cat "${run_dir}/command.sh"
    return
  fi

  echo
  echo "== ${run_id} (${phase_name}, ${source_pr}) =="
  (
    cd "${run_dir}"
    ./command.sh 2>&1 | tee train.log
  )
}

for spec in "${run_specs[@]}"; do
  run_one "${spec}"
done

echo
echo "done: run_open_pr_matrix"
echo "artifacts root: ${run_root}"
echo "manifest: ${manifest_path}"
