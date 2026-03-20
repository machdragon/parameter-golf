#!/usr/bin/env bash
set -euo pipefail

VOLUME_NAME="${1:-parameter-golf-data}"

LOCAL_DATASET_DIR="./data/datasets/fineweb10B_sp1024"
LOCAL_TOKENIZER_FILE="./data/tokenizers/fineweb_1024_bpe.model"

if [[ ! -d "${LOCAL_DATASET_DIR}" ]]; then
  echo "Missing dataset directory: ${LOCAL_DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LOCAL_TOKENIZER_FILE}" ]]; then
  echo "Missing tokenizer file: ${LOCAL_TOKENIZER_FILE}" >&2
  exit 1
fi

echo "Ensuring Modal volume exists: ${VOLUME_NAME}"
.venv-modal/bin/modal volume create "${VOLUME_NAME}" 2>/dev/null || true

# Dataset: re-run without --force fails with "already exists" if a previous sync completed.
# We treat that as OK so tokenizer upload still runs. Use MODAL_SYNC_FORCE=1 to overwrite shards.
echo "Uploading dataset directory to ${VOLUME_NAME}:/datasets/fineweb10B_sp1024"
if [[ "${MODAL_SYNC_FORCE:-}" == "1" ]]; then
  echo "(MODAL_SYNC_FORCE=1: overwriting existing shards; may take a long time)"
  .venv-modal/bin/modal volume put --force "${VOLUME_NAME}" "${LOCAL_DATASET_DIR}" "/datasets/"
else
  set +e
  .venv-modal/bin/modal volume put "${VOLUME_NAME}" "${LOCAL_DATASET_DIR}" "/datasets/"
  ds_rc=$?
  set -e
  if [[ "${ds_rc}" -ne 0 ]]; then
    echo ""
    echo "Note: dataset upload exited with code ${ds_rc}."
    echo "  If the message was 'already exists', shards are already on the volume — OK."
    echo "  To force a full re-upload: MODAL_SYNC_FORCE=1 $0 ${VOLUME_NAME}"
    echo ""
  fi
fi

echo "Uploading tokenizer file to ${VOLUME_NAME}:/tokenizers/fineweb_1024_bpe.model"
.venv-modal/bin/modal volume put --force "${VOLUME_NAME}" "${LOCAL_TOKENIZER_FILE}" "/tokenizers/fineweb_1024_bpe.model"

echo "Volume contents:"
.venv-modal/bin/modal volume ls "${VOLUME_NAME}" /datasets/fineweb10B_sp1024
.venv-modal/bin/modal volume ls "${VOLUME_NAME}" /tokenizers

echo
echo "Modal paths now available:"
echo "  DATA_PATH=/vol/datasets/fineweb10B_sp1024"
echo "  TOKENIZER_PATH=/vol/tokenizers/fineweb_1024_bpe.model"
