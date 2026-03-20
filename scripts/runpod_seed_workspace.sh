#!/usr/bin/env bash
# Populate /workspace on a Pod that uses a RunPod **network volume** (persistent).
# Safe to re-run: pulls repo if already cloned; re-downloads only what cached script needs.
#
# Run from anywhere on the pod (e.g. fresh terminal on first boot):
#   curl -fsSL "https://raw.githubusercontent.com/machdragon/parameter-golf/main/scripts/runpod_seed_workspace.sh" | bash
#
# Or after a manual clone:
#   cd /workspace/parameter-golf && ./scripts/runpod_seed_workspace.sh
#
# Smaller dataset for quick tests:
#   TRAIN_SHARDS=1 curl -fsSL "https://raw.githubusercontent.com/machdragon/parameter-golf/main/scripts/runpod_seed_workspace.sh" | bash
#
# Docs: https://docs.runpod.io/storage/network-volumes
set -euo pipefail

WORKSPACE="${RUNPOD_WORKSPACE:-/workspace}"
REPO_URL="${GIT_REPO_URL:-https://github.com/machdragon/parameter-golf.git}"
REPO_DIR_NAME="${GIT_REPO_NAME:-parameter-golf}"
TRAIN_SHARDS="${TRAIN_SHARDS:-}"

resolve_repo_root() {
  if [[ -f "$(pwd)/data/cached_challenge_fineweb.py" ]]; then
    pwd
    return
  fi
  if [[ -f "$WORKSPACE/$REPO_DIR_NAME/data/cached_challenge_fineweb.py" ]]; then
    echo "$WORKSPACE/$REPO_DIR_NAME"
    return
  fi
  echo ""
}

REPO_ROOT="$(resolve_repo_root || true)"
if [[ -z "$REPO_ROOT" ]]; then
  mkdir -p "$WORKSPACE"
  cd "$WORKSPACE"
  if [[ ! -d "$REPO_DIR_NAME/.git" ]]; then
    echo "Cloning $REPO_URL -> $WORKSPACE/$REPO_DIR_NAME"
    git clone "$REPO_URL" "$REPO_DIR_NAME"
  fi
  REPO_ROOT="$WORKSPACE/$REPO_DIR_NAME"
fi

cd "$REPO_ROOT"
echo "Using repo: $REPO_ROOT"
git pull --ff-only || true

echo "Downloading FineWeb (sp1024)..."
if [[ -n "$TRAIN_SHARDS" ]]; then
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
else
  python3 data/cached_challenge_fineweb.py --variant sp1024
fi

echo ""
echo "Seed complete. Typical next steps:"
echo "  cd $REPO_ROOT"
echo "  ./scripts/runpod_preflight.sh 8"
echo "  ./scripts/runpod_train.sh 8x --run-id prod_8xh100_\$(date -u +%Y%m%d_%H%MZ)"
