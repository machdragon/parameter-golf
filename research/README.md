# Local research (hypothesis sweeps)

This tree is **not** part of the official [Parameter Golf](https://github.com/openai/parameter-golf) submission layout. Keep it on a **fork branch** or local clone; do not open a PR to `openai/parameter-golf` that only adds personal research scaffolding unless you intend to share it.

## Contents

| Path | Purpose |
|------|---------|
| [`../arc_inputs/case_matrix_v3_full.json`](../arc_inputs/case_matrix_v3_full.json) | Full hypothesis grid (canonical); see also [`arc_inputs/README.md`](../arc_inputs/README.md) |
| [`lane_env_presets.json`](lane_env_presets.json) | **Env-only** presets mappable to upstream `train_gpt.py` |
| [`scripts/quick_rank_lanes.py`](scripts/quick_rank_lanes.py) | Run baseline + presets → JSON/CSV ranking |
| [`../scripts/quick_harness.sh`](../scripts/quick_harness.sh) | Same quick gate as **parameter-golf-old**: baseline/candidate + JSON snapshots |
| [`../scripts/run_lane5_quick_gate.sh`](../scripts/run_lane5_quick_gate.sh) | Lane_5 quick-gate runner: baseline + `MDL_only` + `sparsity_only` + `full` |
| [`WINNING_PATTERNS.md`](WINNING_PATTERNS.md) | Themes from top `records/**/README.md` in this repo |
| [`autoresearch_sidecar.md`](autoresearch_sidecar.md) | Using [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for ideas only |
| [`PR_CHECKLIST.md`](PR_CHECKLIST.md) | Checklist for a real leaderboard PR |
| [`recorded_runs/`](recorded_runs/) | Committed **quick_harness** baseline snapshots (`quick_metric` metrics) |

## Quick start

1. Prepare data + tokenizer per upstream README (`data/cached_challenge_fineweb.py`, etc.).
2. Activate your venv; ensure `torchrun` is on `PATH`.
3. **Optional — old quick gate** (same defaults as `parameter-golf-old`, `USE_COMPILE=0`):

```bash
cd /path/to/parameter-golf
chmod +x scripts/quick_harness.sh   # if you see "Permission denied"
./scripts/quick_harness.sh baseline
# or without execute bit:  bash scripts/quick_harness.sh baseline
```

4. Or run a preset sweep:

```bash
cd /path/to/parameter-golf
export DATA_PATH=./data/datasets/fineweb10B_sp1024_1train
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model

python3 research/scripts/quick_rank_lanes.py \
  baseline lane_2_kv_heads_2 lower_lr_like_sota
```

Results go to `research/sweeps/output/` (gitignored).

5. Lane_5 quick gate (baseline + minimal ablations):

```bash
cd /path/to/parameter-golf
export DATA_PATH=./data/datasets/fineweb10B_sp1024_1train
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
./scripts/run_lane5_quick_gate.sh
```

This writes per-ablation snapshots/logs and a summary JSON under `logs/quick_harness/`.

## Caveats

- **Upstream `train_gpt.py`** uses full validation at the end of the training loop; “quick” = fewer **training** steps + wallclock cap, not a smaller val split unless you patch `train_gpt.py` yourself.
- Many **case-matrix lanes** need **code changes** (recurrence, custom quant, etc.). Those are listed in `lane_env_presets.json` under `lanes_not_env_only`; the sweep only runs **env-only** presets unless you extend the JSON.
