# Local research (hypothesis sweeps)

This tree is **not** part of the official [Parameter Golf](https://github.com/openai/parameter-golf) submission layout. Keep it on a **fork branch** or local clone; do not open a PR to `openai/parameter-golf` that only adds personal research scaffolding unless you intend to share it.

## Contents

| Path | Purpose |
|------|---------|
| [`local/case_matrix_v3_full.json`](local/case_matrix_v3_full.json) | Hypothesis lanes (from your prior ARC matrix); reference only |
| [`lane_env_presets.json`](lane_env_presets.json) | **Env-only** presets mappable to upstream `train_gpt.py` |
| [`scripts/quick_rank_lanes.py`](scripts/quick_rank_lanes.py) | Run baseline + presets → JSON/CSV ranking |
| [`WINNING_PATTERNS.md`](WINNING_PATTERNS.md) | Themes from top `records/**/README.md` in this repo |
| [`autoresearch_sidecar.md`](autoresearch_sidecar.md) | Using [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for ideas only |
| [`PR_CHECKLIST.md`](PR_CHECKLIST.md) | Checklist for a real leaderboard PR |

## Quick start

1. Prepare data + tokenizer per upstream README (`data/cached_challenge_fineweb.py`, etc.).
2. Activate your venv; ensure `torchrun` is on `PATH`.
3. Run a ranking sweep (example):

```bash
cd /path/to/parameter-golf-fresh
export DATA_PATH=./data/datasets/fineweb10B_sp1024_1train
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model

python3 research/scripts/quick_rank_lanes.py \
  --presets baseline lane_2_kv_heads_2 lower_lr_like_sota
```

Results go to `research/sweeps/output/` (gitignored).

## Caveats

- **Upstream `train_gpt.py`** uses full validation at the end of the training loop; “quick” = fewer **training** steps + wallclock cap, not a smaller val split unless you patch `train_gpt.py` yourself.
- Many **case-matrix lanes** need **code changes** (recurrence, custom quant, etc.). Those are listed in `lane_env_presets.json` under `lanes_not_env_only`; the sweep only runs **env-only** presets unless you extend the JSON.
