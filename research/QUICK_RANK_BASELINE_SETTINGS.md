# Effective settings: `quick_rank_lanes.py` preset **`baseline`**

See also **[`../scripts/quick_harness.sh`](../scripts/quick_harness.sh)** — same **parameter-golf-old** defaults (`USE_COMPILE=0`, `ITERATIONS=20`, etc.) and writes `logs/quick_harness/baseline.json` via `tools/quick_harness_report.py`.

`research/scripts/quick_rank_lanes.py` sets env vars from:

1. **`quick_protocol.env`** in [`lane_env_presets.json`](lane_env_presets.json) (short screening run)
2. **Preset `baseline`** (`env`: `{}` — no overrides)
3. Your shell: **`DATA_PATH`**, **`TOKENIZER_PATH`** (required by the script)

Everything else comes from **`train_gpt.py` `Hyperparameters`** defaults unless you export more env vars.

## Screening overrides (`quick_protocol`)

| Variable | Value | Notes |
|----------|-------|--------|
| `SEED` | `1337` | |
| `ITERATIONS` | `20` | Few train steps |
| `WARMUP_STEPS` | `0` | |
| `VAL_LOSS_EVERY` | `0` | No mid-run val loss logging |
| `TRAIN_LOG_EVERY` | `99999` | Effectively no train step logs |
| `MAX_WALLCLOCK_SECONDS` | `180` | Hard stop ~3 min |
| `USE_COMPILE` | `0` | Matches `scripts/quick_harness.sh` (no `torch.compile`) |
| `SDP_*` | cudnn 0, flash 1, mem_eff 0, math 0 | Same as old quick harness |

## `baseline` preset

No extra env keys — model shape and optimizer LRs are **stock** `Hyperparameters` defaults.

## Stock `Hyperparameters` used for baseline (high level)

| Topic | Default (unless overridden) |
|-------|------------------------------|
| Model | 9 layers, `MODEL_DIM` 512, `NUM_HEADS` 8, `NUM_KV_HEADS` 4, `MLP_MULT` 2, `VOCAB_SIZE` 1024, `TRAIN_SEQ_LEN` 1024 |
| Train step | `TRAIN_BATCH_TOKENS` 524288 (~512k tokens/step) |
| LRs | e.g. `MATRIX_LR` / `SCALAR_LR` 0.04, `EMBED_LR` 0.6, etc. (see `train_gpt.py`) |
| Val | Full val split; `VAL_BATCH_SIZE` 524288 |

**Why it can feel slow:** even with **20 iterations**, the run still does **tokenizer/dataset setup**, **`torch.compile` warmup**, and **full validation + int8/zlib + TTT LoRA** at the end of `train_gpt.py` — that tail work dominates wall time for a “quick” sweep unless you patch `train_gpt.py`.
