# XSA + EMA 0.997 + KURE 0.001 + Tanh Reparam

Based on PR #287 (11L XSA+EMA baseline). This variant removes Overtone and adds a
lightweight entropy-shaping path aimed at preserving quality while creating artifact
headroom for a future 12L run.

## Base (PR #287)

11L, XSA on last 4 layers, EMA decay=0.997, MLP3x relu², FA3, SmearGate,
BigramHash, OrthoInit, U-Net skips, WD=0.04, GQA, int6+zstd, sliding eval.

## Changes from base

1. **Light KURE regularization** (`KURE_LAMBDA=0.001` default) to nudge weight
   distributions toward lower entropy without heavy quality regression.
2. **Optional R2 margin regularization** (`R2_LAMBDA`, default `0.0`) kept as a dial,
   disabled in the default run.
3. **Tanh reparameterization** (`TANH_REPARAM=1` default) on large `CastedLinear`
   matrices during training.
4. **Export materialization**: before serialization, tanh(P) is materialized into
   linear weights so quantization/eval sees effective weights directly.
5. **Overtone removed** from embedding init to avoid extra compression entropy.

## Default Modal run config

| Variable | Value |
|----------|-------|
| NUM_LAYERS | 11 |
| BIGRAM_VOCAB_SIZE | 2048 |
| XSA_LAST_N | 4 |
| EMA_ENABLED | 1 |
| EMA_DECAY | 0.997 |
| SWA_ENABLED | 0 |
| KURE_LAMBDA | 0.001 |
| R2_LAMBDA | 0.0 |
| TANH_REPARAM | 1 |
| ITERATIONS | 9000 |
| MAX_WALLCLOCK_SECONDS | 600 |
| EVAL_STRIDE | 64 |

## Run plan

| Run | Config delta | Goal |
|-----|-------------|------|
| A | default config above | Find quality/size point at KURE 0.001 |
| B | `NUM_LAYERS=12` | Check if 12L now fits under 16MB |
| C | `KURE_LAMBDA=0.003` | More entropy reduction if B is still near cap |
| D | best config, 2-3 seeds | Confirmation before submission |
