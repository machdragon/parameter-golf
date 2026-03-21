# XSA + EMA 0.997 + Periodic KURE (No Tanh)

Based on PR #287 (11L XSA+EMA baseline). This variant focuses on periodic
KURE regularization (`KURE_EVERY`) to recover wallclock throughput while
preserving entropy reduction.

## Base (PR #287)

11L, XSA on last 4 layers, EMA decay=0.997, MLP3x relu², FA3, SmearGate,
BigramHash, OrthoInit, U-Net skips, WD=0.04, GQA, int6+zstd, sliding eval.

## Changes from base

1. **Periodic KURE regularization** via `KURE_EVERY`.
2. **Explicit gated KURE condition**:
   `(kure_lambda > 0 or r2_lambda > 0) and step % kure_every == 0`.
3. **Optional R2 margin regularization** (`R2_LAMBDA`, default `0.0`).
4. **No tanh reparameterization by default** (`TANH_REPARAM=0`) for speed.
5. **Overtone removed** from embedding init.

## Default Modal run config (Track A)

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
| KURE_EVERY | 8 |
| TANH_REPARAM | 0 |
| ITERATIONS | 9000 |
| MAX_WALLCLOCK_SECONDS | 600 |
| EVAL_STRIDE | 64 |

## Run plan

| Run | Config delta | Goal |
|-----|-------------|------|
| A | `KURE_LAMBDA=0.001`, `KURE_EVERY=8`, `TANH_REPARAM=0` | Validate speed/size recovery and quality |
| B | `KURE_LAMBDA=0.005`, `KURE_EVERY=8`, `TANH_REPARAM=0` | Test stronger entropy push for 12L-candidate regime |
