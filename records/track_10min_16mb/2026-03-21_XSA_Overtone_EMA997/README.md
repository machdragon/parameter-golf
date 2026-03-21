# XSA + Overtone Init + EMA 0.997

Based on PR #287 (SOTA val_bpb=1.1271). Single change: add Overtone init to
embedding table.

## Base (PR #287)

11L, XSA on last 4 layers, EMA decay=0.997, SWA fallback, MLP3x relu²,
FA3, SmearGate, BigramHash, OrthoInit, U-Net skips, WD=0.04, GQA,
Int6+zstd, sliding window eval.

## Changes from base

1. **Overtone init** — After normal init of `tok_emb.weight`, reshape
   singular value spectrum to power-law decay (S_i ∝ i^{-0.5}). This
   smooths the embedding spectrum for better int6 quantization fidelity.
   Proven in PRs #60, #201, #4. Zero training cost (init-time only, 5 lines).

## Hyperparameters (PR #287 proven config)

| Variable | Value | Notes |
|----------|-------|-------|
| NUM_LAYERS | 11 | (sweep: try 12) |
| BIGRAM_VOCAB_SIZE | 2048 | |
| XSA_LAST_N | 4 | (sweep: try 2, 5, 6) |
| EMA_ENABLED | 1 | |
| EMA_DECAY | 0.997 | (sweep: 0.996, 0.998) |
| SWA_ENABLED | 0 | |
| MATRIX_LR | 0.025 | |
| SCALAR_LR | 0.025 | |
| TIED_EMBED_LR | 0.035 | |
| MUON_MOMENTUM | 0.99 | |
| MUON_MOMENTUM_WARMUP_START | 0.92 | |
| MUON_MOMENTUM_WARMUP_STEPS | 1500 | |
| WARMDOWN_ITERS | 3000 | |
| ITERATIONS | 9000 | |
| MAX_WALLCLOCK_SECONDS | 600 | |
| EVAL_STRIDE | 64 | |
| MUON_WD | 0.04 | |
| ADAM_WD | 0.04 | |

## Run plan

| Run | Config delta | Goal |
|-----|-------------|------|
| A | Overtone only (11L) | Baseline: does Overtone help? |
| B | Overtone + NUM_LAYERS=12 | 12L capacity (artifact ~16MB?) |
| C | Overtone + EMA_DECAY=0.996 | Bracket EMA optimum |
| D | Overtone + EMA_DECAY=0.998 | Bracket EMA optimum |
| E | Best config, 3 seeds | Submission validation |

## What we dropped (from our PR #4)

- **KURE + R2** — Redundant with EMA 0.997 (quant gap 0.0067 without them)
- **Tanh reparam** — Cost ~55ms/step overhead (4309 vs 7103 steps)
- **TTT LoRA** — Regressed +0.147 BPB

## Expected outcome

~1560 lines (+5 from base). Training at ~84ms/step. Target: val_bpb ≤ 1.1271.
