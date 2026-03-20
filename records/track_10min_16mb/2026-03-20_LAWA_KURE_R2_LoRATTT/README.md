# LAWA + KURE/R2 + Tanh Reparam + LoRA TTT

Built on top of PR #201 (LAWA-EMA + Int6 + Overtone + MLP3x).

## Changes from base

1. **KURE + R2 regularization** — Quantization-aware training regularizers.
   KURE pushes weight kurtosis toward 1.8 (uniform-like), R2 penalizes
   outliers beyond 2*std. Both improve int6 quantization fidelity.

2. **Tanh reparameterization** — CastedLinear stores latent P, computes
   W = tanh(P) on the fly during forward. Bounds effective weights to [-1,1],
   reducing quantization dynamic range. LAWA shadows track tanh(P) directly.

3. **Parallel EMA tracks** — Three decay rates (0.995, 0.999, 0.9995) run
   in parallel. At export, proxy eval picks the best. Default raised from
   0.995 to 0.999.

4. **Causal LoRA TTT** — Test-time training with per-document low-rank
   adapters on Q, V, and lm_head. Ported from PR #77. Per-doc reset,
   score-before-train, causal attention. Rank 8, lr 0.01.

## Hyperparameters (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| KURE_LAMBDA | 0.01 | Kurtosis regularization strength |
| R2_LAMBDA | 0.01 | Range regularization strength |
| TANH_REPARAM | 1 | Enable tanh weight reparameterization |
| LAWA_EMA_DECAY | 0.999 | Default EMA decay (parallel tracks: 0.995/0.999/0.9995) |
| TTT_LORA_ENABLED | 1 | Enable LoRA TTT at eval |
| TTT_LORA_RANK | 8 | LoRA rank |
| TTT_LORA_LR | 0.01 | LoRA learning rate |
| TTT_CHUNK_SIZE | 256 | Chunk size for TTT |
| TTT_EVAL_SEQ_LEN | 1024 | Context window for TTT |
| TTT_BATCH_SIZE | 64 | Batch size for TTT eval |
