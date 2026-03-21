# LAWA + KURE/R2 + Tanh Reparam + LoRA TTT

Built on top of PR #201 (LAWA-EMA + Int6 + Overtone + MLP3x).

This folder now includes the first full `8xH100` Modal validation run for this
recipe: `RUN_ID=kure-r2-ttt-001`. The run completed successfully, saved
artifacts to the Modal volume, and was downloaded locally for inspection.

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

## Validated run

Command:

```bash
.venv-modal/bin/modal run scripts/modal_train_kure_r2_ttt.py --run-id kure-r2-ttt-001
```

Hardware / runtime:

- `8xH100` on Modal
- Wallclock cap hit at `4309/20000` steps after `600.136s`
- Peak memory: `20315 MiB allocated`, `20750 MiB reserved`

Core train/eval configuration observed in the canonical run:

- `TRAIN_BATCH_TOKENS=786432`, `TRAIN_SEQ_LEN=2048`
- `NUM_LAYERS=11`, `BIGRAM_VOCAB_SIZE=2048`
- `KURE_LAMBDA=0.01`, `R2_LAMBDA=0.01`, `TANH_REPARAM=1`
- `LAWA_ENABLED=1` with parallel decay candidates `0.995`, `0.999`, `0.9995`
- `EVAL_STRIDE=64`
- `TTT_LORA_ENABLED=1`, `TTT_LORA_RANK=8`

## Results

- Pre-export eval at stop: `val_loss=2.0231`, `val_bpb=1.1982`
- LAWA selected `decay=0.995`
- Final int6 roundtrip exact: `val_loss=2.06612423`, `val_bpb=1.22367515`
- Final int6 sliding-window exact: `val_loss=2.02852027`, `val_bpb=1.20140397`
- Final TTT LoRA exact: `val_loss=2.27474682`, `val_bpb=1.34723312`
- Serialized model int6+zstd: `10344428` bytes
- Code size: `79614` bytes
- Total submission size int6+zstd: `10424042` bytes

## What we learned

1. The recipe is operational on Modal `8xH100` end to end: training, LAWA
selection, export, and volume persistence all worked without manual recovery.
2. Sliding-window int6 evaluation is the right headline metric for this run.
It stayed close to the live pre-export score (`1.1982 -> 1.2014`), so the
quantized sliding path appears healthy.
3. The LoRA TTT path regressed badly on this configuration (`1.3472` bpb),
which means TTT should remain a research toggle here rather than the default
submission path.
4. Among the parallel EMA tracks, the shortest decay (`0.995`) still won on
proxy BPB for this 10-minute run, despite the higher default decay in the
script.
5. The recipe is well under the 16 MB cap, so future work should focus on
quality rather than compression.

## Artifact notes

The validated Modal run produced:

- `final_model.pt`
- `final_model.int6.ptz`
- `logs/kure-r2-ttt-001.txt`

These artifacts were downloaded to `modal_runs/kure-r2-ttt-001/` for local
inspection; they are not committed into this record folder.
