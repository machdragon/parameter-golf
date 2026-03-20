# LAWA-EMA Frontier Fork (PR #198 base)

**Base:** PR #198 (11L, 1.1318 BPB — int6 + MLP3x + relu² + FA3 + NTK RoPE + SmearGate + BigramHash + OrthoInit + U-Net skips + SWA + WD=0.04)
**Changes from base:**
1. SWA → LAWA-EMA (exponential moving average of weights)
2. Overtone init added (SVD power-law embedding spectrum from PR #60)
**Status:** Pre-ablation — no runs yet

## What's in this fork

Everything from PR #198 (the current leaderboard record at 1.1318 BPB), plus two modifications:

| Method | Source | What it does |
|--------|--------|-------------|
| 11 layers | pr198 | 5 encoder + 6 decoder (U-Net split), +4.4M params vs 9L |
| Int6 per-row quantization | pr162 | 6-bit quantization for MLP/attn weights, fp16 passthrough for embeddings |
| zstd-22 compression | pr162 | ~15% smaller artifacts than zlib-9 |
| MLP 3x + relu² | pr162/pr164 | hidden=1536, relu(x)² activation for sparsity + int6 tolerance |
| FA3 | pr198 | flash_attn_3 interface, ~13ms/step faster than SDPA on H100 |
| NTK-aware RoPE | pr60/pr198 | Auto-scales base frequency when eval seq_len > train seq_len |
| SmearGate | pr162 | Learned gate blending current/previous token embedding |
| BigramHash embedding | pr162 | 2048-bucket hash of token pairs → 128d → 512d |
| Orthogonal weight init | pr162 | Proj layers scaled by 1/√(2·n_layers) |
| U-Net skip connections | pr162 | Encoder-decoder skip with learned per-channel weights |
| WD=0.04 | pr198 | Higher weight decay shrinks magnitudes → better int6 quantization |
| GQA (8 heads, 4 KV) | pr162 | Grouped-query attention |
| Logit softcap (30.0) | pr162 | Prevents logit explosion |
| Sliding window eval (stride=64) | pr162 | Nearly full 2048-token context per scored token |
| **LAWA-EMA** | **pr197 (new)** | **Replaces SWA. Float32 EMA shadow, updated every step** |
| **Overtone init** | **pr60 (new)** | **SVD power-law decay on embedding spectrum for smoother int6** |

## What we changed

### Change 1: SWA → LAWA-EMA

**SWA in PR #198:** Collects 8 checkpoints every 200 steps during warmdown (scale < 0.5), then uniform-averages them. Effective window ~1400 steps.

**LAWA-EMA in this fork:**
```
lawa_enabled = 1
lawa_ema_decay = 0.995   # configurable via LAWA_EMA_DECAY
```
- At step 0: `shadow = model.state_dict().float().clone()` (float32 on GPU)
- Every step: `shadow = decay * shadow + (1 - decay) * current_weights`
- After training: copy shadow → model → int6 quantize → zstd compress → export
- Effective window: 1/(1-decay) ≈ 200 steps at default decay

### Change 2: Overtone init

Added after normal embedding init in `_init_weights`:
```python
U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
self.tok_emb.weight.data = (U * target_S[None, :]) @ V
```
SVD decomposes the random embedding, replaces singular values with power-law decay (1/√k). Produces smoother per-row value ranges → tighter int6 quantization. Confirmed beneficial in PR #60, costs nothing at training time.

## Effective window comparison

| Method | Effective window | Weighting | When it runs |
|--------|-----------------|-----------|-------------|
| SWA (pr198) | ~1400 steps (8 × 200) | Uniform | Warmdown only |
| LAWA decay=0.990 | ~100 steps | Exponential (recent-heavy) | Every step |
| LAWA decay=0.995 | ~200 steps | Exponential | Every step |
| LAWA decay=0.997 | ~333 steps | Exponential | Every step |
| LAWA decay=0.999 | ~1000 steps | Exponential | Every step |

## Why we expect this might beat PR #198's 1.1318

### The improvement chain

1. **Stronger base (PR #198 vs old PR #162).** We now ablate on the best confirmed config: 11L, WD=0.04, relu², FA3. Any LAWA win here is a real improvement over SOTA.

2. **LAWA-EMA vs SWA.** Exponential weighting naturally down-weights older checkpoints. Float32 shadow preserves precision. Every-step updates vs every-200-step snapshots.

3. **Overtone init.** Power-law embedding spectrum → smoother weight geometry at init → tighter per-row ranges → less int6 quantization error. Same causal chain as LAWA but orthogonal to it.

### The bear case

1. **SWA's broader window may be better.** 1400-step uniform average explores more of the loss basin than a 200-step exponential window.
2. **The improvement may be tiny.** SWA already works well on PR #198's base.
3. **GPU memory cost.** ~107MB for float32 shadow (26.8M params × 4 bytes). Fine on H100.

## Ablation plan

| # | What | Config | Gate |
|---|------|--------|------|
| 1 | Reproduce pr198 | Original pr198 with SWA | Within 0.002 of 1.1318 |
| 2 | No averaging | `LAWA_ENABLED=0` | Expect worse |
| 3 | LAWA default | `LAWA_EMA_DECAY=0.995` | Must beat or match Exp 1 |
| 4 | Decay sweep | {0.990, 0.995, 0.997, 0.999} | Pick best |
| 5 | 3-seed validation | Best decay, SEED={1337, 42, 7} | Mean beats 1.1318 |

## Run config

```bash
NUM_LAYERS=11 MUON_WD=0.04 ADAM_WD=0.04 BIGRAM_VOCAB_SIZE=2048 \
LAWA_ENABLED=1 LAWA_EMA_DECAY=0.995 \
torchrun --nproc_per_node=8 records/track_10min_16mb/lawa_frontier/train_gpt.py
```
