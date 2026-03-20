# LAWA-EMA Frontier Fork

**Base:** PR #162 (9L, 1.1483 BPB — int6 + MLP3x + SmearGate + BigramHash + OrthoInit + SWA)
**Change:** Replace SWA with LAWA-EMA (running exponential moving average of weights)
**Status:** Pre-ablation — no runs yet

## What frontier methods are already in this fork?

Everything from pr162, which is the strongest validated 9L config:

| Method | Source | What it does |
|--------|--------|-------------|
| Int6 per-row quantization | pr162 | 6-bit quantization for MLP/attn weights, fp16 passthrough for embeddings/control tensors |
| zstd-22 compression | pr162 | Better compression than zlib-9 (~15% smaller artifacts) |
| MLP 3x expansion | pr162 | hidden=1536 (vs 2x=1024), biggest single contributor per pr162 author |
| SmearGate | pr162 | Learned gate blending current/previous token embedding |
| BigramHash embedding | pr162 | 4096-bucket hash of consecutive token pairs → 128d → 512d |
| Orthogonal weight init | pr162 | Better conditioning at init, proj layers scaled by 1/√(2·n_layers) |
| Muon + AdamW optimizers | pr162 | Muon for matrix params (WD=0.02), AdamW for scalars/embeddings (WD=0.01) |
| Muon momentum warmup | pr162 | 0.92→0.99 over 1500 steps |
| Grad clipping 0.3 | pr162 | Global norm clipping |
| Sliding window eval | pr162 | stride=64 final evaluation |
| GQA (8 heads, 4 KV) | pr162 | Grouped-query attention |
| RoPE | pr162 | Rotary positional embeddings |
| Logit softcap (30.0) | pr162 | Prevents logit explosion |
| U-Net skip connections | pr162 | Encoder-decoder skip with learned weights |

**NOT yet in this fork** (would come from pr198 for 11L scale-up):
- 11 layers (we're still at 9)
- WD=0.04 (we're at Muon 0.02 / Adam 0.01)
- FA3 (flash-attn-3)
- Bigram vocab 2048 (we're at 4096)

## What we changed: SWA → LAWA-EMA

### What SWA did in pr162

```
swa_start_frac = 0.5    # start collecting when lr_mul < 0.5 (warmdown phase)
swa_every = 200          # grab checkpoint every 200 steps
```

With ~8000 total steps and warmdown_iters=3000:
- Warmdown starts at ~step 5000
- SWA collects ~15 checkpoints at steps 5000, 5200, ..., 7800
- After training, uniform average: final_weights = mean(checkpoint_i for i in 1..15)
- **Effective averaging window: ~3000 steps, uniformly weighted**

### What LAWA-EMA does instead

```
lawa_enabled = 1
lawa_ema_decay = 0.995   # configurable
```

- At step 0: `shadow = model.state_dict().float().clone()` (float32 on GPU)
- Every step: `shadow = decay * shadow + (1 - decay) * current_weights`
- After training: copy shadow → model → quantize → export
- **Effective averaging window: 1/(1-decay) steps, exponentially weighted**

### Effective window comparison

| Method | Effective window | Weighting | When it runs |
|--------|-----------------|-----------|-------------|
| SWA (pr162) | ~3000 steps (15 × 200) | Uniform | Warmdown only (last ~37% of training) |
| LAWA decay=0.990 | ~100 steps | Exponential (recent-heavy) | Every step from step 0 |
| LAWA decay=0.993 | ~143 steps | Exponential | Every step from step 0 |
| LAWA decay=0.995 | ~200 steps | Exponential | Every step from step 0 |
| LAWA decay=0.997 | ~333 steps | Exponential | Every step from step 0 |
| LAWA decay=0.999 | ~1000 steps | Exponential | Every step from step 0 |

**Important:** At the default decay=0.995, our EMA window (~200 steps) is much
narrower than SWA's ~3000-step window. This means the default LAWA is averaging
a smaller neighborhood of the final weights. This could be better (less dilution
from earlier, worse checkpoints) or worse (less smoothing of late-stage noise).
The decay sweep is critical.

## Why we expect this might improve over SWA

### The bull case

1. **Exponential weighting is more principled than uniform.** Recent weights are
   better than weights from 3000 steps ago. SWA weights them equally. EMA
   naturally down-weights older, worse checkpoints.

2. **Float32 shadow preserves precision.** SWA in pr162 collects to CPU and sums
   in whatever dtype. Our shadow stays float32 on GPU — no precision loss from
   dtype casting or CPU round-trips during collection.

3. **Continuous tracking, no collection schedule.** SWA only samples every 200
   steps, missing the 199 in between. EMA sees every step, so the average is
   based on 40x more data points.

4. **Better interaction with int6 quantization.** A smoother weight distribution
   → smaller per-row ranges → less quantization error. EMA's smooth exponential
   blend may produce tighter distributions than SWA's coarser uniform average.

5. **Simpler code.** 2 hyperparams vs 3. No special warmdown-phase logic.

### The bear case

1. **SWA's broader window may be better.** Averaging 15 checkpoints across 3000
   steps explores more of the loss basin than a 200-step exponential window.
   This broader exploration could find a flatter minimum.

2. **SWA intentionally ignores early training.** It only collects during warmdown
   when the model is already good. EMA starts from random init — but at
   decay=0.995, step-0 weights have mass 0.995^8000 ≈ 0, so this is not
   actually a concern in practice.

3. **The improvement may be tiny.** SWA already works well. The frontier PRs
   report SWA gives smooth, quantization-friendly weights. The marginal gain
   from EMA vs uniform averaging may be within noise.

4. **GPU memory cost.** ~68MB for float32 shadow (17M params × 4 bytes). Not a
   problem on H100, but it's nonzero.

### What we learned from PR #197

PR #197 applied LAWA-EMA to a weaker baseline (staging profile, no SmearGate,
no BigramHash, no int6, 10L, stride-512 eval) and got **1.18926 BPB**. This is:
- Better than some older merged entries
- Still far from the frontier (1.1483 pr162, 1.1326 pr198)
- **Not proof that LAWA works well** — just proof it doesn't break things

The real test is this fork: same strong base as pr162, LAWA vs SWA head-to-head.

## What specifically we expect to improve

The metric that matters is **final_int8_zlib_roundtrip val_bpb** — the BPB after
quantization → compression → decompression → dequantization. This is the
submission metric.

The causal chain:

```
smoother averaged weights
  → tighter per-row value ranges
    → less int6 quantization error
      → better roundtrip fidelity
        → lower final val_bpb
```

We do NOT expect LAWA to improve:
- Raw training loss (it doesn't affect the training dynamics at all)
- Pre-quantization val_bpb (the model trains identically; averaging happens after)
- Step time (EMA overhead is ~0.1ms on ~70ms steps)

We DO expect LAWA to potentially improve:
- Post-quantization val_bpb (the submission metric)
- The gap between pre-quant and post-quant metrics (quantization robustness)

## Ablation plan

| # | What | Config | Gate |
|---|------|--------|------|
| 1 | Reproduce pr162 | Original pr162 script with SWA | Within 0.002 of 1.1483 |
| 2 | No averaging | `LAWA_ENABLED=0` on this fork | Expect +0.005-0.010 worse |
| 3 | LAWA default | `LAWA_EMA_DECAY=0.995` | Must beat or match Exp 1 |
| 4 | Decay sweep | {0.990, 0.993, 0.995, 0.997, 0.999} | Pick best, beat Exp 1 |
| 5 | 11L scale-up | Best decay + pr198 hyperparams | Must beat 1.1326 |
| 6 | 3-seed validation | SEED={1337, 42, 7} | Mean beats SOTA by ≥0.005 |

**Critical insight from the window analysis:** decay=0.995 (our default) gives a
~200-step window vs SWA's ~3000-step window. If the sweep shows 0.999 (1000-step
window) works best, that suggests broader averaging is better and SWA's intuition
was right — we'd just be doing it with exponential instead of uniform weights.
