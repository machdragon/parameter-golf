# Patterns from top `records/` submissions (this repo)

Synthesized from READMEs under [`records/track_10min_16mb/`](../records/track_10min_16mb/) and the [public leaderboard](https://github.com/openai/parameter-golf/). Use this to prioritize **what to try before** a full 10-minute / 8×H100 submission run.

## Recurring themes

1. **Depth vs. bytes** — Many records add **`NUM_LAYERS=10`** (or more) and recover artifact budget via **Muon weight decay**, **mixed int8/int6** on middle layers, or **FP16 tied embedding** export so int8+zlib still fits 16MB.

2. **Learning rates** — Several wins use **lower** `MATRIX_LR` / `SCALAR_LR` / `TIED_EMBED_LR` than the starter defaults (often ~**half** defaults). Example: [`2026-03-18_LowerLR`](../records/track_10min_16mb/2026-03-18_LowerLR/README.md).

3. **Attention / eval** — **Sliding-window validation**, longer **TRAIN_SEQ_LEN** (e.g. 2048), or **stride** tricks change effective context at eval time; often shipped as **custom `train_gpt.py` inside the record folder**, not env-only.

4. **Mixed precision in export** — Per-layer quantization (`INT4_LAYERS` / step in record scripts) to shave **zlib** size without collapsing quality. See [`2026-03-19_10L_MixedPrecision`](../records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md).

5. **Test-time training (LoRA)** — Starter script includes **TTT LoRA** eval knobs (`TTT_LORA_RANK`, etc.); see [`2026-03-17_LoRA_TTT`](../records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md).

6. **Statistics for records** — Beating SOTA requires **≥ 0.005 nats** improvement with **p &lt; 0.01** evidence (often **3 seeds**). Plan full runs accordingly; quick sweeps here are **screening only**.

## Suggested reading order

1. [`2026-03-17_NaiveBaseline`](../records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) — reference point  
2. [`2026-03-18_LowerLR`](../records/track_10min_16mb/2026-03-18_LowerLR/README.md) — simple LR lesson  
3. [`2026-03-19_10L_MixedPrecision`](../records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md) — depth + export tradeoff  
4. [`2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`](../records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) — current strong recipe (complex)
