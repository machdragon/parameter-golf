# Snippet for `program.md` (in your **autoresearch** clone)

Paste or merge into [karpathy/autoresearch](https://github.com/karpathy/autoresearch) `program.md` if you use that repo for ideation:

---

## Context: Parameter Golf (separate repo)

We use this autoresearch repo only to **prototype** training ideas. Production work targets **[openai/parameter-golf](https://github.com/openai/parameter-golf)** with `train_gpt.py`, FineWeb, tokenizer-agnostic **val_bpb**, and a **16MB** int8+zlib artifact cap.

- Do **not** treat `val_bpb` from this nanochat-style setup as comparable to the official leaderboard.
- Prefer changes that can be explained as: optimizer schedule, LR, depth/width, attention pattern, or quantization — so they can be **ported** to `train_gpt.py` later.

---
