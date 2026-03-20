# karpathy/autoresearch as a sidecar (ideas only)

[autoresearch](https://github.com/karpathy/autoresearch) is a **separate** project: one GPU, agent-edited **`train.py`**, fixed time budget, **`program.md`** for instructions.

## Why clone it next to Parameter Golf

- Good **habit loop**: tight edit → train → metric → keep/discard.
- **Bad** as a direct submission vehicle for [Parameter Golf](https://github.com/openai/parameter-golf): different data, metric plumbing, and artifact rules.

## Suggested layout

```text
~/Projects/parameter-golf-fresh/    # upstream-aligned; PRs add records/ + train_gpt
~/Projects/autoresearch/            # clone of karpathy/autoresearch — brainstorm only
```

## Workflow

1. Try an architectural or optimizer idea in **autoresearch** `train.py` until `val_bpb` moves the way you expect.
2. **Port** the minimal equivalent into **`train_gpt.py`** (or a **`records/.../train_gpt.py`**) in small, reviewable steps.
3. Validate on **FineWeb** + **int8+zlib** roundtrip the way the challenge requires.

## Optional `program.md` snippet (for your autoresearch clone)

Add a short section reminding the agent: *Parameter Golf scores are FineWeb `val_bpb` after training with official data paths; do not claim PG leaderboard numbers from this repo’s runs.*
