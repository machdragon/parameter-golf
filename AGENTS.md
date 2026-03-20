# AGENTS.md

## Deprecated sibling repos

Do not edit or maintain **`parameter-golf-old`** (or other legacy checkouts). Active work and canonical ARC inputs live in this repo — see [`arc_inputs/README.md`](arc_inputs/README.md).

## Surprise alerts

- `train_gpt.py` writes the full source code into `logs/<RUN_ID>.txt` before the runtime metrics. Any parser that scans those logs must anchor on concrete metric lines and ignore template strings in the source dump, or it will mis-parse placeholders like `{q_val_loss:.8f}` as real values.

See [`scripts/extract_run_metrics.py`](scripts/extract_run_metrics.py) (vendored from [GLDRoger/parameter-golf#13](https://github.com/openai/parameter-golf/pull/13)) for a parser that follows this rule.

## Staging profile

Set `STAGING_PROFILE=1` to inject merged-baseline defaults (PR60 + PR124) before `Hyperparameters` reads env vars. Individual env vars always override staging defaults.

| Env var | Staging default | Vanilla default |
|---------|-----------------|-----------------|
| `NUM_LAYERS` | 10 | 9 |
| `WARMDOWN_ITERS` | 2500 | 1200 |
| `TIED_EMBED_LR` | 0.10 | 0.05 |
| `LAWA_ENABLED` | 1 | 0 |
| `EVAL_STRIDE` | 512 | 0 (no sliding) |
| `MUON_WEIGHT_DECAY` | 0.02 | 0.0 |
| `ADAM_WEIGHT_DECAY` | 0.01 | 0.0 |

Sliding window eval activates when `EVAL_STRIDE > 0`. It replaces the standard `eval_val` call in the post-train int8 roundtrip with `eval_sliding_roundtrip` from [`train_gpt_sliding.py`](train_gpt_sliding.py).
