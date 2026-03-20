# AGENTS.md

## Deprecated sibling repos

Do not edit or maintain **`parameter-golf-old`** (or other legacy checkouts). Active work and canonical ARC inputs live in this repo — see [`arc_inputs/README.md`](arc_inputs/README.md).

## Surprise alerts

- `train_gpt.py` writes the full source code into `logs/<RUN_ID>.txt` before the runtime metrics. Any parser that scans those logs must anchor on concrete metric lines and ignore template strings in the source dump, or it will mis-parse placeholders like `{q_val_loss:.8f}` as real values.

See [`scripts/extract_run_metrics.py`](scripts/extract_run_metrics.py) (vendored from [GLDRoger/parameter-golf#13](https://github.com/openai/parameter-golf/pull/13)) for a parser that follows this rule.
