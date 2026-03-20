# Leaderboard PR checklist (Parameter Golf)

Official rules live in the [openai/parameter-golf README](https://github.com/openai/parameter-golf/blob/main/README.md) and **Submission Process** section.

## Before you open the PR

- [ ] **Beat current SOTA by ≥ 0.005 nats** with statistical evidence (**p &lt; 0.01**), typically **multiple seeds** and logs.
- [ ] **Train + eval** within published limits (**10 min on 8×H100** for record track; separate eval time limits apply).
- [ ] **Artifact ≤ 16,000,000 bytes** (decimal MB) — **code bytes + compressed model**, self-contained, no network at eval.
- [ ] Add **only** a new folder under the correct **`records/...`** track, e.g. `records/track_10min_16mb/<your_run>/`.

## Folder should include (per README)

- [ ] `README.md` — what you changed and why.
- [ ] `submission.json` — metadata template matching other records.
- [ ] **Train log(s)** showing reproducibility and significance vs prior SOTA.
- [ ] `train_gpt.py` (and any deps) that **run from that folder** for verification.

## After local quick-rank screening

Quick runs in `research/scripts/quick_rank_lanes.py` are **indicative only**. For the real PR:

1. Copy the winning recipe into a **standalone `train_gpt.py`** under the new `records/...` folder (see existing records).
2. Run full **8×H100** (or official eval environment) with the same **data + tokenizer** paths as documented.
3. Archive **complete logs** and compute **significance vs** the current leaderboard row you are beating.

## References

- [Parameter Golf — GitHub](https://github.com/openai/parameter-golf/)
- Submission and non-record rules are described in the repo README.
