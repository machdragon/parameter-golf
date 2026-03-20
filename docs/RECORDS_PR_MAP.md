# Record track: PR provenance

Maps `records/track_10min_16mb/` submission folders to how they landed on `openai/parameter-golf` `main` and who authored them. Use this when diffing snapshot `train_gpt.py` files or tracing leaderboard entries.

| Record folder | How it landed | Primary author (GitHub) | PR / commit |
|---------------|---------------|-------------------------|-------------|
| [`2026-03-17_NaiveBaseline`](../records/track_10min_16mb/2026-03-17_NaiveBaseline) | Initial repo **Launch snapshot** (not a community PR) | [0hq](https://github.com/0hq) (commit author: Will DePue) | Commit [`a15093ad`](https://github.com/openai/parameter-golf/commit/a15093adad328a650d421e53c078cbd2c45beb0e) |
| [`2026-03-18_FP16Embed_WD3600`](../records/track_10min_16mb/2026-03-18_FP16Embed_WD3600) | Merged PR | [chonchiog](https://github.com/chonchiog) | [PR #42](https://github.com/openai/parameter-golf/pull/42) (`chonchiog:submission`) |
| [`2026-03-19_10L_MixedPrecision`](../records/track_10min_16mb/2026-03-19_10L_MixedPrecision) (+ [`2026-03-18_LowerLR`](../records/track_10min_16mb/2026-03-18_LowerLR) from same PR) | Merged PR | [nanlliu](https://github.com/nanlliu) | [PR #39](https://github.com/openai/parameter-golf/pull/39) (`nanlliu:lower-lr-submission`) |
| [`2026-03-18_LongContextSeq2048`](../records/track_10min_16mb/2026-03-18_LongContextSeq2048) | Merged PR | [spokane-way](https://github.com/spokane-way) | [PR #49](https://github.com/openai/parameter-golf/pull/49) (`spokane-way:main`) |
| [`2026-03-17_LoRA_TTT`](../records/track_10min_16mb/2026-03-17_LoRA_TTT) | Merged PR | [samacqua](https://github.com/samacqua) | [PR #77](https://github.com/openai/parameter-golf/pull/77) (`samacqua:main`) |
| [`2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`](../records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit) | Merged PR | [notapplica](https://github.com/notapplica) | [PR #60](https://github.com/openai/parameter-golf/pull/60) (`notapplica:submission/ntk-eval-overtone-init`) |

**Sliding-window eval fix:** [PR #124](https://github.com/openai/parameter-golf/pull/124) corrects the final partial window in [`2026-03-19_SlidingWindowEval/train_gpt.py`](../records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py) (`window_starts` length ≥ 1; score start `max(wlen - stride, 0)`). This repo’s copy includes that fix.

**Optional sweep tooling (not from the rows above):** [GLDRoger](https://github.com/GLDRoger)’s draft [PR #13](https://github.com/openai/parameter-golf/pull/13) branch `docs/experiment-workflow-and-sweep-tools` was vendored into [`scripts/`](../scripts/) and [`docs/experiment_workflow.md`](experiment_workflow.md). Upstream `openai/parameter-golf` does not ship these by default.
