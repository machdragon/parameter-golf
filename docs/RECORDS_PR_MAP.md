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

## Learning Review Plan (Closed + Open PRs)

Goal: extract transferable ideas and failure modes from the linked PRs/issues, then decide what we should test, adopt, or ignore in this repo.

**Scope note:** [PR #77](https://github.com/openai/parameter-golf/pull/77) was listed twice in the request; it is tracked once below.

### Review order

1. **Merged lineage first (historical progression):** [#39](https://github.com/openai/parameter-golf/pull/39), [#42](https://github.com/openai/parameter-golf/pull/42), [#49](https://github.com/openai/parameter-golf/pull/49), [#52](https://github.com/openai/parameter-golf/pull/52), [#60](https://github.com/openai/parameter-golf/pull/60), [#61](https://github.com/openai/parameter-golf/pull/61), [#77](https://github.com/openai/parameter-golf/pull/77)
2. **Correctness/measurement context:** [#124](https://github.com/openai/parameter-golf/pull/124), [Issue #83](https://github.com/openai/parameter-golf/issues/83)
3. **Open frontier proposals:** [#150](https://github.com/openai/parameter-golf/pull/150), [#156](https://github.com/openai/parameter-golf/pull/156), [#160](https://github.com/openai/parameter-golf/pull/160), [#162](https://github.com/openai/parameter-golf/pull/162), [#164](https://github.com/openai/parameter-golf/pull/164)

### Per-item extraction rubric

For each PR/issue, capture:

1. **Claimed gain:** exact metric claim (`val_bpb`, mean vs single-run, and whether sliding-window eval is used).
2. **Mechanism:** what changed in code/params (architecture, quantization, optimizer/lr schedule, context/eval settings, init, tokenization/compression tricks).
3. **Evidence quality:** merged vs open, review feedback, unresolved questions, reproducibility signals (seed count, script clarity, deterministic settings).
4. **Cost/risk:** complexity added, training/eval overhead, fragility, and compatibility with current `records/track_10min_16mb` conventions.
5. **Action:** classify as `Adopt now`, `Ablate next`, `Watchlist`, or `Reject`, with one-sentence rationale.

### Tracker (completed pass 1 on 2026-03-19 UTC)

| Item | State | Author | Score(s) | Description | Relevant code | Evidence quality | Action |
|------|-------|--------|----------|-------------|---------------|------------------|--------|
| [PR #39](https://github.com/openai/parameter-golf/pull/39) | closed (merged) | [nanlliu](https://github.com/nanlliu) | `1.2147` exact merged record; PR discussion reports `1.2139` mean over 5 seeds | 10L depth plus mixed int8/int6 middle-layer compression, plus lower LR from companion LowerLR run | [10L README](../records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md), [10L train_gpt.py](../records/track_10min_16mb/2026-03-19_10L_MixedPrecision/train_gpt.py), [LowerLR train_gpt.py](../records/track_10min_16mb/2026-03-18_LowerLR/train_gpt.py) | Strong multi-seed signal, but author flagged H200 vs H100 mismatch caveat | `Ablate next` to separate depth gain from quantization/layout effects on H100 |
| [PR #42](https://github.com/openai/parameter-golf/pull/42) | closed (merged) | [chonchiog](https://github.com/chonchiog) | `1.2197` and `1.2201` on H100; additional H200 runs `~1.216-1.218` | Keep tied embedding in fp16 during export and retune warmdown/LR | [README](../records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md), [train_gpt.py](../records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/train_gpt.py) | Good reproducibility for an early submission and later methods repeatedly confirm fp16 embedding sensitivity | `Adopt now` as a standard ablation axis whenever artifact budget allows |
| [PR #49](https://github.com/openai/parameter-golf/pull/49) | closed (merged) | [spokane-way](https://github.com/spokane-way) | `1.20576485` exact; `1.2064` mean over 3 seeds | Seq2048 long-context training with tuned LRs and tied embeddings | [README](../records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md), [train_gpt.py](../records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py) | Strong 3-seed reproducibility with full logs | `Adopt now` as a baseline long-context recipe |
| [PR #52](https://github.com/openai/parameter-golf/pull/52) | closed (merged) | [spokane-way](https://github.com/spokane-way) | `1.20143417` exact; `1.20136` mean over 3 seeds | Seq4096 training, 3/4 batch, Muon momentum warmup, lower LR, longer warmdown | [README](../records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md), [train_gpt.py](../records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py) | Strong and well-documented; clean standalone reruns | `Adopt now` for high-context training experiments |
| [PR #60](https://github.com/openai/parameter-golf/pull/60) | closed (merged) | [notapplica](https://github.com/notapplica) | `1.1748` mean over 3 seeds | 6-technique stack: sliding eval, fp16 embed export, 10L, Muon WD, overtone init, phase-mix init | [README](../records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md), [train_gpt.py](../records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py) | Strong headline score, but many coupled changes | `Adopt now` as operating baseline; ablate components in follow-up |
| [PR #61](https://github.com/openai/parameter-golf/pull/61) | closed (merged) | [saml212](https://github.com/saml212) | `1.2154` single-run claim in final title/body | Aggressive always-decaying warmdown schedule, fp16 tied embed, NTK eval-length tuning | [README](../records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md), [train_gpt.py](../records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py) | Medium; discussion shows earlier score-title mismatch and no multi-seed confirmation in-thread | `Watchlist` until the warmdown claim is revalidated under matched settings |
| [PR #77](https://github.com/openai/parameter-golf/pull/77) | closed (merged) | [samacqua](https://github.com/samacqua) | Submission `1.195`; internal eval mean `1.1928` over 4 runs | Document-isolated sliding eval plus LoRA test-time training | [README](../records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md), [record train_gpt.py](../records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py), [root train_gpt.py](../train_gpt.py) | Medium; interesting idea but evaluation-time adaptation and global script touch increase integration risk | `Watchlist` for optional eval-time research, not default leaderboard baseline |
| [PR #124](https://github.com/openai/parameter-golf/pull/124) | closed (merged) | [mattqlf](https://github.com/mattqlf) | No model score; correctness patch to scoring | Fixes final partial-window handling in sliding eval | [train_gpt.py](../records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py) (`window_starts` at lines ~860-861 and scoring start at ~902) | High confidence bugfix and merged quickly | `Adopt now` as measurement baseline dependency |
| [Issue #83](https://github.com/openai/parameter-golf/issues/83) | open | [jordankzf](https://github.com/jordankzf) | N/A; community snapshot thread | Unofficial leaderboard rollup of open PR claims and techniques | [Issue thread](https://github.com/openai/parameter-golf/issues/83) | Useful for discovery, but not canonical and includes val-only/memorization entries | `Watchlist` only; do not use as source of truth for accepted records |
| [PR #150](https://github.com/openai/parameter-golf/pull/150) | open | [yahya010](https://github.com/yahya010) | `1.1593` mean over 3 seeds (WIP) | Int6 QAT plus BigramHash, MLP 1344, zstd, seq2048, sliding eval | [README](https://github.com/openai/parameter-golf/blob/5930991ebde94544128e3f27e50f13f91b015969/records%2Ftrack_10min_16mb%2F2026-03-20_Int6QAT_BigramHash_MLP1344%2FREADME.md), [train_gpt.py](https://github.com/openai/parameter-golf/blob/5930991ebde94544128e3f27e50f13f91b015969/records%2Ftrack_10min_16mb%2F2026-03-20_Int6QAT_BigramHash_MLP1344%2Ftrain_gpt.py) | Medium; promising but explicitly marked WIP | `Ablate next` after freezing a non-WIP config |
| [PR #156](https://github.com/openai/parameter-golf/pull/156) | open | [dexhunter](https://github.com/dexhunter) | `1.16019` mean over 3 seeds | Int6 STE, NorMuon, SWA, 3x MLP, sliding eval, zstd | [README](https://github.com/openai/parameter-golf/blob/effbd870163a0c5f02bc13f16cf875b698bb81ce/records%2Ftrack_10min_16mb%2F2026-03-20_Int6STE_NorMuon_SWA_SlidingWindow%2FREADME.md), [train_gpt.py](https://github.com/openai/parameter-golf/blob/effbd870163a0c5f02bc13f16cf875b698bb81ce/records%2Ftrack_10min_16mb%2F2026-03-20_Int6STE_NorMuon_SWA_SlidingWindow%2Ftrain_gpt.py) | Medium-high; multi-seed and detailed, but still unmerged | `Ablate next` with focus on NorMuon and SWA marginal gain |
| [PR #160](https://github.com/openai/parameter-golf/pull/160) | open | [ChaseWNorton](https://github.com/ChaseWNorton) | `1.16230441` sliding score from under-cap repack | 3x MLP, seq2048, grouped low-bit export with lzma, int8 token embedding fallback | [README](https://github.com/openai/parameter-golf/blob/9c4146af4b677f92bffa83282b4d3584a91b2d3b/records%2Ftrack_10min_16mb%2F2026-03-19_ChaseNorton%2FREADME.md), [train_gpt.py](https://github.com/openai/parameter-golf/blob/9c4146af4b677f92bffa83282b4d3584a91b2d3b/records%2Ftrack_10min_16mb%2F2026-03-19_ChaseNorton%2Ftrain_gpt.py) | Medium-low; score is strong but appears single-run and QAT never activated in timed run | `Watchlist` pending multi-seed confirmation |
| [PR #162](https://github.com/openai/parameter-golf/pull/162) | open | [raahilshah](https://github.com/raahilshah) | `1.1483` mean over 3 seeds | Int6 MLP3x plus SmearGate, BigramHash, Muon WD, SWA, orthogonal init | [README](https://github.com/openai/parameter-golf/blob/14cdf6f7a4c84d5016dbd6adc6df6d5208b83261/records%2Ftrack_10min_16mb%2F2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA%2FREADME.md), [train_gpt.py](https://github.com/openai/parameter-golf/blob/14cdf6f7a4c84d5016dbd6adc6df6d5208b83261/records%2Ftrack_10min_16mb%2F2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA%2Ftrain_gpt.py), [review note](https://github.com/openai/parameter-golf/pull/162#discussion_r2963789299) | Medium; one review comment flags possible tail-token double-counting in sliding eval | `Watchlist` until scoring bug concern is resolved |
| [PR #164](https://github.com/openai/parameter-golf/pull/164) | open | [jfprincz](https://github.com/jfprincz) | `1.1524` best seed; `1.1538` mean over 3 seeds | Builds on prior int6+MLP3x with orthogonal init, SmearGate, BigramHash, tuned Muon, seq2048 sliding | [v2 README](https://github.com/openai/parameter-golf/blob/7c44af135b0b314f78758460c3132cadcd480ddd/records%2Ftrack_10min_16mb%2F2026-03-20_OrthoInit_Muon_Int6_MLP3x_SmearBigram_1.1524%2FREADME.md), [v2 train_gpt.py](https://github.com/openai/parameter-golf/blob/7c44af135b0b314f78758460c3132cadcd480ddd/records%2Ftrack_10min_16mb%2F2026-03-20_OrthoInit_Muon_Int6_MLP3x_SmearBigram_1.1524%2Ftrain_gpt.py) | Medium-high; strong seeds and clean narrative, but unmerged | `Ablate next` with first priority on orthogonal init plus bigram/smear additions |

### Operational Split (use now vs experiment only)

Detailed run sheet: [`docs/open_pr_experiment_matrix.md`](open_pr_experiment_matrix.md)

#### A) Merged PRs: immediate baseline we should use now

| Baseline layer | Use now | Source PR(s) | Why this is baseline | Code starting point |
|----------------|---------|--------------|----------------------|---------------------|
| Measurement correctness | Required | [#124](https://github.com/openai/parameter-golf/pull/124) | Prevents under/over-counting tail tokens in sliding eval | [SlidingWindowEval train_gpt.py](../records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py) |
| Primary model/eval recipe | Default | [#60](https://github.com/openai/parameter-golf/pull/60) | Best merged score among reviewed merged PRs (`mean val_bpb=1.1748`) with 3-seed evidence | [PR60 train_gpt.py](../records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py) |
| Stable long-context control arm | Default control | [#52](https://github.com/openai/parameter-golf/pull/52), [#49](https://github.com/openai/parameter-golf/pull/49) | Strong reproducible long-context baselines for non-stacked comparisons | [PR52 train_gpt.py](../records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py), [PR49 train_gpt.py](../records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py) |
| Export sensitivity guardrail | Enabled unless size breaks | [#42](https://github.com/openai/parameter-golf/pull/42) | Tied embedding precision repeatedly shows large quantization sensitivity | [PR42 train_gpt.py](../records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/train_gpt.py) |

#### B) Open PRs: unverified ideas to feed experiment matrix

| Matrix factor | Candidate values | Pulled from open PRs | Keep/skip notes |
|---------------|------------------|----------------------|-----------------|
| `bigram_signal` | `off`, `hash4096_d128`, `smear_only`, `hash+smear` | [#150](https://github.com/openai/parameter-golf/pull/150), [#162](https://github.com/openai/parameter-golf/pull/162), [#164](https://github.com/openai/parameter-golf/pull/164) | High upside, low parameter cost; prioritize early |
| `init_scheme` | `baseline_init`, `orthogonal_mup` | [#162](https://github.com/openai/parameter-golf/pull/162), [#164](https://github.com/openai/parameter-golf/pull/164) | Isolate from bigram/smear to avoid attribution blur |
| `optimizer_variant` | `muon`, `muon_wd`, `normuon` | [#156](https://github.com/openai/parameter-golf/pull/156), [#162](https://github.com/openai/parameter-golf/pull/162) | Needs strict apples-to-apples seed matching |
| `swa` | `off`, `last50pct_q200` | [#156](https://github.com/openai/parameter-golf/pull/156), [#162](https://github.com/openai/parameter-golf/pull/162) | Cheap to test once checkpoint cadence is fixed |
| `quant_train_mode` | `post_training_only`, `int6_ste_qat` | [#150](https://github.com/openai/parameter-golf/pull/150), [#156](https://github.com/openai/parameter-golf/pull/156) | WIP risk; evaluate only after freezing other knobs |
| `export_mix` | `int6+zstd22`, `grouped_lzma`, `tok_emb_fp16`, `tok_emb_int8` | [#150](https://github.com/openai/parameter-golf/pull/150), [#160](https://github.com/openai/parameter-golf/pull/160), [#164](https://github.com/openai/parameter-golf/pull/164) | Keep size cap and reload correctness as hard constraints |
| `seq_len_train` | `1024`, `2048` | [#150](https://github.com/openai/parameter-golf/pull/150), [#160](https://github.com/openai/parameter-golf/pull/160), [#164](https://github.com/openai/parameter-golf/pull/164) | Couple with step-time accounting to avoid hidden compute shifts |
| `sliding_stride` | `64`, `256` | [#150](https://github.com/openai/parameter-golf/pull/150), [#160](https://github.com/openai/parameter-golf/pull/160), [#164](https://github.com/openai/parameter-golf/pull/164) | Evaluate only with PR124-corrected scorer |

#### Recommended matrix phase order

1. **Phase 0 (lock baseline):** PR60 + PR124 as canonical run; PR52 as control arm.
2. **Phase 1 (cheap additive):** `bigram_signal`, `init_scheme`, `sliding_stride`.
3. **Phase 2 (optimizer-level):** `optimizer_variant`, `swa`.
4. **Phase 3 (high-risk/high-coupling):** `quant_train_mode`, `export_mix`, `seq_len_train`.

### Top transferable ideas (current pass)

1. Sliding-window evaluation and correct partial-window scoring are the largest low-cost lever in this track.
2. Int6 mixed precision plus stronger compression (`zstd`/`lzma`) reliably buys model-capacity headroom.
3. MLP widening to 3x repeatedly improves quality when byte savings from quantization are reinvested.
4. Optimizer/schedule tuning (`LR`, `momentum warmup`, `warmdown`) materially changes quantization robustness.
5. Embedding-path precision is uniquely sensitive; fp16 passthrough or careful int8 handling is often decisive.

### High-risk ideas to avoid (for now)

1. Treating open-PR sliding scores as final without verifying scoring logic and seed stability.
2. Adopting large multi-technique stacks without isolated ablations of each component.
3. Depending on unofficial leaderboard snapshots as canonical evidence for acceptance.

### Follow-up ablation queue

1. `Ablate next`: PR #164 technique deltas over PR #70 baseline path (orthogonal init, SmearGate, BigramHash).
2. `Ablate next`: PR #156 NorMuon and SWA contribution after controlling for int6+MLP3x+sliding.
3. `Ablate next`: PR #60 six-way stack as one-toggle-at-a-time knockouts from its record script.
4. `Ablate next`: PR #39 depth vs mixed-precision compression under matched H100 hardware.
5. `Ablate next`: PR #150 BigramHash with a fixed non-WIP run configuration.
