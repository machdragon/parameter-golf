# Open PR Experiment Matrix (Run Sheet)

Concrete run sheet split into:

1. merged PR baseline runs we should execute immediately
2. open PR idea replays that are still unverified and should feed ablations

This matrix is scoped to the `10min_16mb` track.

## Automation

Use the launcher script:

```bash
# phase 0 + phase 1 (default)
./scripts/run_open_pr_matrix.sh --fetch-open-scripts

# phase 0 only, preview commands
./scripts/run_open_pr_matrix.sh --phase0 --dry-run

# phase 1 only
./scripts/run_open_pr_matrix.sh --phase1 --fetch-open-scripts
```

## Contract

- Hardware: `8xH100`
- Wallclock cap: `MAX_WALLCLOCK_SECONDS=600`
- Dataset/tokenizer:
  - `DATA_PATH=./data/datasets/fineweb10B_sp1024`
  - `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`
- Common launch:

```bash
RUN_ID=<run_id> \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 <train_script>
```

## Run ID convention

`mx_<phase>_<family>_<variant>_s<seed>`

Examples:
- `mx_p0_pr60_base_s1337`
- `mx_p1_open162_bigram_s42`

## Phase 0: merged baseline (use now)

| Run ID | Train script | Env overrides | Purpose |
|--------|--------------|---------------|---------|
| `mx_p0_pr60_base_s1337` | `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py` | `SEED=1337` | Canonical merged baseline |
| `mx_p0_pr60_base_s42` | same as above | `SEED=42` | Baseline variance estimate |
| `mx_p0_pr60_base_s7` | same as above | `SEED=7` | Baseline variance estimate |
| `mx_p0_pr52_ctrl_s1337` | `records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py` | `SEED=1337` | Long-context merged control arm |
| `mx_p0_pr52_ctrl_s42` | same as above | `SEED=42` | Control variance check |

Required scorer hygiene in all matrix runs:
- include PR124 sliding-window fix semantics (`window_starts >= 1`, `score_start=max(wlen-stride,0)`)

## Phase 1: open PR feature extraction (single-seed screen)

Use `SEED=1337` first. Promote only if score delta is meaningful and artifact remains under cap.

Note: open PR scripts are not merged in this repo. For replay, use the upstream script snapshot from each PR.

Suggested one-time fetch:

```bash
mkdir -p research/open_pr_replays
curl -L -o research/open_pr_replays/pr150_train_gpt.py https://github.com/openai/parameter-golf/raw/5930991ebde94544128e3f27e50f13f91b015969/records%2Ftrack_10min_16mb%2F2026-03-20_Int6QAT_BigramHash_MLP1344%2Ftrain_gpt.py
curl -L -o research/open_pr_replays/pr156_train_gpt.py https://github.com/openai/parameter-golf/raw/effbd870163a0c5f02bc13f16cf875b698bb81ce/records%2Ftrack_10min_16mb%2F2026-03-20_Int6STE_NorMuon_SWA_SlidingWindow%2Ftrain_gpt.py
curl -L -o research/open_pr_replays/pr160_train_gpt.py https://github.com/openai/parameter-golf/raw/9c4146af4b677f92bffa83282b4d3584a91b2d3b/records%2Ftrack_10min_16mb%2F2026-03-19_ChaseNorton%2Ftrain_gpt.py
curl -L -o research/open_pr_replays/pr162_train_gpt.py https://github.com/openai/parameter-golf/raw/14cdf6f7a4c84d5016dbd6adc6df6d5208b83261/records%2Ftrack_10min_16mb%2F2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA%2Ftrain_gpt.py
curl -L -o research/open_pr_replays/pr164_train_gpt.py https://github.com/openai/parameter-golf/raw/7c44af135b0b314f78758460c3132cadcd480ddd/records%2Ftrack_10min_16mb%2F2026-03-20_OrthoInit_Muon_Int6_MLP3x_SmearBigram_1.1524%2Ftrain_gpt.py
```

| Run ID | Source PR | Train script source | Env overrides (exact knobs) | Candidate signal |
|--------|-----------|---------------------|------------------------------|------------------|
| `mx_p1_open150_qat_on_s1337` | #150 | PR #150 `train_gpt.py` | `SEED=1337 USE_INT6=1 USE_ZSTD=1 ZSTD_LEVEL=22 EVAL_STRIDE=64` | Int6 STE/QAT + BigramHash stack |
| `mx_p1_open150_qat_off_s1337` | #150 | PR #150 `train_gpt.py` | `SEED=1337 USE_INT6=0 USE_ZSTD=1 ZSTD_LEVEL=22 EVAL_STRIDE=64` | Isolate QAT/low-bit effect |
| `mx_p1_open156_normuon_s1337` | #156 | PR #156 `train_gpt.py` | `SEED=1337 SWA_ENABLED=0 EVAL_STRIDE=64` | NorMuon without SWA |
| `mx_p1_open156_normuon_swa_s1337` | #156 | PR #156 `train_gpt.py` | `SEED=1337 SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=200 EVAL_STRIDE=64` | NorMuon + SWA delta |
| `mx_p1_open160_qgv3_lzma_s1337` | #160 | PR #160 `train_gpt.py` | `SEED=1337 COMPRESSOR=lzma QUANT_BITS=6 TOK_EMBED_BITS=8 FP16_EMBED=0 EVAL_STRIDE=256` | Grouped export (`QGv3`) + lzma packaging |
| `mx_p1_open160_zlib_ref_s1337` | #160 | PR #160 `train_gpt.py` | `SEED=1337 COMPRESSOR=zlib QUANT_BITS=6 TOK_EMBED_BITS=8 FP16_EMBED=0 EVAL_STRIDE=256` | Compression backend sensitivity |
| `mx_p1_open162_bigram_smear_s1337` | #162 | PR #162 `train_gpt.py` | `SEED=1337 BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 SWA_ENABLED=0 EVAL_STRIDE=64` | BigramHash + SmearGate stack |
| `mx_p1_open162_bigram_off_s1337` | #162 | PR #162 `train_gpt.py` | `SEED=1337 BIGRAM_VOCAB_SIZE=0 SWA_ENABLED=0 EVAL_STRIDE=64` | Turn off bigram path for attribution |
| `mx_p1_open164_v2_s1337` | #164 | PR #164 v2 `train_gpt.py` | `SEED=1337 EVAL_STRIDE=256 BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128` | Ortho+bigram+smear combined recipe |
| `mx_p1_open164_v2_nobigram_s1337` | #164 | PR #164 v2 `train_gpt.py` | `SEED=1337 EVAL_STRIDE=256 BIGRAM_VOCAB_SIZE=0` | Bigram contribution under v2 stack |

## Phase 2: promote winners to 3-seed confirmation

For each promoted Phase-1 variant, run:
- `SEED in {1337, 42, 7}`
- same script and knobs

Run IDs:
- `mx_p2_<family>_<variant>_s1337`
- `mx_p2_<family>_<variant>_s42`
- `mx_p2_<family>_<variant>_s7`

## Promotion gates

Promote from Phase 1 -> Phase 2 only if all pass:

1. `val_bpb` improvement vs `mx_p0_pr60_base_s1337` is at least `0.004`
2. artifact size <= `16_000_000` bytes
3. no scorer anomalies (especially sliding tail handling)

Adopt into baseline-candidate queue only if Phase 2 confirms:

1. 3-seed mean beats Phase-0 baseline mean by >= `0.005`
2. no single-seed collapse (max-min spread <= `0.004`)
3. result reproducible with the same script and env snapshot

## Matrix export fields (minimum)

Store per run:

- `run_id`
- `script_path_or_url`
- `source_pr`
- `seed`
- `step_stop`
- `step_avg_ms`
- `final_int8_zlib_roundtrip_exact_val_bpb` (or equivalent exact final metric line)
- `sliding_val_bpb` (if emitted)
- `artifact_bytes_total`
- `notes` (scoring caveats, failed constraints, unusual behavior)
