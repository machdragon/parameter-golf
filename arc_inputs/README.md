# ARC inputs (canonical)

This directory holds the **single source of truth** for the full hypothesis grid used with AutoResearchClaw / lane tooling:

- **`case_matrix_v3_full.json`** — Lane definitions, priorities, and evidence notes (version in the JSON `version` field).
- **`SOTA_CROSSWALK_RUN_CHECKLIST.md`** — Deduped run checklist vs merged READMEs and open PRs.

## Lane 5 (P0) — compression-aware training in stock `train_gpt.py`

Hypothesis: **artifact-centric / MDL-style** penalties on block weights. Environment variables (all default **0** = off):

| Variable | Role |
|----------|------|
| `COMPRESSION_OBJECTIVE_WEIGHT` | Weight on STE **fake-int8** MSE on 2D block matrices |
| `MDL_PENALTY_LAMBDA` | Weight on **mean(w²)** on those matrices |
| `STRUCTURAL_SPARSITY_LAMBDA` | Weight on **mean(\|w\|)** (L1-style) |

**Quick gate:** `scripts/quick_harness.sh` runs `train_gpt.py` with **`SKIP_TTT_EVAL=1`** by default. Compare baseline vs candidate with the same `SEED`, `ITERATIONS`, etc.; `tools/quick_harness_report.py compare` uses **`quick_metric`** only.

**Promotion metric (post-roundtrip):** run with **`SKIP_POST_TRAIN_EVAL=0`** (default) and **`SKIP_TTT_EVAL=1`** to log `final_int8_zlib_roundtrip_exact` without the long TTT phase, or set `SKIP_TTT_EVAL=0` for the full leaderboard tail. Snapshot JSON may include **`final_int8_zlib_roundtrip_exact`** when that line appears in the log.

Example candidate:

```bash
export COMPRESSION_OBJECTIVE_WEIGHT=0.01
export MDL_PENALTY_LAMBDA=1e-6
export STRUCTURAL_SPARSITY_LAMBDA=1e-6
./scripts/quick_harness.sh candidate
```
