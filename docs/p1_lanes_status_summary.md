# P1 lanes status (1, 2, 7, 8, 9, 10) — merged PR overlap and testing plan

Synthesis from [`arc_inputs/case_matrix_v3_full.json`](../arc_inputs/case_matrix_v3_full.json), [`docs/p1_lane_knob_audit.md`](p1_lane_knob_audit.md), [`docs/RECORDS_PR_MAP.md`](RECORDS_PR_MAP.md), [`arc_inputs/SOTA_CROSSWALK_RUN_CHECKLIST.md`](../arc_inputs/SOTA_CROSSWALK_RUN_CHECKLIST.md), [`docs/experiment_workflow.md`](experiment_workflow.md), and the lane-5 / quick-harness tooling.

## 1. Have merged PRs covered any P1 hypotheses?

**No.** Merged records under `records/track_10min_16mb/` and the crosswalk checklist only touch *adjacent* ideas. None validate the exact P1 hypotheses framed in the matrix. Per-lane `external_evidence_notes` in the case matrix already document this.

| Lane | Hypothesis (matrix) | Merged PR overlap (what was noted) | Status |
|------|---------------------|------------------------------------|--------|
| **lane_1** | Recurrence + LoRA + gating composite | FP16Embed: recurrence fails under 10 min wall-clock. LoRA TTT ([PR #77](https://github.com/openai/parameter-golf/pull/77)) is *eval-time* only. | Open gap (training-time composite unproven) |
| **lane_2** | Recurrence + MQA/GQA efficiency | Baseline already ships GQA (8/4 heads). No dedicated KV-head sweep in any merged README. | Partial (KV knobs exist; recurrence reinvestment missing) |
| **lane_7** | LAWA checkpoint averaging | Absent from merged records. Appears in open (unmerged) PRs. | High-value gap |
| **lane_8** | MTP training-only aux heads | Absent. Open PRs combine MTP with int6 stack (hints only). | High-value gap |
| **lane_9** | Value projections from shared x0 | Explicit gap in merged READMEs. | Open |
| **lane_10** | U-Net mirrored skips | Sparse “skip” mentions in open PR titles only. | Open |

**Sources**

- `case_matrix_v3_full.json` → each lane’s `external_evidence_notes`
- `SOTA_CROSSWALK_RUN_CHECKLIST.md` → § “Still high value vs README gaps”
- `p1_lane_knob_audit.md` → which matrix knobs are **not wired** in current `train_gpt.py`

Open PRs (unmerged) are **hypothesis hints only** — not accepted records.

## 2. How to test (same contract as lane_5)

Use the **quick-harness** pipeline.

```bash
# 1. Baseline (once)
./scripts/quick_harness.sh baseline

# 2. Candidate — export lane-specific env from matrix starter_pseudocode / primary_knobs
./scripts/quick_harness.sh candidate
```

`quick_harness.sh candidate` already runs `quick_harness_report.py compare` against `baseline.json` (runtime factor 1.10). To compare saved snapshots explicitly:

```bash
python3 tools/quick_harness_report.py compare \
  --baseline logs/quick_harness/baseline.json \
  --candidate logs/quick_harness/candidate.json \
  --runtime-factor 1.10
```

**Quick gate** (every P1 `evaluation_plan`): `val_bpb` lower than baseline **and** `train_time_ms` ≤ baseline × 1.10.

**Promotion** (after a pass): keep `SKIP_POST_TRAIN_EVAL=0` (default in harness when not smoke-testing). Aggregate roundtrip + int8+zlib bytes vs 16 MiB:

```bash
./scripts/collect_lane5_promotion_metrics.sh
```

Default output: `logs/quick_harness/lane5_promotion_metrics.json` (override with `LANE5_PROMOTION_OUT=...`). Uses `quick_harness_report.py promotion`.

### Runnable today without new `train_gpt.py` features

- **lane_2:** `NUM_HEADS` / `NUM_KV_HEADS` only — e.g. `NUM_KV_HEADS=2` vs default `4`. Convenience runner: [`scripts/run_p1_lane2_kv_quick_gate.sh`](run_p1_lane2_kv_quick_gate.sh). Latest recorded result: [`docs/p1_ablation_results.md`](p1_ablation_results.md) (**quick gate PASS** on that slice; still not “recurrence reinvestment”).

### Other five lanes

1. Wire env-controlled behavior in `train_gpt.py` (see `p1_lane_knob_audit.md`).
2. Start from matrix `minimal_ablations` (single knob / single composite step).
3. Reuse lane-5-style wrappers as templates: [`scripts/run_lane5_quick_gate.sh`](run_lane5_quick_gate.sh), [`scripts/run_lane5_compression_weight_sweep.sh`](run_lane5_compression_weight_sweep.sh).

## 3. Suggested priority (low effort → leverage)

1. **lane_7 LAWA** or **lane_8 MTP** first (matrix story: no deploy-time byte cost once heads are dropped / averaging is post-hoc) — *after* trainer support exists.
2. Quick gate: baseline + one candidate.
3. If any candidate passes: run `collect_lane5_promotion_metrics.sh` and archive a full record folder per [`docs/experiment_workflow.md`](experiment_workflow.md).

The harness and promotion tooling are already aligned with lane_5; remaining work is **trainer feature flags** for lanes 1, 7, 8, 9, and 10, plus **recurrence** if pursuing the full lane_1 / lane_2 matrix wording.
