# P1 ablation results (quick harness)

Companion to [`docs/p1_lane_knob_audit.md`](p1_lane_knob_audit.md) and the arc matrix [`arc_inputs/case_matrix_v3_full.json`](../arc_inputs/case_matrix_v3_full.json).

## Summary

| Lane | Minimal ablation attempted | Harness | Quick gate (val_bpb + 1.10× time) | Notes |
|------|----------------------------|---------|-------------------------------------|-------|
| **lane_1** | recurrence_only / +LoRA / +gating | **Blocked** | — | No `NUM_RECURRENCE_LOOPS`, training LoRA, or gating envs in [`train_gpt.py`](../train_gpt.py). |
| **lane_2** | `NUM_KV_HEADS=2` vs default `4` (8 heads) | **Run** | **PASS** | Script: [`scripts/run_p1_lane2_kv_quick_gate.sh`](../scripts/run_p1_lane2_kv_quick_gate.sh). Does **not** include recurrence (matrix sub-lane). |
| **lane_7** | LAWA EMA (`LAWA_ENABLED=1`, `LAWA_MODE=ema`) | **Run** | **TBD** (re-run on idle GPU) | [`scripts/run_p1_lane7_lawa_quick_gate.sh`](../scripts/run_p1_lane7_lawa_quick_gate.sh). Implementation: [`train_gpt_lawa.py`](../train_gpt_lawa.py). |
| **lane_8** | MTP | **Blocked** | — | No `MTP_*` in trainer. |
| **lane_9** | value_from_x0 | **Blocked** | — | No `VALUE_FROM_X0_*` in trainer. |
| **lane_10** | mirrored skip | **Blocked** | — | No `MIRRORED_SKIP_*` in trainer. |

## lane_2 run artifact (2026-03-20)

- Summary JSON: `logs/quick_harness/p1_lane2_kv_quick_gate_20260319_201836.json`
- Snapshots: `logs/quick_harness/baseline.json`, `logs/quick_harness/candidate_p1_lane2_kv2.json`

| Profile | num_kv_heads | quick_metric val_bpb | train_time_ms | Roundtrip exact val_bpb (log) |
|---------|--------------|----------------------|---------------|-------------------------------|
| baseline | 4 | 3.34271319 | 51879 | 3.33462305 |
| candidate | 2 | 3.33353185 | 48382 | 3.32680095 |

**Interpretation:** Under the **same** harness settings (20 steps, seed 1337), fewer KV heads improved `quick_metric` and was faster. Parameter count dropped (~15.88M vs ~17.06M). This supports the **GQA/MQA efficiency** slice of lane_2 only; the matrix hypothesis also mentions **reinvesting** saved budget into **recurrence**—not tested until recurrence exists.

**Promotion:** Both runs reported `Total submission size int8+zlib` under 16 MiB. Re-run [`scripts/collect_lane5_promotion_metrics.sh`](../scripts/collect_lane5_promotion_metrics.sh) if you want a fresh merged promotion table including `candidate_p1_lane2_kv2`.

## lane_7 LAWA (implementation 2026-03-20)

**Env (see `Hyperparameters` in `train_gpt.py`):** `LAWA_ENABLED`, `LAWA_MODE` (`ema` \| `checkpoint`), `LAWA_EMA_DECAY`, `LAWA_INTERVAL`, `LAWA_WINDOW`.

**Behavior:** Online EMA maintains a **float32 shadow on GPU** per floating `state_dict` entry; checkpoint mode appends **CPU** snapshots every `LAWA_INTERVAL` steps (deque length `LAWA_WINDOW`). Before the **last** validation, `lawa_finalize_to_model` writes the shadow or mean snapshot into `base_model` so **`quick_metric` and export match** the averaged weights.

**Harness:** Default runner uses `LAWA_EMA_DECAY=0.97` for short runs. Candidate harness may exit `2` on gate **FAIL**; the runner uses `set +e` so summaries still write.

**Sample diagnostic run (pre–GPU-shadow optimization):** Per-step EMA used CPU copies → wall time blew the 1.10× gate; after **GPU shadow** EMA, cost should track baseline. **Re-run** `./scripts/run_p1_lane7_lawa_quick_gate.sh` on an **idle** GPU and record `p1_lane7_lawa_quick_gate_*.json` + `candidate_p1_lane7_lawa.json` here when stable.

## PR overlap (merged vs open)

See [`docs/RECORDS_PR_MAP.md`](RECORDS_PR_MAP.md) for merged record provenance. Matrix narrative per lane remains in `case_matrix_v3_full.json` → `external_evidence_notes`.
