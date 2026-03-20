# P1 lane knob audit (`train_gpt.py`)

Crosswalk between [`arc_inputs/case_matrix_v3_full.json`](../arc_inputs/case_matrix_v3_full.json) **P1** lanes (`lane_1`, `lane_2`, `lane_7`, `lane_8`, `lane_9`, `lane_10`) and **environment variables / code** in [`train_gpt.py`](../train_gpt.py) as of this audit.

| Lane | Matrix `primary_knobs` | In `train_gpt.py`? | Notes |
|------|------------------------|--------------------|-------|
| **lane_1** | `num_shared_blocks`, `num_recurrence_loops`, `lora_rank`, `lora_alpha`, `lora_dropout`, `residual_gate_enabled`, `per_head_gate_enabled` | **No** | Training-time recurrence + LoRA + gating not present. `TTT_LORA_*` is **eval-time** TTT only. |
| **lane_2** | `num_heads`, `num_kv_heads`, `head_dim`, `num_recurrence_loops` | **Partial** | `NUM_HEADS`, `NUM_KV_HEADS` **yes** (defaults 8 / 4). Head dim is `model_dim // num_heads` (no separate `HEAD_DIM` env). `NUM_RECURRENCE_LOOPS` **no**. |
| **lane_7** | `lawa_enabled`, `lawa_interval`, `lawa_window` | **No** | LAWA / late checkpoint averaging not implemented. |
| **lane_8** | `mtp_num_heads`, `mtp_loss_weight`, `mtp_start_step`, `mtp_end_step` | **No** | MTP auxiliary heads not implemented. |
| **lane_9** | `value_from_x0_enabled`, `value_projection_rank`, `value_projection_dropout` | **No** | Shared value-from-x0 path not implemented. |
| **lane_10** | `mirrored_skip_enabled`, `skip_projection_rank`, `skip_scale` | **No** | Mirrored U-Net-style skips not implemented (only fixed `skip_weights` on blocks unrelated to this lane). |

## Runnable minimal slice today

- **lane_2 (GQA / KV only):** A/B with fixed `NUM_HEADS=8` and `NUM_KV_HEADS` ∈ `{4, 2}` matches the **attention-efficiency** part of the lane. It does **not** test recurrence reinvestment (no recurrence in trainer). Runner: [`scripts/run_p1_lane2_kv_quick_gate.sh`](../scripts/run_p1_lane2_kv_quick_gate.sh); results: [`docs/p1_ablation_results.md`](p1_ablation_results.md).

## Testing contract (all P1 lanes, when wired)

Same as matrix `evaluation_plan`: [`scripts/quick_harness.sh`](../scripts/quick_harness.sh) baseline → candidate; gate via [`tools/quick_harness_report.py`](../tools/quick_harness_report.py) `compare` (`val_bpb` ↓ and `train_time_ms` ≤ 1.10× baseline); promotion: `SKIP_POST_TRAIN_EVAL=0`, int8+zlib bytes ≤ 16 MiB (see [`scripts/collect_lane5_promotion_metrics.sh`](../scripts/collect_lane5_promotion_metrics.sh) pattern).
