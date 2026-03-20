# P1 lanes testing blueprint

Quick-harness gate on 20-step runs, `SEED=1337`, same contract as lane_5. See also [`docs/p1_lanes_status_summary.md`](p1_lanes_status_summary.md) and [`docs/p1_lane_knob_audit.md`](p1_lane_knob_audit.md).

## Philosophy

From each P1 `evaluation_plan` in [`arc_inputs/case_matrix_v3_full.json`](../arc_inputs/case_matrix_v3_full.json) and [`docs/experiment_workflow.md`](experiment_workflow.md):

1. Run **baseline** once: `./scripts/quick_harness.sh baseline`.
2. Run **one candidate** (minimal ablation): `./scripts/quick_harness.sh candidate` with lane env set.
3. **Gate:** lower `val_bpb` than baseline **and** `train_time_ms` ≤ 1.10× baseline (`tools/quick_harness_report.py compare`; also auto-run at end of `quick_harness.sh candidate`).
4. If pass → promotion: `SKIP_POST_TRAIN_EVAL=0` (default) + `./scripts/collect_lane5_promotion_metrics.sh` (roundtrip + int8+zlib ≤ 16 MiB).
5. **Sweep** only after a quick-pass win (pattern: [`scripts/run_lane5_compression_weight_sweep.sh`](run_lane5_compression_weight_sweep.sh)).

**Quick-pass first** for every lane. Sweeps only after a gate-passer.

## Summary table

| Lane | Hypothesis (1-line) | Primary knobs | Minimal ablations (matrix) | Recommended strategy | Code status |
|------|---------------------|---------------|----------------------------|----------------------|-------------|
| **1** | Recurrence + LoRA + gating composite beats naive depth | shared blocks/loops, LoRA rank/α, gates | recurrence_only → +LoRA → full | Quick pass one composite, then sweep `lora_rank` if pass | Not in `train_gpt.py` |
| **2** | KV reduction + (eventually) recurrence reinvestment | `NUM_KV_HEADS`, future recurrence | recurrence+GQA2 (full matrix) | **KV-only quick pass today**; recurrence later | **Partial** — heads wired |
| **7** | Late-window checkpoint / param averaging | `LAWA_*` | LAWA_interval10_window5 | Quick pass only | Not wired |
| **8** | MTP aux heads (train-only) | `MTP_*` | MTP_weight0.15 | Quick pass only | Not wired |
| **9** | Value from shared x0 | `VALUE_FROM_X0_*` | value_from_x0_rank16 | Quick pass only | Not wired |
| **10** | Mirrored U-Net-style skips | `MIRRORED_SKIP_*` | mirrored_skip_rank8 | Quick pass only | Not wired (existing `skip_weights` ≠ this lane) |

## Per-lane notes

### Lane 1 — Recurrence + LoRA + gating

After wiring, env shape like matrix `starter_pseudocode`:

```bash
export NUM_SHARED_BLOCKS=3 NUM_RECURRENCE_LOOPS=3 LORA_RANK=8 RESIDUAL_GATE_ENABLED=1
./scripts/quick_harness.sh candidate
```

Implementation touches GPT forward, optimizer param groups, and export path (ensure LoRA/gates don’t inflate artifact or are merged for export). “Minimal ~30 lines” is a lower bound; expect broader changes + tests.

### Lane 2 — MQA/GQA (+ recurrence later)

**Runnable now** without recurrence:

```bash
export NUM_KV_HEADS=2
./scripts/quick_harness.sh baseline
./scripts/quick_harness.sh candidate
```

Or use [`scripts/run_p1_lane2_kv_quick_gate.sh`](run_p1_lane2_kv_quick_gate.sh). A **quick-pass PASS** on the KV-only slice is recorded in [`docs/p1_ablation_results.md`](p1_ablation_results.md) (same seed/steps contract; not recurrence reinvestment).

### Lane 7 — LAWA

**Wired** in [`train_gpt.py`](../train_gpt.py) + [`train_gpt_lawa.py`](../train_gpt_lawa.py). Quick gate runner: [`scripts/run_p1_lane7_lawa_quick_gate.sh`](../scripts/run_p1_lane7_lawa_quick_gate.sh).

```bash
export LAWA_ENABLED=1 LAWA_MODE=ema LAWA_EMA_DECAY=0.97
./scripts/quick_harness.sh candidate
# checkpoint-style (needs enough steps for K snapshots), e.g. 20 iters:
# LAWA_MODE=checkpoint LAWA_INTERVAL=4 LAWA_WINDOW=5
```

EMA shadow stays on **GPU** for speed; finalize runs **before** final `eval_val` so `quick_metric` reflects averaged weights.

### Lane 8 — MTP

After wiring:

```bash
export MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15
./scripts/quick_harness.sh candidate
```

Aux heads and loss only in training; strip heads on export so byte budget unchanged.

### Lane 9 — Value from x0

After wiring:

```bash
export VALUE_FROM_X0_ENABLED=1 VALUE_PROJECTION_RANK=16
./scripts/quick_harness.sh candidate
```

### Lane 10 — Mirrored skips

After wiring:

```bash
export MIRRORED_SKIP_ENABLED=1 SKIP_PROJECTION_RANK=8
./scripts/quick_harness.sh candidate
```

Do not conflate with existing block `skip_weights` until spec matches matrix intent.

## Workflow (lane_5 style)

1. Pick a lane and wire **one** minimal ablation.
2. Add optional wrapper `scripts/run_p1_lane<N>_quick_gate.sh` (copy [`scripts/run_lane5_quick_gate.sh`](run_lane5_quick_gate.sh) / [`scripts/run_p1_lane2_kv_quick_gate.sh`](run_p1_lane2_kv_quick_gate.sh)).
3. Baseline + candidate → compare.
4. On pass → `collect_lane5_promotion_metrics.sh`.
5. On continued promise → knob sweep script.

**Infra:** `quick_harness.sh`, `quick_harness_report.py`, promotion collector — no new pipeline required.

## Which lane first? (recommendation)

- **Already done (KV slice):** lane_2 with `NUM_KV_HEADS=2` — use results doc; next for lane_2 is **recurrence** if you want the full matrix hypothesis.
- **Next implementation (if choosing between LAWA and “smallest delta”):** **LAWA (lane_7)** is a contained training-loop / state-copy concern and matches the “zero extra parameters at deploy” story — but it is **not** the smallest change in lines of code versus flipping one existing env; it **is** often smaller than MTP or attention-structure refactors.
- **Larger lifts:** lane_8 (MTP), lane_9 (attention path), lane_10 (topology), lane_1 (recurrence + LoRA + gating).

If you want a concrete next step: **implement lane_7 LAWA (EMA or last-K checkpoint average)** behind `LAWA_ENABLED`, then add `run_p1_lane7_quick_gate.sh` mirroring lane_2’s script.
