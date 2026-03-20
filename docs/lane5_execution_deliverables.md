# Lane 5 execution deliverables (quick harness + sweep)

Operational artifacts for the lane_5 screening workflow (see `.cursor/plans` runbook). **Do not treat this file as the source of truth for Cursor plan todos**—it records what was run in-repo.

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/run_lane5_quick_gate.sh`](../scripts/run_lane5_quick_gate.sh) | Baseline + MDL_only / sparsity_only / full ablations; summary `lane5_quick_gate_*.json` |
| [`scripts/run_lane5_compression_weight_sweep.sh`](../scripts/run_lane5_compression_weight_sweep.sh) | Phase 2: sweep `COMPRESSION_OBJECTIVE_WEIGHT` with fixed MDL + sparsity λ; use `LANE5_SKIP_BASELINE=1` to reuse `logs/quick_harness/baseline.json` |
| [`scripts/collect_lane5_promotion_metrics.sh`](../scripts/collect_lane5_promotion_metrics.sh) | Phase 3: aggregate `final_int8_zlib_roundtrip_exact` + `Total submission size int8+zlib` vs 16 MiB |
| [`tools/quick_harness_report.py`](../tools/quick_harness_report.py) | `promotion` subcommand (used by collect script) |

### Promotion CLI

```bash
python3 tools/quick_harness_report.py promotion --out logs/quick_harness/lane5_promotion_metrics.json \
  --entry baseline=logs/quick_harness/baseline.latest.log \
  --entry myrun=logs/quick_harness/candidate_full.latest.log
```

## Latest JSON outputs (regenerate paths after new runs)

- Quick gate summary: `logs/quick_harness/lane5_quick_gate_20260319_191730.json`
- Compression weight sweep: `logs/quick_harness/lane5_comp_weight_sweep_20260319_194735.json`
- Promotion table: `logs/quick_harness/lane5_promotion_metrics.json`

## Compression weight sweep (2026-03-20)

Fixed `MDL_PENALTY_LAMBDA=1e-6`, `STRUCTURAL_SPARSITY_LAMBDA=1e-6`, baseline reused (`LANE5_SKIP_BASELINE=1`).

| weight | quick_metric val_bpb | train_time_ms | gate |
|--------|----------------------|---------------|------|
| 0.0025 | 3.34422483 | 53407 | FAIL (worse bpb; runtime OK) |
| 0.005  | 3.34434496 | 55150 | FAIL (worse bpb; runtime over 1.10×) |
| 0.01   | 3.34498845 | 54377 | FAIL (worse bpb; runtime over 1.10×) |

Best sweep row by `val_bpb`: **0.0025** — still **fails** vs baseline `3.33682083`.

## Phase 4 (optional wave1)

**Not run.** No quick-harness row passed both **bpb_improved** and **runtime_ok**, so `run_wave1_screen.sh` / `extract_run_metrics.py` long confirmation was skipped.

## Next steps

- Try smaller `STRUCTURAL_SPARSITY_LAMBDA` / `MDL_PENALTY_LAMBDA`, or architecture changes; re-run quick gate then sweep.
- After any candidate **PASS**es the quick gate, re-run `./scripts/collect_lane5_promotion_metrics.sh` and optionally wave1.
