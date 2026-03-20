# Recorded quick harness runs

Committed snapshots from `./scripts/quick_harness.sh` (parsed `quick_metric` line). Raw logs live under `logs/quick_harness/` (gitignored).

| File | Description |
|------|-------------|
| `quick_harness_baseline_20260319.json` | Baseline profile, 20 train steps, `USE_COMPILE=0` |
| `lane5_quick_gate_20260320.json` | Lane_5 quick-gate run (`MDL_only`, `sparsity_only`, `full`) against refreshed baseline |
