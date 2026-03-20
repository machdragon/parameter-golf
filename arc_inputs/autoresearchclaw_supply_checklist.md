# AutoResearchClaw Supply Checklist (ARC-First)

This is the concrete package ARC needs for the Parameter Golf workflow.

- [x] `arc_inputs/topic_package.json`
  - Canonical topic string and domains.
  - Objective, quick-gate contract, and promotion policy.
- [x] `arc_inputs/source_corpus.manifest.json`
  - Canonical 7-doc source list with hashes and sizes.
- [x] `arc_inputs/source_corpus.zip`
  - Portable corpus bundle for ARC ingestion.
- [x] `arc_inputs/guardrails.json`
  - Allowed edits policy and traceability policy.
  - Protected files and allowed write areas.
- [x] `arc_inputs/case_matrix_v1.json`
  - Four lanes with hypothesis, direction, knobs, pass metric, and promotion requirements.
- [x] `arc_inputs/case_matrix_v2_full.json`
  - Full-coverage matrix for all high-priority topics.
  - Includes per-lane `hypothesis_statement`, `initial_research_seeds`, and `starter_pseudocode`.
- [x] `arc_inputs/case_matrix_v3_full.json`
  - Full-coverage matrix with execution details.
  - Adds per-lane `why_this_might_work`, `minimal_ablations`, `null_result_interpretation`, `evaluation_plan`, `what_we_are_doing`, and `what_still_needs_to_be_done`.
- [x] `arc_inputs/contracts/guardrails_schema.json`
- [x] `arc_inputs/contracts/experiment_spec_schema.json`
- [x] `arc_inputs/contracts/run_output_schema.json`
- [x] `arc_inputs/contracts/decision_output_schema.json`

## Four Required Research Lanes

1. `lane_1`: Recurrence + LoRA specialization
2. `lane_2`: Recurrence + MQA/GQA efficiency
3. `lane_3`: Compression-aware / QAT
4. `lane_4`: Tokenizer lane (strict gate)

## Full-Coverage Extension

Use `arc_inputs/case_matrix_v3_full.json` when you want all prioritized topics represented in the research grid with lane-execution briefs.
It keeps lanes `1-4` and adds lanes `5-12` for artifact-objective, serialization, LAWA, MTP, shared-value projection, mirrored-skip, position/attention tweaks, and optimizer/schedule tuning.

## Evaluation Plan (Existing Harness)

Quick gate:

```bash
./scripts/quick_harness.sh candidate
python tools/quick_harness_report.py compare --baseline logs/quick_harness/baseline.json --candidate logs/quick_harness/candidate.json --runtime-factor 1.10
python tools/parameter_golf_claw/pgclaw.py quick-gate logs/quick_harness/baseline.json logs/quick_harness/candidate.json
```

Promotion to full-path checks:

- Require quick-gate pass.
- Require post-roundtrip metrics and artifact-size compliance (`<= 16000000` bytes).
- Require reproducibility metadata.
- Apply stricter manual review for tokenizer lane (`lane_4`).
