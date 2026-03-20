# AutoResearchClaw + Parameter Golf Workflow

This note maps the local AutoResearchClaw 23-stage pipeline to a Parameter Golf implementation workflow. AutoResearchClaw owns orchestration and decisioning; Parameter Golf owns the training code, experiment artifacts, and submission packaging.

## 23-Stage Mapping

| Stage | AutoResearchClaw stage | Parameter Golf mapping | Owner |
|---|---|---|---|
| 1 | `TOPIC_INIT` | Define the Parameter Golf objective, constraints, and success metric | AutoResearchClaw |
| 2 | `PROBLEM_DECOMPOSE` | Split the challenge into model, data, system, and evaluation subproblems | AutoResearchClaw |
| 3 | `SEARCH_STRATEGY` | Decide where to search: repo history, docs, records, baselines, and notes | AutoResearchClaw |
| 4 | `LITERATURE_COLLECT` | Collect candidate ideas, prior runs, and relevant implementation evidence | AutoResearchClaw |
| 5 | `LITERATURE_SCREEN` `[gate]` | Approve the short list of ideas and references before spec work starts | Shared gate |
| 6 | `KNOWLEDGE_EXTRACT` | Turn sources into constraint cards, baseline facts, and reusable findings | AutoResearchClaw |
| 7 | `SYNTHESIS` | Combine findings into a small set of Parameter Golf design directions | AutoResearchClaw |
| 8 | `HYPOTHESIS_GEN` | Write testable model/training hypotheses with measurable predictions | AutoResearchClaw |
| 9 | `EXPERIMENT_DESIGN` `[gate]` | Lock the experiment spec, baselines, metrics, and budget | Shared gate |
| 10 | `CODE_GENERATION` | Generate or patch experiment code from the locked spec | Parameter Golf |
| 11 | `RESOURCE_PLANNING` | Plan runs, compute budget, and execution order | Parameter Golf |
| 12 | `EXPERIMENT_RUN` | Execute the runs and collect raw outputs | Parameter Golf |
| 13 | `ITERATIVE_REFINE` | Do bounded repair and rerun loops against the same spec | Parameter Golf |
| 14 | `RESULT_ANALYSIS` | Summarize metrics, tradeoffs, failures, and deltas | AutoResearchClaw |
| 15 | `RESEARCH_DECISION` | Decide proceed, pivot, or refine based on observed evidence | AutoResearchClaw |
| 16 | `PAPER_OUTLINE` | Draft the implementation note outline and final report skeleton | AutoResearchClaw |
| 17 | `PAPER_DRAFT` | Draft the implementation note / runbook content | AutoResearchClaw |
| 18 | `PEER_REVIEW` | Check the draft for missing evidence, false claims, and unclear steps | AutoResearchClaw |
| 19 | `PAPER_REVISION` | Revise the workflow note and runbook after review | AutoResearchClaw |
| 20 | `QUALITY_GATE` `[gate]` | Final approval of the packaged workflow, outputs, and reproducibility | Shared gate |
| 21 | `KNOWLEDGE_ARCHIVE` | Archive lessons learned, run metadata, and reusable settings | AutoResearchClaw |
| 22 | `EXPORT_PUBLISH` | Package the final Parameter Golf artifact bundle | Parameter Golf |
| 23 | `CITATION_VERIFY` | Verify links, references, and any external claims used in the note | AutoResearchClaw |

## Ownership Boundary

AutoResearchClaw owns the orchestration logic: stage sequencing, gate handling, hypothesis/decision generation, review, and archival. Parameter Golf owns the codepath that actually changes the training artifact: experiment spec consumption, source edits, run execution, metrics capture, and final packaging.

The handoff point is the experiment spec. After that, the workflow should avoid free-form code exploration unless it is explicitly approved as a spec update.

## Gate Usage

- Stage 5 filters out weak or off-target ideas before they become a Parameter Golf spec.
- Stage 9 is the main lock point: once approved, the experiment spec becomes the contract for code generation and execution.
- Stage 20 is the final release gate: it checks that the package is reproducible, internally consistent, and ready to hand off.

## Stage 10-13 Constraints

Stages 10-13 should run under a strict "no free baseline rewrite" rule.

- Preserve the approved baseline implementation unless the experiment spec explicitly changes it.
- Allow bug fixes, harness fixes, logging, and adapter code, but not silent architecture changes.
- Keep edits tied to the spec: every code change should map to a named hypothesis, metric, or failure mode.
- If a baseline must change, treat that as a new spec revision and send it back through Stage 9.
- During refinement, prefer targeted patches and reruns over re-deriving the whole baseline from scratch.

## Planned 6-Step Pipeline

1. Ingestion: pull in the local repo context, prior runs, docs, and constraints.
2. Hypothesis: turn the research prompt into a small set of falsifiable Parameter Golf hypotheses.
3. Design: freeze the experiment spec, baselines, metrics, budget, and allowed edits.
4. Execution: generate code, plan resources, run experiments, and capture outputs.
5. Decision loop: analyze results, then proceed, refine, or pivot with bounded edits.
6. Packaging: archive the run, export the final bundle, and verify release readiness.

## Minimal Integration Contracts

### 1) Experiment Spec

`experiment_spec.yaml` or equivalent.

Required fields:

- `spec_id`
- `objective`
- `baseline_ref`
- `hypotheses`
- `metrics`
- `budget`
- `allowed_edits`
- `acceptance_criteria`

### 2) Run Output

`run_output.json` plus artifact paths.

Required fields:

- `run_id`
- `spec_id`
- `command`
- `status`
- `metrics`
- `artifact_paths`
- `logs`
- `timestamps`

### 3) Decision Output

`decision_output.md` or `decision_output.json`.

Required fields:

- `decision` (`proceed`, `refine`, or `pivot`)
- `rationale`
- `evidence`
- `next_actions`
- `spec_changes` if any

## Implementation Rule of Thumb

If a stage output cannot be traced back to the locked spec, it should be treated as an orchestration bug, not as an acceptable exploratory edit.
