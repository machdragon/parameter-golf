# Autonomous Research Plan for Parameter Golf

## 1. Challenge constraints and what success means

OpenAI's **Parameter Golf** challenge asks participants to minimize held-out loss on a fixed FineWeb validation set while staying within a **strict 16 MB artifact limit** and a **10-minute training budget on 8xH100s**. In this repo, the score that matters is tokenizer-agnostic **`val_bpb`** on FineWeb validation, and the current baseline shown in the README is **Naive Baseline 1.2244**. The same README also makes the workflow constraints explicit: submissions should be made as PRs that add a new `records/...` folder, the final artifact is counted as code bytes plus compressed model bytes, and the best models should live in `/records` rather than expanding the complexity of the shared baseline scripts.

The repo adds two more constraints that matter for workflow design. First, new SOTA records must beat the previous result by at least `0.005` nats and show `p < 0.01`, so any serious pipeline has to support repeated runs and significance checking rather than one-off wins. Second, tokenizer and dataset changes receive extra scrutiny, so correctness and accounting checks have to be stronger there than for ordinary architectural sweeps.

The PR landscape reinforces the same operational lesson. Merged work so far has been small and low-risk, while ambitious architectural changes are appearing mainly as self-contained `records/...` submissions. That means "pipeline success" for us should not mean "rewrite the root training stack automatically." It should mean:

1. Generate and run many candidate experiments with reproducible metadata and strict compliance checks.
2. Optimize the **compressed artifact lifecycle**, not just the float checkpoint.
3. Package the best outcomes as merge-friendly `records/...` submissions with clear writeups and evidence.

In other words, the automation should be aggressive in research operations and conservative in the benchmark-critical execution path.

## 2. Research synthesis from PR analysis, Grok, and Tao notes

The strongest idea cluster in the current Parameter Golf queue is already fairly clear from `PR_APPROACHES.md` and `PR_APPROACHES_DEEP_DIVE.md`: **depth recurrence**, **cheap specialization such as LoRA**, and **compression-aware training** are becoming the dominant composite strategy. Recurrence buys effective depth under a storage cap, LoRA repairs some of the rigidity introduced by aggressive weight tying, and QAT or roundtrip-aware methods attack the quality drop that appears after serialization and compression. MQA/GQA-style attention variants are a natural complement because they reduce KV overhead and let more of the artifact budget go toward useful compute.

The Tao notes provide a good way to organize why this cluster makes sense. The mechanics of LLMs are simple enough, but the hard part is prediction: natural language is neither pure noise nor fully structured mathematics. It is **partly structured and partly random**. That matters here because Parameter Golf is effectively asking: under severe byte constraints, which parts of text structure are worth explicitly representing, and which parts of the residual are too close to noise to deserve scarce capacity?

The Enigma analogy is useful only when bounded correctly. What transfers is not the historical machinery of decryption, but the idea that language contains **redundancy, recurring templates, and high-probability motifs**. Cribs in Enigma are analogous to high-mass patterns in natural text: common phrases, boilerplate, collocations, formatting regularities, and highly predictable function-word structure. What does **not** transfer is the adversarial key-recovery framing. Parameter Golf is probability modeling and compression, not cryptanalysis. The value of the analogy is that it reminds us that recurring patterns are real, exploitable, and disproportionately important for reducing total code length.

This is where the Grok synthesis helps sharpen the research objective. Parameter Golf should be treated as a **compressed-artifact optimization problem**, not a pure model-quality problem. The metric is downstream of tokenization, training, quantization, serialization, compression, and final roundtrip evaluation. A design that looks good in float weights but degrades badly after quantization is not actually good for this challenge. That is why QAT, roundtrip proxies, and custom serialization ideas keep reappearing in the strongest submissions: they are targeting the thing the benchmark actually scores.

Together, these notes suggest a practical math stack for the project:

- **Entropy / cross-entropy / code length**: `val_bpb` is the direct language of the challenge.
- **Structure vs randomness**: model gains come from compressing the structured part of text more efficiently.
- **Scaling-law humility**: smooth loss trends exist, but theory still does not tell us enough about which constrained design will win, so automation and ablations matter.

That combination leads to a disciplined research stance: use theory to choose promising levers, but use empirical sweeps to decide what actually survives the artifact budget and roundtrip path.

## 3. Why the promising ideas cluster around structured residual reduction

The most useful heuristic from the Tao notes is the two-bucket error model:

- **Structured residual**: missed recurring patterns that are still compressible with the right inductive bias.
- **Near-random residual**: weakly predictable tail behavior that is hard to improve under tight storage limits.

This turns the current PR frontier into a coherent story rather than a bag of tricks.

**Recurrence** is a bet that more iterative computation helps capture structured residual without paying for fully unique layers. If repeated blocks can progressively refine useful structure, then recurrence is exactly the kind of compute-for-storage trade the challenge invites.

**LoRA and other cheap specialization mechanisms** are a bet that pure recurrence is too rigid. If every loop pass is behaviorally identical, the model may fail to adapt to different structural roles across depth. Low-rank deltas restore some loop-specific behavior at much lower storage cost than untied blocks.

**QAT and compression-aware objectives** are a bet that part of the remaining error is introduced not by training failure but by the deployment path itself. If quantization destroys fine-grained structure in the learned distribution, then training against that deployment distortion is really a way of preserving structured signal through the artifact lifecycle.

**MQA/GQA** fits the same pattern. It reduces attention-side overhead so more of the fixed artifact budget can be spent on the components that capture useful structure.

**Tokenizer work** is best treated as a second-phase lever. It can be high upside because `val_bpb` depends on both token prediction and tokens-per-byte, but it must come after metric accounting is airtight. Tokenizer experiments are important, just not the first place to loosen correctness discipline.

This gives us a practical priority order for experiments:

1. Moderate recurrence plus cheap specialization.
2. Compression-aware training and roundtrip instrumentation.
3. Lighter attention variants such as MQA/GQA.
4. Tokenizer changes only after baseline accounting and reproducibility checks are strong.

The key insight is that these are not independent ideas. They are all trying to reduce the structured residual while respecting the compressed-artifact endpoint.

## 4. Workflow pipeline design with AutoResearchClaw as orchestrator

`/home/alex/Projects/AutoResearchClaw` turns out to be a much more concrete system than the earlier draft assumed. The local repo defines a **23-stage, 8-phase** pipeline, with gate stages at **5, 9, and 20**, prompt customization through `prompts.default.yaml`, multiple experiment modes (`sandbox`, `docker`, `ssh_remote`, plus a simulated mode), ACP support for persistent agent-backed sessions, and structured artifacts such as `exp_plan.yaml`, `results.json`, `analysis.md`, `decision.md`, and `verification_report.json`.

That means we do not need to describe it as a vague agentic shell. We can map it directly onto Parameter Golf.

### AutoResearchClaw's role

AutoResearchClaw should be the **primary orchestrator** for research operations:

- literature search and synthesis
- hypothesis generation
- experiment matrix design
- run scheduling
- result analysis
- REFINE / PIVOT decisions
- write-up generation

### Parameter Golf's role

Parameter Golf should remain the source of truth for the executable benchmark path:

- training scripts
- artifact-byte accounting
- compressed roundtrip evaluation
- wallclock compliance
- submission-folder packaging
- reproducible final record scripts

This boundary is critical. The orchestrator can be autonomous, but the score-producing path must remain deterministic, challenge-compliant, and repo-owned.

### Stage mapping

The right mapping between AutoResearchClaw and Parameter Golf is:

- **Stages 1-8**: directly useful for this project.
  - Use them for problem framing, literature collection, knowledge extraction, synthesis, and bounded hypothesis generation.
- **Stage 9**: the place where Parameter Golf-specific constraints are locked.
  - `16,000,000` byte cap
  - 10-minute training budget
  - record vs non-record track
  - tokenizer-variant choice
  - allowed search space and safety constraints
- **Stages 10-13**: useful only under constraint.
  - Generate configs, experiment specs, and isolated submission-folder code when needed.
  - Do **not** let these stages freely mutate the shared baseline path.
  - Prefer config-driven generation and reusable harnesses over open-ended code rewriting.
- **Stages 14-15**: highly useful.
  - Use `analysis.md` and PROCEED / REFINE / PIVOT decisions to drive the sweep loop.
- **Stages 16-23**: optional for the competition loop.
  - Useful for reports, internal docs, record README drafting, and paper-style exports.
  - Not required for the minimal score-optimization path.

### Planned pipeline for this repo

The concrete workflow should look like this:

1. **Source ingestion**
   - Feed PR notes, world-record notes, Tao notes, scaling-law notes, and prior run summaries into AutoResearchClaw's knowledge base.
2. **Hypothesis generation**
   - Produce bounded theses such as recurrence+LoRA, recurrence+MQA, recurrence+QAT, or tokenizer-first follow-up.
3. **Experiment design**
   - Emit a declarative experiment matrix with fixed knobs, allowed ranges, and hard compliance constraints.
4. **Execution**
   - Launch repo-owned training/eval jobs that produce canonical metrics and artifact-size outputs.
5. **Decision loop**
   - Use AutoResearchClaw's REFINE / PIVOT logic to kill weak branches, tighten promising ones, and escalate only the best candidates to repeated validation.
6. **Packaging**
   - Generate record-ready `records/...` folders and companion writeups for the top candidates.

The main practical recommendation is to **disable or constrain any AutoResearchClaw behavior that would freely rewrite the shared baseline scripts**. The repo's current maintainer signals point the other way: broad architectural autonomy belongs inside isolated experiments and record folders, not in the canonical entrypoints.

### Minimal integration contracts

To make this reliable, the workflow should standardize three doc-level contracts.

**Experiment spec**

- `hypothesis_id`
- architecture knobs
- compression knobs
- training knobs
- compliance limits

For this project, the main knobs should include recurrence layout, KV-head pattern, LoRA rank/placement, QAT schedule, tokenizer variant, seed, and runtime tier.

**Run output**

- `val_loss`
- `val_bpb`
- compressed bytes
- code bytes when applicable
- wallclock
- seed
- tokenizer variant
- pass/fail compliance flags

This should be the canonical interface between Parameter Golf execution and AutoResearchClaw analysis. The orchestrator should reason over structured results, not raw logs.

**Decision output**

- `PROCEED`, `REFINE`, or `PIVOT`
- rationale
- next experiment deltas
- confidence / evidence summary

This is exactly the kind of structured handoff AutoResearchClaw already supports through its decision stages.

## 5. Recommended near-term implementation path and risks

The near-term path should be a config-first pipeline rather than a code-generation-first pipeline.

### Recommended build order

1. Create a deterministic experiment spec for the main architecture lanes:
   - recurrence + LoRA
   - recurrence + QAT
   - recurrence + MQA/GQA
   - tokenizer lane held behind stricter validation
2. Build a canonical run-output contract around the repo's existing training and roundtrip evaluation path.
3. Use AutoResearchClaw to generate experiment matrices and drive the REFINE / PIVOT loop.
4. Escalate only top candidates to repeated runs and significance validation.
5. Package wins as `records/...` submissions with machine-assisted but human-auditable writeups.

### Immediate experiment priorities

Based on the PR deep dive and the synthesis above, the most credible first stack is:

- moderate recurrence
- low-rank loop-specific specialization
- compression-aware or QAT-style training
- lighter KV parameterization where it preserves quality

Tokenizer work should stay gated until we have a deterministic validator that reproduces baseline `val_bpb` and catches accounting regressions.

### Main risks

The first risk is **orchestrator overreach**. If AutoResearchClaw is allowed to mutate shared training code freely, the workflow becomes harder to audit and much less aligned with how this repo appears to accept contributions.

The second risk is **optimizing the wrong metric**. Any loop that ranks runs on float loss or pre-quantization quality without preserving post-roundtrip `val_bpb` is misaligned with the challenge.

The third risk is **over-bundling ideas**. Recurrence, LoRA, QAT, tokenizer changes, serialization tweaks, and optimizer changes can stack, but the pipeline still needs ablations or it will be impossible to tell which component is carrying the result.

The fourth risk is **tokenizer correctness drift**. Because tokenizer changes touch both compression and accounting, they need stronger evidence and better regression tests than ordinary architecture changes.

### Bottom line

The strongest current interpretation of Parameter Golf is: compress the structured part of language as effectively as possible, preserve that structure through quantization and serialization, and automate empirical search without surrendering the benchmark-critical execution path to uncontrolled code generation.

That makes **AutoResearchClaw the right research orchestrator** and **Parameter Golf the right execution substrate**. The winning workflow is not "let the agent rewrite everything." It is "let the agent search, synthesize, schedule, analyze, and document, while the repo-owned training/eval path remains deterministic and submission-ready."
