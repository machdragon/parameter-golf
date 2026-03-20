# Run checklist (dedupe vs Parameter Golf SOTA READMEs)

Use this **before** spending harness budget on lanes that community records already explored (openai/parameter-golf `records/track_10min_16mb/`, Mar 2026 crosswalk).

## Open PRs on `openai/parameter-golf` (unmerged — not accepted solutions)

Pull requests that are still **open** are **not** merged into `main` and are **not** official accepted records. They only indicate what participants are **currently trying**.

- Treat them as **hypothesis hints** and **possible future** merges, not as evidence on par with merged READMEs in `records/`.
- Per-lane hints are appended in `case_matrix_v3_full.json` (v4.2+) under `external_evidence_notes` with the prefix **Open PRs (unmerged):**.
- Dominant **pending** themes (titles/descriptions, subject to change): int6 + MLP 3× + fp16 embed + STE int6 QAT + sliding eval + zstd; SWA/LAWA; MTP combined with that stack; vocab 4096/8192; depth recurrence / shared-heavy still in flight; int8 QAT ablation arguing overhead (aligns with merged FP16 QAT negative).

Re-scan open PRs periodically—the set and claims drift as authors update branches.

## Skip or require a new hypothesis

- **QAT (lane_3)** — FP16Embed record: full + late QAT not worth step overhead vs quant gap.
- **Naive LR-only sweeps (lane_12)** — Multiple records mined lower matrix/scalar/embed LRs, warmdown, Muon WD; only sweep LRs when tied to a **new** architecture or a **narrow** optimizer partition test.
- **Depth recurrence under 10 min (lane_1)** — FP16Embed: recurrence needs more steps than the wall-clock allows unless compute budget changes.
- **SwiGLU** — Not a formal lane; record notes ~45% slower on 8-GPU (net negative for step count).
- **lzma vs zlib (lane_6)** — FP16Embed: lzma worse for int8 weight payloads.

## Keep orthogonal

- **Eval protocol** — LoRA TTT README: doc boundaries + stride dominate metrics vs TTT alone; compare leaderboard numbers only with matching eval assumptions.
- **Sliding-window eval** — Ensure [PR #124](https://github.com/openai/parameter-golf/pull/124)-style final partial window if reusing sliding eval code.

## Still high value vs README gaps

- LAWA / MTP / value-from-x0 / U-Net skips / tokenizer lane — not primary themes in the six READMEs; see per-lane `external_evidence_notes` in `case_matrix_v3_full.json`.
