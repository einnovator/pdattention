# Proposed publication review — Paper 1

Reviewed 2026-09-11 against fetched branch `main` at `b50249700e5e764ef009599c6c79219a4cdb2da5`, equal to `origin/main`. Source: [paper.tex](paper.tex). This includes the September 1 pretrained context-dilution addition. Source-based review with selected JSON/TeX evidence checks; no training/inference rerun or PDF layout certification. Line anchors refer to this commit.

## Assessment and independent contribution

**Major revision, with a credible bounded-mechanism paper already present.** The latest rewrite is substantially more focused than the historical manuscript. Its strongest contribution is an explicit implementation contract: contextual encoding produces native state; fine routing views select original slices; one attention operation admits only a bounded payload. The fragmentation controls and separate fixed-breadth/fixed-fraction analyses are useful.

The strongest reviewer objection is that bounded payload size follows from imposing a budget, while existing latent-memory and K/V-retrieval systems already select small working sets. The paper therefore needs to establish what is preserved under that budget, how contextualization changes the result, and what the complete cost is. Do not make nominal boundedness carry the novelty claim by itself.

**Proposed main claim:** “Separating contextual encoding from routing granularity enables sparse access to stored native K/V under an explicit attention budget; controlled counterfactuals identify when the selected state retains useful information.”

## Priority revisions

### P0 — Resolve the definition of “native operation length”

**Locations:** lengths at 206–215; budget at 297–306; bounded experiment at 558–605.

The paper defines native operation length as participating K/V tokens, then reports that every native operation is at most 20 tokens while allowing 8 direct plus up to 20 selected memory tokens. A source-encoding call of 20 tokens does not by itself prove that the combined attention K/V axis is at most 20. This may be a diagnostic-definition mismatch rather than a broken bound.

Inspect the stored per-call traces and explicitly report maximum source input length, direct query length, actual retained memory length, and combined attention K/V length. If the final axis reaches 28, retain the valid <=32 claim and correct “largest operation 20.” If it never reaches 28, explain the actual admission outcome. State whether the four-token reserve is unused capacity or counted state. This affects the abstract, figure caption, and claimed logical/native ratios.

### P0 — Explain perfect task benefit with incomplete target coverage

**Locations:** scale table at 485–504; interpretation at 541–545.

At 256 units the small model has RCB 1.000 but target coverage .700/.759. Redundant or correlated evidence is a possible explanation, not yet an identified cause. Historical slicing may also allow the answer to influence later token states whose own URI lacks the annotated target. Then target-URI recall and information availability measure different things.

Use existing traces to stratify answer loss by target hit/miss. Inspect where the explicit answer-code anchor sits relative to selected historical states. A decisive control changes/removes the answer-bearing source before cache construction and rebuilds all affected downstream state; changing only one cached chunk cannot exclude contextual information propagation. Alternatively, keep the present evidence and explicitly limit the result to recovery on an answer-coded probe. Do not claim semantic retrieval at natural QA difficulty.

### P0 — Restore enough experimental detail for independent reproduction

**Locations:** evaluation design, 324–354; abbreviated reimplementation guide, 890 onward.

The compressed draft gives broad model sizes and default sample counts but no complete main-experiment configuration table. Add layers, width, heads, tokenizer, training objective and duration, split identities, model selection rule, exact seed list, source construction, position policy, gist/query reducer, and artifact per experiment. Distinguish repartitioning a fixed source into 256 units from increasing source-token length. Candidate units near one token are not evidence of retrieval across 256 independent documents.

No new training is needed to recover these facts from the existing configuration/results files. Include model and data setup in the supplement if necessary, with an explicit main-text pointer.

### P1 — Correct quantitative and inferential wording

- **Lines 392–395:** the prose says F1 agrees with likelihood direction only at 14B/32B. The included `generated_context_dilution_table.tex` also gives 8B/8K `Delta LP=+1.192`, `Delta F1=+.021`: both favor full context. The 8B/32K point is the opposing-sign case. Correct the sentence against the generated table.
- **Lines 513 and 844:** “systems law” and “both scaling laws” overstate three sizes and a bounded-budget construction. Use “measured scaling behavior” or “observed fixed-budget trend.”
- **RCB, 343–353:** specify ratio of means versus mean of ratios, treatment of nonpositive/small denominators, and pairing. RCB above one need not be sampling noise; selected memory may actually yield lower loss than full context. Show raw losses beside ratios and uncertainty over independent units.
- **Transport table, 434–435:** the displayed Hotpot means produce an RCB near .998 rather than .996. This can arise from averaging individual ratios. Verify the aggregation, rather than changing the number from rounded inputs.

### P1 — Delimit the new pretrained calibration and shared evidence

**Locations:** abstract 64–70; pretrained section 356–406; limitations 806–810.

The addition is useful and must remain in this review's scope. However, selected identities are annotation based; it does not test a learned large-model router. Explain whether E2 is built from a selected-text context or sliced from the full distractor context, which positional layout is used, and how those paths differ from the standalone experiment. Sequence agreement is an output test, not proof that all internal tensors are exact.

The MLX summary in this branch is byte-identical to the summary in Paper 1.5's `mac_long_context` directory. Label this as a shared campaign with different analyses, not independent replication. Allocate context dilution and payload scaling here; coordinate/reset effects belong to Paper 1.5. Paper 2's consumer study must separately identify overlapping questions/checkpoints.

Distinguish all-layer K/V payload bytes from measured total unified-memory allocation. The 39.6-versus-4,608 MiB contrast is compelling representation accounting, but is not equivalent to an observed 116-fold whole-process memory reduction.

### P1 — Add the closest comparison and complete cost accounting

The related-work section already handles several neighboring families responsibly. Add a direct comparison to [RetrievalAttention](https://arxiv.org/abs/2409.10516), whose retrieval over stored K/V is closer than generic RAG. Include [CacheBlend](https://arxiv.org/abs/2405.16444) when discussing context lost during independent cache construction. Specify what PRA adds beyond these mechanisms.

For a systems-quality claim, the minimum useful baseline is the same selected evidence as text, with identical output length and both cold and repeated-use timing. Include publication, index build, routing, transfer, attention, and backing-store bytes. The existing 94–103x result against scalar Python should stay a local optimization result. A production benchmark is optional if production superiority is explicitly excluded.

## Readability and structure

Use mechanism -> controlled transport/fragmentation -> sparse scaling -> bounded execution -> pretrained calibration -> costs/limits. Moving the new pretrained section later would avoid introducing E0/E2 before the primary evidence and correct the introduction's outdated “rather than a production-scale language model” description. Define E0/E2 in words beside their first table.

Suggested result sentence: “At a fixed eight-chunk budget, the selected payload stays nearly constant while the source index grows. Task loss remains low on the controlled answer-code task, although the router does not always select the annotated source.”

Replace repeated assertions that selected payload is full-detail with one precise definition plus tests. Spend the saved space on source construction and cost denominators.

## Publication boundary and next pass

This paper owns the bounded transport contract, contextual-slicing intervention, and logical/active scaling experiment. It should not re-derive RoPE or make pretrained retrofit the central novelty. Independent publication remains plausible after those boundaries and the shared MLX campaign are disclosed.

Suggested audience: empirical ML architecture and memory systems. **TMLR** is a plausible claim-focused journal direction: its criteria emphasize convincing support for claims and contribution to knowledge, rather than requiring a leaderboard win. [Acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html). This is a fit judgment, not a promise of acceptance.

**Sol handoff:** resolve P0 accounting and protocol questions first; correct the F1 sentence and scaling-law wording; build the target-hit/miss analysis from frozen traces; then revise abstract and conclusion. If causal source rebuilding is not run, narrow the retrieval interpretation. Return a claim-to-artifact ledger and identify any remaining new experiment separately from completed edits.

Preflight: literal TeX inputs and cited keys resolve; no undefined cross-reference labels or duplicate labels were found. The generated pretrained table was inspected. Full numerical replication and rendered-PDF checks remain outstanding.
