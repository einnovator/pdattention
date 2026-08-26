# Paper 3.1 — Codex continuation contract

## Mission
Develop **Paper 3.1: Generated Summaries as Retrieval Indices for Progressive Retrieval Attention**. A small model generates a compact persistent natural-language routing index for each source chunk; PRA still materializes and consumes the chunk's **unchanged native K/V**. The summary is an address, not replacement evidence.

## Read first
Inspect the latest papers/artifacts for 2.5 iterative PRA, 2.6 lexical/hybrid routing, 2.7 graph query facets, 2.8 low-rank native-QK compression, 2.9 temporal query windows, and Paper 3 native-K/V materialization. Also inspect `GistIndex`, lexical sidecars, routing provenance, HF adapters, dataset loaders, frozen feature artifacts, and Paper-2.8 checkpoints. Reuse routing/materialization code rather than forking it.

## Constraints inherited from prior results
- 2.5: iterative closure did not reliably improve natural multihop retrieval and can multiply comparisons. Do not start with iterative summary traversal.
- 2.6: lexical and semantic channels are complementary. Include BM25/exact-token controls and channel-overlap analysis.
- 2.7: faceting is not automatically useful. Treat faceted summaries as a falsifiable index-format hypothesis.
- 2.8: learned low-rank QK improves some semantic cohorts; BM25 remains stronger on lexical cohorts; better discovery does not guarantee better frozen-consumer use. Reuse rank-16 and compact rank-8/eight-centroid baselines where possible.
- 2.9: temporal query pooling did not improve held-out retrieval; do not combine it with summaries before the basic summary gate passes. Slower routing-clock results may later inform query-time cost.
- Paper 3: conceptual selection and physical native-K/V materialization are distinct. Preserve this boundary exactly.

## Branch/layout
Suggested branch: `research/paper3-1-summary-index`.
Create `docs/papers/paper3_1_summary_index/{paper_3_1.tex,AGENTS.md,README.md}`, `experiments/paper3_1_summary_index/`, and `docs/papers/shared/results/paper3_1_summary_index/`.

## Phase 0 — provenance and parity
Record exact source branches/commits for Papers 2.5--3. Reproduce dataset identities, chunk boundaries, evidence mappings, mean-gist/BM25 metrics, and Paper-2.8 selectors where reusable. Assert summary experiments never rewrite native K/V or materialization policy. Save model/tokenizer revisions, prompts, generation parameters, seeds, dataset hashes, and artifact hashes. Stop if parity fails.

## Phase 1 — frozen summarization feasibility
Implement an ingestion sidecar with stable `(URI, chunk_id)` alignment:
`chunk -> source text -> generated summary/facets -> summary-index record`.
Cache summaries so generation is paid once.

Test practical local/licensable sizes around: 0.1--0.2B diagnostic lower bound; 0.5--0.6B; 1--2B; 3--4B; 7--8B teacher/headroom. Mandatory logical comparison: tiny student, ~1--2B candidate, strong teacher. For each, generate bounded deterministic generic, retrieval-oriented, and (where useful) faceted summaries.

## Phase 2 — scorer decomposition
Summary content and scorer are separate variables. Implement where practical: BM25 over summary text; exact/token lexical score; frozen embedding score; PRA-compatible semantic score; validation-frozen hybrid. Do not attribute a gain to summarization if it is actually caused by a stronger external scorer.

## Phase 3 — primary matched-selection study
Reuse identity-disjoint HotpotQA, QASPER, 2Wiki, and MuSiQue cohorts. Prefer the inherited four-chunk endpoint. Compare mean gist, multi-representative, BM25, prior hybrid, Paper-2.8 rank-16, Paper-2.8 compact rank-8/eight-centroid, generic summary, retrieval-prompt summary, teacher summary, and non-compressed source-text control.

Primary metric: evidence recall at matched selected-chunk budget. Also report complete recovery, precision, reciprocal rank, channel overlap, and unique evidence. Use paired bootstrap intervals over identities, not overlapping windows or multiple summaries as independent samples.

## Phase 4 — matched persistent-index budget
Compare summary token/byte budgets against compact vector indices. At minimum: 1x32, 2x16, 4x8, 8x4 summary/facet tokens. State FP32/FP16/etc. assumptions. Keep source/native-KV backing storage separate from routing-index storage. Measure bytes/chunk, ingestion time, routing time, peak ingestion memory, query-time memory, and native K/V materialized.

## Phase 5 — omission benchmark and controls
Construct chunks with a salient main theme plus low-salience facts targeted by held-out queries. Track retention of entities, aliases, relations/events, dates/numbers, and rare lexical strings. Add shuffled-summary, entity/rare-term extractive, salient-sentence extractive, oracle-identity, summary-only-answer, correct-identity/native-source, and teacher-headroom controls.

## Gate A — train or stop
Train only if frozen retrieval-prompt summaries show a positive held-out signal OR teacher summaries materially beat tiny summaries and establish headroom. If teacher summaries fail, do not scale students; diagnose the summary-index concept instead.

## Phase 6 — LoRA distillation
Use training identities only. Teacher targets should preserve future retrieval affordances rather than prose elegance. Begin with a small LoRA rank and narrow attention projections; expand only with evidence of capacity limitation. Keep generic-summary LoRA as a control against retrieval-summary LoRA. Never expose held-out questions/evidence during summary creation or training.

## Phase 7 — optional retrieval-aware objective
Only after distilled retrieval summaries work, test retrieval-aware preference/ranking or auxiliary representation losses. End-to-end differentiable discrete generation is not required. Do not let this phase delay a clean Paper-3.1 result.

## Phase 8 — native-K/V causal confirmation
Only for discovery conditions passing gates, map selected identities through Paper-3 materialization unchanged. Compare answer likelihood/accuracy/generation against mean-gist, lexical/hybrid, Paper-2.8, and oracle identity. Also run summary-only answering to demonstrate the distinction between routing index and evidence.

## Cost accounting
For N chunks and Q future queries report one-time summarization cost, persistent index bytes, query-time routing cost, native materialization/attention cost, and amortized cost at Q={1,10,100,1000}. External/CPU generation is allowed but not free. Compute break-even versus vector-only routing.

## Claim rules
Never call summaries native-K/V compression, lossless memory, or active-K/V compression. Never infer end-to-end answer quality from retrieval recall. Keep logical memory, requested identities, routing-index residency, backing state, and materialized native K/V separate. Validation selects prompts/scorers/hyperparameters; held-out identities are opened once for headline effects.

## Deliverables
Track row-level metrics, paired effects/CIs, manifests, prompts, summaries with provenance, omission audits, cost tables, and publication plots. Update `paper_3_1.tex` only from measured results. Compile PDF and visually inspect it. Run relevant tests plus new summary-index tests. Preserve unrelated files. Commit and push only the Paper-3.1 branch when the experiment state is internally consistent.
