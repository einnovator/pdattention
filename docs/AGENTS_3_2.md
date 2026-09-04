# AGENTS.md â€” Paper 3.2
## RAG, PRA, and Composable Native Memory: Retrieval, Materialization, Ordering, and Positional Geometry

### Status
Inception plan. This paper is NEW work.

Paper 3.2 may reuse measured preliminary RAG results and infrastructure from Paper 4.5, but it owns a different scientific question and must rerun/qualify the decisive comparisons under a frozen Paper-3.2 protocol.

Do not modify Paper 3.1 claims or reopen its frozen summary-index conclusions merely to support this paper.

---

# 1. Mission

Paper 3.2 asks:

> Across realistic retrieval-augmented generation workloads, when should a model consume all candidate documents directly, conventional RAG-selected text, or PRA-selected/native memory; and when retrieved resources are independently stored, how should their order, positional coordinates, and materialization policy be defined?

The paper unifies four questions that must be experimentally separated:

1. **Retrieval/indexing** â€” which external search/index strategy finds useful evidence?
2. **Routing/selection** â€” after candidates are available, which chunks/resources should be selected?
3. **Materialization/realization** â€” how much selected evidence should be exposed as text or native K/V?
4. **Composition geometry** â€” when several independent resources are consumed together, what ordering and positional frame should they occupy?

The practical target is RAG. The mechanistic target is composable persistent transformer memory.

---

# 2. Boundary with adjacent PRA papers

## Paper 1.5
Owns correct source-relative positional transport, logical offsets, RoPE rebinding algebra, deferred/pre-RoPE machinery, and the distinction between positional and contextualization fragmentation.

Paper 3.2 reuses that machinery. Do not re-prove RoPE algebra except where needed to define composition policies.

## Old Paper 1.6 plan
The abandoned/not-executed Paper-1.6 multi-frame plan is incorporated into Paper 3.2 as the positional-composition experimental program:
- source/global vs packed coordinates;
- resource-adjacent overlapping frames;
- rank/score-derived distance;
- near bands;
- random-distance negative control;
- D1,D2 vs D2,D1/permutation sensitivity;
- source-distance sensitivity;
- distractor sensitivity;
- multi-resource scaling.

Paper 1.6 therefore remains historical planning context, not a competing active paper.

## Paper 3.1
Owns compact retrieval-address representations, especially generated summaries vs lexical/extractive/QK indices. Its frozen result remains independent.

Paper 3.2 may use lexical, embedding, QK, hybrid, and optionally summary indices as routing arms, but must not turn into another summary-index paper.

## Paper 3.0
Owns the causal native-K/V materialization question after selection has been
fixed. It distinguishes human-readable evidence from the contextual native
states that are computationally sufficient for a frozen model, and separates:

```text
encoding granularity != search granularity != materialization granularity
```

Paper 3.2 reuses its logical source intervals, per-domain interval union,
cross-shard gather, fixed-budget allocation, evidence-density accounting, and
matched wrong-memory controls. Whole selected parents remain a local physical
control, not an oracle optimum. Routing gists remain addresses, not sufficient
answer detail.

Inherited Paper-3.0 measurements are background rather than new RAG evidence:
- controlled exact cores averaged 5.25 K/V tokens and used 92.9% less K/V than
  whole selected parents;
- under oracle selection, exact disclosure reduced active K/V by 20.9% on
  MuSiQue and 44.4% on 2Wiki relative to whole-parent disclosure;
- fixed budgets worked only when all required evidence regions remained
  covered, and a common 128-token budget did not transfer cleanly to MuSiQue.

Paper 3.2 must therefore report evidence coverage together with density and
token savings, freeze selected identities before varying materialization, and
include equal-budget wrong-memory and whole-parent controls where labels allow.

## Paper 4.5
Owns runtime/product synthesis, engine integration, portable PRA semantics, deployment modes, and cross-engine qualification.

Paper 3.2 may inherit:
- RAG harness code;
- candidate/selection receipt schemas;
- MultiHop-RAG work;
- strong-reranker controls;
- Selected Context vs Native Memory modes;
- warm/cold/native cache instrumentation;
- preliminary RAG results.

Paper 3.2 owns the controlled scientific comparison of RAG/PRA retrieval, materialization, ordering, and positional composition.

---

# 3. Preliminary inherited evidence from Paper 4.5

Treat these as PRELIMINARY / INHERITED, not final Paper-3.2 results.

A strong conventional reranker was used to freeze selected chunks. Under identical selection receipts:
- strong-reranker Selected Context and corresponding PRA Selected Context produced identical generated outputs;
- Native Memory reuse of the same contiguous selected block also preserved generated outputs.

Representative warm Qwen3-4B measurements:
- Selected Context: mean total latency ~3.031 s, mean TTFT ~2.640 s;
- Native Memory: mean total latency ~0.638 s, mean TTFT ~0.166 s.

Representative 4K Qwen3-4B:
- Selected Context warm ~6.121 s;
- Native Memory warm ~0.799 s.

Representative Qwen3-8B:
- Selected Context warm ~5.541 s;
- Native Memory warm ~0.990 s;
- TTFT ~4.908 s -> ~0.276 s.

A changing-selection persistent-chunk experiment showed partial physical reuse:
- ~22% mean native chunk-hit fraction;
- 101,294 selected tokens -> 80,117 uniquely materialized native tokens;
- cumulative runtime ~151.19 s -> ~131.78 s (~12.8% reduction).

But independently cached/recomposed chunks changed output quality:
- Selected Context official score ~0.440;
- persistent independently cached native chunks ~0.380.

Current interpretation:
- exact reuse of the same contiguous selected native block is qualified;
- recomposition of independently contextualized chunks is not quality-qualified;
- source-local vs composition-position mismatch is a leading causal hypothesis;
- contextualization fragmentation may remain after positional rebinding.

Paper 3.2 must reproduce the critical rows before promoting them to headline evidence.

---

# 4. Primary hypotheses

## H1 â€” Full-document context is a reference, not automatically an upper bound
When all candidate documents fit the native context window, Full-Doc-In-Context gives a direct reference for model behavior without retrieval omission.

RAG or PRA may outperform it if retrieval removes distractors/context dilution.

Therefore never describe Full Context as a guaranteed quality ceiling.

## H2 â€” Strong conventional RAG and PRA are complementary
A mature BM25/vector/hybrid/reranker stack may remain stronger than generic PRA routing.

A viable architecture is:

```text
external retriever/reranker
-> selected evidence
-> PRA Selected Context / Native Memory
-> persistent reuse
```

The paper must separately test PRA as:
- a replacement selector;
- a second-stage selector;
- a realization/native-memory layer after conventional RAG selection.

## RAG+PRA system design and profile matrix

Treat RAG+PRA as an explicit composed system, not as a loosely named condition:

```text
query + corpus revision
-> external retrieval/index (BM25, dense/FAISS, hybrid, reranker, or service)
-> immutable candidate receipt
-> optional PRA second-stage routing
-> immutable selection receipt
-> materialization and positional-composition policy
-> Selected Context or Native Memory realization
-> persistent cache/reuse
-> generation + quality/system receipts
```

The external retriever and the PRA stage have independent contracts. The
candidate receipt freezes source identities, spans, scores, ranks, backend,
and index revision. The selection receipt freezes the evidence used by every
matched realization. A PRA realization-only condition MUST consume the same
selection receipt as its conventional-RAG control and MUST NOT rerun retrieval.

Use these canonical composed profiles:

| Profile | Selector | Realization | Purpose |
| --- | --- | --- | --- |
| `RAG_ONLY_TEXT` | external RAG | freshly packed visible text | strong conventional baseline |
| `RAG_PLUS_PRA_SELECTED` | external RAG, optionally PRA-refined | freshly packed Selected Context | isolate PRA routing without native transport |
| `RAG_PLUS_PRA_NATIVE_CONTIGUOUS` | frozen external selection | one reusable packed native block | isolate native reuse under exact packed geometry |
| `RAG_PLUS_PRA_NATIVE_INDEPENDENT` | frozen external selection | independently cached source-position K/V | expose recomposition error |
| `RAG_PLUS_PRA_NATIVE_REBOUND` | frozen external selection | independent K/V rebound to composition positions | isolate positional repair |
| `RAG_PLUS_PRA_REPAIR` | frozen external selection | rebound K/V plus bounded contextual repair | measure the minimum recomputation needed for parity |

For every external backend, first compare `RAG_ONLY_TEXT` with the
realization-only native profiles. Then enable PRA second-stage routing and,
separately, PRA replacement routing. This yields three interpretable claims:

1. **transport/reuse:** same evidence, different representation;
2. **refinement:** same candidates, external retrieval followed by PRA routing;
3. **replacement:** PRA constructs the selection without an external ranker.

The roadmap must report these three claims independently. A gain in one stage
must not be used as evidence for another stage.

## H3 â€” Materialization has a quality/economics frontier
Full materialization should maximize evidence exposure but may spend more tokens/compute.
Partial materialization may retain quality with lower active context.

Measure the frontier rather than assuming one policy dominates.

Use Paper 3.0's logical-interval semantics. Deduplicate overlap only within the
same source domain; never merge equal numerical offsets from different
resources. Report requested pre-dedup tokens, unique materialized tokens,
evidence tokens, non-evidence tokens, evidence density, shards touched, and
cross-shard intervals. A fixed token budget is invalid as a sufficiency claim
when it drops one of several required evidence regions.

## H4 â€” Serialization order can create artificial semantics
For independent retrieved resources, ordinary RAG forces a total sequence:

```text
D1 -> D2 -> ... -> Dn -> Q
```

even when inter-document order is arbitrary.

Prediction sensitivity under D1,D2 vs D2,D1 may therefore reflect positional serialization rather than evidence content.

## H5 â€” Source position and composition position are distinct
Each persistent resource has source-local coordinates, but a current RAG composition may assign new effective positions.

Conceptually preserve:

```text
logical identity
physical residency
source positional frame
request-specific composition frame
```

Do not infer one from another.

## H6 â€” RoPE rebinding may repair only part of independent-chunk recomposition
Composition-coordinate rebinding can repair positional phase, but cannot recreate cross-resource hidden-state contextualization that did not exist during independent encoding.

The residual fresh-concat vs rebound-native gap estimates contextualization fragmentation.

---

# 5. Dataset ladder

Use multiple RAG/long-context datasets. Freeze revisions and example identities.

## Tier A â€” fast development
- controlled synthetic multi-document facts;
- small held-out subsets from natural datasets.

Purpose: mechanism debugging, exact receipts, ordering/position causal isolation.

## Tier B â€” inherited PRA QA datasets
- HotpotQA;
- QASPER;
- 2WikiMultiHopQA;
- MuSiQue.

These connect Paper 3.2 with Papers 1.xâ€“3.x and provide multi-hop/long-document structure.

## Tier C â€” flagship RAG datasets inherited/planned from Paper 4.5
- MultiHop-RAG â€” mandatory flagship;
- KILT Natural Questions;
- KILT HotpotQA.

Do not begin a large KILT campaign before the small-model MultiHop-RAG protocol is stable.

For every dataset document:
- corpus construction;
- query split;
- gold/supporting evidence where available;
- whether all candidate documents fit Full-Doc-In-Context;
- chunking;
- embedding/index build;
- official answer metric.

---

# 6. Model scaling ladder

Start small. Prove mechanics before spending large-model compute.

## M0 â€” mechanism/debug
Primary:
- Qwen3-0.6B.

Secondary cross-family where practical:
- Llama 3.2-1B.

Use for the broadest policy sweeps.

## M1 â€” small practical
After M0 protocol/correctness gates:
- ~1â€“2B model;
- optionally one Gemma-family checkpoint with validated attention/RoPE topology.

Reduce weak arms before scaling.

## M2 â€” mid-size
- Qwen3-4B â€” mandatory because Paper 4.5 preliminary RAG evidence exists;
- Qwen3-8B â€” first scaling replication.

## M3 â€” larger confirmation
Only strongest conditions:
- Qwen3-14B;
- optionally 32B if compute permits;
- at least one non-Qwen family if feasible.

Do not rerun the full combinatorial grid at every size.
Use small models for discovery, mid-size for qualification, larger models for replication.

At every scale preserve:
- same cohort where token/window constraints allow;
- same retrieval receipts;
- same generation settings;
- same condition semantics.

---

# 7. External retrieval/index families

Retrieval is a first-class factor. Keep candidate retrieval distinct from PRA internal routing.

## 7.1 Custom/local baselines

### Lexical
- BM25 with a pinned implementation;
- simple exact/token lexical baseline where useful.

### Dense semantic
Interpret the requested "FAS" as **FAISS** unless repository context proves otherwise.
Use:
- one pinned embedding model;
- exact flat index as a small-corpus reference where feasible;
- FAISS ANN configuration for scalable local semantic retrieval.

### Hybrid
Combine lexical+dense under a predeclared method:
- RRF preferred as the first robust control;
- optionally normalized weighted fusion.

Do not tune fusion on test data.

### Strong reranker
Use a frozen cross-encoder/strong conventional reranker as a high-quality conventional RAG baseline and as a selector-frozen input to PRA realization.

Record candidate top-N before reranking and selected top-k after reranking.

## 7.2 Real service/vector-store integrations

Run services on the remote machine via Docker/Compose where practical.

Required initial service set:
1. Elasticsearch / OpenSearch-compatible vector+BM25/hybrid path;
2. Qdrant;
3. one of Weaviate or Milvus;
4. pgvector if Postgres deployment is cheap enough.

The user also requested **Derby**. Before implementation:
- identify the exact intended product/project and version;
- do not silently treat Apache Derby as a vector ANN database;
- if it is a specific RAG/vector service, add it as a pinned adapter;
- otherwise record `BACKEND_IDENTITY_UNRESOLVED` and continue with the validated service matrix.

Do not turn Paper 3.2 into a vector-database benchmark.
The purpose of real services is to test whether conclusions survive realistic retrieval infrastructure.

For each backend record:
- version/container digest;
- index type and parameters;
- embedding model;
- BM25/analyzer settings;
- hybrid/reranker path;
- filtering if used;
- index build time;
- query latency;
- candidate IDs/scores;
- recall@k / MRR / nDCG where supported by labels.

---

# 8. Canonical evaluation conditions

Every condition must have an explicit ID and receipt.

## 8.1 Full-context references

### FULL_DOC_IN_CONTEXT
Put all candidate documents in the model prompt in canonical order when they fit.

No retrieval pruning.

### FULL_DOC_PERMUTED
Same documents/tokens, changed irrelevant document order.

Used for order sensitivity.

### FULL_SELECTED_CONTEXT
Freeze a selected set of chunks and freshly concatenate all selected text.

This is the critical matched-content reference for PRA Native Memory.

---

# 8.2 Conventional RAG

### RAG_BM25
BM25 candidate retrieval and standard text packing.

### RAG_DENSE_FAISS
Dense semantic retrieval through FAISS.

### RAG_HYBRID
Lexical+dense hybrid.

### RAG_STRONG_RERANK
Candidate retrieval + strong conventional reranker.

### RAG_SERVICE_<BACKEND>
Equivalent service-backed paths, e.g. Elasticsearch/Qdrant/Weaviate/Milvus/pgvector.

For fair realization comparisons, freeze the final selection receipt and reuse it across Selected Context and Native Memory.

---

# 8.3 PRA routing/search conditions

PRA can start from:
- entire corpus where tractable;
- externally retrieved candidate documents;
- strong-reranker-selected resources.

Required routing families:
- lexical;
- native semantic/gist;
- rank-16/native-QK where available;
- lexical+semantic hybrid;
- strongest inherited generic PRA profile;
- optional summary index only as a secondary inherited Paper-3.1 arm.

Do not let each routing mode see a different candidate corpus unless that difference is the explicit independent variable.

---

# 8.4 PRA materialization conditions

At minimum compare:

### PRA_FULL_MATERIALIZATION
After PRA selects resources/chunks, materialize all tokens for selected objects.

Purpose:
separate routing from detail truncation.

### PRA_PARTIAL_FIXED_K
Materialize a fixed chunk/token budget.

### PRA_PARTIAL_SCORE
Materialize until a score/coverage threshold or token budget is reached.

### PRA_PARTIAL_HIERARCHICAL
Use hierarchical selection/materialization where existing implementation supports it.

### PRA_SELECTED_CONTEXT
Selected evidence is presented as ordinary freshly encoded text.

### PRA_NATIVE_CONTIGUOUS
Selected evidence is encoded once as one contiguous native block and then reused.

### PRA_NATIVE_INDEPENDENT
Selected chunks/resources are independently cached and recomposed.

Always report:
- logical candidate tokens;
- selected tokens;
- visible text tokens;
- native materialized tokens;
- newly materialized tokens;
- reused native tokens;
- selected/full ratio.

---

# 9. Selection Ã— realization decomposition

This decomposition is mandatory.

For the SAME frozen selected chunks compare:

```text
Fresh Selected Context
PRA Native contiguous reuse
PRA Native independent/source-position
PRA Native independent/composition-rebound
PRA Native rebound + optional contextual repair
```

The selection receipt must be byte-identical across these conditions.

This separates:
- retrieval quality;
- PRA selection quality;
- native realization fidelity;
- cache/reuse economics.

Do not attribute No-RAG -> Native-Memory deltas to Native Memory alone.

---

# 10. Composition coordinates and positional policies

Reuse validated Paper-1.5 rebinding machinery.

For resource r and token i:

```text
source position:       p_src(r,i)
composition position:  p_cmp(r,i)
```

A general resource-preserving transform may be written:

\[
p'_{r,i} = b_r + p_{r,i}.
\]

Required policies:

## P0 SOURCE_LOCAL
Preserve each resource's source-local/source-frame positions.

This is not automatically equivalent to fresh RAG packing.

## P1 GLOBAL_PACKED
Assign selected resources contiguous positions matching the ordinary selected-text composition.

This is the primary parity hypothesis.

## P2 RESOURCE_ADJACENT
Preserve internal resource geometry, but place each resource independently near the query. Ranges may overlap.

## P3 RANK_DISTANCE
Map retrieval rank to distance from the query.

## P4 SCORE_DISTANCE
Map normalized retrieval score to distance under a frozen mapping.

## P5 NON_OVERLAPPING_NEAR_BANDS
Use bounded nearby bands while preserving monotonic resource order.

## P6 RANDOM_DISTANCE
Randomized matched-envelope offsets; negative control.

Optional only after P0â€“P6:
- clipped/log distance;
- order-only/coarse distance;
- task/type-specific topology.

Do not promote any positional policy to runtime default from synthetic-only evidence.

---

# 11. D1 D2 vs D2 D1 and permutation experiment

This is a primary experiment, not an appendix ablation.

For a fixed evidence set:
- D1,D2,Q;
- D2,D1,Q.

For 3+ resources sample multiple permutations:
- canonical retrieval order;
- reverse;
- random permutations;
- relevance-ranked;
- where labels allow, causal/logical evidence order.

Two experiment classes must remain separate.

## 11.1 Serialization/composition sensitivity
Freeze:
- resource identities;
- resource content;
- selected spans;
- token budget.

Change only order and/or positional policy.

Do NOT rerun retrieval.

## 11.2 Retrieval/ranking sensitivity
Allow each retrieval strategy to produce its natural rank/order and packing.

This measures the complete RAG pipeline and must not be confused with pure composition sensitivity.

Metrics:
- task score;
- NLL;
- target probability;
- answer flip rate;
- target-probability variance;
- mean pairwise JS divergence across permutations;
- KL secondary;
- generated-output exact agreement.

Define an order-sensitivity summary, for example:

\[
S_{\rm order} =
\operatorname{mean}_{a<b}
JS(P_a \Vert P_b).
\]

Keep per-example distributions; do not rely only on a mean.

---

# 12. Source-distance, distractor, and multi-resource experiments

Inherited from the old Paper-1.6 plan.

## Source distance
Hold evidence content fixed, vary source/global distance from q.

Measure:
- accuracy/NLL;
- target probability;
- evidence attention;
- answer flips;
- slope vs distance.

## Distractors
Sweep irrelevant selected resources:
- 0, 2, 4, 8, 16, 32 where context allows.

Test whether query-adjacent/overlapping frames make distractors over-salient.

## Relevant resource count
Sweep independently relevant resources:
- 1, 2, 4, 8.

Include:
- redundant evidence;
- complementary multi-hop evidence;
- conflicting evidence.

---

# 13. Positional vs contextualization decomposition

For a frozen selected sequence A,B,C,Q:

## C0 FRESH_PACKED
Freshly encode [A B C Q].

## C1 NATIVE_SOURCE
Independently cached native A/B/C with source-local positions.

## C2 NATIVE_REBOUND_PACKED
Same independently cached hidden/KV states, but RoPE K is rebound to the positions A/B/C occupy in the packed composition.

Use validated Paper-1.5 pre/post-RoPE algebra.

## C3 NATIVE_REBOUND_REPAIR
C2 plus selective contextual recomputation.

## C4 FULL_RECOMPUTE
Equivalent to fresh selected text / full reprefill; correctness ceiling for the repair sweep.

Primary diagnostics:
- logit RMSE;
- KL/JS;
- top-1 parity;
- generated sequence parity;
- official task score.

Approximate positional recovery may be summarized as:

\[
R_{\rm pos}
=
1 -
\frac{D(P_{\rm fresh}, P_{\rm rebound})}
     {D(P_{\rm fresh}, P_{\rm source})},
\]

with a clearly specified divergence D.

Do not imply rebinding can reconstruct missing cross-resource contextualization.

---

# 14. Contextual repair ladder

Only if C2 does not recover sufficient quality/parity.

Compare limited repair against full reprefill.

Candidate policies:
- first N tokens of each appended chunk;
- boundary windows;
- fixed fractions: 5%, 10%, 25%, 50%;
- selected layers;
- attention/change-triggered tokens if a simple deterministic signal is available.

Primary curve:

```text
quality/parity recovered
vs
fraction of selected tokens/layers recomputed
vs
TTFT/prefill cost
```

Do not over-engineer adaptive repair before fixed-fraction controls are understood.

---

# 15. Repeated-query / session RAG

After one-query mechanics are stable, evaluate partially overlapping selections:

```text
Q1 -> A,B,C
Q2 -> B,C,D
Q3 -> A,D,E
...
```

Compare:
1. conventional RAG full reprefill;
2. conventional engine prefix/KV caching where applicable;
3. PRA contiguous exact-block reuse;
4. PRA independent-chunk reuse/source positions;
5. PRA independent-chunk reuse/packed rebinding;
6. rebound + repair.

Measure:
- selection overlap;
- native chunk-hit fraction;
- newly encoded tokens;
- reused tokens/KV bytes;
- cumulative TTFT;
- cumulative prefill;
- cumulative wall time;
- memory residency;
- official quality;
- output parity/flip rate.

This is the key practical test of PRA as a memory/runtime layer under existing RAG.

---

# 16. Prefix-cache / ordinary serving comparison

Paper 4.5 preliminary warm numbers are not sufficient without a strong ordinary-cache control.

Where engine support allows compare:

```text
RAG no cache
RAG + prefix cache/APC
RAG + semantic/exact prompt cache if available
RAG + PRA Native Memory
RAG + prefix cache + PRA
```

Keep exact-prefix reuse conceptually separate from query-addressed non-prefix PRA reuse.

Report cache hit tokens/blocks and whether changing document order destroys ordinary prefix reuse.

---

# 17. Metrics

## Quality
- official task metric: EM/F1/accuracy as appropriate;
- answer availability only as a diagnostic;
- NLL/gold logP;
- correct-answer margin;
- generated-output exact agreement;
- semantic judge only if official metrics are insufficient and judge is frozen.

## Retrieval
- document Recall@k;
- supporting-document coverage;
- evidence/span recall;
- MRR;
- nDCG where meaningful;
- candidate count;
- reranker displacement.

## Context/materialization
- total corpus/candidate tokens;
- full-context tokens;
- selected visible tokens;
- selected native tokens;
- newly materialized tokens;
- reused native tokens;
- materialization avoidance;
- active K/V bytes.

## Composition
- order flip rate;
- mean pairwise JS;
- KL;
- target-probability variance;
- source-distance slope;
- distractor sensitivity;
- rebound recovery.

## Systems
- index build time;
- retrieval p50/p95/p99;
- rerank latency;
- tokenization;
- prefill;
- TTFT p50/p95/p99;
- ITL;
- completion latency;
- tokens/s;
- requests/s;
- peak RAM/VRAM/unified memory;
- K/V bytes;
- H2D/storage traffic;
- warm/cold/hot state.

## Economic
Where defensible:
- GPU seconds/request;
- GPU seconds/correct answer;
- input/materialized tokens per correct answer;
- cumulative repeated-query compute.

Do not invent monetary cost for local hardware.

---

# 18. Statistical design

Use paired comparisons whenever conditions share query/evidence.

Requirements:
- split by example identity;
- frozen validation/test cohorts;
- bootstrap 95% CIs;
- paired effect distributions;
- multiple random permutations/seeds where randomness is part of the experiment;
- show per-model/per-dataset sign reversals;
- do not average away family/workload dependence.

For large grids:
1. discovery on M0;
2. freeze promising policies;
3. qualification on M2;
4. minimal replication on M3.

---

# 19. Fairness and receipts

Persist immutable receipts for:
- dataset/query identity;
- corpus revision;
- candidate retrieval;
- candidate scores;
- reranker scores;
- selected resources/chunks;
- selected order;
- token spans;
- materialization;
- positional policy;
- effective positions;
- model revision;
- tokenizer;
- generation configuration;
- backend/index version.

Required assertion:
Selected Context and Native Memory realization comparisons MUST use identical selection receipts.

For positional comparisons, selected contents MUST be identical and only positional policy/order may change.

---

# 20. Failure taxonomy

At minimum:

```text
FULL_CONTEXT_OVERFLOW
FIRST_STAGE_RETRIEVAL_MISS
RERANKER_MISS
RAG_PACKING_MISS
PRA_SELECTOR_MISS
PRA_DISTRACTOR_SELECTION
PRA_MATERIALIZATION_MISS
NATIVE_REALIZATION_MISMATCH
POSITION_COMPOSITION_MISMATCH
CONTEXTUALIZATION_FRAGMENTATION
GENERATION_FAILURE
ANSWER_FORMAT_FAILURE
CACHE_REUSE_MISS
BACKEND_IDENTITY_UNRESOLVED
```

Use one primary failure reason plus optional secondary tags.

---

# 21. Experimental execution order

## Phase 0 â€” inherit and audit
- pull Paper 4.5 RAG code/results;
- copy, do not silently mutate, preliminary artifacts;
- verify dataset/model/retriever revisions;
- label inherited results PRELIMINARY.

## Phase 1 â€” M0 full-document/RAG baseline
On Qwen3-0.6B:
- FULL_DOC_IN_CONTEXT;
- BM25 RAG;
- FAISS dense RAG;
- hybrid RAG;
- strong reranker;
- full selected-context control.

Use small natural cohorts plus synthetic.

Freeze candidate and selection receipts here. Run `RAG_ONLY_TEXT` and
`RAG_PLUS_PRA_NATIVE_CONTIGUOUS` first so the initial RAG+PRA result isolates
representation and reuse rather than changing retrieval.

## Phase 2 â€” M0 PRA routing/materialization
Cross:
- strongest routing families;
- full vs partial materialization;
- identical candidate receipts.

Prune clearly dominated arms.

Run the complete composed profile ladder. The second-stage and replacement
selector arms receive the same initial candidate set; all realization arms
within a selector condition receive the same selection receipt.

## Phase 3 â€” M0 positional composition
Run:
- SOURCE_LOCAL;
- GLOBAL_PACKED;
- RESOURCE_ADJACENT;
- RANK_DISTANCE;
- SCORE_DISTANCE;
- NEAR_BANDS;
- RANDOM_DISTANCE.

Run D1D2/D2D1 and permutation suite.

## Phase 4 â€” composition fidelity
Fresh packed vs source vs rebound packed.

If required, run repair ladder.

## Phase 5 â€” real service indices
Run remote Docker services:
- Elasticsearch;
- Qdrant;
- one of Weaviate/Milvus;
- pgvector if practical;
- requested Derby backend once exact identity is resolved.

Use same embedding/chunking/query cohort when backend capabilities permit.

## Phase 6 â€” Qwen3-4B powered qualification
Reproduce Paper-4.5 relevant rows and run only frozen best conditions.

Mandatory:
- MultiHop-RAG;
- Full-Doc when fit;
- BM25/dense/hybrid/strong RAG;
- strongest PRA selector;
- strong-RAG -> PRA Selected Context;
- strong-RAG -> PRA Native contiguous;
- independent native source;
- independent native rebound;
- partial-overlap repeated-query experiment.

## Phase 7 â€” Qwen3-8B replication
Reduced matrix, same decisive conditions.

## Phase 8 â€” larger/cross-family
Qwen3-14B and one non-Qwen model if compute permits.

Do not scale failed mechanisms.

## Phase 9 â€” KILT and broad natural validation
Only after protocol is frozen:
- KILT NQ;
- KILT HotpotQA;
- larger Hotpot/QASPER/2Wiki/MuSiQue cohorts as useful.

---

# 22. Primary tables

## Table A â€” End-to-end quality/context frontier
Rows:
- Full Doc;
- RAG BM25;
- RAG dense;
- RAG hybrid;
- strong RAG;
- PRA full materialization;
- PRA partial;
- strong RAG + PRA Selected Context;
- strong RAG + PRA Native.

Columns:
quality, evidence recall, visible/materialized tokens, TTFT, completion, K/V memory.

## Table B â€” Retrieval/index comparison
BM25, FAISS, hybrid, strong reranker, Elasticsearch, Qdrant, Weaviate/Milvus, pgvector, requested Derby if resolved.

## Table C â€” Matched realization
Fresh selected text vs contiguous native vs independent source vs rebound vs repair.

## Table D â€” Order/position robustness
Policy Ã— quality Ã— pairwise JS Ã— flip rate Ã— source-distance slope Ã— distractor sensitivity.

## Table E â€” Repeated-query reuse
No cache vs prefix cache vs PRA variants; quality plus cumulative cost.

## Table F â€” Scaling
0.6B -> 4B -> 8B -> 14B+ decisive rows.

---

# 23. Primary figures

1. Full Doc vs RAG vs PRA quality/context/latency frontier.
2. Retrieval/index recall vs latency.
3. PRA full vs partial materialization Pareto frontier.
4. D1D2 vs D2D1 and permutation divergence.
5. Diagram of source coordinates vs packed composition coordinates.
6. Quality by positional policy.
7. Fresh concat -> source native -> rebound native -> repaired native fidelity.
8. Quality vs contextual-repair fraction.
9. Partial-overlap session cumulative TTFT/runtime.
10. Model-size scaling of the best configurations.
11. Optional source-distance curve.
12. Optional distractor-count curve.

---

# 24. Claim hierarchy

## Strong positive A â€” PRA as RAG realization layer
Conventional retrieval/reranking + PRA Native Memory preserves quality while reducing repeated prefill/TTFT and enabling non-prefix evidence reuse.

## Strong positive B â€” PRA routing/materialization
A PRA routing/materialization policy improves the quality/context frontier over strong conventional RAG at matched evidence/token budget.

Do not claim this unless strong reranker/service baselines are beaten.

## Strong positive C â€” composable native memory
Independent resource K/V can be rebound/repaired into new compositions with near-fresh quality at substantially less recomputation.

## Strong positive D â€” multi-frame geometry
A non-global positional topology maintains or improves quality while reducing arbitrary serialization-order/source-distance sensitivity.

## Conservative
PRA is most useful under a mature external RAG retriever as a persistent realization/reuse layer; positional composition requires explicit request-specific semantics.

## Negative
Strong RAG plus ordinary serving caches dominate PRA, or independent native fragments require near-full recomputation, or pretrained models consistently prefer conventional packed global topology.

Negative findings remain publishable and must remain visible.

---

# 25. Falsification

Report explicitly if:
- Full Doc dominates every retrieval method whenever it fits;
- PRA selector loses to BM25/dense/strong reranking;
- partial materialization loses too much quality;
- Native Memory benefits vanish against prefix/APC caching;
- rebinding fails to recover the independent-chunk gap;
- repair requires nearly full reprefill;
- resource-adjacent/overlapping frames hurt quality;
- reduced order variance comes with worse accuracy;
- gains appear only on synthetic data;
- effects reverse by model family/size;
- real vector services materially change retrieval conclusions.

---

# 26. Implementation architecture

Keep adapters modular:

```text
Dataset/Corpus
  -> Chunker
  -> IndexAdapter
       BM25
       FAISS
       Elasticsearch
       Qdrant
       Weaviate/Milvus
       pgvector
       Derby? (after identification)
  -> CandidateReceipt
  -> OptionalReranker
  -> SelectionReceipt
  -> Realizer
       FreshText
       PRASelectedText
       NativeContiguous
       NativeIndependent
  -> PositionPolicy
  -> OptionalContextRepair
  -> Model
  -> Metrics/Artifacts
```

Core interfaces should allow a candidate set from any retriever to feed either conventional RAG or PRA.

No backend-specific candidate semantics in the evaluator.

---

# 27. Remote service deployment

Create a dedicated compose stack, e.g.:

```text
experiments/paper3_2_rag/docker/
  compose.yml
  elasticsearch/
  qdrant/
  weaviate_or_milvus/
  pgvector/
```

Pin image digests when feasible.

Remote machine records:
- CPU/RAM/GPU if used;
- OS;
- Docker version;
- storage;
- network path from model runner;
- service versions;
- index sizes.

Separate:
- retrieval service latency;
- network latency;
- LLM inference latency.

Do not compare local in-process FAISS latency directly with remote service latency without decomposition.

---

# 28. Required artifacts

Suggested structure:

```text
docs/papers/paper3_2_rag_composition/
  AGENTS.md
  paper_3_2.tex
  README.md

experiments/paper3_2_rag/
  datasets/
  retrieval/
  services/
  routing/
  materialization/
  position/
  repair/
  session_reuse/

docs/papers/shared/results/paper3_2_rag/
  manifest.json
  inherited_paper4_5_results.json
  full_context_results.csv
  retrieval_results.csv
  service_retrieval_results.csv
  pra_routing_results.csv
  materialization_results.csv
  realization_results.csv
  position_policy_results.csv
  permutation_results.csv
  source_distance_results.csv
  distractor_results.csv
  repair_results.csv
  session_reuse_results.csv
  scaling_results.csv
  failure_summary.csv
  canonical_evidence.json
  plots/
```

---

# 29. Minimum tests

- exact candidate receipt serialization;
- exact selection freeze between paired realization conditions;
- Full Selected Context token identity;
- source/composition position metadata;
- Paper-1.5 pre/post RoPE rebinding equivalence;
- GLOBAL_PACKED effective position audit;
- internal resource distance preservation;
- overlapping resource frames remain distinct K/V objects;
- decode/query positions remain correct;
- D1D2/D2D1 changes only declared ordering;
- service adapter candidate-ID consistency;
- cache invalidation by model/tokenizer/source revision;
- metric table regeneration from raw artifacts.

---

# 30. Definition of done

Paper 3.2 is ready for editorial consolidation when:

1. Qwen3-0.6B broad mechanism matrix is complete.
2. Full Doc, conventional RAG, PRA full materialization, and PRA partial materialization exist on multiple natural datasets.
3. BM25, FAISS dense, hybrid, and strong reranker are all measured.
4. At least two real remote retrieval services are measured, including Elasticsearch and one dedicated vector DB.
5. D1D2/D2D1/permutation experiments exist.
6. At least SOURCE_LOCAL, GLOBAL_PACKED, RESOURCE_ADJACENT, and a negative positional control are validated.
7. Fresh/source/rebound decomposition is complete.
8. If rebound is insufficient, at least one repair curve exists.
9. Qwen3-4B powered MultiHop-RAG qualification is complete.
10. Qwen3-8B reduced replication is complete.
11. Repeated-query partial-overlap RAG is measured.
12. Prefix/APC cache control exists on at least one production-style engine if practical.
13. Paper-4.5 inherited preliminary numbers are either reproduced or explicitly marked non-reproduced.
14. Negative results remain visible.
15. Tests, structured artifacts, plots, PDF build, references, and visual QA pass.

---

# Core principle

Paper 3.2 is not "PRA versus vector databases."

It asks how retrieval, selection, materialization, persistence, and transformer positional composition interact.

The practical architecture is allowed to be:

```text
mature external RAG
-> high-quality selected evidence
-> PRA persistent/native realization
-> request-specific composition
-> bounded transformer consumption
```

The strongest result may therefore be complementarity rather than selector replacement.
