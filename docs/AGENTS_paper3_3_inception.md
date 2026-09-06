# AGENTS.md — Paper 3.3 Inception
## Sparse Learned Cross-Document Contextualization for Persistent Native Memory

Suggested branch:
`research/paper3-3-crossdoc-context`

Base branch/commit:
`research/paper3-2-rag` at `de90606a1859f18200f3b9be121a857c140ace2b`

Paper 3.3 must be independently publishable.

It builds on Paper 3.2 but must restate all required background, definitions, baselines, and key inherited empirical facts needed to understand the new problem.

---

# 1. Paper boundary

Paper 3.2 established:
- persistent contiguous native transport;
- canonical persistent PRA records;
- pre-RoPE request-time rebinding;
- causal isolation of the packed-RAG/record difference;
- finite-precision runtime effects;
- failure/limits of fixed partial, gist-append, mask, and first residual repair mechanisms.

Paper 3.3 owns a new positive objective:

> Recover packed-RAG task quality using a small, learned, request-specific set of cross-document interactions while preserving persistent independent PRA record K/V.

Paper 3.3 is not another broad RAG benchmark. It is a focused sparse-composition paper.

---

# 2. Central research question

Ordinary packed RAG computes dense cross-document causal attention:

```text
D1 -> D2 -> D3 -> ... -> Q
```

Persistent PRA records avoid that cost:

```text
D1   D2   D3
 \    |    /
      Q
```

Paper 3.2 showed:
- independent records lose quality on the powered natural comparison;
- fixed sparse masks are non-monotonic;
- extremely sparse boundary interaction can approach packed-RAG F1 on some metrics;
- a small trained residual recovers part of the gap but remains uncertain.

Paper 3.3 asks:

> Which cross-document interactions actually matter, and can they be selected and executed sparsely enough to recover RAG quality without rebuilding the packed prefix?

---

# 3. Primary hypotheses

## H1 — Dense packed-RAG cross-document attention is highly redundant

An oracle small subset of cross-document interactions should recover most of packed-RAG task quality.

## H2 — Useful interactions have predictable structure

The important interactions should be predictable from some combination of:
- query;
- record/chunk gist;
- reranker score/rank;
- document pair identity;
- layer/head;
- token role/boundary;
- semantic overlap;
- native attention features.

## H3 — Executing real base-model attention on selected interactions is more robust than synthesizing arbitrary post-hoc K/V

Paper 3.2's parameter-free appended K/V was off-manifold and failed badly.

Paper 3.3 should prefer:

```text
select sparse interaction
→ execute original transformer operation
```

over:

```text
invent arbitrary K/V
→ append to decoder
```

unless a learned K/V adapter is explicitly trained and validated.

## H4 — Task-first sparse composition can recover ≥80% of the packed-RAG quality gap with ≤10% of dense cross-document interaction cost

This is the main quantitative target.

---

# 4. Inherited baseline values

For the powered Paper 3.2 endpoint:

```text
Packed RAG:
  F1 ≈ .199
  official ≈ .700

Independent PRA:
  F1 ≈ .135
  official ≈ .433

Trained rank-8 residual:
  F1 ≈ .154
  official ≈ .533
```

Paper 3.3 must reproduce or explicitly inherit these values with provenance.

Target for 80% gap recovery:

```text
F1 target:
  .135 + .8 * (.199 - .135) ≈ .186

Official target:
  .433 + .8 * (.700 - .433) ≈ .647
```

These are target gates, not guaranteed claims.

---

# 5. Experiment 0 — reproduction gate

Before new mechanisms:

1. reproduce the frozen Paper 3.2 packed-RAG baseline;
2. reproduce independent PRA;
3. reproduce the trained residual baseline if code/checkpoint is inherited;
4. verify canonical `ContextRecord`, BGE selector, token budget, and prompt semantics;
5. verify pre-RoPE request binding and precision/runtime contracts.

Do not train the new selector before this gate passes.

---

# 6. Experiment 1 — oracle cross-document sparsity

This is the highest-priority experiment.

Instrument ordinary packed RAG and record document→document attention edges.

For each layer/head/query-token/key-token pair where source and target documents differ, capture:
- attention probability/mass;
- source/target document/chunk;
- layer;
- head;
- token positions;
- boundary distances;
- reranker ranks/scores;
- query relevance if available.

Do not persist prohibitively large dense tensors if unnecessary. Prefer top-k edge summaries, per-layer/head sparse records, or streaming aggregation.

---

# 7. Oracle sparsification methods

Test at least two oracle definitions.

## O1 — top attention mass

Keep the highest-attention cross-document edges from the full packed teacher.

Budgets:

```text
0%
0.01%
0.05%
0.1%
0.5%
1%
2%
5%
10%
100%
```

## O2 — cumulative attention mass

Keep the minimum edge set explaining:

```text
50%
75%
90%
95%
99%
```

of cross-document attention mass.

Optional later:
- gradient/ablation importance oracle;
- output-sensitivity oracle.

Do not start with expensive causal attribution unless attention-mass oracle shows headroom.

---

# 8. Oracle execution semantics

The oracle sparse condition must execute **real transformer attention**, not post-hoc synthetic K/V.

Preferred implementation:

```text
persistent independent record states
        ↓
selected cross-document edges
        ↓
request-local sparse attention/update using base-model weights
        ↓
query
```

If exact sparse state propagation is difficult, begin with a packed teacher-mask oracle:

```text
same packed token sequence
+ sparse document→document mask
```

This establishes the empirical headroom before engineering the PRA runtime realization.

---

# 9. Oracle success gate

Proceed to learned interaction selection only if an oracle sparse condition reaches approximately:

```text
F1 >= .19
official >= .67
```

or is statistically indistinguishable from packed RAG while using:

```text
<= 5%
```

of dense document→document interactions.

If no oracle sparse mask approaches packed quality, report that boundary before training a selector.

---

# 10. Experiment 2 — interaction localization

Analyze oracle-selected edges by:

- layer;
- head;
- source document rank;
- target document rank;
- source/target chunk;
- token boundary distance;
- token type;
- entity overlap;
- query overlap;
- retrieval/reranker score;
- gist similarity.

Questions:

1. Are useful interactions concentrated in a few layers?
2. Are they concentrated in a few heads?
3. Are boundary tokens unusually important?
4. Does the highest-ranked document act as a hub?
5. Are interactions mostly document-pair level rather than token-specific?
6. Can chunk gists predict them?

This determines the selector granularity.

---

# 11. Experiment 3 — learned pair selector

Start coarse.

Predict whether selected document/chunk pair `(i,j)` should interact.

Features:

```text
query gist
source gist
target gist
source reranker score/rank
target reranker score/rank
pairwise semantic similarity
entity/token overlap
resource type/metadata
```

Output:

```text
s_ij ∈ [0,1]
```

Train against oracle pair-level labels/importance.

Evaluate:
- ROC/AUC;
- precision/recall at interaction budgets;
- oracle-quality retained by selected pairs.

Do not immediately predict token-level edges.

---

# 12. Experiment 4 — layer/head routing

If oracle localization shows concentration, extend selector:

```text
s_ijlh
```

or factored:

```text
s_pair(i,j)
s_layer(l | i,j,q)
s_head(h | l,i,j,q)
```

Prefer low-complexity factorization.

Target: avoid executing all layers/heads for every selected pair.

---

# 13. Experiment 5 — sparse base-model interaction

For selected pair/layer/head regions, perform request-local genuine transformer interaction.

Potential mechanisms, in increasing complexity:

## S1 — sparse packed mask teacher/runtime
Recompute only a small request-local composition block under selected edges.

## S2 — boundary token interaction
Use selected real record boundary tokens and original Q/K/V projections.

## S3 — selected-token interaction
Use a learned token gate within selected record pairs.

## S4 — low-rank request state
Only after S1–S3 establish headroom.

Do not synthesize untrained K/V.

---

# 14. State propagation strategy

A major design question is how selected interaction affects later layers.

Evaluate:

### P0 — single-layer correction
Interact at selected layer only.

### P1 — selected-layer stack
Interact at a small contiguous set of layers.

### P2 — request-local corrected tokens propagated through remaining layers
Potentially more accurate but more expensive.

Measure:
- corrected token count;
- layer count;
- total request FLOPs;
- quality.

---

# 15. Query-conditioned selection

The final selector must depend on the query.

Do not optimize a fixed global boundary policy.

Train:

```text
π(q, selected records, layer state)
```

Compare:
- query-independent learned selector;
- query-conditioned selector.

---

# 16. Training data

The Paper 3.2 residual used only 12 training questions.

Paper 3.3 should use a proper train/validation/test split.

Initial target:

```text
train: 500–2,000 questions
validation: 100–200
test: fixed five-seed held-out cohort, >=150 questions
```

Use additional public MultiHop-RAG training examples if licensing/split semantics allow.

Never train on the Paper 3.2 final evaluation examples. Persist exact IDs.

---

# 17. Training objective

Task performance is primary.

Suggested objective:

```text
L =
  λ_task * L_answer
+ λ_seq  * KL(P_sparse || P_packed)
+ λ_sel  * L_oracle_selector
+ λ_cost * interaction_budget
```

Start with:

```text
λ_task = 1.0
λ_seq  = 0.25
λ_sel  = 0.25
λ_cost = tuned small positive value
```

Do not give global K/V reconstruction a dominant weight.

Optional state distillation should be local, selected-layer, selected-token, and low weight.

Paper 3.2 showed that global K/V similarity is not the same as task quality.

---

# 18. Packed teacher vs task target

Use packed RAG as a useful teacher, not an unquestioned oracle.

Because Paper 3.2 found:
- cross-document interaction can hurt some models/cohorts;
- NLL and generated quality can disagree.

Therefore evaluate:
- packed-teacher KL;
- gold task loss;
- official generation quality

separately.

A sparse PRA system may legitimately outperform the teacher on some examples.

---

# 19. Order robustness objective

Preserve PRA's large order-robustness advantage.

For the same evidence set, train/evaluate permutations:

```text
canonical
reverse
random-1
random-2
```

Optional consistency term:

```text
L_order = mean JS(P_pi_a, P_pi_b)
```

Do not sacrifice task quality solely to reduce variance.

Positive result requires RAG-matched quality with lower or equal order sensitivity.

---

# 20. Precision/runtime qualification

Inherit Paper 3.2's finding:
- shape-matched pre-RoPE rebinding is exact;
- lower precision amplifies separate-shape drift.

For the best Paper 3.3 mechanism, qualify:
- FP16;
- INT8;
- INT4;
- FP32 if affordable.

Do not rerun the full precision matrix on every exploratory mechanism.

---

# 21. Model ladder

## M0 — Qwen3-1.7B
Full discovery/training matrix.

## M1 — Qwen3-4B
Reduced qualification.

## M2 — Qwen3-8B
Reduced qualification.

## M3 — Llama-3.1-8B
Cross-family qualification.

Only expand to larger models after a learned sparse mechanism clears the 1.7B quality/economics gate.

---

# 22. Baselines

Mandatory:

```text
Packed RAG
Independent PRA
Paper 3.2 rank-8 residual
Fixed boundary-8 mask
Fixed boundary-16/32 where useful
Oracle sparse mask
Learned sparse selector
```

Optional:
- contiguous native block;
- full repair;
- prefix cache for systems comparison.

---

# 23. Metrics

## Quality
- F1;
- official score;
- exact match;
- gold NLL;
- answer log probability.

## Distribution
- first-step JS;
- sequence KL where feasible;
- output parity.

## Sparsity
- cross-document edges;
- percentage of dense edges;
- selected record pairs;
- selected tokens;
- selected layers/heads.

## Systems
- persistent K/V reuse;
- newly computed token states;
- request-local FLOPs;
- composition latency;
- TTFT;
- total latency;
- memory.

## Robustness
- order sensitivity;
- seed CI;
- model/family transfer.

---

# 24. Main plots

Paper 3.3 should target:

1. **Oracle quality vs interaction budget**
2. **Learned quality vs interaction budget**
3. **Where useful interactions occur: layers/heads/pairs**
4. **Packed vs PRA vs learned sparse composition**
5. **Quality vs request-local recomputation**
6. **Order robustness at matched quality**
7. **Scale/family replication**

---

# 25. Failure taxonomy

Add:

```text
ORACLE_SPARSITY_INSUFFICIENT
PAIR_SELECTOR_MISS
LAYER_SELECTOR_MISS
HEAD_SELECTOR_MISS
TOKEN_SELECTOR_MISS
SPARSE_STATE_PROPAGATION_FAILURE
OFF_MANIFOLD_CORRECTION
TASK_DISTILLATION_CONFLICT
ORDER_ROBUSTNESS_REGRESSION
COMPUTE_BUDGET_REGRESSION
QUANTIZATION_TRANSFER_FAILURE
```

---

# 26. Claim hierarchy

## Claim A — oracle sparsity
A small subset of cross-document interactions preserves packed-RAG quality.

## Claim B — predictable sparsity
Those interactions can be predicted from query/record features.

## Claim C — practical sparse composition
A learned PRA mechanism recovers >=80% of the RAG gap with <=10% request-local interaction cost.

## Claim D — stronger outcome
Sparse PRA matches packed RAG while preserving better order robustness and non-prefix reuse.

Do not claim C/D unless powered held-out results support them.

---

# 27. Falsification

Paper 3.3 remains scientifically useful if:
- oracle needs dense interaction;
- learned selector cannot approach oracle;
- task-first training overfits;
- interaction routing costs exceed saved prefill;
- order robustness disappears at matched quality;
- cross-family transfer fails.

Report these explicitly.

---

# 28. Systems target architecture

If successful:

```text
ContextRecord store
    ↓
retrieval + reranker
    ↓
persistent pre-RoPE native K/V
    ↓
query-conditioned sparse interaction selector
    ↓
small amount of real cross-record transformer computation
    ↓
request-local corrected state
    ↓
query decode
```

Persistent record K/V remains immutable.

Request-local contextualization is ephemeral and auditable.

---

# 29. Paper 4.5 integration

Reuse Paper 4.5:
- `ContextRecord`;
- storage metadata;
- resolver;
- selection receipts;
- policy abstractions.

Do not fork a new record format.

Any learned interaction policy should be representable later as a request/session policy in the runtime.

Keep research implementation modular.

---

# 30. Definition of done for inception iteration

The inception iteration is complete when:

1. Paper 3.3 branch exists;
2. Paper 3.2 reproduction gate is documented;
3. oracle attention-edge extraction is implemented;
4. oracle sparse-mask budget sweep runs on a small natural cohort;
5. oracle headroom decision is made;
6. interaction-localization artifacts exist;
7. initial learned pair-selector design is specified;
8. training/evaluation splits are frozen;
9. draft paper compiles;
10. no Paper 3.2 result is silently changed or re-owned.

The first go/no-go gate is oracle sparsity.
