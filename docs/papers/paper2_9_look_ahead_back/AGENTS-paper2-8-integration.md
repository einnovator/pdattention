# AGENTS.md Add-on — Paper 2.9 Integration with Paper 2.8 Low-Rank Routing

## Purpose

Patch the existing Paper 2.9 plan, Temporal Semantic Discovery for PRA, so its temporal-query experiments explicitly integrate the strongest current Paper 2.8 routing representations.

This add-on extends the existing Paper 2.9 AGENTS.md; it does not replace it.

Paper 2.9 remains a temporal-query paper. Paper 2.8 remains the memory-side / routing-subspace paper.

Combined question:

> Does a temporally richer query trajectory extract more useful information from the compact low-rank memory-routing space discovered in Paper 2.8 than a single instantaneous query state?

## Freeze inherited Paper 2.8 memory-side configurations

### A. Rank-16 all-token
- 32 projected token states/chunk;
- rank 16;
- 2,048 FP32 bytes/chunk;
- current strongest QASPER point: recall 0.2542.

### B. Rank-8 all-token
- 32 projected token states;
- rank 8;
- 1,024 FP32 bytes/chunk;
- current QASPER recall about 0.2456.

### C. Rank-8, eight-centroid
- eight representatives/chunk;
- rank 8;
- 256 FP32 bytes/chunk;
- current QASPER recall 0.1829.

Controls:
- native mean gist;
- Paper 2.6 exact;
- BM25;
- best hybrid;
- full native-K teacher as analysis-only where useful.

Do not retune 2.8 memory representations on Paper 2.9 test identities. If fresh Paper 2.8 confirmation later supersedes these weights, use the confirmed frozen projections.

## Conceptual separation

Paper 2.8:
K_C -> Z_C

where Z_C is m x r or 32 x r.

Paper 2.9:
q_t -> Q_{t-B:t+F}

Combined:
A_ij = z_{q_i}^T z_{C_j}

Do not average query and memory representations before interaction except as an explicit baseline.

Core hypothesis:

> Late interaction between temporal query states and compact memory modes preserves information lost by early averaging on either side.

## Mandatory 2x2 interaction experiment

Before predictive look-ahead, run:

1. B=1 current query × native mean gist
2. B>1 temporal query × native mean gist
3. B=1 current query × Paper 2.8 low-rank memory
4. B>1 temporal query × Paper 2.8 low-rank memory

At minimum B in {1,4,8}.

For low-rank memory use:
- rank-16 all-token;
- rank-8 centroid-8.

Estimate:
- query-side gain;
- memory-side gain;
- query×memory interaction.

Classify combined gain as additive, subadditive, or synergistic. Do not claim synergy without an explicit interaction contrast.

## Temporal projection into frozen Paper 2.8 space

For each temporal query state q_i:
z_qi = W_q q_i

First freeze Paper 2.8 W_q/W_k.

Do not retrain them during the first temporal experiment. This isolates whether temporal evidence helps inside the existing compact space.

Only afterward may a jointly temporal-trained projection be tested. Keep backbone frozen and separate this result clearly.

## P1 integration — causal look-behind

For B in {1,2,4,8,16}, add:
- temporal query vs rank-16 all-token memory;
- temporal query vs rank-8 centroid-8 memory.

Compare query aggregation:
- current/last;
- mean;
- validation-fixed weighted pooling;
- late interaction.

Preferred late-interaction controls:

S_max(C) = sum_i w_i max_j z_qi^T z_Cj

and a top-mean reducer over token×memory-mode scores.

Validate reducer once and freeze before test.

## P2 integration — delayed commitment

For D in {1,2,4,8}:
- accumulate actual future observed query states;
- project them with frozen W_q;
- score against frozen low-rank memory;
- compare with immediate B=1 routing at the same final K/V budget.

Measure:
- evidence recall;
- complete evidence;
- selected-set churn;
- top-1 margin;
- entropy;
- delay;
- router calls/token;
- low-rank dots/token.

Key question:

> Does waiting for a few actual tokens improve low-rank semantic routing enough to justify reduced routing frequency or latency?

## P3 integration — oracle and prefill future

For oracle future context:
- project already-known future query states with frozen W_q;
- score against rank-16 and centroid-8 memory.

This isolates temporal query headroom without changing memory representation.

For non-causal prompt prefill sidecar:
- keep causal backbone states unchanged;
- use sidecar only to form routing queries;
- compare against causal look-behind in the same low-rank space.

Do not call known prompt future tokens predictive look-ahead.

## P4 clarification — query × memory compression

Update P4 primary memory conditions to:
1. native mean gist;
2. historical multi-gist/prototype controls;
3. Paper 2.8 rank-16 all-token low-rank index;
4. Paper 2.8 rank-8 centroid-8 index;
5. full-dimensional native landmarks only as historical controls.

Cross best:
- causal temporal query;
- delayed query;
- prefill-sidecar query where allowed;

with:
- mean;
- rank 16;
- rank-8 centroid-8.

Do not reopen a large m×r search. Paper 2.8 owns that axis.

## P5 facets integration

For Paper 2.7 syntax/graph facets:
- project each retained facet query into frozen 2.8 low-rank query space;
- compare facet late interaction with temporal-token late interaction;
- optionally add global temporal query + facets as a bounded control.

Preserve the Paper 2.7 lesson: partition quality does not imply retrieval quality.

## P6 slower routing clock

High-priority combined systems experiment.

For stride s in {1,2,4,8}:
- reuse prior selection between updates;
- accumulate a short temporal query window;
- route at stride boundary or confidence/drift trigger.

Report:
- router calls/token;
- low-rank dots/token;
- routing latency/token;
- evidence recall;
- selection churn.

Important comparisons:
- B=1, stride 1;
- B=4, stride 4;
- B=8, stride 8.

Test whether slower routing is both cheaper and semantically better.

## P7 predictive look-ahead

Only after existing Paper 2.9 oracle-future gates pass.

Prefer predicting future low-rank query states:

hat z_q,t+1:t+F

rather than full native q where possible.

Compare against:
- predicting full q;
- equal-latency delayed commitment.

Do not claim predictive look-ahead useful unless it beats delay at matched latency.

## Dataset strategy

Primary:
- QASPER;
- HotpotQA.

When the Paper 2.8 MuSiQue/2Wiki extension yields compatible confirmed indexes, add:
- 2WikiMultiHopQA;
- MuSiQue.

Diagnostic roles:
- QASPER: strongest candidate for temporal×low-rank gain;
- HotpotQA: lexical-dominant control;
- 2Wiki/MuSiQue: test temporal relational composition and multi-hop evidence.

Do not assume 2Wiki/MuSiQue will resemble QASPER.

## Hybrid lexical + temporal-low-rank routing

Add one bounded hybrid:
- exact/BM25 lexical score;
- temporal low-rank semantic score.

Use inherited Paper 2.6 combination semantics where possible.

Validation-select weights only.

Compare:
- lexical only;
- B=1 low-rank;
- temporal low-rank;
- lexical + B=1 low-rank;
- lexical + temporal low-rank.

Report evidence uniquely recovered by each channel.

## Added metrics

Interaction diagnostics:
- temporal-query × low-rank-memory interaction gain;
- additive vs synergistic decomposition;
- number of query states contributing to selected chunk score;
- number of memory centroids/tokens contributing;
- late-interaction sparsity;
- score concentration.

Cost:
- routing-index bytes;
- temporal-query buffer bytes;
- low-rank dots/router call;
- low-rank dots/generated token;
- router calls/token;
- cached latency;
- active native K/V;
- transfer bytes.

Keep routing state separate from backing native K/V.

## Integration gates

### I0 — Frozen Paper 2.8 parity
B=1 low-rank routing must reproduce frozen Paper 2.8 rows on matched identities.

### I1 — Temporal gain in frozen low-rank space
Pass if B>1 or delayed commitment improves low-rank retrieval on at least one natural dataset with acceptable non-regression.

### I2 — Query × memory interaction
Pass if temporal-query + low-rank-memory improves beyond either side alone or gives a clearly better quality-cost frontier.

### I3 — Slow-clock efficiency
Pass if temporal accumulation reduces router calls/token without material retrieval degradation.

### I4 — Hybrid complementarity
Pass if lexical + temporal-low-rank recovers evidence unavailable to either alone at fixed final K/V budget.

### I5 — Generation eligibility
Only after retrieval gates pass, run downstream native-K/V answer generation.

## Claim discipline

Do not say:
- look-ahead for known prompt future tokens;
- native QK compression if the space is evidence-supervised with low teacher overlap;
- universal semantic router based on QASPER;
- memory reduction when only routing-index memory shrinks;
- synergy without explicit interaction analysis.

Preferred terms:
- compact low-rank native interaction space;
- temporal low-rank query;
- late query-memory interaction;
- routing-index compression;
- native K/V payload unchanged.

## Likely positive story

1. Paper 2.8 finds a tiny memory-side routing subspace.
2. Paper 2.9 shows a single instantaneous query underuses it.
3. Short causal windows or delayed commitment improve retrieval.
4. Late interaction preserves temporal and memory multimodality.
5. Routing can run on a slower clock.
6. Lexical and temporal-low-rank channels are complementary.
7. Selected native K/V improves generation.

Potential supported claim:

> Compact memory-side interaction spaces and temporally extended query trajectories jointly improve sparse long-memory routing without changing the frozen backbone or native K/V payload.

Use only if interaction and generation evidence support it.

## Likely negative story

If temporal extent does not improve low-rank routing:
- conclude the low-rank router already extracts most available information from the current state.

If temporal queries help mean gists but not low-rank memory:
- conclude low-rank routing already compensates for some query ambiguity.

If gains stay QASPER-specific:
- retain dataset/channel-specialization interpretation.

Negative interaction results are publishable.

## Immediate execution order

1. B=1 Paper 2.8 parity inside Paper 2.9 harness.
2. 2x2 temporal-query × memory-representation experiment.
3. causal B sweep with rank-16 and centroid-8 memory.
4. delayed commitment in the same frozen low-rank space.
5. slower routing clock.
6. lexical + temporal-low-rank hybrid.
7. prefill future sidecar if existing gates permit.
8. predictive look-ahead only if oracle future headroom remains.
9. downstream generation only after retrieval gates pass.

Do not jointly retrain the 2.8 projection until frozen-space experiments are complete.
