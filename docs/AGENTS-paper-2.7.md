# AGENTS.md - Paper 2.7: Graph-Structured Query Facet Discovery for PRA

## Mission
Implement Paper 2.7 as the direct successor to Paper 2.6. Paper 2.5 established contextual multi-facet query representations and iterative PRA; Paper 2.6 adds lexical + semantic hybrid discovery. Paper 2.7 asks whether query facets themselves can be discovered from sparse graph structure over the encoded query.

This is a mechanistic retrieval paper first. Do not turn it into generic graph-RAG, GNN training, or the full multi-mode LLM architecture.

## Branch and dependency rules
- Start from the completed/frozen Paper 2.6 branch/commit, not Paper 2.5 or `main`.
- Record the exact base commit and config lineage in machine-readable results and the manuscript.
- Reuse Paper 2.5/2.6 query facets, hybrid lexical/semantic scoring, datasets, routing, materialization, accounting, and output validation.
- Preserve Paper 2.6 defaults; graph discovery is opt-in until a gate promotes it.
- Do not rewrite frozen Paper 2.5/2.6 results.

## Core research question
Can a sparse graph over contextual query units discover retrieval facets that improve PRA evidence recall and downstream quality at matched memory/materialization cost relative to:
1. one global query;
2. Paper 2.5 fixed multiscale windows/clauses/token facets;
3. Paper 2.6 lexical-semantic hybrid search;
4. embedding-only clustering?

## Required architecture
Implement a clean boundary:

`encoded query -> graph builder -> sparsifier -> clustering -> facet pooler -> existing Paper 2.6 hybrid scorer`

Do not mix graph clustering with memory graph traversal. Paper 2.7 clusters the **query side**. Existing PRA memory/evidence graphs remain a separate subsystem.

Suggested modules, adapted to actual repository layout:
```text
src/pra_hf/
  query_graph.py
  query_graph_cluster.py
  query_graph_facets.py
experiments/paper2_7_query_graph/
  run_graph_primitives.py
  run_controlled_facets.py
  run_natural_retrieval.py
  run_algorithm_cross.py
  run_encoding_mode_pilot.py
tests/
  test_query_graph.py
  test_query_graph_cluster.py
  test_query_graph_facets.py
docs/papers/paper2_7_query_graph/
  paper.tex
  README.md
```
Reuse existing modules instead of duplicating loaders, feature caches, hybrid scorers, materializers, or judges.

## Query graph contract
Represent nodes by stable query-unit IDs and retain provenance:
- token/span boundaries;
- decoded text for audit only;
- layer/head source;
- contextual vector source;
- lexical features;
- original query position.

Represent sparse edges as PyTorch tensors:
- `src: LongTensor[E]`
- `dst: LongTensor[E]`
- `weight: Tensor[E]`
- optional per-edge component scores for contextual, lexical, attention, positional, and residual-update signals.

Never require NetworkX in the inference path.

## Graph construction
Start with:
`w_ij = alpha*contextual + beta*lexical + gamma*attention + delta*position`

Residual-update similarity is optional and must not become a dependency on the residual paper line.

Sparsify immediately with per-node top-k and optional threshold:
`edge = topk(W_i, k) AND w > tau`

Required `k` values: 2, 4, 8, 16, 32, subject to query length.

Support explicit policies:
- directed;
- union-symmetrized;
- mutual-top-k;
- averaged reciprocal weight when both directions exist.

Do not silently symmetrize causal attention.

## Algorithm B1 - connected components
Implement tensor-native minimum-label propagation:
```python
labels = torch.arange(N, device=device)
for _ in range(max_iter):
    candidate = labels[src]
    recv = torch.full_like(labels, N)
    recv.scatter_reduce_(0, dst, candidate, reduce="amin", include_self=False)
    new = torch.minimum(labels, recv)
    if torch.equal(new, labels):
        break
    labels = new
```
For undirected CC, ensure both edge directions exist.

Tests:
- isolated nodes;
- two components;
- chain;
- star;
- duplicate edges;
- unsorted edges;
- directed-vs-symmetric behavior;
- convergence;
- CPU/CUDA parity.

Do not claim this is the fastest possible GPU CC implementation. It is the minimal tensor-native baseline.

## Algorithm B2 - threshold filtration
Sweep validation thresholds. Record component birth/split/persistence. Components may split as threshold rises but must never merge if the graph is constructed from one fixed weighted edge set.

Test whether persistent components are better facets than one validation-selected threshold.

## Algorithm B3 - weighted label propagation
Primary simple candidate:
`c_i <- argmax_c sum_j w_ij * 1[c_j=c]`

Implement with tensor gather plus scatter/segment reductions. Avoid Python loops over nodes or edges. Define deterministic tie-breaking. Report iterations and convergence.

## Algorithm B4 - Graclus/heavy-edge matching
Use PyTorch Geometric as the reference implementation if available. Preserve hierarchy/provenance from original query units to supernodes. Do not write a custom implementation until the reference experiment demonstrates value.

## Algorithm B5 - Leiden
Use a GPU implementation when practical as a strong classical baseline/teacher. It is not a required production dependency. Conversion to/from external graph structures must be included in latency accounting.

## Algorithm B6 - soft memberships
Only after hard graph methods pass the natural retrieval gate.

Implement a bounded soft assignment `Q[N,K]` with propagation using sparse matrix multiplication where possible. Do not use oracle `K` at test time without an explicit control; define a bounded initialization/seed policy.

Facet pooling:
`Z = Q.T @ H`
with normalization by membership mass.

## Causal representation rule
Main Paper 2.7 experiments must work with the pretrained decoder-only PRA model.

Causal hidden states satisfy `h_i = f(x_<=i)`. Native attention is directed. State this limitation explicitly.

Allowed causal graph signals:
- hidden-state cosine similarity;
- Paper 2.6 lexical relations;
- directed attention summaries;
- reciprocal/symmetrized attention only as an explicit derived feature;
- positional/local edges.

Do not remove the causal mask from a pretrained model and present the result as a valid encoder.

## Bidirectional encoding pilot
This is Gate 5, not the starting point.

Compare:
- frozen causal query representations;
- an exploratory unmasked pass only if technically meaningful and clearly labelled OOD;
- preferably a separately trained/custom encoder-mode model if available.

The clean future architecture is:
`Encode_bidir -> query graph -> Q -> Z -> Generate/Review/Verify`

Paper 2.7 may motivate this multi-mode architecture but must not depend on it.

## Controlled dataset
Create compositional queries with known latent facets. Vary:
- 1-6 facets;
- facet ordering;
- contiguous vs non-contiguous evidence;
- shared entities;
- lexical overlap across facets;
- distractor clauses;
- pronouns/references;
- one phrase belonging to multiple facets;
- displacement up to 2K/4K/8K where feasible.

Do not let templates leak facet IDs through fixed punctuation or ordering. Include paraphrase and permutation controls.

## Natural datasets
Reuse existing PRA adapters/features first:
- HotpotQA;
- 2WikiMultiHopQA;
- MuSiQue;
- QASPER.

Do not add new datasets until existing ones have produced interpretable graph-vs-baseline results.

## Baselines
Mandatory:
- global query;
- Paper 2.5 multiscale windows;
- Paper 2.5 clause/cue facets;
- Paper 2.6 lexical-only;
- semantic-only;
- Paper 2.6 hybrid;
- embedding-only k-means or comparable vector clustering;
- graph CC;
- weighted label propagation.

Graclus, Leiden, and soft propagation are gated extensions.

## Metrics
Facet recovery where labels exist:
- ARI;
- NMI;
- pairwise F1;
- boundary F1;
- cluster-count error;
- hierarchy consistency;
- overlap metrics for soft facets.

PRA utility:
- evidence/supporting-fact recall;
- routing precision;
- RCB if inherited;
- selected references/chunks;
- materialized tokens;
- downstream answer metrics;
- quality at matched materialization budget.

Efficiency:
- graph construction ms;
- clustering ms;
- total routing ms;
- N, E, k;
- iterations to convergence;
- peak GPU memory;
- graph overhead as percent of end-to-end inference.

## Experimental gates
### G0 - inherited parity
Reproduce frozen Paper 2.6 baseline rows within tolerance before graph experiments.

### G1 - primitive correctness
All graph primitive tests and CPU/GPU parity pass.

### G2 - controlled facet recovery
At least one graph method beats fixed windows and embedding-only clustering without oracle test-time cluster count.

### G3 - natural PRA utility
At matched materialization budget, graph facets improve evidence recall or quality/cost on at least one natural benchmark without material regression on the rest.

If G3 fails, diagnose and write the negative result before adding more algorithms.

### G4 - algorithm expansion
Only after G3 compare Graclus, Leiden, soft propagation, and more elaborate layer/head aggregation.

### G5 - encoding-mode pilot
Freeze the best graph method before causal-vs-bidirectional representation comparison.

## Key ablations
- contextual vs lexical vs attention vs hybrid edges;
- top-k 2/4/8/16/32;
- threshold and threshold persistence;
- directed vs union vs mutual-top-k graph;
- early/middle/late layer;
- single layer vs cross-layer persistence;
- hard vs soft membership;
- fixed spans vs discovered facets;
- matched facet count where useful as a diagnostic, never as the only result;
- matched final materialization budget.

Avoid the full Cartesian product. Use validation gates to narrow.

## Causal tests
A discovered community is more convincing if its removal selectively damages its corresponding retrieval/answer component.

Add cluster ablation:
1. run normal graph facets;
2. suppress one facet before routing;
3. measure which evidence and answer components degrade;
4. compare against size-matched random token/span ablation.

Do not interpret high ARI/NMI alone as evidence of computational subtasks.

## Falsification criteria
Treat these as valid outcomes:
- graph methods do not beat Paper 2.5 windows;
- Paper 2.6 lexical hybrid explains all gains;
- attention edges add no value;
- threshold instability prevents robust facets;
- graph clusters look semantically plausible but do not improve retrieval;
- graph overhead exceeds retrieval savings;
- causal representations are the limiting factor.

Do not tune until every benchmark shows a positive result.

## Artifact requirements
Every run writes:
- config;
- git commit;
- model/tokenizer revision;
- dataset split/hash;
- seeds;
- graph parameters;
- edge-family weights;
- threshold selection provenance;
- timing/memory accounting;
- per-example facet assignments and provenance;
- retrieval selections;
- aggregate metrics.

Keep validation threshold/policy selection separate from held-out test evaluation.

## Paper-writing constraints
The manuscript must clearly distinguish:
1. query graph construction;
2. facet/community discovery;
3. memory retrieval;
4. materialization;
5. answer generation.

Never describe memory graph traversal from Paper 2.5 as query clustering.

Do not claim that causal decoder attention provides a symmetric semantic graph.

Do not claim bidirectional query encoding is validated unless it was trained/evaluated as such.

## Stop conditions
Stop and report before expanding scope if:
- Paper 2.6 parity cannot be reproduced;
- graph construction changes baseline retrieval when disabled;
- CPU/GPU graph results disagree beyond defined tolerance;
- test-time oracle facet count/labels leak into selection;
- materialization budgets are not matched;
- graph latency accounting excludes conversion or synchronization costs.

## Preferred final contribution
A successful Paper 2.7 should support a narrow claim:

> Sparse graph structure over contextual query representations can discover retrieval facets that improve the PRA quality/cost frontier over predetermined query-span families, and the useful graph operations can be expressed largely as GPU-friendly tensor primitives.

The bidirectional Encode-mode result is secondary and should be framed as a bridge to the multi-mode LLM paper series.
