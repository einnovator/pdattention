# Paper 2.7 natural-retrieval claim audit

- Frozen Paper 2.5/2.6 query and memory states; no model forward or training.
- Graph policy selected on controlled validation only.
- All conditions request exactly four aligned 32-token chunks.
- Physical native K/V materialization and answer generation were not run.
- Query clustering is distinct from inherited memory traversal.
- Attention and residual-update query edges were unavailable in this frozen cohort.
- Causal hidden-state graphs are not described as bidirectional semantic encoders.
- Paper 2.6 route parity: True.
- G3 directional retrieval-budget proxy passed: True.
- G3 paired-interval criterion passed: False.
