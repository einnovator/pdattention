# Paper 2.6 claim audit

- Frozen Qwen3-0.6B features; no model weights are loaded by this analysis.
- Every primary channel requests at most four aligned 32-token chunks.
- Iterative routing reports total unique chunks and all semantic/token comparisons.
- Natural dataset identities use deterministic validation/test partitions inherited from Paper 2.5.
- The heuristic selector receives only query/routing observables; its API rejects gold geometry fields.
- Gold evidence geometry is explanatory only.
- No native K/V is materialized and no generation metric is reported.
- Cohorts are below the requested 50 held-out examples per dataset; conclusions are cohort-bounded.
- Approximate matching is normalized tokenizer-piece sequence similarity, not character edit distance.
