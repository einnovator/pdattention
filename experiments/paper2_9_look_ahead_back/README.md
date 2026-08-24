# Paper 2.9: Temporal Semantic Discovery

This experiment family changes only the temporal extent and cadence of PRA's
query-side routing representation. The frozen causal backbone, Paper 2.8
memory-side projections, final chunk budget, and backing native K/V remain
unchanged.

Execution is gated:

1. capture aligned tokenwise query states and reproduce Paper 2.8 at `B=1`;
2. run the frozen-space query-by-memory interaction and causal window sweep;
3. evaluate causal delay against analysis-only known-future routing;
4. test a slower routing clock and one validation-selected lexical hybrid;
5. run downstream native-K/V generation only if retrieval gates pass.

Large feature caches and resumable shards are local artifacts. Manifests retain
their hashes and provenance; compact tables, plots, and selected checkpoints are
tracked with the paper.
