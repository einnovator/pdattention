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

Capture tokenwise queries while reusing the local Paper 2.8 source-key caches:

```powershell
python experiments/paper2_9_look_ahead_back/precompute_temporal_queries.py --device cuda
```

Run the frozen-index retrieval study. CPU is usually faster than an older GPU
for the many small routing matrices; this stage does not execute the backbone:

```powershell
python experiments/paper2_9_look_ahead_back/run_temporal_study.py --device cpu
```

Both commands checkpoint their expensive phases and resume by default. Pass
`--no-resume` only when intentionally replacing a completed local run.
