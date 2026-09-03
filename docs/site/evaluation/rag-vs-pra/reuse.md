# Multi-Query Reuse

L6 asks multiple questions over one persistent document corpus. Standard RAG serializes selected text for every request. PRA can retain document identity and native chunks, materializing a chunk only when it has not already become resident.

The current 50-query diagnostic uses real BM25 candidates, 20 documents per query, and a 2K physical budget. It counts unique selected chunk IDs as newly materialized state:

| Measure after 50 queries | Tokens |
| --- | ---: |
| Standard RAG cumulative visible selected text | 101,730 |
| PRA cumulative unique newly materialized chunks | 80,535 |
| Avoided repeated materialization | 21,195 (20.8%) |

This is a corpus-overlap accounting result, not a latency claim. The questions do not heavily reuse the same evidence, so the curve also shows an important boundary: PRA's reuse advantage depends on resource overlap. Warm repeated questions over a common document collection should produce a larger separation than unrelated questions over the same corpus namespace.

Engine qualification must add physical cache hits, bytes promoted, eviction, TTFT, queue time, and cumulative wall time before this accounting becomes an economic result.
