# Baselines

## Standard RAG

The baseline is global BM25 chunk selection over the frozen candidate documents:

1. Split every candidate into 256-token chunks with 32-token overlap.
2. Include title and document identity in the retained resource, while ranking chunk body text.
3. Compute BM25 with `k1=1.2` and `b=0.75` across all candidate chunks.
4. Sort by descending score with stable chunk ID as the tie breaker.
5. Pack whole chunks in rank order when they fit the physical-token budget.
6. Report both packed and discarded candidate tokens.

This is not intentionally weakened. A stronger dense/reranked RAG baseline remains a required follow-up.

## Parameter-Free PRA

Generic PRA uses two independently ranked views of the same chunks:

- the same BM25 score used by the baseline;
- a deterministic 128-dimensional hashed-semantic view.

Reciprocal-rank fusion with constant 60 combines the two rankings without mixing incomparable raw scores. The resulting exact character intervals point back into typed, versioned document records.

## Adaptor bundle

No learned document-RAG router is currently qualified for this cohort. The canonical third condition is therefore retained with state `NO_QUALIFIED_ADAPTER`. Existing QASPER-specific routers are not silently reused because their transfer to other datasets is mixed.

## Oracle controls

Gold-document and later gold-span oracles locate headroom. They answer where a failure occurs; they do not represent a deployable system and are excluded from RAG/PRA deltas.
