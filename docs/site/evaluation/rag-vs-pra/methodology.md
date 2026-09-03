# Methodology

## Frozen first-stage retrieval

Each question produces one immutable candidate receipt before a context condition runs. The receipt records:

- dataset and corpus revisions;
- corpus and retriever-index digests;
- retriever name and implementation revision;
- ordered document IDs, ranks, and scores;
- document fingerprints;
- chunker configuration and seed.

Loading a receipt rejects missing documents, changed document content, rank gaps, duplicate IDs, or a mismatched receipt digest. Both Standard RAG and PRA must consume this exact identity.

## Conditions

| Condition | Context behavior |
| --- | --- |
| No PRA | Global BM25 ranks fixed-token chunks from the frozen candidate documents; whole chunks are packed into visible context up to the budget. |
| PRA, no adaptor | BM25 and a parameter-free hashed-semantic view are rank-fused; selected typed-document chunks are materialized up to the same budget. |
| PRA adaptor bundle | Uses an exact immutable bundle only when its document-routing adaptor is qualified for this cohort. Otherwise the state is `NO_QUALIFIED_ADAPTER`. |
| Oracle gold documents | Packs available gold-document chunks. This is a research diagnostic and is excluded from headline deltas. |

The model, prompt, generation settings, engine, and hardware must match within a model-backed comparison. Oracle rows never count as PRA improvements.

## Metrics remain separate

**First-stage retrieval:** document Recall@K and whether all gold documents entered the candidate set.

**Context selection:** supporting-document and supporting-span coverage, MRR, NDCG, gold/false selected-document fractions.

**Physical context:** logical candidate tokens, packed/materialized tokens, selected/full ratio, discarded tokens, and materialization avoidance.

**Answer:** dataset-native EM and token F1. The evidence-availability probe is reported under its own label and is never substituted for generation EM/F1.

**Serving:** TTFT, ITL, completion latency, output tokens/s, ingestion/index/native-encoding time, active detail bytes, and reuse. A value is absent rather than estimated when an engine cannot expose it.

## Failure labels

Per-example results distinguish first-stage retrieval, Standard-RAG packing, PRA selection, materialization, generation, distractor confusion, and insufficient budget. This prevents an answer miss from being attributed automatically to routing.
