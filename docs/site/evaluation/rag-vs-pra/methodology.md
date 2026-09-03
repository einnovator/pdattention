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
| `NO_PRA_STANDARD_RAG` | Global BM25, or the declared strong conventional reranker, packs frozen candidate chunks as visible context. |
| `PRA_SELECTED_CONTEXT_NO_ADAPTOR` | The generic or strong PRA selector's exact intervals are serialized as visible selected text. |
| `PRA_NATIVE_MEMORY_NO_ADAPTOR` | The same frozen selection receipt is encoded as detached native K/V. |
| `PRA_SELECTED_CONTEXT_BUNDLE` | Exact bundle-selected intervals are serialized as visible text. It is `NO_QUALIFIED_ADAPTER` when no exact qualified bundle exists. |
| `PRA_NATIVE_MEMORY_BUNDLE` | The same bundle selection is consumed as native K/V, or remains `NO_QUALIFIED_ADAPTER`. |
| Oracle gold documents | Packs available gold-document chunks. This is a research diagnostic and is excluded from headline deltas. |

The model, prompt, generation settings, engine, and hardware must match within a model-backed comparison. Selected Context and Native Memory must carry the same condition-independent selection-receipt digest. Oracle rows never count as PRA improvements.

## Metrics remain separate

**First-stage retrieval:** document Recall@K and whether all gold documents entered the candidate set.

**Context selection:** supporting-document and supporting-span coverage, MRR, NDCG, gold/false selected-document fractions.

**Physical context:** logical candidate tokens, packed/materialized tokens, selected/full ratio, discarded tokens, and materialization avoidance.

**Answer:** dataset-native EM and token F1. The evidence-availability probe is reported under its own label and is never substituted for generation EM/F1.

**Serving:** TTFT, ITL, completion latency, output tokens/s, ingestion/index/native-encoding time, active detail bytes, and reuse. A value is absent rather than estimated when an engine cannot expose it.

## Failure labels

Per-example results use the stable failure classes `FIRST_STAGE_RETRIEVAL_MISS`, `STANDARD_RAG_PACKING_MISS`, `PRA_SELECTOR_MISS`, `PRA_DISTRACTOR_SELECTION`, `PRA_MATERIALIZATION_MISS`, `GENERATION_FAILURE`, `ANSWER_FORMAT_FAILURE`, `NATIVE_REALIZATION_MISMATCH`, and `BUNDLE_SELECTOR_REGRESSION`. This prevents an answer miss from being attributed automatically to routing.
