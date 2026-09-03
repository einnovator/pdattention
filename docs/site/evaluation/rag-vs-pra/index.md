# RAG and PRA

In the initial RAG evaluation, PRA does not replace first-stage corpus retrieval. The external retriever determines a candidate document set. PRA changes how those candidate documents remain addressable, are selected, materialized, reused, and presented to the model.

The central comparison is therefore:

```text
same corpus + same external retriever + same ordered candidate documents
    |-- Standard RAG: rank chunks and serialize selected text
    `-- RAG + PRA: retain typed documents and materialize selected regions
```

This design asks whether PRA preserves access to more useful evidence at a matched physical-context budget, and whether persistent native state improves repeated-query economics. It does not assume that every workload benefits.

## Current evidence

- **L0:** a deterministic 60-document fixture validates identities, chunk boundaries, budgets, receipts, and failure labels.
- **L1:** 50 official MultiHop-RAG questions use gold-present candidate sets of 5, 10, 20, and 50 documents at 2K, 4K, 8K, and 16K physical-token budgets.
- **L2:** the same cohort uses real BM25 first-stage retrieval over all 609 official corpus documents.
- **L6 diagnostic:** 50 queries measure cumulative visible text and unique newly materialized chunks over a persistent corpus.

The L1/L2 grid currently uses a deterministic answer-availability probe. It is selection evidence, not a model answer-quality result. Model-backed and engine-native rows are labeled separately.

## Read next

Start with [Methodology](methodology.md), then review [Results](results.md). The [Reproduction](reproduction.md) page includes exact commands and artifact locations.
