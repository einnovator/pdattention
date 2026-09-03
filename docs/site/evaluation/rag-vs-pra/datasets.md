# Datasets

## MultiHop-RAG

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) is the first natural benchmark. Its official ODC-BY release contains 2,556 questions, 609 news documents, and two to four evidence documents per question. The loader pins upstream repository and dataset blob identities, maps evidence by canonical URL, and retains source metadata.

The first held-out cohort contains 50 questions selected with seed 11. Two stages use those same question identities:

- **L1 fixed candidates:** missing gold documents are inserted deterministically after retrieval so context selection can be studied independently of first-stage misses.
- **L2 real retrieval:** unmodified BM25 top-K results expose the complete retrieval-plus-context pipeline.

## Controlled fixture

L0 contains 60 versioned documents and 15 lookup, bridge, and synthesis questions. It includes paraphrased/shared-entity distractors and long irrelevant records. It is a software-validation fixture, not publishable benchmark evidence.

## Next datasets

| Level | Dataset | Role | Status |
| --- | --- | --- | --- |
| L3 | KILT Natural Questions | Large-corpus open-domain QA | Adapter and knowledge-source acquisition pending |
| L4 | KILT HotpotQA | Open-corpus multi-hop QA | Pending after NQ |
| L4B | KILT TriviaQA | Independent open-domain replication | Pending |
| L5 | BEIR NQ, HotpotQA, SciFact, FiQA, NFCorpus | Retrieval diagnostics | Pending |
| L9 | CRAG and dynamic retrieval | Agentic evaluation | Deliberately deferred |

KILT rows require the pinned KILT knowledge source, not a convenience corpus assembled from validation answers. Until that source and index are present, they remain unmeasured rather than approximated.
