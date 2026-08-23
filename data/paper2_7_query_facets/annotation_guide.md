# Natural query-facet annotation guide

## Scope

The benchmark maps dataset-authored reasoning metadata onto the original query
surface. MuSiQue contributes its `question_decomposition`; 2WikiMultiHopQA
contributes ordered `evidences` relation triples. These records are natural
questions, but the span mapping is deterministic dataset-derived supervision,
not a new human annotation campaign.

## Unit and facet rules

- Units are Unicode word/punctuation spans with exact character offsets.
- Each MuSiQue decomposition step is one facet.
- 2Wiki compositional questions use one facet per evidence relation; comparison
  questions group connected evidence relations into their parallel query branches.
- Exact multi-token source phrases anchor otherwise ambiguous surface words.
- Unique lexical matches anchor units to facets.
- Answer-type words map to the terminal facet.
- Unmatched content words map to the nearest source anchor.
- Function words, punctuation, and multiply matched terms remain shared/global.
- Shared/global units are excluded from primary ARI, NMI, and pairwise F1.
- Non-contiguous facets are permitted and measured.

## Reliability boundary

The mapper is deterministic and rerunnable, but no human inter-annotator
agreement is claimed. Source strings, token decisions, confidence, and mapping
reasons are persisted for audit. Generated LLM subquestions are evaluated only
as predictions and never used as gold labels.
