# AGENTS.md — PRA Paper 2.6: Hybrid Reference Discovery

## Mission and placement
Paper 2.6 sits between Paper 2.5 associative discovery and Paper 3 materialization.

- Paper 2.5: can bounded iterative PRA traverse memory-to-memory associations?
- Paper 2.6: which signals should discover candidate references/chunks?
- Paper 3: which discovered candidates, and how much detail, should be materialized?
- Paper 3.5: when should retrieval retry, stop, or spend more compute?

The central hypothesis is that contextual hidden-state gists are useful semantic indexes but are not universal addresses. Because PRA runs beside the LLM and therefore has the model tokenizer available, reference resolution should operate primarily in the model's own token space rather than through a separate word-level lexical representation. Multi-hop evidence can expose exact token spans, names, titles, aliases, approximate subword sequences, or explicit references that should be exploited directly.

## Required repository audit
Before coding, read the latest Paper 0, Paper 1.5, Paper 2, Paper 2.5 branch, and Paper 3 branch/design. Record current router/index APIs and reuse them. In particular preserve Paper 2.5's bounded iteration rather than creating a second iterative engine.

Paper 0's vocabulary is authoritative: distinguish logical, requested, resident, and materialized memory. Preserve the separation among naming, discovery/routing, admission, and native attention.

## Claims to test
H1. Hybrid discovery improves supporting-evidence recall at a matched candidate/materialization budget.
H2. Entry hops and later hops favor different signals: semantic evidence may dominate entry routing, while exact or approximate token-reference evidence may dominate some later hops.
H3. Sparse reference following can match broad semantic top-k with fewer distractors.
H4. Candidate provenance and calibrated confidence are useful to later materialization/control.
H5. Hybrid retrieval is regime-dependent and may not help tasks with weak token-level continuity.
H6. Discovery improvements may or may not survive end-to-end answer generation; this must be tested explicitly rather than inferred from retrieval gains.

### Two-level success gate
Report two separate success levels.

**Discovery-level positive result:** a proposed condition improves required-evidence recall at matched active/materialization cost, or matches recall with materially lower active memory/cost, with statistical support and causal controls.

**End-to-end positive result:** the discovery improvement also produces a resolved downstream QA improvement (e.g. EM/F1 with an appropriate confidence interval or paired test) on at least one natural multi-hop dataset.

A discovery-level result is scientifically valid for Paper 2.6 because discovery and materialization/consumption are separate mechanisms. However, the abstract, title, and conclusion must never imply improved reasoning or QA accuracy unless the end-to-end gate is also passed.

Do not claim that token matching itself is novel, that classical lexical retrieval is unnecessary, that gists are obsolete, that answer-code probes are full HotpotQA, or that better retrieval necessarily implies better generation.

## Candidate representation
Do not prematurely collapse all evidence into one scalar. Introduce a provenance-preserving candidate record with:
- URI/reference/chunk/layer identity;
- semantic score;
- exact token-span score/flag;
- weighted token-overlap score;
- ordered-subsequence / token n-gram score;
- approximate token-match score;
- optional token-embedding similarity score;
- classical lexical/BM25 score (baseline channel);
- entity/name score;
- explicit-match flag;
- associative-edge score;
- discovery hop and parent;
- provenance set;
- calibrated confidence/rank.

### Confidence semantics
Treat candidate confidence as an estimate of referent validity, not merely a normalized retrieval score:

`c_i ~= P(candidate i is a valid referent | query, evidence channels, provenance)`.

Calibration may be imperfect initially, but the semantics must remain stable across experiments. Preserve raw channel scores and provenance so confidence can be calibrated conditionally on discovery mode. Exact-token confidence, approximate-token confidence, and semantic-gist confidence must not be assumed directly comparable.

Evaluate Brier score, ECE/reliability diagrams, and selective accuracy/coverage where sample size permits. This interface is intentionally useful downstream:
- high confidence -> materialize / permit action;
- intermediate confidence -> search more / disambiguate;
- low confidence -> abstain or request clarification.

Paper 2.6 evaluates retrieval/control implications only; external agent actions belong to the later agent paper.

Conceptually:
C_h = C_explicit ∪ C_token-exact ∪ C_token-approx ∪ C_semantic ∪ C_assoc,
with classical lexical/BM25 candidates retained as an external baseline/comparison channel rather than the main proposed representation.

Exact explicit references may bypass approximate retrieval. Approximate channels generate candidates; materialization remains downstream.

## Mechanisms
Implement each behind independent flags. The proposed method is **token-native reference matching**, using the same tokenizer as the host LLM. Classical word-level lexical retrieval remains a baseline.

1. Existing gist router — unchanged semantic baseline.
2. Explicit reference / URI resolution — deterministic when a valid typed handle is present.
3. Exact token-span matching — contiguous tokenizer-ID sequence matches over titles, aliases, names, and optionally chunk text.
4. Token normalization — conservative handling of case, whitespace/punctuation markers, tokenizer-specific leading-space artifacts, and decoded surface spans. Preserve raw token IDs as well as normalized spans.
5. Information-poor token suppression/weighting — compare a fixed stop-token strategy with corpus-derived token IDF. Prefer IDF-like weighting because tool/domain corpora can make words such as `get`, `create`, `api`, or `tool` effectively stop terms.
6. Weighted token overlap — score candidate spans using discriminative token weights rather than unweighted set overlap.
7. Ordered token matching — longest-common-subsequence-like, ordered-subsequence, and tokenizer n-gram matches to preserve multi-token identity while tolerating intervening material.
8. Approximate token matching — only after exact/high-confidence matching fails or is ambiguous. Compare bounded token edit distance and similarity over decoded token/subword strings. Keep thresholds explicit and auditable.
9. Optional model-token embedding similarity — use the host model's input embedding matrix to compare near tokens/spans as a bridge between discrete token identity and fully contextual gist similarity. Treat this as an ablation, not automatically as the default.
10. Classical lexical retrieval (BM25/FTS) over words/text — baseline for comparison with token-native matching, not the centerpiece of the proposed method.
11. Dataset-provided entity/title anchors first; automatic anchor extraction only as a later ablation.
12. Token-native + semantic candidate union with deduplication and provenance preservation.
13. Token-native candidate generation → semantic reranking.
14. Semantic candidate generation → token-native reranking.
15. Iterative hybrid traversal — selected evidence exposes token anchors for the next bounded Paper-2.5 hop.

### Progressive resolution cascade
Evaluate a cascade rather than only score fusion:

`explicit ref -> exact token span -> weighted/ordered token match -> approximate token match -> semantic gist`

A high-confidence earlier stage may resolve or sharply narrow candidates without running all later stages. Always retain a union/fusion condition as a comparison.

### Token-index representation
For every indexed resource keep, where feasible:
- raw tokenizer IDs;
- normalized tokenizer spans;
- decoded token/subword strings;
- token document frequencies / IDF;
- title/name/alias boundaries;
- optional token n-gram postings.

Do not assume BPE/subword tokenization gives stable word boundaries. Evaluate leading-space and alternate-segmentation artifacts explicitly.

Measure index build separately from cold/warm query cost.

## Why token-native rather than purely lexical?
The matching runtime already has the host model tokenizer. Reusing it avoids a second vocabulary and lets the resolver compare the same discrete symbols that produced the model state and cached K/V.

The token-native matcher should be hierarchical:
- exact contiguous token span;
- normalized exact span;
- weighted content-token overlap;
- ordered subsequence / token n-gram evidence;
- approximate token match;
- contextual semantic gist fallback.

This is deliberately not “semantic similarity at token level.” Exact and approximate identity evidence should remain distinguishable from contextual semantic evidence. The model embedding matrix may be tested as an intermediate ablation.

Classical BM25 remains important as a strong external baseline. If BM25 dominates token-native matching at equal cost, that is a meaningful negative result.

## First-class Tokenization Perturbation Suite
Do not bury tokenizer fragility inside a generic tokenizer-family ablation. Build a dedicated invariance/robustness suite in which the underlying referent is held constant while its surface/token realization changes.

Perturbations should include:
- case changes;
- leading/trailing whitespace and tokenizer leading-space artifacts;
- punctuation insertion/removal;
- hyphenation and separator changes;
- concatenation/splitting of multiword names;
- alternate subword segmentations when naturally induced;
- minor misspellings/typos;
- abbreviations and aliases;
- stop/information-poor token insertion;
- Unicode/normalization variants where supported safely.

Measure degradation separately for exact token span, weighted/ordered token matching, approximate token matching, cascade, and iterative hybrid conditions.

The key quantity is not only average accuracy but the invariance curve: how quickly each method degrades as the same referent moves farther from exact token identity.

## Datasets
Stage A: controlled synthetic multi-hop chains with exact token-span aliases, alternate tokenizer segmentations, stop-token insertions, fuzzy/approximate aliases, ambiguous names, semantic-only hops, token-only hops, semantic→token and token→semantic paths.

Stage B: HotpotQA as the primary natural benchmark. Separate supporting-document/fact retrieval from full answer generation.

Stage C: add only 2–3 structurally distinct multi-hop datasets after auditing their reference structure. Strong candidates are 2WikiMultiHopQA, MuSiQue, and QASPER as a non-entity-centric contrast.

## Minimum experimental matrix
- B0 gist-only PRA.
- B1 classical BM25/word-lexical only.
- B2 exact token-span/name only.
- B3 weighted/ordered token matching only.
- B4 approximate token matching only.
- H1 semantic ∪ token-native union.
- H2 token-native → semantic rerank.
- H3 semantic → token-native rerank.
- H4 progressive token-resolution cascade + semantic fallback.
- H5 iterative hybrid token/semantic traversal — main proposal.
- A1 broad semantic top-k — “buy recall with breadth.”
- O1 annotated/oracle routing ceiling.

Match final candidate and materialization budgets. A hybrid method must not silently get twice the active budget.

## Metrics
Discovery: supporting-document recall@k, supporting-fact recall, MRR, NDCG, precision, missed-evidence rate, hop success, path completion.

Sparsity/materialization interface: discovered candidates, requested chunks, materialized chunks/tokens/KV, active fraction, distractor materialization, required-evidence recall per materialized token.

Task: answer EM/F1 and loss/calibration where appropriate. Downstream QA is a primary validation axis, not an afterthought; always report whether retrieval gains survive generation.

Cost: index construction, cold/warm routing, per-hop cost, transfer, materialization/attention, end-to-end latency.

Causality: valid, disabled, shuffled, irrelevant, empty, oracle, and full-context controls consistent with Paper 0.

## Critical ablations
Semantic top-k; token-native top-k; BM25 top-k; shared union budget; hop depth; branching factor; raw token IDs vs normalized spans; fixed stop-token removal vs token IDF weighting; exact span vs token n-gram vs ordered subsequence; exact-only vs approximate fallback; token edit-distance threshold; decoded-subword similarity; optional model-token embedding similarity; title/name/alias-only vs full chunk token index; provided vs extracted anchors; per-layer vs shared token candidate sets; cascade vs fusion vs union vs reranking; static query vs anchors exposed by newly retrieved evidence; confidence calibration with/without provenance; abstention thresholds; contradictory token-vs-semantic evidence; model size; tokenizer family; chunk granularity; gist count; sparse vs all PRA layers.

## Wrong-reference and contradictory-evidence stress tests
Hybrid routing must be evaluated when token-native evidence is confidently wrong or ambiguous, not only when it helps.

Construct controlled cases with:
- two resources sharing most discriminative tokens;
- aliases referring to different entities;
- high token overlap with the wrong referent;
- a typo that is closer to the wrong candidate;
- exact/near-exact token evidence that conflicts with stronger contextual semantic evidence;
- semantic evidence favoring one candidate while token evidence favors another;
- nonexistent requested references;
- distractor candidates deliberately engineered to produce high token scores.

Measure:
- top-1/top-k referent accuracy;
- confidence assigned to wrong candidates;
- recovery after an incorrect first hop;
- downstream evidence/QA degradation;
- selective accuracy as the confidence threshold changes.

Report whether errors degrade gracefully or catastrophically. A hybrid method that improves average recall by becoming overconfident in wrong token matches is not acceptable.

## Fairness controls
Prevent gold-label leakage through metadata. Log exactly what text/state is queried at each hop. Do not count “discovered” as “materialized.” Keep final budgets matched. Establish retrieval improvements before attributing downstream QA gains to reasoning.

## Paper 3 handoff
Paper 2.6 should leave a generic interface:
`discover(query_state, memory_index, budget, policy) -> list[Candidate]`

Paper 3 owns:
`materialize(candidates, kv_budget, policy) -> ActiveMemory`

If hybrid wins, Paper 3 should adopt it as a principal discovery condition while retaining gist-only and other discovery modes as baselines. Paper 3 must remain valid even if hybrid is dataset-dependent.

## Codex execution phases
0. Audit current branches/APIs; write `paper2_6_audit.md`; no algorithm changes.
1. Add tokenizer-native indexes (raw/normalized spans, IDF, n-grams) plus BM25 baseline and tests.
2. Add provenance-preserving candidate records while preserving exact old behavior when disabled.
3. Run synthetic and one-hop HotpotQA retrieval.
4. Reuse Paper-2.5 iteration for iterative hybrid discovery.
5. Run full QA only after retrieval gains are established.
6. Replicate on selected datasets.
7. Produce Paper-3 handoff note and update Paper 0 roadmap if warranted.

## Acceptance gates
### Gate A — discovery
A headline discovery result requires a hybrid/token-native condition to improve required-evidence recall at matched active budget, or match recall with materially lower active memory/cost, with statistical support and while surviving shuffled/irrelevant controls, tokenization perturbations, and wrong-reference stress tests.

### Gate B — end-to-end utility
A headline QA/reasoning result additionally requires the discovery improvement to produce a resolved downstream EM/F1 gain on at least one natural multi-hop dataset. If Gate A passes but Gate B does not, report the result explicitly as a discovery/efficiency improvement and treat the missing downstream gain as a principal limitation and handoff to Paper 3.

A clean negative/regime result is preferable to forcing a positive conclusion.

## Writing rules
Separate observation, interpretation and hypothesis. State model, dataset, seeds and budget for every headline number. Use “discovery” for candidate finding and “materialization” for native-KV admission. Prefer “token-native reference matching” for the proposed non-neural matcher; reserve “lexical/BM25” for the classical baseline. Never describe token/BM25 retrieval as attention. Do not imply online decode-resume recursion unless implemented.
