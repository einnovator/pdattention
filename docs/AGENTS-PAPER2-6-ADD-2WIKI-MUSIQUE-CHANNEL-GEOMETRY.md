# AGENTS — Paper 2.6 Extension: 2Wiki + MuSiQue and Retrieval-Channel Geometry

## Mission
Extend Paper 2.6 (`hybrid-pra`) beyond QASPER/HotpotQA by adding:
- 2WikiMultiHopQA
- MuSiQue

The current result already shows opposite discovery regimes:
- QASPER favors exact/token-native routing
- HotpotQA favors BM25 on the current slice
- static hybrid improves over gist on Hotpot but does not beat the best lexical channel
- iterative hybrid can recover newly exposed lexical references in controlled tests

The next question is:

> Are these differences merely dataset-specific, or can retrieval-channel preference be explained by query/evidence geometry?

Do not merely add two benchmark rows. Use 2Wiki and MuSiQue to identify what properties make semantic, exact, BM25, approximate-token, or iterative hybrid retrieval useful.

## 1. Keep one channel matrix across all four datasets

Evaluate:

| Channel | QASPER | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|---|
| Native/gist semantic | yes | yes | yes | yes |
| Exact token/span | yes | yes | yes | yes |
| BM25 | yes | yes | yes | yes |
| Approximate token | yes | yes | yes | yes |
| Static hybrid | yes | yes | yes | yes |
| Iterative hybrid | yes | yes | yes | yes where applicable |

Do not silently change chunk budgets or candidate limits across channels.

## 2. Match retrieval budget

For fair comparisons match:
- requested chunks
- requested roots
- max returned chunks
- chunk granularity
- iteration depth where applicable

For iterative methods report:
- budget per step
- total retrieved chunks
- total search comparisons

Do not compare a 4-chunk one-shot baseline against a 16-chunk iterative method without explicit accounting.

## 3. Reuse Paper 2.6 metrics

Keep:
- evidence recall
- precision
- MRR where defined
- root recall
- complete-path recovery where relevant
- paired differences
- bootstrap CIs

Add where available:
- evidence-region recall
- evidence-token coverage
- channel disagreement
- selected distractor count

## 4. Add 2Wiki

Hypothesis to test, not assume:
- more explicit entity/relation mentions
- strong successor topology from Paper 2.5
- relatively compact evidence

Questions:
- exact vs semantic?
- BM25 vs exact?
- hybrid vs best lexical?
- does iterative hybrid help after first-hop evidence exposes the next entity/relation?
- is successor retrieval easier than root retrieval?

## 5. Add MuSiQue

Hypothesis to test:
- more distributed evidence
- harder root routing
- larger oracle-facet gap
- lexical overlap may be weaker or less uniformly useful

Questions:
- does semantic gist recover evidence missed by exact/BM25?
- does lexical identity become useful at later hops?
- does static hybrid improve recall but reduce precision?
- does iterative hybrid become more useful than static fusion?
- does evidence dispersion increase the value of state-dependent retrieval?

## 6. Dataset/query geometry descriptors

For each example derive where possible:

### Query lexicality
- fraction of query tokens appearing verbatim in gold evidence
- rare-token overlap
- IDF-weighted overlap
- named-entity overlap
- relation-token overlap
- answer/evidence lexical overlap if appropriate

### Evidence geometry
- number of evidence regions
- number of evidence documents
- total evidence tokens
- maximum evidence span
- average evidence gap
- chain depth
- evidence compactness

### Root ambiguity
- semantic root rank
- exact root rank
- BM25 root rank
- top1/top2 score gap
- channel disagreement

### Successor geometry
- successor rank by channel
- lexical exposure at hop 1
- whether later evidence contains a newly exposed reference absent from the original query

## 7. Channel advantage variables

Compute per example:

`delta_exact_gist = recall_exact - recall_gist`

`delta_bm25_gist = recall_bm25 - recall_gist`

`delta_approx_gist = recall_approx - recall_gist`

`delta_hybrid_gist = recall_hybrid - recall_gist`

Also:

`delta_hybrid_best_single = recall_hybrid - max(recall_gist, recall_exact, recall_bm25, recall_approx)`

This is critical. Hybrid beating gist but losing to BM25 is not a hybrid win.

## 8. Precision–recall analysis

Report both evidence precision and recall for every channel.

If later materialization is evaluated also track:
- distractor chunks
- distractor K/V
- attention-weighted evidence share

Paper 2.5/3 show false positives can hurt downstream computation.

Therefore higher recall with much lower precision may not be the best PRA discovery policy.

## 9. Retrieval regimes

Try to identify examples as:

### Semantic regime
Relevant evidence is semantically related but lexically weak.

### Identity/reference regime
Rare names, exact spans, IDs, URLs, citation markers, or specific terms dominate.

### Bridge regime
Correct root is ambiguous, but retrieved evidence exposes the next address.

### Mixed regime
Semantic and lexical channels recover complementary pieces.

Do not force a label when evidence is ambiguous.

## 10. Adaptive channel-selector baseline

Before a learned selector, implement simple diagnostic rules such as:

```text
if strong rare exact span:
    exact/BM25
elif lexical and semantic agree:
    intersection/rerank
elif semantic strong and lexical weak:
    gist
elif new reference exposed after hop:
    iterative lexical/hybrid
else:
    keep candidates from multiple channels
```

These are baselines, not final policy.

## 11. Learned channel selector only after geometry analysis

If sample size allows, train an observable-only selector.

Input may include:
- query lexicality
- entity overlap
- root score gaps
- channel disagreement
- deployment-available prompt features

Output:
- gist
- exact
- BM25
- approx
- hybrid
- iterative hybrid

Do NOT use gold-derived evidence geometry as deployment features. Gold geometry is explanatory only.

## 12. Oracle channel upper bound

Compute per example:

`R_oracle_channel = max_c R_c`

Then compare:
- best fixed channel
- static hybrid
- adaptive selector
- oracle per-example channel

Define:

`headroom = R_oracle_channel - R_best_fixed`

This tells us whether adaptive channel selection is worth pursuing.

## 13. Channel diversity

Measure overlap of recovered evidence across channels.

Ask:
- are semantic and lexical channels redundant?
- do they retrieve complementary gold evidence?
- does fusion add useful diversity or mainly distractors?

Use Jaccard/overlap statistics where appropriate.

## 14. Static fusion review

Do not assume one weighted formula should dominate.

Secondary comparisons may include:
- union/max
- rank fusion
- reciprocal-rank fusion
- calibrated score fusion only if channel scores are comparable

Primary question remains adaptive channel selection vs static fusion.

## 15. Iterative hybrid analysis

Separate:

### Hop 0
What is retrievable from the original query?

### Hop 1+
What new tokens/entities/references become visible after retrieving evidence?

Define a useful metric such as:

`NewAddressRate = P(retrieved evidence exposes a useful address absent from Q)`

Then test whether iterative-hybrid gains are concentrated on those examples.

## 16. Wrong-reference robustness

Keep current controls:
- clean
- case variation
- punctuation
- typo
- confidently wrong reference

Add if cheap:
- near-entity collision
- shared-prefix entity
- numeric-ID collision
- URL/domain overlap
- alias/synonym

Measure:
- target recovery
- wrong-target recovery
- confidence
- abstention/retry opportunity

## 17. Approximate matcher

Review whether current matching is:
- character-level
- token-level
- edit-distance
- BPE-aware

Do not turn this into a fuzzy-search paper. Goal is robustness characterization.

## 18. Sample size

Expand natural cohorts before stronger dataset claims.

Target if practical:
- >=50 held-out units per dataset
- preferably 100+

Use validation for fusion/calibration choices.

Prefer more examples before more fusion complexity.

## 19. Required dataset table

| Dataset | Best fixed channel | Gist recall | Exact recall | BM25 recall | Hybrid recall | Precision | Oracle-channel headroom |
|---|---|---:|---:|---:|---:|---:|---:|

Add uncertainty.

## 20. Required geometry table

| Geometry feature | Semantic advantage | Lexical advantage | Hybrid advantage |
|---|---:|---:|---:|
| high exact entity overlap | | | |
| low lexical overlap | | | |
| high IDF overlap | | | |
| distributed evidence | | | |
| new address after hop | | | |
| high channel disagreement | | | |

Only fit regression/effect sizes if sample size supports it.

## 21. Paper 3.5 handoff

If preference is heterogeneous, add a forward connection:

`M_search ∈ {semantic, exact, BM25, approx, hybrid, iterative}`

Paper 3.5 can then choose:
- retrieval channel
- search effort

Do not move the full adaptive-controller study into Paper 2.6.

## 22. Paper 3 handoff

Keep stages distinct:

`channel discovery -> conceptual chunks -> materialization -> attention`

Do not imply retrieval recall improvements are answer-quality improvements.

## 23. Preferred main conclusion

If supported:

> PRA memory discovery is multi-representational. Semantic similarity, lexical identity, and dynamically exposed references carry complementary but query-dependent information. Static hybrid fusion can repair some semantic failures but does not dominate the best single channel. The stronger opportunity is adaptive channel selection and state-dependent retrieval.

Avoid:
> Hybrid search is universally better.

## 24. Required artifacts

Create/update:
- `channel_results_qasper.csv`
- `channel_results_hotpot.csv`
- `channel_results_2wiki.csv`
- `channel_results_musique.csv`
- `channel_geometry_features.csv`
- `channel_advantage_rows.csv`
- `channel_precision_recall.csv`
- `channel_oracle_headroom.csv`
- `channel_overlap.csv`
- `iterative_new_address.csv`
- `wrong_reference_robustness.csv`
- `channel_selector_baselines.csv`
- `paper2_6_findings.json`
- `claim_audit.md`

## 25. Required plots

1. Recall by channel × dataset
2. Precision by channel × dataset
3. Precision-recall scatter
4. Oracle-channel headroom by dataset
5. Lexical overlap vs lexical-semantic advantage
6. Evidence dispersion vs channel advantage
7. Channel disagreement vs oracle headroom
8. New-address rate vs iterative-hybrid gain
9. Wrong-reference robustness
10. Static hybrid vs adaptive-selector frontier if selector is implemented

## 26. Tests

Add:
- 2Wiki loader/evidence mapping
- MuSiQue loader/evidence mapping
- equal requested-budget assertions
- exact/BM25/approx/hybrid parity
- deterministic channel results
- geometry-feature extraction
- no gold leakage into selector
- oracle-channel computation
- precision/recall accounting
- iterative new-address detection

Keep offline reproducibility.

## 27. Stop rules

Do NOT:
- add many fusion algorithms before the four-dataset comparison
- tune on held-out
- make dataset-level laws from tiny slices
- optimize recall while ignoring precision
- claim QA/output improvement without generation
- fold Paper 3.5 adaptive control into this paper prematurely

## Core Principle

Paper 2.6 should move from “lexical helps semantic search” to a more general result: **PRA discovery is multi-representational, and the best retrieval channel depends on query/evidence geometry**. QASPER and Hotpot already demonstrate different lexical regimes. 2Wiki and MuSiQue should test whether this variation is explained by entity/reference explicitness, lexical overlap, evidence dispersion, bridge structure, and dynamically exposed references. The eventual architecture should not assume one static fusion rule; retrieval channel itself should become an adaptive control decision.
