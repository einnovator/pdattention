# AGENTS — Paper 2.6 Final Iteration: Cohorts, Disambiguation, Handoff, and Related Work

## Mission
Continue `hybrid-pra` from commit `48f8ac2`. Perform a focused strengthening pass, not another mechanism expansion.

Goals:
1. slightly enlarge natural cohorts if feasible;
2. strengthen confidence/referential-disambiguation diagnostics;
3. finalize the root/successor search-method action-space handoff to Paper 3.5;
4. add a proper Related Work section and explicit links to earlier PRA papers.

Current findings to preserve:
- root/successor channel differs on 3/4 datasets;
- QASPER exact→exact;
- Hotpot exact→native semantic;
- 2Wiki gist→exact-new-address;
- MuSiQue approximate→native semantic;
- static hybrid and RRF do not beat best fixed channels;
- true adaptive headroom exists but is modest;
- simple observable selectors do not close it;
- useful-address exposure is conditionally useful;
- typo, alias, and confidently-wrong references remain weak points.

## 1. Cohort expansion
Current study has 132 identities / 792 matched-budget routes.

Increase held-out coverage where cached/offline execution is cheap. Target if practical:
- >=50 identities/dataset;
- preferably 75–100/dataset.

Priority: Hotpot, 2Wiki, MuSiQue, then QASPER.

Keep exactly the same:
- chunking;
- root/successor budgets;
- channel definitions;
- evidence mapping;
- metrics.

Do not change the mechanism while increasing N.

Where possible stratify new examples by deployment-observable geometry:
- high/low IDF overlap;
- explicit reference vs diffuse lexical query;
- bridge/direct;
- short/long query;
- low/high channel disagreement;
- candidate hop-one address presence.

Never stratify by final winning channel.

## 2. Re-estimate stability
Recompute all root/successor recalls, precisions, transition frequencies, selector results, oracle headroom, and useful-address effects with paired/bootstrap intervals.

Add subsampling/bootstrap stability:
> How often would each channel be selected as “best” at different cohort sizes?

Separate:
`H_selection = oracle per-example - best held-out fixed`
from
`H_validation = best held-out fixed - validation-selected`.

This must remain explicit.

## 3. Confidence/disambiguation focus
The main remaining weakness is not raw matching but knowing whether a strong match is referentially valid.

Distinguish:
`match confidence != referential validity`.

For each channel record confidence signals.

Semantic:
- top score;
- top1/top2 gap;
- entropy/effective support where meaningful;
- query/current-state similarity.

Exact:
- span length;
- matched-token count;
- IDF/rarity;
- corpus occurrence count;
- candidate count sharing reference.

BM25:
- top score;
- top1/top2 gap;
- matching-term count;
- rare-term contribution;
- score concentration.

Approximate:
- edit/token similarity;
- matched span length;
- ambiguity count;
- exact-vs-approx gap;
- alias/partial-mention indicator.

Cross-channel:
- top-candidate agreement;
- top-k overlap;
- semantic consistency of lexical candidate;
- lexical support for semantic candidate.

## 4. Referential-validity diagnostics
Test whether wrong lexical matches can be detected using:
- entity/type consistency;
- surrounding semantic compatibility;
- relation compatibility;
- current-path consistency;
- candidate uniqueness;
- source/document consistency;
- alias tables where available without gold leakage.

Do not claim entity resolution is solved.

## 5. Robustness controls
Retain:
- clean;
- case;
- punctuation;
- typo;
- alias/synonym;
- confidently wrong reference;
- near entity;
- shared prefix;
- numeric ID;
- URL/domain overlap.

Add only if cheap:
- same name, wrong entity;
- same class, wrong instance;
- correct entity, wrong relation;
- stale/alternate alias;
- two plausible references with similar lexical confidence.

These are more useful than arbitrary noise.

## 6. Confidence calibration
For deployment-observable signals evaluate prediction of:
- correct root;
- correct successor;
- true UsefulAddress;
- wrong-reference failure.

Report as appropriate:
- AUROC;
- AUPRC;
- ECE/calibration;
- reliability bins;
- precision at conservative thresholds.

Tune thresholds on validation only.

## 7. Cross-channel consistency gate
Test a simple diagnostic:
> trust lexical candidate only when sufficiently consistent with semantic/current-state evidence.

Compare:
- lexical confidence alone;
- semantic confidence alone;
- consistency-gated lexical;
- static fusion.

This is a diagnostic gate, not a new universal hybrid algorithm.

## 8. Abstention/retry opportunity
Measure whether conservative confidence thresholds can reject wrong routes:
- wrong routes rejected;
- correct routes rejected;
- retained recall;
- precision after abstention.

Export these signals for Paper 3.5. Do not implement the full retry agent here.

## 9. UsefulAddress proxy
Keep gold diagnostic:
`UsefulAddress = Exposed AND GoldLinked AND CompetitiveRank`.

Add observable proxies:
- rarity;
- uniqueness;
- candidate count;
- exact/approx confidence;
- semantic consistency;
- relation consistency.

Measure whether these predict true UsefulAddress.

## 10. Finalize action-space handoff
Export/update `search_method_action_spec.json`.

Root actions:
- semantic
- exact
- bm25
- approximate
- hybrid

Successor actions:
- native_semantic
- exact_new_address
- bm25_state
- approximate_new_address
- hybrid_state

For each specify:
- implementation identifier;
- root/successor/both;
- required state/index;
- parameters;
- confidence outputs;
- cost metrics;
- known failure modes.

Paper 3.5 must be able to consume this without redefining methods.

## 11. Explicit Paper 3.5 handoff section
State clearly:
> Paper 2.6 establishes the search-method action space and its empirical heterogeneity. Paper 3.5 treats root and successor methods as adaptive actions jointly with query interpretation, breadth/depth, and admission budgets.

Keep `S_root` and `S_succ` separate.

## 12. Relation to earlier PRA papers
Add a concise “Relation to the PRA Series” subsection.

### Paper 0 / position paper
- logical/unbounded external memory;
- on-demand activation;
- virtual-memory analogy.
Paper 2.6 studies discovery over that logical memory.

### Paper 2 / HF integration
- frozen pretrained backbones;
- native K/V memory;
- routing bottleneck.
Paper 2.6 improves discovery; it does not retrain the consumer.

### Paper 2.5 / associative memory
Strong link:
- query→root often harder than memory→memory traversal;
- topology/traversal;
- distractor competition;
- dataset-dependent root geometry.
Paper 2.6 adds multiple retrieval representations for entering/traversing associative memory.

### Paper 3 / materialization
Boundary:
`discovery -> conceptual selection -> materialization -> attention`.
Paper 2.6 chooses candidate conceptual memory; Paper 3 decides physical K/V participation.

### Paper 3.5 / adaptive control
Paper 2.6 exports methods/confidence signals; Paper 3.5 selects them adaptively.

### Paper 5 / scaling
Optional forward link:
at large logical memory, retrieval/index choice affects search cost. Do not duplicate scaling experiments.

Mention Paper 1.5 only where positional/native-memory transport is genuinely relevant.

## 13. Add a focused Related Work section
Organize into four groups.

### A. Sparse/retrieval-augmented attention and long-context memory
Compare:
- retrieval unit;
- external text vs native K/V;
- semantic/lexical representation;
- iterative/state-dependent retrieval;
- selective physical materialization.

### B. RAG and hybrid retrieval
Cover:
- dense retrieval;
- BM25/sparse retrieval;
- dense+sparse hybrid;
- reranking;
- reciprocal-rank fusion;
- multi-stage retrieval.

Be explicit: hybrid IR itself is not the novelty. Paper 2.6 applies multiple representations to PRA-native memory and finds root/successor preference changes with state.

### C. Multi-hop / iterative retrieval
Cover systems where retrieved evidence:
- reformulates/expands the next query;
- exposes intermediate entities;
- changes subsequent retrieval.

Compare with PRA’s state-dependent native-memory traversal.

### D. Entity/reference resolution and approximate lexical matching
Cover:
- entity linking/disambiguation;
- aliases;
- fuzzy matching;
- approximate string retrieval.

Use this literature to frame typo/alias/wrong-reference limitations.

## 14. Literature search requirements
Before editing Related Work, perform a current literature search. Prefer original papers, conference/arXiv versions, and official repos.

Search themes:
- hybrid dense sparse retrieval
- BM25 + dense retrieval / hybrid RAG
- iterative retrieval multi-hop QA
- query reformulation multi-hop retrieval
- retrieval-augmented attention / KV retrieval
- sparse attention memory transformer
- entity linking retrieval disambiguation
- approximate lexical/string matching
- reciprocal rank fusion neural retrieval

Also inspect bibliographies/citations already used in Papers 0, 2, 2.5, 3, and 3.5 to maintain consistent terminology and avoid duplicate/mismatched citations.

Do not use blogs when primary literature exists.

## 15. Related-work comparison table
If space allows:

| Family/system | External text | Native K/V | Lexical | Semantic | Iterative/state-dependent | Adaptive channel |
|---|---|---|---|---|---|---|

Use representative systems only. Put PRA 2.6 last.

Do not mark absence of a feature unless verified.

## 16. Citation/novelty discipline
Every related-work claim must be traceable.

Avoid:
> No prior work does X.

Prefer:
> In the systems reviewed here, X is typically fixed/external/one-shot, whereas Paper 2.6 evaluates state-dependent channel changes inside PRA memory discovery.

Keep novelty narrow.

## 17. Introduction update
Motivate in this order:
1. PRA must find a tiny active working set in large logical memory.
2. Native semantic similarity is only one discovery representation.
3. Exact/BM25/approx channels recover evidence semantic routing misses.
4. Static fusion does not dominate.
5. Root and successor states often prefer different representations.
6. Therefore retrieval representation should be exposed as a state-dependent action.

## 18. Abstract update
Keep compact. Mention:
- four datasets;
- root/successor mismatch on 3/4;
- static fusion/RRF negative result;
- adaptive headroom but selector limitation;
- remaining disambiguation weakness.

Do not list every recall number.

## 19. Discussion update
End with:

### Established
Multi-representational, state-dependent discovery.

### Unresolved
Observable channel selection and referential disambiguation.

### Handed off
Adaptive method selection -> Paper 3.5.
Physical K/V admission -> Paper 3.

## 20. Required artifacts
Create/update:
- expanded row-level route results
- `cohort_stability.csv`
- `channel_confidence_rows.csv`
- `channel_confidence_metrics.csv`
- `reference_disambiguation.csv`
- `useful_address_proxy.csv`
- `search_method_action_spec.json`
- `related_work_comparison.csv`
- bibliography additions
- updated findings JSON
- claim audit
- readability audit

## 21. Required plots
1. Best-channel stability vs cohort size
2. Root confidence calibration
3. Successor confidence calibration
4. Wrong-reference abstention/precision-recall curve
5. Semantic-consistency gate effect
6. UsefulAddress proxy calibration
7. Root→successor transition heatmap on expanded cohort
8. Adaptive headroom with updated CIs

## 22. Tests
Add:
- deterministic expanded-cohort sampling
- confidence-feature extraction
- no gold leakage
- calibration metrics
- referential-validity fixtures
- action-spec schema validation
- bibliography/reference build checks
- related-work table consistency if generated

Keep full suite green.

## 23. Stop rules
Do NOT:
- add more datasets after this;
- invent another large fusion family;
- build the full adaptive controller;
- add generation just to expand scope;
- claim referential disambiguation solved;
- overstate novelty relative to hybrid IR;
- duplicate Paper 3/3.5 experiments.

## Core Principle
This iteration should make Paper 2.6 a mature study of **state-dependent retrieval representation**. Larger cohorts should test stability; confidence/disambiguation diagnostics should clarify when lexical evidence can be trusted; a stable machine-readable action specification should hand root/successor method choice to Paper 3.5; and related work should position the contribution precisely within hybrid retrieval, multi-hop retrieval, retrieval-augmented attention, and entity/reference-resolution research.
