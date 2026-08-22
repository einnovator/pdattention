# AGENTS — Paper 2.6 Next Iteration: Channel Selection, Geometry, and Robustness

## Mission
Continue `hybrid-pra` from the four-dataset result. Do not add more datasets yet. The goal is to explain why QASPER favors exact, HotpotQA/2Wiki favor BM25, MuSiQue favors approximate-token routing, why static hybrid never wins, and how root-search and successor-search channel choice should be exposed to Paper 3.5.

## Core questions
1. Which observable query/evidence properties predict the best retrieval channel?
2. Does the best root-search channel differ from the best successor-search channel?
3. How much true per-example channel headroom exists after separating it from validation-selection noise?
4. Why does MuSiQue benefit from approximate-token matching?
5. Why do rare hop-one addresses help only some natural examples?
6. Why does static hybrid lose: score calibration, budget dilution, or complementary distractors?

## Separate root and successor search
Evaluate channel performance separately for:
- root: `Q -> C1`
- successor: `(Q,Ct) -> C(t+1)`

Root methods:
- semantic/gist
- exact
- BM25
- approximate
- static hybrid

Successor methods:
- native semantic
- exact newly exposed reference
- BM25 over updated state/evidence
- approximate newly exposed reference
- hybrid/cascade

Required table:
`dataset | best root channel | best successor channel | same/different | recall | precision | cost`

Do not assume root and successor should use the same representation.

## Root and successor oracle headroom
Compute per-example oracle root channel and, where possible, per-hop successor channel.

Report:
- best held-out fixed channel
- validation-selected fixed channel
- per-example oracle
- observable selector

Separate:
`H_selection = oracle - best-heldout-fixed`
from:
`H_validation = best-heldout-fixed - validation-selected`

Do not conflate adaptive headroom with small-sample validation instability.

## Channel-transition matrix
Estimate transitions such as:
- semantic -> exact
- semantic -> BM25
- BM25 -> semantic
- BM25 -> approximate
- exact -> semantic

Required:
`root channel | successor channel | frequency | path gain`

This directly tests dynamic multi-representational search versus static fusion.

## Dataset-specific win analyses

### MuSiQue approximate wins
Inspect examples where approximate beats exact/BM25/gist. Classify causes:
- morphology
- alias variation
- punctuation/case
- BPE/tokenization variation
- partial entity mention
- spelling variation
- lexical near-match
- other

Do not assume the cause.

### QASPER exact wins
Inspect:
- technical phrase reuse
- section/title references
- rare terminology
- explicit evidence wording
- identity-style references

### Hotpot/2Wiki BM25 wins
Inspect whether BM25 benefits from:
- multiple moderately informative terms
- entity+relation co-occurrence
- bridge-query lexical structure
- exact routing overfocusing on one span

## Channel disagreement
Formalize:
- selected-set Jaccard
- rank overlap
- score disagreement
- root disagreement

Test whether high disagreement predicts higher oracle-channel headroom.

## Query-only observable features
Deployment-safe features:
- query length
- rare-token count / IDF stats
- named-entity count
- numeric/URL/ID markers
- exact confidence
- BM25 gap
- approximate-match confidence
- gist score gap
- channel disagreement
- query-region/layout
- facet disagreement

Gold evidence geometry is explanatory only; never use it as a deployment selector feature.

## Selector baselines
Keep modest:
- H0 best fixed
- H1 simple rules
- H2 linear/logistic
- H3 small MLP only if sample size supports it

Do not build a large controller here. Paper 3.5 owns that.

## Static-hybrid postmortem
Measure:
- unique gold evidence added by each channel
- unique distractors added by each channel
- duplicate budget consumption
- precision after fusion
- score/rank calibration mismatch

If cheap, add one robust rank-fusion control such as reciprocal-rank fusion. Do not turn this into a fusion survey.

## Iterative address analysis
Replace simple “new address exposed” with:

`UsefulAddress = Exposed AND GoldLinked AND CompetitiveRank`

Measure:
- validity
- uniqueness
- rarity
- candidate count
- correct successor rank
- semantic consistency with current state

Test whether UsefulAddress predicts iterative gain better than exposure alone.

## Wrong-reference robustness
Keep clean/case/punctuation/typo/confident-wrong controls. If cheap add:
- near-entity collision
- shared prefix
- numeric-ID collision
- URL/domain overlap
- alias/synonym

Measure target recovery, wrong-target recovery, confidence, and abstention/retry opportunity.

## Precision–recall
Report root and successor precision/recall separately. A channel may be excellent for root discovery and poor for iterative expansion.

Do not optimize recall alone; false positives later consume conceptual budget and potentially K/V/attention budget.

## Search-method cost accounting
Per channel record:
- comparisons
- index lookups
- token/span operations
- latency
- index memory
- CPU/GPU placement where relevant

No production-serving claim.

## Paper 3.5 handoff
Export `search_method_action_spec.json` with:

`root_search_methods = {gist, exact, bm25, approximate, hybrid}`

`successor_search_methods = {native_semantic, exact_new_address, bm25_state, approximate_new_address, hybrid_state}`

For each method include allowed params, confidence signals, cost metrics, and failure indicators.

## Required artifacts
- `root_channel_results.csv`
- `successor_channel_results.csv`
- `channel_transition_matrix.csv`
- `channel_true_oracle_headroom.csv`
- `validation_instability.csv`
- `musique_approx_win_analysis.csv`
- `qasper_exact_win_analysis.csv`
- `bm25_win_analysis.csv`
- `channel_disagreement_features.csv`
- `selector_observable_features.csv`
- `selector_results.csv`
- `static_hybrid_postmortem.csv`
- `iterative_useful_address.csv`
- `address_confidence.csv`
- `search_method_action_spec.json`
- updated `paper2_6_findings.json`
- claim audit

## Required plots
1. Root recall by channel × dataset
2. Successor recall by channel × dataset
3. Root→successor transition heatmap
4. Oracle headroom vs validation-instability decomposition
5. Channel disagreement vs oracle headroom
6. MuSiQue approximate-win taxonomy
7. Useful-address rate vs iterative gain
8. Root vs successor precision-recall
9. Static hybrid unique-evidence vs unique-distractor contribution
10. Selector vs oracle frontier

## Stop rules
Do NOT:
- add more datasets
- tune many hybrid weights
- build a large adaptive router
- claim generation/output gains
- conflate root and successor performance
- conflate oracle headroom with validation instability

## Core principle
Paper 2.6 should finish by showing that retrieval representation is state- and task-dependent. The best similarity space for root discovery need not be the best one after new evidence changes the search state. Quantify that geometry and export the action space; Paper 3.5 should learn how to control it.
