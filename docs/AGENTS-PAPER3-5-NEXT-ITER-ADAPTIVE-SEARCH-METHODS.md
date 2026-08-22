# AGENTS — Paper 3.5 Next Iteration: Adaptive Search-Method Selection

## Mission
Extend `adaptive-pra` so the controller chooses not only effort parameters but also the retrieval/search representation itself.

Paper 2.6 now establishes heterogeneous channel preference:
- QASPER -> exact
- HotpotQA -> BM25
- 2Wiki -> BM25
- MuSiQue -> approximate-token
- static hybrid never beats the best single channel
- per-example oracle channel choice has headroom
- address exposure alone is not enough for reliable iterative reference following

The controller must distinguish root-search method from successor-search method.

## Extend action vector
From approximately:
`(Q_regions,F,R,K,H,B,theta,L,G,M)`

to:
`(Q_regions,S_root,S_succ,F,R,K,H,B_search,B_KV,theta,L,G,M)`

Root method:
`S_root in {semantic, exact, BM25, approx, hybrid}`

Successor method:
`S_succ in {native_semantic, exact_new_address, BM25_state, approx_new_address, hybrid_state}`

Do not force them to match.

## Interpret / Search / Admit
Use:

### Interpret
`(Q_regions, F, S_root)`
- identify query
- facet it
- choose root-retrieval representation

### Search
`(R,K,H,S_succ)`
- choose breadth/depth
- choose successor-retrieval representation

### Admit
`(B_search,B_KV,theta,L,G,M)`
- decide how much discovered memory becomes active

This is the central metacontrol decomposition.

## Import Paper 2.6 action specification
Consume `search_method_action_spec.json` when available. Reuse method names, confidence signals, cost metrics, and allowed states. Do not duplicate incompatible definitions.

## Baselines
Compare:
- S0 one fixed method for root+successor
- S1 dataset-fixed best method (diagnostic only)
- S2 static hybrid
- S3 fixed root/successor pair
- S4 adaptive root only
- S5 adaptive successor only
- S6 adaptive root+successor
- S7 oracle root+successor

This decomposes where adaptivity matters.

## Search-method oracle
Before learning selectors derive:
- `S_root*` per example
- `S_succ*` per hop where possible
- matched-budget quality/cost

Compute root, successor, and combined oracle headroom.

## Factorized oracle with method choice
Extend factorized control from `(F,R,K,H,B)` to staged optimization over:
`(S_root,S_succ,F,R,K,H,B_search,B_KV)`.

Use:
1. search-method oracle under current good profiles
2. factorized parameters conditional on method
3. joint Pareto frontier

Do not brute-force an uncontrolled Cartesian explosion.

## Recompute profile quantization regret
Measure separately:
- savings from factorized params
- savings from adaptive method selection
- combined savings

Do not hide which source creates headroom.

## Controller target variants

### T0 global profiles
Profiles may include root/successor method, but derive them from Pareto data.

### T1 group profiles — preferred next learned variant

Interpret head predicts:
- query-region policy
- facet policy
- root search method

Search head predicts:
- R/K/H profile
- successor search method

Admit head predicts:
- search/KV budget
- threshold/layer/granularity/materialization profile

### T2 independent heads
Separate categorical heads for methods and parameter values.

### T3 autoregressive/interaction-aware
Candidate order:
`Q_region -> S_root -> F -> R -> S_succ -> K -> H -> B_search -> B_KV`

Only after oracle interaction analysis.

## Query representation
Use current Paper-3.5 findings:
- R0 observable router remains supported default
- `S2_embed_last` is the strongest cheap self-query representation selected in the current study
- deeper hidden/native Q/K representations did not reliably improve routing
- contextual encoding barely improved over embedding-only
- external encoder remains deferred

For root-method selection compare:
- R0 observable features
- embedding-only self query
- combined observable + self embedding

Do not reopen the full layer sweep.

## Root-method features
Deployment-safe:
- query length
- IDF/rare-token statistics
- entities
- numeric/URL/ID markers
- exact confidence
- BM25 gap
- approximate confidence
- gist gap
- channel disagreement
- query-region/layout
- self-query embedding
- facet disagreement

No gold features.

## Successor-method features
At hop t additionally use:
- newly exposed rare tokens/entities
- address rarity/uniqueness
- candidate count sharing address
- semantic consistency with query/current node
- semantic successor score gap
- lexical-vs-semantic disagreement
- previous method/hop state

This is likely where adaptive method switching matters most.

## Useful-address gate
Paper 2.6 shows exposure is insufficient.

Model:
`P(useful_address | observables)`

using:
- rarity
- uniqueness
- entity/type consistency
- semantic consistency
- candidate count
- approximate confidence

Only choose exact/approx successor following when confidence is adequate.

## Non-learned cascades first
Test interpretable baselines:

- semantic root -> if weak gap, BM25
- BM25 root -> native semantic successor
- semantic root -> unique rare address -> exact successor
- lexical root -> native successor -> weak successor -> approximate retry
- channel disagreement -> search two channels but keep B_KV tight

These may be stronger than static hybrid.

## Search more, admit less
Maintain the Paper-3 principle:

`B_search ↑` does not imply `B_KV ↑`.

A controller may search multiple channels but materialize only a tight high-precision core.

Track separately:
- candidates searched
- conceptual nodes retained
- K/V materialized

## Precision–recall–cost objective
Report:
- evidence recall
- evidence precision
- path recovery
- search comparisons/latency
- active K/V
- output quality where live generation exists

Primary control objective:
`min cost subject to sufficient downstream quality`.

Do not optimize channel classification accuracy in isolation.

## Retry actions
Expand targeted retry actions:

- reinterpret query
- change root method
- change successor method
- increase F
- increase R
- increase K
- increase H
- increase B_search
- narrow B_KV
- preserve strong root
- stop

A retry may change representation without increasing effort.

## Evaluator-side retry decomposition
Current evaluator-side bounded retry raises quality strongly. Decompose potential corrections into:
- query reinterpretation
- root-method switch
- successor-method switch
- parameter change
- admission narrowing
- combined action

This determines what a learned retry controller actually needs to predict.

## Learned targeted retry
Only after upper-bound decomposition. Predict the cheapest corrective `delta_action`, not a full new profile.

Compare against:
- always widen
- always switch channel
- oracle correction

## Search-method confidence calibration
Calibrate:
- semantic gap
- exact confidence
- BM25 gap
- approximate confidence
- channel disagreement
- useful-address confidence

Use for selection, retry, and abstention.

## Dataset behavior
Report QASPER, HotpotQA, 2Wiki, MuSiQue separately.

Do not feed dataset identity as a deployment feature.

## Cross-dataset generalization
If sample size permits, train on three datasets and evaluate on the fourth.

This tests whether the selector learns query/channel geometry rather than dataset labels.

## Query-region × method interaction
Measure whether correct query localization changes the best root method.

## Facet × method interaction
Measure whether:
- exact prefers local facets
- BM25 prefers broader text
- semantic gist benefits from multiscale facets

Only add interaction-aware architecture when oracle data supports these interactions.

## Method-specific systems cost
Account for:
- semantic GEMM/index
- lexical/BM25 index
- exact/approx token index
- lookup latency
- index memory
- CPU/GPU placement
- batching/cache compatibility

Do not reduce unlike operations to one misleading count.

## Required primary experiments
A. Root-method oracle headroom
B. Successor-method oracle headroom
C. Root→successor transition matrix
D. Factorized+method oracle
E. Group-profile controller with method selection
F. Independent/interaction method heads after E
G. Leave-one-dataset-out generalization
H. Targeted retry with method switching
I. Query-region × method interaction

## Required artifacts
- `search_method_action_space.json`
- `root_method_oracle.csv`
- `successor_method_oracle.csv`
- `root_successor_transition.csv`
- `factorized_method_oracle.csv`
- `method_profile_quantization_regret.csv`
- `method_selector_features.csv`
- `method_selector_results.csv`
- `method_cross_dataset_results.csv`
- `useful_address_features.csv`
- `method_retry_upper_bound.csv`
- `method_targeted_retry.csv`
- `search_vs_admission_results.csv`
- `method_cost_accounting.csv`
- updated `paper3_5_findings.json`

## Required plots
1. Root-method oracle distribution
2. Successor-method oracle distribution
3. Root→successor method heatmap
4. Fixed vs adaptive quality-cost frontier
5. Channel disagreement vs selected method
6. Useful-address confidence vs lexical-successor gain
7. Search breadth vs K/V admission breadth
8. Cross-dataset generalization
9. Retry correction by action type
10. Factorized+method oracle regret

## Tests
Add:
- root/successor method independence
- method-specific config validation
- factorized+method oracle correctness
- no dataset-ID leakage
- no gold-feature leakage
- transition tracing
- useful-address gating
- targeted method retry
- search vs K/V budget separation
- method-specific cost accounting
- deterministic Paper-2.6 action-spec import

## Paper narrative update
After results, evolve Paper 3.5 from “adaptive effort selection” to:

`QUERY/SESSION -> INTERPRET -> SEARCH -> ADMIT -> EVALUATE -> targeted retry`

where interpretation includes query region, facets, and root-search representation; search includes successor representation and breadth/depth; admission controls conceptual and physical working sets.

## Stop rules
Do NOT:
- assume one channel for root and successor
- use dataset identity as a selector feature
- build a large generative controller
- optimize method classification rather than downstream quality/cost
- merge search breadth with K/V materialization
- endlessly tune static hybrid
- duplicate Paper 2.6 mechanism experiments

## Core principle
Retrieval representation itself is an adaptive action. Root discovery may need exact identity, BM25 relevance, approximate lexical matching, or semantic similarity; successor traversal may need a different representation after new evidence changes the state. Adaptive PRA should therefore control **interpretation, search method, search effort, and admission**, and retry should be able to change the representation rather than merely spend more compute.
