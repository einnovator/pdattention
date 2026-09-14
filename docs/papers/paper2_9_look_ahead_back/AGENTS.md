# AGENTS.md — Paper 2.9: Temporal Semantic Discovery for PRA

## Mission
Test whether PRA semantic discovery should route from temporally extended query evidence rather than one current token/state. Keep the frozen causal backbone and downstream native-K/V materialization unchanged. Paper 2.6 lexical/indexed retrieval remains the default for explicitly structured sources such as tools and APIs.

## Inherited context
- **2.5:** iterative traversal; root/propagation/stopping separation; query-entry and multiscale-query diagnostics.
- **2.6:** semantic vs exact/BM25/approximate/hybrid retrieval; tokenizer-native sparse index; stable identities; matched discovery/materialization accounting.
- **2.7:** graph facets can recover plausible partitions yet hurt retrieval. Facet discovery is not enough; consolidation/weighting matters.
- **2.8:** one mean gist can lose sparse QK signals; compact native-key landmarks have limited/query-dependent headroom. Do not confound query-window gains with memory expansion.

## Hypotheses
1. B=2–8 observed-token windows improve semantic routing over B=1 on ambiguity-sensitive examples.
2. Delayed commitment recovers much of oracle look-ahead causally.
3. Prompt prefill benefits from a tiny non-causal routing sidecar without changing causal backbone states.
4. Late token×gist interaction can beat mean pooling when several tokens jointly disambiguate a chunk.
5. Temporal query extent interacts with number/type of chunk gists.
6. Routing can run on a slower token clock and sparse layer checkpoints.
7. Predictive look-ahead is justified only if oracle future context leaves material headroom after delay.

## Non-goals
- Do not replace Paper 2.6 tool/API lexical indexes with neural routing.
- Do not alter native K/V, RoPE, or Paper 3 materialization semantics.
- Do not fine-tune the backbone.
- Do not tune on test identities.
- Do not run generation before retrieval gates pass.

## P0 — Reproduce and freeze
- Reproduce exact 2.6/2.7/2.8 rows used as baselines.
- Record branch/commit, model revision, layer, chunk size, budgets, identity hashes.
- Add identity-disjoint temporal fixtures: shared prefixes, compositional phrases, relation order, distractors, progressive candidate collapse.
- Assert B=1 reproduces current semantic routing.

## P1 — Causal look-behind
Implement B={1,2,4,8,16} over token embeddings and selected hidden layers with exact provenance/masks.
Aggregators: last/current; mean; validation-fixed weighted/attention pool; late interaction `sum_i w_i max_j sim(q_i,g_cj)`.
Run first with inherited one-mean-gist chunks.

**G1:** look-behind improves one natural dataset with paired 95% CI > 0 and regresses <0.02 on the other primary dataset.

## P2 — Delayed commitment
Implement immediate, fixed D={1,2,4,8}, entropy/margin threshold, and optional phrase-boundary policies.
Log candidate entropy, top-1 margin, candidate count, selected-set churn, delay, and router calls/token.

## P3 — Oracle and prefill look-ahead
First measure oracle F={1,2,4,8} using already-known future tokens. This is analysis, not deployable generation routing.
Then implement a 1–2-layer local bidirectional prefill sidecar over embeddings/frozen states. It must not modify backbone states; disabled-PRA parity must remain exact.

**G2:** if delay recovers >=70% of oracle gain with median delay <=4, prefer delay for generation.

## P4 — Query × memory compression
Cross best temporal query with m={1,2,4,8} representatives:
- inherited mean gist;
- local/prototype gists;
- existing k-means/farthest-first controls where available;
- compatible 2.8 native-key landmarks.
Keep final selected-chunk budget fixed and report routing dots/bytes separately.

## P5 — Facets
Reuse 2.7 syntax/windows/graph facets only as controls. Add consolidation:
- weighted facet centroid;
- top-f facets by informativeness;
- global-query + facets mixture;
- late interaction over facet set.
Do not claim partition ARI predicts retrieval.

## P6 — Slower routing clock
Test update stride s={1,2,4,8} and sparse layer checkpoints. Reuse selection until confidence/identity/drift trigger.
Measure quality vs router calls, candidates scored, latency, and churn.

## P7 — Predictive generation look-ahead (conditional)
Run only if oracle future context has substantial unrecovered headroom after P2.
Order:
1. expected next-token embedding;
2. small future-routing probe;
3. direct coarse destination/subtree predictor.
Compare against **equal-latency delayed commitment**, not just immediate routing.

## Datasets and baselines
Natural: inherited HotpotQA, QASPER, 2Wiki, MuSiQue where feature contracts match.
Baselines: B=1 semantic; whole-query semantic; 2.6 exact/BM25/approx/best hybrid; fixed windows; 2.7 syntax/graph; one-mean-gist; oracle future.
Tool-like fixtures are mechanism checks only; production tool discovery remains lexical/index-first.

## Metrics
Evidence recall/precision, complete evidence, MRR, top-k stability, entropy, candidate count, delay, churn, router calls/token, candidates scored/token, routing latency, physical K/V tokens, and active-memory fraction. Bootstrap by identity.

## Gates
- **G0:** inherited baseline parity.
- **G1:** causal look-behind material retrieval gain.
- **G2:** delayed-commitment recovery of oracle future gain.
- **G3:** prefill-sidecar gain + exact disabled-PRA backbone parity.
- **G4:** late interaction/multi-gist gain justifies cost.
- **G5:** predictive probe beats equal-latency delay.
- **G6:** only then run downstream K/V consumption/generation.

## Required ablations
- token IDs/embeddings vs hidden states;
- early/mid/late layers;
- B and F independently;
- mean vs weighted vs late interaction;
- one vs multiple chunk gists;
- immediate vs delayed;
- per-token vs strided routing;
- per-layer vs checkpoint routing;
- semantic-only vs 2.6 lexical/indexed reference where appropriate.

## Claim discipline
Do not claim look-ahead if only prefill future tokens are used; call it non-causal prefill routing.
Do not call predicted futures useful unless they beat equal-latency delay.
Do not count candidates as materialized memory.
Do not merge retrieval and consumption claims.
Negative gates are publishable outcomes.
