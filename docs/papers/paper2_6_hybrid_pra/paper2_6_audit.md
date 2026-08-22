# Paper 2.6 implementation audit

## Scope and vocabulary

Paper 0 defines four distinct quantities: logical memory is all addressable
source content; requested memory is the subset selected by discovery; resident
memory is encoded/cache state available to the runtime; materialized memory is
the native K/V admitted to active attention. Paper 2.6 changes discovery only.
It must not count a candidate as materialized or alter Paper 3's physical
interval-materialization contract.

Paper 1.5 establishes that selected native K/V retains source-relative position
semantics. Token-native matching can select identities and spans, but it does
not rotate, concatenate, or otherwise rewrite K/V. Paper 2 supplies the frozen
Qwen3-0.6B feature protocol and shows that contextual gists are useful but leave
relation-near misses. Paper 2.5 supplies the bounded traversal engine. Paper 3
owns the subsequent mapping from selected conceptual identities to physical
native-K/V intervals.

## Current routing interfaces

- `src/pra_hf/iterative.py::GistIndex` packs layer-local semantic gists in
  stable `(URI, chunk_id)` order without copying native K/V.
- `IterativeRoutingConfig` owns depth, per-frontier branching, beam width,
  confidence threshold, path reduction, and the hard `max_unique_chunks`
  discovery budget.
- `IterativeGistRouter.route()` owns the Paper-2.5 loop: score, propose,
  deduplicate, admit, update the frontier, stop, and emit retrieval-graph
  provenance. This loop must remain the only iterative engine.
- `RetrievalGraph` schema 2.0 is the versioned handoff from discovery to later
  materialization. Its nodes are identities and diagnostics, not active K/V.
- `PRAForCausalLM._route_once()` maps final selected identities to consumption
  layers only after traversal stops. That is the correct integration boundary
  for hybrid discovery.
- `PRAConfig.routing_mode` currently supports `one_shot`, `iterative`, and
  `local_iterative`; disabled/default behavior must remain byte-for-byte
  compatible.

## Implemented lexical support

`experiments/paper2_hf/routing/precompute_router_features.py` stores a one-hop
question/chunk token-ID set-Jaccard score. It is an experimental diagnostic,
not a reusable index. It discards order, token frequency, exact spans,
normalization state, aliases, provenance, and hop context.

Paper 2.6 now provides:

1. a tokenizer-native index aligned exactly with `GistIndex.records`;
2. raw and normalized token sequences, IDF, n-grams, aliases, and BM25 words;
3. a candidate record retaining every channel score and provenance;
4. shared-budget score policies for token-only, BM25, union, reranking,
   cascade, and iterative hybrid conditions;
5. token queries exposed by admitted evidence at later hops;
6. validation-only confidence calibration and held-out reliability metrics;
7. controlled perturbation and confidently-wrong-reference tests.

## Reuse boundary

The token-native index is a sidecar to `GistIndex`, checked for exact
identity alignment. `IterativeGistRouter.route()` accepts an optional
sidecar and policy. Semantic and token-native channels will propose within the
existing Paper-2.5 loop and compete for the same branch, beam, and unique-chunk
budgets. The default call path will not construct or consult the sidecar.

`PRAForCausalLM` constructs the sidecar only for explicit token-native or
hybrid routing modes. The root token query comes from the already-tokenized
prompt; each admitted chunk contributes its indexed token sequence to the next
frontier. Final identities continue through the existing layer mapping and
materialization path unchanged.

## Experiment assets

The frozen Paper-2 Qwen feature files provide aligned semantic query/chunk
states for validation and held-out HotpotQA/QASPER examples. Source text is
reconstructed from the identity-stable local dataset loaders. Controlled chains
exercise exact, normalized, weighted, ordered, approximate, semantic-only, and
mixed-hop cases. The primary endpoint is evidence discovery at a matched final
candidate budget. Any generated-answer experiment is reported separately and
does not retroactively redefine discovery success.

The channel-geometry extension also reuses Paper-2.5's frozen 2Wiki and MuSiQue
token states and labelled evidence mapping. It adapts them to the same one-mean-
gist-per-32-token-chunk contract without loading model weights. The tracked
runner emits the six-channel matrix, per-example geometry and advantage rows,
precision/recall intervals, channel overlap, rare new-address diagnostics,
validation-fixed and oracle selectors, robustness controls, and all paper
plots under `shared/results/paper2_6_hybrid_pra/channel_geometry`.

`src/pra_hf/channel_geometry.py` keeps the selector boundary explicit. Gold
geometry may explain observed channel advantage, but the diagnostic selector
rejects gold-derived fields. Every primary route is asserted to return exactly
four unique chunks. Discovery comparisons are logged, materialized K/V remains
zero, and the claim audit records that no answer generation is performed.

The next iteration separates the four-chunk budget into two root requests and
two successor requests. All 25 root/successor channel pairs are retained for
audit, while successor headlines hold the validation-selected root fixed.
`headroom_decomposition()` separates true per-example selector opportunity from
validation-selection instability. A validation-only linear selector uses 24
query/score/rank observables; no MLP is fitted. Reciprocal-rank fusion is the
only added fusion control.

Natural successor semantics remain explicit: MuSiQue/2Wiki use annotated graph
dependencies, while QASPER/Hotpot use a separately labelled unordered evidence
remainder. `UsefulAddress` requires exposure, a gold link, and rank within four.
The extended robustness matrix distinguishes target and wrong-target recovery
for near entities, shared prefixes, numeric IDs, URL paths, and aliases. The
Paper-3.5 handoff is serialized in `search_method_action_spec.json`; it exposes
search actions and costs but no gold geometry or materialization decision.
