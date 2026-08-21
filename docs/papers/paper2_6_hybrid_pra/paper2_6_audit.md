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

## Existing lexical support and missing mechanisms

`experiments/paper2_hf/routing/precompute_router_features.py` stores a one-hop
question/chunk token-ID set-Jaccard score. It is an experimental diagnostic,
not a reusable index. It discards order, token frequency, exact spans,
normalization state, aliases, provenance, and hop context.

Paper 2.6 therefore needs:

1. a tokenizer-native index aligned exactly with `GistIndex.records`;
2. raw and normalized token sequences, IDF, n-grams, aliases, and BM25 words;
3. a candidate record retaining every channel score and provenance;
4. shared-budget score policies for token-only, BM25, union, reranking,
   cascade, and iterative hybrid conditions;
5. token queries exposed by admitted evidence at later hops;
6. validation-only confidence calibration and held-out reliability metrics;
7. controlled perturbation and confidently-wrong-reference tests.

## Reuse plan

The token-native index will be a sidecar to `GistIndex`, checked for exact
identity alignment. `IterativeGistRouter.route()` will accept an optional
sidecar and policy. Semantic and token-native channels will propose within the
existing Paper-2.5 loop and compete for the same branch, beam, and unique-chunk
budgets. The default call path will not construct or consult the sidecar.

`PRAForCausalLM` will construct the sidecar only for explicit token-native or
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
