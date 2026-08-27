# Architecture

## Prompt and references

A prompt can contain lightweight handles such as `<REF_1>`. A row-local
`ReferenceTable` maps each handle to a URI and optional summary or nested reference
metadata. Handles are symbolic addresses; their referenced text is not automatically
inserted into the prompt.

## Cache construction

`build_cache_from_metadata()` converts collator metadata into resolver documents and asks
`RecursiveReferenceCacheBuilder` to construct memory child-first. Depth, reference-count,
token-count, cycle, and missing-reference policies bound recursive expansion.

Each retained document is partitioned into chunks. `TinyPRAModel` independently encodes
each chunk through the same block stack used by the prompt. Before a PRA sublayer runs,
the model captures that layer's projected token K/V:

```text
K, V: [1, heads, reference_tokens, head_width]
```

One or more paired routing gists are built from projected keys and values for each chunk and
layer. Optional URI-level gist sets compress all chunk gists at that layer and are cached for
true reference-first routing. The default remains one mean chunk gist, preserving existing
configs while keeping cheap routing state separate from detailed attention memory.

## Hierarchical routing

At each PRA layer, the final prompt token supplies one projected routing query per batch
row. The configured search strategy selects references and chunks under independent
budgets:

```text
query [B, model_width]
  -> optionally score cached URI gist sets [G_ref,D]
  -> select up to top_k_references URIs
  -> score chunk gist sets [G_chunk,D] in selected URIs
  -> select up to top_k_chunks_per_reference chunks per URI
```

The `hierarchical`, `reference_first`, and `global_chunks` strategies provide controlled
alternatives for experiments.

## Long prompt history

`prepare_prompt_batch_for_pra()` splits oversized prompts on exact token IDs. The recent
tail, bounded by `max_prompt_direct_tokens` and `max_seq_len`, remains in ordinary causal
self-attention. The displaced prefix becomes the request-local URI
`pra://implicit/prompt/head` (`#__head`) in that row's existing cache. It uses the same
chunk and gist strategies as explicit references, while `max_prompt_gists` independently
controls its chunk cap. A null cap keeps all prompt-head chunks.

Mixed-length direct tails are padded with an attention mask, and routing uses each row's
last valid token. Long initial generation prompts are supported. Streaming migration of
generated history into the implicit reference is not implemented yet.

## Batch isolation

Every prompt row owns an independent reference namespace. `PRABatchedMemoryCache` wraps
the completed row caches and guarantees that `query[i]` searches only `row_caches[i]`.
Duplicate URI strings across rows are therefore safe.

The prompt model runs once for `input_ids [B,T]`. Selected memory may have a different
length `M_i` in each row. `dynamic_memory_attention()` buckets similar lengths, masks
padding, and restores the original row order.

## Attention fusion

`PRAttention` computes normal causal self-attention and optional external-memory attention.
After materialization, the two branches are combined as:

\[
y = y_{\mathrm{local}} + \alpha y_{\mathrm{memory}}.
\]

`detail_materialization` controls whether selected chunks, full selected references, or
winning gist-only positions enter the memory branch. Passing `use_pra_memory=False` bypasses the
branch for controlled disabled-reference evaluation.

## Training boundary

The `common` package owns model-independent optimization, checkpointing, timing, logging,
and metric history. `pra_torch.pra_train` injects PRA-aware batch and evaluation callbacks.
`PRAStandaloneTrainer` is a thin object-oriented facade over those functional APIs.

## Typed capabilities

Tools and skills enter the runtime as immutable typed records. Their selection
views contain only identity, description, and use conditions. A bounded palette
is encoded first; selecting one stable ID activates its complete schema or
instructions without semantic rediscovery. Tool visibility remains separate
from host execution authorization.

## Compact result backing

Tool responses, database rows, logs, terminal output, graphs, files, RAG
results, and generic API payloads can be stored as session-scoped exact backing.
Type-aware compactors produce bounded prompt views while lexical, entity,
rare-term, schema, and optional summary addresses remain retrieval-only.
Materialization can return the full record, selected fields or ranges, search
matches, or cursor pages.

Native result routing is explicit. `register_result_backing()` encodes exact
backing only after the host opts in; `route_result_backing()` uses the production
PRA query/router path without generation; and teardown removes every reference
registered for that session. This path requires an isolated model session so
model-resident K/V cannot cross tenant boundaries.
