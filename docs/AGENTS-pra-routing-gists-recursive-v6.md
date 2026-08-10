# AGENTS — Fix PRA Reference Routing

> **Metrics integration notice**
>
> The final section titled **“Authoritative correction: integrate retrieval metrics into the existing training stack”** overrides earlier wording that suggests creating separate logging or metric-output infrastructure. Extend `train.py`, `pra_train.py`, `trainer.py`, `metrics.py`, and `logging.py` through their existing contracts.


> **Authoritative architecture notice**
>
> The section titled **“Architectural correction: references, chunks, routing gists, and token K/V”** is authoritative and overrides any earlier wording in this file about arbitrary gist regions, multiple gists per chunk, or `top_k_regions`. Implement the corrected hierarchy exactly.


## Goal

Correct the current PRAttention reference-routing implementation so that:

1. reference retrieval is correct for every item in a batch;
2. datasets and cache entries no longer require textual summaries;
3. reference routing vectors are derived from the actual reference content;
4. each PRA layer uses its own cached key representation;
5. the initial routing representation is the arithmetic mean of that layer's cached keys;
6. head-specific routing is **not** introduced in this change;
7. each reference may produce a variable number of routing gists;
8. gist count is bounded by a configurable small maximum;
9. fixed-size and marker-based chunking are implemented;
10. semantic/dynamic chunking is represented by an interface and configuration mode, but the semantic algorithm itself is not implemented here;
11. nested references are resolved recursively with explicit depth, cycle, and budget controls.

This remains correctness-first, but it also establishes the extensible gist/chunk/recursive-reference abstractions needed by later experiments.

---

## Current problems

### 1. Batch routing bug

The current cache search accepts a query shaped `[batch, d_model]`, but reduces it to the first item:

```python
if query.dim() == 2:
    query = query[0]
```

As a consequence, all examples in a batch use the references selected for batch item `0`.

The memory tensors are then expanded across the batch, which silently gives every example the first example's retrieved references.

This must be fixed.

### 2. Incorrect dependence on textual summaries

The current cache entry stores:

```python
summary: str
summary_vector: torch.Tensor
```

and reference construction requires:

```python
encode_reference_to_cache(
    uri,
    text,
    summary,
    tokenizer,
    device,
)
```

The summary vector is built by averaging raw token embeddings of the supplied textual summary.

This is not the intended PRA mechanism.

A reference must be routable from its own encoded content. No manually supplied, generated, or dataset-provided textual summary should be required.

### 3. Routing uses raw embeddings instead of layer-specific keys

The current router compares a contextual layer input against a mean of raw token embeddings.

Instead, each PRA layer must route using the same key space already used for memory attention.

For reference `r` and PRA layer `l`, the cache already stores:

```text
K[r, l]: [1, n_heads, ref_len, head_dim]
```

For this first corrected implementation, derive a single layer-specific routing vector by:

1. merging the head and head-dimension axes back into `d_model`;
2. averaging over the reference-token axis.

The result is:

```text
routing_key[r, l]: [d_model]
```

Mathematically:

```text
K_layer: [1, H, T_ref, D_h]

K_tokens = transpose(K_layer, head/token axes)
K_tokens: [1, T_ref, H, D_h]

K_tokens = reshape(K_tokens)
K_tokens: [1, T_ref, D_model]

routing_key = mean(K_tokens, dim=token)
routing_key: [D_model]
```

Equivalently:

\[
ar{k}_{r,l}
=
rac{1}{T_r}
\sum_{t=1}^{T_r}
\operatorname{concat}_{h=1}^{H} K_{r,l,h,t}
\]

Do not average across heads before reconstructing the full model-width key vector.

---

## Scope

Inspect all repository code, tests, examples, scripts, notebooks, documentation, and dataset builders that reference any of the following:

```text
summary
summary_vector
search_by_summary
encode_reference_to_cache
PRACacheEntry
PRAMemoryCache
PRASimpleMemoryCache
last_selected_references
reference dataset
ref dataset
```

Update every affected call site.

Likely primary files include:

```text
src/pra_torch/memory.py
src/pra_torch/model.py
src/pra_torch/attention.py
src/pra_torch/train.py
tests/
examples/
scripts/
```

Do not assume this list is complete. Search the repository.

---

# Required design

## 1. Cache entry

Replace summary-oriented fields with content-derived, layer-specific routing keys.

Preferred structure:

```python
@dataclass
class LayerKV:
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class PRACacheEntry:
    uri: str
    text: str
    layer_kv: dict[int, LayerKV] = field(default_factory=dict)
    layer_routing_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
```

Required routing-key shape:

```text
layer_routing_keys[layer_id]: [d_model]
```

Alternative: derive the routing key lazily from `layer_kv[layer_id].k`.

However, precomputing it once during cache construction is preferred because:

- it avoids repeated mean pooling on every forward pass;
- it makes routing behavior explicit;
- it is easy to test;
- it avoids unnecessary GPU work.

Do not retain `summary` or `summary_vector` as required fields.

If backward compatibility is genuinely needed, make old fields optional and deprecated, but do not use them for routing. Prefer removing them entirely unless existing serialized artifacts require a transition.

## 2. Reference cache construction

Change the API from:

```python
encode_reference_to_cache(
    uri: str,
    text: str,
    summary: str,
    tokenizer,
    device,
)
```

to:

```python
encode_reference_to_cache(
    uri: str,
    text: str,
    tokenizer,
    device,
    metadata: dict | None = None,
)
```

During reference encoding, for each PRA layer:

1. obtain the layer-specific `LayerKV`;
2. store it in `entry.layer_kv[layer_id]`;
3. derive the mean-pooled full-width routing key;
4. detach the routing key;
5. store it in `entry.layer_routing_keys[layer_id]`.

Suggested helper:

```python
def mean_pool_layer_keys(layer_k: torch.Tensor) -> torch.Tensor:
    """
    Convert cached keys from [1, H, T, D_h] to one [D_model]
    routing key by concatenating heads per token and averaging tokens.
    """
    if layer_k.ndim != 4:
        raise ValueError(
            f"Expected layer_k with shape [batch, heads, tokens, head_dim], "
            f"got {tuple(layer_k.shape)}"
        )

    if layer_k.shape[0] != 1:
        raise ValueError(
            "Reference cache construction currently expects one reference "
            "document per encoding call."
        )

    token_keys = (
        layer_k
        .transpose(1, 2)
        .contiguous()
        .view(1, layer_k.shape[2], -1)
    )

    return token_keys.mean(dim=1).squeeze(0).detach()
```

Do not tokenize or encode a separate summary string.

## 3. Layer-specific search API

Replace:

```python
search_by_summary(query, top_k)
```

with a layer-aware name and API, such as:

```python
search_by_routing_key(
    query: torch.Tensor,
    layer_id: int,
    top_k: int = 2,
)
```

or:

```python
search(
    query: torch.Tensor,
    layer_id: int,
    top_k: int = 2,
)
```

The API must support:

```text
query: [batch, d_model]
```

It must return independent results for every batch item.

Recommended return type:

```python
list[list[tuple[PRACacheEntry, float]]]
```

where:

```text
outer list index = batch item
inner list = ranked references for that item
```

Example:

```python
[
    [(entry_a, 0.91), (entry_b, 0.72)],
    [(entry_c, 0.88), (entry_a, 0.60)],
]
```

Do not collapse the batch dimension.

### Search implementation

For a specified `layer_id`:

1. include only entries containing a routing key for that layer;
2. stack routing keys into `[n_refs, d_model]`;
3. normalize queries over `d_model`;
4. normalize routing keys over `d_model`;
5. compute:

```python
scores = normalized_queries @ normalized_keys.T
```

Result:

```text
scores: [batch, n_refs]
```

6. apply `topk` independently across the reference dimension.

Suggested shape-safe implementation:

```python
def search_by_routing_key(
    self,
    query: torch.Tensor,
    layer_id: int,
    top_k: int = 2,
) -> list[list[tuple[PRACacheEntry, float]]]:
    if query.ndim == 1:
        query = query.unsqueeze(0)

    if query.ndim != 2:
        raise ValueError(
            f"Expected query [batch, d_model] or [d_model], "
            f"got {tuple(query.shape)}"
        )

    entries = [
        entry
        for entry in self.all_entries()
        if layer_id in entry.layer_routing_keys
    ]

    if not entries:
        return [[] for _ in range(query.shape[0])]

    routing_keys = torch.stack(
        [
            entry.layer_routing_keys[layer_id].to(
                device=query.device,
                dtype=query.dtype,
            )
            for entry in entries
        ],
        dim=0,
    )

    query_norm = torch.nn.functional.normalize(query, dim=-1)
    key_norm = torch.nn.functional.normalize(routing_keys, dim=-1)

    scores = query_norm @ key_norm.transpose(0, 1)
    k = min(top_k, len(entries))
    values, indices = torch.topk(scores, k=k, dim=-1)

    results = []
    for batch_index in range(query.shape[0]):
        batch_results = []
        for value, entry_index in zip(
            values[batch_index],
            indices[batch_index],
        ):
            batch_results.append(
                (
                    entries[int(entry_index)],
                    float(value.detach().cpu()),
                )
            )
        results.append(batch_results)

    return results
```

Use a clearer production-quality implementation if preferred, but preserve these semantics.

## 4. Attention integration

The PRA layer currently has:

```text
q: [batch, heads, seq, head_dim]
x: [batch, seq, d_model]
```

For this change, use the last token from the projected query, merged back to model width:

```python
routing_query = (
    q[:, :, -1, :]
    .contiguous()
    .view(b, self.d_model)
)
```

This is preferable to routing with:

```python
x[:, -1, :]
```

because cached routing vectors are now derived from projected keys.

The router comparison is therefore:

```text
projected query space vs projected key space
```

Call the cache with the current layer:

```python
retrieved_by_batch = self.pra_cache.search_by_routing_key(
    routing_query,
    layer_id=self.layer_id,
    top_k=self.top_k_refs,
)
```

## 5. Batch-specific memory attention

Each batch item may select a different number or set of references after thresholding.

Do not concatenate one reference set and expand it across the batch.

Implement correct per-example memory attention.

For the current small experimental system, a clear loop over batch items is acceptable and preferred over fragile vectorization.

Suggested approach:

```python
mem_outputs = []

for batch_index, retrieved in enumerate(retrieved_by_batch):
    selected_k = []
    selected_v = []
    selected_metadata = []

    for entry, similarity in retrieved:
        if similarity < self.trigger_threshold:
            continue

        kv = entry.layer_kv.get(self.layer_id)
        if kv is None:
            continue

        selected_k.append(
            kv.k.to(device=x.device, dtype=q.dtype)
        )
        selected_v.append(
            kv.v.to(device=x.device, dtype=q.dtype)
        )
        selected_metadata.append((entry.uri, similarity))

    # Record selections for this batch item.

    if not selected_k:
        # Append a zero memory output for this item.
        continue

    mem_k = torch.cat(selected_k, dim=2)
    mem_v = torch.cat(selected_v, dim=2)

    q_item = q[batch_index : batch_index + 1]

    mem_scores = (
        q_item @ mem_k.transpose(-2, -1)
        / math.sqrt(self.head_dim)
    )
    mem_weights = F.softmax(mem_scores, dim=-1)
    mem_item = self.merge_heads(mem_weights @ mem_v)
    mem_outputs.append(mem_item)
```

Requirements:

- item `i` must only use references retrieved for item `i`;
- no batch item may inherit the reference set of item `0`;
- items with no selected references must receive zero memory contribution;
- local self-attention output must remain unchanged;
- gradients through the active query and memory-attention branch must remain valid;
- cached reference K/V and routing keys remain detached.

## 6. Selection diagnostics

The current field:

```python
last_selected_references: list[tuple[str, float]]
```

cannot represent batch-specific results.

Change it to something explicit, such as:

```python
last_selected_references: list[list[tuple[str, float]]]
```

where the first index is the batch item.

Update:

```python
selected_references_by_layer()
```

and any logging or tests.

Document the exact return structure.

Do not silently flatten selections across the batch.

---

# Dataset changes

## Remove summaries completely

Search every dataset builder, fixture, sample, loader, collator, serialization format, CLI argument, and example for summary fields.

Remove required fields such as:

```text
summary
ref_summary
reference_summary
summaries
summary_text
```

A reference dataset item should require only the data actually needed by the mechanism, for example:

```python
{
    "uri": "doc://example/1",
    "text": "Full referenced content...",
    "metadata": {},
}
```

If the current data format contains:

```python
{
    "uri": ...,
    "text": ...,
    "summary": ...,
}
```

migrate it to:

```python
{
    "uri": ...,
    "text": ...,
}
```

Update generated WikiText-with-references data and any synthetic reference examples.

Do not replace textual summaries with copied or truncated text. The point is to eliminate the summary dependency.

## Backward-compatible parsing

If existing generated data files may already include a `summary` field:

- loaders may ignore unknown legacy `summary` fields temporarily;
- new generated output must not emit them;
- no model or cache API may consume them;
- add a migration note if persisted datasets are checked into the repository.

---

# Variable-count gist architecture

## Core invariant

A reference must no longer map to exactly one routing vector.

The required abstraction is:

```text
reference URI
    -> resolved content
    -> zero or more chunks/partitions
    -> one or more gists per chunk
    -> bounded list of layer-specific routing keys
```

For reference `r` and layer `l`:

```text
routing_gists[r, l]: [G_r,l, d_model]
```

where:

```text
0 <= G_r,l <= max_gists_per_reference
```

`G_r,l` is not tied to `n_heads` and must not be assumed fixed across references, layers, chunking modes, or batches.

The default configuration should normally produce one gist for a short reference, but the data model and search code must support more than one from the start.

## Cache schema

Replace the earlier single-key proposal:

```python
layer_routing_keys: dict[int, torch.Tensor]  # [d_model]
```

with a structured variable-count form:

```python
@dataclass
class ReferenceGist:
    gist_id: str
    layer_id: int
    vector: torch.Tensor          # [d_model]
    chunk_id: str | None = None
    token_start: int | None = None
    token_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_uri: str | None = None
    depth: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class PRACacheEntry:
    uri: str
    text: str
    layer_kv: dict[int, LayerKV] = field(default_factory=dict)
    layer_gists: dict[int, list[ReferenceGist]] = field(default_factory=dict)
    child_uris: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

Required invariants:

- `ReferenceGist.vector` is always rank 1 with shape `[d_model]`;
- `layer_gists[layer_id]` is a list and may be empty;
- list length must never exceed `max_gists_per_reference`;
- every gist retains provenance back to the reference and, when applicable, the chunk;
- do not represent gists by padding inside cache entries;
- padding/masking may be introduced only transiently for batched similarity computation;
- do not overwrite full `layer_kv` when producing multiple gists;
- full K/V and routing gists are separate concepts.

## Configuration

Add explicit configuration with validation:

```yaml
reference_routing:
  gist_mode: mean
  max_gists_per_reference: 4
  chunking_mode: none
  fixed_chunk_tokens: 64
  fixed_chunk_overlap_tokens: 0
  marker_rules: []
  semantic_chunker: null
  gist_overflow_policy: truncate
  score_aggregation: max
  recursive_refs:
    enabled: true
    max_depth: 2
    max_total_references: 16
    max_total_tokens: 2048
    cycle_policy: skip
    missing_ref_policy: warn
```

Suggested Python types:

```python
GistMode = Literal[
    "mean",
    "last",
    "ref_end",
    "gru",
]

ChunkingMode = Literal[
    "none",
    "fixed",
    "markers",
    "semantic",
]

GistOverflowPolicy = Literal[
    "truncate",
    "merge_tail",
    "error",
]

GistScoreAggregation = Literal[
    "max",
    "mean",
    "logsumexp",
]
```

`mean` is the default gist mode.

`none` is the default chunking mode for short-reference compatibility.

The `semantic` mode must fail with a clear `NotImplementedError` unless a concrete chunker plugin/callback is supplied. Do not silently fall back to fixed or mean chunking, because that would invalidate experiments.

# Gist computation modes

## Shared input convention

All gist modes operate on the current layer's contextualized projected reference keys, not raw token embeddings and not external summaries.

Starting tensor:

```text
layer_k: [1, H, T, D_h]
```

Merge heads per token:

```python
token_keys = (
    layer_k
    .transpose(1, 2)
    .contiguous()
    .view(1, T, H * D_h)
    .squeeze(0)
)
```

Result:

```text
token_keys: [T, d_model]
```

When chunking is enabled, compute each gist from the token slice belonging to that chunk.

Never average across references.

Never average across chunks unless an explicit overflow policy requests it.

## Mode: `mean` — default

For chunk token keys `X in [T_chunk, d_model]`:

```python
gist = X.mean(dim=0)
```

Properties:

- deterministic;
- parameter-free;
- parallel;
- stable baseline;
- one gist per chunk;
- detached before cache storage.

Mean is the default because it is the simplest content-derived baseline, not because it is assumed optimal.

## Mode: `last`

Use:

```python
gist = X[-1]
```

This means the final projected key in the chunk.

Do not confuse:

- the final content token;
- an EOS token;
- a reference-end marker;
- a padding token.

Padding must be removed before selecting `last`.

Store enough metadata to identify which token supplied the gist.

## Mode: `ref_end`

Use the projected key at an explicit reference/chunk terminator token.

Requirements:

- the tokenizer must preserve the token atomically;
- each encoded chunk/reference must contain exactly one selected terminator;
- padding cannot be mistaken for it;
- the code must validate the index;
- if the marker is missing or duplicated, raise a clear error unless an explicit fallback policy is configured;
- do not treat arbitrary punctuation as `ref_end`.

This mode may use `<REF_END>` or a dedicated `<GIST>` token, but naming must be centralized in tokenizer/config code.

## Mode: `gru`

The GRU is a learned gist encoder over contextual projected keys:

```text
[T_chunk, d_model]
    -> optional down projection
    -> GRU
    -> final hidden state
    -> output projection
    -> [d_model]
```

Suggested module:

```python
class GRUGistPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        ...

    def forward(
        self,
        token_keys: torch.Tensor,  # [B, T, d_model]
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:             # [B, d_model]
        ...
```

For the causal/reference-cache baseline:

- default to unidirectional;
- use packed sequences or explicit lengths so padding is ignored;
- initialize hidden state deterministically;
- ensure parameters are registered in the model and included in checkpoints;
- ensure optimizer construction includes gist-pooler parameters;
- do not create a new GRU inside each cache-building call;
- do not detach inputs before the GRU when training the gist mechanism;
- detach only the cached output after it is computed;
- decide explicitly whether one pooler is shared across PRA layers or layer-conditioned.

Preferred initial design:

```text
one shared GRU gist pooler
+ learned layer embedding or layer-specific output projection
```

This avoids one full GRU per layer while preserving layer identity.

However, if cache building is permanently under `torch.no_grad()`, the GRU cannot learn from downstream objectives. Codex must not quietly add a GRU that never receives gradients.

Therefore introduce two explicit cache-building modes:

```text
offline/detached cache construction
trainable gist construction
```

Document which one is used by each experiment.

For initial correctness tests, GRU mode may be exercised in forward-only form, but its trainability must be tested separately.

## Future modes

Design the interface so these can be added without changing cache/search schemas:

```text
attention_pool
multi_latent
state_space
recurrent_attractor
```

Do not implement them in this patch unless needed by existing tests.

# Chunking and partitioning

## General requirements

Chunking must produce structured objects, not raw strings alone:

```python
@dataclass(frozen=True)
class ReferenceChunk:
    chunk_id: str
    source_uri: str
    text: str
    char_start: int
    char_end: int
    token_start: int | None = None
    token_end: int | None = None
    marker_type: str | None = None
    metadata: dict = field(default_factory=dict)
```

Chunk IDs must be deterministic for the same source, configuration, and content.

Suggested form:

```text
<canonical-uri>#chunk=<index>
```

or a stable content/config hash.

Do not use Python's process-randomized `hash()`.

Chunk boundaries and provenance must survive into `ReferenceGist`.

## Mode: `none`

Treat the whole resolved reference content as one chunk.

If the reference exceeds model/cache token limits:

- do not silently truncate without diagnostics;
- apply an explicit configurable policy;
- record original and retained lengths;
- emit a warning or trace field;
- preserve deterministic behavior.

## Mode: `fixed`

Split by tokenizer token count, not bytes and not Python characters.

Configuration:

```yaml
chunking_mode: fixed
fixed_chunk_tokens: 64
fixed_chunk_overlap_tokens: 0
```

Requirements:

- `fixed_chunk_tokens > 0`;
- `0 <= overlap < fixed_chunk_tokens`;
- no empty chunks;
- exact token spans stored;
- decoded chunk text is optional; token IDs/spans are authoritative;
- special reference markers must not be broken into partial tokens;
- references at chunk boundaries require defined handling;
- overlap must not produce duplicate `gist_id` values;
- final short chunk is retained;
- maximum number of produced gists is enforced after chunking.

For RNN gist experiments, overlap should default to zero so recurrence receives a clean partition. Overlap is experimental and must be surfaced in run metadata.

## Mode: `markers`

Partition using structural markers supplied by a parser or document adapter.

Examples:

```text
Markdown headings
HTML sections
XML elements
JSON fields/objects
source-code functions/classes
notebook cells
table sections
explicit PRA anchors
```

Do not implement marker-based chunking as one giant regex over every format.

Define an adapter interface:

```python
class ReferencePartitioner(Protocol):
    def partition(
        self,
        uri: str,
        text: str,
        metadata: dict,
    ) -> list[ReferenceChunk]:
        ...
```

Provide at least:

```text
plain explicit-marker partitioner
Markdown heading partitioner
```

if the current repository already contains corresponding document stages.

Requirements:

- parser failures are explicit;
- malformed nesting does not cause infinite loops;
- empty structural sections are skipped or represented according to a documented rule;
- parent heading/context may be included in chunk metadata;
- markers may be retained or removed according to explicit config;
- chunk text and spans must remain traceable to source;
- structure metadata must be serializable;
- parser-specific code stays outside attention/model modules.

Marker-based chunks may produce variable counts naturally.

## Mode: `semantic`

This is an extension point, not a hard-coded algorithm in this patch.

Define:

```python
class SemanticChunker(Protocol):
    def partition(
        self,
        uri: str,
        text: str,
        token_ids: list[int],
        metadata: dict,
        max_chunks: int,
    ) -> list[ReferenceChunk]:
        ...
```

The core implementation must support receiving variable chunks from this interface.

Do not implement a simplistic cosine-threshold splitter and call the problem solved.

A future semantic chunker may use:

- embedding discontinuities;
- topic-boundary prediction;
- discourse segmentation;
- model-predicted boundaries;
- change-point detection;
- recurrent state shifts;
- learned end-to-end segmentation.

Semantic chunking should be treated as a separate research track/paper.

# Bounding variable gists

## Maximum count

Add:

```text
max_gists_per_reference
```

with a small default such as `4`.

This is a hard safety and compute bound.

The bound applies after:

- recursive expansion decisions;
- chunk generation;
- gist generation.

Do not confuse it with:

```text
top_k_refs
top_k_gists
n_heads
max_recursive_references
```

## Overflow policies

Implement explicit policies:

### `truncate`

Keep the first `max_gists_per_reference` deterministic chunks.

This is simple but structurally biased; traces must report discarded chunks.

### `merge_tail`

Keep the first `max_gists_per_reference - 1`, then merge all remaining chunk token ranges into the final gist input.

This preserves broad coverage but may create an oversized heterogeneous tail.

### `error`

Raise a descriptive exception.

Useful for controlled experiments where silent information loss is unacceptable.

Do not randomly sample chunks unless a separate explicitly seeded policy is later added.

## Search semantics with multiple gists

Flatten gists for similarity calculation while retaining parent-entry indices:

```text
all_gists: [N_total_gists, d_model]
gist_to_entry: [N_total_gists]
gist_to_chunk: metadata mapping
```

For each batch query:

```text
gist_scores: [B, N_total_gists]
```

Then aggregate gist scores to reference scores.

Default:

```text
reference score = max score among that reference's gists
```

Formally:

```text
score(q, r) = max_g cosine(q, gist[r, g])
```

This corresponds to "a reference is relevant if at least one of its gists is relevant."

Support configurable:

```text
max
mean
logsumexp
```

Do not average gist vectors first; that would collapse the variable-gist abstraction back into a single vector.

Return both:

- aggregate reference score;
- winning gist/chunk provenance.

Recommended result type:

```python
@dataclass(frozen=True)
class RetrievalHit:
    entry: PRACacheEntry
    score: float
    gist_id: str
    chunk_id: str | None
    gist_index: int
    metadata: dict
```

Batch result:

```python
list[list[RetrievalHit]]
```

Reference `top_k` must deduplicate parent references after gist scoring.

If the same reference has several high-scoring gists, it should occupy one reference slot unless a separate chunk-level retrieval mode is explicitly configured.

# Recursive references

## Current status

The current standalone implementation is not recursive.

The cache builder runs each reference through blocks with:

```python
x = block(x, use_pra_memory=False)
```

which explicitly disables reference-memory use while encoding a reference.

Therefore a reference containing another reference marker does not resolve or attend to its child during cache construction.

The README describes recursive anchor expansion as a goal, not a completed implementation.

## Required semantics

A resolved reference may contain atomic reference markers mapped to child URIs.

Example:

```text
parent URI: doc://manual/overview

content:
"The authentication policy is defined in <REF_7>."

local reference table:
<REF_7> -> doc://manual/auth-policy
```

Recursive cache construction should:

1. resolve parent URI;
2. parse parent content and local reference table;
3. discover child URIs;
4. recursively ensure child cache entries exist;
5. encode the parent with child PRA memory enabled, subject to policy;
6. store parent K/V and gists;
7. retain provenance/dependency edges.

## Resolver contract

Do not parse child URIs from arbitrary visible text alone.

Use an explicit resolver result:

```python
@dataclass
class ResolvedReference:
    uri: str
    text: str
    reference_table: dict[int, str] = field(default_factory=dict)
    anchors: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    version: str | None = None
```

Reference markers must be atomic tokenizer tokens and mapped through the local table.

The resolver, tokenizer, dataset reference-table representation, and cache builder must agree on marker identity.

Do not use global `<REF_n>` meaning across unrelated documents unless explicitly namespaced.

## Recursive builder

Introduce a dedicated builder/service rather than embedding recursion directly in `TinyPRAModel.encode_reference_to_cache`.

Suggested interface:

```python
class RecursiveReferenceCacheBuilder:
    def __init__(
        self,
        model,
        resolver,
        tokenizer,
        cache,
        config,
    ):
        ...

    def ensure_cached(
        self,
        uri: str,
        *,
        depth: int = 0,
        ancestry: tuple[str, ...] = (),
        budget: ResolutionBudget | None = None,
    ) -> PRACacheEntry:
        ...
```

Do not make model modules responsible for filesystem/network resolution.

## Recursion ordering

Use depth-first child-first construction for the initial implementation:

```text
resolve parent
discover children
cache children recursively
encode/cache parent with available child memory
```

This makes child K/V available when encoding the parent.

Document that parent representations may therefore incorporate child memory.

A later mode may preserve parent-local-only encoding, but do not mix semantics silently.

## Cycle safety

Cycles are expected in real document graphs.

Example:

```text
A -> B -> C -> A
```

Maintain the current ancestry stack.

Before descending to child:

```python
if child_uri in ancestry or child_uri == current_uri:
    apply cycle_policy
```

Default:

```text
cycle_policy = skip
```

Record a trace event and dependency edge marked as cyclic.

Supported policies may include:

```text
skip
error
link_only
```

Never recurse until Python raises `RecursionError`.

## Depth and global budgets

Required controls:

```text
max_depth
max_total_references
max_total_tokens
max_children_per_reference
```

All apply to one root-resolution operation.

Use a mutable/shared `ResolutionBudget` object for the traversal, not separate counters reset at each recursive call.

When exhausted:

- stop expansion deterministically;
- record the reason;
- optionally retain unresolved child links;
- do not partially mutate a cache entry and present it as complete.

## Cache construction states

Avoid duplicate work and re-entrant corruption.

Track:

```text
MISSING
BUILDING
READY
FAILED
```

If a URI is encountered while `BUILDING`, treat it as a cycle/re-entrant dependency.

Only expose a cache entry as `READY` after K/V, gists, and metadata are complete.

On failure:

- clear or mark the partial entry as `FAILED`;
- preserve the exception/trace;
- do not leave malformed tensors in the searchable cache.

## Cache identity and invalidation

Recursive cache correctness requires identity beyond URI when content changes.

At minimum record:

```text
canonical_uri
content/version fingerprint
model/checkpoint fingerprint
tokenizer fingerprint
chunking configuration fingerprint
gist configuration fingerprint
recursive-policy fingerprint
```

An entry produced with different gist/chunk/model configuration must not be reused silently.

A complete invalidation framework is not required now, but cache metadata and equality checks must make stale reuse detectable.

## Recursive training and gradient semantics

The current reference cache is detached and built under `torch.no_grad()`.

Keep this default for stable experiments.

However, recursive child influence means:

```text
child cached K/V
    -> parent encoding
    -> parent cached K/V/gists
```

This remains detached between cache-building stages.

Do not accidentally retain a graph through the entire document tree.

If trainable recursive gist construction is later enabled, it must use an explicit bounded training mode and cannot reuse stale detached caches as though gradients were available.

## Recursive chunk/gist interaction

A parent may produce its own chunks and gists while also depending on child references.

Do not automatically append every child gist to the parent's `layer_gists`.

Keep identities distinct:

```text
parent gists describe parent encoded content
child gists belong to child cache entries
dependency graph links them
```

If parent encoding uses child PRA memory, its contextualized keys may indirectly incorporate child information. That is sufficient for the initial recursive mode.

A future explicit inherited-gist mode must be separately named and tested.

# Implementation order

Codex must implement in this order:

1. Fix batch-specific routing and memory attention.
2. Remove textual summary fields and dataset dependence.
3. Introduce variable-count `ReferenceGist` schema.
4. Implement whole-reference `mean` gist and tests.
5. Implement `last` and optional validated `ref_end`.
6. Implement fixed-size chunking.
7. Implement marker-partitioner interface and at least the parsers required by current datasets.
8. Implement flattened gist search plus parent-reference aggregation.
9. Add bounded recursive resolver/cache builder.
10. Add GRU gist mode with checkpoint/optimizer/gradient tests.
11. Add semantic chunker interface that fails clearly without an implementation.
12. Update documentation and run all smoke/regression tests.

Do not combine all changes into one unreviewable function.

Prefer small modules:

```text
gist.py
chunking.py
resolution.py
memory.py
attention.py
```

or equivalent repository-consistent names.


# Dynamic memory batching and bucketed rectangular attention

## Motivation

Different batch elements may select:

- different numbers of references;
- different numbers of gists;
- different chunk spans;
- different total numbers of detailed K/V positions.

Therefore, for batch item `i`, the selected memory length is:

```text
M_i
```

and generally:

```text
M_0 != M_1 != ... != M_(B-1)
```

The implementation must support this without leaking memory across examples and without requiring one fixed global memory length.

The rectangular tensor size may vary between forward passes.

## Configuration

Add:

```yaml
reference_routing:
  memory_bucket_count: 1
```

Semantics:

```text
memory_bucket_count = 0
    Process each batch item independently.
    No padding rectangle across different examples.
    This is the reference/debug implementation.

memory_bucket_count = 1
    Create one rectangular memory tensor for the whole current batch.
    Pad every item to the maximum selected memory length in that batch.
    Use an explicit validity mask.

memory_bucket_count >= 2
    Partition batch items into up to this many memory-length buckets.
    Items with similar selected memory lengths share one padded rectangle.
    Run one rectangular attention operation per non-empty bucket.
    Restore outputs to original batch order.
```

The parameter controls the maximum number of dynamic buckets, not a fixed bucket width.

Validate:

```text
memory_bucket_count >= 0
```

Do not overload this parameter with token chunking or gist count.

Recommended defaults:

```text
development/debug: 0
normal experiments: 1
performance experiments: 2, 4, or 8
```

## Mode 0: per-element iteration

For each batch item:

```python
q_i:     [1, H, Q, D_h]
mem_k_i: [1, H, M_i, D_h]
mem_v_i: [1, H, M_i, D_h]
```

Run attention independently and concatenate outputs in original order.

This mode is the semantic reference implementation.

Requirements:

- support `M_i = 0`;
- no padding is required;
- return exact zero memory contribution for empty memory;
- preserve autograd through query and active memory-attention computations;
- use this mode as the oracle for numerical comparison with rectangular modes.

## Mode 1: one dynamic rectangle

Compute:

```text
M_max = max_i M_i
```

Construct:

```text
mem_k:    [B, H, M_max, D_h]
mem_v:    [B, H, M_max, D_h]
mem_mask: [B, 1, 1, M_max]
lengths:  [B]
```

Each item occupies:

```text
mem_k[i, :, :M_i, :]
mem_v[i, :, :M_i, :]
```

and the remaining positions are padding.

The rectangle width is dynamic and may differ across batches.

Example:

```text
forward batch A lengths: [18, 40, 31]
rectangle width: 40

forward batch B lengths: [90, 85, 120]
rectangle width: 120
```

### Masking

Before softmax:

```python
scores = scores.masked_fill(
    ~mem_mask,
    torch.finfo(scores.dtype).min,
)
```

Do not rely on zero K/V padding without a mask.

### Empty-memory elements

If `M_i = 0`, masking every position can generate NaNs.

Use one of these correct approaches:

1. temporarily expose a dummy zero K/V position and multiply the final output by `has_memory`; or
2. exclude empty-memory items from the rectangular operation and scatter exact zeros back.

The second approach is preferred when practical.

Required final invariant:

```text
memory_output[i] == 0 exactly when M_i == 0
```

within floating-point representation.

## Mode 2+: dynamic length bucketing

After gist/reference selection and detail-region materialization, calculate:

```python
memory_lengths = [M_0, M_1, ..., M_(B-1)]
```

Partition batch indices into up to:

```text
K = memory_bucket_count
```

groups with similar lengths.

For each group:

1. gather the corresponding query rows;
2. create a rectangle padded only to the group's local maximum;
3. apply the validity mask;
4. compute memory attention;
5. scatter outputs back to original batch indices.

Example:

```text
memory lengths: [12, 210, 18, 95, 200, 0, 105, 14]
memory_bucket_count: 3

possible deterministic buckets:
bucket 0: lengths [0, 12, 14, 18], max 18
bucket 1: lengths [95, 105], max 105
bucket 2: lengths [200, 210], max 210
```

This computes substantially less padded attention than one width-210 rectangle.

## Bucket construction strategy

Implement a deterministic strategy.

Preferred initial strategy:

```text
1. separate zero-memory items;
2. stable-sort non-empty items by memory length;
3. partition sorted items into at most K contiguous groups;
4. minimize or approximately minimize padding cost;
5. preserve original indices for scatter-back.
```

### Baseline equal-count strategy

A simple initial strategy may divide the sorted list into approximately equal-size groups.

However, this can be poor when lengths are highly skewed.

### Preferred padding-cost strategy

Define bucket cost:

```text
cost(bucket) = number_of_items(bucket) * max_length(bucket)
```

Total padded positions:

```text
total_cost = sum_b cost(bucket_b)
```

Useful padding waste metric:

```text
waste =
    total_cost
    - sum_i M_i
```

For small batch sizes and small `K`, dynamic programming can find the optimal contiguous partition after sorting.

Suggested objective:

```text
minimize sum_b |bucket_b| * max_length(bucket_b)
```

subject to:

```text
number of buckets <= K
```

Because sorted buckets are contiguous, the maximum is the last length in each bucket.

Codex may implement:

- exact dynamic programming for controlled experimental batches; or
- a deterministic greedy approximation behind a separate strategy flag.

Expose:

```yaml
memory_bucket_strategy: optimal_contiguous
```

Optional later alternatives:

```text
equal_count
power_of_two
greedy_padding
fixed_boundaries
```

Default for `memory_bucket_count >= 2`:

```text
optimal_contiguous
```

if batch sizes remain modest.

Do not use nondeterministic clustering.

## Suggested bucket planner interface

```python
@dataclass(frozen=True)
class MemoryBucket:
    original_indices: torch.Tensor
    lengths: torch.Tensor
    max_length: int


class MemoryBucketPlanner:
    def plan(
        self,
        lengths: Sequence[int],
        max_buckets: int,
    ) -> list[MemoryBucket]:
        ...
```

Planner requirements:

- every batch index appears exactly once;
- bucket indices are disjoint;
- empty-memory items are represented explicitly or returned separately;
- bucket order is deterministic;
- original batch order can be restored;
- no tensor data is copied during planning;
- planning may run on CPU using integer lengths;
- do not synchronize GPU repeatedly to extract lengths one item at a time if lengths are already known from Python-side selection metadata.

## Rectangular attention helper

Create one reusable, tested helper rather than duplicating mask logic:

```python
def padded_memory_attention(
    q: torch.Tensor,
    memory_k_by_item: Sequence[torch.Tensor],
    memory_v_by_item: Sequence[torch.Tensor],
    *,
    indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    q:
        [B_group, H, Q, D_h]

    each memory tensor:
        [1, H, M_i, D_h]

    returns:
        [B_group, H, Q, D_h]
    """
```

Or use an equivalent structure consistent with the repository.

The helper must:

- allocate on the query device;
- use query dtype;
- preserve head count and head dimension;
- validate K/V length equality;
- validate consistent H and D_h;
- mask all padding;
- return exact zero outputs for empty-memory rows;
- avoid accidental `.expand()` of one item's memory into another;
- support mixed precision safely;
- avoid NaNs with FP16/BF16;
- keep gradients through active query/memory paths.

## Output restoration

For bucketed execution:

```python
final_mem_out = torch.zeros_like(q)
```

For each bucket:

```python
bucket_q = q.index_select(0, bucket_indices)
bucket_out = padded_attention(...)
final_mem_out.index_copy_(0, bucket_indices, bucket_out)
```

Equivalent safe scatter logic is acceptable.

Verify that output order exactly matches input batch order.

Do not concatenate bucket outputs without restoring order.

## Relationship to gist selection

The selected memory length must be computed after:

```text
query
-> gist matching
-> top-k gist selection
-> duplicate-region policy
-> recursive/reference resolution
-> associated detail-region materialization
```

For batch item `i`:

```text
M_i = total number of selected detailed K/V positions
```

If operating in compressed gist-only mode:

```text
M_i = number of selected gist K/V positions
```

The same batching architecture supports both.

## Duplicate and overlapping regions

Multiple selected gists may point to:

- the same region;
- overlapping token spans;
- the same reference and chunk.

Before calculating `M_i`, apply an explicit region policy:

```yaml
selected_region_merge_policy: union
```

Recommended default:

```text
union
```

Meaning:

- identical regions are deduplicated;
- overlapping spans from the same reference/layer are merged;
- token K/V is not repeated unnecessarily.

Alternative experimental policy:

```text
keep_duplicates
```

must be explicit because duplicate K/V changes attention mass.

Do not silently duplicate overlapping token regions.

## Complexity metrics and diagnostics

Record per layer and batch:

```text
selected_memory_lengths
number_of_nonempty_items
requested_bucket_count
actual_bucket_count
bucket_membership
bucket_max_lengths
valid_memory_positions = sum_i M_i
allocated_memory_positions = sum_b |bucket_b| * max_length_b
padding_positions
padding_fraction
planner_time
attention_time
```

Padding fraction:

```text
padding_fraction =
    padding_positions / max(allocated_memory_positions, 1)
```

These metrics are essential for determining whether bucketing gives a real systems benefit.

## Tests for memory batching

Add all of the following.

### Numerical equivalence

For deterministic tensors, compare:

```text
memory_bucket_count = 0
memory_bucket_count = 1
memory_bucket_count = 2
memory_bucket_count = 4
```

Outputs must match within dtype-appropriate tolerance.

Also compare gradients for selected model parameters and queries.

### Different lengths

Test lengths such as:

```text
[0, 1, 7, 32, 3, 17]
```

### All empty

Every item has zero selected memory.

Expected:

- exact zero memory output;
- no NaNs;
- local attention unchanged.

### One non-empty item

Only one batch element has memory.

Verify no leakage.

### Bucket count greater than batch size

For:

```text
memory_bucket_count > number of non-empty items
```

create no more than one bucket per non-empty item.

Do not create empty compute buckets.

### Equal lengths

All `M_i` equal.

Bucketed and single-rectangle modes should avoid unnecessary complexity and produce equivalent allocation cost.

### Highly skewed lengths

Example:

```text
[8, 9, 10, 11, 1024]
```

Verify that `K >= 2` isolates or sensibly groups the outlier and reduces allocated positions compared with `K = 1`.

### Stable ordering

Permute input batch order while retaining values and expected outputs.

Verify scatter-back correctness.

### Overlapping gist regions

Verify union/deduplication policy before length calculation.

### Mixed precision

Run FP32 and, where hardware permits, FP16/BF16 smoke tests.

### Compile compatibility

If the project uses `torch.compile`, test each mode.

Dynamic bucket planning and variable shapes may cause graph breaks. Report them honestly rather than hiding them.

## Acceptance criteria additions

- [ ] `memory_bucket_count = 0` runs independent per-item memory attention.
- [ ] `memory_bucket_count = 1` runs one masked rectangle dynamically sized to the current batch.
- [ ] `memory_bucket_count >= 2` groups similar memory lengths into at most the configured number of rectangles.
- [ ] Bucketed outputs and gradients match mode 0 within tolerance.
- [ ] Empty-memory elements never produce NaNs.
- [ ] Outputs are restored to the original batch order.
- [ ] Padding and bucket-efficiency metrics are logged.
- [ ] Highly skewed batches show reduced allocated memory positions with multiple buckets.
- [ ] No item can access another item's selected K/V.
- [ ] Bucket count is independent of attention-head count, gist count, chunk count, and top-k settings.

## Non-goals for this batching patch

Do not implement yet:

- distributed cross-device bucket migration;
- persistent global batching queues;
- asynchronous memory prefetch;
- custom CUDA kernels;
- block-sparse kernels;
- paged K/V allocators;
- production-level compiler specialization;
- reinforcement-learned bucket assignment.

Keep interfaces open for these future directions.

# Critical safeguards for Codex

- Do not reintroduce textual summaries under another name.
- Do not use raw token embeddings for routing.
- Do not assume one gist per reference.
- Do not assume gist count equals head count.
- Do not pad cached gist lists permanently.
- Do not select top-k gists and accidentally return duplicate references.
- Do not average all gists back into one vector.
- Do not use byte offsets as token offsets.
- Do not split atomic reference-marker tokens.
- Do not silently truncate references or gist lists.
- Do not silently fall back from semantic chunking.
- Do not put parsing/network/filesystem work inside attention forward.
- Do not resolve references during every token-generation step.
- Do not let one batch item's selected references leak into another.
- Do not enable recursive PRA without depth/cycle/global-budget controls.
- Do not expose partially built recursive entries to search.
- Do not create GRU modules dynamically during cache construction.
- Do not place trainable GRU work permanently inside `torch.no_grad()`.
- Do not change local self-attention behavior while implementing routing.
- Do not claim recursion is complete merely because child URIs are parsed; verify child memory affects parent encoding in a dedicated test.

# Tests

Add or update tests covering all the following.

## 1. Mean pooling shape and value

Construct a deterministic key tensor:

```text
[1, H, T, D_h]
```

Verify:

- output shape is `[H * D_h]`;
- heads are concatenated per token before averaging;
- arithmetic result matches a manually computed expected tensor.

Use values that distinguish:

```text
mean(concatenated per-token heads)
```

from accidental alternatives such as:

```text
mean over heads
mean over all dimensions
flatten without transpose
```

## 2. Layer-specific routing

Create two entries whose routing keys differ by layer.

Verify that the same query can select different references for:

```text
layer_id = 1
layer_id = 2
```

This proves that routing no longer uses one global summary vector.

## 3. Batch-specific retrieval

Create at least two orthogonal routing keys and two query examples:

```text
query[0] should retrieve ref A
query[1] should retrieve ref B
```

Verify independent top-k results.

This test must fail under the old `query = query[0]` behavior.

## 4. Batch-specific memory integration

Run a PRA attention forward pass with two batch items whose correct references contain distinguishable K/V tensors.

Verify:

- batch item `0` records reference A;
- batch item `1` records reference B;
- changing item `1`'s query does not alter item `0`'s selection;
- memory output is not simply expanded from the first item.

## 5. No-match behavior

With a threshold above all similarities:

- output must equal local attention output;
- memory contribution must be zero;
- diagnostic selection lists must be empty for every batch item;
- no NaNs or shape errors.

## 6. Empty cache behavior

Verify normal local attention with an empty cache.

## 7. Fewer references than top-k

Verify correct operation when:

```text
n_refs < top_k_refs
```

## 8. Missing layer data

If a cache entry lacks `layer_routing_keys[layer_id]` or `layer_kv[layer_id]`:

- skip it safely;
- do not crash;
- do not return it as usable for that layer.

## 9. No summary fields

Add an integration test that:

1. creates a reference from URI and text only;
2. encodes it into the cache;
3. performs retrieval;
4. runs memory attention.

The word `summary` must not be required anywhere in this path.

## 10. Training smoke test

Run the smallest available training or forward/backward smoke test with:

```text
batch_size > 1
PRA memory enabled
at least two references
```

Verify:

- forward pass completes;
- loss is finite;
- backward pass completes;
- parameters expected to receive gradients do so;
- cached tensors stay detached.

---

# Naming cleanup

Rename summary-specific concepts throughout production code:

```text
search_by_summary       -> search_by_routing_key
summary_vector          -> layer_routing_keys[layer_id]
summary similarity      -> routing similarity
summary retrieval       -> reference routing
```

Use `routing_key`, `reference_key`, or `pooled_key` consistently.

Avoid calling these vectors "embeddings" when they are projected attention keys. Prefer:

```text
reference routing key
pooled reference key
layer-specific routing key
```

---

# Documentation updates

Update README, architecture docs, comments, diagrams, and paper-support notes that describe summary-based retrieval.

The corrected implementation should be described as:

```text
For each cached reference and PRA layer, the implementation mean-pools the
reference's contextualized, layer-specific projected keys into one model-width
routing key. During a forward pass, the layer compares each batch item's
projected last-token query with these routing keys using cosine similarity,
selects references independently per batch item, and performs normal multi-head
attention over the selected references' full cached K/V tensors.
```

Clearly state that:

- textual summaries are not required;
- routing is based on reference content;
- routing keys are layer-specific;
- mean pooling is the initial baseline;
- chunking/partitioning is future work;
- head-specific routing is an experimental extension, not part of this patch.

Correct any comments that still say:

```text
summary vector
search by summary
raw summary embeddings
```

---

# Non-goals

Do not implement these in this patch:

- independent top-k retrieval per attention head;
- IDF or TF-IDF weighting;
- Sentence Transformer encoders;
- separate external retrieval models;
- a concrete semantic/dynamic chunking algorithm;
- approximate nearest-neighbor indexes;
- a full production cache invalidation framework;
- new large-scale experiments;
- changes to the fundamental local-attention branch.

These should remain separate experiments or future issues.

---

# Performance guidance

Correctness comes first.

The cache and models are currently small enough that:

- stacking routing keys per search;
- cosine similarity by dense matrix multiplication;
- looping over batch items for memory attention;

are acceptable.

Avoid premature vectorization that reintroduces cross-example contamination.

After correctness tests pass, minor refactoring is allowed, but retain simple, auditable tensor semantics.

---

# Acceptance criteria

The work is complete only when all of the following are true:

- [ ] No production dataset requires or generates textual reference summaries.
- [ ] `encode_reference_to_cache` accepts URI and text without a summary.
- [ ] `PRACacheEntry` no longer requires `summary` or `summary_vector`.
- [ ] Every PRA layer stores a variable-length list of layer-specific routing gists.
- [ ] Mean pooling over projected reference keys is the default gist mode.
- [ ] `last` is supported as an experimental gist mode.
- [ ] GRU gist mode is structurally supported and its parameters/checkpoint/gradient path are tested.
- [ ] One reference can produce multiple gists up to `max_gists_per_reference`.
- [ ] Fixed-size and marker-based chunking produce deterministic provenance-preserving chunks.
- [ ] Semantic chunking is exposed only through an explicit interface and does not silently fall back.
- [ ] Gist-level scores are aggregated and deduplicated into reference-level top-k results.
- [ ] The routing query is the current layer's projected last-token query.
- [ ] Retrieval is independent for every batch item.
- [ ] Memory K/V is not expanded from batch item `0` to other items.
- [ ] Selection diagnostics preserve batch identity.
- [ ] Empty-cache and no-match behavior remain correct.
- [ ] Tests cover layer specificity, batch specificity, and exact mean pooling.
- [ ] A batch-size-greater-than-one PRA forward/backward smoke test passes.
- [ ] Existing non-PRA and batch-size-one behavior remains operational.
- [ ] Documentation no longer presents textual summaries as the core router.
- [ ] Nested references are resolved child-first with bounded depth and global budgets.
- [ ] Cycles cannot cause unbounded recursion.
- [ ] Recursive cache entries have explicit build states and partial entries are never searchable.
- [ ] Tests prove that child memory can affect parent reference encoding when recursion is enabled.

---

# Final Codex report

After implementation, report:

1. files changed;
2. old and new cache schemas;
3. old and new tensor shapes;
4. how batch-specific routing is represented;
5. tests added;
6. commands run and results;
7. any legacy datasets or serialized caches that require regeneration;
8. any remaining references to `summary` and why they are unrelated or intentionally retained;
9. recommended follow-up experiments, without implementing them.

Recommended follow-up issues:

```text
1. Compare mean, last, ref-end, and GRU gists under equal routing budgets.
2. Compare fixed and marker-based chunking on the same references.
3. Develop dynamic semantic chunking as a separate research line/paper.
4. Compare shared full-width routing with head-aware routing.
5. Add learned attention pooling and multiple latent gist states.
6. Evaluate recursion depth, cycle structure, and cache-staleness effects.
```


# Architectural correction: references, chunks, routing gists, and token K/V

This section overrides any earlier wording in this file that suggests:

- arbitrary "regions" inside a chunk;
- multiple gists per chunk as part of the initial implementation;
- one global set of gist regions detached from chunk identity;
- selecting a gist and then materializing an undefined subregion;
- using `top_k_regions_per_reference`.

The initial implementation must use the following precise hierarchy:

```text
Reference
    -> one or more Chunks
        -> exactly one routing gist per chunk
        -> the full token-level K/V sequence for that chunk
```

A chunk is the smallest materializable detailed-memory unit in this patch.

There are no additional "regions inside a chunk."

A gist does not represent an arbitrary token subspan inside its chunk.

The gist is the routing representation for the entire chunk.

## Required object model

Use or adapt the following structure:

```python
@dataclass
class ChunkKV:
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class ChunkRoutingGist:
    k: torch.Tensor
    v: torch.Tensor | None
    method: str


@dataclass
class ReferenceChunkMemory:
    chunk_id: str
    source_uri: str

    token_start: int
    token_end: int

    char_start: int | None
    char_end: int | None

    token_kv: ChunkKV
    routing_gist: ChunkRoutingGist

    metadata: dict = field(default_factory=dict)


@dataclass
class LayerReferenceMemory:
    chunks: list[ReferenceChunkMemory]


@dataclass
class PRACacheEntry:
    uri: str
    text: str
    layer_memory: dict[int, LayerReferenceMemory]
    child_uris: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

Equivalent repository-consistent types are acceptable, but the semantics must be identical.

Required invariants:

- each chunk contains exactly one `routing_gist` in the initial implementation;
- each chunk contains its own token-level K/V;
- each gist points to exactly one chunk;
- the detailed K/V materialized after chunk selection is that chunk's entire token-level K/V sequence;
- no undefined "region" abstraction exists below chunk level;
- a reference may have a variable number of chunks;
- chunk count may vary by reference;
- chunk token length may vary;
- different batch elements may select different references and chunks;
- the resulting detailed-memory length may vary per batch item.

# Terminology

Use these names consistently:

```text
reference
chunk
chunk routing gist
chunk token K/V
reference score
chunk score
selected reference
selected chunk
```

Avoid these ambiguous names in the initial implementation:

```text
region
gist region
detail region
subregion
top_k_regions
```

If legacy identifiers contain `region`, rename them unless they refer to something genuinely distinct outside this design.

# Routing gist semantics

Each chunk has exactly one routing gist per PRA layer.

For reference `r`, chunk `c`, and layer `l`:

```text
gist_key[r, c, l]: [d_model]
```

or in head form if the implementation keeps it projected:

```text
gist_key[r, c, l]: [n_heads, head_dim]
```

The gist is derived from that chunk's contextualized, layer-specific projected keys.

Supported initial gist modes:

```yaml
gist_mode: mean
```

Default:

```text
mean over the chunk's projected token keys
```

Experimental alternatives:

```yaml
gist_mode: last
gist_mode: ref_end
gist_mode: gru
```

The gist mode changes only how the one routing gist for each chunk is computed.

It does not change chunk identity.

It does not create multiple gists per chunk in this patch.

Future work may support multiple gists per chunk, but that requires a separate schema/configuration extension and must not be inferred from this implementation.

# Two independent top-k parameters

Add these parameters:

```yaml
reference_routing:
  top_k_references: 4
  top_k_chunks_per_reference: 2
```

They control different levels of the hierarchy.

## `top_k_references`

This limits how many distinct references survive reference-level routing for each batch item and PRA layer.

Example:

```text
available references:
A, B, C, D, E

top_k_references = 2

selected references:
B, D
```

A reference occupies at most one reference slot regardless of how many chunks it contains.

## `top_k_chunks_per_reference`

This limits how many chunks survive inside each selected reference.

Example:

```text
reference B chunks:
B0, B1, B2, B3

top_k_chunks_per_reference = 2

selected chunks:
B1, B3
```

The selected chunks' token K/V are concatenated for detailed memory attention.

If a selected reference contains fewer chunks than `top_k_chunks_per_reference`, keep all available chunks.

If a reference contains zero chunks for the current layer, it cannot contribute memory at that layer.

## Hard upper bound

For one batch item, the maximum selected chunk count is:

```text
top_k_references * top_k_chunks_per_reference
```

before duplicate handling or recursion-specific constraints.

The maximum token K/V count is still variable because chunks may have different token lengths.

# Search strategies

Add:

```yaml
reference_routing:
  search_strategy: hierarchical
```

Supported values:

```python
SearchStrategy = Literal[
    "hierarchical",
    "reference_first",
    "global_chunks",
]
```

These strategies must share the same cache schema and produce the same result type.

Recommended result type:

```python
@dataclass(frozen=True)
class SelectedChunk:
    entry: PRACacheEntry
    reference_score: float

    chunk_id: str
    chunk_score: float

    layer_id: int
    source_uri: str

    token_start: int
    token_end: int

    metadata: dict
```

Batch/layer result:

```python
list[list[SelectedChunk]]
```

Outer list:

```text
batch item
```

Inner list:

```text
selected chunks for that batch item and layer
```

Selected chunks must retain both:

- the reference-level score;
- the chunk-level score.

## Strategy: `hierarchical`

This is the recommended default.

Algorithm for each batch item and layer:

```text
1. Score all chunk routing gists.
2. Group chunk scores by parent reference.
3. Derive one score per reference.
4. Select top_k_references distinct references.
5. Within each selected reference, select top_k_chunks_per_reference chunks.
6. Materialize only the selected chunks' token K/V.
```

### Reference score aggregation

Add:

```yaml
reference_score_aggregation: max
```

Supported initial options:

```text
max
mean
logsumexp
```

Default:

```text
max
```

For reference `r` with chunk scores:

```text
s[r, 0], s[r, 1], ..., s[r, C_r - 1]
```

default reference score:

```text
reference_score[r] = max_c s[r, c]
```

This score is used only to choose references.

After references are selected, chunk selection still uses the individual chunk scores.

Do not replace selected chunk scores with the aggregate reference score.

Do not materialize the full reference.

Do not select chunks from references that failed the reference top-k.

### Hierarchical pseudocode

```python
chunk_hits_by_reference = score_all_chunk_gists(
    query=query_i,
    layer_id=layer_id,
)

reference_scores = {}

for reference_uri, chunk_hits in chunk_hits_by_reference.items():
    reference_scores[reference_uri] = aggregate_reference_score(
        [hit.chunk_score for hit in chunk_hits],
        mode=reference_score_aggregation,
    )

selected_reference_uris = top_k_distinct(
    reference_scores,
    k=top_k_references,
)

selected_chunks = []

for reference_uri in selected_reference_uris:
    chunk_hits = chunk_hits_by_reference[reference_uri]

    best_chunks = top_k(
        chunk_hits,
        key=lambda hit: hit.chunk_score,
        k=top_k_chunks_per_reference,
    )

    selected_chunks.extend(best_chunks)
```

## Strategy: `reference_first`

Algorithm:

```text
1. Compute or retrieve a separate reference-level routing representation.
2. Score and select top_k_references.
3. Only then score chunk routing gists inside selected references.
4. Select top_k_chunks_per_reference.
5. Materialize selected chunks' token K/V.
```

Important:

The current corrected design derives routing from chunks.

Therefore `reference_first` requires an explicit reference-level representation.

Supported reference-level representation modes may include:

```text
mean of chunk gist keys
last chunk gist
learned reference aggregator
GRU over chunk gists
```

For this patch:

- implement `reference_first` only if a clear reference-level gist is explicitly configured;
- otherwise raise a clear configuration error;
- do not silently emulate `hierarchical`;
- do not use textual summaries;
- do not average raw token embeddings.

This strategy is useful for systems experiments because it can reduce chunk scoring when references are numerous.

## Strategy: `global_chunks`

Algorithm:

```text
1. Score every chunk gist globally.
2. Rank chunks across all references.
3. Apply reference and per-reference chunk budgets.
4. Materialize selected chunks' token K/V.
```

This strategy must still respect both top-k parameters.

A correct constrained global selection procedure is:

```text
- consider chunks in descending global chunk score;
- skip a chunk if its parent reference already has
  top_k_chunks_per_reference selected chunks;
- admit a new parent reference only if fewer than
  top_k_references distinct references are currently selected;
- continue until no more candidates can be admitted.
```

Do not simply take:

```text
top_k_references * top_k_chunks_per_reference
```

global chunks, because that may violate:

- distinct reference budget;
- per-reference chunk budget.

### Global constrained pseudocode

```python
ranked_chunks = sorted(
    all_chunk_hits,
    key=lambda hit: hit.chunk_score,
    reverse=True,
)

selected_chunks = []
selected_reference_uris = set()
chunk_count_by_reference = defaultdict(int)

for hit in ranked_chunks:
    uri = hit.entry.uri

    if chunk_count_by_reference[uri] >= top_k_chunks_per_reference:
        continue

    if (
        uri not in selected_reference_uris
        and len(selected_reference_uris) >= top_k_references
    ):
        continue

    selected_chunks.append(hit)
    selected_reference_uris.add(uri)
    chunk_count_by_reference[uri] += 1
```

This strategy favors the strongest individual chunks globally.

It may select fewer than the theoretical maximum number of chunks if constraints prevent further admissions.

# Materialization semantics

After search, materialize exactly the selected chunks.

For each selected chunk:

```text
chunk.token_kv.k
chunk.token_kv.v
```

Concatenate these token K/V tensors independently for each batch item and layer:

```python
mem_k_i = torch.cat(
    [selected.chunk.token_kv.k for selected in selected_chunks_i],
    dim=2,
)

mem_v_i = torch.cat(
    [selected.chunk.token_kv.v for selected in selected_chunks_i],
    dim=2,
)
```

Do not materialize:

- the entire parent reference;
- unselected chunks;
- all chunks associated with a selected reference;
- arbitrary subregions within chunks;
- only the gist unless compressed-memory mode is explicitly enabled.

Default:

```yaml
detail_materialization: selected_chunks
```

Compatibility/ablation mode:

```yaml
detail_materialization: full_reference
```

`full_reference` must not be the default once chunk routing is enabled.

# Chunk-level versus reference-level scoring

Maintain two separate concepts:

```text
chunk score
reference score
```

## Chunk score

Similarity between the current layer query and one chunk routing gist.

```text
chunk_score(query, chunk)
```

Used to rank chunks.

## Reference score

Aggregation over a reference's chunk scores.

```text
reference_score(query, reference)
```

Used only where the search strategy requires reference selection.

Do not conflate them.

Example:

```text
Reference A:
  chunk A0 score = 0.91
  chunk A1 score = 0.20
  reference score with max = 0.91

Reference B:
  chunk B0 score = 0.70
  chunk B1 score = 0.69
  reference score with max = 0.70
```

With:

```text
top_k_references = 1
top_k_chunks_per_reference = 2
search_strategy = hierarchical
reference_score_aggregation = max
```

select:

```text
Reference A
```

then materialize:

```text
A0 and A1
```

if both fit the per-reference chunk top-k.

If the desired behavior is to prefer references with several supporting chunks, use `logsumexp` or another explicit aggregation mode.

# Search strategy tests

Add tests for all strategies.

## Hierarchical

Construct:

```text
Reference A chunks:
A0 = 0.95
A1 = 0.10
A2 = 0.05

Reference B chunks:
B0 = 0.80
B1 = 0.79
B2 = 0.78
```

With:

```text
top_k_references = 1
top_k_chunks_per_reference = 2
reference_score_aggregation = max
```

Expected:

```text
selected reference A
selected chunks A0 and A1
```

This verifies that:

- reference selection and chunk selection are distinct stages;
- selected reference does not imply only its winning chunk;
- full reference is not materialized.

## Hierarchical with logsumexp

Use a case where several moderately matching chunks can beat one isolated high chunk.

Verify configurable aggregation.

## Global chunks

Verify constrained selection respects:

```text
top_k_references
top_k_chunks_per_reference
```

even when the highest-ranked global chunks all belong to one reference.

## Reference first

Verify:

- reference-level representation is required;
- textual summaries are not used;
- only selected references have their chunks scored;
- invalid configuration fails clearly.

## Strategy equivalence edge case

With:

```text
one chunk per reference
top_k_chunks_per_reference = 1
```

`hierarchical` and `global_chunks` should often produce equivalent selections under `max` aggregation.

Test the exact configured behavior.

# Top-k boundary tests

Add tests for:

```text
top_k_references = 0
top_k_chunks_per_reference = 0
top_k_references > number of references
top_k_chunks_per_reference > chunks in a reference
empty cache
references with zero chunks
batch items selecting different references
batch items selecting different chunk counts
```

Define:

```text
0 means select none
```

Do not reinterpret zero as unlimited.

Validate non-negative values.

# Interaction with dynamic memory bucketing

The memory length for one batch item is:

```text
M_i =
sum of token lengths of all selected chunks
after chunk selection and deduplication
```

The dynamic memory batching modes operate after the full search strategy has completed.

Correct order:

```text
query
-> chunk gist scoring
-> search strategy
-> top_k_references constraint
-> top_k_chunks_per_reference constraint
-> selected chunk deduplication
-> chunk token K/V materialization
-> compute M_i
-> memory_bucket_count strategy
-> detailed memory attention
```

Do not bucket based on:

- number of references alone;
- number of chunks alone;
- number of gists alone.

Bucket based on actual selected token K/V length.

# Interaction with recursive references

Recursive references remain separate cache entries.

A parent reference may have child references, but the search hierarchy remains:

```text
reference
-> chunks
-> one routing gist per chunk
-> chunk token K/V
```

Do not flatten child chunks into the parent's chunk list automatically.

Do not count child chunks against the parent's `top_k_chunks_per_reference` unless the child is explicitly represented as a parent-owned materialized chunk, which is not the default.

Reference budgets apply to distinct cache-entry URIs selected for the current query.

If recursion causes parent encoding to incorporate child memory, that affects the parent's contextualized chunk K/V and gist representations, but identity and budgets remain explicit.

# Configuration example

Use a complete configuration example:

```yaml
reference_routing:
  search_strategy: hierarchical

  top_k_references: 4
  top_k_chunks_per_reference: 2

  reference_score_aggregation: max

  chunking_mode: fixed
  fixed_chunk_tokens: 64
  fixed_chunk_overlap_tokens: 0

  gist_mode: mean

  detail_materialization: selected_chunks

  memory_bucket_count: 2
  memory_bucket_strategy: optimal_contiguous

  recursive_refs:
    enabled: true
    max_depth: 2
    max_total_references: 16
    max_total_tokens: 2048
    cycle_policy: skip
```

Expected maximum selected chunk count per batch item:

```text
4 references * 2 chunks = 8 chunks
```

Actual detailed memory length remains variable.

# Critical implementation warnings

Codex must not:

- introduce a "region" layer below chunks;
- treat token K/V items as regions;
- call individual token K/V pairs gists;
- create several gists per chunk in the initial patch;
- use one global top-k without enforcing both budgets;
- select top-k references and then materialize all their chunks;
- select top-k chunks and then materialize full references;
- collapse all chunk gists into one reference vector unless the selected search strategy explicitly requires a reference-level representation;
- confuse `top_k_references` with `top_k_chunks_per_reference`;
- confuse chunk count with token K/V count;
- compute memory buckets before selected chunks are known;
- let recursive child chunks silently become parent chunks;
- use textual summaries for either reference or chunk routing;
- silently map an unsupported strategy to another one;
- leave old `top_k_refs` semantics ambiguous.

If the old configuration contains:

```yaml
top_k_refs: N
```

migrate it explicitly to:

```yaml
top_k_references: N
```

and add:

```yaml
top_k_chunks_per_reference: 1
```

for behavior closest to one selected chunk per selected reference.

Do not preserve a misleading alias indefinitely without a deprecation warning.

# Acceptance criteria for corrected hierarchy

- [ ] Cache hierarchy is exactly reference -> chunks -> one routing gist per chunk -> chunk token K/V.
- [ ] No undefined region abstraction exists below chunks.
- [ ] `top_k_references` is implemented and tested.
- [ ] `top_k_chunks_per_reference` is implemented and tested.
- [ ] `search_strategy` supports `hierarchical`, `reference_first`, and `global_chunks`.
- [ ] `hierarchical` is the default.
- [ ] Reference score and chunk score remain distinct.
- [ ] Only selected chunks are materialized by default.
- [ ] Full-reference materialization is an explicit ablation/compatibility mode.
- [ ] Both top-k limits are enforced independently per batch item and layer.
- [ ] Variable selected token K/V lengths flow correctly into dynamic memory bucketing.
- [ ] Search results retain parent reference and chunk provenance.
- [ ] Tests prove that selecting one reference does not automatically materialize all its chunks.
- [ ] Tests prove that selecting one chunk does not materialize the whole reference.
- [ ] Documentation and code contain no ambiguous `region` terminology for this path.



# Retrieval metrics, diagnostics, training, evaluation, and inference

This section is mandatory. Retrieval quality must be measured independently of
language-model quality. A lower perplexity alone is insufficient to understand
PRAttention behavior.

## Separate metric groups

Always report metrics in four categories:

1. Routing quality
2. Memory utilization
3. Language-model quality
4. Systems performance

Never report only loss/perplexity.

---

## Routing quality

Record for every PRA layer:

- reference recall@K
- reference precision@K
- chunk recall@K
- chunk precision@K
- MRR
- nDCG
- hit@1
- hit@K
- mean selected references
- mean selected chunks
- mean selected token K/V
- routing cosine distributions
- routing score histogram
- fraction of queries with zero retrieved chunks
- fraction of retrieved chunks actually attended

If ground truth references are unavailable, compute weak labels from the
synthetic reference generator and report that clearly.

---

## Memory utilization

For every layer:

- available references
- available chunks
- available token K/V
- selected references
- selected chunks
- selected token K/V
- selected/available ratio
- average chunk size
- average selected memory length
- maximum selected memory length
- bucket count
- padding fraction
- duplicate chunk merges
- recursive expansion depth
- recursion budget usage

---

## Attention behaviour

For selected chunks report:

- attention entropy
- attention concentration
- percentage of attention on retrieved memory vs local context
- per-head memory usage
- memory gate alpha statistics
- layer-wise retrieved memory contribution

---

## Training metrics

For every epoch and validation pass:

Language model:

- train loss
- validation loss
- perplexity
- token accuracy

Retrieval:

- reference recall@1
- reference recall@K
- chunk recall@1
- chunk recall@K
- retrieval precision
- retrieval F1
- average retrieved chunks
- average selected memory tokens
- routing failures

Learning:

- gradient norm of gist pooler
- gradient norm of routing projections
- gradient norm of memory projection
- cache construction time
- retrieval time
- memory attention time

---

## Inference metrics

For benchmark evaluation record:

Quality:

- perplexity
- QA accuracy
- exact match
- F1

Retrieval:

- retrieved references
- retrieved chunks
- selected chunk IDs
- recursive path
- retrieved memory tokens

Efficiency:

- prompt latency
- retrieval latency
- memory attention latency
- total latency
- peak GPU memory
- cache hit ratio
- cache reuse ratio

---

## Ablation table

Require automatic support for comparing:

- no PRA
- full reference materialization
- selected chunk materialization
- mean gist
- last gist
- ref_end gist
- GRU gist
- fixed chunking
- marker chunking
- bucket count 0
- bucket count 1
- bucket count 2
- bucket count 4
- bucket count 8
- hierarchical search
- global chunk search
- reference-first search

Generate one CSV/JSON row per run with every configuration parameter and every
metric so experiments can be aggregated automatically.

---

## Logging

Every run should emit:

- machine-readable JSON metrics
- CSV summary
- TensorBoard scalars
- optional MLflow metrics

Store retrieval diagnostics separately from language-model metrics.

# Authoritative correction: integrate retrieval metrics into the existing training stack

This section overrides any earlier instruction in this document that could be
read as asking Codex to build a separate metrics, logging, evaluation, trace, or
experiment-output subsystem.

The repository already contains the relevant infrastructure:

```text
src/pra_torch/train.py
    Generic training loop
    batch_step metric dictionaries
    RunningAverages
    train_batch/train/train_epoch logging
    val/test evaluation hooks
    timing metrics
    logger integration
    checkpoints and epoch history

src/pra_torch/pra_train.py
    PRA-specific batch adapter
    PRA evaluation
    selected-reference traces
    retrieval metrics
    cache metrics
    reference ablations
    prediction/trace JSONL output

src/pra_torch/trainer.py
    PRAStandaloneTrainer compatibility/object API

src/pra_torch/metrics.py
    shared metric helpers and RunningAverages

src/pra_torch/logging.py
    configured logger backends
```

Do not create a competing metrics framework.

Do not introduce a second trainer.

Do not bypass `train_model()`.

Do not add ad hoc direct TensorBoard, CSV, MLflow, or JSON writers inside the
attention layer or model forward pass.

All new metrics must flow through the existing batch-step/eval-step/logger
contracts.

## Existing behavior that must be preserved

The generic training loop currently expects:

```python
loss, batch_metrics = batch_step(model, batch, device)
```

and already logs generic metrics such as:

```text
train_loss
perplexity
learning_rate
grad_norm
examples_per_second
tokens_per_second
gpu_memory_allocated
train_batch_duration_seconds
```

using the configured logger at:

```text
train_batch
train
train_epoch
val
val_epoch
test
run
```

The PRA adapter already returns rich batch objects including:

```text
batch
cache
caches
selections
logits
```

The PRA evaluator already computes metrics including:

```text
answer_accuracy
reference_retrieval_accuracy
reference_selection_top1_accuracy
reference_selection_topk_accuracy
reference_selection_mrr
selected_ref_count
expected_anchor_hit
expansion_depth
expanded_ref_count
average_retrieved_tokens
cache_hit_ratio
latency
examples_per_second
tokens_per_second
gpu_memory_allocated
```

and can emit prediction and trace JSONL.

Extend this implementation rather than replacing it.

## Required refactor for the corrected chunk architecture

The existing evaluation assumes reference-only selections shaped approximately
as:

```python
dict[layer_id, list[tuple[uri, score]]]
```

The corrected architecture requires chunk-aware selections.

Introduce a structured trace type such as:

```python
@dataclass(frozen=True)
class SelectedChunk:
    reference_uri: str
    reference_score: float

    chunk_id: str
    chunk_score: float

    layer_id: int
    token_start: int
    token_end: int
    selected_token_count: int

    rank_within_reference: int
    reference_rank: int

    metadata: dict = field(default_factory=dict)
```

Model diagnostics should expose:

```python
selected_chunks_by_layer() -> dict[int, list[list[SelectedChunk]]]
```

where:

```text
dictionary key = PRA layer
outer list = batch item
inner list = selected chunks for that item
```

If a temporary compatibility method remains:

```python
selected_references_by_layer()
```

it must derive deduplicated reference results from the selected chunks and emit
a deprecation warning. It must not remain the primary source of truth.

Update `_pra_batch_step()` to return chunk-aware selections through the existing
`batch_metrics` dictionary.

Do not write metrics from inside `_pra_batch_step()`; return sufficient
statistics and let the existing loop/logger aggregate them.

## Training-time retrieval metric integration

The generic `train_model()` currently creates its own `metrics` dictionary after
`batch_step()` and does not merge arbitrary scalar values from `batch_metrics`
into that dictionary.

Modify the existing generic contract carefully.

Preferred approach:

```python
loss, batch_metrics = batch_step(...)

extra_scalars = batch_metrics.get("metrics", {})
```

Then merge only validated scalar metrics:

```python
metrics = {
    "train_loss": ...,
    "perplexity": ...,
    ...
    **extra_scalars,
}
```

Requirements:

- scalar values only;
- no tensors retaining computation graphs;
- convert tensors using `.detach().item()`;
- reject or skip nested structures;
- reserve generic metric names to prevent accidental overwrite;
- use stable prefixes such as `retrieval_`, `memory_`, `bucket_`, `cache_`;
- preserve existing generic metrics unchanged.

Suggested return shape from `_pra_batch_step()`:

```python
return loss, {
    "tokens": ...,
    "examples": ...,

    "metrics": {
        "retrieval_selected_reference_count": ...,
        "retrieval_selected_chunk_count": ...,
        "memory_selected_token_count": ...,
        "memory_zero_selection_fraction": ...,
        "bucket_padding_fraction": ...,
        "bucket_actual_count": ...,
    },

    "batch": batch,
    "caches": caches,
    "selections": selections,
    "logits": logits,
}
```

Use the existing `RunningAverages` and logger.

Do not create another accumulator.

### Metric weighting

The current generic loop weights batch metrics by number of examples.

Not every metric should be weighted identically.

Add an optional metric-weight contract only if needed:

```python
"metric_weights": {
    "retrieval_selected_chunk_count": examples,
    "memory_selected_token_count": examples,
    "bucket_padding_fraction": allocated_positions,
}
```

A simpler first implementation may use example weighting for all new per-example
means, but it must document the denominator.

Do not average already-global counters as though they were per-example values.

## Evaluation-time retrieval metric integration

Extend `evaluate_pra_model()` directly.

Do not create a separate `evaluate_retrieval_model()` that duplicates the
loader pass.

The current evaluator already:

- loops over the evaluation loader;
- obtains PRA caches and selections;
- derives expected reference URIs;
- computes top-1/top-k/MRR;
- records selected references;
- emits traces;
- measures latency and throughput.

Extend that same pass to compute both reference-level and chunk-level metrics.

## Ground-truth metadata requirements

Reference metrics require:

```text
target_reference_ids or target reference URIs
```

Chunk metrics require explicit target chunk identity or answer-support spans.

Extend dataset metadata to support optional fields:

```python
target_reference_ids: list[int]
target_reference_uris: list[str]

target_chunk_ids: list[str]
target_chunk_spans: list[{
    "uri": str,
    "token_start": int,
    "token_end": int,
}]
```

Do not fabricate chunk ground truth from selected chunks.

When chunk labels are absent:

- omit supervised chunk metrics or return them as unavailable;
- continue reporting unsupervised utilization metrics;
- include a metric/flag such as `chunk_labels_available_fraction`;
- do not report zero recall as though labels existed.

Synthetic datasets should emit exact reference and chunk labels.

For natural QA datasets, supporting-document/chunk labels may be derived only
when the dataset provides sufficient evidence or an explicitly documented
mapping procedure.

## Reference-level retrieval metrics

Retain and refine the existing metrics.

For each query unit/layer compute:

```text
reference_hit_at_1
reference_hit_at_k
reference_recall_at_k
reference_precision_at_k
reference_reciprocal_rank
reference_selected_count
```

Distinguish:

```text
accuracy/hit@k:
    whether at least one expected reference appears

recall@k:
    fraction of all expected references retrieved

precision@k:
    fraction of selected references that are expected
```

The existing `reference_selection_topk_accuracy` is effectively a hit rate, not
general multi-label recall. Preserve it for compatibility but add correctly
named recall and precision metrics.

Recommended names:

```text
reference_hit_at_1
reference_hit_at_k
reference_recall_at_k
reference_precision_at_k
reference_mrr
```

Legacy names may remain as aliases for one release with documentation.

## Chunk-level retrieval metrics

For each layer/query unit compute:

```text
chunk_hit_at_1
chunk_hit_at_k
chunk_recall_at_k
chunk_precision_at_k
chunk_mrr
selected_chunk_count
```

A selected chunk is correct when:

1. its `chunk_id` is explicitly listed as a target; or
2. its source URI and token span overlaps a target support span according to an
   explicitly configured overlap rule.

Add:

```yaml
chunk_match_mode: exact_id
```

Supported future/optional values:

```text
exact_id
any_overlap
iou_threshold
contains_answer_span
```

For IoU threshold mode:

```yaml
chunk_iou_threshold: 0.5
```

Do not mix matching rules across runs without recording configuration.

## Layer-specific and aggregate metrics

The current evaluator iterates `layer_selections.values()` and aggregates all
selection units.

Preserve global aggregates, but also emit per-layer metrics using stable names:

```text
layer_0/reference_hit_at_k
layer_0/chunk_hit_at_k
layer_0/selected_chunk_count
layer_0/selected_memory_tokens

layer_1/...
```

If the logger cannot safely support `/`, use a documented delimiter such as:

```text
layer_0_reference_hit_at_k
```

Also emit model-level macro averages.

Do not merge layers before computing diagnostics; progressive retrieval by layer
is a central scientific question.

## Memory and materialization metrics

The existing `average_retrieved_tokens` counts tokens in all cache entries,
which is not the same as selected/materialized token K/V.

Retain a cache-size metric, but rename or clarify it:

```text
cache_reference_token_count
```

Add metrics based on actual selected chunks:

```text
available_reference_count
available_chunk_count
available_reference_token_count

selected_reference_count
selected_chunk_count
selected_memory_token_count

selected_reference_fraction
selected_chunk_fraction
selected_memory_token_fraction
```

Definitions:

```text
available_reference_token_count:
    source/cache tokens available to route over

selected_memory_token_count:
    token K/V positions actually materialized for detailed memory attention

allocated_memory_position_count:
    rectangular/bucketed positions allocated, including padding

padding_position_count:
    allocated minus valid selected memory positions
```

These must not be conflated.

## Dynamic bucketing metrics

Return bucket statistics from the memory-attention implementation as detached
diagnostic data, for example:

```python
@dataclass(frozen=True)
class MemoryBatchingStats:
    selected_lengths: tuple[int, ...]
    requested_bucket_count: int
    actual_bucket_count: int
    bucket_max_lengths: tuple[int, ...]
    valid_positions: int
    allocated_positions: int
    padding_positions: int
    padding_fraction: float
```

Expose these diagnostics after forward without retaining autograd graphs.

Aggregate through `_pra_batch_step()["metrics"]` and `evaluate_pra_model()`.

Required metrics:

```text
memory_valid_positions
memory_allocated_positions
memory_padding_positions
memory_padding_fraction
memory_actual_bucket_count
memory_max_selected_length
memory_mean_selected_length
memory_zero_selection_fraction
```

For performance measurements, also collect carefully timed:

```text
cache_build_duration_seconds
routing_duration_seconds
materialization_duration_seconds
memory_attention_duration_seconds
bucket_planning_duration_seconds
```

Avoid adding `torch.cuda.synchronize()` inside every model layer by default.

Provide a profiling flag:

```yaml
collect_detailed_timing: false
```

When false, collect cheap counters only.

When true, synchronize only at well-defined profiling boundaries and document
the overhead.

## Attention diagnostics

Do not log full attention tensors every batch.

Expose optional reduced statistics:

```text
memory_attention_entropy
memory_attention_max_weight
memory_attention_effective_token_count
memory_output_norm
local_output_norm
memory_to_local_output_norm_ratio
```

Per-head diagnostics should be behind a flag:

```yaml
collect_per_head_metrics: false
```

If enabled, use layer/head-prefixed scalar names.

Do not store full matrices in `RunningAverages`.

Full attention samples may be written only to bounded trace artifacts for a
small configured number of examples.

## Trace schema update

Extend the existing JSONL trace written by `evaluate_pra_model()`.

Current trace fields such as:

```text
available_references
cached_references
selected_references_by_layer
expanded_anchors
cache_hits
retrieved_token_counts
```

should be retained or migrated clearly.

Add:

```json
{
  "selected_chunks_by_layer": {
    "0": [
      {
        "reference_uri": "doc://...",
        "reference_score": 0.91,
        "chunk_id": "doc://...#chunk=2",
        "chunk_score": 0.88,
        "token_start": 128,
        "token_end": 192,
        "selected_token_count": 64,
        "reference_rank": 1,
        "rank_within_reference": 1
      }
    ]
  },
  "memory_lengths_by_layer": {
    "0": 128
  },
  "bucket_stats_by_layer": {
    "0": {
      "actual_bucket_count": 2,
      "valid_positions": 420,
      "allocated_positions": 480,
      "padding_fraction": 0.125
    }
  },
  "recursive_paths": [],
  "routing_configuration": {}
}
```

Keep trace emission optional.

Do not add large tensors to JSONL.

## Existing ablation framework

`pra_train.py` already provides `evaluate_reference_ablation()` with conditions:

```text
valid
disabled
shuffled
irrelevant
empty
oracle
```

Extend this framework instead of building a second ablation runner.

Add chunk-aware conditions where meaningful:

```text
full_reference
selected_chunks
oracle_chunks
shuffled_chunks
irrelevant_chunks
gist_only
```

Keep the existing reference conditions functional.

Record the exact ablation configuration in returned metrics and run logs.

Do not materialize all loader batches in memory unless required; the existing
ablation implementation currently converts the loader to a list. Codex should
evaluate whether this remains acceptable and refactor carefully if datasets grow.

## Logger and artifact integration

Use `build_logger()` and the existing logger abstraction.

Before adding claims about supported outputs, inspect `logging.py`.

If CSV, TensorBoard, JSON, or another backend already exists, extend it through
the logger interface.

If a backend does not exist, do not add direct one-off writes from PRA code.
Either:

1. extend the generic logger abstraction cleanly; or
2. leave it as a documented future backend.

Prediction and retrieval traces may continue using the existing explicit JSONL
artifact paths because those are per-example artifacts, not scalar metric logs.

## `trainer.py` compatibility

`PRAStandaloneTrainer` delegates to the canonical functions.

Keep it thin.

Do not add duplicate metric calculations to the class.

Its `validate()` and `test()` methods should receive the expanded dictionaries
from `evaluate_pra_model()` automatically.

Its `train()` method should continue returning the canonical result.

## Tests

Add tests that prove integration with the existing stack.

### Generic metric passthrough

A custom batch step returns:

```python
"metrics": {
    "retrieval_selected_chunk_count": 2.0,
}
```

Verify `train_model()` logs and epoch-aggregates it without overwriting generic
metrics.

### Existing metric compatibility

Verify existing keys remain available:

```text
reference_selection_top1_accuracy
reference_selection_topk_accuracy
reference_selection_mrr
selected_ref_count
```

unless deliberately deprecated with aliases.

### Chunk metric correctness

Use deterministic references/chunks and labels to verify exact hit, recall,
precision, and MRR values.

### Missing chunk labels

Verify chunk supervised metrics are omitted/marked unavailable rather than
incorrectly reported as zero.

### Per-layer aggregation

Verify layer metrics do not leak across layers.

### Materialized versus cached tokens

Construct a cache with many tokens but select one short chunk.

Verify:

```text
cache_reference_token_count != selected_memory_token_count
```

### Bucket metric integration

Verify mode 0, 1, and 2+ return batching statistics through the normal metric
pipeline and preserve numerical output equivalence.

### Trace compatibility

Verify JSONL traces are serializable, bounded, and include selected chunk
provenance.

### Logger lifecycle

Verify no direct writer is opened outside the configured logger or explicit
prediction/trace artifact writers.

## Acceptance criteria

- [ ] New metrics extend `train.py`/`pra_train.py`; no parallel trainer exists.
- [ ] `_pra_batch_step()` supplies retrieval/memory scalars through the existing
      batch metrics contract.
- [ ] `train_model()` logs validated extra scalar metrics through the configured
      logger.
- [ ] `evaluate_pra_model()` computes reference, chunk, memory, bucket, and
      systems metrics in one evaluation pass.
- [ ] Existing reference metrics and ablations remain functional.
- [ ] Cached-token counts and selected/materialized-token counts are distinct.
- [ ] Layer-specific metrics are emitted.
- [ ] Missing labels do not generate misleading supervised metrics.
- [ ] JSONL trace output is extended rather than replaced.
- [ ] `PRAStandaloneTrainer` remains a thin compatibility shell.
- [ ] No attention/model module writes files or calls logger backends directly.
