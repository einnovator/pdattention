# PRAttention Code Walkthrough

Repository: `einnovator/pdattention`

This document follows the current implementation by symbol rather than by fixed line
number. The most important source files are under `src/pra_torch` for PRA behavior and
`src/common` for reusable experiment infrastructure.

## 1. The system in one picture

PRAttention is a decoder-only Transformer whose attention layer has two branches:

1. ordinary causal self-attention over the prompt;
2. routed cross-attention over external reference memory.

The reference path does not concatenate every reference token to the prompt. It first
routes with compact vectors, then exposes detailed K/V only for the selected chunks.

```text
dataset sample and reference metadata
        |
        v
cache_services.py + resolution.py
  resolve URI documents recursively
  build one isolated cache per batch row
        |
        v
model.py
  partition each reference into chunks
  encode chunks through the model stack
  capture layer-specific token K/V
  build routing-gist sets per chunk and URI layer
        |
        v
memory.py
  row -> URI -> layer -> chunk
  route [B,D] queries without crossing row namespaces
        |
        v
attention.py
  compute local causal attention
  route from the final prompt query token
  materialize selected memory for each row
        |
        v
memory_batching.py
  bucket unequal memory lengths
  run masked memory attention in batches
        |
        v
local_output + memory_alpha * memory_output
```

The memory hierarchy is:

```text
batch row
  -> URI / symbolic address
      -> layer-specific chunk gist
          -> selected chunk
              -> layer-specific token K/V
```

This separation is central. A gist answers "which memory looks useful?" Detailed K/V
answers "what information should the prompt attend to?"

## 2. Package boundaries

The implementation now has a deliberate split between generic experiment machinery and
PRA-specific behavior.

### 2.1 `src/common`

`src/common` is model-agnostic and can be reused by unrelated Transformer experiments.

- `config.py` defines the base `TrainConfig` and YAML merge helpers.
- `train.py` owns `TrainingState`, optimization, gradient accumulation, validation,
  checkpoint cadence, CUDA timing, and the generic callback contract.
- `metrics.py` owns running averages, perplexity, gradient norm, CUDA memory, and
  throughput helpers.
- `logging.py` composes console, TensorBoard, Weights & Biases, ClearML, and local metric
  history loggers.
- `plots.py` writes `metrics.json`, a Markdown summary, and optional per-metric plots.
- `checkpointing.py` and `callbacks.py` provide reusable persistence and training
  controls.

Nothing in this package imports PRA cache, routing, resolver, or reference types.

### 2.2 `src/pra_torch`

`src/pra_torch` contains the model and every operation that understands references.

- `config.py` extends the common training configuration with resolver/cache services and
  defines `PRAConfig`.
- `model.py` defines the decoder blocks and reference encoder.
- `attention.py` implements local attention plus routed memory attention.
- `memory.py` defines cache data structures, search, and batched row isolation.
- `memory_batching.py` handles unequal selected-memory lengths efficiently.
- `chunking.py`, `gist.py`, `resolver.py`, and `resolution.py` build reference memory.
- `cache_services.py` joins dataset metadata to the resolver and cache builder.
- `pra_train.py` adapts PRA cache construction, retrieval metrics, and evaluation to the
  generic training loop.
- `trainer.py` is a compatibility object shell over the functional APIs.
- `lm_train.py` is the non-reference language-model adapter used by vanilla baselines.

There is no longer a duplicated `src/pra_torch/train.py`. Generic training lives only in
`src/common/train.py`.

## 3. Configuration

### 3.1 `PRAConfig`

`PRAConfig` in `src/pra_torch/config.py` controls model structure and reference behavior.
Hidden states entering a block have shape `[B,T,D]`, where:

- `B` is prompt batch size;
- `T` is local sequence length;
- `D` is `d_model`.

The main decoder dimensions are `vocab_size`, `d_model`, `n_heads`, `n_layers`, `d_ff`,
`max_seq_len`, and `dropout`. Each head has width `Dh = D / n_heads`.

The block mix is controlled by `n_vanilla_layers` and `n_mixed_layers`. Any remaining
layers are PRA-only blocks.

Named model variants normalize those counts:

| Variant | Layer arrangement |
| --- | --- |
| `td_sa` | all layers use vanilla self-attention |
| `td_pra` | all layers use PRAttention |
| `tdx_pra` | leading layers are vanilla; the final two use PRAttention |
| `custom` | use the explicit vanilla/mixed counts |

`pra_layer_ids` is retained as experiment metadata and is normalized for named variants.
The executable layer stack is determined by the block counts.

### 3.2 Routing budgets

PRA uses two independent budgets:

- `top_k_references`: maximum distinct URIs selected per row and layer;
- `top_k_chunks_per_reference`: maximum chunks retained inside each selected URI.

The old `top_k_refs` name is only a deprecated alias. Keeping the budgets separate is
important: selecting two references with one chunk each is different from selecting two
chunks globally.

`trigger_threshold` removes selected chunks below the cosine-score threshold.
`memory_alpha` controls the final fusion:

\[
y = y_{\mathrm{local}} + \alpha y_{\mathrm{memory}}.
\]

Setting either top-k budget to zero yields no selected memory. Setting
`use_pra_memory=False` on the model forward is the stronger disabled-reference ablation:
it skips search and memory attention entirely.

### 3.3 Search strategies

`PRASimpleMemoryCache.search()` supports three strategies.

**`hierarchical`** scores all chunk gists, aggregates chunk evidence to URI scores,
selects URIs, then selects chunks within each URI.

**`reference_first`** first constructs an explicit URI-level representation, selects
URIs, then ranks chunks only inside them. `reference_level_gist_mode` controls the URI
vector.

**`global_chunks`** ranks chunks globally while still enforcing both the distinct-URI and
per-URI budgets. It is useful as an ablation against explicit hierarchical routing.

`reference_score_aggregation` may be `max`, `mean`, or `logsumexp` when chunk evidence is
collapsed to a URI score.

### 3.4 Gists, chunks, and detailed memory

`gist_mode` controls how projected token keys become a compact routing key:

- `mean`: mean pool token keys;
- `last`: use the final token key;
- `ref_end`: pool at the configured `<REF_END>` marker;
- `gru`: use the registered `GRUGistPooler`.

`max_gists_per_reference` caps independently routable chunks per URI.
`gist_overflow_policy` chooses `truncate`, `merge_tail`, or `error` when that cap is
exceeded.

`chunking_mode` may be `none`, `fixed`, `markers`, or `semantic`. Fixed chunking uses
`fixed_chunk_tokens` and `fixed_chunk_overlap_tokens`. Semantic chunking requires an
explicit plugin implementation.

Long prompts add a preparation boundary before the bounded model forward.
`max_prompt_direct_tokens` selects the recent direct tail; `prompt_overflow_mode` chooses
backward-compatible truncation, a clear error, or an implicit `#__head` PRA reference for
the displaced prefix. The split operates on exact token IDs. `max_prompt_gists` is an
independent cap for prompt-head chunks, and `None` avoids the ordinary explicit-reference
cap so long history is not silently discarded.

After routing, `detail_materialization` decides what enters memory attention:

| Mode | Materialized memory |
| --- | --- |
| `selected_chunks` | full token K/V only for selected chunks |
| `full_reference` | all chunks from every selected URI |
| `gist_only` | winning gist K/V pair from each selected chunk |

Routing and materialization are separate decisions. `full_reference`, for example, still
routes to URIs before expanding their detail.

### 3.5 Summaries and recursive references

When `use_summary=True`, a summary is independently encoded into a routing-side vector.
It is not appended to detailed token memory. `summary_mode` selects how it participates:

- `replace`: route with summary evidence in place of the ordinary chunk gist;
- `hybrid`: combine summary and ordinary candidate scores;
- `augment`: add normalized summary evidence.

Recursive references are resolved child-first. The relevant limits cover depth, total
references, total tokens, and children per document. Cycle and missing-reference policies
make failure behavior explicit. A parent can be encoded while attending to completed
children, but it cannot attend to a half-built cache entry.

### 3.6 `TrainConfig`

`src/common/config.py::TrainConfig` owns generic loop, device, loader, logging, and
artifact settings. `src/pra_torch/config.py::TrainConfig` subclasses it and adds typed
`resolver_config` and `cache_config` fields.

This inheritance direction matters: the generic engine knows only the common base class;
the PRA adapter may use the extra service settings.

## 4. The decoder and reference encoder

### 4.1 Block types

`VanillaTransformerBlock` wraps a batch-first, pre-norm
`nn.TransformerEncoderLayer` with a causal mask. It has no external-memory path.

`PRATransformerBlock` applies:

```text
x = x + PRAttention(LN1(x))
x = x + FF(LN2(x))
```

`PRAttention` already combines local causal attention and routed memory attention. The
block supplies the residual connection around that combined branch.

`PRASATransformerBlock` is the explicit mixed experimental block:

```text
vanilla causal self-attention
-> PRA local-plus-memory attention
-> feed-forward network
```

It intentionally performs a conventional self-attention sublayer and then a PRA sublayer
whose implementation also has a local attention path. It should not be interpreted as a
standard single-self-attention block with a small memory add-on.

### 4.2 `TinyPRAModel`

`TinyPRAModel` owns token embeddings, absolute position embeddings, the block stack, final
layer norm, and language-model head. The same learned components are used for prompt
processing and reference encoding.

For prompt IDs `[B,T]`, `forward()` creates positions `[T]`, forms hidden states `[B,T,D]`,
runs all blocks, and returns logits `[B,T,V]`.

`set_pra_cache()` attaches one cache object to every PRA-capable block. During normal
batched execution this object is a `PRABatchedMemoryCache`, not one flat cache shared by
all examples.

`selected_chunks_by_layer()` exposes the latest selection as:

```text
layer_id -> batch row -> SelectedChunk list
```

`pra_diagnostics_by_layer()` exposes aggregate attention and memory-batching diagnostics
for each PRA layer.

## 5. Building layer-specific reference memory

### 5.1 `_encode_reference_tokens()`

A reference chunk of `M` token IDs is given a singleton batch dimension and encoded as
`[1,M,D]`. Its local absolute positions currently restart at zero for each independently
encoded chunk.

Before each PRA sublayer executes, `project_reference_kv()` captures the representation
that the matching prompt layer will consume:

```text
state entering PRA sublayer L
        |
        +-> layer norm and K/V projection
        |      K_L, V_L: [1,H,M,Dh]
        |
        +-> execute block L
        v
state entering the next block
```

Reference memory is therefore layer-specific. It is not one embedding copied to every
decoder depth.

For a mixed block, reference encoding first applies the same vanilla self-attention
transformation used by the prompt and then projects the state consumed by its PRA
sublayer. This keeps prompt and reference representations aligned.

### 5.2 `encode_reference_to_cache()`

`TinyPRAModel.encode_reference_to_cache()` performs the neural part of cache construction:

1. partition reference text with `partition_reference()`;
2. tokenize and enforce `reference_overflow_policy`;
3. encode each chunk through the shared model stack;
4. capture detailed `LayerKV` at each PRA layer;
5. pool projected K and V into a `ChunkRoutingGist`;
6. create `ReferenceChunkMemory` with provenance and truncation metadata;
7. insert the chunk under the URI and layer in a `PRACacheEntry`.

The routing gist is pooled after layer-specific K/V projection. That puts routing queries
and routing keys in the same learned attention space.

`cache_build_mode="detached"` uses no-grad construction for reusable runtime memory.
`cache_build_mode="trainable_gist"` preserves a graph so gist construction can receive
training signal. Detailed cache tensors still obey the model's configured detach path.

### 5.3 Memory data structures

The main types in `src/pra_torch/memory.py` are:

- `LayerKV`: token K/V, each `[1,H,M,Dh]`;
- `ChunkRoutingGist`: paired compact key/value sets, each `[G_chunk,D]`;
- `ReferenceRoutingGists`: cached URI-level key/value sets, each `[G_ref,D]`;
- `ReferenceChunkMemory`: detail, gist, source offsets, and chunk identity;
- `LayerReferenceMemory`: all chunks for one URI at one layer;
- `PRACacheEntry`: one resolved URI and all of its layer memories;
- `SelectedChunk`: the chosen chunk plus URI/chunk scores and ranks.

The same URI is stored approximately as:

```text
PRACacheEntry(uri)
  layer 0 -> URI gists [G_ref,D] -> chunk gists [G_chunk,D] -> token K/V
  layer 1 -> URI gists [G_ref,D] -> chunk gists [G_chunk,D] -> token K/V
  layer 2 -> URI gists [G_ref,D] -> chunk gists [G_chunk,D] -> token K/V
```

## 6. Resolution and cache services

`src/pra_torch/cache_services.py` is the bridge from collator metadata to runtime memory.
It does not implement neural encoding itself.

`collect_reference_metadata()` converts each collated reference into resolver documents,
summary records, nested document records, and root handles. `create_resolver()` and
`create_cache()` instantiate the configured backends.

`build_cache_from_metadata()` then invokes `RecursiveReferenceCacheBuilder`. The builder
resolves child references first, enforces traversal budgets, records dependency events,
and calls `TinyPRAModel.encode_reference_to_cache()` for each ready document.

The function has an important `attach_to_model` switch:

- `True` is convenient for a standalone cache build or singleton inference;
- `False` returns a completed row cache without replacing the model cache.

PRA training uses `False` while constructing each row. It attaches a combined batched
wrapper only after every row cache is ready.

## 7. Routing and strict batch isolation

### 7.1 `PRASimpleMemoryCache`

`PRASimpleMemoryCache.search(query, layer_id, config)` accepts either `[D]` or `[B,D]` and
returns one selected-chunk list per query row. In an ordinary simple cache, every query
row searches that cache's entries.

The search path scores each normalized query against every gist in a candidate set, reduces
those scores with `max`, `mean`, or `logsumexp`, and records the winning gist index. In
`reference_first`, cached URI gist sets choose URIs before any chunk gists inside those URIs
are scored. The attention layer later applies the materialization threshold and detail policy.

### 7.2 `PRABatchedMemoryCache`

A logical training batch contains independent reference tables. URI strings are not
globally unique: row 0 and row 1 may both use `doc://same` for different content. Flattening
their caches would leak memory across examples.

`PRABatchedMemoryCache` solves this by wrapping completed caches in prompt-row order:

```text
PRABatchedMemoryCache
  row_caches[0] -> references belonging to input_ids[0]
  row_caches[1] -> references belonging to input_ids[1]
  ...
```

Its `search()` validates that query batch size equals row-cache count, then evaluates:

```python
row_caches[i].search(query[i:i + 1], layer_id, config)
```

for each row. The result still has the public shape
`list[list[SelectedChunk]]`.

Flat `entries`, URI-only `get()`, and URI-only `has()` are intentionally rejected because
they would be ambiguous across namespaces. The wrapper is read-only after construction.

The row loop in routing is currently deliberate. The expensive prompt Transformer runs
once for `[B,T]`; vectorizing independent cache searches is a separate optimization.

## 8. `PRAttention.forward()`

### 8.1 Local causal attention

For hidden states `x [B,T,D]`, the layer computes:

```text
q, k, v: [B,H,T,Dh]
scores:  [B,H,T,T]
local:   [B,T,D]
```

The causal mask prevents each prompt position from seeing future prompt positions.

When `use_pra_memory=False`, no cache is attached, or the cache is empty, the function
returns the local branch. This makes disabled-reference evaluation use the same model
weights and prompt tokens while bypassing external memory.

### 8.2 Last-token routing

The current router uses the newest prompt token's projected query:

```text
q[:, :, -1, :]       [B,H,Dh]
flatten heads         [B,D]
```

One routing query is therefore produced per prompt row and PRA layer. The selected memory
is then visible to all `T` query positions in that layer's memory-attention branch.

### 8.3 Materialization

Selections are materialized independently per row. `_materialize()`:

1. optionally expands selected URIs for `full_reference`;
2. applies `trigger_threshold`;
3. removes duplicate `(URI, chunk_id)` selections;
4. chooses gist-only or detailed token K/V;
5. removes duplicated overlap tokens between adjacent chunks of one URI;
6. concatenates retained pieces along memory length.

Each row receives K/V shaped `[1,H,M_i,Dh]`. `M_i` may be zero and may differ across rows.

### 8.4 Unequal-memory batching

`dynamic_memory_attention()` in `src/pra_torch/memory_batching.py` accepts:

```text
q:        [B,H,T,Dh]
memory_k: list of B tensors [1,H,M_i,Dh]
memory_v: list of B tensors [1,H,M_i,Dh]
```

`MemoryBucketPlanner` groups rows by memory length. Each bucket is padded only to its own
maximum `M`, padding is masked before softmax, and outputs are restored to original row
order. This avoids cross-row leakage and reduces padding compared with one global memory
rectangle.

Rows with no retained memory are explicitly zeroed after memory attention. The final
attention result is:

```text
local_out + memory_alpha * memory_out
```

Diagnostics retain selected lengths, valid and padded positions, bucket counts, padding
fractions, duplicate overlap tokens, output norms, and optional detailed timings.

## 9. Generic training in `src/common`

### 9.1 Callback contract

`common.train.train_model()` owns one canonical optimization loop. Model-specific behavior
is injected through:

```python
batch_step(model, batch, device) -> (loss, metadata)
eval_step(model, loader, device, split=...) -> metrics
```

The batch-step metadata supplies example/token counts and optional scalar metrics. The
generic engine rejects attempts to overwrite reserved engine metrics.

### 9.2 `TrainingState`

`TrainingState` holds the model, optimizer, scheduler, device, logger, checkpoint paths,
AMP scaler, early-stopping state, optimizer step, consumed batch step, and epoch history.
An optional `checkpoint_extra` callback adds model-specific reproducibility data without
making checkpointing PRA-aware.

### 9.3 Loop and timing metrics

The common loop handles:

- AdamW and warmup scheduling;
- gradient accumulation and clipping;
- optional CUDA automatic mixed precision;
- periodic and end-of-epoch validation;
- latest/best checkpoints and resume state;
- batch-wise and epoch-wise metric histories;
- final test evaluation.

CUDA is synchronized around measured sections so asynchronous kernels do not understate
execution time. The run records batch duration, train/validation/test duration, wall-clock
time, processed tokens, sequences, optimizer steps, and throughput.

`MetricsHistory` preserves distinct `train_batch`, `train`, `train_epoch`, `val`,
`val_epoch`, `test`, and `run` records. Closing the logger writes JSON, Markdown, and
optional plots. This is why both noisy batch evolution and epoch-wise evolution can be
reported without collapsing history to one point.

## 10. PRA-specific training adapter

### 10.1 `_pra_batch_step()`

`src/pra_torch/pra_train.py::_pra_batch_step()` now performs one prompt forward for the
whole logical batch.

```text
move batch tensors to device
for each metadata row:
    build an isolated cache with attach_to_model=False
wrap row caches in PRABatchedMemoryCache
attach wrapper once
logits = model(input_ids)                  # one [B,T] forward
read selections and diagnostics by layer
compute causal-LM loss and retrieval metrics
```

Reference cache construction is still per row because documents and lengths are unrelated.
The previous implementation also ran `B` singleton prompt forwards; that bottleneck has
been removed.

The adapter records `logical_batch_size`, `prompt_forward_calls`, prompt-forward duration,
per-row average cache-build duration, and total cache-build duration. Tests assert that
`prompt_forward_calls == 1` for batch sizes 1, 2, and 4 and that batched logits match the
legacy isolated-forward result within floating-point tolerance.

### 10.2 Correct diagnostic ownership

Selections are converted from:

```text
layer -> row -> hits
```

to:

```text
row -> layer -> hits
```

for per-example traces and retrieval labels.

Some metrics are row-local, such as selected references, selected chunks, hit@k, MRR,
nDCG, and each row's valid memory positions. Other diagnostics describe the single batched
attention execution, such as aggregate output norms, total padding, and layer timing.

`_retrieval_metrics()` now averages row-local values per row and layer, while adding
batch-level diagnostics only once per layer. This avoids multiplying a batch aggregate by
the number of examples.

### 10.3 Evaluation and ablations

`evaluate_pra_model()` uses the same batched step as training, aggregates LM and retrieval
metrics, and can persist predictions and detailed traces.

`evaluate_reference_ablation()` compares controlled reference conditions with the same
model weights and prompt targets. It directly implements `valid`, `disabled`, `empty`,
`shuffled`, `irrelevant`, and `oracle` URI/cache conditions; `oracle_chunks`,
`shuffled_chunks`, and `irrelevant_chunks` detail interventions; and the
`selected_chunks`, `full_reference`, and `gist_only` materialization variants. The
disabled path calls the model with `use_pra_memory=False` rather than pretending an empty
retrieval result is equivalent to executing the reference path. This controlled evaluator
currently processes examples one at a time because each condition mutates an isolated
cache or K/V payload; ordinary PRA training and evaluation use the batched path above.

`train_pra_model()` creates PRA-aware `batch_step` and `eval_step` closures and delegates
the actual loop to `common.train.train_model()`.

### 10.4 `PRAStandaloneTrainer`

`src/pra_torch/trainer.py::PRAStandaloneTrainer` is intentionally a thin compatibility
shell. It owns no training algorithm.

- construction calls `create_pra_training_state()`;
- `train()` delegates to `train_pra_model()`;
- `validate()` and `test()` delegate to `evaluate_pra_model()`;
- `resume()` and `save()` delegate to common checkpoint functions;
- properties expose the shared functional `TrainingState`.

New code can use the functional API directly. Older object-oriented callers retain the
same convenient surface.

## 11. End-to-end batched execution

For a logical batch:

```text
input_ids: [B,T]
metadata:  B independent reference tables
```

the current path is:

```text
pra_train.py::_pra_batch_step
  |
  | build one cache per metadata row
  v
cache_services.py::build_cache_from_metadata(attach_to_model=False)
  |
  v
RecursiveReferenceCacheBuilder
  resolve children first
  enforce depth/reference/token budgets
  record resolution events
  |
  v
model.py::encode_reference_to_cache
  partition each document
  encode every retained chunk
  capture per-layer K/V [1,H,M,Dh]
  build chunk gists [G_chunk,D]
  cache URI gists [G_ref,D]
  |
  v
PRABatchedMemoryCache([row_cache_0, ..., row_cache_B-1])
  |
  v
model.py::forward(input_ids=[B,T])               ONE prompt forward
  |
  v
each PRA layer
  local causal attention
  route final-token queries [B,D]
  search query[i] only in row_cache[i]
  materialize memory [1,H,M_i,Dh] per row
  bucket and mask unequal M_i values
  fuse local and memory outputs
  |
  v
logits [B,T,V]
  |
  v
ordinary next-token cross-entropy
  plus retrieval, memory, timing, and throughput metrics
```

The key corrected invariant is:

```text
one expensive prompt Transformer forward per logical batch
AND
row i can route only against references belonging to row i
```

## 12. Modes at a glance

| Question | Configuration or API |
| --- | --- |
| No PRA layers | `model_variant="td_sa"` |
| PRA in every layer | `model_variant="td_pra"` |
| PRA in final two layers | `model_variant="tdx_pra"` |
| Disable memory for an ablation | `model(..., use_pra_memory=False)` |
| Route normally, expose selected detail | `detail_materialization="selected_chunks"` |
| Route to URIs, expose all their chunks | `detail_materialization="full_reference"` |
| Attend only to compact gist positions | `detail_materialization="gist_only"` |
| Build reusable cache without graph | `cache_build_mode="detached"` |
| Preserve gist-training graph | `cache_build_mode="trainable_gist"` |
| Use recursive child memory | `recursive_refs_enabled=True` |
| Minimize memory-padding waste | increase `memory_bucket_count` and measure |

## 13. Important invariants

Changes to the implementation should preserve these properties.

1. **Prompt/reference projection alignment.** Reference K/V must use the same layer norm
   and K/V projections as the matching prompt PRA sublayer.
2. **Layer specificity.** A chunk's layer 0 memory must not be reused as its layer 3
   memory.
3. **Row isolation.** Duplicate URI strings in different batch rows must never share
   content or selections.
4. **One logical prompt forward.** `_pra_batch_step()` should call the prompt model once,
   independent of `B`.
5. **Masked variable memory.** Padded K/V positions must receive zero attention weight.
6. **Provenance preservation.** Chunk IDs and token/character offsets must survive routing
   and materialization.
7. **Recursive readiness.** A parent may consume completed child memory, never an entry in
   the middle of construction.
8. **Metric ownership.** Per-row retrieval values and per-batch execution diagnostics must
   not be averaged as though they had the same sampling unit.
9. **Generic/PRA boundary.** `src/common` must remain free of PRA imports.

## 14. Current limitations and next optimizations

### 14.1 Reference encoding is still serialized

The prompt forward is batched, but cache construction loops over rows and documents.
Persistent cache reuse, fingerprinting, or batched reference encoding could reduce this
cost. Those changes should be benchmarked separately from prompt batching.

### 14.2 Routing search loops over row caches

`PRABatchedMemoryCache.search()` keeps isolation obvious by dispatching one query slice to
each row cache. A future packed search could vectorize this while retaining an owner mask.

### 14.3 Routing uses only the final prompt token

One `[D]` query per row and layer can match several chunk or URI gists, but the current
router still reduces those matches to one score per candidate. Multiple routing positions,
learned router tokens, or per-head routing remain experimental extensions.

### 14.4 Chunk positions restart at zero

Each independently encoded chunk uses positions starting at zero. Source offsets remain in
metadata, but absolute document offset is not represented neurally.

### 14.5 The mixed block contains two local-attention stages

This is intentional for the current experimental architecture. Any simplification must be
treated as a model change and compared under controlled seeds, not as a transparent code
cleanup.

## 15. Tests that protect the design

The most relevant suites are:

- `tests/test_pra_batching.py`: one prompt forward, singleton parity, duplicate-URI row
  isolation, unequal/empty memory, overlap removal, summaries, recursion, and gradients;
- `tests/test_pra_routing.py`: search strategies, routing budgets, gists, materialization,
  recursion, and cache behavior;
- `tests/test_shapes.py`: model and attention tensor contracts;
- `tests/test_common.py`: generic training/config/metrics infrastructure;
- `tests/test_trainer.py`: the compatibility shell and functional delegation;
- `tests/test_eval.py`: language-model and PRA evaluation behavior;
- `tests/test_notebook_utils.py`: multi-seed and split-count experiment/report plumbing.

When changing batch routing, the highest-value regression check is not only that loss is
finite. It is that batched logits match isolated execution and that two rows with the same
URI string but different text retrieve only their own content.

## 16. Key takeaways

1. PRAttention is local causal attention plus routed cross-attention over external memory.
2. Compact layer-specific gists perform routing; selected layer-specific token K/V carries
   detailed evidence.
3. Resolver traversal, cache storage, routing, materialization, and memory attention are
   separate stages with explicit contracts.
4. The corrected batch path builds isolated row caches, wraps them in
   `PRABatchedMemoryCache`, and performs one `[B,T]` prompt forward.
5. Unequal selected-memory lengths are bucketed and masked without leaking information
   across rows.
6. Generic optimization and artifact handling live in `src/common`; PRA-specific adapters
   live in `src/pra_torch`.
7. `PRAStandaloneTrainer` is now mostly a compatibility facade over the functional APIs.
8. Metrics preserve batch history, epoch history, retrieval quality, memory sparsity,
   timings, and report artifacts with the correct row-level or batch-level ownership.
