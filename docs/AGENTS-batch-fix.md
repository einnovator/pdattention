# AGENTS-BATCH-FIX.md

## Mission

Fix PRAttention training so a logical minibatch executes **one batched Transformer prompt forward** instead of one forward per example, while preserving strict isolation between each example's reference namespace.

Current bottleneck:

- `src/pra_torch/pra_train.py:190-211`
- especially `src/pra_torch/pra_train.py:201`

Current code effectively does:

```python
for index, metadata in enumerate(batch["metadata"]):
    cache = build_cache_from_metadata(..., [metadata], ...)
    logits_by_example.append(model(batch["input_ids"][index:index+1]))
logits = torch.cat(logits_by_example, dim=0)
```

This preserves reference isolation but turns a logical batch of size `B` into `B` Transformer forwards.

The target is:

```python
# Build/assemble row-isolated reference memory.
batch_cache = ...

# Attach once.
model.set_pra_cache(batch_cache)

# One prompt forward.
logits = model(batch["input_ids"])  # [B,T,V]
```

Do not weaken isolation to achieve batching.

---

# 1. Existing capabilities to preserve and exploit

Before modifying code, inspect these exact paths.

### Batched prompt model

`src/pra_torch/model.py:224-233`

`TinyPRAModel.forward()` already accepts `[B,T]` input and performs one batched block-stack execution. Do not rewrite it into per-row execution.

### Batched routing contract

`src/pra_torch/memory.py:166-199`

`PRAMemoryCache.search()` already accepts queries `[B,d_model]` and is specified to return one independent ordered `SelectedChunk` list per batch row.

Concrete routing is at:

`src/pra_torch/memory.py:424-454`

Preserve that interface if possible.

### Batched PRA query construction

`src/pra_torch/attention.py:151-179`

`PRAttention.forward()` creates:

```python
routing_query = q[:, :, -1, :].contiguous().view(b, self.d_model)
```

at `src/pra_torch/attention.py:176`.

This is already `[B,D]`.

### Per-row materialization

`src/pra_torch/attention.py:180-191`

The attention layer already loops through `selected_by_batch` and creates independent variable-length K/V memory for each row.

### Variable-length memory batching

`src/pra_torch/attention.py:192-204`

and

`src/pra_torch/memory_batching.py:160-204`

already support one query batch with unequal memory lengths.

The batching fix should make these existing mechanisms useful; do not replace them with a giant common K/V pool that allows cross-row retrieval.

---

# 2. Fundamental invariant

For every logical minibatch row `i`:

```text
query row i may search and materialize only references owned by row i.
```

For `i != j`:

```text
row i MUST NOT retrieve, score, select, or materialize row j memory.
```

This must hold even when:

- two rows use identical URI strings,
- two rows contain different content behind the same URI,
- one row has no references,
- rows have different numbers of references,
- rows have different chunk counts,
- recursive references are enabled,
- summaries are enabled,
- memory lengths differ drastically.

A URI string alone is therefore not sufficient as a global batch identity.

Use row ownership/namespace explicitly.

---

# 3. Preferred architecture

Introduce a batch-aware cache abstraction while keeping the existing per-example cache implementation useful.

Recommended conceptual API:

```python
class BatchedPRAMemoryCache(PRAMemoryCache):
    def __init__(self, row_caches: list[PRAMemoryCache]):
        self.row_caches = row_caches

    def search(
        self,
        query: torch.Tensor,       # [B,D]
        layer_id: int,
        config,
    ) -> list[list[SelectedChunk]]:
        assert query.shape[0] == len(self.row_caches)
        return [
            row_cache.search(query[i:i+1], layer_id, config)[0]
            for i, row_cache in enumerate(self.row_caches)
        ]
```

This simple first implementation is acceptable even if search itself loops over rows. The critical performance goal is to eliminate repeated **Transformer prompt forwards**, not necessarily every Python loop.

However, design the abstraction so a later optimized backend can vectorize routing.

Possible names:

- `PRABatchedMemoryCache`
- `BatchedPRAMemoryCache`
- `BatchIsolatedPRAMemoryCache`

Use the repository's naming style consistently.

---

# 4. Cache visibility API

The current `PRAttention.forward()` fast path checks:

`src/pra_torch/attention.py:172`

```python
if not use_pra_memory or not self.pra_cache.entries:
    return local_out
```

A batched wrapper cannot safely expose one flat URI dictionary if duplicate URI names can occur in separate rows.

Modify the cache interface cleanly.

Preferred options:

### Option A

Add:

```python
def is_empty(self) -> bool:
    ...
```

to `PRAMemoryCache`.

Then change the attention fast path to:

```python
if not use_pra_memory or self.pra_cache.is_empty():
    return local_out
```

Implement it for simple and batched caches.

### Option B

Use `len(self.pra_cache) == 0` if the abstract cache's `__len__` semantics can remain correct.

Prefer an explicit abstraction over relying on a flattened `entries` compatibility dictionary.

Do not introduce ambiguous duplicate-URI flattening.

---

# 5. Cache construction

Current per-example cache building is at:

`src/pra_torch/pra_train.py:190-200`.

It is acceptable in phase 1 to continue building each row's cache independently:

```python
row_caches = []
for metadata in batch["metadata"]:
    row_cache = build_cache_from_metadata(
        model,
        tokenizer,
        [metadata],
        device,
        ...
    )
    row_caches.append(row_cache)
```

But there is a problem:

`build_cache_from_metadata()` currently attaches each newly built cache to the model at:

`src/pra_torch/cache_services.py:88`

```python
model.set_pra_cache(cache)
```

This side effect is convenient for singleton training but awkward for batch assembly.

Refactor it.

Preferred API:

```python
build_cache_from_metadata(
    ...,
    attach_to_model: bool = True,
)
```

or better, separate construction from attachment:

```python
cache = build_cache_from_metadata(...)
model.set_pra_cache(cache)
```

The long-term cleaner design is for `build_cache_from_metadata()` to build and return the cache and for callers to decide when it is attached.

Preserve backward compatibility if other tests/callers rely on automatic attachment.

For the new batched path:

```python
row_caches = [
    build_cache_from_metadata(..., attach_to_model=False)
    ...
]
batch_cache = BatchedPRAMemoryCache(row_caches)
model.set_pra_cache(batch_cache)
```

Then execute one forward.

---

# 6. Important reference-encoding interaction

`build_cache_from_metadata()` invokes `RecursiveReferenceCacheBuilder`, which can call `TinyPRAModel.encode_reference_to_cache()`.

Reference encoding itself uses the model stack:

`src/pra_torch/model.py:234-249`

and can optionally use PRA memory for recursive parent construction.

Therefore do not attach an incomplete global/batched cache while a row's recursive cache is being built unless its semantics are explicitly correct.

Phase-1 safe solution:

1. Build each row cache independently under existing semantics.
2. Do not expose other rows during that build.
3. After all row caches are complete, wrap them in the batch-aware cache.
4. Attach the wrapper exactly once for the prompt forward.

This isolates cache construction and greatly reduces implementation risk.

---

# 7. Refactor `_pra_batch_step()`

Target:

`src/pra_torch/pra_train.py:170-222`.

Replace the current singleton-forward structure.

Desired pseudocode:

```python
batch = move_batch(batch, device)

row_caches = []
cache_build_duration = 0.0

for metadata in batch["metadata"]:
    start = time.perf_counter()
    row_cache = build_cache_from_metadata(
        model,
        tokenizer,
        [metadata],
        device,
        resolver_config=resolver_config,
        cache_config=cache_config,
        attach_to_model=False,
    )
    cache_build_duration += time.perf_counter() - start
    row_caches.append(row_cache)

batch_cache = BatchedPRAMemoryCache(row_caches)
model.set_pra_cache(batch_cache)

# ONE batched model forward.
logits = model(batch["input_ids"])

selections = model.selected_chunks_by_layer()
diagnostics = model.pra_diagnostics_by_layer()

# Convert model's layer -> batch-row structure into whatever
# _retrieval_metrics() expects without losing row alignment.

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    batch["labels"].view(-1),
    ignore_index=0,
)
```

Be careful with the existing `selections` shape.

`model.selected_chunks_by_layer()` returns:

`src/pra_torch/model.py:188-195`

```text
layer_id -> list[batch row] -> SelectedChunk list
```

The current singleton implementation converts it into one dictionary per example.

Update `_retrieval_metrics()` or adapt the data shape once, cleanly.

Avoid repeated transpose/repack logic scattered across functions.

---

# 8. Diagnostics shape

`model.pra_diagnostics_by_layer()` is defined at:

`src/pra_torch/model.py:213-223`.

Some diagnostics are aggregate per PRA layer rather than per row.

The old singleton-forward path naturally generated one diagnostic dictionary per example.

After batched forward, determine which metrics genuinely require per-example values and which are layer/batch aggregates.

Do not fabricate per-row values by copying a batch aggregate.

Refactor metrics types if needed.

Minimum requirements:

- retrieval correctness metrics remain row-level,
- selected chunks/references remain row-level,
- memory lengths remain row-level,
- padding/allocated positions can remain batch/layer aggregate,
- timing can remain batch/layer aggregate.

Document the distinction.

---

# 9. Required tests

Add focused tests before optimizing further.

## 9.1 Forward equivalence

For a deterministic model (`dropout=0`, fixed seed):

```text
old singleton execution
vs
new batched execution
```

must produce numerically equivalent logits within floating-point tolerance when cache semantics are identical.

Test at least:

- `B=1`
- `B=2`
- `B=4`

## 9.2 Strict isolation

Create two batch rows whose reference content intentionally conflicts.

Example:

```text
row 0 URI "doc://same" -> "answer is ALPHA"
row 1 URI "doc://same" -> "answer is BETA"
```

Both rows use the **same URI string**.

Assert row 0 cannot retrieve row 1's chunk and row 1 cannot retrieve row 0's chunk.

This test is mandatory because it catches accidental global URI flattening.

## 9.3 Unequal reference counts

Test one batch containing rows with:

```text
0 references
1 reference
3 references
many chunks
```

Ensure forward succeeds and empty rows receive zero memory contribution.

## 9.4 Unequal memory lengths

Force selected memory lengths such as:

```text
0, 8, 31, 128
```

Verify `dynamic_memory_attention()` receives all rows in one call and restores original row order.

## 9.5 Routing strategies

Run isolation/equivalence tests for:

- `hierarchical`
- `reference_first`
- `global_chunks`

## 9.6 Materialization modes

Test:

- `selected_chunks`
- `full_reference`
- `gist_only`

## 9.7 Chunk overlap

Ensure overlap removal in:

`src/pra_torch/attention.py:136-143`

still behaves identically under batching.

## 9.8 Recursive references

If recursive references are already covered by tests, add a batched isolation case with recursive child URIs.

## 9.9 Summaries

If `use_summary=True`, verify row-local summary keys cannot leak across batch rows.

## 9.10 Backpropagation

Run an actual training step with `B>1`:

- loss finite,
- backward succeeds,
- optimizer step succeeds,
- gradients finite.

Also test `cache_build_mode="trainable_gist"` if currently supported in training.

---

# 10. Performance instrumentation

Add a regression/performance metric that makes the fix observable.

At minimum record:

```text
logical_batch_size
prompt_forward_calls
```

For a normal batch:

```text
prompt_forward_calls == 1
```

Do not count reference-encoding forwards as prompt forwards.

If useful, also record:

```text
cache_build_seconds
prompt_forward_seconds
routing_seconds
memory_attention_seconds
```

Existing detailed PRA timing hooks live at:

`src/pra_torch/attention.py:221-228`.

Do not mix cache-building cost with prompt-forward cost.

---

# 11. Benchmark

Add a lightweight benchmark script or test utility comparing:

```text
legacy singleton prompt forwards
new true batched prompt forward
```

Use at least:

```text
B = 1, 2, 4, 8
```

and report:

```text
examples/s
tokens/s
wall-clock prompt-forward time
peak CUDA memory when CUDA is available
memory padding fraction
```

Do not require CUDA for correctness tests.

The benchmark may skip unavailable hardware.

---

# 12. Preserve generic training separation

Do not move PRA cache logic into `src/pra_torch/train.py`.

`train.py` is intentionally model-agnostic; `TrainingState` states this at:

`src/pra_torch/train.py:21-40`.

Keep the fix primarily in:

- `src/pra_torch/memory.py`
- `src/pra_torch/cache_services.py`
- `src/pra_torch/pra_train.py`
- tests

Touch `attention.py` only for clean cache-interface changes such as `is_empty()`.

Touch `model.py` only if needed for clean selection/diagnostic shape handling.

---

# 13. Avoid these incorrect fixes

Do **not**:

1. Merge all row references into one `PRASimpleMemoryCache` without ownership information.
2. Assume URI strings are globally unique across a minibatch.
3. Pad every reference document into one huge token sequence and run unrestricted cross-row attention.
4. Duplicate the whole model per batch row.
5. Keep the current singleton model-forward loop and claim batching is fixed because logits are concatenated afterward.
6. Remove `memory_batching.py`; it solves a different and still necessary problem.
7. Make retrieval metrics silently compare row `i` selections with row `j` labels.
8. Sacrifice recursive-reference semantics just to simplify the batch wrapper.
9. Break `B=1`.
10. optimize reference encoding and prompt batching simultaneously in one opaque refactor. Make the prompt-forward fix correct first.

---

# 14. Suggested implementation stages

## Stage 1 — abstraction and correctness

- Add batch-aware cache wrapper.
- Remove ambiguous reliance on flattened `.entries` for emptiness.
- Make cache construction optionally non-attaching.
- Refactor `_pra_batch_step()` to one prompt forward.
- Preserve retrieval metric correctness.
- Add isolation/equivalence tests.

## Stage 2 — observability

- Add prompt-forward count.
- Add timing separation.
- Add benchmark.

## Stage 3 — optional routing vectorization

Once correctness is established, consider vectorizing gist scoring across row caches.

Do not block Stage 1 on this.

## Stage 4 — optional batched reference encoding

Reference encoding currently uses singleton chunks by design in:

`src/pra_torch/model.py:234-249`.

Only optimize this after prompt batching is stable.

This is a separate research/engineering task because reference chunks vary in length and recursive construction creates dependencies.

---

# 15. Acceptance criteria

The batch fix is complete when all of the following hold:

- `src/pra_torch/pra_train.py` performs one prompt `model(batch["input_ids"])` call per logical batch.
- Every batch row can only search its own reference namespace.
- Duplicate URI strings across different rows are safe.
- `PRAttention` receives `[B,D]` routing queries and row-local selections.
- `dynamic_memory_attention()` receives all logical batch rows together when PRA memory is active.
- `B=1` behavior remains compatible.
- Existing routing/materialization modes remain functional.
- LM loss and retrieval metrics remain correct.
- deterministic batched logits match singleton-reference execution within tolerance.
- backward/optimizer step works for `B>1`.
- tests explicitly detect cross-row leakage.
- benchmark demonstrates that increasing logical batch size no longer increases prompt Transformer forward-call count linearly.

---

# 16. Documentation requirement

Update the relevant code comments/docstrings to explain the distinction among:

```text
logical training batch
row-local reference namespace
row-local selected memory
bucketed memory tensor execution
```

Document tensor shapes, especially:

```text
input_ids        [B,T]
routing_query    [B,D]
local q          [B,H,T,Dh]
row memory K_i   [1,H,M_i,Dh]
bucket K         [B_bucket,H,M_max,Dh]
logits           [B,T,V]
```

Keep comments concise but pedagogical.

