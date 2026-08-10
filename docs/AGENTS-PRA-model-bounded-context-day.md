# AGENTS.md — PRA Model-Bounded Long-Context, Chunking, Materialization Budgeting, and Streaming

## Mission

Extend PRAttention so that it can safely and cleanly operate on top of real pretrained/open models with finite native context limits while still exposing a much larger logical PRA context.

The implementation must enforce one hard architectural invariant:

> **No underlying-model encoding call or attention/materialization step may exceed the configured maximum context supported by the base model/deployment environment.**

This applies to:

- explicit references/documents;
- implicit prompt history (`#__head`);
- routing/materialization;
- streaming generation;
- future persistent PRA memory.

The goal is to separate four scales cleanly:

\[
L_{\text{logical}}
\gg
L_{\text{model,max}}
\ge
L_{\text{encode}}
>
L_{\text{route}}
\]

where:

- `L_logical` = total PRA-addressable context;
- `L_model,max` = configured hard maximum native model context;
- `L_encode` = bounded source block used to produce native K/V;
- `L_route` = smaller addressable routing chunk.

The current PRA architecture already has most of the pieces. This task should generalize them rather than create a parallel long-context subsystem.

---

# 1. Core Architectural Principles

Maintain these distinctions throughout the codebase.

## 1.1 Logical context is not native model context

PRA may expose logical contexts far larger than the base model's native window. The underlying model may support only 8K, 32K, 128K, etc. per native encoding/attention operation.

PRA must bridge that difference through bounded encoding, routing, and selective materialization.

## 1.2 Encoding granularity is not routing granularity

A source may be encoded in large model-safe blocks but exposed to routing as much smaller chunks.

Example:

```text
model_max_context_tokens = 32768
encoding block = 16384
routing chunk = 1024
```

A 128K document can therefore become approximately:

```text
8 encoding blocks
128 routing chunks
```

Do not force each 1K routing chunk to be independently encoded.

The split-64 experiments already demonstrated why that is harmful: routing granularity and contextual encoding granularity must remain independent.

## 1.3 Routing is not selection is not materialization

Maintain three explicit stages:

```text
routing
    -> relevance scores / rankings

budgeted selection
    -> choose feasible chunks under context/memory constraints

materialization
    -> obtain native token-level K/V for selected chunks
```

`top_k` is an upper bound on candidates, not necessarily the exact number of chunks finally materialized.

---

# 2. Add a Hard Model Context Configuration

Introduce a clear hard limit:

```python
model_max_context_tokens: int | None = None
```

Meaning:

> Maximum number of token positions the underlying model/deployment is allowed to process in any one native operation.

For HF/open models this will normally be derived from or default to the model's configured context length, but it must be overridable downward for deployment constraints.

If unset in current tiny-model experiments, preserve backward-compatible behavior using the existing model context configuration.

Validate:

```text
model_max_context_tokens > 0
```

Never silently allow an actual model call above this value.

---

# 3. Generalize Chunking — One Abstraction, Separate Config Instances

Do not build a new custom encoding splitter.

Encoding blocks and routing chunks are both partitioning operations and should reuse a common generic chunking abstraction.

Create/refactor toward a reusable configuration type, conceptually:

```python
@dataclass
class ChunkingConfig:
    mode: str = "fixed"
    chunk_tokens: int | None = None
    overlap_fraction: float = 0.0
    markers: list[str] | None = None

    # Existing/future mode-specific parameters:
    # semantic params
    # paragraph/section params
    # learned params
    # etc.
```

Use separate instances:

```python
encoding_chunking: ChunkingConfig
routing_chunking: ChunkingConfig
```

If nested dataclasses would be too disruptive for current config serialization, equivalent namespaced flat parameters are acceptable for this change, but keep the internal abstraction shared.

Preferred public semantics:

```yaml
encoding_chunking:
  mode: fixed
  chunk_tokens: 16384
  overlap_fraction: 0.25

routing_chunking:
  mode: fixed
  chunk_tokens: 1024
  overlap_fraction: 0.10
```

---

# 4. Existing Chunking Modes Should Work at Both Levels

Where technically meaningful, reuse current chunking modes for both encoding and routing.

Examples:

```text
fixed
markers
semantic
```

and future:

```text
paragraph
section
message
learned
summary-assisted
```

Do not duplicate fixed/marker/semantic algorithms into separate files for encoding and routing.

Instead expose a generic partition function such as:

```python
partition_source(
    source,
    config: ChunkingConfig,
    ...
)
```

and call it from the appropriate stage.

---

# 5. Encoding Chunking

The encoding chunker decides how much source content can be passed to the base model to generate native K/V.

Hard invariant:

```text
actual encoding input tokens <= model_max_context_tokens
```

This applies to:

- explicit references;
- `#__head`;
- future archived/streaming history.

If an encoding chunk configuration asks for more than the model maximum, reject as invalid or normalize explicitly with a clear warning. Prefer validation over silent behavior changes.

---

# 6. Encoding Overlap / Context

Encoding overlap and routing overlap solve different problems.

Encoding overlap gives source tokens near an encoding-block boundary additional contextual history.

However, distinguish:

```text
encoding input span
```

from:

```text
stored native-KV span
```

For continuous historical encoding, overlap/history tokens may be present only to contextualize the new source block.

Do not necessarily store duplicate K/V for overlapping history.

Conceptually:

```text
input:
[previous context][new source span]

store:
                  [new source span]
```

This avoids duplicated cache storage and duplicate routing units.

---

# 7. Encoding Modes for Continuous vs Independent Sources

External documents and prompt history have related but not identical semantics.

Support at least:

```python
encoding_context_mode = "independent"
encoding_context_mode = "overlap"
```

If current code already supports historical/native slicing cleanly, optionally name/use:

```text
historical_window
```

## independent

Each encoding block is encoded without previous source context.

Useful for unrelated docs and strict compatibility.

## overlap / historical_window

Each new source block receives bounded previous context, subject to:

```text
context_tokens + new_block_tokens <= model_max_context_tokens
```

Store native K/V only for the new source span where possible.

Use this preferentially for continuous `#__head` history if compatible with the current model path.

---

# 8. `#__head` Must Obey the Native Model Limit

The previous idealized behavior:

```text
encode whole 28K head once
-> slice native K/V
```

works only when 28K is within the configured model maximum.

For larger heads:

```text
#__head
    -> encoding chunker
    -> bounded model-safe blocks
    -> native K/V
    -> routing chunker
```

Example:

```text
#__head = 96K
model max = 32K
encoding chunk = 16K
routing chunk = 1K
```

The model must never receive the whole 96K at once.

The logical head can still expose approximately 96 routing chunks.

---

# 9. Preserve Logical Source Offsets Separately from Model Encoding Positions

For RoPE and other pretrained models, do not assume arbitrary global positions beyond the supported native context are legal.

Maintain separate metadata:

```text
logical source position
vs
base-model encoding position
```

Example:

```python
logical_start_token = 65536
logical_end_token = 73727
```

while the base model may encode that block using positions valid under its own context policy.

Routing/index metadata should retain logical offsets for provenance and reconstruction.

Do not invent unsupported RoPE positions merely to preserve absolute logical numbering.

Follow actual base-model positional/cache semantics.

---

# 10. Routing Chunking Happens After Contextual Encoding

Preferred pipeline:

```text
logical source
    ↓
encoding partition
    ↓
bounded contextual model encoding
    ↓
native K/V
    ↓
routing partition / native-KV slicing
    ↓
gists
    ↓
packed index
```

A routing chunk should ideally correspond to a slice of already-contextualized native K/V.

Do not independently re-encode every small routing chunk when a larger valid contextual encoding block is available.

---

# 11. Add Materialization Budgeting

After routing, introduce an explicit budget-selection stage before K/V materialization.

The model must satisfy:

```text
direct/local tokens
+ selected PRA memory tokens
+ required current/new token reserve
<= model_max_context_tokens
```

Add config:

```python
max_materialized_memory_tokens: int | None = None
```

This is a deployment/compute cap, distinct from the hard model limit.

Conceptually:

```python
hard_remaining = (
    model_max_context_tokens
    - direct_context_tokens
    - required_current_tokens
    - configured_safety_reserve
)

memory_budget = min(
    hard_remaining,
    max_materialized_memory_tokens
        if configured
        else hard_remaining,
)
```

Never produce a negative budget silently.

---

# 12. Add Optional Generation / Safety Reserve

Introduce a small explicit reserve if useful:

```python
context_safety_reserve_tokens: int = 0
```

or equivalent naming consistent with current generation code.

Purpose:

- prevent exact-limit edge errors;
- reserve required current/query tokens;
- allow deployment-specific slack.

Do not overcomplicate this.

---

# 13. Score-Ordered Greedy Budget Selection — V1

For the first implementation, use the routing score as selection priority.

Conceptually:

```python
ranked_hits = sorted(
    routed_hits,
    key=lambda hit: hit.score,
    reverse=True,
)

selected = []
remaining = memory_budget

for hit in ranked_hits:
    cost = hit.materialized_token_count

    if cost <= remaining:
        selected.append(hit)
        remaining -= cost
    else:
        continue
```

Important:

- do not terminate when a high-score chunk is too large;
- skip it and allow smaller lower-ranked chunks to fill the budget;
- preserve whole chunks initially.

Do not implement score-per-token, knapsack optimization, partial chunks, or learned allocation in this task.

Those are future materialization-policy research.

---

# 14. Hierarchical Routing + Budgeting

Budgeting should happen after existing routing hierarchy has produced candidate hits.

Do not replace:

```text
top_k_references
top_k_chunks_per_reference
```

These remain routing/candidate caps.

New semantics:

```text
all candidates
    ↓
hierarchical exact routing
    ↓
top-k candidate set
    ↓
global materialization budgeter
    ↓
feasible final selected set
```

If necessary, allow router to expose more than the final expected number of chunks so the budgeter has alternatives after oversized chunks are skipped.

Avoid new knobs unless actually needed.

---

# 15. Budget Across All PRA Memory Sources

The materialization budget must cover all selected PRA memory regardless of provenance:

```text
explicit docs
#__head
streaming history
future persistent memory
```

Do not give each source an independent full model budget.

The final materialized K/V union must fit the single native attention budget.

---

# 16. Direct/Recent Context Has Priority

The direct recent context is the high-resolution foreground.

Allocation order should be:

```text
1. required current/query token(s)
2. configured direct/recent context
3. safety/generation reserve
4. remaining capacity available to PRA memory
```

PRA memory fills only the remainder.

Do not evict recent direct tokens merely because memory chunks score highly, except through the explicit streaming rollover policy.

---

# 17. Streaming Generation Must Maintain the Same Invariant

Long generation cannot allow direct history to grow forever.

Add/complete streaming rollover behavior.

When direct window exceeds:

```python
max_prompt_direct_tokens
```

migrate oldest excess direct tokens into PRA prompt-history memory.

Conceptually:

```text
direct history grows
    ↓
exceeds direct budget
    ↓
oldest expired region
    ↓
append/merge into #__head / streaming PRA history
    ↓
bounded direct tail remains
```

Every generation step should satisfy:

```text
bounded direct tail
+ selected PRA memory
+ current generation token(s)
<= model_max_context_tokens
```

Streaming memory must reuse the same encoding chunking, routing chunking, gist/index, and budgeted materialization machinery.

Avoid a separate streaming-only attention mechanism.

---

# 18. Streaming Rollover Granularity

Do not migrate one token at a time if that causes pathological cache/index rebuilding.

Prefer bounded rollover units, such as routing-chunk-sized, encoding-chunk-sized, or a small configurable rollover block.

Choose the smallest change that fits current architecture.

Preserve causal ordering and exact token conservation.

Metrics should expose:

```text
direct_tokens
head_tokens
rollover_events
tokens_migrated
```

---

# 19. Cache / Index Invalidation for Streaming

Appending new prompt-history memory may invalidate or extend packed gist indexes.

Prefer incremental extension if straightforward. Otherwise rebuild correctly.

Correctness before optimization.

Add tests proving after rollover:

- old head chunks remain available;
- newly migrated chunks become routable;
- stale indexes are not used.

---

# 20. Overlapping Routing Chunks and Budget Accounting

If routing chunks overlap, naive length summation is safe but conservative.

V1 may count both full chunk lengths even if some K/V positions overlap. This guarantees the hard model-context bound.

Do not risk overflow by undercounting.

If overlap is common and budget utilization becomes poor, optionally implement second-stage deduplication based on native source spans:

```text
selected chunks
    ↓
union logical/native spans
    ↓
unique K/V materialization
```

Treat this as lower priority unless trivial with current representation.

Document whether accounting is conservative or deduplicated.

---

# 21. Whole-Chunks Only in This Paper

If remaining budget is smaller than the next selected chunk:

```text
skip chunk
```

Do not partially truncate materialized K/V.

Partial-chunk materialization belongs to the later materialization-strategy paper.

Preserve current clear semantics:

> selected routing chunk -> full native token K/V for that chunk.

---

# 22. Metrics for Materialization Budgeting

Add lightweight diagnostics:

```text
model_max_context_tokens
direct_context_tokens
memory_budget_tokens
routing_candidates
routing_topk_candidates
chunks_materialized
chunks_budget_rejected
memory_tokens_requested
memory_tokens_materialized
materialization_budget_utilization
lowest_materialized_score
highest_budget_rejected_score
```

For streaming:

```text
head_tokens
direct_tokens
rollover_events
tokens_migrated
```

Integrate with existing benchmark/result structures.

---

# 23. Tests — Hard Context Safety

Add tests that directly enforce the base-model limit.

Instrument or mock model calls and assert:

```text
every encoding call length <= model_max_context_tokens
```

and:

```text
every attention/materialization operation <= model_max_context_tokens
```

Test:

- short refs;
- very long refs;
- huge `#__head`;
- mixed refs + head;
- streaming generation;
- multiple selected chunks.

A regression that exceeds configured maximum should fail loudly.

---

# 24. Tests — Shared Chunking

Verify same chunking abstraction works independently for:

```text
encoding_chunking
routing_chunking
```

Test at least:

```text
fixed
fixed + overlap
marker mode
```

where already supported.

Check boundaries, logical offsets, overlap amount, no token loss, intended duplicate/context behavior, and max encoding length.

---

# 25. Tests — Encoding vs Routing Granularity

Construct a source where:

```text
encoding chunk = 16 tokens
routing chunk = 4 tokens
```

Verify:

- one encoding call covers multiple routing chunks;
- routing chunks slice contextual native K/V;
- small routing chunks are not independently re-encoded.

This should become a key regression test.

---

# 26. Tests — Budgeted Materialization

Create deterministic candidate chunks, for example:

```text
A score=.97 length=8
B score=.93 length=8
C score=.90 length=12
D score=.81 length=4
```

with constrained budget.

Verify:

- candidates considered by descending score;
- oversized candidates skipped, not terminal;
- smaller later candidates may fill remaining capacity;
- total selected token cost never exceeds budget;
- final full model-context invariant holds.

Add tie-score tests if deterministic tie-breaking matters.

---

# 27. Tests — All Memory Sources Share One Budget

Construct `#__head` chunks plus explicit-reference chunks where each source independently could fill the available budget.

Verify final materialized union fits one global budget.

No source may bypass allocator.

---

# 28. Tests — Streaming Generation

Create a generation scenario with a tiny artificial context limit, e.g.:

```text
model max = 32
direct budget = 8
```

Generate enough tokens to force repeated rollover.

Verify:

- direct history remains bounded;
- old history migrates to PRA memory;
- all tokens remain represented exactly once logically;
- new head chunks become routable;
- no underlying operation exceeds 32 tokens;
- explicit refs continue to work alongside streaming history.

Do not rely on language quality for unit correctness.

---

# 29. Tests — CPU Residency Compatibility

The CPU-resident K/V mode must remain compatible with the budgeter.

Verify:

- budget selection occurs before transfer/materialization;
- only finally selected chunks are transferred;
- budget-rejected chunks are never transferred;
- CPU and GPU-resident modes choose same chunks;
- output parity remains within tolerance.

---

# 30. Benchmark — Materialization Budget Sweep

Add a controlled benchmark varying:

```text
max_materialized_memory_tokens
```

For example:

```text
2K
4K
8K
16K
unbounded-under-native-limit
```

Measure:

```text
RCB / task metric
retrieval recall
chunks selected
materialized tokens
budget utilization
attention time
transfer bytes/time
peak GPU memory
```

Run at a representative long-context scale such as 256 units.

The result should show quality/compute-memory tradeoff without introducing new allocation policies.

---

# 31. Benchmark — Underlying Context Limit

Add at least one test where logical source exceeds configured native model maximum by a large factor.

Example conceptual setup:

```text
logical context: 128K
model max:       16K or 32K
encoding chunk:   8K or 16K
routing chunk:    1K
direct prompt:    4K or 8K
```

Use model-scaled equivalent if tiny research model cannot support absolute sizes.

Verify:

```text
logical context >> native model context
```

while every underlying operation stays bounded.

Report:

```text
logical_context / model_max_context
```

as explicit metric.

---

# 32. Benchmark — `#__head` Beyond Native Context

Extend current `#__head` experiment so implicit head itself is larger than `model_max_context_tokens`.

Compare:

```text
truncate
PRA head with independent encoding chunks
PRA head with overlap/historical encoding
oracle selected chunk
dense control where feasible only
shuffle/wrong-memory control
```

Measure:

```text
answer-code/task metric
routing recall
active K/V
encoding calls
max encoding length
materialized memory
total logical context
```

Do not claim dense equivalence beyond where dense execution is actually possible.

---

# 33. Benchmark — Streaming Long Generation Smoke Test

Add synthetic long-generation smoke test where:

```text
generated length > model_max_context_tokens
```

but direct window remains bounded through rollover.

Objective is mechanism correctness, not generation quality.

Record:

```text
generated tokens
max direct tokens observed
head tokens accumulated
rollover count
max native operation length
routing time
materialized tokens
```

This demonstrates PRA can continue operating beyond native context horizon without violating hard maximum.

---

# 34. Configuration Documentation

Document distinction among:

```text
model_max_context_tokens
max_prompt_direct_tokens
encoding_chunking
routing_chunking
max_materialized_memory_tokens
context/generation reserve
```

Example:

```yaml
model_max_context_tokens: 32768

max_prompt_direct_tokens: 8192

encoding_chunking:
  mode: fixed
  chunk_tokens: 16384
  overlap_fraction: 0.25

routing_chunking:
  mode: fixed
  chunk_tokens: 1024
  overlap_fraction: 0.10

max_materialized_memory_tokens: 16384
context_safety_reserve_tokens: 1
```

Explain clearly that logical context may exceed 32768 but no underlying model operation may.

---

# 35. Backward Compatibility

Existing experiments/configs must continue to run.

If nested config objects would break current YAML/CLI behavior, provide migration aliases.

Old chunking names may map into `routing_chunking` because existing chunking historically referred to routing/addressable chunks.

Document deprecations rather than silently changing semantics.

---

# 36. Avoid Config Explosion

Do not immediately create separate flat copies for:

```text
doc_encoding_*
head_encoding_*
doc_routing_*
head_routing_*
stream_encoding_*
```

Use generic defaults plus per-source override metadata/config only where genuinely necessary.

Preferred architecture:

```text
global encoding_chunking
global routing_chunking

optional source overrides:
    explicit reference
    #__head
    streaming history
```

Most sources should reuse defaults.

---

# 37. Future-Compatible but Out of Scope

Design should leave room for future encoding context modes:

```text
summary
summary + raw overlap
compressed history
learned context
hierarchical context
```

and future materialization policies:

```text
score-per-token
knapsack
partial chunk
subchunk disclosure
gist-only
prototype reconstruction
```

Do not implement them now.

---

# 38. Profiling

After correctness, measure overhead added by materialization budgeter.

It should be tiny relative to routing.

If ranking already exists on GPU, avoid reconstructing full Python rankings unnecessarily.

Preserve deterministic exact semantics before optimizing.

Do not undo recent tensorized routing speedups.

---

# 39. Paper 1 Update

Update Paper 1 after implementation and experiments.

Add/revise architecture sections to introduce the four-scale model:

\[
L_{\text{logical}}
\gg
L_{\text{model,max}}
\ge
L_{\text{encode}}
>
L_{\text{route}}
\]

Explain:

1. logical context;
2. model-native context;
3. encoding blocks;
4. routing chunks;
5. budgeted materialization;
6. direct-window priority;
7. streaming rollover.

Clarify PRA does **not** require the base model to natively forward the entire logical context.

Add implementation pseudocode with stable paper line numbers for encoding partition, native-KV slicing, routing partition, budget allocator, and stream rollover.

Add source file/line references where current reproducibility style uses them.

---

# 40. Paper 1 Results Update

Only include completed measurements.

Desired new tables/figures:

```text
logical context vs model native limit
#__head beyond native context
materialization budget sweep
streaming rollover smoke test
```

If CPU-resident K/V remains enabled, show how budgeted selection limits actual host->GPU transfer.

Maintain strict distinction:

```text
logical context
cached K/V
GPU-resident K/V
selected/materialized K/V
direct K/V
```

---

# 41. Fix Paper 0 After Code/Paper 1 Work

Paper 0 currently needs a consistency pass with latest implementation/results.

Do this after new mechanism stabilizes so it only needs one update.

Update:

- abstract;
- current empirical status;
- scalability discussion;
- routing/runtime section;
- long-context discussion;
- memory section;
- conclusion.

Paper 0 should state high-level architecture and latest evidence, not duplicate all Paper 1 engineering detail.

It should no longer present the old scalar 0.5–1.0 s router as the current implementation.

Mention, if completed and measured:

- exact tensorized routing;
- warm packed-index reuse;
- `#__head`;
- CPU-resident K/V;
- hard native-model context budgeting;
- bounded encoding;
- budgeted materialization;
- streaming support.

Be cautious with any capability not fully benchmarked.

---

# 42. Paper 0 Conceptual Addition

Add the key architectural statement:

> PRA separates the size of the logical addressable context from the maximum context processed by the underlying model in any one operation.

Use a concise formal statement:

\[
L_{\text{logical}}
\gg
L_{\text{model,max}}
\ge
L_{\text{encode}}
>
L_{\text{route}}.
\]

Then explain that direct context plus materialized memory is also bounded:

\[
L_{\text{direct}}
+
L_{\text{materialized}}
+
L_{\text{reserve}}
\le
L_{\text{model,max}}.
\]

This is important to PRA positioning for real open/SOTA models.

---

# 43. Reproducibility Artifacts

Generate JSON/CSV/plots for new experiments.

Suggested artifacts:

```text
pra_context_budget.json
pra_context_budget.csv
pra_context_budget.pdf

pra_head_beyond_native_context.json
pra_head_beyond_native_context.csv
pra_head_beyond_native_context.pdf

pra_materialization_budget.json
pra_materialization_budget.csv
pra_materialization_budget.pdf

pra_streaming_rollover.json
pra_streaming_rollover.csv
pra_streaming_rollover.pdf
```

Preserve git SHA, device, PyTorch/CUDA, config, seeds, and dataset metadata.

---

# 44. Roadmap Update

Update:

```text
docs/AGENTS-PRA-Roadmap.md
```

Document:

- new context-scale abstraction;
- config names;
- shared chunking abstraction;
- encoding/block semantics;
- materialization budgeter;
- streaming rollover;
- benchmarks;
- remaining limitations.

Move future summary/compressed-history encoding and alternative materialization policies into later-paper work.

---

# 45. Day-Work Priority Order

## Priority A — architecture/correctness

1. hard `model_max_context_tokens`;
2. shared encoding/routing chunking abstraction;
3. bounded reference + `#__head` encoding;
4. separate encoding blocks from routing chunks;
5. materialization budgeter;
6. global budget across head + refs;
7. tests for hard context limits.

## Priority B — streaming

8. bounded direct generation window;
9. rollover into PRA head/history;
10. cache/index correctness after rollover;
11. streaming smoke tests.

## Priority C — measurements

12. `#__head` larger than native context experiment;
13. materialization budget sweep;
14. logical/native-context ratio benchmark;
15. streaming > native-context smoke benchmark;
16. CPU residency + budget transfer metrics.

## Priority D — papers/docs

17. Paper 1 architecture/results update;
18. Paper 0 full consistency/fix pass;
19. roadmap;
20. rebuild PDFs and verify.

---

# 46. Stop Conditions

If time becomes constrained:

Must complete:

- hard context invariant;
- encoding/routing separation;
- budgeted materialization;
- correctness tests;
- Paper 0 stale routing correction.

Strongly desired:

- `#__head` beyond native context benchmark;
- streaming rollover;
- Paper 1 formal architecture update.

Can defer:

- overlap deduplication;
- incremental packed-index append optimization;
- advanced encoding modes;
- ANN;
- alternative materialization policies;
- custom CUDA/fused kernels.

---

# 47. Final Validation

Before completion:

1. run full test suite;
2. ensure no regression in existing routing parity;
3. rerun representative current PRA scaling tests;
4. verify no underlying operation exceeds `model_max_context_tokens`;
5. verify CPU-resident cache mode still works;
6. verify mixed `#__head` + explicit refs;
7. verify streaming rollover if implemented;
8. build Paper 0;
9. build Paper 1;
10. verify no unresolved references/layout warnings;
11. inspect generated plots/tables;
12. update roadmap;
13. commit;
14. push;
15. report commit SHA and exact completed/incomplete items.

---

# 48. Interpretation Guardrails

Do not claim:

> PRA makes the base model itself natively support arbitrary context.

Correct claim:

> PRA can expose a logical context much larger than the underlying model's native context while ensuring each base-model encoding and attention operation remains within the configured native limit.

Do not claim:

> top-k always materializes k chunks.

Correct semantics after this change:

> routing identifies high-value candidates; the materialization budgeter selects the highest-scoring feasible whole chunks under the current native-context and deployment budget.

Do not claim:

> overlapping encoding makes separated blocks fully equivalent to one unlimited dense forward.

It reduces boundary/context fragmentation but remains bounded by base-model context.

---

# 49. Strategic Outcome

After this work, PRA should have a clean systems abstraction suitable for real pretrained models:

```text
arbitrarily large logical prompt + references
        ↓
shared source partitioning
        ↓
model-safe contextual encoding blocks
        ↓
native K/V
        ↓
smaller routing chunks + gists
        ↓
exact tensorized routing
        ↓
score-ranked context budgeter
        ↓
only feasible selected K/V materialized/transferred
        ↓
bounded direct + memory attention
        ↓
streaming rollover of expired direct context
```

The architectural invariant is simple:

> **PRA can grow the logical memory without ever requiring the underlying model to process more context than it is configured to support.**

That is the key prerequisite before moving the mechanism onto real SOTA/open pretrained models.
