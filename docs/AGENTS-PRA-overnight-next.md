# AGENTS.md — PRA Overnight Next-Step Program

## Mission

Build on the completed exact tensorized router and use the next development cycle to close the most important remaining gaps in the standalone PRA paper and implementation.

Current baseline:

- historical native-KV slicing through 256 addressable units;
- exact tensorized hierarchical routing with `torch.topk`;
- exact prediction/loss parity with the legacy router;
- approximately 16.5–19.4x routing speedup at 256 splits;
- bounded active K/V with fixed `top_k`;
- long-prompt `#__head` support in the code path;
- 170 passing tests.

Do not reopen settled architecture questions unless a correctness failure is found.

Priorities, in order:

1. reconcile Paper 0 with the new routing results;
2. measure and optimize reusable packed routing indexes;
3. empirically validate long-prompt `#__head`;
4. prototype true GPU-memory savings with CPU-resident native K/V and selective transfer;
5. obtain honest cold/warm end-to-end timing and peak-memory measurements.

Do not start pretrained Hugging Face integration unless the earlier goals are complete and stable.

---

## 0. Inspect Current HEAD

Before changes:

1. pull latest `main`;
2. record commit SHA;
3. run full tests;
4. verify existing routing benchmark artifacts;
5. verify Paper 0 and Paper 1 build cleanly.

Preserve current results. Do not overwrite historical JSON/CSV with incompatible schemas.

Benchmark metadata should include git SHA, device, CUDA/PyTorch versions, seed, model tier, dataset, split count, backend, warmup count, and measured example count.

---

## 1. Fix Paper 0 Staleness

Paper 0 still describes the exhaustive 0.5–1.0 s router as the current runtime state. Update it to distinguish:

```text
historical scalar router
vs
current exact tensorized router
vs
future end-to-end serving system
```

Report the new 256-split measurements:

```text
tiny:
~522–542 ms/example -> ~27.7–28.0 ms/example

small:
~908–981 ms/example -> ~53.2–55.5 ms/example

speedup:
~16.47–19.38x
```

State explicitly:

- routing semantics unchanged;
- selected chunks unchanged;
- loss delta zero at recorded precision;
- full native token K/V is still materialized after routing;
- this is not yet an end-to-end serving-speed claim.

If old routing-cost figures remain, label them clearly as pre-tensorization baselines.

Preferred claim:

> Exact PRA routing need not incur candidate-wise Python/CUDA synchronization. Tensorization removes most of that software overhead while preserving exact selection.

---

## 2. Measure Packed-Index Reuse

The current published tensorized benchmark includes packed-index construction. Measure reuse explicitly.

### Add benchmark modes

```text
legacy_scalar
tensorized_cold_index
tensorized_warm_index
```

Definitions:

- `legacy_scalar`: existing legacy path.
- `tensorized_cold_index`: rebuild packed index for each routed example/query.
- `tensorized_warm_index`: construct packed index once and issue repeated queries against the same cache/index.

Warm-index mode approximates autoregressive decoding over stable memory and repeated queries against a persistent/session cache.

### Timing decomposition

Measure separately:

```text
index_build_ms
query_scoring_ms
reference_topk_ms
chunk_topk_ms
selected_hit_serialization_ms
total_routing_ms
```

Use CUDA events where appropriate. Avoid synchronization except at explicit timing boundaries.

### Scaling

Benchmark real caches at:

```text
32, 64, 128, 256
```

If cheap and stable, add routing-only synthetic scaling at:

```text
512, 1024
```

Label synthetic routing-only results clearly.

### Metrics

Record:

```text
candidate chunks
candidate gists
index build time
warm query time
cold query time
legacy time
cold speedup
warm speedup
index bytes
GPU peak allocated
GPU peak reserved
```

### Correctness

Warm-index reuse must select the same reference URIs, chunk IDs, ranks, and equivalent scores as cold tensorized routing. Cache mutation must invalidate the packed index.

Add parity tests.

---

## 3. Profile Remaining CPU Serialization

The tensorized path still converts selected tensors back to CPU/Python after `torch.topk`.

Profile first.

If this is now a meaningful fraction of warm routing time:

- transfer only top-k selected indices/scores;
- avoid moving complete candidate score arrays;
- preserve deterministic tie-breaking;
- preserve exact routing semantics.

Do not attempt a large custom CUDA/C++ rewrite this cycle.

---

## 4. Empirically Validate Long-Prompt `#__head`

The mechanism is documented but needs controlled evidence.

### Core experiment

Use a fixed direct context budget, e.g. 4096 tokens, or a scaled equivalent for the tiny model.

Test total prompt lengths near:

```text
1x, 2x, 4x, 8x direct budget
```

For example:

```text
4K, 8K, 16K, 32K
```

Place a required answer/fact entirely in the displaced prefix.

Compare:

```text
A. direct truncation
B. implicit #__head PRA
C. dense/full historical context control
D. oracle selected head chunk
E. shuffled/wrong head chunk
```

Where dense full context is impossible at the largest size, run it only where feasible and mark missing points clearly.

### Dataset

Start with deterministic answer-code/key-value probes:

```text
prefix:
key_173 = ZEBRA731

large irrelevant context

direct tail:
"What is key_173?"
```

Vary target position across early/middle/late implicit head locations and include chunk-boundary cases.

Required conclusions to test:

1. truncation loses access;
2. `#__head` preserves access through routing;
3. oracle head memory approaches dense historical control;
4. shuffled memory destroys the benefit.

### Small sensitivity sweep

Only vary a few important knobs:

```text
chunk size: 1–3 sensible values
overlap: 0 and one modest overlap
top_k: 2, 4, 8, 16
```

Do not create a huge grid.

Primary plots:

```text
quality / retrieval recall vs total prompt length
active K/V fraction vs total prompt length
routing time vs total prompt length
materialized K/V tokens vs total prompt length
```

### Token conservation

Verify exactly:

```text
implicit_head_tokens + direct_tail_tokens == original_valid_prompt_tokens
```

with no missing token, duplicated split token, or padding leakage.

Include mixed-length batches.

---

## 5. Preserve Historical Context in `#__head`

This is critical.

The split-64 work already showed that routing granularity and encoding context are separate. Do not reintroduce fragmentation by encoding every tiny `#__head` chunk independently.

Determine current behavior.

Preferred semantics:

```text
encode prompt head historically once
-> preserve global/continued positions
-> slice native K/V into routable chunks
```

rather than:

```text
encode each tiny head chunk independently
```

Add a controlled comparison:

```text
independent head-chunk encoding
vs
block/historical encoding
vs
native historical slicing
```

Where feasible, verify:

```text
full dense forward
vs
historical head K/V + continued direct tail
```

to numerical tolerance.

If `#__head` currently resets positions/context, fixing this is higher priority than broad parameter sweeps.

---

## 6. CPU-Resident Native K/V Prototype

The current results show active-attention savings, but the full native K/V cache remains resident.

Implement a minimal experimental tiered-cache mode.

Suggested config:

```python
kv_cache_residency = "gpu"   # current behavior
kv_cache_residency = "cpu"
```

Optional:

```python
kv_cache_pin_memory = True
```

Do not add disk, remote storage, distributed cache, or production eviction in this cycle.

### Desired layout

```text
GPU:
    routing gists/index
    current query
    selected materialized K/V
    local prompt K/V

CPU:
    complete native historical K/V
```

Routing must not require transferring full token K/V.

After selection:

```text
selected chunk IDs
-> gather corresponding CPU K/V
-> transfer selected tensors
-> native-KV attention
```

### Correctness

CPU-resident and GPU-resident modes must produce numerically equivalent outputs within ordinary device-transfer tolerance.

Selected URI/chunk IDs and token counts must match.

### Metrics

Measure:

```text
total cached KV bytes
GPU-resident gist/index bytes
selected KV bytes transferred per layer
selected KV transfer time
materialization time
GPU peak allocated
GPU peak reserved
```

Compare at:

```text
32, 64, 128, 256 splits
```

with `top_k=8`.

The expected qualitative pattern is:

```text
total cached source grows
selected transferred K/V remains approximately bounded
```

Do not hard-code an expected speedup.

### Pinned/nonblocking transfer

Only after basic CPU mode is correct:

- optionally use pinned memory;
- try `non_blocking=True` where correct;
- optionally test a dedicated CUDA stream.

Do not claim overlap unless measured.

---

## 7. End-to-End Runtime Decomposition

Add a benchmark for an entire PRA request.

Measure:

```text
prompt preparation
reference resolution
tokenization
reference/head encoding
native K/V creation
gist/index construction
routing
selected K/V gather
host->device transfer
memory attention
local model forward
total request
```

Also measure a warm-cache repeated-query path:

```text
cache already encoded
index already built
new query/tail only
```

Report separately:

```text
cold request latency
warm request latency
```

Do not fold reference encoding into steady-state query latency.

---

## 8. Peak GPU Memory Benchmark

Record real CUDA peak memory for each scale condition:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Compare where feasible:

```text
dense/full historical control
PRA full GPU cache
PRA CPU-resident cache
```

Record:

```text
peak allocated MiB
peak reserved MiB
cached source KV MiB
active selected KV MiB
```

Report allocated and reserved separately.

---

## 9. Do Not Start ANN Yet Unless Needed

The tensorized exact router has already changed the runtime picture.

First answer:

> How fast is exact routing when index construction is amortized?

Only start FAISS/HNSW/ANN if 512/1024+ exact warm routing is still a material bottleneck relative to model forward, K/V transfer, or memory attention.

If ANN is explored, keep it behind:

```python
routing_index_backend = "exact"
routing_index_backend = "ann"
```

and measure disagreement/recall against exact routing.

Never replace exact results with ANN results.

---

## 10. Tensorize Optional Fallback Paths Only If Cheap

The roadmap says summary-combination and some reference-first paths still use scalar fallbacks.

If straightforward, tensorize them.

Priority:

1. canonical hierarchical content routing;
2. reference-first;
3. summary-combination.

Do not spend most of the cycle on historical/rare modes.

All tensorized paths need legacy parity tests.

---

## 11. Update Paper 1

Only add results that actually complete.

### Routing reuse

Add cold-index vs warm-index routing and index-build cost.

### Long-prompt `#__head`

Add a dedicated experiment subsection with:

- direct-tail budget;
- implicit prefix size;
- historical encoding/slicing behavior;
- routing config;
- active K/V;
- retrieval quality;
- truncation control;
- dense/oracle/shuffle controls.

### CPU-resident K/V

If successful, distinguish explicitly:

```text
routing index memory
stored source K/V
selected transferred K/V
active attention K/V
```

Report actual peak CUDA memory.

### End-to-end timing

If collected, report cold and warm separately.

Do not merge cache building with warm query latency.

---

## 12. Update Paper 0 Again After Experiments

Keep Paper 0 concise.

Only add high-level status for:

- exact tensorized routing;
- `#__head` empirical validation if successful;
- real GPU-capacity reduction if CPU-resident K/V succeeds.

Do not copy all Paper 1 implementation detail into Paper 0.

---

## 13. Reproducibility Artifacts

Generate JSON, CSV, and plot artifacts.

Suggested names:

```text
pra_routing_index_reuse.json
pra_routing_index_reuse.csv
pra_routing_index_reuse.pdf

pra_long_prompt_head.json
pra_long_prompt_head.csv
pra_long_prompt_head.pdf

pra_kv_residency.json
pra_kv_residency.csv
pra_kv_residency.pdf

pra_end_to_end_runtime.json
pra_end_to_end_runtime.csv
pra_end_to_end_runtime.pdf
```

Follow existing shared-results/figures conventions.

---

## 14. Tests

Minimum additions:

### Packed-index reuse
- warm/cold selection parity;
- mutation invalidation;
- repeated-query reuse;
- multi-gist parity;
- batched parity.

### `#__head`
- exact split;
- mixed batches;
- token conservation;
- explicit-reference coexistence;
- cache isolation;
- historical position continuation;
- native-slice head encoding if added;
- long initial generation prompt.

### CPU K/V residency
- GPU/CPU output parity;
- selected-chunk identity parity;
- no transfer of unselected token K/V in an instrumented test;
- correct devices;
- pinned/unpinned correctness;
- empty-memory path;
- unequal selected lengths.

Keep all existing tests passing.

---

## 15. Profiling Discipline

For every optimization:

1. benchmark current path;
2. profile;
3. change one subsystem;
4. rerun parity tests;
5. rerun benchmark;
6. preserve baseline artifact;
7. document included/excluded costs.

Timing-only CUDA synchronization must remain behind instrumentation flags.

---

## 16. Stop Conditions

### Priority A — must complete

- Paper 0 consistency update.
- Packed-index warm reuse benchmark.
- `#__head` controlled functional experiment.
- Full test suite.

### Priority B — strongly desired

- CPU-resident native K/V.
- selected host->GPU transfer metrics.
- peak GPU memory comparison.

### Priority C — only if time remains

- pinned/nonblocking transfer;
- full cold/warm end-to-end timing;
- 512/1024 routing-only scaling;
- optional fallback-path tensorization.

### Explicitly defer

- distributed cache;
- remote object store;
- disk paging;
- production eviction;
- custom fused CUDA attention kernel;
- ANN unless exact warm routing is still limiting;
- Hugging Face pretrained-model integration;
- alternative materialization research.

---

## 17. Interpretation Guardrails

Maintain these distinctions everywhere.

### Gists are routing state

They answer:

> Which chunks should be opened?

They are not canonical transported memory.

### Selected chunks materialize full native K/V

Canonical `selected_chunks` mode still reads complete token-level K/V for selected chunks.

### Active K/V is not total cached K/V

Before off-device residency:

> active attention sparsity != GPU capacity saving.

After CPU residency:

> measure actual GPU peak memory and transfer before claiming capacity savings.

### Routing benchmark is not end-to-end latency

Report separately:

```text
index build
routing
materialization
transfer
attention
encoding
total
```

### Long-prompt support is not automatically historical equivalence

If `#__head` chunks are encoded independently, positional/context fragmentation can return.

Prefer historical encode-once + native-KV slicing.

---

## 18. Questions the Final Report Must Answer

1. How much of the remaining 27–55 ms tensorized routing time is index construction?
2. How fast is exact routing when the gist index is persistent?
3. Does `#__head` recover facts completely outside the direct window?
4. Does quality remain stable as total prompt length grows with a fixed direct window?
5. Does historical head encoding/slicing outperform independent head-chunk encoding?
6. Can complete source K/V live on CPU while only selected chunks occupy GPU memory?
7. How much does actual peak GPU memory fall?
8. What is the selected K/V transfer cost?
9. After routing optimization and cache offload, what dominates warm PRA inference?

These answers are more valuable now than adding more architecture knobs.

---

## 19. Final Deliverables

Before finishing:

1. run full tests;
2. rebuild Paper 0 PDF;
3. rebuild Paper 1 PDF;
4. verify no unresolved LaTeX references/layout problems;
5. inspect plots/tables;
6. update `docs/AGENTS-PRA-Roadmap.md`;
7. commit code, tests, benchmark artifacts, paper sources, and PDFs;
8. push to `main`;
9. report commit SHA;
10. summarize exact results and incomplete Priority-B/C items.

Report negative results too.

If CPU offload increases latency but substantially lowers GPU memory, report both.

If `#__head` fails due to contextual fragmentation, fix or document it rather than hiding the failure.

---

## Strategic Goal

The next milestone is not another feature.

It is to move PRA from:

> sparse native-KV attention with a fast exact router

toward:

> a measured long-context memory system whose routing can be reused, whose displaced prompt history remains recoverable, and whose inactive native K/V no longer needs to occupy scarce GPU memory.

That creates the strongest launch point for the subsequent pretrained/Hugging Face integration paper.
