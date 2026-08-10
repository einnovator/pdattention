# AGENTS.md — Multi-Gist PRAttention

## Mission

Extend PRAttention so both **chunks** and **references/URIs** can own multiple routing gists.

Current behavior is effectively:

```text
reference
  -> one reference-level routing vector (in reference-first mode)

chunk
  -> one routing gist
  -> detailed token K/V
```

Target behavior:

```text
reference / URI
  -> G_ref reference-level gists
       |
       v
  selected references
       |
       v
chunk
  -> G_chunk chunk-level gists
       |
       v
  selected chunks
       |
       v
detailed token K/V
```

Default behavior must remain backward-compatible:

```python
gists_per_chunk = 1
reference_gists_per_reference = 1
```

Existing single-gist modes must continue to behave exactly as before unless explicitly configured otherwise.

The new design must support multiple-gist strategies including:

- `kmeans`
- `som`
- `prototype`
- `hybrid`

Each strategy implementation MUST live in its own source file for maintainability and experimentation.

Do not implement all strategies in one giant `gist.py`.

---

# 1. Primary architectural goals

The implementation must make the following distinctions explicit:

```text
gist construction
!=
routing
!=
detail materialization
```

## Gist construction

How a chunk or reference is compressed into one or more routing vectors.

## Routing

How a query compares against the gist set and how multiple gist scores are reduced to one chunk/reference score.

## Detail materialization

What detailed K/V is exposed after selection.

These mechanisms must remain independently configurable and testable.

---

# 2. Core tensor invariant

All routing-gist collections MUST use the same shape convention:

```text
[G, D]
```

where:

- `G` = number of gists actually produced,
- `D` = `d_model`.

Never use mixed conventions such as `[D]` for one gist and `[G,D]` for many gists.

Single-gist modes MUST return `[1,D]`.

This invariant applies at both levels:

```text
chunk routing gists:      [G_chunk, D]
reference routing gists:  [G_ref, D]
```

If value gists are stored:

```text
chunk gist K: [G_chunk, D]
chunk gist V: [G_chunk, D]

reference gist K: [G_ref, D]
reference gist V: [G_ref, D]
```

---

# 3. Configuration changes

Inspect and extend:

```text
src/pra_torch/config.py
```

Add clear top-level controls close to the existing gist-related configuration.

Recommended fields:

```python
gist_mode: str = "mean"
gists_per_chunk: int = 1

reference_level_gist_mode: str = "mean"
reference_gists_per_reference: int = 1

gist_score_aggregation: str = "max"
reference_gist_score_aggregation: str = "max"
```

Initial supported score aggregation:

```text
max
```

Optionally support later:

```text
mean
logsumexp
topk_mean
```

If adding these now is low effort, validate them explicitly. Otherwise keep the public config field but implement only `max` and reject unsupported values.

The default configuration MUST reproduce the old one-gist routing behavior.

---

# 4. Existing single-gist modes

Existing modes such as:

```text
mean
last
ref_end
gru
```

must continue to produce exactly one gist:

```text
[1,D]
```

even when `gists_per_chunk > 1`.

Do not manufacture duplicate copies merely to satisfy the requested count.

Record both values in metadata if useful:

```text
requested_gists
actual_gists
```

Do not fail merely because a single-gist strategy received `gists_per_chunk > 1`.

For reference-level single-gist modes, apply the same rule.

---

# 5. Multi-gist strategy code organization

Create a dedicated package.

Recommended layout:

```text
src/pra_torch/gists/
    __init__.py
    base.py
    common.py
    single.py
    kmeans.py
    som.py
    prototype.py
    hybrid.py
```

Alternative names are acceptable if consistent, but each of the following MUST be in its own file:

```text
kmeans.py
som.py
prototype.py
hybrid.py
```

Do not combine them.

## `base.py`

Define shared interfaces/data structures.

Example:

```python
@dataclass
class ComputedGists:
    k: torch.Tensor          # [G,D]
    v: torch.Tensor | None   # [G,D]
    metadata: dict[str, Any]
```

Define an interface or protocol conceptually like:

```python
class GistStrategy(Protocol):
    def compute(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor | None,
        num_gists: int,
        config: PRAConfig,
        context: GistContext,
    ) -> ComputedGists:
        ...
```

The interface must be usable for both chunk-level and reference-level gist construction.

## `common.py`

Put reusable utilities here:

- flatten projected heads `[1,H,T,Dh] -> [T,D]`
- normalization helpers
- deterministic seed helpers
- cluster assignment utilities
- K/V paired aggregation
- empty/small-input handling
- score reduction helpers

## `single.py`

Keep existing simple one-gist strategies here:

- mean
- last
- ref_end
- GRU adapter if appropriate

If GRU needs a registered module, preserve the existing registered-module design in the model.

---

# 6. K-means mode

File:

```text
src/pra_torch/gists/kmeans.py
```

Mode name:

```text
kmeans
```

## Required semantics

Cluster in **key space**, not independently in K and V space.

Given:

```text
keys:   [T,D]
values: [T,D]
```

find cluster assignments using `keys`.

For cluster `j`:

```text
K gist j = centroid of assigned keys
V gist j = aggregate of values belonging to the SAME assigned tokens
```

Formally:

\[
g_j^K = rac{1}{|C_j|}\sum_{t \in C_j} K_t
\]

\[
g_j^V = rac{1}{|C_j|}\sum_{t \in C_j} V_t
\]

Do NOT run separate k-means on K and V.

Keys determine addressing. Values remain associated with the same selected region.

## K-means parameters

Add explicit config fields, for example:

```python
gist_kmeans_max_iters: int = 8
gist_kmeans_init: str = "kmeans++"
gist_kmeans_tol: float = 1e-4
gist_kmeans_normalize: bool = True
gist_kmeans_seed: int = 0
gist_kmeans_empty_cluster_policy: str = "farthest"
```

For reference-level k-means, either reuse the same parameters or allow prefixed overrides such as:

```python
reference_gist_kmeans_max_iters: int | None = None
reference_gist_kmeans_init: str | None = None
```

If an override is `None`, fall back to the chunk-level setting.

## Small-input behavior

If `num_points < requested_gists`, produce at most `actual_gists = num_points` unless a deliberate duplicate-centroid policy is configured.

Default should be no artificial duplication.

---

# 7. SOM mode

File:

```text
src/pra_torch/gists/som.py
```

Mode name:

```text
som
```

The first implementation should be intentionally simple and local to one gist-construction call.

Do not build a large global online SOM system in this task unless explicitly required elsewhere.

## Suggested semantics

Given keys `[N,D]`:

1. choose/sample input vectors,
2. find best matching unit by cosine or Euclidean distance,
3. update winning prototype,
4. optionally update neighboring prototypes,
5. normalize if configured.

The final SOM prototype vectors become routing K gists.

V gists should be computed by assigning each input token/child gist to its best matching SOM prototype and aggregating the corresponding V vectors.

Again, preserve K/V association through shared assignment.

## SOM parameters

Add fields such as:

```python
gist_som_steps: int = 32
gist_som_learning_rate: float = 0.2
gist_som_final_learning_rate: float = 0.05
gist_som_neighborhood_radius: float = 1.0
gist_som_final_neighborhood_radius: float = 0.0
gist_som_distance: str = "cosine"
gist_som_normalize: bool = True
gist_som_init: str = "sample"
gist_som_seed: int = 0
gist_som_topology: str = "line"
```

Keep topology simple initially.

---

# 8. Prototype mode

File:

```text
src/pra_torch/gists/prototype.py
```

Mode name:

```text
prototype
```

This mode should be designed as an extensible strategy for selecting or learning representative vectors rather than assuming one specific interpretation forever.

For the first implementation, use a simple deterministic diversity-based baseline.

Recommended baseline:

```text
farthest-point / diversity-based prototype selection in normalized K space
```

Example:

1. choose the first prototype from mean-nearest or according to config,
2. iteratively choose the point with maximum distance to the current prototype set,
3. assign points to nearest selected prototype,
4. optionally replace selected points with cluster means,
5. compute V prototypes from matching assignments.

## Prototype parameters

Add fields such as:

```python
gist_prototype_method: str = "farthest"
gist_prototype_init: str = "mean_nearest"
gist_prototype_refine: bool = True
gist_prototype_normalize: bool = True
gist_prototype_distance: str = "cosine"
gist_prototype_seed: int = 0
```

Potential future values may include `farthest`, `medoid`, `learned`, and `dictionary`, but only implement explicitly supported modes.

---

# 9. Hybrid mode

File:

```text
src/pra_torch/gists/hybrid.py
```

Mode name:

```text
hybrid
```

Recommended default:

```text
gist 0 = global mean
remaining G-1 = k-means prototypes
```

Alternative configurable global gist:

```text
mean
last
ref_end
```

when valid for the source context.

Example:

```python
gists_per_chunk = 4
gist_mode = "hybrid"
gist_hybrid_global_mode = "mean"
gist_hybrid_local_mode = "kmeans"
```

produces:

```text
[global_mean, prototype_1, prototype_2, prototype_3]
```

## Hybrid parameters

Add fields such as:

```python
gist_hybrid_global_mode: str = "mean"
gist_hybrid_local_mode: str = "kmeans"
gist_hybrid_global_count: int = 1
gist_hybrid_deduplicate: bool = True
gist_hybrid_min_cosine_separation: float = 0.0
```

The local strategy should delegate to the appropriate implementation rather than duplicate k-means/SOM/prototype code.

---

# 10. Chunk-level gist construction

Preserve the current architecture in which reference encoding produces layer-specific projected K/V.

The multi-gist builder must operate on:

```text
layer-specific projected keys
layer-specific projected values
```

not raw token embeddings.

Expected path:

```text
reference chunk
  -> same model stack
  -> layer-specific K/V
  -> flatten K/V to [T,D]
  -> gist strategy
  -> [G_chunk,D]
```

Do not introduce one universal external embedding space.

---

# 11. Reference-level gist construction

Reference-first routing must support multiple reference-level gists.

Add:

```python
reference_gists_per_reference: int = 1
```

Reference-level gist construction should operate on the available representations of that URI for the current layer.

Preferred source representation:

```text
union of all chunk routing K gists for that reference/layer
```

If a reference has `C` chunks and `G_chunk` gists/chunk, the reference-level strategy receives approximately `[C * G_chunk, D]` and compresses/selects this into `[G_ref, D]`.

For reference-level V gists, use corresponding chunk gist V vectors when needed.

Single-gist reference modes:

```text
mean
last
gru if correctly supported
```

Multi-gist reference modes:

```text
kmeans
som
prototype
hybrid
```

Use the same strategy package for both levels.

---

# 12. Cache reference-level gists

Do not recompute expensive reference-level clustering on every query if it can be computed during cache construction.

Extend the cache representation so each URI can own per-layer reference gists.

Recommended structure:

```python
@dataclass
class ReferenceRoutingGists:
    k: torch.Tensor
    v: torch.Tensor | None
    mode: str
    metadata: dict
```

Then extend `PRACacheEntry` conceptually with:

```python
reference_gists_by_layer:
    dict[int, ReferenceRoutingGists]
```

Desired hierarchy:

```text
PRACacheEntry
  |
  +-- layer 0 reference gists [G_ref,D]
  +-- layer 0 chunks
  |      +-- chunk 0 gists [G_chunk,D]
  |      +-- chunk 1 gists [G_chunk,D]
  |
  +-- layer 1 reference gists [G_ref,D]
  +-- layer 1 chunks
         ...
```

---

# 13. True reference-first routing

Refactor `reference_first` so it becomes genuinely progressive where feasible:

```text
query
  |
  v
score reference-level gists only
  |
  v
top-k references
  |
  v
score chunk gists only inside selected references
  |
  v
top-k chunks
  |
  v
materialize detailed K/V
```

Do not score every chunk in every reference before reference selection when cached reference gists are available.

Keep the other routing strategies available:

```text
hierarchical
global_chunks
```

and update them to understand multi-gist chunks.

---

# 14. Multi-gist scoring

Create one common helper for gist-set scoring.

Conceptually:

```python
def score_gist_set(
    query: torch.Tensor,
    gists: torch.Tensor,
    aggregation: str,
) -> GistScore:
    ...
```

Return at least:

```python
@dataclass
class GistScore:
    aggregate_score: float
    winning_index: int | None
    per_gist_scores: torch.Tensor | None
```

For initial `max` semantics:

\[
s_j = \cos(q, g_j)
\]

\[
s = \max_j s_j
\]

\[
j^* = rg\max_j s_j
\]

Reuse this helper for both chunk and reference scoring.

---

# 15. Selected-chunk diagnostics

Extend `SelectedChunk` or equivalent diagnostics with:

```python
winning_gist_index: int | None
winning_gist_score: float | None
gist_count: int
```

For reference-first routing, track:

```python
winning_reference_gist_index
winning_reference_gist_score
reference_gist_count
```

Do not bury these entirely in unstructured metadata.

---

# 16. `gist_only` materialization semantics

When:

```python
detail_materialization = "gist_only"
```

and a chunk contains multiple gists, materialize the **winning gist K/V pair only** by default.

Example:

```text
chunk gists:
  g0 score .21
  g1 score .87
  g2 score .65

gist_only materialization:
  K = gist_k[1]
  V = gist_v[1]
```

Do not automatically materialize all gists unless a future explicit mode requests that.

---

# 17. Edge cases

Handle explicitly:

- empty input -> `[0,D]`
- requested count larger than available points -> at most `N` gists
- single input point -> `[1,D]`
- empty clusters -> no NaNs
- one-gist modes ignore larger requested counts without duplicating vectors

---

# 18. Determinism

All stochastic gist strategies must support deterministic tests.

Use explicit seeds from config.

Do not silently rely on global random state.

---

# 19. Gradient semantics

Preserve existing `cache_build_mode`.

For `detached`, multi-gist computation must not accidentally retain autograd graphs.

For `trainable_gist`, differentiable strategies should preserve gradients where practical.

Hard k-means assignments, SOM winner selection, and farthest-point prototype selection are not fully differentiable. Do not pretend otherwise.

Document this clearly.

---

# 20. Mode-specific config organization

Avoid making `PRAConfig` unreadable if possible.

If compatible with current serialization, prefer nested dataclasses such as:

```python
@dataclass
class KMeansGistConfig:
    max_iters: int = 8
    init: str = "kmeans++"
    tol: float = 1e-4
    normalize: bool = True
    seed: int = 0
    empty_cluster_policy: str = "farthest"
```

Equivalent config objects may be added for SOM, Prototype, and Hybrid.

If nested dataclasses conflict with current CLI/config loading, use flat prefixed fields instead.

Do not rewrite the configuration framework as part of this task.

---

# 21. Backward compatibility

With:

```python
gists_per_chunk = 1
reference_gists_per_reference = 1
```

and existing modes:

```text
mean
last
ref_end
gru
```

results should remain numerically compatible with current behavior within floating-point tolerance.

Old configs that omit the new fields must load with defaults.

---

# 22. Required tests

Add tests covering:

1. shape invariants for every strategy;
2. K/V pairing for k-means, SOM, and prototype;
3. chunk routing with multiple gists and correct winning gist index;
4. reference-first routing with multiple reference gists;
5. duplicate URI batch isolation;
6. `gist_only` materializing only the winner;
7. reference-level gist caching/reuse;
8. one-gist backward compatibility;
9. edge cases with zero/one/few points;
10. deterministic behavior under fixed seed.

---

# 23. Metrics and diagnostics

Add research-useful metrics where inexpensive:

```text
chunk_gists_requested
chunk_gists_actual_mean
chunk_gists_actual_max

reference_gists_requested
reference_gists_actual_mean
reference_gists_actual_max

winning_chunk_gist_index
winning_reference_gist_index

chunk_best_gist_score
reference_best_gist_score
```

Optional diagnostics:

```text
gist utilization histogram
fraction of gists ever selected
prototype occupancy
cluster-size entropy
mean cosine separation between gists
```

Keep expensive diagnostics optional.

---

# 24. Performance considerations

Approximate routing cost should remain:

```text
reference stage:
O(R * G_ref * D)

chunk stage after reference selection:
O(K_ref * C * G_chunk * D)
```

Do not materialize detailed token K/V merely to score gists.

Use vectorized matrix operations for gist scoring where possible.

---

# 25. Suggested implementation order

## Stage 1 — representation

- change gist tensors to always `[G,D]`;
- add count parameters;
- preserve current modes as `[1,D]`;
- adapt routing to max over gist sets;
- update diagnostics;
- update `gist_only`;
- add regression tests.

## Stage 2 — K-means

- add `gists/kmeans.py`;
- paired K/V assignments;
- parameters;
- tests.

## Stage 3 — reference-level multi-gist

- cache per-layer reference gists;
- add `reference_gists_per_reference`;
- true reference-first two-stage routing;
- tests.

## Stage 4 — Prototype

- add `gists/prototype.py`;
- diversity baseline;
- tests.

## Stage 5 — SOM

- add `gists/som.py`;
- deterministic local SOM;
- tests.

## Stage 6 — Hybrid

- add `gists/hybrid.py`;
- compose existing strategies;
- no duplicated algorithms;
- tests.

## Stage 7 — diagnostics and benchmarks

- gist utilization;
- cache-build cost;
- routing cost;
- single-gist baseline comparison.

---

# 26. Avoid these mistakes

Do NOT:

1. sometimes return `[D]` and sometimes `[G,D]`;
2. cluster K and V independently;
3. flatten multiple reference gists back to one vector before scoring;
4. implement reference-first by scoring every chunk globally first;
5. combine k-means, SOM, prototype, and hybrid implementations in one file;
6. duplicate k-means code inside hybrid mode;
7. make URI strings global identities across batch rows;
8. make multi-gist routing force detailed K/V materialization;
9. duplicate points merely to fill requested gist counts by default;
10. silently present non-differentiable clustering as differentiable;
11. break existing one-gist configs/checkpoints;
12. mix unrelated architecture changes into this refactor.

---

# 27. Acceptance criteria

The task is complete when:

- chunks can store `[G_chunk,D]` routing gists;
- references can store `[G_ref,D]` routing gists;
- default counts remain 1;
- old one-gist modes still work;
- `kmeans.py`, `som.py`, `prototype.py`, and `hybrid.py` exist separately;
- each mode has explicit validated parameters;
- K/V prototype correspondence is preserved;
- chunk routing records the winning gist;
- reference-first records the winning reference gist;
- reference-first selects references before scoring their chunks;
- `gist_only` materializes only the winning chunk gist by default;
- batched row isolation remains correct;
- cached reference gists are reused;
- regression tests demonstrate one-gist backward compatibility;
- multi-gist chunk and reference routing are covered by tests.

---

# 28. Research intent

This extension is not merely a throughput optimization.

The goal is to test whether a chunk or document is better represented for retrieval as a **set of semantic attractors/prototypes** rather than one pooled vector.

Examples:

```text
one mean gist:
  compresses all semantic regions into one point

multiple k-means gists:
  represent several local semantic regions

SOM gists:
  preserve a structured family of representative regions

prototype gists:
  emphasize diverse representative states

hybrid gists:
  combine a global summary with specialized local prototypes
```

At the reference level, the same principle gives progressive routing:

```text
query
  ->
reference-level prototypes
  ->
selected references
  ->
chunk-level prototypes
  ->
selected chunks
  ->
detailed token K/V
```

Keep this research interpretation visible in the code structure and experiment metrics.
