# AGENTS-PRA-PARAM-SENSITIVITY-SCALE.md

## Mission

Improve native-KV PRA routing and fragmentation robustness through controlled parameter sensitivity experiments, then scale evaluation from 32/64 to 128/256 addressable units without confusing more addressable context with smaller and smaller encoding fragments.

The work should separate four questions:

1. Retrieval breadth: how much does selecting more references/chunks improve routed quality?
2. Routing representation: how much do multi-gist and alternative gist modes improve ranking without increasing transported token K/V?
3. Fragmentation/composition: why does oracle quality degrade at fine partitioning, and can routing granularity increase without shrinking encoding context?
4. Model capacity: after routing and fragmentation are controlled, does a larger/better-trained SA backbone preserve PRA quality at 64, 128 and 256 units?

Do not tune merely for benchmark score. Identify the bottleneck and the quality-cost scaling law.

## Current baseline

Treat current native-KV results as a deliberately weak, untrained-router baseline.

The benchmark currently uses approximately:

```text
memory_transport = native_kv
top_k_references = 1
top_k_chunks_per_reference = 1
trigger_threshold = -inf
detail_materialization = selected_chunks
gist_mode = mean
gists_per_chunk = 1
```

Important observations:

- split 32 oracle RCB is ~0.996 HotpotQA-derived and ~0.999 QASPER-derived;
- split 32 routed RCB is ~0.634 and ~0.602;
- sparse oracle active K/V is about 9%;
- shuffled memory strongly degrades quality;
- split 64 oracle quality itself degrades, so split 64 is not just a routing problem.

Interpretation:

```text
split 32 -> primarily routing gap
split 64 -> routing gap + fragment encoding/composition problem
```

## Chunk overlap and content-aware chunking

Chunking strategy is an additional independent experimental axis.

Do not assume non-overlapping fixed-size chunks are optimal.

Fine fragmentation can damage native K/V quality because tokens near chunk boundaries lose nearby left/right semantic context during independent encoding. A small overlap between consecutive encoding chunks may reduce this boundary effect while preserving PRA's overall routing design.

### Add overlap configuration

Introduce an explicit parameter such as:

```python
chunk_overlap_fraction: float = 0.0
```

with semantics:

0.0 -> no overlap
0.1 -> consecutive chunks overlap by 10% of chunk length
0.25 -> overlap by 25%

Require:

0.0 <= chunk_overlap_fraction < 1.0

If the codebase already uses token counts for chunk overlap, support both forms cleanly, for example:

chunk_overlap_fraction: Optional[float]
chunk_overlap_tokens: Optional[int]

but avoid ambiguous simultaneous use. Define precedence or reject conflicting values.

The default must remain:

chunk_overlap_fraction = 0.0

to preserve existing experiment reproducibility.

Overlap sensitivity sweep

For fragmentation experiments, test at least:

0.0
0.05
0.10
0.20

Optionally test 0.25 if early results indicate continuing benefit.

Do not start with very large overlap because excessive duplication can obscure whether PRA is actually solving fragmentation.

Run overlap sweeps first on:

split 64

then on:

128
256

only after the encoding-scale experiment is healthy.

Hypothesis

Overlap may improve:

oracle RCB
native-all quality
boundary-sensitive retrieval
gist quality
fragment composition

because each chunk retains more neighboring semantic context.

Expected costs:

more K/V storage
more encoding work
more gist/index entries or larger chunks
possible duplicate K/V in final attention
higher transfer volume
possible double-counting of repeated tokens

Therefore report quality and duplication costs together.

Required overlap accounting

Capture:

```
chunk_overlap_fraction
chunk_overlap_tokens
unique_source_tokens
encoded_tokens_including_overlap
stored_kv_tokens_including_overlap
duplication_factor
retrieved_physical_kv_tokens
retrieved_unique_source_tokens
```

Define:
```
duplication_factor =
    encoded_tokens_including_overlap
    /
    unique_source_tokens
```

Do not report only physical active K/V fraction when overlap is enabled.

Also report a unique-context-normalized active fraction where feasible:

```
active_unique_fraction =
    retrieved_unique_source_tokens
    /
    total_unique_accessible_tokens
```

while retaining the physical K/V fraction used for actual systems cost.

Deduplication policy

When two retrieved chunks overlap, avoid accidental double insertion of identical token K/V where practical.

Support an explicit policy such as:

overlap_materialization = "deduplicate"
overlap_materialization = "keep_duplicates"

Use deduplicate as the preferred native-KV evaluation mode if token identity / source offsets make exact deduplication reliable.

Keep keep_duplicates as a diagnostic control because repeated K/V may alter attention probability mass.

Record which policy was used.

Separate encoding overlap from routing units

Overlap belongs primarily to the encoding/contextualization strategy.

Do not require routing units themselves to overlap identically.

Support the distinction:

encoding block
routing region
materialized K/V region

For example:

encode 64-token block with 10% overlap
expose several non-overlapping routing subregions
materialize selected native K/V subregions

This can preserve local context without inflating final active K/V as much as directly routing overlapping chunks.

This should be compared with simple overlapping chunks.

Content-aware chunking as future strategy family

Leave the chunking abstraction open for later strategies beyond fixed token counts.

Plan an extensible interface such as:

chunking_mode = "fixed"
chunking_mode = "paragraph"
chunking_mode = "section"
chunking_mode = "section_hierarchy"
chunking_mode = "semantic"
chunking_mode = "custom"

Do not implement all modes immediately unless already easy.

The architecture should not assume that chunk boundaries are defined only by token count.

Potential future strategies include:

paragraph boundaries
sentence groups
Markdown headings
document sections/subsections
code functions/classes/modules
semantic-change boundaries
hierarchical document structure
Why content-aware chunking matters

PRA routing operates over representations of chunks.

A chunk is likely to produce a better gist and more coherent native K/V when it corresponds to a semantically coherent unit rather than an arbitrary token interval.

Content-aware chunking may therefore improve both:

representation quality
routing quality

while also reducing the need for large overlap.

However, content-aware chunking introduces variable-length chunks and additional confounds.

Therefore experimental order should be:

1. fixed non-overlapping baseline
2. fixed-size overlap sweep
3. larger-block encode + slicing
4. fixed-size vs paragraph-aware
5. section/hierarchy-aware chunking
6. semantic/custom chunking

Do not mix content-aware chunking into the first parameter sweep.

New fragmentation comparison

Expand the fragmentation experiment to compare:

independent fixed chunks, no overlap
independent fixed chunks + overlap
larger encoding blocks + slicing
native historical K/V slicing
paragraph-aware chunks
section-aware chunks

For each strategy report:

oracle RCB
routed RCB
routing MRR
Recall@k
active physical K/V
active unique K/V
storage duplication
routing cost
KV transfer

The key question is:

Can PRA increase the number of addressable context regions while preserving enough local semantic context that oracle quality remains high?

Paper 1 interpretation

If modest overlap substantially improves split-64/128 oracle quality, state cautiously:

Part of the observed degradation under fine partitioning arises from chunk-boundary contextualization loss. Modest overlap recovers some of this loss at the cost of additional K/V storage and encoding.

Do not claim overlap solves fragmentation universally.

If content-aware chunking later outperforms fixed-size chunking, frame it as evidence that:

PRA quality depends not only on retrieval policy but also on how context is partitioned into representationally coherent addressable units.

Tests

Add tests for:

exact overlap fraction/token calculation;
deterministic chunk boundaries;
no missing source tokens;
expected duplicated-token count;
overlap metadata correctness;
deduplication of overlapping retrieved K/V;
fixed-target invariants under overlap;
correct active physical and unique K/V accounting;
variable-length content-aware chunks when those modes are added;
backward compatibility with chunk_overlap_fraction=0.0.

I would definitely add this. In fact, for the current split-64 degradation, **overlap is one of the cheapest experiments that can distinguish boundary-context loss from deeper representation fragmentation**, so it belongs before more invasive routing training.

## 1. Make all routing knobs explicit

Remove benchmark hard-coding where practical and expose/store:

```text
top_k_references
top_k_chunks_per_reference
search_strategy
gist_mode
gists_per_chunk
gist_score_aggregation
reference_level_gist_mode
reference_gists_per_reference
reference_score_aggregation
trigger_threshold
```

Every result row must record actual values used. Do not silently rely on mutable global defaults.

## 2. Capture complete rank diagnostics first

For each layer/example capture:

```text
all candidate reference ids
scores or sufficiently deep ranked list
target reference ids
rank of each target
MRR
Recall@1
Recall@2
Recall@4
Recall@8
Recall@16
Recall@32
best target score
best non-target score
score margin
```

For multi-reference evidence capture:

```text
any_target_hit@k
all_targets_hit@k
fraction_targets_covered@k
```

Do the same at chunk level when references contain multiple chunks.

This determines whether top-1 is the main problem or whether gist ranking itself is poor.

## 3. Sweep `top_k_references` first

### Split 32

```text
k = 1, 2, 4, 8
```

### Split 64

```text
k = 1, 2, 4, 8, 16
```

### Later 128/256

```text
k = 1, 2, 4, 8, 16, 32
```

only after fragmentation is decoupled from reference count.

Expected tradeoff:

```text
higher k
 -> higher evidence recall / lower routing gap
 -> more active K/V / transfer / attention work
 -> potentially more irrelevant-memory composition noise
```

Plot the Pareto frontier, not only best loss:

```text
RCB vs active_fraction
RCB vs retrieved_tokens
Recall@k vs active_fraction
```

## 4. Sweep gist representation second

Baseline mean pooling may dilute localized concepts.

Test:

```text
mean
last
prototype
kmeans
som
hybrid
```

Use GRU later because it introduces learned routing parameters.

For multi-gist modes test:

```text
gists_per_chunk = 1, 2, 4
```

optionally 8 only for sufficiently large chunks.

Important: do not treat `gists_per_chunk > 1` as meaningful for inherently single-gist modes such as `mean`.

Start with:

```text
gist_score_aggregation = max
```

then test `logsumexp` / `mean` only for promising modes.

Multiple gists should primarily increase routing-index cost, not transported token K/V.

Report:

```text
RCB
MRR
Recall@k
gist comparisons
routing latency
active K/V
```

## 5. Make reference-level routing meaningful

The current one-chunk-per-reference layout makes several hierarchical parameters effectively inert.

Add a layout where:

```text
one reference/URI -> multiple internal chunks
```

Then test:

```text
top_k_references
top_k_chunks_per_reference
reference_level_gist_mode
reference_gists_per_reference
reference_score_aggregation
```

PRA is intended to route progressively:

```text
reference -> chunk -> token K/V
```

Do not infer the value of `top_k_chunks_per_reference` from a one-chunk-per-reference benchmark.

## 6. Adaptive threshold only after ranking improves

Top-k is the controlled baseline.

Once ranking recall is strong, test:

```text
trigger_threshold
relative-to-best thresholds
normalized-mass thresholds
hybrid threshold + optional cap
```

Measure distribution of retrieved memory:

```text
mean
median
p90
p99
```

The natural PRA mode should be open-ended/adaptive where practical, with hard caps only as resource ceilings.

## 7. Critical: separate reference count from fragment size

Do NOT scale 64 -> 128 -> 256 by splitting the same short source into ever smaller independently encoded fragments.

That would mostly measure pathological fragmentation.

Create three independent variables:

```text
routing/addressable unit count
encoding block size
materialization/retrieval unit size
```

Hold encoding context reasonably large while increasing addressable units.

## 8. Fragmentation strategies

Compare:

### A. Independent microchunks
Current stress-test baseline.

### B. Larger-block encode + K/V slicing
Example:

```text
encode 32-token block
expose four independently routable 8-token subregions
```

This increases routing resolution without destroying contextualization.

### C. Full/native historical encode + K/V slicing
Encode the source/prefix once through the causal model and slice already-computed native K/V into routing regions.

This is the cleanest test of PRA indexing and should be a primary scale-up condition.

### D. Small overlap
Try minimal overlap such as 1–2 tokens or a small percentage.

### E. Atomic/semantic boundary preservation
Keep known inseparable evidence units together as a diagnostic control.

## 9. Position sensitivity

Compare where possible:

```text
reset/local positions
original/global positions
larger-block encoding + slicing
native historical K/V slicing
```

Do not attribute split-64 degradation to position alone; missing left context can alter deeper states too.

For Paper 2/HF later test RoPE-preserving and virtual-position policies.

## 10. Stage 1: close the split-32 routing gap

Run in this order:

```text
1. baseline rank diagnostics
2. top-k sweep
3. select Pareto candidates
4. multi-gist sweep
5. gist aggregation sweep for best modes
6. adaptive threshold only after ranking is good
```

Suggested engineering target:

```text
routed RCB >= 0.90
target evidence recall >= 0.90
active_fraction <= 0.20
```

These are targets, not conditions for suppressing negative results.

## 11. Stage 2: repair split 64

At split 64, oracle first.

Compare:

```text
independent microchunks
larger-block encode + slice
full-native encode + K/V slice
position-preserving variant
small-overlap variant
```

If native/larger-block slicing restores high oracle RCB, interpret the earlier split-64 decline as fragmentation/composition rather than a hard limit on addressable regions.

Only then rerun routing sensitivity.

Suggested targets:

```text
oracle RCB >= 0.90 if achievable
routed RCB >= 0.90 * oracle_RCB
complete target coverage >= 0.90
active_fraction <= 0.25
```

## 12. Scale to 128 and 256

Proceed only after split 64 has a healthy oracle condition.

Increase total source/context size while keeping encoding blocks meaningful.

Conceptually prefer:

```text
32 units  -> N
64 units  -> ~2N
128 units -> ~4N
256 units -> ~8N
```

rather than keeping N constant and shrinking chunks toward single tokens.

At every scale:

> Oracle first. Router second.

Capture:

```text
accessible_tokens
addressable_units
encoding_block_size
retrieval_unit_size
oracle_RCB
routed_RCB
Recall@k
complete evidence coverage
active_tokens
active_fraction
gist_count
routing comparisons
routing latency
KV bytes
attention latency
```

Main question:

```text
Does active/retrieved memory grow much more slowly than accessible context while RCB stays high?
```

## 13. Model-capacity sensitivity

Current natural-text probe models are tiny, roughly:

```text
d_model=128
layers=2
heads=4
d_ff=256
```

Do not assume this predicts SOTA K-space geometry.

After algorithmic issues are isolated, add SA backbone tiers. Train SA only, then convert to native PRA.

Example tiers:

### Tiny
```text
d_model=128
layers=2
heads=4
d_ff=256
```

### Small
```text
d_model=256
layers=4
heads=4 or 8
d_ff=768 or 1024
```

### Medium local-hardware tier
```text
d_model=384 or 512
layers=6
heads=6 or 8
d_ff ~= 4*d_model
```

Choose exact values based on hardware.

Measure whether model capacity improves:

```text
full-context quality
oracle RCB
routing MRR / Recall@k
fragmentation robustness
```

Do not claim larger models improve routing until measured.

## 14. Training-budget sensitivity

Separate capacity from undertraining.

Track:

```text
train loss
validation loss
full-context target quality
tail/full dependency gain
```

Use adequate SA training per capacity tier. Do not compare models solely at equal step counts if that leaves larger models undertrained.

Native PRA transport remains untrained.

## 15. Avoid full Cartesian search

Use staged search:

```text
top-k
 -> Pareto filter
 -> gist mode/count
 -> Pareto filter
 -> aggregation/threshold
 -> fragmentation
 -> capacity
 -> 128/256 scale
```

Use successive halving if helpful.

Record the search protocol so final parameters are not presented as a priori.

## 16. Pareto metrics

Quality:

```text
RCB
loss
accuracy
routing_gap
MRR
Recall@k
complete evidence coverage
```

Cost:

```text
active_fraction
retrieved_tokens
gist comparisons
routing latency
KV transfer bytes
attention tokens
peak memory
```

Prefer Pareto-optimal settings over a single “best score”.

## 17. Required plots

Add:

```text
Top-k frontier:
    x=active_fraction
    y=routed_RCB
    label=top_k_references

Recall curve:
    x=k
    y=Recall@k / complete coverage@k

Gist sensitivity:
    x=gists_per_chunk
    y=RCB or MRR
    series=gist_mode

Fragmentation decomposition:
    x=addressable units
    y=oracle_RCB
    series=independent / larger-block-slice / native-slice / overlap

Scale frontier:
    x=accessible tokens
    y=RCB

Scale sparsity:
    x=accessible tokens
    y=active_fraction

Model capacity:
    x=model size tier
    y=oracle_RCB / routed_RCB / MRR
```

## 18. Paper 1 updates

Keep the current published native result as the explicit baseline:

> Mean-gist, top-1, untrained routing is a deliberately minimal selector.

Add a parameter-sensitivity/scaling section that distinguishes:

```text
transport capacity
retrieval breadth
routing representation
fragmentation/contextualization
base-model capacity
```

Do not rewrite history as if tuned settings produced the original result.

If split-64 improves with larger-block/native slicing, state that the earlier degradation was primarily a fragmentation/composition effect only if supported.

If 128/256 work, emphasize that context/addressability increased while encoding granularity remained reasonable and active fraction stayed low.

## 19. Relation to Paper 2

Use these experiments to determine the best settings to transfer into pretrained HF models.

Paper 2 should test:

```text
native historical K/V
meaningful chunk size
multi-gist routing
adaptive/top-k retrieval
64/128/256+ addressable regions
```

The tiny-model sweep does not prove SOTA behavior. It identifies which mechanisms and parameters deserve scaling.

## 20. Only train routing after inference-only sensitivity

If unsupervised routing still plateaus after top-k/multi-gist/fragmentation work:

```text
freeze decoder
freeze native K/V transport
train router / routing projection / learned gist pooler only
```

Use ranking loss with hard negatives and target-reference labels.

Do not reintroduce `mem_o_proj` to solve a ranking failure.

## 21. Artifacts

Save:

```text
parameter_sweep_manifest.json
raw_rankings.csv/parquet
raw_runs.csv/parquet
aggregate_pareto.csv
aggregate_by_topk.csv
aggregate_by_gist.csv
aggregate_by_fragmentation.csv
aggregate_by_scale.csv
aggregate_by_model_capacity.csv
figures/
tables/
```

Every paper number must trace to raw artifacts and exact config/checkpoint metadata.

## 22. Tests

Add tests for:

1. benchmark overrides reaching `PRAConfig`;
2. correct single-vs-multi-gist semantics;
3. requested gist counts for valid multi-gist modes;
4. top-k selection counts;
5. multi-target complete coverage;
6. deterministic rank capture;
7. larger-block encode + slicing ranges;
8. native-cache slicing without re-encoding;
9. fixed-target invariants at 128/256;
10. active-fraction accounting excluding routing gists.

## 23. Execution order

```text
1. explicit benchmark routing config
2. rank/Recall@k/MRR capture
3. baseline diagnostics
4. split-32 top-k sweep
5. split-32 multi-gist sweep
6. choose Pareto candidates
7. split-64 fragmentation sweep
8. split-64 routing sweep on best encoding strategy
9. model-capacity sensitivity
10. build meaningful-scale split 128
11. oracle 128
12. routed 128
13. build/run 256 only if 128 oracle is healthy
14. update Paper 1 tables/plots/text
```

## Guiding principle

Do not ask only:

> Which parameter gives the best score?

Ask:

> Which combination of retrieval breadth, gist richness, encoding granularity, and model capacity preserves full-context benefit while making the smallest useful fraction of native K/V active as accessible context and addressability grow?

The 64/128/256 experiments must scale **context and addressability**, not merely fragmentation pathology.
