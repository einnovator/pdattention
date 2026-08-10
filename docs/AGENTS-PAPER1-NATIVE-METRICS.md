# AGENTS-PAPER1-NATIVE-METRICS.md

## Mission

Extend the native-KV PRA evaluation so Paper 1 can make a clean, causal, systems-relevant comparison between full SelfAttention context, local/tail-only context, PRA with all displaced native K/V restored, PRA with sparse oracle-selected native K/V, PRA with routed native K/V, PRA with shuffled/wrong memory, and PRA with memory disabled.

The goal is not just to collect raw loss values. Quantify:

1. how closely native PRA approximates full self-attention;
2. how much of the useful long-context benefit sparse PRA recovers;
3. how much K/V must actually be active;
4. how routing quality differs from transport quality;
5. how these quantities scale as context is partitioned into more chunks;
6. whether sparse native PRA provides a favorable quality-vs-active-context frontier.

Use the fixed-target WikiText-2 split experiment already added for:

```text
2, 3, 5, 8, 16, 32, 64
```

Do not alter user-provided instruction files.

## 1. Experimental invariants

For every split-count condition, keep constant:

- source document;
- final direct/local tail;
- prediction target;
- evaluated target token positions;
- tokenizer;
- model checkpoint;
- seed pairing where applicable;
- decoding/evaluation mode;
- direct local-context size.

Only the displaced prefix partitioning may change.

Add assertions or metadata checks that verify target identity.

## 2. Required native-KV conditions

Capture results for:

### `sa_full`
Ordinary SelfAttention with the full accessible context. This is the quality ceiling / dense-context baseline.

### `sa_tail`
Only the direct local/tail context. This measures how much useful information is lost when the prefix is displaced.

### `pra_native_all`
Native-KV PRA with all displaced historical K/V restored. This measures transport / partition / position fidelity.

### `pra_native_oracle`
Restore only known relevant chunk(s). This measures whether a sparse K/V subset can recover the useful benefit of full context.

### `pra_native_routed`
Use normal PRA gist/reference routing. This measures actual selection quality relative to the oracle upper bound.

### `pra_native_shuffled`
Use the same memory amount/shape but wrong or shuffled content. This tests content causality.

### `pra_native_disabled`
Enable PRA machinery but materialize no external memory. This controls for wrapper/path effects and should reproduce local-only behavior.

## 3. Raw metrics

For every evaluation row, capture at minimum:

```text
experiment_id
timestamp
checkpoint_id
model_name
seed
dataset
example_id
split_count
condition
transport_mode
routing_mode
gist_mode
top_k_or_threshold
local_tokens
accessible_tokens
displaced_tokens
retrieved_tokens
active_tokens
active_fraction
num_references
num_chunks
num_selected_chunks
num_selected_references
loss
perplexity
token_accuracy_if_available
```

Where available also capture:

```text
routing_score_statistics
oracle_selected_chunk_ids
routed_selected_chunk_ids
selection_hit
recall_at_k
reference_top1
chunk_top1
chunk_recall_at_k
attention_latency
routing_latency
kv_materialization_latency
kv_transfer_bytes
peak_cuda_memory
host_memory_use
```

Do not block core quality/sparsity experiments if some systems metrics are not yet available.

## 4. Derived metrics

Implement these centrally so all plots/tables use the same canonical calculations.

### 4.1 Transport gap

```text
transport_gap =
    loss(pra_native_all) - loss(sa_full)
```

Interpretation:

```text
0     -> full fidelity
> 0   -> transport/partition/position cost
< 0   -> PRA-all happened to outperform dense baseline
```

### 4.2 Sparse approximation gap

```text
sparse_gap =
    loss(pra_native_oracle) - loss(pra_native_all)
```

Small means a sparse oracle subset preserves most of the all-memory result.

### 4.3 Memory benefit

```text
memory_benefit =
    loss(sa_tail) - loss(pra_condition)
```

Compute for at least all, oracle, routed, and shuffled PRA. Positive is better.

### 4.4 Full-context benefit / dependency gain

```text
full_context_benefit =
    loss(sa_tail) - loss(sa_full)
```

This measures whether the target actually benefits from displaced history.

### 4.5 Recovered Context Benefit (RCB)

Add this as a first-class Paper 1 metric:

```text
RCB =
    (loss(sa_tail) - loss(pra_condition))
    /
    (loss(sa_tail) - loss(sa_full))
```

Compute at least:

```text
RCB_all
RCB_oracle
RCB_routed
RCB_shuffled
```

Interpretation:

```text
RCB = 0  -> PRA recovers none of the full-context benefit
RCB = 1  -> PRA recovers all measured full-context benefit
RCB > 1  -> PRA exceeds the full-context baseline on that target
RCB < 0  -> retrieved memory is harmful relative to tail-only
```

If the denominator is near zero, mark RCB undefined / low-dependency rather than exploding numerically.

Do not clamp RCB to [0,1].

### 4.6 Content causality gap

```text
content_causality_gap =
    loss(pra_native_shuffled) - loss(pra_native_valid)
```

Use oracle and routed valid baselines where appropriate. Positive means correct content matters.

### 4.7 Routing gap

```text
routing_gap =
    loss(pra_native_routed) - loss(pra_native_oracle)
```

This isolates routing error after transport and sparse-oracle feasibility have been measured.

### 4.8 Active fraction

Define consistently:

```text
active_fraction =
    active_token_kv / total_accessible_token_kv
```

Prefer:

```text
active_token_kv =
    local token K/V
    +
    materialized retrieved token K/V
```

Do not include gist/index vectors in token-level active fraction. Record gist/index cost separately.

## 5. Error decomposition

Paper 1 should explicitly decompose error relative to full SA:

```text
full SA
  -> PRA native all
      transport / partition / position error

PRA native all
  -> PRA native oracle
      sparsification error

PRA native oracle
  -> PRA native routed
      routing error
```

Corresponding loss gaps:

```text
transport_gap
sparse_gap
routing_gap
```

This decomposition should appear in code comments, result tables, plots, and the paper narrative.

## 6. Dependency-sensitive analysis

WikiText-2 targets often do not strongly require distant context.

Use:

```text
dependency_gain =
    loss(sa_tail) - loss(sa_full)
```

Create low/medium/high dependency strata, preferably using quantiles or another data-driven rule. Do not choose thresholds to favor PRA.

Report:

- overall results;
- high-dependency subset;
- optionally low/medium subsets.

High-dependency examples are especially important for RCB, memory benefit, and causality analysis.

## 7. Aggregation

Aggregate at:

```text
per-example
per-seed
per-split
across-seed summary
```

Use paired comparisons whenever the same seeds/examples occur across conditions.

Report where practical:

```text
mean
standard deviation
median
95% confidence interval
```

For very small seed counts, do not overstate p-values. Distinguish effect size, directional consistency, and significance.

## 8. Main Paper 1 plots

Create publication-ready plots and save machine-readable source CSV/JSON next to figures.

### Figure A — Quality vs split count

X-axis:

```text
2,3,5,8,16,32,64
```

Y-axis: loss.

Series:

```text
SA full
SA tail
PRA native all
PRA native oracle
PRA native routed
PRA native shuffled
```

### Figure B — Recovered Context Benefit vs split count

X-axis: split count.

Y-axis: RCB.

Series:

```text
all-memory
oracle
routed
shuffled
```

Include reference lines at RCB=1 and optionally RCB=0.

### Figure C — Quality gap vs active K/V fraction

X-axis:

```text
active_fraction
```

Y-axis:

```text
loss(PRA) - loss(SA_full)
```

Use oracle budget sweeps where available.

This is a central Paper 1 figure.

### Figure D — RCB vs active K/V fraction

X-axis: active fraction.

Y-axis: RCB.

This directly shows how much full-context benefit is recovered for a given active K/V fraction.

### Figure E — Active K/V vs accessible context

X-axis: accessible tokens or split-derived accessible context.

Y-axis: active token K/V.

Series where available:

```text
full SA
oracle PRA
routed PRA
```

### Figure F — Active fraction vs accessible context

X-axis: accessible tokens.

Y-axis: active fraction.

Show whether active fraction falls as logical context grows.

### Figure G — Error decomposition

For each split count show:

```text
transport_gap
sparse_gap
routing_gap
```

Use grouped bars or another clear representation.

### Figure H — Dependency-sensitive RCB

Compare all examples versus high-dependency examples for at least oracle, routed, and shuffled.

## 9. Optional systems plots

If timing/memory capture is robust, add:

```text
latency vs accessible context
peak CUDA memory vs accessible context
KV transfer bytes vs retrieved K/V
routing latency vs chunks/gists
```

Do not delay core quality/sparsity analysis waiting for perfect profiling.

## 10. Machine-readable result layout

Prefer the project's existing convention. If none exists, use something like:

```text
results/native_kv/
    raw_runs.csv
    per_example.csv
    per_seed.csv
    aggregate_by_split.csv
    aggregate_by_budget.csv
    dependency_strata.csv
    figures/
    tables/
```

Every paper table/figure must be reproducible from committed scripts plus raw result files.

Do not hard-code paper numbers into plotting scripts.

## 11. Paper 1 tables

### Table 1 — Native-KV main results

Columns:

```text
split
SA full loss
SA tail loss
PRA all loss
PRA oracle loss
PRA routed loss
PRA shuffled loss
oracle RCB
routed RCB
active fraction
```

### Table 2 — Error decomposition

Columns:

```text
split
transport gap
sparse gap
routing gap
content causality gap
```

### Table 3 — Dependency-sensitive results

Columns:

```text
dependency stratum
N examples
full-context benefit
oracle RCB
routed RCB
shuffled RCB
```

## 12. Paper 1 narrative

Update Paper 1 so the native-KV story proceeds causally.

### First: equivalence

Establish that the trained SA checkpoint can be converted to native-KV PRA without reference-conditioned training and that restoring all exact historical K/V reproduces ordinary attention within expected tolerance.

### Second: context matters

Show SA full vs SA tail and quantify full_context_benefit.

### Third: sparse oracle feasibility

Compare PRA native oracle with PRA native all/full SA and report sparse_gap, RCB_oracle, and active_fraction.

This answers the fundamental sparsity question.

### Fourth: routing

Only after oracle feasibility, compare routed vs oracle using routing_gap, selection metrics, and RCB_routed.

### Fifth: content causality

Use shuffled memory and report content_causality_gap. Do not infer causal use merely from a disabled-memory penalty.

### Sixth: scaling

Use the 2–64 split progression to ask:

> As accessible context is partitioned/scaled, can PRA recover most of the useful context benefit while making only a small fraction of token K/V active?

This should become one of the main Paper 1 conclusions if supported.

## 13. Preferred wording

Prefer statements like:

> Native-KV PRA introduces no reference-specific transport parameters and can therefore be evaluated directly from a normally trained SelfAttention checkpoint.

> The native experiment separates transport fidelity, sparsification error, and routing error.

> Recovered Context Benefit measures the fraction of the predictive advantage of full context that PRA recovers over a tail-only baseline.

> Active-KV fraction measures the fraction of accessible token-level K/V that participates in the final attention computation.

> The key scaling question is whether RCB remains high while active-KV fraction falls as accessible context increases.

Do not claim infinite context, constant total cost, industrial-scale savings, routing success, or causal memory use unless measured results support them.

## 14. Cross-attention results placement

Keep historical cross-attention results in a later section labeled clearly as optional adapted cross-attention transport.

Retain findings on frozen adaptation, forgetting, scratch/joint training, weak shuffled/oracle effects, and routing.

Do not mix those numbers into native-KV main tables.

A short later comparison of native inference-only transport vs adapted cross-attention transport is fine, but do not claim superiority without comparable experiments.

## 15. Statistical handling

For native-KV results:

- preserve paired seeds/examples;
- report effect sizes;
- report confidence intervals where practical;
- use exact paired tests only where appropriate;
- do not over-focus on p<0.05 with very small seed counts.

For deterministic equivalence tests, numerical tolerance matters more than significance.

For split scaling, emphasize trends and effect sizes.

## 16. Validation requirements

Before accepting plots/paper tables:

1. verify split conditions use identical targets;
2. verify `sa_full` and `sa_tail` use the same checkpoint;
3. verify native all-memory restores intended K/V;
4. verify shuffled memory genuinely changes content while preserving comparable memory volume;
5. verify active-fraction denominators are consistent;
6. verify RCB denominator handling;
7. verify no cross-attention result enters native tables;
8. verify no PRA training occurred for native-KV runs;
9. verify every paper number traces to a machine-readable result row;
10. rebuild Paper 1 PDF after updates.

## 17. Minimum deliverables

Implement or update:

```text
native metric capture
derived metric computation
result aggregation
plotting script/notebook
Paper 1 tables
Paper 1 figures
Paper 1 text/results interpretation
```

Prefer reusable Python modules/scripts over notebook-only logic. If notebooks are used for exploration, final plots/tables must also be reproducible non-interactively.

## 18. Completion criteria

Complete when:

- all required native conditions emit compatible raw metrics;
- `transport_gap`, `sparse_gap`, `routing_gap`, `memory_benefit`, `full_context_benefit`, `content_causality_gap`, `active_fraction`, and `RCB` are implemented;
- split 2/3/5/8/16/32/64 results aggregate cleanly;
- dependency-sensitive analysis exists;
- main figures are generated automatically;
- Paper 1 contains the native-KV metrics/results section;
- existing cross-attention results are moved later and clearly labeled;
- all figures/tables trace to result files;
- Paper 1 PDF rebuilds successfully.

## Guiding principle

The key Paper 1 result is not simply:

```text
PRA loss < some baseline loss
```

The important relationship is:

```text
accessible context increases
active K/V fraction decreases
recovered context benefit remains high
```

or, if that does not occur, a clear decomposition showing whether the limitation comes from transport, sparsification, routing, or weak context dependence in the dataset.

Design the metrics and paper so either outcome is scientifically interpretable.
