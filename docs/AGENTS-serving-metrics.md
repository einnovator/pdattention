# AGENTS.md — PRA Paper 2: Systems and Serving Metrics

## Purpose

Paper 2 must evaluate whether the algorithmic advantages demonstrated by PRA Paper 1 translate into practical advantages when PRA is integrated into pretrained Hugging Face models.

The central systems question is:

> **Does reducing the amount of K/V state that must be actively materialized and attended translate into lower memory use, lower latency, higher throughput, greater concurrency, and ultimately lower serving cost while maintaining model quality?**

Paper 2 should therefore measure both model-quality metrics and real systems metrics.

Do not report only theoretical FLOPs or active-token counts. Measure actual GPU behavior wherever hardware permits.

---

# 1. Relationship to Paper 1

Paper 1 already establishes several lower-level PRA metrics.

Based on the current Paper 1 experimental program, metrics already present or substantially represented include:

* logical context / logical token count;
* native/direct attention limit;
* active K/V tokens;
* materialized K/V tokens or materialization budget;
* transfer/materialization volume;
* routing/reference selection behavior;
* attention sparsity;
* causality / causality violations;
* routing or memory benefit;
* loss;
* perplexity;
* token accuracy where applicable;
* comparisons against full attention, truncation, independent blocks, and other relevant baselines.

Paper 2 should preserve these metrics where applicable so results remain comparable with Paper 1.

However, Paper 1's resource metrics are primarily **algorithmic/model-level metrics**.

Paper 2 must additionally determine whether those reductions translate into **real hardware and serving improvements**.

A useful distinction is:

```text
Paper 1
    ↓
Algorithmic efficiency

active K/V
materialized K/V
transfer volume
attention sparsity
routing quality
loss / perplexity

    ↓

Paper 2
    ↓
Hardware + serving efficiency

HBM
KV-cache capacity
TTFT
TPOT
throughput
concurrency
GPU utilization
memory bandwidth
$/M tokens
```

Do not remove Paper 1 metrics from Paper 2 merely because higher-level serving metrics are added.

The relationship between both sets of metrics is itself scientifically important.

---

# 2. Primary Paper 2 Systems Metrics

## 2.1 HBM Usage

Measure actual GPU High-Bandwidth Memory consumption.

Record at minimum:

* idle/base model HBM;
* HBM after model loading;
* HBM after prefill;
* HBM during decode;
* peak HBM;
* average HBM where meaningful;
* HBM attributable to KV cache where measurable.

Report in GiB and as a fraction of total available GPU memory.

Measure:

```text
peak_hbm_gib
avg_hbm_gib
kv_cache_hbm_gib
hbm_fraction
```

The important PRA question is whether reduced active/materialized K/V produces actual GPU-memory savings.

Do not infer HBM savings solely from K/V token counts.

---

## 2.2 KV-Cache Capacity

Measure how much K/V state the system can support before exhausting the configured memory budget.

Report useful forms such as:

```text
maximum physical KV tokens
maximum logical context
maximum batch size
maximum concurrent sequences
KV bytes per request
KV bytes per logical token
```

Distinguish carefully between:

```text
logical context size
```

and

```text
physically resident/materialized KV size
```

This distinction is central to PRA.

A major Paper 2 result would be demonstrating that PRA supports a much larger logical context than would normally fit into the same GPU KV-cache budget.

---

# 3. Latency

## 3.1 TTFT — Time To First Token

TTFT is the elapsed wall-clock time between submitting a request and receiving the first generated token.

Measure:

```text
TTFT mean
TTFT median
TTFT p90
TTFT p95
TTFT p99
```

Test TTFT as a function of:

```text
logical context length
physical/materialized context
number of references
routing configuration
materialization configuration
batch size
concurrency
```

TTFT is especially important for identifying PRA routing and materialization overhead during prefill.

---

## 3.2 TPOT — Time Per Output Token

Measure decode latency after the first token has been generated.

Report:

```text
mean TPOT
median TPOT
p90
p95
p99
```

Where useful also report:

```text
inter-token latency
```

PRA may have a particularly strong effect during decode because attention against the K/V state is repeatedly executed for every generated token.

---

# 4. Throughput

Separate prefill and decode performance.

Measure:

```text
prefill tokens/s
decode tokens/s
total tokens/s
requests/s
```

Do not collapse all of these into a single throughput number.

Run throughput experiments across:

```text
context length
output length
batch size
concurrency
materialization budget
routing budget
```

The central question is:

> How much of the reduction in active attention work translates into real end-to-end throughput?

For example, a 10× reduction in active K/V does not imply a 10× throughput improvement.

Measure the conversion empirically.

---

# 5. Concurrency

Measure the number of simultaneous sequences/requests that can be supported under fixed hardware.

Test increasing concurrency until one or more constraints become binding:

* HBM exhaustion;
* unacceptable TTFT;
* unacceptable TPOT;
* throughput saturation;
* GPU saturation;
* routing/materialization bottleneck.

Report:

```text
max feasible concurrency
max concurrency under latency SLA
throughput vs concurrency
HBM vs concurrency
TTFT vs concurrency
TPOT vs concurrency
```

This may be one of the most commercially important PRA metrics.

Reduced KV-cache requirements should potentially allow significantly more simultaneous requests per GPU.

---

# 6. GPU Utilization

Measure actual GPU utilization rather than assuming reduced FLOPs imply improved hardware efficiency.

Capture where possible:

```text
GPU compute utilization %
memory utilization %
SM utilization %
tensor-core utilization
kernel occupancy
```

The purpose is to detect whether PRA creates:

* routing stalls;
* synchronization overhead;
* small inefficient kernels;
* excessive transfers;
* poor batching;
* CPU/GPU synchronization;
* underutilized GPU execution.

A theoretically cheaper attention mechanism that substantially lowers GPU utilization may fail to deliver proportional serving gains.

---

# 7. Memory Bandwidth

Measure memory traffic and achieved memory bandwidth where tooling/hardware permits.

Record:

```text
GB/s
bytes read
bytes written
KV bytes transferred
host → device transfer
device → host transfer
device → device transfer
```

This is particularly important because LLM decoding is often strongly memory-bandwidth constrained.

Connect these measurements to the existing Paper 1 transfer/materialization metrics.

Paper 1 may show:

```text
fewer K/V vectors transferred/materialized
```

Paper 2 should determine whether this produces:

```text
less physical memory traffic
```

and ultimately:

```text
lower TPOT / higher throughput
```

---

# 8. Cost per Million Tokens

Estimate or directly measure:

```text
$/M input tokens
$/M output tokens
$/M total served tokens
```

State all assumptions explicitly:

```text
GPU type
GPU hourly cost
utilization assumption
batching/concurrency
input/output ratio
context distribution
```

A simple first-order calculation is:

```text
cost_per_million_tokens =
    gpu_cost_per_second
    * seconds_required_for_1M_tokens
```

For realistic serving, account for utilization and concurrency rather than assuming one request owns the entire GPU.

The important comparison is:

```text
baseline $/M tokens
vs.
PRA $/M tokens
```

at comparable model quality and workload.

---

# 9. Quality Must Remain Coupled to Efficiency

Never present efficiency improvements without the associated model quality.

For every major operating point, retain metrics such as:

```text
loss
perplexity
task accuracy
retrieval-dependent task score
routing accuracy/recall where available
```

The relevant object is not maximum speed in isolation.

It is the Pareto frontier:

```text
quality
    vs.
memory
    vs.
latency
    vs.
throughput
    vs.
cost
```

Paper 2 should identify Pareto-efficient PRA configurations rather than claiming one universally optimal configuration.

---

# 10. Required Baselines

Where technically possible, compare equivalent workloads using:

1. Standard full/self attention.
2. Context truncation.
3. PRA.
4. Relevant efficient-attention / KV-management baselines where scientifically appropriate.

Keep constant whenever possible:

```text
model weights
precision
GPU
batching policy
input
output length
sampling configuration
software stack
compiler settings
```

Do not compare PRA on one optimized serving stack against a poorly configured baseline.

---

# 11. Scaling Sweeps

Run experiments over increasing logical context.

Suggested conceptual sweep:

```text
1× native
2×
4×
8×
16×
32×
```

Extend further where PRA remains practical.

For each context scale capture at least:

```text
quality
active KV
materialized KV
peak HBM
KV-cache size
TTFT
TPOT
prefill tokens/s
decode tokens/s
total throughput
GPU utilization
concurrency
```

This creates the empirical foundation for the later PRA scaling-theory work.

---

# 12. Batch and Concurrency Sweeps

Do not benchmark only batch size 1.

At minimum investigate approximately:

```text
batch = 1, 2, 4, 8, 16, ...
```

until hardware saturation.

Also perform concurrency sweeps independently when the serving runtime supports continuous batching.

PRA may provide modest latency improvements at batch 1 while producing much larger throughput improvements under memory-constrained concurrent serving.

That distinction is important.

---

# 13. Prefill vs Decode

Treat these as separate computational regimes.

## Prefill

Measure:

```text
prefill latency
prefill tokens/s
peak HBM
routing time
materialization time
attention time
```

## Decode

Measure:

```text
TPOT
decode tokens/s
active KV
materialization updates
routing overhead
memory bandwidth
```

Do not assume PRA affects both phases equally.

---

# 14. Decompose PRA Overhead

Instrument PRA-specific work separately.

Where possible record:

```text
gist construction time
reference routing time
chunk routing time
materialization time
KV transfer time
PRA bookkeeping time
attention time
```

Then estimate:

```text
PRA overhead
```

versus:

```text
attention work avoided
```

This allows the paper to explain why observed wall-clock speedups differ from theoretical attention reductions.

---

# 15. Recommended Summary Table

Every major benchmark configuration should ultimately permit a table approximately of the form:

| Metric           | Full Attention | Truncation | PRA | PRA improvement |
| ---------------- | -------------: | ---------: | --: | --------------: |
| Quality          |                |            |     |                 |
| Logical context  |                |            |     |                 |
| Active K/V       |                |            |     |                 |
| Materialized K/V |                |            |     |                 |
| Transfer volume  |                |            |     |                 |
| Peak HBM         |                |            |     |                 |
| KV cache/request |                |            |     |                 |
| TTFT             |                |            |     |                 |
| TPOT             |                |            |     |                 |
| Prefill tok/s    |                |            |     |                 |
| Decode tok/s     |                |            |     |                 |
| Total tok/s      |                |            |     |                 |
| Max concurrency  |                |            |     |                 |
| GPU utilization  |                |            |     |                 |
| Memory bandwidth |                |            |     |                 |
| $/M tokens       |                |            |     |                 |

Use ratios such as:

```text
logical-context expansion
KV reduction
HBM reduction
TTFT speedup
TPOT speedup
throughput speedup
concurrency multiplier
cost reduction
```

where meaningful.

---

# 16. Expected Causal Chain

Paper 2 should explicitly test, rather than assume, the following causal chain:

```text
PRA routing
    ↓
less active/materialized K/V
    ↓
less KV storage / memory traffic
    ↓
lower HBM pressure
    ↓
larger feasible logical contexts
and/or
higher concurrency
    ↓
lower latency and/or higher throughput
    ↓
lower serving cost
```

A failure at any arrow is scientifically useful.

For example:

```text
10× active-KV reduction
→ 8× HBM reduction
→ 4× concurrency
→ 2.5× throughput
→ 2.2× cost reduction
```

would be considerably more informative than simply reporting:

```text
10× fewer K/V tokens.
```

---

# 17. Negative and Nonlinear Results

Do not hide cases where PRA's theoretical reduction fails to translate proportionally to hardware performance.

Investigate them.

Examples:

* routing overhead dominates for short contexts;
* materialization causes excessive memory transfers;
* GPU utilization falls;
* small kernels become inefficient;
* batching becomes fragmented;
* HBM improves without TPOT improving;
* TTFT worsens while decode improves;
* concurrency improves much more than single-request latency.

These results help identify the operating regime in which PRA is useful.

The objective is to characterize the **PRA operating frontier**, not merely produce favorable benchmark numbers.

---

# 18. Paper 1 vs Paper 2 Metric Map

Use the following conceptual classification when writing Paper 2.

## Already established / substantially covered by Paper 1

```text
loss / perplexity
token/task accuracy where applicable
logical context length
native/direct context limit
active K/V
materialization budget / materialized K/V
transfer/materialization volume
attention sparsity
routing behavior
causality
memory benefit
algorithmic comparisons against truncation/full attention/etc.
```

## Added or substantially expanded in Paper 2

```text
actual HBM consumption
peak vs average HBM
physical KV-cache bytes
maximum KV-cache capacity
TTFT
TPOT
latency percentiles
prefill throughput
decode throughput
end-to-end throughput
requests/s
concurrency
concurrency under latency SLA
GPU utilization
SM/kernel utilization where measurable
memory bandwidth
actual transfer bandwidth
$/M input tokens
$/M output tokens
$/M served tokens
PRA wall-clock overhead decomposition
```

## Bridge metrics

These should appear in both papers because they connect algorithmic behavior to systems behavior:

```text
logical context
active K/V
materialized K/V
transfer volume
quality
```

These bridge metrics are essential for explaining **why** a particular systems result occurs.

---

# 19. Reproducibility

Every benchmark must record enough environment information to reproduce it:

```text
GPU model
GPU count
VRAM/HBM capacity
driver
CUDA version
PyTorch version
Transformers version
serving-runtime version
model/checkpoint
dtype
quantization
attention backend
batch size
concurrency
logical context
physical/materialized context
output length
PRA configuration
gist configuration
routing configuration
materialization configuration
random seed
warmup policy
number of measured runs
```

Store raw measurements as machine-readable CSV/JSON/Parquet in addition to producing paper tables.

Benchmark scripts should produce the metrics automatically rather than requiring manual transcription.

---

# 20. Long-Term Scaling-Theory Compatibility

Design Paper 2's result schema so later experiments can model quantities such as:

```text
Q = quality
N = logical context
A = active KV
M = materialized KV
G = gist count
K = routing budget
B = memory bandwidth
H = HBM
C = concurrency
L = latency
T = throughput
D = serving cost
```

The eventual scaling work should be able to ask questions such as:

```text
Q = f(N, A, M, G, K, ...)
```

and:

```text
T = f(A, M, B, C, hardware, ...)
```

and ultimately characterize the constrained frontier:

```text
maximize quality and throughput
subject to:

HBM <= memory budget
latency <= SLA
cost <= serving budget
```

Therefore preserve raw experimental data, not merely aggregated values used in plots.

---

# Core Principle

Paper 1 asks:

> **Can Progressive Retrieval Attention work?**

Paper 2 should additionally ask:

> **Does PRA continue to work on pretrained models, and do its algorithmic savings become real memory, latency, throughput, concurrency, and economic savings?**

Always connect the two.

The strongest Paper 2 result is not merely a lower perplexity or a smaller theoretical attention matrix.

It is evidence that:

```text
larger logical context
+
comparable model quality
+
smaller physical KV footprint
+
better serving efficiency
```

can coexist in the same pretrained model.
