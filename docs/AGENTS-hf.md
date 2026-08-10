
# AGENTS-PAPER2.md — PRA on Pretrained Hugging Face LLMs

## Mission

Paper 2 should test the highest-value practical hypothesis of Progressive Retrieval Attention (PRA):

> **Can an existing pretrained causal LLM obtain substantially larger effective context through an inference-time PRA wrapper, without retraining its base weights, while keeping active token-level attention far smaller than total accessible context?**

The paper is not primarily about training a new long-context architecture.

It is about determining whether PRA can be added to already trained Hugging Face models as an inference/system transformation over their native attention K/V.

The intended progression is:

```text
existing pretrained HF model
    -> expose native per-layer K/V
    -> externalize old/reference K/V
    -> build small layer-specific routing gists
    -> progressively retrieve relevant native K/V
    -> inject selected K/V into ordinary attention
    -> keep original model weights frozen
```

The strongest result would show that:

```text
accessible/effective context grows by large factors
active token K/V grows much more slowly
quality remains near a relevant full-context baseline
latency and accelerator-memory growth are substantially below dense long-context inference
```

This is the core Paper 2 question.

---

# 1. Strategic framing

Do not frame Paper 2 as merely “adding references to Hugging Face models”.

Do not frame PRA as RAG inside attention.

Do not assume a fixed active-memory budget is fundamental to PRA.

The high-impact hypothesis is:

> **PRA may make attention cost scale more closely with relevant context than with available context.**

Let:

```text
N = total accessible historical/context tokens
Q = current layer query state
R(N,Q) = token K/V materialized by PRA for Q
```

The desired regime is:

```text
N -> very large

R(N,Q) is adaptively determined by relevance

R(N,Q) << N for typical sparse-dependency queries

R(N,Q) / N -> 0 where context sparsity permits

quality_gap_to_full_attention remains small
```

A fixed `top_k`, token budget, or maximum retrieved K/V is useful as:

- an experimental control;
- a serving resource limit;
- a latency/SLA ceiling;
- a safety fallback.

It should **not** be presented as the essential PRA algorithm.

The default conceptual algorithm should permit open-ended adaptive retrieval.

---

# 2. PRA versus RAG

Maintain a strict distinction.

## PRA

PRA manages context that is already available to the inference process:

```text
native layer K/V
 -> external storage
 -> routing/index
 -> selective reactivation
 -> native self-attention
```

## RAG/search

RAG or external search is only needed when the desired information has not been provided and is not already referenced/available:

```text
query
 -> external retrieval/search/tool
 -> retrieved text/data
 -> tokenizer/model
 -> native K/V
 -> PRA may then manage this state
```

Use the short distinction:

> **RAG acquires context. PRA manages and reactivates context.**

If a user supplies a 1M-token prompt, PRA should be able to manage that prompt without invoking RAG.

If a user explicitly references a known document/context region, PRA should resolve/materialize it.

If the information has never been supplied or referenced, search/RAG may precede PRA as a separate preprocessing step.

---

# 3. Canonical transport: native K/V

Paper 2 must treat native K/V injection as the canonical PRA transport.

At layer `l`, the current model has:

```text
Q_l
K_local_l
V_local_l
```

PRA retrieves:

```text
K_selected_l
V_selected_l
```

and computes ordinary attention over:

```text
[K_selected_l ; K_local_l]
[V_selected_l ; V_local_l]
```

using:

- one shared attention normalization;
- the model's existing `o_proj`;
- the model's existing head organization;
- the model's existing attention semantics.

Do not require:

- `mem_o_proj`;
- a separate memory softmax;
- `memory_alpha`;
- a learned transport adapter;
- reference-conditioned base-model fine-tuning.

Cross-attention remains an optional later ablation inherited from Paper 1.

---

# 4. Hugging Face target models

Select a small but representative set of open pretrained causal LLMs.

Prefer models that cover important attention/cache variants without making the paper unmanageably broad.

Target diversity should include, where feasible:

- standard MHA;
- GQA;
- possibly MQA if convenient;
- RoPE;
- modern RMSNorm/SwiGLU decoder blocks;
- models supported by standard HF cache interfaces;
- at least one model small enough for extensive local experimentation;
- at least one more realistic multi-billion-parameter model.

Do not depend on a single architecture-specific hack.

The goal is to demonstrate that PRA can be implemented as a reusable inference transformation across standard decoder-only model families.

Use model families and exact checkpoints that are practical and legally distributable at implementation time.

Keep the wrapping API conceptually simple:

```python
model = AutoModelForCausalLM.from_pretrained(...)

pra_model = PRA.wrap(
    model,
    direct_context=...,
    transport="native_kv",
    retrieval_policy="threshold",
)
```

The real implementation may need architecture adapters, but preserve this product-level abstraction.

---

# 5. Wrapping architecture

Prefer a modular architecture such as:

```text
PRAWrapper
    |
    +-- model adapter
    |
    +-- attention adapter per supported HF family
    |
    +-- KV external store
    |
    +-- gist/index store
    |
    +-- reference/chunk manager
    |
    +-- retrieval policy
    |
    +-- metrics/tracing
```

Avoid forking complete HF model implementations unless unavoidable.

Where possible:

- wrap/replace attention modules;
- hook cache creation/consumption;
- keep base weights untouched;
- retain compatibility with `generate`;
- preserve HF model/config/tokenizer usage.

Separate generic PRA logic from model-family-specific adaptation.

---

# 6. Native K/V lifecycle

Implement and document the lifecycle clearly.

```text
tokens processed
    -> layer-native K/V computed
    -> local active K/V retained
    -> older/reference K/V externalized
    -> layer-specific gists generated
    -> full K/V stored in external hierarchy
    -> query scores gists
    -> selected K/V materialized
    -> selected K/V injected into ordinary attention
```

Distinguish:

1. **hot/local K/V**
   Accelerator-resident active context.

2. **warm K/V**
   Potentially host-RAM-resident chunks.

3. **cold/archive K/V**
   Lower-priority or disk-backed state if experiments reach this scale.

4. **gists/index**
   Small routing representation, ideally cheap enough to remain readily accessible.

Do not require all tiers in the first implementation, but design interfaces so memory hierarchy experiments can be added without rewriting attention logic.

---

# 7. Adaptive retrieval, not fixed memory by definition

Implement retrieval-policy abstraction.

Suggested modes:

```python
retrieval_policy = "threshold"
retrieval_policy = "topk"
retrieval_policy = "hybrid"
retrieval_policy = "oracle"
```

## Threshold mode

Retrieve all chunks/references above a relevance criterion.

Potential criteria include:

```text
absolute score threshold
relative-to-best threshold
score-margin threshold
normalized mass threshold
```

## Top-k mode

Useful primarily for controlled experiments and hard serving budgets.

## Hybrid mode

Example:

```text
retrieve all chunks above threshold
subject to optional maximum tokens/chunks
```

## Oracle mode

Bypasses routing to measure the upper bound of transport/sparsity.

Configuration should allow open-ended retrieval:

```python
max_retrieved_chunks: Optional[int] = None
max_retrieved_tokens: Optional[int] = None
```

`None` means no explicit hard cap.

These fields are resource ceilings, not PRA's core selection rule.

---

# 8. Progressive/hierarchical retrieval

Paper 2 should begin moving beyond a flat scan when accessible context becomes large.

Conceptual hierarchy:

```text
query
 -> reference gists
 -> relevant references
 -> chunk gists
 -> relevant chunks
 -> optional subchunk/prototype level
 -> materialize native token K/V
```

The retrieval hierarchy should be separable from the model.

Measure both:

```text
retrieval quality
retrieval cost
```

Do not claim long-context scaling if the routing step simply scans every token K/V.

A flat gist scan is acceptable as an early baseline.

For large-N experiments, add at least one indexed/hierarchical strategy if practical.

---

# 9. Gists

Gists are routing/index structures, not normally the memory transported into attention.

Support/reuse Paper 1 gist modes where available:

```text
mean
last
multiple gists
K-means
SOM
Hebbian/prototype
hybrid
GRU
```

The main transport remains the original native token K/V of selected chunks.

Layer-specific gists are important.

Do not collapse all layers into a single external sentence-embedding retrieval space unless explicitly testing that as a baseline.

For each model layer `l`:

```text
Q_l -> G_l(K_l) -> select -> K_l,V_l
```

This preserves PRA's key idea of searching in the Transformer's own learned attention geometry.

---

# 10. RoPE and positional handling

This is a major Paper 2 technical topic.

Differentiate:

## Exact historical K/V

If K/V were produced during the same causal history and are reactivated with compatible positions, they are the cleanest case.

## Independently materialized reference chunks

These may differ because:

- their missing original left context changes deeper hidden states;
- their position assignment differs;
- RoPE rotations may differ.

Do not assume position is the only difference.

Test positional policies explicitly.

Candidate policies:

```text
original_global_position
contiguous_before_local
fixed_virtual_distance
distance_clipped
```

Investigate practical mechanisms such as:

- cache pre-RoPE K where architecture permits;
- undo/reapply RoPE rotations;
- regenerate positional rotation at materialization time;
- preserve intra-chunk order while changing inter-chunk distance.

A potentially important hypothesis is:

> Exact local/intra-chunk relative position may matter much more than preserving huge absolute/relative distances between a remote reference and the live prompt.

Treat this as an empirical question.

---

# 11. MHA, GQA and MQA

PRA must correctly preserve each model's native head semantics.

For MHA:

```text
num_q_heads == num_kv_heads
```

For GQA:

```text
num_q_heads > num_kv_heads
```

Selected memory K/V must retain the model's native KV-head layout.

Do not expand GQA K/V unnecessarily in storage.

Perform whatever repeat/interleave operation the base implementation normally performs only at the appropriate computation stage.

Measure storage savings accurately using actual KV-head count.

Support architecture adapters rather than assuming MHA everywhere.

---

# 12. HF cache compatibility

Study current Hugging Face cache abstractions and support them cleanly where practical.

Requirements:

- ingest ordinary generated K/V;
- externalize selected cache regions;
- restore/materialize regions;
- preserve generation correctness;
- distinguish prefill and decode;
- support batch dimensions where feasible.

Do not build a PRA cache API that is inseparable from the toy model.

Prefer a PRA cache layer with conversion/adaptation to HF cache structures.

---

# 13. Prefill versus decode

Paper 2 should report prefill and decode separately.

PRA can affect them differently.

## Prefill

Very long supplied prompts may be partitioned:

```text
early prefix -> implicit references / externalized K/V
final direct tail -> active prompt
```

The recent `#__head` concept should be supported:

```text
if prompt length > max_direct_access:
    early prefix -> implicit reference #__head
    final tail -> direct context
```

The user should not need explicit reference syntax for ordinary long prompts.

## Decode

During generation:

```text
new local K/V accumulate
old local chunks externalize
gists are maintained
relevant historical chunks are reactivated when needed
```

Measure decode token latency separately from initial prefill cost.

---

# 14. Experimental progression

Do not jump directly to learned routing.

## Experiment 1 — Native HF equivalence

Take a pretrained HF model.

For an exact historical prefix, externalize native K/V and restore all of them.

Compare output logits to ordinary HF cached/dense attention under equivalent masks and positions.

Target:

```text
PRA-native-all ~= standard model
```

within expected numerical tolerance.

This validates the wrapper.

## Experiment 2 — Remove/reinsert historical chunks

Externalize chunks from the cache and selectively reinsert them.

Use oracle selection first.

Measure output/logit/loss difference.

## Experiment 3 — Context multiplier sweep

Test logical/access context multipliers such as:

```text
1x
2x
4x
8x
16x
32x
64x
```

subject to hardware/data feasibility.

Do not require all models to reach the same maximum multiplier.

## Experiment 4 — Fixed-budget frontier

Although fixed budget is not fundamental, it is an important controlled experiment.

Hold retrieved K/V budget approximately fixed while accessible context increases.

Measure the quality/cost frontier.

## Experiment 5 — Adaptive retrieval

Enable threshold/hybrid retrieval.

Now let retrieved memory vary according to query relevance.

Measure:

```text
R(N,Q)
R/N
quality
latency
```

The desired result is not constant R by definition.

It is that **R grows substantially more slowly than N where the task permits**.

## Experiment 6 — Oracle versus routed

Compare:

```text
oracle
routed
shuffled
disabled
all-memory
```

This separates transport, routing and content causality.

## Experiment 7 — Independent references

Materialize chunks independently rather than using exact historical cache.

Measure degradation and positional effects.

## Experiment 8 — Memory hierarchy

Where hardware allows:

```text
GPU only
GPU + CPU
GPU + CPU + local disk
```

Measure whether transfer/index overhead preserves the expected savings.

## Experiment 9 — Cross-attention/adapted transport

Only after native zero-training baselines are established, optionally test:

- separate cross-attention;
- `mem_o_proj`;
- learned memory gates;
- LoRA memory adaptation;
- learned positional correction.

Treat these as optional improvements, not requirements.

---

# 15. Datasets/tasks

Use multiple task types because ordinary LM loss alone may hide context dependence.

Include where feasible:

## Natural-language continuation

Useful for measuring degradation against normal model operation.

## Long-context retrieval

Needle-style or controlled retrieval tasks can test whether relevant distant state remains accessible.

## Multi-evidence reasoning

Use tasks where multiple distant chunks are genuinely required.

This is especially important for adaptive retrieval because some queries should naturally materialize more chunks than others.

## Long documents/code

Where practical include:

- repository-scale code;
- technical documents;
- books/reports;
- multi-turn agent traces.

The paper should not depend exclusively on synthetic retrieval tasks, but controlled tasks are useful for causal diagnosis.

---

# 16. Metrics

Do not report only accuracy/loss.

For every long-context condition, record where possible:

```text
accessible tokens N
local active tokens
retrieved token K/V R
R/N
number of references
number of chunks
number of gists inspected
number of hierarchy nodes visited
routing latency
KV materialization latency
KV transfer bytes
attention latency
prefill latency
decode latency/token
total latency
tokens/sec
peak accelerator memory
host-memory usage
archive storage size
quality/loss/perplexity/accuracy
gap to full/native baseline
```

For adaptive retrieval additionally report distributions:

```text
mean R
median R
p90 R
p99 R
R versus query/task difficulty
```

---

# 17. Key plots

Generate plots such as:

```text
quality gap vs accessible context
active K/V vs accessible context
R/N vs accessible context
latency vs accessible context
GPU memory vs accessible context
quality gap vs active-KV budget
quality vs routing cost
retrieved K/V distribution under adaptive retrieval
```

The strongest pattern would be:

```text
N increases greatly
R increases slowly/adaptively
R/N declines
accelerator memory/attention cost grow far slower than dense context
quality remains close to the relevant baseline
```

Do not hide cases where broad reasoning requires large R.

Those cases are important evidence about when context is or is not sparse.

---

# 18. FlashAttention / SDPA positioning

Do not describe PRA as replacing FlashAttention.

Use:

> FlashAttention/SDPA makes attention over the active K/V set more hardware-efficient. PRA attempts to reduce how much historical K/V needs to enter that active set.

Intended stack:

```text
very large logical context
 -> PRA progressive retrieval
 -> selected native K/V
 -> SDPA / FlashAttention
 -> accelerator
```

Where possible benchmark PRA using the same optimized kernel used by the baseline for the active attention operation.

The systems comparison should therefore distinguish:

```text
dense attention with optimized kernel
vs
PRA-selected attention with optimized kernel
```

This demonstrates whether PRA changes the scaling problem rather than merely benefiting from a slower baseline.

---

# 19. Memory hierarchy and industrial economics

Paper 2 should include a systems discussion, grounded in measurements.

Potential hierarchy:

```text
accelerator HBM/VRAM
 -> hot local K/V

CPU/host RAM
 -> warm retrieved/history K/V

NVMe/local SSD
 -> cold K/V chunks

optional distributed/object storage
 -> archival long-lived state
```

Gists/index structures can remain much smaller than full K/V.

The economic hypothesis is:

> Very-long context may become increasingly a storage/indexing/transfer problem instead of requiring all accessible history to reside in the most expensive active neural-computation tier.

Measure whether this is actually advantageous.

Do not assume PCIe/host transfer or indexing is free.

Report transfer and routing overhead explicitly.

---

# 20. Approximate-unbounded-context claim

Use careful language.

Good formulations:

- approximately unbounded context;
- very-large accessible context with bounded or sublinear active attention;
- attention cost related to relevant rather than available context;
- logical context decoupled from active context.

Avoid claiming literal infinite context or O(1) total cost unless mathematically and empirically justified.

Storage remains at least related to retained information unless compression/eviction is introduced.

Routing can also become expensive unless indexed/hierarchical.

---

# 21. Ablations

Important ablations include:

```text
native exact historical K/V vs independently encoded chunks
original positions vs virtual/remapped positions
flat vs hierarchical routing
single gist vs multiple gists
mean vs K-means/prototype/SOM/etc.
fixed top-k vs threshold/adaptive retrieval
all-memory vs sparse oracle
oracle vs routed
valid vs shuffled
GPU-only vs tiered storage
native K/V vs cross-attention adapter
```

Do not run every combinatorial combination on every model.

Use staged experiments and select representative conditions.

---

# 22. Correctness tests

Before large experiments, require:

## Native equivalence

All relevant native K/V restored:

```text
PRA logits ~= baseline logits
```

where exact equivalence should hold.

## Empty-memory equivalence

No retrieved memory:

```text
PRA local output ~= local-only baseline
```

## Cache round-trip

```text
HF cache
 -> PRA external store
 -> restore
```

preserves K/V within expected numerical tolerance.

## GQA/MQA shape tests

Validate K/V shapes and repeat behavior.

## Position tests

Verify RoPE handling using controlled examples.

## Generation tests

Ensure greedy generation matches baseline in equivalence conditions.

## No-training test

Native PRA must run with all base model weights frozen and no required learned transport parameters.

---

# 23. Performance engineering

Do not optimize prematurely, but avoid an implementation that makes meaningful benchmarking impossible.

Track separately:

```text
Python routing overhead
tensor copy overhead
CPU<->GPU transfer
attention kernel time
cache indexing
gist construction/update
```

Prefer batched transfers/materialization.

Avoid per-token Python loops in the final benchmark path where possible.

Support profiling with PyTorch profiler/Nsight-compatible ranges if convenient.

---

# 24. Paper structure

Recommended Paper 2 structure:

```text
1. Introduction
2. PRA inference-time hypothesis
3. Native-KV formulation
4. Relationship to dense long-context attention, KV compression and RAG
5. Hugging Face integration architecture
6. Position/RoPE and cache semantics
7. Adaptive progressive retrieval
8. Experimental setup
9. Native equivalence results
10. Context multiplier and active-KV scaling
11. Oracle vs routed retrieval
12. Adaptive retrieval results
13. Memory hierarchy / systems results
14. Independent-reference and position ablations
15. Optional learned/cross-attention adaptations
16. Limitations
17. Industrial/system implications
18. Conclusion
```

---

# 25. Paper narrative

The paper should tell a causal story:

### Step 1
Can native K/V be externalized and reinserted without changing model behavior?

### Step 2
Can only a subset be reinserted while preserving quality?

### Step 3
Can that subset be found cheaply?

### Step 4
Does the selected subset grow much more slowly than accessible context?

### Step 5
Do memory/storage/transfer costs preserve the computational advantage?

### Step 6
Does this hold on existing pretrained HF models without changing base weights?

Only after these questions should the paper discuss learned adapters.

---

# 26. Negative results are important

The project is high-upside but must remain falsifiable.

Report clearly if:

- routing cost approaches dense attention cost;
- R grows nearly linearly with N;
- broad reasoning requires most of the history;
- RoPE remapping harms performance;
- independent chunks are not usable without adaptation;
- CPU/GPU transfer dominates runtime;
- native sparse attention loses too much quality;
- certain HF families are difficult to wrap cleanly.

These results identify the true limits of the PRA hypothesis.

---

# 27. Implementation constraints

Keep core PRA independent from Paper 1 toy-model code where possible, but reuse general abstractions that are sound.

Avoid duplicating model-family code.

Keep:

```text
routing
storage
materialization
transport
HF adaptation
position policy
metrics
```

as separate modules/interfaces.

Do not make cross-attention assumptions leak into native-KV infrastructure.

Do not make a fixed top-k budget mandatory.

Do not make gists part of transported attention state by default.

---

# 28. Completion criteria

Paper 2 implementation is ready for serious evaluation when:

- at least one standard HF causal LM can be wrapped without modifying pretrained weights;
- exact historical K/V can round-trip through PRA;
- native-KV reinjection passes equivalence tests;
- generation works;
- explicit and implicit references can be represented;
- long prompts can externalize an early `#__head` prefix;
- fixed and adaptive retrieval policies exist;
- no hard memory budget is required by the algorithm;
- oracle routing can bypass learned/heuristic routing;
- routed, shuffled and disabled controls exist;
- active K/V and total accessible K/V are measured;
- latency and accelerator-memory metrics are captured;
- RoPE handling is explicit;
- at least one GQA model is supported if feasible;
- Paper 2 experiments include context growth beyond the model's directly active PRA window;
- existing optimized attention kernels remain usable on the selected active K/V set where practical.

---

# 29. Guiding principle

When choices are ambiguous, preserve this invariant:

> **PRA should first try to make the right native K/V available to the pretrained attention operation, and should retrieve as much relevant memory as the query actually needs—not as much as an arbitrary fixed budget dictates.**

The central scientific and systems question is:

> **Can accessible context become vastly larger than active context without proportional quality or inference-cost growth?**

That is the result Paper 2 should be designed to test.

```md
### Paper 2 decisive experiment

Treat the highest-value Paper 2 question as:

> **Can an existing pretrained Hugging Face causal LLM obtain substantially larger effective context through PRA as an inference-time wrapper, without changing the pretrained base weights?**

The ideal experimental setup is conceptually:

```python
model = AutoModelForCausalLM.from_pretrained(...)

pra_model = PRA.wrap(
    model,
    direct_context=...,
    active_memory_budget=...,
)

The actual implementation may require attention-module replacement, cache hooks, RoPE handling, GQA/MQA support and custom kernels, but preserve this product-level abstraction.

Test increasing accessible context such as:

1x
2x
4x
8x
16x
32x
64x

Measure:

quality relative to native/full-context baseline
latency
GPU memory
CPU/host memory
KV transfer bandwidth
routing cost
tokens/sec
time-to-first-token
decode cost

Only after establishing the zero-training baseline should Paper 2 test optional:

learned routing;
learned positional correction;
memory adapters;
cross-attention transport;
fine-tuning.


I would **not change the existing implementation instructions otherwise**. This patch mainly prevents Codex—or future paper revisions—from losing the central thesis: **the potentially important result is not that references work; it is that native sparse K/V reactivation may decouple effective context length from active attention cost.**