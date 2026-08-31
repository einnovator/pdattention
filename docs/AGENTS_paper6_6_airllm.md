# AGENTS.md — Paper 6.6: PRA on AirLLM
## Semantic non-prefix memory under layer-streamed, low-VRAM inference

> Numbering note: use Paper 6.6 provisionally. Paper 6.5 has already been used in the PRA roadmap for tools/skills work. Do not overwrite that paper.

## 0. Target

Integrate the shared PRA runtime into:

```text
https://github.com/lyogavin/airllm
```

Create the AirLLM engine companion paper using the same semantic/runtime contract established in Paper 4.5 and the engine-paper methodology used by vLLM, SGLang, MLX, TensorRT-LLM, and OpenVINO.

This is not a conventional serving-engine integration.

AirLLM's distinctive mechanism is:

```text
model checkpoint
    ↓ split into layer shards
disk / host
    ↓ prefetch
one or a few layers resident on GPU
    ↓ compute
release layer weights
```

Recent AirLLM code deliberately delegates full model forward/generation semantics to Hugging Face Transformers and attaches pre/post hooks to large modules, streaming weights before execution and freeing them after. This is strategically useful for PRA because the model architecture and KV-cache semantics remain close to HF even though weight residency is radically different.

The paper should therefore study:

> Can PRA provide selected, persistent non-prefix context while AirLLM independently streams model weights, and does the combination enable large-model + long-context inference under very small VRAM budgets?

---

# 1. Do not duplicate Paper 4.5 semantic policy

The shared PRA runtime remains responsible for:

```text
typed records
routing
selection
materialization policy
consumer-layer profile
task/session metadata
authorization
HOT/WARM/COLD/SOURCE
storage retention
wire protocol
gateway/agent integration
product profile registry
```

AirLLM-specific code should own only:

```text
weight-streaming lifecycle
device layer residency
HF model invocation
AirLLM-specific cache/device movement
profiling of weight I/O
```

Do not implement a second record or storage policy inside AirLLM.

---

# 2. AirLLM architecture audit

Start by pinning the exact AirLLM commit/version.

Document:

```text
AutoModel path
AirLLMBaseModel
meta-device model skeleton
per-layer disk shards
forward pre-hooks
forward post-hooks
prefetch executor
pinned-memory behavior
compression
per-expert MoE streaming
Mac/MLX path if relevant
Transformers DynamicCache interaction
```

The current implementation states that the actual Transformers model owns the full forward/generation path while AirLLM hooks big modules to stream weights. Treat this as a major simplification for PRA integration.

Create:

```text
airllm_capability_audit.json
```

with:

```text
hf_attention_path
hf_cache_type
position_api
weight_streaming_mode
prefetching
compression
expert_streaming
device
resident_modules
supported_model_family
PRA integration level
```

---

# 3. Central scientific question

The paper should not merely ask "does PRA run?"

The central question is:

\[
\boxed{
\text{Can weight streaming and semantic context streaming be optimized independently?}
}
\]

AirLLM attacks:

```text
model-weight residency
```

PRA attacks:

```text
context/KV residency
```

The combination potentially permits:

```text
very large model
+
large backing context
+
small accelerator memory
```

This is a distinct and publishable systems question.

---

# 4. Decompose memory explicitly

Every experiment should separate:

```text
M_weights_hot
M_local_KV
M_PRA_hot
M_PRA_warm
M_temporary
M_framework
```

Report:

\[
M_{\rm peak}
=
M_{\rm weights}
+
M_{\rm localKV}
+
M_{\rm PRA}
+
M_{\rm temporary}
+
M_{\rm runtime}.
\]

Do not report only total VRAM.

The value of the integration is precisely in showing which pressure source dominates.

---

# 5. Stage A — exact disabled-PRA reference

Before adding native PRA:

1. choose a small model already supported by AirLLM and PRA;
2. compare AirLLM against the same HF checkpoint;
3. verify output/generation compatibility;
4. verify HF cache semantics under AirLLM streaming;
5. capture layer-streaming timing.

Preferred initial families:

```text
Qwen
Llama
Gemma if topology works
```

Then add a larger model that would not normally fit on the test GPU.

---

# 6. Stage B — E0 PRA gateway/runtime baseline

Use AirLLM unchanged as a PRA-unaware engine first.

Conditions:

```text
FULL_CONTEXT
SELECTED_TEXT
PRA gateway G10 selected text
```

Freeze selection.

Measure:

```text
quality
visible tokens
local KV
peak VRAM
weight I/O
TTFT
ITL
completion
tokens/s
```

This establishes whether context reduction helps even without native K/V integration.

---

# 7. Stage C — native PRA through HF semantics

Because AirLLM delegates generation to Transformers, first attempt to reuse the Paper 4.5 HF-native PRA path rather than patching AirLLM attention independently.

Desired architecture:

```text
PRA runtime
    ↓
HF PRA-aware model/cache
    ↓
AirLLM model skeleton
    ↓
AirLLM weight-streaming hooks
```

The key test is whether:

```text
native PRA K/V persists
```

while:

```text
model layer weights stream independently.
```

Do not create an AirLLM-specific attention implementation if the HF reference path can be reused.

---

# 8. Native cache/device placement

Investigate where native PRA K/V resides while weights stream.

Candidate modes:

## A. PRA HOT stays device-resident

```text
weights stream
PRA KV remains GPU
```

Pros:
- fast consumption.

Cons:
- reduces the tiny VRAM budget available for streamed weights.

## B. Per-layer PRA K/V streaming

For a consumer layer:

```text
load layer weights
+
load selected PRA K/V for that layer
+
compute
+
release both
```

This may be especially attractive for AirLLM.

## C. Hybrid

```text
small frequently-used PRA layers HOT
remaining layer-specific PRA K/V WARM
```

This should be experimentally evaluated rather than assumed.

---

# 9. AirLLM-specific PRA insight: layer-local K/V streaming

PRA detail K/V is already layer-specific.

AirLLM is also layer-streamed.

Therefore evaluate a fused lifecycle:

```text
for layer L:
    prefetch weights[L]
    prefetch PRA_KV[L] if L is consumer layer
    execute
    release weights[L]
    optionally release PRA_KV[L]
```

This could reduce native PRA HOT memory dramatically.

It may be one of the most interesting novel contributions of the paper.

---

# 10. Consumer-layer profiles become especially important

Test:

```text
REFERENCE_CORRECTNESS
QUALITY_MAX
BALANCED
ECONOMY
```

The AirLLM-specific cost of a PRA consumer layer includes:

```text
PRA K/V transfer
PRA attention work
potential memory pressure
```

A reduced consumer-layer band could therefore save more than native K/V bytes alone.

Report per-profile:

```text
quality
consumer layers
PRA K/V bytes transferred
PRA K/V peak residency
weight I/O
TTFT
ITL
total latency
```

---

# 11. Weight/PRA double prefetch

AirLLM already prefetches the next layer's weights.

Evaluate coordinated prefetch:

```text
while computing layer L:
    prefetch weights[L+1]
    prefetch PRA_KV[L+1] if needed
```

Required conditions:

```text
weight-only prefetch
PRA-only prefetch
independent parallel prefetch
coordinated prefetch
no prefetch
```

Measure:

```text
disk read time
host staging
H2D time
demand stall
GPU idle time
```

Avoid Python-thread multiplication that increases contention without overlap.

---

# 12. Storage tiers map naturally onto AirLLM

The AirLLM checkpoint itself is a disk-streamed weight store.

Do NOT mix model-weight storage identity with PRA source/native storage identity.

Keep:

```text
AirLLM weight shards
```

separate from:

```text
PRA SOURCE
PRA COLD
PRA WARM
PRA HOT
```

But measure contention when both use the same SSD/PCIe path.

This is an important AirLLM-specific systems issue.

---

# 13. Disk contention experiment

AirLLM can be disk-bound.

PRA WARM/COLD can also create disk reads.

Run:

```text
weights only
weights + PRA SOURCE reconstruction
weights + PRA WARM mmap
weights + PRA COLD compressed
```

Measure:

```text
disk throughput
read amplification
queue depth if observable
weight-layer stall
PRA promotion stall
TTFT
tokens/s
```

The correct PRA policy may differ on AirLLM because retaining K/V on disk can contend with model-layer streaming.

---

# 14. SOURCE may be especially competitive

The existing PRA results show that small resources may be cheaper to reconstruct from SOURCE than restore from WARM.

AirLLM makes this even more relevant because disk bandwidth is precious.

Measure:

\[
C_{\rm source}(r)
\quad \text{vs} \quad
C_{\rm warm}(r)
\]

including the effect on concurrent layer-weight reads.

Derive AirLLM-specific admission thresholds.

---

# 15. Large-model experiment

After correctness on small models, run at least one model that is materially larger than GPU VRAM.

Examples depend on available hardware, but AirLLM currently targets model families such as:

```text
Qwen
Llama
DeepSeek
Mistral/Mixtral
Gemma
```

Choose a model where AirLLM's weight streaming is necessary, not merely optional.

Evaluate:

```text
full visible context
selected-text PRA
native PRA
```

under the same small VRAM budget.

This is the paper's strongest use case.

---

# 16. MoE experiment

If feasible, include one sparse MoE model.

AirLLM can stream only routed experts for supported MoE architectures.

This creates a conceptual combination:

```text
expert selection
+
PRA context selection
```

Keep these independent in interpretation.

Measure:

```text
experts loaded/token
weight bytes/token
PRA selected bytes/token
peak VRAM
latency
```

Do not claim learned coordination between the two selectors unless implemented.

---

# 17. E0/E2 matched benchmark

Use the common engine benchmark:

```text
cold one-shot
warm repeat
multi-query same resource
shared-resource reuse
```

AirLLM is not primarily a high-throughput serving scheduler, so do not force vLLM-style continuous-batching claims unless the engine supports them.

Focus instead on:

```text
low-VRAM feasibility
peak memory
weight/context I/O
single/few-request latency
reuse
```

---

# 18. Quality vs memory frontier

The key plot should be:

\[
\text{task quality}
\quad \text{vs} \quad
\text{peak VRAM}
\quad \text{vs} \quad
\text{latency}.
\]

Conditions should include:

```text
AirLLM full context
AirLLM selected text
AirLLM native PRA full consumer profile
AirLLM native PRA balanced profile
```

This directly answers whether PRA extends AirLLM's low-memory frontier.

---

# 19. Context-length sweep

Sweep source/backing sizes such as:

```text
2K
8K
32K
64K
128K where model allows
```

Keep evidence amount controlled.

Measure whether PRA prevents local KV growth from becoming the next dominant memory bottleneck after AirLLM removes weight residency.

This is likely an important finding:

> Once model weights no longer dominate VRAM, the KV cache becomes proportionally more important.

---

# 20. Persistent resource reuse

Test repeated queries over one large document/resource.

Compare:

```text
selected text re-prefill
PRA SOURCE rebuild
PRA WARM per-layer K/V
PRA HOT K/V
```

Measure:

```text
bytes read
TTFT
peak VRAM
total latency
```

---

# 21. Correctness ladder

Required:

```text
HF baseline
AirLLM disabled-PRA
AirLLM E0 selected text
AirLLM native full-selected PRA
AirLLM per-layer streamed PRA
AirLLM repeated decode
```

Metrics:

```text
first-token agreement
exact sequence
logit error where accessible
F1/task score
gold-answer log probability
```

Geometry must be matched before interpreting exact-sequence divergence.

---

# 22. Position and cache invariants

Verify:

```text
source/query position geometry
RoPE
DynamicCache lifetime
selected K/V survives model-layer unload
selected K/V cannot alias freed/reused buffers
no duplicate attachment
```

AirLLM's weight unload must never invalidate PRA cache state.

---

# 23. Compression interactions

AirLLM supports compressed on-disk weight shards.

Keep these axes distinct:

```text
weight compression
PRA K/V quantization
PRA persistence compression
```

Create matched conditions.

Do not describe AirLLM weight quantization as PRA K/V quantization.

---

# 24. CPU and Mac paths

Only after CUDA/Linux correctness.

If practical:

```text
CPU AirLLM + PRA
Mac/MLX AirLLM path + PRA
```

But avoid duplicating Paper 6.2's MLX work.

The Mac experiment should focus on AirLLM weight streaming specifically.

---

# 25. Metrics schema

Capture:

## Quality
```text
task score
exact output parity
gold log probability
```

## Context
```text
source tokens
visible tokens
selected tokens
active PRA K/V
```

## Weight streaming
```text
layer bytes read
expert bytes read
weight disk time
weight H2D time
prefetch hit/stall
```

## PRA streaming
```text
PRA bytes read
PRA H2D
PRA restore/rebuild
PRA prefetch stall
```

## Memory
```text
peak VRAM
weight HOT
local KV
PRA HOT
temporary
host RAM
```

## Latency
```text
TTFT
ITL
completion
tokens/s
```

---

# 26. Tables

## Table A — capability mapping

| AirLLM mechanism | PRA interaction | Status |
|---|---|---|

## Table B — correctness

| Model | Condition | Exact | Quality | Gold logP |
|---|---:|---:|---:|---:|

## Table C — memory frontier

| Condition | Weight VRAM | Local KV | PRA KV | Peak VRAM | Quality |
|---|---:|---:|---:|---:|---:|

## Table D — I/O economics

| Condition | Weight reads | PRA reads | Disk time | H2D | Tok/s |
|---|---:|---:|---:|---:|---:|

## Table E — profiles

| Profile | Consumer layers | PRA peak | PRA transfer | Quality | Latency |
|---|---:|---:|---:|---:|---:|

---

# 27. Figures

Create:

1. AirLLM + PRA architecture;
2. VRAM decomposition;
3. quality-memory-latency Pareto frontier;
4. disk bandwidth decomposition;
5. context length vs peak VRAM;
6. weight-prefetch/PRA-prefetch overlap;
7. consumer-profile tradeoff.

---

# 28. Product matrix integration

Emit Paper 4.5-compatible product rows:

```text
engine = airllm
hardware
model
PRA profile
quality
visible-token reduction
active-KV reduction
peak VRAM
TTFT
ITL
tok/s
weight I/O
PRA I/O
status
```

Add an explicit AirLLM-specific metric:

```text
minimum measured VRAM for model/workload
```

---

# 29. Editorial organization

Structure the paper independently:

```text
1. Introduction
2. PRA background
3. AirLLM weight-streaming architecture
4. Orthogonal weight/context residency
5. PRA integration
6. Layer-local PRA streaming
7. Correctness
8. Memory and I/O experiments
9. Large-model experiments
10. Related work
11. Limitations
12. Reproducibility
```

Do not make the reader study Paper 4.5 first.

---

# 30. Related work

Cover:

```text
AirLLM
FlexGen
DeepSpeed-Inference/offload where relevant
Accelerate CPU/disk offload
layer/expert streaming
KV cache offload/compression
PRA engine papers
RAG/context compression
```

The distinguishing combination is:

```text
weight offload/streaming
+
query-selected semantic KV
```

---

# 31. Falsification

PRA is not useful for AirLLM if:

```text
selected-text E0 achieves the same memory/latency frontier
```

or if:

```text
PRA K/V disk traffic materially worsens AirLLM's weight-streaming bottleneck
```

or if:

```text
native PRA does not lower peak KV/context memory under long-context workloads.
```

Report negative results.

---

# 32. Stop gate

Complete when:

- AirLLM architecture is audited;
- disabled-PRA HF/AirLLM compatibility is established;
- G10 selected-text baseline is measured;
- HF-native PRA is attempted before AirLLM-specific attention changes;
- native PRA correctness is established or a precise blocker is documented;
- HOT-resident vs per-layer-streamed PRA is compared;
- coordinated weight/PRA prefetch is measured;
- disk contention is measured;
- context-length memory frontier is measured;
- at least one genuinely too-large-for-VRAM model is evaluated;
- product matrix rows are emitted;
- tests pass;
- PDF is rebuilt and visually inspected.

---

# Core paper message

> AirLLM removes model-weight residency as the dominant VRAM requirement by streaming layers or experts. PRA can remove persistent long-context residency by selecting and materializing semantic K/V independently. The paper should determine whether these two forms of streaming compose into a practical large-model, long-context inference regime on small GPUs.
