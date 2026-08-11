# AGENTS-PRA-Roadmap

## Goal
Create a staged research program. Every experiment should answer one scientific question.

## Implemented foundation (August 2026)

The implementation baseline for all new experiments is now native-KV PRA. The former
adapted cross-attention transport remains available only for historical reproduction and
tail-section ablations.

### Native-KV transport

- A trained SelfAttention model can be converted to PRA by copying its Q/K/V and output
  projections. Canonical native transport adds no memory-specific projection.
- Ordered references can be encoded once as historical causal context and sliced back into
  independently addressable URI/chunk K/V. Slices retain global source positions and the
  direct tail continues at the historical position offset.
- Selected memory K/V and local K/V share one attention softmax. Historical memory is
  visible to every local query; local K/V retains causal and padding masks.
- The default materialization mode loads complete token K/V for selected chunks. Gist-only
  and whole-reference materialization remain explicit experimental alternatives.
- Five-seed tests through 256 nominal splits verify dense native-all parity and show that
  historical slicing removes the false degradation caused by independently resetting
  context and positions for every small reference.

Relevant modules: `src/pra_torch/model.py`, `src/pra_torch/attention.py`,
`src/pra_torch/memory_batching.py`, and `src/pra_torch/memory.py`.

### Model-bounded logical context

- `model_max_context_tokens` is the hard per-operation native limit. It defaults to the
  tiny model's `max_seq_len` and may be lowered for a deployment constraint.
- `encoding_chunking` and `routing_chunking` are independent instances of the shared
  `ChunkingConfig`. One bounded encoding block can produce multiple smaller routing
  chunks while retaining global logical offsets.
- Encoding overlap is context-only: left context enters the base-model call, but only the
  new core span is retained as K/V. Fixed, marker, and semantic-plugin partitioning reuse
  the same source partitioner at both levels.
- All routed candidates, including `#__head` and explicit references, share one
  score-ordered whole-chunk allocator. The enforced invariant is
  `direct + materialized + reserve <= model_max_context_tokens`.
- Budgeting runs before K/V transfer. Oversized chunks are skipped rather than ending the
  scan, allowing smaller lower-ranked chunks to fill remaining capacity.
- Diagnostics now distinguish routing candidates, requested/materialized tokens,
  budget-rejected chunks, utilization, score boundaries, encoding calls, and maximum
  encoding input.
- A five-seed CUDA probe uses a 32-token native limit with 16-token encoding cores and
  four-token routing chunks. A 184-token head requires 12 calls whose largest input is 20
  tokens. No encoded or attended operation violates the limit.

Relevant modules: `src/pra_torch/chunking.py`, `src/pra_torch/config.py`,
`src/pra_torch/model.py`, `src/pra_torch/attention.py`, and
`scripts/run_pra_bounded_context.py`.

### Long-context implicit `#__head`

- Prompts longer than the direct window can use `prompt_overflow_mode=implicit_reference`.
- The recent tail remains in ordinary causal SelfAttention. Displaced leading tokens are
  encoded once when they fit or in bounded overlap-aware blocks when they exceed the
  native limit. Native layer K/V is sliced into a request-local cache entry with URI
  `pra://implicit/prompt/head` and display name `#__head`.
- Each batch row owns a separate cache namespace. Identical implicit URIs therefore cannot
  leak K/V between examples.
- Mixed-length batches carry one historical tail-position offset per row. Exact tests show
  that restoring all head slices reproduces dense full-sequence tail logits.
- `max_prompt_gists` independently bounds how many head chunks remain routable. Metrics
  report total, direct, implicit, chunk, and gist counts.
- A five-seed fixed-target probe now covers 1x/2x/4x/8x the direct window. Historical
  routing matches dense loss through 4x and beats truncation at 8x; wrong-memory and
  independent-chunk controls confirm content dependence and the cost of context resets.
- Streaming generation migrates expired routing-sized prefixes into `#__head`, invalidates
  stale packed indexes, and rebuilds bounded history. A 48-token smoke continuation uses
  11 rollovers and 40 routed steps while keeping direct history at eight tokens or fewer.
  The current rebuild is correct but intentionally not an incremental K/V append.

Relevant module: `src/pra_torch/prompt.py` with integration in `src/pra_torch/pra_train.py`.

### Exact tensorized routing

- `routing_backend=tensorized` is the default; `legacy` is retained for parity and timing
  controls.
- Per-layer chunk gists are normalized and packed as `[chunks, gists, d_model]`. A query
  batch scores all candidates with matrix multiplication/einsum rather than one Python/CUDA
  call per gist.
- URI aggregation remains exact for max, mean, and log-sum-exp policies. `torch.topk`
  selects references and per-reference chunks without changing materialization semantics.
- Packed indexes are invalidated whenever entries or gist representations change. Detached
  caches reuse indexes across queries; trainable-gist mode preserves its computation graph.
- After reference top-k, the normal path gathers only selected reference rows for chunk
  top-k and transfers only selected identifiers/scores to the host. Full rankings remain
  available in diagnostic mode.
- Summary-combination and reference-first modes currently retain the exact scalar fallback;
  tensorizing those optional paths is Phase-9 runtime work.
- Detailed timing now fences CUDA only in measurement mode, preventing asynchronous kernels
  from being charged to the wrong phase. Normal inference remains asynchronous.
- The original five-seed cold benchmark reduced 256-way scalar routing from 522--981 ms to
  27.7--55.5 ms. A serving-oriented follow-up on real encoded caches reduces cold routing
  further to 14.8--33.7 ms and persistent-index warm routing to 5.5--11.6 ms, a 94--103x
  improvement over scalar selection with exact URI/chunk/score/loss parity. At 256 units,
  index construction is 45--54% of cold exact routing.

Relevant modules: `src/pra_torch/memory.py`, `src/pra_torch/config.py`, and
`scripts/run_pra_routing_speed.py`.

### Runtime interpretation

At fixed top-k, PRA now demonstrates both reduced active attention K/V and a prototype
reduction in complete GPU-resident source K/V. `kv_cache_residency=cpu` keeps gists/indexes
on GPU, leaves detached token K/V in pinned CPU memory, and transfers selected chunks only.
At 256 units it removes 0.73--3.42 MiB from the measured GPU cache with zero loss or
selection delta. Warm transfer costs 4--9 ms; cold cache construction is 3--4x slower
because offload is currently per chunk and synchronous. End-to-end claims must still include
resolver, encoding, indexing, transfer, materialization, attention, and synchronization.

### Overnight checkpoint (August 2026)

Completed:
- exact reusable packed indexes and selected-only ranking serialization
- historical encode-once/slice behavior for implicit prompt heads
- per-row historical offsets for mixed batches
- CPU-resident native K/V with actual transfer-byte and peak-CUDA metrics
- five-seed CUDA routing, long-prompt, sensitivity, and residency experiments
- hard native-context checks, shared encoding/routing chunking, global materialization
  budgeting, streaming rollover, and five-seed bounded-context artifacts

Next systems work:
- batch CPU offload during cache construction instead of issuing one transfer per chunk
- overlap selected K/V transfer with routing and local projection
- fuse or bucket variable-length memory attention
- append generated-history K/V and packed-index rows incrementally instead of rebuilding
  the request-local head cache
- benchmark pretrained RoPE/GQA models and genuine QA targets

## Phase 1
Finish WikiText-2:
- 5+ seeds
- parameter matching
- oracle references
- irrelevant references
- distance bins
- recursive references
- summary-only vs summary+text

## Phase 2
Controlled synthetic datasets:
- arithmetic
- graph traversal
- symbolic references
- repository navigation
- nested references

## Phase 3
QA datasets:
- HotpotQA
- 2WikiMultihopQA
- MuSiQue
- Natural Questions
- NarrativeQA
- Qasper

## Phase 4
Long-context:
- LongBench
- InfiniteBench
- ZeroSCROLLS
- RULER
- RepoBench
- Needle-in-a-Haystack

Engineering prerequisites:
- test pretrained RoPE and grouped-query attention without position drift
- compare direct truncation, implicit `#__head`, and dense full-context controls
- vary direct-tail length, head chunk size/overlap, top-k, and gist count
- measure selection quality separately from materialization and language-model quality

Status: controlled implicit-head transport, exact dense parity, and 1x--8x sensitivity are
complete on the tiny fixed-target probe. Source-relative offsets now survive rollover,
overlap, overflow splitting, routing, and materialization. With a 192-token logical source,
all continuous-head model operations remained at or below 24 tokens under a 32-token native
limit. Public long-context suites and pretrained models remain pending.

Paper 1.5 adds matched learned-absolute, sinusoidal, and RoPE tiny/small controls, pre/post-
position cache metadata, five-seed split scaling, positional/contextual fragmentation
decomposition, overlap, and bounded implicit-head evaluation. Source-relative offsets remove
layer-0 reset error exactly and improve final-layer K fidelity in all 30 paired validation runs.
That representation result survives WikiText-2, but 8L next-token loss improves in only 5/30
pairs. Controlled HotpotQA/QASPER answer-code probes are also mechanism-dependent: absolute and
sinusoidal routed means improve, whereas RoPE routed means worsen in both tiers and datasets.
Position continuity is therefore an auditable transport requirement, not a sufficient quality
intervention. A four-cell storage matrix verifies exact pre/post-RoPE parity at matched effective
positions; intentional K-only rebinding changes attention. The unfused deferred path is about
3.6x slower in the small GTX 950M microbenchmark and does not reduce K/V bytes. Overlap,
routing, and memory composition remain separate targets. Pretrained RoPE/GQA, unrestricted
natural QA, selective recomputation, and fused deferred-position kernels remain Phase 4/8 work.

## Phase 5
Software engineering:
- SWE-bench
- RepoBench
- CodeSearchNet

## Phase 6
Enterprise knowledge:
- Confluence
- Jira
- Documentation
- Wikis

## Phase 7
Data analytics:
- SQL schemas
- Semantic layers
- Dashboards
- Business glossaries
- Snowflake
- BigQuery
- DuckDB

## Phase 8
Pretrained models (ONLY after earlier phases are successful)
Do not attempt on the personal PC.
Use the higher-memory company Mac/workstation.
Recommended progression:
- SmolLM
- TinyLlama
- Qwen 0.5B
- Gemma 270M/1B
then:
- Llama
- Qwen
- Mistral
using PEFT/LoRA before full finetuning.

## Phase 9
Runtime:
- tensorize summary and reference-first routing paths
- [done] benchmark packed-index reuse separately from per-request index construction
- add exact dense and approximate-nearest-neighbor index backends
- shared caches
- paging
- admission/eviction
- distributed cache
- persistent memory
- [prototype] off-GPU K/V storage and selective transfer
- batch and asynchronously overlap cache offload/selected transfer
- fused variable-length native-KV attention kernels
- end-to-end latency, throughput, peak-memory, and energy accounting

## Phase 10
Theory:
- information theory
- complexity
- embodied cognition
- differentiable memory routing
- memory graphs

## Publication roadmap
Paper 0 -> Position
Paper 1 -> Standalone prototype
Paper 1.5 -> Positional semantics for retrieved native-KV
Paper 2 -> Pretrained integration [routing representation complete; learned router next]
Paper 3 -> Runtime systems
Paper 4 -> Theory
Paper 5 -> Enterprise and analytics applications

## Paper 2 Qwen checkpoint (August 2026)

The pretrained phase has started on `research/paper2-hf` from the frozen Paper 1.5 RoPE head.
The controlled attention implementation now delegates routing, whole-chunk budgeting,
materialization, selected transfer, native-K/V composition, and metrics to `PRAExecutionCore`.
A thin HF contract preserves family projection, position, mask, cache, and output semantics.

Completed first Qwen gate:
- pinned `Qwen/Qwen3-0.6B` revision `c1899de` under Transformers 4.55.4 eager attention
- one wrapped upper layer with all pretrained Q/K/V/QK-norm/output parameters reused
- bit-exact disabled logits and hidden states, 100% greedy-token agreement, and identical
  generation-cache shapes across all 28 layers
- native GQA storage with 8 K/V heads for 16 query heads; exact synthetic replay against
  explicit K/V repetition
- CPU-resident explicit-reference K/V with selected-only transfer and cached generation
- 408-token logical prompt split into a 280-token `#__head` and 128-token direct tail with
  source-relative offsets and zero native-limit violations
- structured JSON/CSV artifacts and unrestricted HotpotQA/QASPER pipeline smokes

Completed routing-representation gate:
- routing and transport are now independent: post-RoPE key means, normalized pre-RoPE key means,
  and attention-input hidden-state means all materialize the same post-RoPE native K/V
- on 16 matched HotpotQA/QASPER examples with 32-token chunks, post-RoPE routing has score-position
  correlation 0.652 and recall@3/8/16 of 0.125/0.250/0.313
- pre-RoPE routing removes the late-position correlation (0.009) and reaches
  0.313/0.563/0.750; hidden-state routing is also position-neutral (-0.021) and gives the best
  sparse recall@3, 0.438
- 64 hidden-state confirmation evaluations across two seeds yield recall@3/8/16 of
  0.391/0.578/0.797, MRR 0.326, and score-position correlation -0.077
- recall@16 requires selecting 23.8% of chunks; QASPER recall@3 remains 0.156
- one hidden-state gist adds 1.57% over 32-token detail K/V; mean packed-index build is 3.50 ms
  and warm exact routing plus selection is 1.90--2.74 ms on the GTX 950M

Current scientific blocker:
- RoPE phase contamination is confirmed as one cause of the original routing failure, but
  zero-parameter semantic ranking remains below the predeclared Qwen-to-Llama promotion gate
- the gate fails combined recall@3, per-dataset recall@3, and MRR while passing sparsity and
  position-bias conditions

Next:
- keep Qwen frozen and train a small evidence-supervised router/gist adapter
- retain pre-RoPE and hidden-state zero-parameter baselines and test cross-dataset generalization
- defer broad multi-gist and overlap sweeps unless the learned-router diagnostic needs them
- do not move to Llama until Qwen passes the documented routing gate; Gemma remains later
