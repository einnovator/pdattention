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

### Long-context implicit `#__head`

- Prompts longer than the direct window can use `prompt_overflow_mode=implicit_reference`.
- The recent tail remains in ordinary causal SelfAttention. Displaced leading tokens are
  encoded into a request-local cache entry with URI `pra://implicit/prompt/head` and display
  name `#__head`.
- Each batch row owns a separate cache namespace. Identical implicit URIs therefore cannot
  leak K/V between examples.
- `max_prompt_gists` independently bounds how many head chunks remain routable. Metrics
  report total, direct, implicit, chunk, and gist counts.

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
  caches reuse indexes; trainable-gist mode preserves its computation graph.
- Summary-combination and reference-first modes currently retain the exact scalar fallback;
  tensorizing those optional paths is Phase-9 runtime work.
- Detailed timing now fences CUDA only in measurement mode, preventing asynchronous kernels
  from being charged to the wrong phase. Normal inference remains asynchronous.
- A five-seed CUDA comparison over HotpotQA-derived and QASPER-derived workloads, tiny and
  small backbones, 32 examples per condition, and 32/64/128/256 splits records exact loss
  parity. At 256 splits, routing falls from 522--542 ms to 27.7--28.0 ms for tiny and from
  908--981 ms to 53.2--55.5 ms for small, a 16.47--19.38x speedup. Packed-index construction
  is included; resolver, encoding, materialization, and memory attention are excluded.

Relevant modules: `src/pra_torch/memory.py`, `src/pra_torch/config.py`, and
`scripts/run_pra_routing_speed.py`.

### Runtime interpretation

At fixed top-k, PRA has demonstrated reductions in active attention K/V and potential
transfer, not yet reductions in the complete resident cache. The prototype still retains
all encoded K/V in memory. Actual GPU-capacity savings require CPU/remote/tiered backing
storage, selective transfer, paging, or eviction. End-to-end claims must include resolver,
encoding, indexing, transfer, materialization, attention, and synchronization time.

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
- benchmark packed-index reuse separately from per-request index construction
- add exact dense and approximate-nearest-neighbor index backends
- shared caches
- paging
- admission/eviction
- distributed cache
- persistent memory
- off-GPU K/V storage and selective asynchronous transfer
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
Paper 2 -> Pretrained integration
Paper 3 -> Runtime systems
Paper 4 -> Theory
Paper 5 -> Enterprise and analytics applications
