# AGENTS.md --- Paper 6.3: PRA on OpenVINO GenAI / OVMS

Work from the Paper 4.5 shared runtime. This is an engine paper: PRA
owns logical records, routing/selection, authorization, task/session
policy, and HOT/WARM/COLD/SOURCE; OpenVINO owns compiled execution,
continuous batching, prefix caching, KV blocks, eviction, and Intel
device execution.

## Research question

Can query-addressed non-prefix PRA memory coexist with OpenVINO GenAI
continuous batching and prefix caching while preserving model semantics
and improving context/memory economics on Intel hardware?

## Plan

1.  **Environment/capability audit.** Record OpenVINO/GenAI/OVMS
    versions, CPU/GPU/NPU, RAM/device memory, model/precision/topology,
    SchedulerConfig support (`cache_size`, `num_kv_blocks`,
    `max_num_seqs`, `max_num_batched_tokens`, prefix caching, eviction,
    split/fuse), and extension seams.
2.  **E0 baseline.** Run no-PRA, prefix-only, selected-text,
    prefix+selected-text, full-context using the common frozen
    manifests. Capture quality, visible tokens, TTFT, ITL, completion,
    req/s, prefix hits, cache blocks, RAM/device memory.
3.  **Prefix/PRA identity.** Establish
    `sequential prefix identity != PRA resource identity`. Prevent
    identical visible prompts with different hidden PRA resources from
    unsafe reuse.
4.  **Native E2 feasibility.** Prefer public GenAI/cache hooks, then
    graph/custom-op seams, then a minimal source patch only if
    necessary. Implement `OpenVINONativeExecutor`.
5.  **Correctness invariants.** Same-model source K/V; correct
    positions/masks; MHA/GQA/MQA topology; one softmax; no duplicates;
    request pinning; cleanup; tenant/session isolation.
6.  **Geometry-matched parity ladder.** Ordinary prefix,
    geometry-matched cached source, full native source,
    sparse/multi-record source, cached decode. Measure K/V error where
    visible, first-token/logit agreement, exact sequence, F1, gold
    log-prob.
7.  **Storage lifecycle.** Map HOT=attention-ready OpenVINO KV,
    WARM=lossless shared PRA representation, COLD=shared persistent
    compact representation, SOURCE=typed backing record. Reuse
    `PRAStorageManager`.
8.  **Continuous batching.** Sweep concurrency, `max_num_seqs`,
    batched-token limit, cache size/KV blocks, prefix caching, PRA HOT
    budget, shared/independent resources.
9.  **Intel device comparison.** CPU vs Intel GPU vs NPU where
    supported. Separate compilation time from serving time.
10. **Matched E0/E2 economics.** Cold, warm, multi-query,
    shared-resource concurrency, independent-resource concurrency.

## Required metrics

Quality; exact parity; Recall@k; visible/selected tokens;
TTFT/ITL/completion p50/p95/p99; req/s; cache-block occupancy;
prefix/PRA hits; evictions/reloads; RAM/device memory;
promotion/source-rebuild latency; bytes moved/shared.

## Paper tables

-   capability mapping;
-   E0/E2 parity;
-   prefix-cache × PRA state;
-   continuous-batching economics;
-   CPU/GPU/NPU comparison;
-   HOT/WARM/SOURCE economics.

## Figures

Architecture; TTFT p99 vs load; memory/cache blocks vs concurrency;
E0/E2 cost ratio; shared-resource savings; SOURCE-vs-WARM break-even.

## Editorial structure

Introduction; minimal PRA background; OpenVINO GenAI architecture;
prefix cache/continuous batching; native PRA design; correctness;
storage lifecycle; serving experiments; device comparison; related work;
limitations; reproducibility.

## Stop gate

E0 reproducible; E2 implemented or a precise API blocker documented;
geometry-matched parity measured; prefix coexistence safe; shared
lifecycle connected; continuous batching/tails measured; physical
memory/cache occupancy measured; product-matrix artifacts emitted; tests
pass; PDF visually checked.

## Core message

OpenVINO already manages KV blocks, prefix caching, eviction and
continuous batching. PRA should add semantic non-prefix resource
identity without replacing those mechanisms.
