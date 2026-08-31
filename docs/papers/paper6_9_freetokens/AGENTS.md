# AGENTS.md — Paper 6.9: PRA with FreeToken

## Semantic memory under bandwidth-adaptive MoE serving

### Central question
Can PRA context selection compose with edge-native expert/model-state selection to improve the quality-memory-bandwidth frontier?

Treat FreeToken as a distinct research/serving target. Do not assume its current architecture from older descriptions: pin and audit the repository first.

### Separation of responsibilities
PRA owns semantic records, routing/selection, profiles, task/session policy, authorization and HOT/WARM/COLD/SOURCE.
FreeToken owns model/expert execution, bandwidth-adaptive scheduling, physical model-state movement and its native cache/scheduler.

Do not merge the PRA router with an expert router merely because both perform selection.

### Phase 0 — architecture audit
Pin FreeToken commit and document:
- model families;
- MoE architecture;
- expert routing;
- model/expert placement;
- CPU/GPU/disk/network movement;
- scheduler;
- cache hierarchy;
- SGLang/vLLM/FlashInfer/llama.cpp-derived components actually present;
- attention/KV interfaces;
- continuous batching;
- extension seams.

### Phase 1 — E0 baseline
Run FULL and E0_SELECTED under matched quality.
Measure visible tokens, task quality, TTFT/ITL, throughput, device memory, model/expert bytes moved and bandwidth.

### Phase 2 — resource-identity integration
Add E1 logical PRA identity if useful for resource reuse and lifecycle. Keep semantic resource identity separate from expert/model-state identity.

### Phase 3 — native E2
Implement detached selected PRA K/V through the smallest native attention/cache seam.
Validate positions, masks, GQA/MQA/MoE topology, one softmax, lifetime, isolation and no duplication.

### Phase 4 — joint physical frontier
Cross:
- expert/model-state policy;
- PRA representation FULL/E0/E2;
- PRA consumer-layer profile;
- HOT/WARM/SOURCE;
- bandwidth limit.

Do not train a joint controller yet. Use fixed/oracle/matched policies first.

### Phase 5 — bandwidth sweep
Artificially or physically sweep available:
- host-device bandwidth;
- disk bandwidth;
- network/off-node bandwidth where relevant.

Measure when weight/expert traffic vs PRA traffic dominates.

### Phase 6 — coordinated prefetch
Evaluate:
- expert/model-state prefetch only;
- PRA prefetch only;
- independent concurrent prefetch;
- coordinated prefetch.

Measure ready-before-demand, wasted prefetch, contention, queue delay and GPU idle time.

### Phase 7 — E3 scheduler integration
Claim E3 only when FreeToken's real scheduler owns PRA prefetch/placement/sharing/eviction alongside its model-state decisions.

Key question:
Can the scheduler exploit two independent predicted needs without one starving the other?

### Phase 8 — memory decomposition
Report:
- model resident bytes;
- expert/model-state transfer;
- local KV;
- PRA HOT;
- PRA WARM;
- temporary memory;
- peak device/host memory.

### Phase 9 — quality controls
Use natural and controlled workloads.
Measure:
- task F1/EM/success;
- evidence recall;
- exact E0/E2 parity where selection is frozen;
- gold logP where feasible.

Distinguish retrieval failure from expert/model execution failure.

### Phase 10 — profiles
Evaluate REFERENCE/QUALITY_MAX/BALANCED/ECONOMY as joint quality-memory-bandwidth profiles, but keep model/expert and PRA configuration fields independently visible.

### Required metrics
Quality; selected/full tokens; active K/V; expert count; model/expert bytes moved; PRA bytes moved; H2D/network/disk bandwidth; TTFT/ITL/completion tails; req/s/tok/s; queue delay; cache hits; evictions/reloads; ready-before-demand; peak memory; successful req/s.

### Tables
Capability mapping; E0/E2 parity; bandwidth sweep; expert/PRA traffic decomposition; profile frontier; E3 scheduling results.

### Figures
Architecture with independent selectors; quality-memory-bandwidth Pareto frontier; bytes/token decomposition; latency vs bandwidth; coordinated-prefetch overlap; profile frontier.

### Editorial structure
Introduction; PRA primer; FreeToken architecture; independent semantic/expert selection; native integration; bandwidth/scheduler experiments; quality frontier; related work; limitations; reproducibility.

### Falsification
Native PRA is not justified if optimized E0 gives the same quality-memory-bandwidth frontier, or if PRA traffic destroys the benefit of FreeToken's bandwidth-adaptive execution.

### Stop gate
Architecture pinned/audited; E0 qualified; E2 implemented or precisely blocked; frozen-selection parity measured; bandwidth decomposition measured; coordinated prefetch tested; E3 claimed only on real scheduler ownership; product matrix emitted; tests/PDF pass.

### Core message
FreeToken and PRA select different things: model/expert state versus semantic context. Paper 6.9 should determine whether keeping those decisions logically independent but physically coordinated improves edge inference under tight memory and bandwidth constraints.
